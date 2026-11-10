# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-image", "matplotlib", "warp-lang>=1.18,<1.19"]
# ///
"""Step 4: simulate the artery over several heartbeats, healthy vs blocked (lbm.py on the GPU, NVIDIA Warp).

Boundary conditions (a lumped circulation around the 3D segment, so a blockage redistributes the flow as in life):
  - inlet (internal carotid end): a pulsatile PRESSURE P_s(t) standing for the heart and the arteries upstream,
    calibrated on the healthy case so that it draws the sourced inflow (240 mL/min mean, physics.json waveform);
  - outlets (middle and anterior cerebral arteries): resistances R_k standing for the brain tissue downstream,
    p_k = R_k Q_k, sized so that the healthy split matches the sourced MCA flow (135 mL/min) at a mean outlet
    pressure of P_OUT_MMHG (assumption, see below). The same P_s(t) and R_k are used for every clot case, so a
    blockage cuts the total inflow and reroutes part of it into the open branch.
Pressures in the solver are gauge values relative to P_OUT_MMHG (small density deviations); results are stored as
absolute mmHg. Flows are measured in slabs inside each open end (beyond the sponge layers).

Grids (Step 3): healthy and occluded at 0.1 mm / 5 us; the 70 % narrowing at 0.05 mm / 1 us (its 0.6 mm throat is
not grid-converged at 0.1 mm), ~3.6 h per heartbeat.

  uv run simulate.py healthy            # first: calibrates the inlet pressure (results/healthy)
  uv run simulate.py occluded
  uv run simulate.py narrow70           # overnight
  uv run simulate.py report             # reports/step4_report.md + figures from whatever cases have finished
"""
import argparse
import json
import math
import os
import time

import numpy as np
import warp as wp

import lbm
import patient as PT
import vessel as V

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MMHG = 133.322
GRIDS = {"healthy": (1.0e-4, 5.0e-6), "occluded": (1.0e-4, 5.0e-6), "narrow70": (5.0e-5, 1.0e-6),
         "narrow30": (1.0e-4, 2.0e-6), "narrow50": (1.0e-4, 2.0e-6), "narrow90": (5.0e-5, 1.0e-6),
         # Step 17: 40 % (1.3 mm throat) like 50 %; 60 % (0.8 mm throat, 8 cells at 0.1 mm) on the fine grid like 70 %
         "narrow40": (1.0e-4, 2.0e-6), "narrow60": (5.0e-5, 1.0e-6),
         # Step 11: patients' own vessels (patient.py, patients/cases.json), 0.1 mm unless the patient sets a grid
         **{f"{p}_{k}": tuple(c.get("grid", (1.0e-4, 2.0e-6))) for p, c in PT.PATIENTS.items() for k in c["scans"]}}
BEATS = 3
P_OUT_MMHG = 80.0       # mean pressure at the outlets, healthy (assumption: mean arterial ~90-95 mmHg minus the
#                         large arteries' losses); only shifts all pressures and sets how strongly flow reroutes
SPONGE_MM, SPONGE_TAU = 2.0, 0.55
FLUX_SLAB_MM = (2.5, 3.5)  # depth inside each open end where its flow is measured
SAMPLE_S = 1.0e-3       # time-series sampling
FRAMES = 40             # field snapshots over the last beat
HARMONICS = 10
OUTLET_RELAX_S = 0.5e-3  # outlet pressures follow R Q with this time constant (stability of the explicit coupling)
CASE_ORDER = ("healthy", "narrow30", "narrow40", "narrow50", "narrow60", "narrow70", "narrow90", "occluded")
OPEN_ENDS = (("trunk", "start", "inlet (ICA)"), ("blocked", "end", "MCA outlet"), ("sibling", "end", "ACA outlet"))


# ---------------------------------------------------------------------------------------------------------------
# kernels: outlet resistances and measurements
# ---------------------------------------------------------------------------------------------------------------


@wp.kernel
def k_flux(idx: wp.array(dtype=int), wgt: wp.array(dtype=wp.vec3), slot: wp.array(dtype=int),
           vel: wp.array(dtype=wp.vec3), Q: wp.array(dtype=float)):
    t = wp.tid()
    wp.atomic_add(Q, slot[t], wp.dot(vel[idx[t]], wgt[t]))


@wp.kernel
def k_bc(Q: wp.array(dtype=float), R: wp.array(dtype=float), pref: float, drive: wp.array(dtype=float),
         n_open: int, alpha: float, p_out: wp.array(dtype=float), q_last: wp.array(dtype=float)):
    """End of a step: inlet pressure from the drive waveform, outlet pressures from their resistances."""
    p_out[0] = drive[0]
    for k in range(n_open):
        q_last[k] = Q[k]
        if k > 0:
            # relaxed towards R Q (explicit p = R Q is unstable: R ~ the vessel's acoustic impedance)
            p_out[k] = p_out[k] + alpha * (R[k] * Q[k] - pref - p_out[k])
        Q[k] = 0.0


@wp.kernel
def k_gather(idx: wp.array(dtype=int), slot: wp.array(dtype=int), rho: wp.array(dtype=float),
             vel: wp.array(dtype=wp.vec3), out: wp.array(dtype=float), umax: wp.array(dtype=float)):
    t = wp.tid()
    wp.atomic_add(out, slot[t], rho[idx[t]] / 3.0)
    wp.atomic_max(umax, 0, wp.length(vel[idx[t]]))


class ArterySolver(lbm.Solver):
    """lbm.Solver + resistance outlets and flow measurement inside each step (captured in the CUDA graphs)."""

    def __init__(self, lat, nu_lu, flux, R_lu, pref_lu, alpha, **kw):
        super().__init__(lat, nu_lu, **kw)
        dev = self.dev
        self.f_idx = wp.array(flux["idx"], dtype=int, device=dev)
        self.f_w = wp.array(flux["w"], dtype=wp.vec3, device=dev)
        self.f_slot = wp.array(flux["slot"], dtype=int, device=dev)
        self.n_open = len(R_lu)
        self.Qacc = wp.zeros(self.n_open, dtype=float, device=dev)
        self.Qlast = wp.zeros(self.n_open, dtype=float, device=dev)
        self.R = wp.array(np.asarray(R_lu, np.float32), dtype=float, device=dev)
        self.pref = float(pref_lu)
        self.alpha = float(alpha)

    def _launch(self, a):
        super()._launch(a)
        wp.launch(k_flux, dim=len(self.f_idx), device=self.dev,
                  inputs=[self.f_idx, self.f_w, self.f_slot, self.vels[1 - a], self.Qacc])
        wp.launch(k_bc, dim=1, device=self.dev,
                  inputs=[self.Qacc, self.R, self.pref, self.drv, self.n_open, self.alpha, self.p_out, self.Qlast])


# ---------------------------------------------------------------------------------------------------------------
# set-up
# ---------------------------------------------------------------------------------------------------------------


def fourier(y, nh=HARMONICS):
    """Samples over one period -> [a0, a1, b1, a2, b2, ...] for a0 + sum a_k cos(2 pi k t) + b_k sin(2 pi k t)."""
    F = np.fft.rfft(y) / len(y)
    c = [F[0].real]
    for k in range(1, nh + 1):
        c += [2 * F[k].real, -2 * F[k].imag]
    return np.array(c)


def from_fourier(c, t):
    y = np.full_like(t, c[0], dtype=float)
    for k in range(1, (len(c) - 1) // 2 + 1):
        y += c[2 * k - 1] * np.cos(2 * math.pi * k * t) + c[2 * k] * np.sin(2 * math.pi * k * t)
    return y


def circuit(R_seg):
    """Lumped circuit (SI): outlet resistances and the inlet pressure waveform's Fourier coefficients (Pa, absolute)
    that draws the sourced inflow through (segment + outlets)."""
    q_in = V.Q_INLET_MEAN * 1e-6 / 60
    q_mca = V.Q_MCA_TARGET * 1e-6 / 60
    p_out = P_OUT_MMHG * MMHG
    R_mca, R_aca = p_out / q_mca, p_out / (q_in - q_mca)
    R_eq = p_out / q_in
    _, shape = V.inflow_waveform(400)
    coef = fourier((R_eq + R_seg) * q_in * shape)
    return {"R_out": [0.0, R_mca, R_aca], "R_seg": R_seg, "P_in_coef": coef.tolist(), "R_eq": R_eq}


def is_patient(case):
    return case.split("_")[0] in PT.PATIENTS


def setup(case, dt_scale=1.0):
    dx, dt = GRIDS[case]
    dt *= dt_scale
    dt = SAMPLE_S / round(SAMPLE_S / dt)   # whole steps per sample, so the beat and the 40 frames land on samples
    un = lbm.Units(dx, dt, V.RHO, V.MU / V.RHO)
    dx_mm = dx * 1e3
    t0 = time.perf_counter()
    if is_patient(case):
        phi, origin, ends, probes, g = PT.build(case, dx_mm)
    else:
        gh = V.build("healthy")
        g = V.build(case)
        lo, hi = V.bounds(gh)
        phi, origin = V.sdf_grid(g, dx_mm, lo, hi)
        ends = []
        for name, key, label in OPEN_ENDS:
            b = g["branches"][name]
            c, nrm = b["caps"][key]
            ends.append({"kind": "outlet", "centre": c, "normal": nrm, "label": label,
                         "radius": float(b["R"][0] if key == "start" else b["R"][-1])})
        probes = probe_points(gh, g)
    lat = lbm.build_lattice(phi, origin, dx_mm, openings=ends)
    P = origin + lat["coords"] * dx_mm
    lat["wall_mm"] = -phi[tuple(lat["coords"].T)]
    del phi
    # flow slabs: inflow positive at the inlet, outflow positive at the outlets
    idx, w, slot = [], [], []
    thick = (FLUX_SLAB_MM[1] - FLUX_SLAB_MM[0]) / dx_mm
    for k, e in enumerate(ends):
        rel = P - e["centre"]
        depth = -(rel @ e["normal"])
        side = np.linalg.norm(rel + depth[:, None] * e["normal"], axis=1) < 1.3 * e["radius"]
        m = np.flatnonzero(side & (depth >= FLUX_SLAB_MM[0]) & (depth < FLUX_SLAB_MM[1]))
        sign = -1.0 if k == 0 else 1.0
        idx.append(m), slot.append(np.full(len(m), k))
        w.append(np.tile(sign * e["normal"] / thick, (len(m), 1)))
    flux = {"idx": np.concatenate(idx).astype(np.int32), "w": np.vstack(w).astype(np.float32),
            "slot": np.concatenate(slot).astype(np.int32)}
    # pressure probes (healthy centrelines; the clot case's lumen decides which cells are inside)
    pidx, pslot = [], []
    for k, pr in enumerate(probes):
        m = np.flatnonzero(np.linalg.norm(P - pr["point"], axis=1) < pr["radius"])
        pr["cells"] = int(len(m))
        pidx.append(m), pslot.append(np.full(len(m), k))
    gather = {"idx": np.concatenate(pidx).astype(np.int32), "slot": np.concatenate(pslot).astype(np.int32)}
    print(f"{case}: grid {lat['shape']}, {lat['nbr'].shape[1] / 1e6:.2f} M fluid cells, links {lat['counts']}, "
          f"set-up {time.perf_counter() - t0:.0f} s", flush=True)
    return un, g, lat, ends, flux, probes, gather


def probe_points(gh, g):
    """Pressure probes along the centrelines (mm): ICA, junction, MCA before / in / after the clot, distal ends."""
    br = gh["branches"]
    Cb = br["blocked"]["C"]
    sb = V._arc(Cb)
    clot = gh["clot"]
    at = lambda s: Cb[int(np.argmin(np.abs(sb - s)))]
    rad = lambda s: float(np.interp(s, sb, br["blocked"]["R0"]))
    rcase = lambda s: float(np.interp(s, sb, g["branches"]["blocked"]["R"]))
    s_up = clot["face_arc"] - clot["ramp"] - 1.0                    # upstream of the taper
    s_mid = clot["face_arc"] + clot["body_length"] / 2
    s_dn = clot["face_arc"] + clot["body_length"] + clot["ramp"] + 3.0  # past the taper
    tr, sib = br["trunk"], br["sibling"]
    pts = [("ICA", tr["C"][tr["segment_from"]], float(tr["R0"][tr["segment_from"]])),
           ("junction", np.zeros(3), 1.0),
           ("MCA before clot", at(s_up), rad(s_up)),
           ("clot throat", at(s_mid), max(min(0.5 * rad(s_mid), 0.8 * rcase(s_mid)), 0.0)),
           ("MCA after clot", at(s_dn), rad(s_dn)),
           ("MCA distal", Cb[br["blocked"]["segment_to"] - 1], float(br["blocked"]["R0"][br["blocked"]["segment_to"] - 1])),
           ("ACA distal", sib["C"][sib["segment_to"] - 1], float(sib["R0"][sib["segment_to"] - 1]))]
    out = []
    for name, p, r in pts:
        out.append({"name": name, "point": np.asarray(p, float), "radius": 0.5 * r if name != "clot throat" else r})
    return out


# ---------------------------------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------------------------------


def coarse_map(lat, dx_mm):
    """Cells -> 0.1 mm bins (identity at 0.1 mm; 2x2x2 averages at 0.05 mm) for the stored fields."""
    f = int(round(0.1 / dx_mm))
    key = lat["coords"] // f
    uk, inv = np.unique(key, axis=0, return_inverse=True)
    cnt = np.bincount(inv.ravel())
    return uk.astype(np.int16), inv.ravel(), cnt, f


def result_name(case, flow=1.0, map_mmhg=None):
    """Results folder: the case, plus the blood-pressure level (e.g. narrow70_map146) or the inflow level when it
    isn't the sourced one (e.g. narrow70_flow130)."""
    if map_mmhg:
        return f"{case}_map{round(map_mmhg)}"
    return case if abs(flow - 1.0) < 1e-6 else f"{case}_flow{round(flow * 100)}"


def run(case, beats=BEATS, flow=1.0, dt_scale=1.0, map_mmhg=None):
    """flow scales the inlet pressure waveform (absolute) and so the inflow; the outlets' resistances stay.
    map_mmhg instead sets the inlet's mean pressure (a patient's blood pressure), i.e. flow = map / calibrated mean."""
    wp.init()
    tag = result_name(case, flow, map_mmhg)
    if map_mmhg:
        flow = map_mmhg * MMHG / json.load(open(os.path.join(RESULTS, "calibration.json")))["P_in_coef"][0]
        print(f"{tag}: inlet mean pressure {map_mmhg:.0f} mmHg = {flow:.3f} x the calibrated one", flush=True)
    os.makedirs(os.path.join(RESULTS, tag), exist_ok=True)
    un, g, lat, ends, flux, probes, gather = setup(case, dt_scale)
    dx_mm = un.dx * 1e3
    cal_path = os.path.join(RESULTS, "calibration.json")
    calibrate = case == "healthy" and flow == 1.0 and not map_mmhg
    if calibrate:
        cal = circuit(0.0)
    else:
        cal = json.load(open(cal_path))
    q_si = un.dx ** 3 / un.dt                      # m^3/s per lattice flow unit
    p_si = V.RHO * un.c ** 2                       # Pa per lattice pressure unit
    # outlets: the story's calibrated brain resistances, by outlet (a patient's MCA drains into the same brain)
    R_out = [cal["R_out"][[e[2] for e in OPEN_ENDS].index(e["label"])] for e in ends]
    R_lu = [r * q_si / p_si for r in R_out]
    pref_lu = P_OUT_MMHG * MMHG / p_si
    n_open = len(ends)
    s = ArterySolver(lat, un.nu_lu, flux, R_lu, pref_lu, un.dt / OUTLET_RELAX_S, model="regularised")
    s.set_outlet_pressure([0.0] + [-pref_lu] * (n_open - 1))
    s.set_sponge(SPONGE_MM / dx_mm, SPONGE_TAU, kinds=("outlet",))
    spb = int(round(V.HEART_PERIOD / un.dt))
    every = int(round(SAMPLE_S / un.dt))
    ramp = int(0.1 / un.dt)

    def drive(cal):
        # flow scales the absolute inlet pressure: the inflow runs through fixed resistances down to 0, so it scales
        # with the whole pressure (scaling only the gauge part above pref moved the inflow by ~5 %, not 30 %)
        c = np.array(cal["P_in_coef"]) / p_si * flow
        c[0] -= pref_lu                            # gauge
        s.set_drive(c, period=spb, ramp=ramp)

    drive(cal)
    g_idx = wp.array(gather["idx"], dtype=int, device=s.dev)
    g_slot = wp.array(gather["slot"], dtype=int, device=s.dev)
    g_out = wp.zeros(len(probes), dtype=float, device=s.dev)
    g_umax = wp.zeros(1, dtype=float, device=s.dev)
    pcount = np.maximum(np.bincount(gather["slot"], minlength=len(probes)), 1)
    cmap = coarse_map(lat, dx_mm)
    frame_every = spb // FRAMES
    if spb % every or frame_every % every:   # the loop only looks every `every` steps
        raise ValueError(f"{tag}: {spb} steps per beat / {frame_every} per frame are not multiples of {every}")
    series = {"t": [], "Q": [], "p_open": [], "p_probe": [], "umax": []}
    frames_p, frames_u, frames_t = [], [], []
    T0 = time.perf_counter()
    total = beats * spb
    while s.steps < total:
        s.run(every, chunk=every)
        g_out.zero_()
        g_umax.zero_()
        wp.launch(k_gather, dim=len(g_idx), device=s.dev,
                  inputs=[g_idx, g_slot, s.rho, s.vels[s.cur], g_out, g_umax])
        Q = s.Qlast.numpy() * q_si * 6e7           # mL/min
        series["t"].append(s.steps * un.dt)
        series["Q"].append(Q)
        series["p_open"].append(un.p(s.p_out.numpy()[:n_open]) / MMHG + P_OUT_MMHG)
        pp = un.p(g_out.numpy() / pcount) / MMHG + P_OUT_MMHG
        pp[np.bincount(gather["slot"], minlength=len(probes)) == 0] = np.nan
        series["p_probe"].append(pp)
        series["umax"].append(float(un.u(g_umax.numpy()[0])))
        beat = s.steps // spb
        if s.steps % spb == 0:
            if not s.ok():
                raise RuntimeError(f"{case}: solver diverged at step {s.steps}")
            tt = np.array(series["t"])
            m = tt > tt[-1] - V.HEART_PERIOD
            Qm = np.array(series["Q"])[m].mean(0)
            el = time.perf_counter() - T0
            flows = ", ".join(f"{e['label'].split()[0]} {q:.1f}" for e, q in zip(ends, Qm))
            print(f"{tag}: beat {beat} done ({el / 60:.1f} min): mean flow {flows} mL/min; "
                  f"max speed {max(np.array(series['umax'])[m]):.2f} m/s", flush=True)
            if calibrate and beat < beats:
                # the inlet pressure that gives the sourced mean inflow: R_seg from this beat's mean
                p_in = np.array(series["p_open"])[m][:, 0].mean() * MMHG
                R_tot = p_in / (Qm[0] * 1e-6 / 60)
                cal = circuit(float(max(R_tot - cal["R_eq"], 0.0)))
                drive(cal)
                print(f"  calibration: segment + extension resistance {cal['R_seg'] / 1e6:.0f} MPa s/m^3 "
                      f"({cal['R_seg'] * V.Q_INLET_MEAN * 1e-6 / 60 / MMHG:.1f} mmHg at mean inflow)", flush=True)
            save(tag, un, lat, ends, probes, series, frames_t, frames_p, frames_u, cmap, cal, beats, spb, done=False,
                 patient=g if is_patient(case) else None)
        elif s.steps % (50 * every) == 0 or s.steps <= 10 * every:
            names = "/".join(e["label"].split()[0] for e in ends)
            print(f"{tag}: {s.steps * un.dt * 1e3:.0f} ms, {(time.perf_counter() - T0) / s.steps * spb / 60:.1f} min/beat: "
                  f"flow {names} {'/'.join(f'{q:.0f}' for q in Q[:n_open])} mL/min, p "
                  f"{'/'.join(f'{p:.0f}' for p in series['p_open'][-1])} mmHg, "
                  f"max speed {series['umax'][-1]:.2f} m/s", flush=True)
        if s.steps > (beats - 1) * spb and (s.steps - (beats - 1) * spb) % frame_every == 0:
            p, u = s.fields()
            uk, inv, cnt, _ = cmap
            pc = np.bincount(inv, weights=p, minlength=len(uk)) / cnt
            uc = np.stack([np.bincount(inv, weights=u[:, k], minlength=len(uk)) / cnt for k in range(3)], 1)
            frames_t.append((s.steps - (beats - 1) * spb) * un.dt)
            frames_p.append((un.p(pc) / MMHG + P_OUT_MMHG).astype(np.float16))
            frames_u.append(un.u(uc).astype(np.float16))
    if calibrate:
        json.dump(cal, open(cal_path, "w"), indent=1, default=float)
    save(tag, un, lat, ends, probes, series, frames_t, frames_p, frames_u, cmap, cal, beats, spb, done=True,
         secs=time.perf_counter() - T0, patient=g if is_patient(case) else None)
    print(f"{tag}: done in {(time.perf_counter() - T0) / 60:.1f} min", flush=True)


def save(case, un, lat, ends, probes, series, frames_t, frames_p, frames_u, cmap, cal, beats, spb, done, secs=None,
         patient=None):
    d = os.path.join(RESULTS, case)
    np.savez_compressed(os.path.join(d, "series.npz"), **{k: np.array(v) for k, v in series.items()})
    meta = {"case": case, "done": done, "dx_mm": un.dx * 1e3, "dt_us": un.dt * 1e6, "tau": un.tau, "beats": beats,
            "steps_per_beat": spb, "cells": int(lat["nbr"].shape[1]), "links": lat["counts"], "secs": secs,
            "p_out_mmhg": P_OUT_MMHG, "calibration": cal,
            "open_ends": [{"label": e["label"], "centre": np.asarray(e["centre"]).tolist(),
                           "normal": np.asarray(e["normal"]).tolist(), "radius": e["radius"]} for e in ends],
            "probes": [{"name": p["name"], "point": p["point"].tolist(), "radius": p["radius"], "cells": p["cells"]}
                       for p in probes]}
    if patient:
        meta["patient"] = patient
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1, default=float)
    if done and frames_p:
        uk, inv, cnt, f = cmap
        wall = np.bincount(inv, weights=lat["wall_mm"], minlength=len(uk)) / cnt
        np.savez_compressed(os.path.join(d, "fields.npz"), coords=uk, origin=lat["origin"], dx_mm=0.1 if f > 1
                            else lat["dx"], wall_mm=wall.astype(np.float32), t=np.array(frames_t),
                            p_mmhg=np.array(frames_p), u_m_s=np.array(frames_u))


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------


def last_beat(series, period=V.HEART_PERIOD):
    t = series["t"]
    m = t > t[-1] - period + 1e-9
    return m


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [c for c in CASE_ORDER if os.path.exists(os.path.join(RESULTS, c, "meta.json"))]
    data = {}
    for c in cases:
        meta = json.load(open(os.path.join(RESULTS, c, "meta.json")))
        ser = dict(np.load(os.path.join(RESULTS, c, "series.npz")))
        fp = os.path.join(RESULTS, c, "fields.npz")
        data[c] = {"meta": meta, "ser": ser, "fields": dict(np.load(fp)) if os.path.exists(fp) else None}
    colors = {"healthy": "#2a6fdb", "narrow70": "#e08a00", "occluded": "#d62728", "narrow50": "#2ca02c"}
    names = [p["name"] for p in data[cases[0]]["meta"]["probes"]]

    # --- numbers per case (last beat)
    rows = {}
    for c in cases:
        ser = data[c]["ser"]
        m = last_beat(ser)
        Q, pp = ser["Q"][m], ser["p_probe"][m]
        dp = pp[:, names.index("MCA before clot")] - pp[:, names.index("MCA after clot")]
        beats_done = int(round(ser["t"][-1] / V.HEART_PERIOD))
        prev = (ser["t"] > ser["t"][-1] - 2 * V.HEART_PERIOD) & ~m
        conv = (np.abs(Q.mean(0) - ser["Q"][prev].mean(0)).max() / Q[:, 0].mean()) if prev.any() else float("nan")
        rows[c] = {"Q_mean": Q.mean(0), "Q_max": Q.max(0), "p_mean": np.nanmean(pp, 0), "p_max": np.nanmax(pp, 0)
                   if np.isfinite(pp).any() else pp[0], "dp_mean": float(np.nanmean(dp)), "dp_max": float(np.nanmax(dp)),
                   "umax": float(ser["umax"][m].max()), "beats": beats_done, "conv": conv,
                   "done": data[c]["meta"]["done"]}

    # --- figure 1: flows and pressures over the last beat
    fig, axs = plt.subplots(2, 3, figsize=(20, 9))
    for c in cases:
        ser = data[c]["ser"]
        m = last_beat(ser)
        t = ser["t"][m] - ser["t"][m][0]
        for k, lab in enumerate(("inflow (ICA)", "MCA (blocked branch)", "ACA (open branch)")):
            axs[0, k].plot(t, ser["Q"][m][:, k], color=colors[c], lw=2, label=c)
            axs[0, k].set_title(f"{lab} flow"), axs[0, k].set_ylabel("mL/min")
        for k, nm in enumerate(("MCA before clot", "MCA after clot")):
            axs[1, k].plot(t, ser["p_probe"][m][:, names.index(nm)], color=colors[c], lw=2, label=c)
            axs[1, k].set_title(f"pressure, {nm}"), axs[1, k].set_ylabel("mmHg")
        dp = ser["p_probe"][m][:, names.index("MCA before clot")] - ser["p_probe"][m][:, names.index("MCA after clot")]
        axs[1, 2].plot(t, dp, color=colors[c], lw=2, label=c)
    axs[1, 2].set_title("pressure drop across the clot (before - after)"), axs[1, 2].set_ylabel("mmHg")
    for ax in axs.ravel():
        ax.set_xlabel("time in the heartbeat (s)"), ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.suptitle("Last simulated heartbeat per case")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step4_beat.png"), dpi=90)
    plt.close(fig)

    # --- figure 2: speed (max along the view) and pressure (mean along the view) at peak systole and their difference
    fc = [c for c in cases if data[c]["fields"] is not None]
    if fc:
        fig, axs = plt.subplots(2, len(fc), figsize=(7.5 * len(fc), 10), squeeze=False)
        ref = data["healthy"]["fields"] if "healthy" in fc else data[fc[0]]["fields"]
        k_peak = int(np.argmax(np.nanmax(ref["u_m_s"].astype(float), axis=(1, 2)) if False else
                               np.linalg.norm(ref["u_m_s"].astype(float), axis=2).max(1)))
        lo = np.min([data[c]["fields"]["coords"].min(0) for c in fc], 0)
        hi = np.max([data[c]["fields"]["coords"].max(0) for c in fc], 0)
        shape = hi - lo + 1
        vmax = max(np.linalg.norm(data[c]["fields"]["u_m_s"][k_peak].astype(float), axis=1).max() for c in fc)
        pall = np.concatenate([data[c]["fields"]["p_mmhg"][k_peak].astype(float) for c in fc])
        plo, phi_ = np.nanpercentile(pall, 1), np.nanpercentile(pall, 99)
        for j, c in enumerate(fc):
            F = data[c]["fields"]
            co = F["coords"].astype(int) - lo
            sp = np.linalg.norm(F["u_m_s"][k_peak].astype(float), axis=1)
            pr = F["p_mmhg"][k_peak].astype(float)
            mip = np.full((shape[0], shape[1]), np.nan)
            np.fmax.at(mip, (co[:, 0], co[:, 1]), sp)
            acc = np.zeros((shape[0], shape[1]))
            cnt = np.zeros((shape[0], shape[1]))
            np.add.at(acc, (co[:, 0], co[:, 1]), pr)
            np.add.at(cnt, (co[:, 0], co[:, 1]), 1)
            pm = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            ext = [lo[0] * 0.1, (hi[0] + 1) * 0.1, lo[1] * 0.1, (hi[1] + 1) * 0.1]
            im = axs[0, j].imshow(mip.T, origin="lower", cmap="turbo", vmin=0, vmax=vmax, extent=ext)
            axs[0, j].set_title(f"{c}: speed (max through the depth), peak systole t = {F['t'][k_peak]:.2f} s")
            plt.colorbar(im, ax=axs[0, j], label="m/s", fraction=0.03)
            im = axs[1, j].imshow(pm.T, origin="lower", cmap="RdYlBu_r", vmin=plo, vmax=phi_, extent=ext)
            axs[1, j].set_title(f"{c}: pressure (mean through the depth)")
            plt.colorbar(im, ax=axs[1, j], label="mmHg", fraction=0.03)
            for ax in axs[:, j]:
                ax.set_aspect("equal"), ax.set_xlabel("grid x (mm)"), ax.set_ylabel("grid y (mm)")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "reports", "step4_fields.png"), dpi=80)
        plt.close(fig)

    # --- report
    L = ["# Step 4 report: the artery, healthy vs blocked", "",
         "Generated by `uv run simulate.py report`. Solver and its accuracy: Step 3. Inlet: pulsatile pressure "
         "calibrated on the healthy case to the sourced inflow; outlets: resistances for the brain downstream, sized for "
         f"the sourced MCA flow at {P_OUT_MMHG:.0f} mmHg mean outlet pressure (assumption); the same inlet pressure and "
         "resistances for every case. Numbers are over the last simulated heartbeat.", "",
         "| case | grid | beats | mean inflow | MCA flow | ACA flow | MCA share | peak speed | clot pressure drop (mean / peak) | "
         "beat-to-beat change |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cases:
        r, meta = rows[c], data[c]["meta"]
        L.append(f"| {c}{'' if r['done'] else ' (running)'} | {meta['dx_mm']} mm | {r['beats']} | "
                 f"{r['Q_mean'][0]:.0f} mL/min | {r['Q_mean'][1]:.0f} | {r['Q_mean'][2]:.0f} | "
                 f"{r['Q_mean'][1] / r['Q_mean'][0]:.0%} | {r['umax']:.2f} m/s | {r['dp_mean']:.1f} / "
                 f"{r['dp_max']:.1f} mmHg | {r['conv']:.1%} |")
    L += ["", "Mean pressure at the probes (mmHg, last beat):", "", "| probe | " + " | ".join(cases) + " |",
          "|---|" + "---|" * len(cases)]
    for k, nm in enumerate(names):
        L.append(f"| {nm} | " + " | ".join(f"{rows[c]['p_mean'][k]:.1f}" if np.isfinite(rows[c]['p_mean'][k])
                                           else "(no lumen)" for c in cases) + " |")
    if "healthy" in rows:
        h = rows["healthy"]
        L += ["", "Change vs healthy:", ""]
        for c in cases:
            if c == "healthy":
                continue
            r = rows[c]
            L.append(f"- {c}: inflow {r['Q_mean'][0] / h['Q_mean'][0] - 1:+.0%}, MCA flow "
                     f"{r['Q_mean'][1] / h['Q_mean'][1] - 1:+.0%}, ACA flow {r['Q_mean'][2] / h['Q_mean'][2] - 1:+.0%}; "
                     f"pressure before the clot {r['p_mean'][2] - h['p_mean'][2]:+.1f} mmHg, after it "
                     f"{r['p_mean'][4] - h['p_mean'][4]:+.1f} mmHg")
    cal = data[cases[0]]["meta"]["calibration"]
    L += ["", f"Circuit: outlet resistances MCA {cal['R_out'][1] / 1e9:.2f}, ACA {cal['R_out'][2] / 1e9:.2f} GPa s/m^3; "
          f"segment (incl. extensions) {cal['R_seg'] / 1e9:.3f} GPa s/m^3 (calibrated).", "",
          "Figures: `step4_beat.png` (flows, pressures, clot pressure drop over the beat), `step4_fields.png` (speed "
          "and pressure maps at peak systole). Data for Step 5: `results/<case>/fields.npz` (40 frames over the last "
          "beat on a 0.1 mm grid), `series.npz` (1 ms time series), `meta.json`."]
    open(os.path.join(HERE, "reports", "step4_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=list(GRIDS) + ["report"])
    ap.add_argument("--beats", type=int, default=BEATS)
    ap.add_argument("--flow", type=float, default=1.0, help="inflow level, x the sourced one (scales the driving pressure)")
    ap.add_argument("--map", type=float, default=None, help="inlet mean pressure in mmHg (patient blood pressure; "
                    "replaces --flow)")
    ap.add_argument("--dt-scale", type=float, default=1.0, help="x the case's time step (smaller for faster flow)")
    a = ap.parse_args()
    if a.case == "report":
        report()
    else:
        run(a.case, a.beats, a.flow, a.dt_scale, a.map)


if __name__ == "__main__":
    main()
