# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "matplotlib", "warp-lang>=1.18"]
# ///
"""Step 3: validate the Warp lattice-Boltzmann solver (lbm.py) against known answers, then time it on the artery.

  1. steady pipe flow (Poiseuille), periodic pipe driven by a body force: velocity profile vs the exact parabola,
     at 10 / 20 / 40 cells across (convergence order), interpolated vs half-way bounce-back walls;
  2. pulsatile pipe flow (Womersley), oscillating body force: velocity profiles through the cycle vs the exact
     Bessel-function solution, at Womersley numbers covering the artery's heartbeat harmonics;
  3. open pipe with the artery's boundary conditions (velocity inlet, pressure outlet), straight and tilted to the
     grid (the artery's ends are oblique): pressure gradient vs Hagen-Poiseuille, velocity profile;
  4. a 70 % narrowing (cosine stenosis, 91 % area reduction) at artery size and speed: pressure drop vs the
     Young-Tsai correlation, and grid convergence (0.1 vs 0.05 mm);
  5. the artery itself (healthy, 0.1 mm): lattice build, stability, speed -> time per heartbeat.
All runs use blood's physical viscosity at the artery's spacing and time step (no rescaled "toy" parameters).

  uv run validate.py            # writes reports/step3_report.md, step3_*.png, geometry/solver.json
"""
import json
import math
import os
import time

import numpy as np
import warp as wp
from scipy.special import jv

import lbm
import vessel as V

HERE = os.path.dirname(os.path.abspath(__file__))
NU = V.MU / V.RHO                       # m^2/s
OPERATING = {                           # spacing (m), step (s): healthy flow and the 70 % narrowing's jet
    "healthy": (1.0e-4, 5.0e-6),
    "narrow70": (1.0e-4, 2.0e-6),
    "narrow70_fine": (5.0e-5, 1.0e-6),
}
SOLVER = {"model": "regularised"}       # lbm.Solver options used throughout (D3Q27 + Galilean correction by default)
SPONGE_TAU = 0.55                       # tau at the open ends (sponge layers, see lbm.Solver.set_sponge)
SPONGE_CELLS = 20                       # sponge depth in the artery (cells; 2 mm at 0.1 mm)
OFFSET = np.array([0.31, 0.17, 0.5])    # pipe axes off the grid's symmetry lines; straight pipes' ends between nodes


def units(key):
    dx, dt = OPERATING[key]
    return lbm.Units(dx, dt, V.RHO, NU)


def l2(a, b):
    return float(np.sqrt(((a - b) ** 2).sum() / (b ** 2).sum()))


# ---------------------------------------------------------------------------------------------------------------
# 1 + 2: periodic pipes
# ---------------------------------------------------------------------------------------------------------------


def periodic_pipe(D):
    R = D / 2.0
    n = int(math.ceil(D)) + 4
    c0 = np.array([n / 2 - 0.5, n / 2 - 0.5]) + OFFSET[:2]
    I, J, _ = np.meshgrid(np.arange(n), np.arange(n), np.arange(2), indexing="ij")
    r = np.hypot(I - c0[0], J - c0[1])
    lat = lbm.build_lattice(r - R, [0, 0, 0], 1.0, periodic=(2,))
    return lat, r[tuple(lat["coords"].T)], R


def test_poiseuille(nu):
    out = []
    for D, bz in ((10, True), (20, True), (40, True), (20, False)):
        lat, r, R = periodic_pipe(D)
        s = lbm.Solver(lat, nu, bouzidi=bz, **SOLVER)
        F = 4 * nu * 0.02 / R ** 2                       # centreline 0.02 lattice units
        s.set_force([0, 0, F])
        s.run(int(14 * R * R / (5.78 * nu)))             # 14 decay times of the slowest mode
        _, u = s.fields()
        ua = F / (4 * nu) * (R ** 2 - r ** 2)
        out.append({"D": D, "bouzidi": bz, "l2": l2(u[:, 2], ua), "flow_err": float(u[:, 2].sum() / ua.sum() - 1),
                    "r": r / R, "u": u[:, 2] / ua.max(), "ok": s.ok()})
        print(f"  Poiseuille D={D} {'interpolated' if bz else 'half-way'}: L2 {out[-1]['l2']:.2e}, "
              f"flow {out[-1]['flow_err']:+.2e}")
    return out


def womersley_exact(r, R, Wo, F0, w, t):
    k = (1j ** 1.5) * Wo
    U = F0 / (1j * w) * (1 - jv(0, k * r / R) / jv(0, k))
    return np.real(U * np.exp(1j * w * t))


def test_womersley(nu, D=20, wos=(1.4, 2.8, 5.6), phases=12):
    out = []
    lat, r, R = periodic_pipe(D)
    for Wo in wos:
        T = int(round(2 * math.pi * R * R / (Wo * Wo * nu) / phases)) * phases
        w = 2 * math.pi / T
        Wo = R * math.sqrt(w / nu)
        rr = np.linspace(0, R, 50)
        peak = max(np.abs(womersley_exact(rr, R, Wo, 1.0, w, t)).max() for t in np.linspace(0, T, 24))
        F0 = 0.02 / peak
        s = lbm.Solver(lat, nu, **SOLVER)
        s.set_force([0, 0, F0])
        s.set_drive([0.0, 1.0, 0.0], period=T)
        s.run(2 * T)
        num, den, snaps = 0.0, 0.0, []
        for k in range(phases):
            s.run(T // phases)
            _, u = s.fields()
            ua = womersley_exact(r, R, Wo, F0, w, s.steps - 0.5)
            num += ((u[:, 2] - ua) ** 2).sum()
            den += (ua ** 2).sum()
            snaps.append((s.steps % T / T, u[:, 2] / 0.02, ua / 0.02))
        out.append({"Wo": Wo, "period_steps": T, "l2": float(math.sqrt(num / den)), "r": r / R, "snaps": snaps,
                    "ok": s.ok()})
        print(f"  Womersley {Wo:.2f} (period {T} steps): L2 over the cycle {out[-1]['l2']:.2e}")
    return out


# ---------------------------------------------------------------------------------------------------------------
# 3 + 4: open pipes (velocity inlet, pressure outlet)
# ---------------------------------------------------------------------------------------------------------------


def open_pipe(axis, L, radius, pad=3):
    """A straight pipe from A along unit `axis`, length L (cells), radius(s) a function of the axial coordinate.
    Returns the lattice, axial and radial coordinate per cell, and the inlet/outlet description."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    rmax = float(max(radius(np.linspace(0, L, 400))))
    ends = np.array([np.zeros(3), axis * L])
    ext = rmax * np.sqrt(np.clip(1 - axis ** 2, 0, 1)) + pad
    lo = ends.min(0) - ext
    hi = ends.max(0) + ext
    n = np.ceil(hi - lo).astype(int) + 1
    A = -lo + OFFSET                               # pipe start in grid coordinates
    G = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3) - A
    s = G @ axis
    rad = np.linalg.norm(G - s[:, None] * axis, axis=1)
    phi = np.maximum(np.maximum(rad - radius(s), -s), s - L).reshape(n)
    R0, R1 = float(radius(np.array([0.0]))[0]), float(radius(np.array([L]))[0])
    openings = [{"kind": "inlet", "centre": A, "normal": -axis, "radius": R0},
                {"kind": "outlet", "centre": A + axis * L, "normal": axis, "radius": R1}]
    lat = lbm.build_lattice(phi, [0, 0, 0], 1.0, openings=openings)
    P = lat["coords"] - A
    sc = P @ axis
    rc = np.linalg.norm(P - sc[:, None] * axis, axis=1)
    return lat, sc, rc, axis


def section_profile(sc, values, L, nb=None):
    """Mean of `values` in 1-cell slabs along the axis -> (s centres, means)."""
    nb = nb or int(L)
    edges = np.linspace(0, L, nb + 1)
    k = np.clip(np.digitize(sc, edges) - 1, 0, nb - 1)
    cnt = np.bincount(k, minlength=nb)
    m = np.bincount(k, weights=values, minlength=nb) / np.maximum(cnt, 1)
    return 0.5 * (edges[1:] + edges[:-1]), m


def flow_rate(sc, u_axial, s0, s1):
    """Volume flow through the pipe, averaged over the slab s0..s1 (lattice units: cells^3 / step)."""
    m = (sc >= s0) & (sc < s1)
    return float(u_axial[m].sum() / (s1 - s0))


def run_steady(s, measure, chunk, min_chunks=4, max_chunks=40, tol=2e-4, avg_last=1):
    hist = []
    for it in range(max_chunks):
        s.run(chunk)
        if not s.ok():
            return hist, False
        hist.append(measure())
        if it + 1 >= min_chunks and abs(hist[-1][0] - hist[-2][0]) <= tol * abs(hist[-1][0]):
            break
    return hist, True


def test_open_pipe(u_key="healthy", D=20, length_d=12, U_phys=0.6):
    un = units(u_key)
    U = float(un.u_lu(U_phys))
    out = []
    for name, axis in (("straight", (0, 0, 1)), ("tilted (1,2,2)", (1, 2, 2))):
        R = D / 2
        L = length_d * D
        lat, sc, rc, ax = open_pipe(axis, L, lambda s: np.full_like(s, R, dtype=float))
        s = lbm.Solver(lat, un.nu_lu, **SOLVER)
        s.set_sponge(D, SPONGE_TAU)
        s.set_drive([U], ramp=4000)
        s.set_outlet_pressure([0.0])
        a, b = 3 * D, (length_d - 3) * D

        def measure():
            p, u = s.fields()
            ua = u @ ax
            Q = flow_rate(sc, ua, a, b)
            sm, pm = section_profile(sc, p, L)
            m = (sm > a) & (sm < b)
            slope = np.polyfit(sm[m], pm[m], 1)[0]
            return -slope, Q, sm, pm, ua

        hist, ok = run_steady(s, measure, chunk=10000, min_chunks=6)
        G, Q, sm, pm, ua = hist[-1]
        G_exact = 8 * un.nu_lu * Q / (math.pi * R ** 4)
        mid = (sc > a) & (sc < b)
        Um = Q / (math.pi * R * R)
        prof_err = l2(ua[mid], 2 * Um * (1 - (rc[mid] / R) ** 2))
        Re = U_phys * D * un.dx / NU
        out.append({"name": name, "cells": lat["nbr"].shape[1], "steps": s.steps, "ok": ok, "Re": Re,
                    "grad_err": G / G_exact - 1, "flow_vs_inlet": Q / (U * math.pi * R * R) - 1,
                    "profile_l2": prof_err, "sm": sm, "pm": pm, "G": G, "G_exact": G_exact, "a": a, "b": b,
                    "rc": rc[mid] / R, "ua": ua[mid] / (2 * Um), "un": un, "counts": lat["counts"]})
        print(f"  open pipe {name}: pressure gradient {out[-1]['grad_err']:+.2%} vs Hagen-Poiseuille, profile L2 "
              f"{prof_err:.2e}, flow vs inlet {out[-1]['flow_vs_inlet']:+.2%}, {s.steps} steps, ok {ok}")
    return out


def young_tsai(U0, D0, D1, Ls):
    """Pressure drop (Pa) across an axisymmetric stenosis, steady flow: Young & Tsai (1973), with the viscous
    coefficient in the Seeley & Young form Kv = 32 (0.83 Ls + 1.64 D1) / D0 (A0/A1)^2 and Kt = 1.52."""
    ar = (D0 / D1) ** 2
    Kv = 32 * (0.83 * Ls + 1.64 * D1) / D0 * ar ** 2
    return Kv * V.MU * U0 / D0 + 1.52 / 2 * V.RHO * (ar - 1) ** 2 * U0 ** 2


def test_stenosis(u_key, Re0, sev=0.7, D0_mm=2.0, Ls_mm=4.0, up_d=4, down_d=18):
    un = units(u_key)
    D = D0_mm * 1e-3 / un.dx                            # cells across
    R0 = D / 2
    Ls = Ls_mm * 1e-3 / un.dx
    s0 = up_d * D
    L = (up_d + down_d) * D + Ls

    def radius(s):
        s = np.asarray(s, float)
        x = np.clip((s - s0) / Ls, 0, 1)
        return R0 * (1 - sev * 0.5 * (1 - np.cos(2 * math.pi * x)))

    lat, sc, rc, ax = open_pipe((0, 0, 1), L, radius)
    U0_phys = Re0 * NU / (D0_mm * 1e-3)
    U = float(un.u_lu(U0_phys))
    s = lbm.Solver(lat, un.nu_lu, **SOLVER)
    s.set_sponge(D, SPONGE_TAU)
    s.set_drive([U], ramp=int(0.3 * L / U))
    s.set_outlet_pressure([0.0])
    up = (sc > 1 * D) & (sc < s0 - 0.5 * D)
    dn = (sc > L - 6 * D) & (sc < L - 1 * D)
    sc_mid = s0 + Ls / 2

    def measure():
        p, u = s.fields()
        sm, pm = section_profile(sc, p, L)
        fu = np.polyfit(sm[(sm > 1 * D) & (sm < s0 - 0.5 * D)], pm[(sm > 1 * D) & (sm < s0 - 0.5 * D)], 1)
        fd = np.polyfit(sm[(sm > L - 6 * D) & (sm < L - D)], pm[(sm > L - 6 * D) & (sm < L - D)], 1)
        dp = np.polyval(fu, sc_mid) - np.polyval(fd, sc_mid)
        return dp, sm, pm, p, u

    flow_through = L / U
    t0 = time.perf_counter()
    hist, ok = run_steady(s, measure, chunk=int(0.25 * flow_through) // 200 * 200, min_chunks=8, max_chunks=24,
                          tol=2e-3)
    wp.synchronize()
    secs = time.perf_counter() - t0
    dp_lu, sm, pm, p, u = hist[-1]
    # unsteady jets: average the last flow-through
    if len(hist) >= 4:
        dp_lu = float(np.mean([h[0] for h in hist[-4:]]))
        spread = float(np.ptp([h[0] for h in hist[-4:]]) / abs(dp_lu))
    else:
        spread = float("nan")
    dp = float(un.p(dp_lu))
    yt = young_tsai(U0_phys, D0_mm * 1e-3, D0_mm * 1e-3 * (1 - sev), Ls_mm * 1e-3)
    on_axis = rc < 0.75
    smu, umax = section_profile(sc[on_axis], u[on_axis, 2], L)
    res = {"key": u_key, "Re0": Re0, "U0": U0_phys, "D_cells": D, "cells": lat["nbr"].shape[1], "steps": s.steps,
           "ok": ok, "dp_Pa": dp, "dp_mmHg": dp / 133.322, "young_tsai_Pa": yt, "vs_young_tsai": dp / yt - 1,
           "spread_last_flow_through": spread, "u_throat_max": float(un.u(np.abs(u[:, 2]).max())),
           "u_lu_max": float(np.abs(u).max()), "tau": un.tau, "secs": secs,
           "mlups": lat["nbr"].shape[1] * s.steps / secs / 1e6, "sm_mm": sm * un.dx * 1e3, "p_Pa": un.p(pm),
           "s_axis_mm": smu * un.dx * 1e3, "u_axis": un.u(umax), "s0_mm": s0 * un.dx * 1e3, "Ls_mm": Ls_mm,
           "lat": lat, "u": u, "sc": sc, "rc": rc}
    print(f"  stenosis 70 % {u_key} (D0 {D:.0f} cells, tau {un.tau:.4f}) Re0 {Re0}: dp {dp:.0f} Pa "
          f"({res['dp_mmHg']:.1f} mmHg) vs Young-Tsai {yt:.0f} Pa ({res['vs_young_tsai']:+.0%}); jet max "
          f"{res['u_throat_max']:.2f} m/s (lattice {res['u_lu_max']:.3f}); spread {spread:.1%}; {s.steps} steps, "
          f"{res['mlups']:.0f} MLUPS, ok {ok}")
    return res


# ---------------------------------------------------------------------------------------------------------------
# 5: the artery
# ---------------------------------------------------------------------------------------------------------------


def artery_lattice(case, dx_mm):
    g = V.build(case)
    phi, origin = V.sdf_grid(g, dx_mm)
    openings = []
    for name, b in g["branches"].items():
        for key, kind in (("start", "inlet"), ("end", "outlet")):
            cap = b["caps"][key]
            if cap is not None:
                r = b["R"][0] if key == "start" else b["R"][-1]
                openings.append({"kind": kind, "centre": cap[0], "normal": cap[1], "radius": float(r),
                                 "branch": name})
    openings.sort(key=lambda o: o["kind"] != "inlet")
    lat = lbm.build_lattice(phi, origin, dx_mm, openings=openings)
    lat["openings"] = openings
    return lat, g


def test_artery(steps=20000):
    un = units("healthy")
    t0 = time.perf_counter()
    lat, g = artery_lattice("healthy", un.dx * 1e3)
    build_s = time.perf_counter() - t0
    R_in = lat["openings"][0]["radius"] * 1e-3
    U_mean = V.Q_INLET_MEAN * 1e-6 / 60 / (math.pi * R_in ** 2)
    s = lbm.Solver(lat, un.nu_lu, **SOLVER)
    s.set_sponge(SPONGE_CELLS, SPONGE_TAU)
    s.set_drive([float(un.u_lu(U_mean))], ramp=10000)
    s.set_outlet_pressure([0.0] * lat["n_outlets"])
    s.run(1000)                                         # warm-up (compile, graph capture)
    wp.synchronize()
    t0 = time.perf_counter()
    s.run(steps)
    wp.synchronize()
    secs = time.perf_counter() - t0
    p, u = s.fields()
    N = lat["nbr"].shape[1]
    mlups = N * steps / secs / 1e6
    res = {"cells": N, "build_s": build_s, "steps": s.steps, "secs": secs, "mlups": mlups, "ok": s.ok(),
           "u_max_m_s": float(un.u(np.linalg.norm(u, axis=1).max())), "counts": lat["counts"],
           "openings": [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items()}
                        for o in lat["openings"]], "U_mean_inlet": U_mean}
    print(f"  artery healthy 0.1 mm: {N / 1e6:.2f} M cells, {mlups:.0f} MLUPS, ok {res['ok']}, "
          f"max speed after {s.steps * un.dt * 1e3:.0f} ms {res['u_max_m_s']:.2f} m/s; links {lat['counts']}")
    return res


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wp.init()
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    unh = units("healthy")
    print(f"operating point healthy: dx {unh.dx * 1e3} mm, dt {unh.dt * 1e6} us, nu_lu {unh.nu_lu:.3e}, "
          f"tau {unh.tau:.5f}, 1 m/s = {unh.u_lu(1.0):.3f} lattice")
    T0 = time.perf_counter()
    print("1. Poiseuille")
    pois = test_poiseuille(unh.nu_lu)
    print("2. Womersley")
    wom = test_womersley(unh.nu_lu)
    print("3. open pipe")
    opn = test_open_pipe()
    print("4. stenosis")
    sten = [test_stenosis("narrow70", 100), test_stenosis("narrow70", 200), test_stenosis("narrow70_fine", 200)]
    print("5. artery")
    art = test_artery()
    total = time.perf_counter() - T0

    # time per heartbeat on the artery
    beat = {}
    for key in ("healthy", "narrow70", "narrow70_fine"):
        un = units(key)
        cells = art["cells"] * (1e-4 / un.dx) ** 3
        steps = V.HEART_PERIOD / un.dt
        beat[key] = {"dx_mm": un.dx * 1e3, "dt_us": un.dt * 1e6, "tau": un.tau, "cells_M": cells / 1e6,
                     "steps_per_beat": steps, "minutes_per_beat": cells * steps / (art["mlups"] * 1e6) / 60}

    # --- figure 1: Poiseuille + Womersley
    fig, axs = plt.subplots(1, 5, figsize=(24, 4.8))
    ax = axs[0]
    rr = np.linspace(0, 1, 100)
    ax.plot(rr, 1 - rr ** 2, "k-", lw=1.5, label="exact")
    for o, c in zip(pois[:3], ("#1f77b4", "#ff7f0e", "#2ca02c")):
        ax.plot(o["r"], o["u"], ".", ms=3, color=c, label=f"{o['D']} cells across (L2 {o['l2']:.1%})")
    ax.set_xlabel("r / R"), ax.set_ylabel("u / u_max(exact)")
    ax.set_title("Steady pipe flow (Poiseuille)")
    ax.legend(fontsize=8), ax.grid(alpha=0.3)
    ax = axs[1]
    Ds = np.array([o["D"] for o in pois[:3]])
    es = np.array([o["l2"] for o in pois[:3]])
    order = -np.polyfit(np.log(Ds), np.log(es), 1)[0]
    ax.loglog(Ds, es, "o-", label=f"interpolated walls (order {order:.1f})")
    ax.loglog([pois[3]["D"]], [pois[3]["l2"]], "s", color="r", label="half-way bounce-back")
    ax.loglog(Ds, es[0] * (Ds / Ds[0]) ** -2.0, "k:", lw=1, label="2nd order")
    ax.set_xlabel("cells across the pipe"), ax.set_ylabel("L2 velocity error")
    ax.set_title("Grid convergence"), ax.legend(fontsize=8), ax.grid(alpha=0.3, which="both")
    for ax, o in zip(axs[2:], wom):
        order_r = np.argsort(o["r"])
        cols = plt.cm.twilight(np.linspace(0, 1, len(o["snaps"]) + 1))
        R = o["r"][order_r]
        for (ph, un_, ua), c in zip(o["snaps"], cols):
            ax.plot(R, un_[order_r], ".", ms=2, color=c)
            ax.plot(R, ua[order_r], "-", lw=0.8, color="k", alpha=0.6)
        ax.set_title(f"Pulsatile (Womersley {o['Wo']:.1f}): L2 {o['l2']:.1%}")
        ax.set_xlabel("r / R"), ax.set_ylabel("u / 0.02")
        ax.grid(alpha=0.3)
    axs[2].text(0.02, 0.02, "dots = solver, 12 phases\nlines = exact", transform=axs[2].transAxes, fontsize=8)
    fig.suptitle(f"Periodic pipes at the artery's operating point (blood, dx 0.1 mm, dt 5 us, tau {unh.tau:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step3_pipes.png"), dpi=90)
    plt.close(fig)

    # --- figure 2: open pipes
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for o, c in zip(opn, ("#1f77b4", "#d62728")):
        un = o["un"]
        x = o["sm"] * un.dx * 1e3
        axs[0].plot(x, un.p(o["pm"]), color=c, lw=1.5, label=f"{o['name']}")
        m = (o["sm"] > o["a"]) & (o["sm"] < o["b"])
        pe = un.p(o["pm"][m].mean() - o["G_exact"] * (o["sm"][m] - o["sm"][m].mean()))
        axs[0].plot(x[m], pe, "k--", lw=1)
        axs[1].plot(o["rc"], o["ua"], ".", ms=1, color=c, alpha=0.3, label=o["name"])
    axs[0].set_xlabel("distance from the inlet (mm)"), axs[0].set_ylabel("cross-section mean pressure (Pa)")
    axs[0].set_title("Pressure along the pipe (dashed: Hagen-Poiseuille gradient)")
    axs[0].legend(fontsize=8), axs[0].grid(alpha=0.3)
    rr = np.linspace(0, 1, 50)
    axs[1].plot(rr, 1 - rr ** 2, "k-", lw=1.5, label="exact")
    axs[1].set_xlabel("r / R"), axs[1].set_ylabel("u / u_max(exact)")
    axs[1].set_title("Velocity profile, middle half of the pipe"), axs[1].legend(fontsize=8), axs[1].grid(alpha=0.3)
    ax = axs[2]
    names = [o["name"] for o in opn]
    vals = np.array([[100 * o["grad_err"], 100 * o["profile_l2"], 100 * o["flow_vs_inlet"]] for o in opn])
    xx = np.arange(3)
    for k, (o, c) in enumerate(zip(opn, ("#1f77b4", "#d62728"))):
        ax.bar(xx + 0.38 * k, vals[k], width=0.36, color=c, label=o["name"])
    ax.set_xticks(xx + 0.19, ["pressure gradient\nvs exact", "profile L2", "flow vs inlet\nsetting"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("%"), ax.set_title(f"Errors (Re {opn[0]['Re']:.0f}, D = 20 cells)"), ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step3_open_pipe.png"), dpi=90)
    plt.close(fig)

    # --- figure 3: stenosis
    fig = plt.figure(figsize=(20, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    fine = sten[2]
    lat = fine["lat"]
    un = units(fine["key"])
    grid_u = lbm.scatter(lat, np.linalg.norm(fine["u"], axis=1))
    xc = int(round(np.median(lat["coords"][fine["rc"] < 0.6][:, 0])))
    sl = un.u(grid_u[xc])                                  # (y, z) plane through the axis
    ax = fig.add_subplot(gs[0, :])
    ext = [0, sl.shape[1] * un.dx * 1e3, 0, sl.shape[0] * un.dx * 1e3]
    im = ax.imshow(sl, origin="lower", extent=ext, cmap="turbo", aspect="equal")
    plt.colorbar(im, ax=ax, label="speed (m/s)", fraction=0.02)
    ax.set_xlabel("mm along the pipe (flow ->)")
    ax.set_title(f"70 % narrowing, Re {fine['Re0']} upstream ({fine['U0']:.2f} m/s): speed in the plane through the "
                 f"axis (0.05 mm grid)")
    ax = fig.add_subplot(gs[1, 0])
    for o, c in zip(sten, ("#2ca02c", "#1f77b4", "#d62728")):
        ax.plot(o["sm_mm"], o["p_Pa"] / 133.322, color=c, lw=1.5,
                label=f"Re {o['Re0']}, {o['D_cells']:.0f} cells (dp {o['dp_mmHg']:.1f} mmHg; Young-Tsai "
                      f"{o['young_tsai_Pa'] / 133.322:.1f})")
    ax.axvspan(sten[0]["s0_mm"], sten[0]["s0_mm"] + sten[0]["Ls_mm"], color="k", alpha=0.1)
    ax.set_xlabel("mm from the inlet"), ax.set_ylabel("cross-section mean pressure (mmHg)")
    ax.set_title("Pressure along the pipe"), ax.legend(fontsize=8), ax.grid(alpha=0.3)
    ax = fig.add_subplot(gs[1, 1])
    for o, c in zip(sten, ("#2ca02c", "#1f77b4", "#d62728")):
        ax.plot(o["s_axis_mm"], o["u_axis"], color=c, lw=1.5, label=f"Re {o['Re0']}, {o['D_cells']:.0f} cells")
    ax.axvspan(sten[0]["s0_mm"], sten[0]["s0_mm"] + sten[0]["Ls_mm"], color="k", alpha=0.1)
    ax.set_xlabel("mm from the inlet"), ax.set_ylabel("mean speed near the axis (m/s)")
    ax.set_title("Jet through the narrowing (core within 0.75 cell of the axis)"), ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step3_stenosis.png"), dpi=90)
    plt.close(fig)

    # --- solver.json + report
    strip = lambda d, keys: {k: v for k, v in d.items() if k not in keys}
    summary = {
        "solver": f"lbm.py: sparse {lbm.LATTICE} lattice Boltzmann (Warp), {SOLVER}, Galilean correction, Bouzidi walls, "
                  f"equilibrium velocity inlet, pressure outlets, sponge tau {SPONGE_TAU}",
        "operating": {k: {"dx_mm": v[0] * 1e3, "dt_us": v[1] * 1e6, "tau": units(k).tau} for k, v in OPERATING.items()},
        "poiseuille": [strip(o, ("r", "u")) for o in pois],
        "womersley": [strip(o, ("r", "snaps")) for o in wom],
        "open_pipe": [strip(o, ("sm", "pm", "rc", "ua", "un")) for o in opn],
        "stenosis": [strip(o, ("sm_mm", "p_Pa", "s_axis_mm", "u_axis", "lat", "u", "sc", "rc")) for o in sten],
        "artery": art, "per_heartbeat": beat, "total_s": total}
    with open(os.path.join(HERE, "geometry", "solver.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    L = ["# Step 3 report: solver validation", "",
         f"Generated by `uv run validate.py`. Solver: `lbm.py`, sparse {lbm.LATTICE} lattice Boltzmann on the GPU with "
         "NVIDIA Warp: regularised collision with a Galilean-invariance correction, interpolated (Bouzidi) walls from "
         "the signed distance, equilibrium velocity inlet and pressure outlets at any orientation, thin sponge layers "
         f"(tau up to {SPONGE_TAU}) inside the open ends. All tests use blood's real viscosity at the artery's grid "
         "spacing and time step (tau ~ 0.502-0.505), where lattice-Boltzmann schemes are least forgiving.", "",
         "## 0. How the solver was chosen (development runs, 20-40 cells across, blood viscosity)", "",
         "| finding | consequence |", "|---|---|",
         "| D3Q19: in a straight pipe, spurious secondary vortices once the grid Reynolds number u dx / nu exceeds "
         "~30 (velocity error 0.8 % -> 6-10 %); D3Q27: none up to 73 (White & Chong 2011 report the same) | D3Q27 |",
         "| two-relaxation-time (TRT) collision: blows up at the velocity inlet within ~3000 steps; with sponges stable "
         "but loses ~5 % of the flow over 12 diameters in a pipe tilted to the grid (interpolated walls are not "
         "mass-conserving with its slow odd-moment relaxation) | regularised collision (0.4 % loss) |",
         "| bounce-back inlet: imposes a profile ~6 % too peaked, which takes ~15 diameters to relax | equilibrium inlet "
         "cells |",
         "| regularised collision near tau = 1/2: walls sit slightly inside (velocity -3.4 % at 20 cells, -0.9 % at 40, "
         "2nd order) | accepted; documented below |", "",
         f"Operating point (healthy): dx 0.1 mm, dt 5 us, lattice viscosity {unh.nu_lu:.2e}, tau {unh.tau:.5f}; "
         f"1 m/s = {unh.u_lu(1.0):.3f} lattice units.", "",
         "## 1. Steady pipe flow (Poiseuille)", "", "| cells across | walls | L2 velocity error | flow error |",
         "|---|---|---|---|"]
    for o in pois:
        L.append(f"| {o['D']} | {'interpolated' if o['bouzidi'] else 'half-way'} | {o['l2']:.2%} | "
                 f"{o['flow_err']:+.2%} |")
    L += ["", f"Convergence order (interpolated walls): {order:.2f}.", "",
          "## 2. Pulsatile pipe flow (Womersley), 20 cells across", "",
          "| Womersley number | period (steps) | L2 error over the cycle |", "|---|---|---|"]
    for o in wom:
        L.append(f"| {o['Wo']:.2f} | {o['period_steps']} | {o['l2']:.2%} |")
    L += ["", "The artery: Womersley 1.5-2.8 for the heartbeat (higher harmonics up to ~9).", "",
          "## 3. Open pipe: velocity inlet, pressure outlet (20 cells across, 12 diameters long, 0.6 m/s)", "",
          "| pipe | Re | pressure gradient vs Hagen-Poiseuille | profile L2 | flow vs inlet setting | steps |",
          "|---|---|---|---|---|---|"]
    for o in opn:
        L.append(f"| {o['name']} | {o['Re']:.0f} | {o['grad_err']:+.2%} | {o['profile_l2']:.2%} | "
                 f"{o['flow_vs_inlet']:+.2%} | {o['steps']} |")
    L += ["", "## 4. 70 % narrowing (cosine, 2 mm pipe, 4 mm long, steady)", "",
          "| grid | tau | Re upstream | upstream speed | pressure drop | Young-Tsai | difference | jet max | "
          "lattice speed max | unsteadiness (last flow-through) | MLUPS |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in sten:
        L.append(f"| {OPERATING[o['key']][0] * 1e3:.2f} mm ({o['D_cells']:.0f} cells) | {o['tau']:.4f} | {o['Re0']} | "
                 f"{o['U0']:.2f} m/s | {o['dp_Pa']:.0f} Pa ({o['dp_mmHg']:.1f} mmHg) | "
                 f"{o['young_tsai_Pa']:.0f} Pa | {o['vs_young_tsai']:+.0%} | {o['u_throat_max']:.2f} m/s | "
                 f"{o['u_lu_max']:.3f} | {o['spread_last_flow_through']:.1%} | {o['mlups']:.0f} |")
    g2 = sten[2]["dp_Pa"] / sten[1]["dp_Pa"] - 1
    L += ["", f"Grid convergence at Re 200: 0.1 mm vs 0.05 mm pressure drop differs by {g2:+.1%}.",
          "Young-Tsai is an empirical fit to experiments on different stenosis shapes; agreement within ~30 % is "
          "the expected level.", "",
          "## 5. The artery (healthy, 0.1 mm)", "",
          f"- {art['cells'] / 1e6:.2f} M fluid cells; lattice built in {art['build_s']:.1f} s; links: {art['counts']}",
          f"- speed: {art['mlups']:.0f} million cell updates per second on the RTX 3090; stable: {art['ok']}", "",
          "| setting | dx | dt | tau | cells | steps per beat | minutes per heartbeat |", "|---|---|---|---|---|---|---|"]
    for k, b in beat.items():
        L.append(f"| {k} | {b['dx_mm']} mm | {b['dt_us']} us | {b['tau']:.4f} | {b['cells_M']:.2f} M | "
                 f"{b['steps_per_beat']:.0f} | {b['minutes_per_beat']:.1f} |")
    L += ["", f"Total run time of this validation: {total / 60:.1f} min.", "",
          "Figures: `step3_pipes.png`, `step3_open_pipe.png`, `step3_stenosis.png`."]
    open(os.path.join(HERE, "reports", "step3_report.md"), "w").write("\n".join(L) + "\n")
    print(f"done in {total / 60:.1f} min")


if __name__ == "__main__":
    main()
