# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "matplotlib", "warp-lang>=1.18"]
# ///
"""Step 2: the clean vessel (life size), the clot cases and the physical set-up.

Geometry: the story's centreline PATH from Step 1 (`geometry/segment.json`), with smooth, typical adult RADII
(Decision 1b), in millimetres with the junction at the origin (axes as the stroke renderer's frame, y up).
  - branch tubes = unions of round cones along each centreline (radius interpolated along every segment);
  - the three branches are joined with a smooth union (blend radius JUNCTION_BLEND) at the junction;
  - straight flow extensions (EXT_DIAMETERS x the end diameter) at the inlet and both outlets, cut flat, so the
    inflow profile can develop and the outlets don't disturb the segment;
  - the clot: the blocked branch's radius is scaled by (1 - severity x w(s)), w = 1 over the story's clot body
    and a cosine ramp of CLOT_RAMP either side; severity = diameter reduction (0 healthy, 0.7 = 70 %, 1 = full
    occlusion, which closes the lumen).
The signed distance (negative inside the lumen) is evaluated on a voxel grid on the GPU (Warp); later steps call
`sdf_grid(build(case), dx)` at whatever resolution they need.

  uv run vessel.py            # writes geometry/vessel.json, geometry/physics.json, reports/step2_*
"""
import json
import math
import os

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))

# --- geometry choices (Decision 1b: story path + typical adult radii; sources in physics.json)
DIAM = {  # mm, along the branch from its start (trunk: from the inlet; branches: from the junction)
    "trunk": [(0.0, 4.0)],                         # internal carotid C4-C6 4.1-4.4 mm; the terminal part is narrower
    "blocked": [(0.0, 2.2), (24.0, 1.9)],          # MCA M1: 2.19 proximal -> 1.88 distal over its 24.1 mm length
    "sibling": [(0.0, 2.0)],                       # ACA A1: study means 1.58-2.26 mm
}
JUNCTION_BLEND = 0.6     # mm, smooth-union blend radius at the junction
EXT_DIAMETERS = 3.0      # straight flow extensions at the inlet and outlets, in end diameters
CLOT_RAMP = 4.0          # mm, cosine taper either side of the clot body (was 1 mm: too abrupt, a "dent")
BEND_LIMIT = 0.5         # max curvature x radius: the story's thin-vessel path is relaxed where adult radii would fold
RESAMPLE = 0.2           # mm, centreline spacing used for the distance field
CASES = {"healthy": 0.0, "narrow30": 0.3, "narrow40": 0.4, "narrow50": 0.5, "narrow60": 0.6, "narrow70": 0.7, "narrow90": 0.9,
         "occluded": 1.0}   # 40 / 60 %: Step 17, more clot sizes for the AI model
#         90 %: its 0.2 mm throat is ~4 cells at 0.05 mm, so that case is approximate (was dropped in Decision 2)

# --- physical set-up (sources in physics.json)
RHO = 1060.0             # kg/m^3
MU = 3.5e-3              # Pa s (Newtonian)
HEART_PERIOD = 1.0       # s (60 beats per minute)
Q_INLET_MEAN = 240.0     # mL/min: half of the anterior circulation's 483 mL/min (both internal carotids)
Q_MCA_TARGET = 135.0     # mL/min, healthy: between 121 (PC-MRI) and ~150 (21 % of 717 mL/min total)
PULSATILITY = 0.83       # (max - min) / mean of the inflow waveform: MCA PI 0.83 +- 0.14 (ages 20-40)


# ---------------------------------------------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------------------------------------------


def _arc(C):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])


def _resample(C, h):
    s = _arc(C)
    t = np.linspace(0.0, s[-1], max(2, int(math.ceil(s[-1] / h)) + 1))
    return np.stack([np.interp(t, s, C[:, k]) for k in range(3)], 1)


def _end_tangent(C, at_end, span=1.5):
    s = _arc(C)
    if at_end:
        k = np.searchsorted(s, s[-1] - span)
        t = C[-1] - C[min(k, len(C) - 2)]
    else:
        k = np.searchsorted(s, span)
        t = C[max(k, 1)] - C[0]
    return t / np.linalg.norm(t)


def _profile(table, s):
    xs, ds = zip(*table)
    return np.interp(s, xs, ds) / 2.0  # radius


def _curvature(C):
    s = _arc(C)
    d1 = np.gradient(C, s, axis=0)
    d2 = np.gradient(d1, s, axis=0)
    return np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)


def _limit_bends(C, table, limit=BEND_LIMIT, max_iter=4000):
    """Relax the path only where a bend is tighter than the tube allows (curvature x radius > limit; at 1 the wall
    on the inside of the bend folds). Laplacian steps on the offending stations and a few neighbours, ends fixed,
    re-sampled as it goes. Returns the path and the largest move away from the input (mm)."""
    C0 = C.copy()
    for it in range(max_iter):
        kr = _curvature(C) * _profile(table, _arc(C))
        bad = kr > limit
        bad[:2] = bad[-2:] = False
        if not bad.any():
            break
        m = np.convolve(bad.astype(float), np.ones(13), "same") > 0
        m[0] = m[-1] = False
        lap = np.zeros_like(C)
        lap[1:-1] = 0.5 * (C[:-2] + C[2:]) - C[1:-1]
        C = C + 0.5 * lap * m[:, None]
        if it % 25 == 24:
            C = _resample(C, RESAMPLE)
    C = _resample(C, RESAMPLE)
    d = np.min(np.linalg.norm(C[:, None] - C0[None], axis=2), axis=1).max()
    return C, float(d)


def build(case="healthy", severity=None):
    """The vessel for one clot case: branches (centreline mm, radius mm, caps), clot, frame."""
    seg = json.load(open(os.path.join(HERE, "geometry", "segment.json")))
    sev = CASES[case] if severity is None else float(severity)
    mm = seg["mm_per_unit"]
    J = np.array(seg["junction_point"])
    to_mm = lambda p: (np.asarray(p, float) - J) * mm
    br = {}
    # centrelines in mm; branches are joined at the junction (origin)
    trunk = to_mm(seg["branches"]["trunk"]["points"])
    blocked = to_mm(seg["branches"]["blocked"]["points"])
    sibling = to_mm(seg["branches"]["sibling"]["points"])
    trunk = np.vstack([trunk, [[0.0, 0.0, 0.0]]])
    blocked = np.vstack([[[0.0, 0.0, 0.0]], blocked])
    sibling = np.vstack([[[0.0, 0.0, 0.0]], sibling])
    for name, C in (("trunk", trunk), ("blocked", blocked), ("sibling", sibling)):
        C = _resample(C, RESAMPLE)
        C, moved = _limit_bends(C, DIAM[name])
        s = _arc(C)
        R = _profile(DIAM[name], s)
        # flow extensions: inlet before the trunk, outlets after the branches (straight, along the end tangent)
        if name == "trunk":
            t = _end_tangent(C, at_end=False)
            L = EXT_DIAMETERS * 2 * R[0]
            ext = C[0] - t[None] * np.linspace(L, 0, int(L / RESAMPLE) + 1)[:-1, None]
            C, R = np.vstack([ext, C]), np.concatenate([np.full(len(ext), R[0]), R])
            seg_start = len(ext)
            caps = {"start": (C[0], -t), "end": None}
        else:
            t = _end_tangent(C, at_end=True)
            L = EXT_DIAMETERS * 2 * R[-1]
            ext = C[-1] + t[None] * np.linspace(0, L, int(L / RESAMPLE) + 1)[1:, None]
            C, R = np.vstack([C, ext]), np.concatenate([R, np.full(len(ext), R[-1])])
            seg_start = 0
            caps = {"start": None, "end": (C[-1], t)}
        br[name] = {"C": C, "R0": R.copy(), "R": R, "caps": caps, "segment_from": seg_start, "path_moved_mm": moved,
                    "segment_to": len(C) - (len(ext) if name != "trunk" else 0)}
    # the clot on the blocked branch: the story's face point projected on the new centreline, the story's body length
    cf = to_mm(seg["clot"]["face_point"])
    Cb = br["blocked"]["C"]
    sb = _arc(Cb)
    s_face = float(sb[np.argmin(np.linalg.norm(Cb - cf, axis=1))])
    body = seg["clot"]["body_length"] * mm
    w = np.zeros_like(sb)
    inside = (sb >= s_face) & (sb <= s_face + body)
    w[inside] = 1.0
    for lo, hi, rising in ((s_face - CLOT_RAMP, s_face, True), (s_face + body, s_face + body + CLOT_RAMP, False)):
        m = (sb > lo) & (sb < hi)
        x = (sb[m] - lo) / CLOT_RAMP
        w[m] = 0.5 - 0.5 * np.cos(np.pi * (x if rising else 1 - x))
    br["blocked"]["R"] = br["blocked"]["R0"] * (1.0 - sev * w)
    return {"case": case, "severity": sev, "branches": br,
            "clot": {"face_arc": s_face, "body_length": body, "ramp": CLOT_RAMP, "weight": w},
            "frame": {"mm_per_unit": mm, "junction_model": J.tolist(),
                      "note": "mm = (model - junction_model) x mm_per_unit; axes = stroke renderer frame, y up"}}


# ---------------------------------------------------------------------------------------------------------------
# signed distance on the GPU
# ---------------------------------------------------------------------------------------------------------------


@wp.func
def _smin(a: float, b: float, k: float):
    h = wp.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


@wp.kernel
def k_sdf(origin: wp.vec3, dx: float, pts: wp.array(dtype=wp.vec3), rad: wp.array(dtype=float),
          br_start: wp.array(dtype=int), br_count: wp.array(dtype=int), cap_p: wp.array(dtype=wp.vec3),
          cap_n: wp.array(dtype=wp.vec3), cap_r: wp.array(dtype=float), blend: float, nbranch: int,
          out: wp.array3d(dtype=float)):
    i, j, l = wp.tid()
    p = origin + wp.vec3(float(i), float(j), float(l)) * dx
    d = float(1.0e9)
    for b in range(nbranch):
        db = float(1.0e9)
        s0 = br_start[b]
        for s in range(s0, s0 + br_count[b] - 1):
            a = pts[s]
            ab = pts[s + 1] - a
            t = wp.clamp(wp.dot(p - a, ab) / wp.max(wp.dot(ab, ab), 1.0e-12), 0.0, 1.0)
            r = rad[s] + (rad[s + 1] - rad[s]) * t
            db = wp.min(db, wp.length(p - (a + ab * t)) - r)
        for e in range(2):  # flat open ends: a half-space, only near the end (the plane must not cut elsewhere)
            c = b * 2 + e
            if cap_r[c] > 0.0:
                q = p - cap_p[c]
                if wp.length(q) < 2.5 * cap_r[c]:
                    db = wp.max(db, wp.dot(q, cap_n[c]))
        d = _smin(d, db, blend)
    out[i, j, l] = d


def _packed(g):
    names = list(g["branches"])
    pts, rad, start, count, cp, cn, cr = [], [], [], [], [], [], []
    for name in names:
        b = g["branches"][name]
        start.append(sum(len(x) for x in pts)), count.append(len(b["C"]))
        pts.append(b["C"]), rad.append(b["R"])
        for key, r in (("start", b["R"][0]), ("end", b["R"][-1])):
            cap = b["caps"][key]
            cp.append(cap[0] if cap else np.zeros(3)), cn.append(cap[1] if cap else np.zeros(3))
            cr.append(float(r) if cap else 0.0)
    return names, np.vstack(pts), np.concatenate(rad), start, count, np.array(cp), np.array(cn), np.array(cr)


def bounds(g, margin=0.5):
    allp = np.vstack([b["C"] for b in g["branches"].values()])
    rmax = max(float(b["R0"].max()) for b in g["branches"].values())
    return allp.min(0) - rmax - margin, allp.max(0) + rmax + margin


def sdf_grid(g, dx, lo=None, hi=None, device=None):
    """Signed distance (mm, negative inside) on voxel centres origin + (i, j, k) dx -> (array (nx, ny, nz), origin)."""
    if lo is None:
        lo, hi = bounds(g)
    n = np.ceil((np.asarray(hi) - np.asarray(lo)) / dx).astype(int) + 1
    names, pts, rad, start, count, cp, cn, cr = _packed(g)
    dev = device or wp.get_preferred_device()
    out = wp.empty(tuple(int(x) for x in n), dtype=float, device=dev)
    wp.launch(k_sdf, dim=out.shape, device=dev, inputs=[
        wp.vec3(*map(float, lo)), float(dx), wp.array(pts, dtype=wp.vec3, device=dev),
        wp.array(rad.astype(np.float32), dtype=float, device=dev), wp.array(np.array(start, np.int32), dtype=int,
                                                                             device=dev),
        wp.array(np.array(count, np.int32), dtype=int, device=dev), wp.array(cp, dtype=wp.vec3, device=dev),
        wp.array(cn, dtype=wp.vec3, device=dev), wp.array(cr.astype(np.float32), dtype=float, device=dev),
        float(JUNCTION_BLEND), len(names)], outputs=[out])
    return out.numpy(), np.asarray(lo, float)


# ---------------------------------------------------------------------------------------------------------------
# physical set-up
# ---------------------------------------------------------------------------------------------------------------


def inflow_waveform(n=200):
    """Synthetic inflow over one beat, Q(t) / Q_mean (mean 1), with (max - min) / mean = PULSATILITY: a systolic
    peak, an augmented late-systolic shoulder (as in older adults' carotid waveforms) and a diastolic decay,
    smoothed to its first 10 harmonics so it is periodic and smooth."""
    t = np.arange(n) / n
    g = lambda c, w: np.exp(-0.5 * ((t - c) / w) ** 2)
    shape = 0.15 + 1.0 * g(0.13, 0.045) + 0.45 * g(0.30, 0.06) + 0.12 * g(0.45, 0.04) + 0.25 * np.exp(-(t / 0.9))
    F = np.fft.rfft(shape)
    F[11:] = 0
    shape = np.fft.irfft(F, n)
    shape = shape / shape.mean()
    shape = 1 + (shape - 1) * PULSATILITY / (shape.max() - shape.min())
    return t * HEART_PERIOD, shape


def physics():
    t, q = inflow_waveform()
    d_trunk, d_mca = DIAM["trunk"][0][1] / 1e3, DIAM["blocked"][0][1] / 1e3
    qi, qm = Q_INLET_MEAN / 6e7, Q_MCA_TARGET / 6e7  # m^3/s
    v_trunk, v_mca = qi / (math.pi * d_trunk ** 2 / 4), qm / (math.pi * d_mca ** 2 / 4)
    nu = MU / RHO
    omega = 2 * math.pi / HEART_PERIOD
    return {
        "blood": {"density_kg_m3": RHO, "viscosity_Pa_s": MU, "kinematic_viscosity_m2_s": nu, "model": "Newtonian"},
        "heart": {"period_s": HEART_PERIOD},
        "inlet": {"mean_flow_mL_min": Q_INLET_MEAN, "pulsatility_index": PULSATILITY,
                  "waveform_t_s": t.round(4).tolist(), "waveform_Q_over_mean": q.round(5).tolist(),
                  "profile": "Womersley/developed via the inlet extension (decided in Step 3)"},
        "outlets": {"healthy_MCA_target_mL_min": Q_MCA_TARGET,
                    "healthy_sibling_mL_min": Q_INLET_MEAN - Q_MCA_TARGET,
                    "model": "resistance (RCR) per outlet, tuned in Step 4 to the healthy split"},
        "walls": "rigid, no-slip",
        "derived": {
            "mean_velocity_trunk_m_s": v_trunk, "mean_velocity_MCA_m_s": v_mca,
            "peak_velocity_MCA_m_s": v_mca * q.max(),
            "Reynolds_trunk": RHO * v_trunk * d_trunk / MU, "Reynolds_MCA_mean": RHO * v_mca * d_mca / MU,
            "Reynolds_MCA_peak": RHO * v_mca * q.max() * d_mca / MU,
            "Womersley_trunk": d_trunk / 2 * math.sqrt(omega / nu), "Womersley_MCA": d_mca / 2 * math.sqrt(omega / nu),
        },
        "geometry_radii_mm": {k: v for k, v in DIAM.items()},
        "sources": {
            "diameters": [
                "ICA C4-C6 inner diameter 4.4 +- 0.45 mm (M), 4.14 +- 0.39 mm (F): https://ama.ba/index.php/ama/article/download/265/pdf",
                "MCA M1 width 2.19 +- 0.19 (proximal), 1.99 +- 0.12 (middle), 1.88 +- 0.15 mm (distal), M1 length 24.1 mm (CTA, normal side): https://ejtcm.gumed.edu.pl/articles/83",
                "A1 2.26 / 2.25 mm (MRA, n=200): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12165878/ ; A1 1.58 / 1.64 mm, M1 2.13 / 2.10 mm: https://journals.viamedica.pl/folia_morphologica/article/download/FM.a2017.0004/41513",
            ],
            "flows": [
                "MCA flow 121 +- 28 mL/min (PC-MRI, n=88): https://link.springer.com/article/10.1007/s003300000378",
                "MCA 21 % of total cerebral blood flow 717 mL/min (Zarrinkoob et al., JCBFM 35:648)",
                "Anterior circulation 483 +- 87 mL/min, total 695 mL/min (QMRA, 83 adults): https://pmc.ncbi.nlm.nih.gov/articles/PMC8134363",
                "Flows fall with age (~2.6 mL/min per year for total flow): https://pmc.ncbi.nlm.nih.gov/articles/PMC4426749",
            ],
            "waveform": [
                "MCA pulsatility index 0.83 +- 0.14 (ages 20-40), 0.82 +- 0.13 (40-60): https://ejns.springeropen.com/articles/10.1186/s41984-021-00114-0/tables/2",
                "Older adults' carotid waveforms: augmented late-systolic peak (Hoi et al. 2010, Physiol Meas 31:291): https://scholarlyworks.lvhn.org/radiology-diagnostic-medical-imaging/1978",
                "The waveform shape itself is synthetic (a smooth stand-in with the sourced pulsatility).",
            ],
            "blood": [
                "Newtonian viscosity 0.0035 Pa s (as in Bernabeu et al. 2013): https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2021.718540/pdf",
                "Newtonian is a reasonable simplification in cerebral arteries, with wall-shear-stress differences up to ~30 % reported: https://reference-global.com/article/10.2478/ama-2024-0002",
                "Density 1060 kg/m^3: the usual value in blood-flow models (not separately sourced here).",
            ],
        },
    }


# ---------------------------------------------------------------------------------------------------------------
# checks + report
# ---------------------------------------------------------------------------------------------------------------


def trilinear(F, origin, dx, P):
    from scipy.ndimage import map_coordinates
    idx = ((np.asarray(P) - origin) / dx).T
    return map_coordinates(F, idx, order=1, mode="nearest")


def section_diameters(F, origin, dx, C, every=0.5, half=4.0, res=0.04):
    """Equivalent lumen diameter (from the area with sdf < 0) on cross-section discs along a centreline."""
    s = _arc(C)
    out = []
    for sk in np.arange(0, s[-1], every):
        k = min(np.searchsorted(s, sk), len(C) - 2)
        t = C[k + 1] - C[k]
        t /= np.linalg.norm(t)
        e1 = np.cross(t, [0, 0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(t, e1)
        u = np.arange(-half, half, res)
        U, V = np.meshgrid(u, u)
        P = C[k][None] + U.reshape(-1, 1) * e1 + V.reshape(-1, 1) * e2
        inside = trilinear(F, origin, dx, P) < 0
        # only the region connected to the centre (another branch can cross the disc near the junction)
        from scipy.ndimage import label
        lab, _ = label(inside.reshape(U.shape))
        c = lab[len(u) // 2, len(u) // 2]
        area = (lab == c).sum() * res * res if c > 0 else 0.0
        out.append((sk, 2 * math.sqrt(area / math.pi)))
    return np.array(out)


def clearance(g):
    """Closest approach between tube walls that should not touch: different branches away from the junction, and
    the same branch folded back on itself (points far apart along it)."""
    rows = []
    br = g["branches"]
    names = list(br)
    for a in names:
        for b in names:
            Ca, Ra, Cb, Rb = br[a]["C"], br[a]["R0"], br[b]["C"], br[b]["R0"]
            D = np.linalg.norm(Ca[:, None] - Cb[None], axis=2) - Ra[:, None] - Rb[None]
            if a == b:
                sa = _arc(Ca)
                far = np.abs(sa[:, None] - sa[None]) > 4 * (Ra[:, None] + Ra[None])
                D = np.where(far, D, np.inf)
            else:
                # away from the junction (origin): both points farther than 3 x the larger radius from it
                na = np.linalg.norm(Ca, axis=1) > 3 * max(Ra.max(), Rb.max()) + 2
                nb = np.linalg.norm(Cb, axis=1) > 3 * max(Ra.max(), Rb.max()) + 2
                D = np.where(na[:, None] | nb[None], D, np.inf)
            if np.isfinite(D).any():
                i, j = np.unravel_index(np.argmin(D), D.shape)
                rows.append((a, b, float(D[i, j]), Ca[i], Cb[j]))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import generate_binary_structure, label

    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    wp.init()
    dx = 0.1
    phys = physics()
    with open(os.path.join(HERE, "geometry", "physics.json"), "w") as f:
        json.dump(phys, f, indent=1)
    g0 = build("healthy")
    lo, hi = bounds(g0)
    results, grids = {}, {}
    import time
    for case in CASES:
        g = build(case)
        t0 = time.perf_counter()
        F, origin = sdf_grid(g, dx, lo, hi)
        ms = (time.perf_counter() - t0) * 1000
        fluid = F < 0
        lab, nlab = label(fluid, generate_binary_structure(3, 2))  # 18-neighbour, like a D3Q19 lattice
        # which open end each fluid component touches
        ends = {}
        for name, b in g["branches"].items():
            for key in ("start", "end"):
                cap = b["caps"][key]
                if cap is None:
                    continue
                p = cap[0] - cap[1] * dx * 1.5  # just inside the open face
                i = np.round((p - origin) / dx).astype(int)
                ends[f"{name}.{key}"] = int(lab[tuple(i)])
        sizes = np.bincount(lab.ravel())[1:]
        results[case] = {"grid": list(F.shape), "fluid_voxels": int(fluid.sum()),
                         "fluid_volume_mm3": float(fluid.sum() * dx ** 3), "components": int(nlab),
                         "component_sizes": sorted(sizes.tolist(), reverse=True)[:4], "ends": ends,
                         "sdf_ms": ms}
        grids[case] = (F, origin, g)
        print(f"{case:9s} grid {F.shape} fluid {fluid.sum() / 1e6:.2f} M voxels ({fluid.sum() * dx ** 3:.0f} mm^3), "
              f"components {nlab}, ends {ends}, SDF {ms:.0f} ms")

    # cross-section diameters along the blocked branch, per case; and along trunk / sibling for the healthy case
    Fh, oh, gh = grids["healthy"]
    diam = {case: section_diameters(F, o, dx, g["branches"]["blocked"]["C"]) for case, (F, o, g) in grids.items()}
    diam_t = section_diameters(Fh, oh, dx, gh["branches"]["trunk"]["C"])
    diam_s = section_diameters(Fh, oh, dx, gh["branches"]["sibling"]["C"])
    clot = gh["clot"]

    # --- figure 1: diameters
    fig, axs = plt.subplots(1, 2, figsize=(16, 5.5), gridspec_kw={"width_ratios": [2, 1]})
    ax = axs[0]
    for case, d in diam.items():
        ax.plot(d[:, 0], d[:, 1], lw=2, label=f"{case} (voxel lumen)")
    sb = _arc(gh["branches"]["blocked"]["C"])
    ax.plot(sb, 2 * gh["branches"]["blocked"]["R0"], "k--", lw=1, label="target (healthy)")
    ax.axvspan(clot["face_arc"], clot["face_arc"] + clot["body_length"], color="k", alpha=0.15, label="clot body")
    seg_end = sb[gh["branches"]["blocked"]["segment_to"] - 1]
    ax.axvline(seg_end, color="grey", ls=":", lw=1)
    ax.text(seg_end + 0.3, 2.5, "outlet extension", fontsize=8, color="grey")
    ax.set_xlabel("distance from the junction along the blocked branch (mm)")
    ax.set_ylabel("lumen diameter (mm)")
    ax.set_title("Blocked branch: lumen diameter per clot case (measured on the 0.1 mm voxel grid)")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    ax = axs[1]
    st = _arc(gh["branches"]["trunk"]["C"])
    ax.plot(diam_t[:, 0] - st[-1], diam_t[:, 1], lw=2, label="trunk (voxel)")
    ax.plot(st - st[-1], 2 * gh["branches"]["trunk"]["R0"], "k--", lw=1)
    ax.plot(diam_s[:, 0], diam_s[:, 1], lw=2, label="sibling (voxel)")
    ss = _arc(gh["branches"]["sibling"]["C"])
    ax.plot(ss, 2 * gh["branches"]["sibling"]["R0"], "k--", lw=1, label="targets")
    ax.set_xlabel("distance from the junction (mm; trunk negative)")
    ax.set_title("Trunk and sibling (healthy)")
    ax.grid(alpha=0.3), ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step2_diameters.png"), dpi=90)
    plt.close(fig)

    # --- figure 2: longitudinal sections through the clot, per case (plane: local tangent x a normal)
    Cb = gh["branches"]["blocked"]["C"]
    k = int(np.searchsorted(sb, clot["face_arc"] + clot["body_length"] / 2))
    t = Cb[min(k + 5, len(Cb) - 1)] - Cb[max(k - 5, 0)]
    t /= np.linalg.norm(t)
    e1 = np.cross(t, [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    u = np.arange(-6, 6, 0.03)
    v = np.arange(-2.5, 2.5, 0.03)
    U, V = np.meshgrid(u, v)
    P = Cb[k][None] + U.reshape(-1, 1) * t + V.reshape(-1, 1) * e1
    fig, axs = plt.subplots(1, len(CASES), figsize=(4 * len(CASES), 2.6))
    for ax, (case, (F, o, g)) in zip(axs, grids.items()):
        S = trilinear(F, o, dx, P).reshape(U.shape)
        ax.imshow(S < 0, extent=[u[0], u[-1], v[0], v[-1]], origin="lower", cmap="Reds", vmin=0, vmax=1.6)
        ax.contour(U, V, S, levels=[0], colors="k", linewidths=0.8)
        ax.set_title(f"{case} (severity {CASES[case]:.0%})", fontsize=10)
        ax.set_xlabel("mm along the vessel (flow ->)")
        ax.set_aspect("equal")
    fig.suptitle("Lumen (red) in a plane along the blocked branch through the clot", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step2_clot_sections.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    # --- figure 3: the whole healthy vessel (lumen projections) + the inflow waveform
    fig, axs = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={"width_ratios": [1.3, 1, 0.9]})
    fluid = Fh < 0
    names = {"trunk": "#2a6fdb", "blocked": "#d62728", "sibling": "#2ca02c"}
    for ax, (axis, (a, b)) in zip(axs[:2], [(2, (0, 1)), (0, (2, 1))]):
        proj = fluid.any(axis=axis)
        ext = [oh[a], oh[a] + dx * fluid.shape[a], oh[b], oh[b] + dx * fluid.shape[b]]
        ax.imshow(proj.T, extent=ext, origin="lower", cmap="Greys", vmin=0, vmax=2.5)
        for name, bch in gh["branches"].items():
            C = bch["C"]
            i0, i1 = bch["segment_from"], bch["segment_to"]
            ax.plot(C[:, a], C[:, b], color=names[name], lw=0.8, ls=":")
            ax.plot(C[i0:i1, a], C[i0:i1, b], color=names[name], lw=1.6, label=name)
        Cc = Cb[(sb >= clot["face_arc"]) & (sb <= clot["face_arc"] + clot["body_length"])]
        ax.plot(Cc[:, a], Cc[:, b], color="k", lw=5, label="clot")
        ax.set_aspect("equal")
        ax.set_xlabel("mm"), ax.set_ylabel("mm")
        ax.set_title(f"Healthy lumen, projected along {'xyz'[axis]} (dotted = flow extensions)")
    axs[0].legend(loc="best", fontsize=8)
    tt, qq = inflow_waveform()
    ax = axs[2]
    ax.plot(tt, qq * Q_INLET_MEAN, lw=2)
    ax.axhline(Q_INLET_MEAN, color="k", ls="--", lw=1)
    ax.set_xlabel("time in the beat (s)"), ax.set_ylabel("inflow (mL/min)")
    ax.set_title(f"Inlet flow over one beat (mean {Q_INLET_MEAN:.0f} mL/min, pulsatility {PULSATILITY})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step2_vessel.png"), dpi=90)
    plt.close(fig)

    # --- geometry description for later steps
    desc = {"cases": CASES, "dx_checked_mm": dx, "grid_origin_mm": lo.tolist(), "grid_hi_mm": hi.tolist(),
            "junction_blend_mm": JUNCTION_BLEND, "extension_diameters": EXT_DIAMETERS, "clot_ramp_mm": CLOT_RAMP,
            "clot": {"face_arc_mm": clot["face_arc"], "body_length_mm": clot["body_length"]},
            "frame": gh["frame"], "diameters_mm": DIAM,
            "branches": {n: {"centreline_mm": b["C"].round(4).tolist(), "radius_healthy_mm": b["R0"].round(4).tolist(),
                             "segment_index_range": [b["segment_from"], b["segment_to"]],
                             "open_end": {k: (None if c is None else {"point": c[0].tolist(), "outward": c[1].tolist()})
                                          for k, c in b["caps"].items()}}
                         for n, b in gh["branches"].items()},
            "checks": results}
    with open(os.path.join(HERE, "geometry", "vessel.json"), "w") as f:
        json.dump(desc, f, indent=1)

    # --- report
    d = phys["derived"]
    cl = clearance(gh)
    md = ["# Step 2 report: clean vessel, clot cases, physical set-up", "",
          "## Geometry", "",
          "- Path: the story's centrelines (Step 1), life size (168.2 mm per model unit), junction at the origin.",
          f"- Radii (Decision 1b, typical adult): trunk {DIAM['trunk'][0][1]} mm; blocked branch "
          f"{DIAM['blocked'][0][1]} -> {DIAM['blocked'][1][1]} mm over its first {DIAM['blocked'][1][0]:.0f} mm; "
          f"sibling {DIAM['sibling'][0][1]} mm.",
          f"- Bends: the story's path was drawn for thin vessels; with adult radii some bends were tighter than the "
          f"tube (wall folding on the inside). The path is relaxed only there, to curvature x radius <= {BEND_LIMIT}: "
          + ", ".join(f"{n} moved at most {b['path_moved_mm']:.2f} mm" for n, b in gh["branches"].items()) + "; "
          + "max curvature x radius now " + ", ".join(
              f"{n} {float((_curvature(b['C']) * b['R0'])[2:-2].max()):.2f}" for n, b in gh["branches"].items()) + ".",
          f"- Junction: smooth union, blend {JUNCTION_BLEND} mm. Flow extensions: {EXT_DIAMETERS:.0f} diameters, "
          "straight, cut flat (inlet before the trunk, outlets after both branches).",
          f"- Clot: at the story's position, {clot['face_arc']:.2f} mm past the junction; body "
          f"{clot['body_length']:.2f} mm (story) + {CLOT_RAMP} mm cosine ramps; severity = diameter reduction.", "",
          "## Checks (0.1 mm voxels)", "",
          "| case | fluid voxels | volume (mm³) | connected pieces | inlet / blocked outlet / sibling outlet piece | min diameter at the clot (mm) | SDF time |",
          "|---|---|---|---|---|---|---|"]
    for case, r in results.items():
        dd = diam[case]
        m = (dd[:, 0] > clot["face_arc"] - 1) & (dd[:, 0] < clot["face_arc"] + clot["body_length"] + 1)
        e = r["ends"]
        md.append(f"| {case} | {r['fluid_voxels'] / 1e6:.2f} M | {r['fluid_volume_mm3']:.0f} | {r['components']} | "
                  f"{e.get('trunk.start')} / {e.get('blocked.end')} / {e.get('sibling.end')} | "
                  f"{dd[m, 1].min():.2f} | {r['sdf_ms']:.0f} ms |")
    # analytic volume of the healthy tubes (round-cone sum, ignores the junction overlap) vs voxels
    va = sum(float(np.sum(np.pi * ((b["R0"][1:] + b["R0"][:-1]) / 2) ** 2 * np.diff(_arc(b["C"]))))
             for b in gh["branches"].values())
    md += ["", f"- Healthy volume: voxels {results['healthy']['fluid_volume_mm3']:.0f} mm³ vs tube formula {va:.0f} mm³ "
           f"({100 * (results['healthy']['fluid_volume_mm3'] / va - 1):+.1f} %; the formula double-counts the junction "
           "overlap and ignores the blend).",
           f"- Healthy diameters on the grid vs targets: trunk median {np.median(diam_t[3:-3, 1]):.2f} mm "
           f"(target {DIAM['trunk'][0][1]}), sibling median {np.median(diam_s[3:-3, 1]):.2f} mm "
           f"(target {DIAM['sibling'][0][1]}).",
           "- Connectivity (18-neighbour, as a D3Q19 lattice sees it): the full occlusion must split the lumen in two "
           "(inlet + sibling vs the blocked branch beyond the clot); every other case is one piece.",
           "- Clearance between walls that must not touch (adult radii on the story's path):"]
    for a, b, dmin, pa, pb in cl:
        md.append(f"  - {a} vs {b}: {dmin:+.2f} mm" + (" **(touching: merged walls)**" if dmin < 0.2 else ""))
    md += ["", "## Physical set-up (`geometry/physics.json`)", "",
           f"- Blood: Newtonian, density {RHO:.0f} kg/m³, viscosity {MU * 1e3:.1f} mPa·s.",
           f"- Heart period {HEART_PERIOD} s. Inlet: mean {Q_INLET_MEAN:.0f} mL/min, synthetic waveform with "
           f"pulsatility index {PULSATILITY}.",
           f"- Outlets: resistances tuned in Step 4 so the healthy MCA carries ~{Q_MCA_TARGET:.0f} mL/min.",
           f"- Derived: mean velocity trunk {d['mean_velocity_trunk_m_s']:.2f} m/s, MCA {d['mean_velocity_MCA_m_s']:.2f} m/s "
           f"(peak {d['peak_velocity_MCA_m_s']:.2f}); Reynolds trunk {d['Reynolds_trunk']:.0f}, MCA "
           f"{d['Reynolds_MCA_mean']:.0f} (peak {d['Reynolds_MCA_peak']:.0f}): laminar; Womersley trunk "
           f"{d['Womersley_trunk']:.1f}, MCA {d['Womersley_MCA']:.1f}.", "",
           "Sources: listed in `geometry/physics.json` (`sources`).", "",
           "## Files", "",
           "- `vessel.py`: `build(case)` and `sdf_grid(geometry, dx)` for later steps (any resolution).",
           "- `geometry/vessel.json`, `geometry/physics.json`.",
           "- `reports/step2_vessel.png`, `step2_diameters.png`, `step2_clot_sections.png`."]
    with open(os.path.join(HERE, "reports", "step2_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
