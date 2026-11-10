# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "matplotlib", "pillow", "pyyaml", "warp-lang>=1.18,<1.19"]
# ///
"""Step 16: wall shear stress (WSS), from the saved velocity fields of every finished run.

At each vertex of the wall surface (the same marching-cubes mesh the renderer draws): the velocity of the nearby
lumen cells, minus its part along the wall normal, is fitted against the distance to the wall as
u_t(d) = a d + b d^2 (no slip at d = 0; weighted towards cells straight below the vertex), and the wall shear stress
is tau = mu a (blood mu = 3.5 mPa s). One sparse weight matrix per run, so all 40 frames of the beat cost little.
Written per run to results/<run>/wss.npz: vertices, faces, tau (40 frames x vertices x 3, Pa), time-averaged WSS
(TAWSS = mean |tau|) and the oscillatory shear index (OSI = (1 - |mean tau| / mean |tau|) / 2); plus a summary in
results/wss_summary.json.

    uv run wss.py --test             the method on Poiseuille pipes (known answer 4 mu Q / (pi R^3))
    uv run wss.py                    every finished run (or name runs: uv run wss.py healthy narrow70)
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MU = json.load(open(os.path.join(HERE, "geometry", "physics.json")))["blood"]["viscosity_Pa_s"]
K = 48            # nearest lumen cells per wall vertex
D_MIN, D_MAX = 0.02, 0.45      # mm: cells used in the fit, by distance to the wall
SIGMA = 0.15      # mm: weight falls off with sideways distance from the vertex


def weights(verts, normals, cells, depth):
    """Per vertex: its K nearest cells and the coefficient of each, so that a = sum c_j u_t,j (per mm)."""
    dist, idx = cKDTree(cells).query(verts, k=K, distance_upper_bound=0.8)
    ok = np.isfinite(dist)
    idx = np.where(ok, idx, 0)
    d = depth[idx]
    rel = cells[idx] - verts[:, None]
    side = rel - (rel * normals[:, None]).sum(-1, keepdims=True) * normals[:, None]
    w = np.exp(-(np.linalg.norm(side, axis=-1) / SIGMA) ** 2) * ok * (d > D_MIN) * (d < D_MAX)
    s11, s12, s22 = (w * d ** 2).sum(1), (w * d ** 3).sum(1), (w * d ** 4).sum(1)
    det = s11 * s22 - s12 ** 2
    quad = (det > 1e-12 * np.maximum(s11 * s22, 1e-30)) & ((w > 0.05).sum(1) >= 6)
    c = np.where(quad[:, None], w * (s22[:, None] * d - s12[:, None] * d ** 2) / np.where(quad, det, 1)[:, None],
                 w * d / np.maximum(s11, 1e-30)[:, None])                # too few cells: linear fit u_t = a d
    c[s11 <= 0] = 0.0
    return idx, c.astype(np.float32), quad, s11 > 0


def shear(U, normals, idx, c):
    """tau (Pa) at every vertex for velocity U (cells x 3, m/s)."""
    u = U[idx]                                                    # vertices x K x 3
    ut = u - (u * normals[:, None]).sum(-1, keepdims=True) * normals[:, None]
    return MU * 1e3 * (c[..., None] * ut).sum(1)                  # (m/s)/mm -> 1/s, x mu


# ------------------------------------------------------------------------------------------------- method test
def test():
    print("Pipes on the saved fields' 0.1 mm grid. Poiseuille u = u_max (1 - (r/R)^2): exact WSS = 2 mu u_max / R;\n"
          "the blunt profile u = u_max (1 - (r/R)^4) (like systole's flattened Womersley profile): 4 mu u_max / R")
    rng = np.random.default_rng(0)
    for R, pw in [(r, p) for p in (2, 4) for r in (0.4, 0.65, 1.0, 1.5, 2.0)]:
        h = 0.1
        g = np.arange(-R - h, R + h + 1e-9, h)
        x, y, z = np.meshgrid(g + rng.uniform(0, h), g + rng.uniform(0, h), np.arange(0, 3, h), indexing="ij")
        r = np.hypot(x, y)
        m = r < R
        cells = np.column_stack([x[m], y[m], z[m]])
        depth = R - r[m]
        umax = 0.5
        U = np.zeros_like(cells)
        U[:, 2] = umax * (1 - (r[m] / R) ** pw)
        th = rng.uniform(0, 2 * np.pi, 400)
        verts = np.column_stack([R * np.cos(th), R * np.sin(th), rng.uniform(1, 2, 400)])
        normals = -np.column_stack([np.cos(th), np.sin(th), np.zeros_like(th)])
        idx, c, quad, _ = weights(verts, normals, cells, depth)
        t = np.linalg.norm(shear(U, normals, idx, c), axis=1)
        exact = pw * MU * umax / (R * 1e-3)
        print(f"  {'Poiseuille' if pw == 2 else 'blunt':10s} R {R:.2f} mm ({R / h:4.1f} cells): exact {exact:6.2f} Pa, estimated {t.mean():6.2f} Pa "
              f"({100 * (t.mean() / exact - 1):+5.1f} %, spread {100 * t.std() / exact:4.1f} %), quadratic fit at "
              f"{100 * quad.mean():.0f} % of vertices")


# ------------------------------------------------------------------------------------------------- the runs
def runs():
    return sorted(r for r in os.listdir(RESULTS) if os.path.exists(os.path.join(RESULTS, r, "fields.npz"))
                  and "misaligned" not in r and not r.startswith("ai_"))     # ai_: AI estimates (Step 15)


def regions(case, verts, geo):
    """Story artery: label wall vertices by stretch: trunk, branch (sibling), MCA before / at / after the clot."""
    from vessel import RESAMPLE  # noqa: F401  (the same centrelines as the solver)
    pts, lab = [], []
    cl = geo["clot"]
    for name in ("trunk", "blocked", "sibling"):
        C = geo["branches"][name]["C"]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        if name == "blocked":
            L = np.where(s < cl["face_arc"] - cl["ramp"], "MCA before the clot",
                np.where(s <= cl["face_arc"] + cl["body_length"] + cl["ramp"], "MCA at the clot", "MCA after the clot"))
        else:
            L = np.full(len(C), "carotid (trunk)" if name == "trunk" else "other branch (ACA)")
        pts.append(C)
        lab.append(L)
    _, i = cKDTree(np.vstack(pts)).query(verts)
    return np.concatenate(lab)[i]


def one(run, device):
    import visualise as VZ
    c = VZ.Case(run, device)
    phi = c.sdf.a.numpy()
    org, h = np.array([c.sdf.origin[0], c.sdf.origin[1], c.sdf.origin[2]]), c.sdf.dx
    at = lambda a, X: map_coordinates(a, ((X - org) / h).T, order=1, mode="nearest")
    depth = -at(phi, c.pos)                                        # mm inside the lumen (phi < 0 inside)
    g = np.gradient(phi, h)
    verts = c.verts.astype(np.float64)
    n = -np.column_stack([at(gi, verts) for gi in g])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)   # into the lumen
    idx, cf, quad, any_ = weights(verts, n, c.pos, depth)
    U = c.u_host.astype(np.float32)
    tau = np.stack([shear(U[f], n, idx, cf) for f in range(len(U))]).astype(np.float32)
    mag = np.linalg.norm(tau, axis=2)
    tawss = mag.mean(0)
    osi = 0.5 * (1 - np.linalg.norm(tau.mean(0), axis=1) / np.maximum(tawss, 1e-9))
    # open ends (flow extensions' cut faces) and vertices with no cells nearby carry no wall
    wall = any_.copy()
    for e in c.ends:
        off = verts - np.asarray(e["centre"])
        wall &= ~((np.abs(off @ np.asarray(e["normal"])) < 0.15) & (np.linalg.norm(off, axis=1) < 1.2 * e["radius"] + 0.2))
    tawss[~wall], osi[~wall], mag[:, ~wall], tau[:, ~wall] = 0, 0, 0, 0
    faces = c.wall.indices.numpy().reshape(-1, 3)
    np.savez_compressed(os.path.join(RESULTS, run, "wss.npz"), verts=c.verts.astype(np.float32), faces=faces.astype(np.int32),
                        tau=tau.astype(np.float16), tawss=tawss.astype(np.float32), osi=osi.astype(np.float32),
                        wall=wall, t=c.t)
    # summary: area-weighted per stretch
    tri = verts[faces]
    area_f = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    area = np.bincount(faces.ravel(), np.repeat(area_f / 3, 3), minlength=len(verts)) * wall
    peak_t = mag.max(0)
    out = {"run": run, "vertices": int(wall.sum()), "quadratic_fit_pct": round(100 * float(quad[wall].mean()), 1),
           "all": stats(area, tawss, peak_t, osi)}
    if not VZ.is_patient(run):
        lab = regions(run, verts, c.geom)
        out["regions"] = {k: stats(area * (lab == k), tawss, peak_t, osi) for k in dict.fromkeys(lab)}
    return out


def stats(area, tawss, peak, osi):
    A = area.sum()
    if A <= 0:
        return None
    o = np.argsort(peak)
    cum = np.cumsum(area[o]) / A
    return {"area_mm2": round(float(A), 1), "tawss_mean_pa": round(float((area * tawss).sum() / A), 2),
            "tawss_max_pa": round(float(tawss[area > 0].max()), 1),
            "peak_pa": round(float(peak[o][np.searchsorted(cum, 0.999)]), 1),      # 99.9 % of the area below
            "low_tawss_area_pct": round(100 * float(area[tawss < 0.4].sum() / A), 1),
            "osi_mean": round(float((area * osi).sum() / A), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        test()
        return
    import warp as wp
    wp.config.quiet = True
    wp.init()
    path = os.path.join(RESULTS, "wss_summary.json")
    summary = json.load(open(path)) if os.path.exists(path) else {}
    for r in a.runs or runs():
        s = one(r, "cuda:0")
        summary[r] = s
        json.dump(summary, open(path, "w"), indent=1)
        al = s["all"]
        line = f"{r:18s} TAWSS {al['tawss_mean_pa']:5.2f} Pa mean, peak {al['peak_pa']:6.1f} Pa, low (<0.4 Pa) {al['low_tawss_area_pct']:4.1f} % of wall"
        if "regions" in s:
            g = s["regions"].get("MCA at the clot")
            line += f" | at the clot: TAWSS {g['tawss_mean_pa']:.2f}, peak {g['peak_pa']:.1f} Pa" if g else ""
        print(line, flush=True)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
