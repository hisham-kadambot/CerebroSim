# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-image", "matplotlib", "warp-lang>=1.18,<1.19"]
# ///
"""Step 11b: simulation geometry from a patient's own MR angiogram (patients/<P>/pre.stl, post.stl).

The lumen is the STL surface itself: a signed distance on the lattice nodes from Warp's winding-number inside test
and closest-point query (no smoothing, no idealised tube), restricted to the vessel of interest:
  - centrelines from the 'pre' scan (voxelised at 0.2 mm, skeletonised, shortest skeleton path from the inlet anchor
    to each outlet's anchor: one path for a single vessel, two for a branching); the scans are in the same
    coordinates, so the same centrelines serve the 'post' scan;
  - the lumen is kept within a tube around the cropped stretches of centreline (side branches become short closed
    stubs, neighbouring vessels are dropped) and cut flat at the inlet and each outlet;
  - each end gets a straight 6 mm extension of the same cross-section area, so the inlet / outlet, the sponge
    layers and the flow slabs sit in round, straight sections (standard practice for image-based CFD).
Only coordinates and radii leave this file: nothing from the reports (names, IDs) is read or stored.

  uv run patient.py P1              # centreline (cached in geometry/patients), lumen check and figure
"""
import argparse
import json
import os

import numpy as np
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "geometry", "patients")
VOX_DX = 0.2            # mm, voxels for the centreline
EXT_MM = 6.0            # straight extension at each open end
TUBE_MARGIN = 1.0       # mm kept beyond the vessel's own radius around the centreline (side branches -> stubs)
MIN_OPEN_MM = 0.25      # an outlet whose cut has a smaller equivalent radius stays closed (~2.5 cells at 0.1 mm)
OFF_CENTRE = 0.3        # x radius: an extension further off its lumen's centre is moved onto it

# per patient (Slicer LPS mm, +x = patient left): the inlet anchor, one path per outlet (its anchor, where the
# simulated stretch ends in mm along that path's centreline, the outlet's label = the story outlet whose calibrated
# brain resistance it gets), where the stretch starts ("start", mm along every path: they share the inlet), the path
# with the lesion (the explorer's close-up and cross-section run along it) and the probes ([name, mm from the inlet
# cut(, path)], on the lesion path unless given). Optional "grid": [dx m, dt s] for simulate.py.
# The patients' own definitions are kept beside their scans, in patients/cases.json ({"P1": {...}, ...}), which is
# not published; EXAMPLE shows the format (made-up coordinates).
EXAMPLE = {
    "PX": {"side": "right", "vessel": "carotid (ICA) and middle cerebral artery (MCA)", "title": "right ICA",
           "inlet": [-15.0, -5.0, -34.0],
           "paths": [{"to": [-26.0, -23.0, -12.0], "end": 50.0, "label": "MCA outlet", "name": "middle cerebral (MCA)"}],
           "start": 2.0, "lesion_path": 0,
           "probes": [["ICA", 5.0], ["before narrowing", 18.0], ["narrowing", 25.0], ["after narrowing", 31.0],
                      ["ICA terminus", 36.0], ["MCA (M1)", 43.0]],
           "lesion": [21.0, 29.0],
           "around": ["before narrowing", "after narrowing"],   # the pressure drop across the lesion
           "terminus": 36.0,                        # mm from the inlet cut: ICA before, the outlet's vessel after
           "scans": {"pre": "narrowed (first scan)", "post": "after (second scan)"},
           "note": "Example only: replace with your own patient's vessel and findings."},
}
CASES = os.path.join(HERE, "patients", "cases.json")
PATIENTS = json.load(open(CASES, encoding="utf-8")) if os.path.exists(CASES) else {}


def load_stl(path):
    """Binary or ASCII STL -> (vertices (n, 3) float32, triangles (m, 3) int32), duplicates merged."""
    b = open(path, "rb").read()
    n = int(np.frombuffer(b[80:84], np.uint32)[0])
    if 84 + 50 * n == len(b):
        d = np.frombuffer(b[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
        v = d["v"].reshape(-1, 3)
    else:
        v = np.array([ln.split()[1:4] for ln in b.decode(errors="ignore").splitlines()
                      if ln.strip().startswith("vertex")], np.float32)
    u, inv = np.unique(v.round(4), axis=0, return_inverse=True)
    return u.astype(np.float32), inv.reshape(-1, 3).astype(np.int32)


def mesh(patient, scan, device=None):
    v, f = load_stl(os.path.join(HERE, "patients", patient, f"{scan}.stl"))
    return wp.Mesh(points=wp.array(v, dtype=wp.vec3, device=device), indices=wp.array(f.ravel(), dtype=int,
                                                                                       device=device)), v


@wp.kernel
def k_sdf(m: wp.uint64, lo: wp.vec3, dx: float, band: float, out: wp.array3d(dtype=float)):
    """Signed distance to the surface (mm, negative inside), clipped at +-band."""
    i, j, k = wp.tid()
    p = lo + wp.vec3(float(i), float(j), float(k)) * dx
    q = wp.mesh_query_point_sign_winding_number(m, p, band, 2.0, 0.5)
    d = band
    if q.result:
        d = wp.min(wp.length(p - wp.mesh_eval_position(m, q.face, q.u, q.v)), band)
        if q.sign < 0.0:
            d = -d
    out[i, j, k] = d


def sdf(m, lo, n, dx, band=3.0, device=None):
    out = wp.empty(tuple(int(x) for x in n), dtype=float, device=device)
    wp.launch(k_sdf, dim=out.shape, inputs=[m.id, wp.vec3(*map(float, lo)), float(dx), float(band)], outputs=[out],
              device=device)
    return out.numpy()


def _arc(c):
    return np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]


def centreline(patient, k=0):
    """Smoothed centreline (0.1 mm steps) from the inlet anchor to path k's anchor in the 'pre' scan, cached."""
    path = os.path.join(GEO, f"{patient}_centreline{k or ''}.npz")
    if os.path.exists(path):
        return np.load(path)["C"]
    from scipy.ndimage import gaussian_filter1d
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree
    from skimage.morphology import skeletonize
    A, B = np.array(PATIENTS[patient]["inlet"], float), np.array(PATIENTS[patient]["paths"][k]["to"], float)
    m, _ = mesh(patient, "pre")
    lo = np.minimum(A, B) - 15.0
    n = np.ceil((np.maximum(A, B) + 15.0 - lo) / VOX_DX).astype(int) + 1
    inside = sdf(m, lo, n, VOX_DX, band=2.0) < 0
    X = lo + np.argwhere(skeletonize(inside)) * VOX_DX
    t = cKDTree(X)
    pr = t.query_pairs(1.75 * VOX_DX, output_type="ndarray")
    w = np.linalg.norm(X[pr[:, 0]] - X[pr[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[pr[:, 0], pr[:, 1]], np.r_[pr[:, 1], pr[:, 0]])), shape=(len(X),) * 2).tocsr()
    a, b = t.query(A)[1], t.query(B)[1]
    dist, pred = dijkstra(G, indices=a, return_predecessors=True)
    if not np.isfinite(dist[b]):
        raise RuntimeError(f"{patient}: no skeleton path between the anchors")
    k = [b]
    while k[-1] != a:
        k.append(pred[k[-1]])
    C = gaussian_filter1d(X[k[::-1]], sigma=5, axis=0, mode="nearest")   # ~1 mm: removes the voxel staircase
    s = _arc(C)
    si = np.arange(0.0, s[-1], 0.1)
    C = np.stack([np.interp(si, s, C[:, d]) for d in range(3)], 1)
    os.makedirs(GEO, exist_ok=True)
    np.savez(path, C=C)
    return C


def _at(C, s_mm):
    i = int(np.clip(round(s_mm / 0.1), 1, len(C) - 2))
    t = C[i + 1] - C[i - 1]
    return C[i], t / np.linalg.norm(t)


def build(case, dx_mm, device=None):
    """case 'P2_pre' -> (phi (nx, ny, nz) mm, origin, ends [inlet, outlet(s)], probes, info) on nodes origin + i dx.
    Cached in geometry/patients/<case>_<dx>.npz (stale when patient.py or the STL is newer)."""
    path = os.path.join(GEO, f"{case}_{round(dx_mm * 1000)}um.npz")
    srcs = [__file__, os.path.join(HERE, "patients", case.split("_")[0], f"{case.split('_', 1)[1]}.stl")]
    if os.path.exists(path) and os.path.getmtime(path) > max(map(os.path.getmtime, srcs)):
        z = np.load(path, allow_pickle=True)
        return z["phi"].astype(np.float32), z["lo"], list(z["ends"]), list(z["probes"]), z["info"].item()
    out = _build(case, dx_mm, device)
    os.makedirs(GEO, exist_ok=True)
    np.savez_compressed(path, phi=out[0], lo=out[1], ends=np.array(out[2], dtype=object),
                        probes=np.array(out[3], dtype=object), info=np.array(out[4], dtype=object))
    return out


def segments(patient):
    """[(centreline, its simulated stretch)] per path; every path starts at the inlet cut."""
    cfg = PATIENTS[patient]
    out = []
    for k, pth in enumerate(cfg["paths"]):
        C = centreline(patient, k)
        out.append((C, C[int(cfg["start"] / 0.1):int(pth["end"] / 0.1) + 1]))
    return out


def _build(case, dx_mm, device=None):
    from scipy import ndimage
    from scipy.spatial import cKDTree
    patient, scan = case.split("_", 1)
    cfg = PATIENTS[patient]
    s0 = cfg["start"]
    segs = segments(patient)
    m, _ = mesh(patient, scan, device)
    # the cuts: the inlet (path 0's start) and each path's outlet: (point, tangent, outward sign, label, path, index)
    c0, t0 = _at(segs[0][0], s0)
    cuts = [(c0, t0, -1.0, "inlet (ICA)", 0, 0)]
    for k, ((C, seg), pth) in enumerate(zip(segs, cfg["paths"])):
        cuts.append((*_at(C, pth["end"]), 1.0, pth["label"], k, len(seg) - 1))
    tips = [c + sign * EXT_MM * t for c, t, sign, *_ in cuts]
    allseg = np.vstack([seg for _, seg in segs])
    lo = np.minimum(allseg.min(0), np.min(tips, 0)) - 4.0
    hi = np.maximum(allseg.max(0), np.max(tips, 0)) + 4.0
    n = np.ceil((hi - lo) / dx_mm).astype(int) + 1
    phi = sdf(m, lo, n, dx_mm, device=device)
    X = lo + np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3) * dx_mm
    # keep the lumen within a tube around the stretches (radius from the 'pre' lumen along each + margin)
    near, tube = [], np.full(len(X), 8.0)
    for C, seg in segs:
        d_c, j = cKDTree(seg).query(X, distance_upper_bound=8.0)
        j = np.minimum(j, len(seg) - 1)
        r_tube = _smooth_max(_radius_along(patient, seg, device)) + TUBE_MARGIN
        tube = np.minimum(tube, np.minimum(d_c - r_tube[j], 8.0))
        near.append((j, np.isfinite(d_c)))
    phi = np.maximum(phi, tube.reshape(n))
    # flat cuts (only beyond the stretch's own ends: a plane would also slice a vessel that curves back past it),
    # then straight extensions of the same area. An outlet with (almost) no lumen at its cut (the vessel gone or a
    # thread on this scan) is not opened: its stretch stays a closed stub
    ends, closed = [], []
    for c, t, sign, label, k, jend in cuts:
        j, ok = near[k]
        nrm = sign * t                                   # outward
        ax = ((X - c) @ nrm).reshape(n)
        near_end = ((np.abs(j - jend) <= 2) & ok).reshape(n)
        phi = np.where(near_end, np.maximum(phi, ax), phi)
        area, cen = _section(phi, lo, dx_mm, c - 1.5 * dx_mm * nrm, t)        # just inside
        r = float(np.sqrt(area / np.pi))
        # the extension sits on the lumen's own centre at the cut when the centreline is well off it (a vessel thinner
        # on this scan than on the 'pre' scan the centreline comes from), else the step at the join makes a false jet
        off = (cen + 1.5 * dx_mm * nrm) - c
        print(f"{case} {label}: lumen centre {np.linalg.norm(off):.2f} mm off the centreline, r {r:.2f} mm", flush=True)
        if np.linalg.norm(off) > OFF_CENTRE * r:
            c = c + off
            ax = ((X - c) @ nrm).reshape(n)
        if r < MIN_OPEN_MM and ends:
            closed.append({"label": label, "radius": r})
            continue
        rad = np.linalg.norm((X - c) - ((X - c) @ nrm)[:, None] * nrm, axis=1).reshape(n)
        cyl = np.maximum(np.maximum(rad - r, -ax), ax - EXT_MM)
        phi = np.minimum(phi, cyl)
        ends.append({"kind": "outlet", "centre": c + EXT_MM * nrm, "normal": nrm, "label": label, "radius": r,
                     "cut_centre": c})
    # only the lumen connected to an open end (seeded in the middle of each extension): drops stray islands, keeps
    # the vessel beyond a closure (present on the scan, without flow from this side)
    lab, _ = ndimage.label(phi < 0)
    seeds = [lab[tuple(np.round((e["centre"] - 0.5 * EXT_MM * e["normal"] - lo) / dx_mm).astype(int))] for e in ends]
    keep = np.isin(lab, [s for s in seeds if s])
    phi = np.where(keep | (phi >= 0), phi, np.abs(phi))
    probes = []
    for name, s, *k in cfg["probes"]:
        p, t = _at(segs[k[0] if k else cfg["lesion_path"]][0], s0 + s)
        r = np.sqrt(_section_area(phi, lo, dx_mm, p, t) / np.pi)
        probes.append({"name": name, "point": p, "s": s, "radius": float(max(0.5 * r, 0.25))})
    info = {"patient": patient, "scan": scan, "label": cfg["scans"][scan], "side": cfg["side"], "vessel": cfg["vessel"],
            "crop": [s0, cfg["paths"][cfg["lesion_path"]]["end"]], "lesion": cfg["lesion"], "around": cfg["around"],
            "terminus": cfg["terminus"], "outlet_connected": bool(len(set(seeds)) == 1 and seeds[0]), "closed": closed}
    return phi.astype(np.float32), lo, ends, probes, info


def geom(case):
    """A story-shaped geometry dict for visualise.py / explore.py's camera, plane and close-up helpers: the whole
    stretch is the 'blocked' branch (the vessel with the lesion) and the lesion is its 'clot'."""
    patient = case.split("_")[0]
    cfg = PATIENTS[patient]
    a, b = cfg["lesion"]
    br = {("blocked" if k == cfg["lesion_path"] else f"path{k}"):
          {"C": seg, "segment_from": 0, "segment_to": len(seg), "name": cfg["paths"][k]["name"]}
          for k, (_, seg) in enumerate(segments(patient))}
    return {"branches": br, "clot": {"face_arc": a, "body_length": b - a, "ramp": 0.0}, "patient": patient,
            "terminus": cfg["terminus"],
            "up": (0.0, 0.0, 1.0), "view_dir": (0.0, 1.0, 0.0)}   # LPS: head up, seen from the front (as in Slicer)


def _smooth_max(r, half=20):
    """Running max of the radius over +-2 mm (keeps a narrowing's neighbours wide enough for the tube)."""
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(r, 2 * half + 1, mode="nearest")


def _radius_along(patient, seg, device):
    """Largest inscribed radius near each centreline point in the 'pre' lumen (distance to the wall)."""
    m, _ = mesh(patient, "pre", device)
    pts = wp.array(seg.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.empty(len(seg), dtype=float, device=device)
    wp.launch(k_wall, dim=len(seg), inputs=[m.id, pts], outputs=[out], device=device)
    return np.maximum(out.numpy(), 0.3)


@wp.kernel
def k_wall(m: wp.uint64, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    t = wp.tid()
    q = wp.mesh_query_point_sign_winding_number(m, pts[t], 5.0, 2.0, 0.5)
    out[t] = 0.0
    if q.result and q.sign < 0.0:
        out[t] = wp.length(pts[t] - wp.mesh_eval_position(m, q.face, q.u, q.v))


def _section_area(phi, lo, dx, c, t):
    return _section(phi, lo, dx, c, t)[0]


def _section(phi, lo, dx, c, t):
    """Area (mm^2) and centroid of the lumen section through c normal to t (0.05 mm samples, 4 mm around c), the piece
    around c only."""
    from scipy.ndimage import label, map_coordinates
    e1 = np.cross(t, [1.0, 0, 0] if abs(t[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    g = np.arange(-4.0, 4.0, 0.05)
    U, W = np.meshgrid(g, g, indexing="ij")
    Q = c + U.ravel()[:, None] * e1 + W.ravel()[:, None] * e2
    v = map_coordinates(phi, ((Q - lo) / dx).T, order=1, mode="nearest").reshape(U.shape) < 0
    lab, _ = label(v)
    mid = lab[len(g) // 2, len(g) // 2]
    if not mid:     # centreline just off the lumen (a bend): the largest piece within 1 mm
        near = lab[(np.hypot(U, W) < 1.0) & (lab > 0)]
        mid = np.bincount(near).argmax() if len(near) else 0
    if not mid:
        return 0.0, c
    q = (lab == mid).ravel()
    return float(q.sum() * 0.05 ** 2), Q[q].mean(0)


def section_profile(phi, lo, dx, C, s_range):
    """Equivalent radius sqrt(area / pi) of the lumen cross-section along the centreline (mm)."""
    out = []
    for s in np.arange(s_range[0] + 0.5, s_range[1] - 0.5, 0.25):
        c, t = _at(C, s)
        out.append((s, np.sqrt(_section_area(phi, lo, dx, c, t) / np.pi)))
    return np.array(out)


def check(patient, dx_mm=0.1):
    """Both scans: lumen, connectivity, cross-section profile -> reports/step11_<P>_geometry.png + .json."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wp.init()
    cfg = PATIENTS[patient]
    segs = segments(patient)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    summary = {}
    for k, scan in enumerate(cfg["scans"]):
        phi, lo, ends, probes, info = build(f"{patient}_{scan}", dx_mm)
        fl = phi < 0
        profs = []
        for (C, _), pth in zip(segs, cfg["paths"]):
            pr = section_profile(phi, lo, dx_mm, C, [cfg["start"], pth["end"]])
            pr[:, 0] -= cfg["start"]
            profs.append(pr)
        prof = profs[cfg["lesion_path"]]
        les = (prof[:, 0] >= cfg["lesion"][0]) & (prof[:, 0] <= cfg["lesion"][1])
        summary[scan] = {"fluid_cells": int(fl.sum()), "grid": list(map(int, phi.shape)), "dx_mm": dx_mm,
                         "outlet_connected": info["outlet_connected"], "closed": info["closed"],
                         "ends": [{"label": e["label"], "radius_mm": round(e["radius"], 3)} for e in ends],
                         "min_radius_in_lesion_mm": round(float(prof[les, 1].min()), 3),
                         "at_mm": round(float(prof[les][np.argmin(prof[les, 1]), 0]), 2),
                         "median_radius_mm": round(float(np.median(prof[:, 1])), 3)}
        # projection along the axis that shows the most of the vessel
        ax = axs[k]
        proj = fl.any(axis=1)
        ax.imshow(proj.T, origin="lower", cmap="Greys", extent=[lo[0], lo[0] + phi.shape[0] * dx_mm,
                                                               lo[2], lo[2] + phi.shape[2] * dx_mm])
        for p in probes:
            ax.plot(p["point"][0], p["point"][2], "o", ms=4, color="tab:orange")
        for e in ends:
            ax.plot(e["centre"][0], e["centre"][2], "s", ms=5, color="tab:blue")
        ax.set_title(f"{patient} {cfg['scans'][scan]}: lumen simulated\n(front view, x = patient left -> right side "
                     f"of image is patient left)", fontsize=9)
        ax.set_xlabel("x mm"), ax.set_ylabel("z mm (up)")
        for kp, (pr, pth) in enumerate(zip(profs, cfg["paths"])):
            axs[2].plot(pr[:, 0], pr[:, 1], color=f"C{k}", ls="-" if kp == cfg["lesion_path"] else ":",
                        label=f"{cfg['scans'][scan]}: to the {pth['label']}")
        print(f"{patient}_{scan}: {summary[scan]}", flush=True)
    axs[2].axvspan(*cfg["lesion"], color="tab:red", alpha=0.12, label="lesion (Step 11a)")
    for name, s, *_ in cfg["probes"]:
        axs[2].axvline(s, color="0.7", lw=0.6)
        axs[2].text(s, 0.02, name, rotation=90, fontsize=7, va="bottom", ha="right")
    axs[2].set_xlabel("mm along the vessel from the inlet cut"), axs[2].set_ylabel("lumen radius sqrt(area/pi), mm")
    axs[2].set_ylim(0, None), axs[2].grid(alpha=0.3), axs[2].legend(fontsize=8)
    axs[2].set_title(f"{patient}: {cfg['side']} {cfg['vessel']}\n(probes on the solid line's path)", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "reports", f"step11_{patient}_geometry.png"), dpi=80)
    json.dump(summary, open(os.path.join(GEO, f"{patient}_check.json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("patient", choices=sorted(PATIENTS))
    ap.add_argument("--dx", type=float, default=0.1)
    a = ap.parse_args()
    check(a.patient, a.dx)
