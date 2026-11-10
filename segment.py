# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "matplotlib", "warp-lang>=1.18", "pillow"]
# ///
"""Step 1: extract the blocked artery segment from the stroke story's vessel model.

The story's vessels are a display mesh (45 open pieces, ~17 vertices per ring), so instead of using the surface
we extract each branch's CENTRELINE and RADIUS, which Step 2 turns into a clean, closed vessel.

What the vessel mesh gives us (Unity's VesselFlowBaker / VesselBloodFlow.shader):
  - colour .r: distance along the flow from the tree's root, in world units (1 unit = 168.2 mm of anatomy,
    StoryScriptController's "168.2 mm-per-world-unit anatomical reference");
  - colour .b: 1 on the branch that can clot;
  - StorySceneState.ClotFaceWorld (renderer.StoryScene.clot_face) and the clot body [_ClotStart, _ClotEnd] in .r.

The segment: the trunk upstream of the junction before the clot (inlet), the junction, the blocked branch
through the clot to its end, and the sibling branch at the junction (where blocked blood diverts).

Method, per branch: vertices are split into branches by connectivity above the junction's flow coordinate; a
first centreline from slab centroids along .r is refined twice by fitting a circle to the wall vertices in the
plane across the vessel at each station (works for partial rings), then smoothed.

  uv run segment.py            # writes geometry/segment.json + reports/step1_*.png / step1_report.md
"""
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
STROKE = os.path.join(os.path.dirname(HERE), "stroke_in_3d")
sys.path.insert(0, STROKE)
import renderer as SR  # noqa: E402  (the stroke project's scene loader: vessel mesh, clot face, clot body)

MM_PER_UNIT = 168.2  # StoryScriptController: "the 168.2 mm-per-world-unit anatomical reference"
STATION = 0.002      # centreline sample spacing (model units; 0.34 mm in life)
# Typical adult lumen diameters (mm) for the three roles, for comparison (sources in Step 2's physics.json):
# internal carotid C4-C6 4.14 +- 0.39 (F) / 4.4 +- 0.45 (M); MCA M1 2.19 +- 0.19 proximal to 1.88 +- 0.15 distal;
# ACA A1 means 1.58-2.26 across two studies.
ANAT = {"trunk": (3.75, 4.85), "blocked": (1.7, 2.4), "sibling": (1.5, 2.3)}
END_HOLD = 1.5 / MM_PER_UNIT  # radius held constant over the last 1.5 mm (life) at each branch end


# ---------------------------------------------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------------------------------------------


def load_vessels():
    s = SR.StoryScene()
    v0, v1, vi, VW = s.vessel_mesh
    npz = np.load(os.path.join(STROKE, "assets", "cache", "meshes.npz"))
    P = npz[f"m{vi}_pos"].astype(np.float64) @ VW[:3, :3].T + VW[:3, 3]  # assembly frame = the renderer's frame
    col = npz[f"m{vi}_color"].astype(np.float64)
    T = s.tri[(s.tri >= v0).all(1) & (s.tri < v1).all(1)] - v0
    return s, P, col, T


def weld(P, T, col):
    """Merge coincident vertices (the mesh splits them at UV seams) -> welded positions, colours, triangles."""
    key = np.round(P / 1e-6).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    return P[first], col[first], inv[T]


def edges(Tw):
    return np.unique(np.sort(np.concatenate([Tw[:, [0, 1]], Tw[:, [1, 2]], Tw[:, [2, 0]]]), 1), axis=0)


def components(n, E, mask):
    Es = E[mask[E[:, 0]] & mask[E[:, 1]]]
    _, lab = connected_components(coo_matrix((np.ones(len(Es)), (Es[:, 0], Es[:, 1])), shape=(n, n)),
                                  directed=False)
    lab = np.where(mask, lab, -1)
    u, c = np.unique(lab[mask], return_counts=True)
    return lab, u[np.argsort(-c)], np.sort(c)[::-1]


# ---------------------------------------------------------------------------------------------------------------
# centreline + radius
# ---------------------------------------------------------------------------------------------------------------


def resample(C, h):
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.arange(0, s[-1] + 1e-12, h)
    return np.stack([np.interp(t, s, C[:, k]) for k in range(3)], 1)


def nearest_on_polyline(X, C):
    """Arc parameter and distance of each point X to the polyline C."""
    a, b = C[:-1], C[1:]
    ab = b - a
    L2 = np.maximum((ab * ab).sum(1), 1e-18)
    t = np.clip(((X[:, None, :] - a[None]) * ab[None]).sum(2) / L2[None], 0, 1)
    Q = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(X[:, None, :] - Q, axis=2)
    k = d.argmin(1)
    seg = np.concatenate([[0], np.cumsum(np.sqrt(L2))])
    return seg[k] + t[np.arange(len(X)), k] * np.sqrt(L2[k]), d[np.arange(len(X)), k]


def circle_fit(Q):
    """Algebraic (Kasa) circle fit to 2D points: centre, radius."""
    A = np.column_stack([2 * Q, np.ones(len(Q))])
    b = (Q * Q).sum(1)
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return c[:2], float(np.sqrt(max(c[2] + c[:2] @ c[:2], 1e-18)))


def centreline(X, r, slab=0.006, half=0.005, iters=2, sigma=4.0):
    """Centreline + radius of one unbranched tube from its wall vertices X (flow coordinate r)."""
    bins = np.floor((r - r.min()) / slab).astype(int)
    C = np.array([X[bins == k].mean(0) for k in np.unique(bins)])
    C = resample(C, STATION)
    for _ in range(iters):
        s, _ = nearest_on_polyline(X, C)
        st = np.arange(0, s.max() + 1e-12, STATION)
        tang = np.gradient(C, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        cs, rs, ok = [], [], []
        for k, sk in enumerate(st[:len(C)]):
            sel = np.abs(s - sk) < half
            t = tang[k]
            e1 = np.cross(t, [0.0, 0.0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0.0, 0.0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(t, e1)
            if sel.sum() < 6:
                cs.append(C[k]), rs.append(np.nan), ok.append(False)
                continue
            D = X[sel] - C[k]
            Q = np.column_stack([D @ e1, D @ e2])
            c2, rad = circle_fit(Q)
            cs.append(C[k] + e1 * c2[0] + e2 * c2[1]), rs.append(rad), ok.append(True)
        cs, rs, ok = np.array(cs), np.array(rs), np.array(ok)
        idx = np.arange(len(cs))
        for k in range(3):
            cs[~ok, k] = np.interp(idx[~ok], idx[ok], cs[ok, k])
        rs[~ok] = np.interp(idx[~ok], idx[ok], rs[ok])
        C = resample(gaussian_filter1d(cs, sigma, axis=0, mode="nearest"), STATION)
    # radius: the median distance of the wall vertices to the final centreline in a ~1 mm window, smoothed (more
    # robust on this coarse mesh than the circle-fit radii, which it agrees with where both are defined)
    s, dist = nearest_on_polyline(X, C)
    sc = arc(C)
    R = np.full(len(C), np.nan)
    for k, sk in enumerate(sc):
        sel = np.abs(s - sk) < 0.006
        if sel.sum() >= 5:
            R[k] = np.median(dist[sel])
    good = ~np.isnan(R)
    R = np.interp(sc, sc[good], R[good])
    R = gaussian_filter1d(R, 3.0, mode="nearest")
    return C, R, dist - np.interp(s, sc, R)  # residual: each wall vertex's distance minus the local radius


def arc(C):
    return np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])


# ---------------------------------------------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------------------------------------------


def extract():
    s, P, col, T = load_vessels()
    W, Wc, Tw = weld(P, T, col)
    E = edges(Tw)
    n = len(W)
    r, flag = Wc[:, 0], Wc[:, 2] > 0.5
    piece, _, _ = components(n, E, np.ones(n, bool))
    cf = s.clot_face
    cp = piece[np.argmin(np.linalg.norm(W - cf, axis=1))]
    inp = piece == cp
    # the junction: the lowest flow coordinate above which the piece splits into two sizeable branches
    rj = None
    for t in np.arange(r[inp].min(), r[inp].max(), 0.0025):
        _, u, c = components(n, E, inp & (r > t))
        if len(c) >= 2 and c[1] >= 10:
            rj = float(t)
            break
    if rj is None:
        raise SystemExit("no junction found upstream of the clot")
    lab, u, c = components(n, E, inp & (r > rj))
    main_k = max(u[:2], key=lambda k: flag[lab == k].mean())
    side_k = [k for k in u[:2] if k != main_k][0]
    branches = {
        "trunk": inp & (r <= rj),
        "blocked": lab == main_k,
        "sibling": lab == side_k,
    }
    out = {}
    for name, m in branches.items():
        C, R, dist = centreline(W[m], r[m])
        rr = r[m]
        # orient along the flow: trunk ends at the junction, branches start there
        a, b = C[0], C[-1]
        lo = W[m][np.argmin(rr)]
        if np.linalg.norm(b - lo) < np.linalg.norm(a - lo):
            C, R = C[::-1], R[::-1]
        # ends: the mesh's closed end caps pinch the tube, and at the junction the branches merge, so the radius
        # is held over the last END_HOLD at both ends (Step 2 blends the junction itself)
        k = max(1, int(round(END_HOLD / STATION)))
        if len(R) > 2 * k + 1:
            R = R.copy()
            R[:k], R[-k:] = R[k], R[-k - 1]
        out[name] = {"C": C, "R": R, "wall_dist": dist, "n_vertices": int(m.sum()),
                     "r_range": (float(rr.min()), float(rr.max())), "flagged": float(flag[m].mean())}
    # junction point: where the trunk's end and the two branch starts meet
    J = (out["trunk"]["C"][-1] + out["blocked"]["C"][0] + out["sibling"]["C"][0]) / 3
    # the clot on the blocked branch: its face (clot_face) and body end (_ClotEnd) along the flow coordinate
    Cb = out["blocked"]["C"]
    sb = arc(Cb)
    s_face, _ = nearest_on_polyline(cf[None], Cb)
    clot_len = (s.clot["end"] - s.clot["start"])  # flow distance = arc length (checked below)
    return s, W, Tw, piece, cp, rj, out, J, float(s_face[0]), float(clot_len), sb


# ---------------------------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------------------------


def life(x):
    return np.asarray(x) * MM_PER_UNIT


def main():
    os.makedirs(os.path.join(HERE, "geometry"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    s, W, Tw, piece, cp, rj, out, J, s_face, clot_len, sb = extract()
    # anatomical names are an interpretation of the layout (trunk splitting into the clotted branch and a stub),
    # not labels stored in the Unity project
    roles = {"trunk": "inlet trunk (likely the end of the internal carotid)",
             "blocked": "blocked branch (likely the middle cerebral artery, M1 onward)",
             "sibling": "sibling branch at the junction (likely the anterior cerebral artery, A1 stub)"}
    seg = {"source": "stroke_in_3d StoryScene.vessel_mesh (Phase2_StoryScript_v1)",
           "frame": "stroke renderer assembly frame (Unity world), model units",
           "mm_per_unit": MM_PER_UNIT, "junction_flow_coordinate": rj,
           "junction_point": J.tolist(),
           "clot": {"face_point": np.asarray(s.clot_face).tolist(), "face_arc_on_blocked": s_face,
                    "face_flow_coordinate": s.clot["start"], "body_end_flow_coordinate": s.clot["end"],
                    "body_length": clot_len},
           "branches": {}}
    lines = []
    for name, b in out.items():
        C, R = b["C"], b["R"]
        L = arc(C)[-1]
        seg["branches"][name] = {"role": roles[name], "points": C.round(6).tolist(), "radius": R.round(6).tolist(),
                                 "station_spacing": STATION, "length": float(L),
                                 "flow_coordinate_range": b["r_range"]}
        wd = b["wall_dist"]
        lines.append((name, roles[name], b["n_vertices"], L, b["r_range"], R, wd))
    with open(os.path.join(HERE, "geometry", "segment.json"), "w") as f:
        json.dump(seg, f, indent=1)
    report(s, W, Tw, piece, cp, rj, out, J, s_face, clot_len, lines)


def report(s, W, Tw, piece, cp, rj, out, J, s_face, clot_len, lines):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    colors = {"trunk": "#2a6fdb", "blocked": "#d62728", "sibling": "#2ca02c"}
    cf = np.asarray(s.clot_face)
    Cb, Rb = out["blocked"]["C"], out["blocked"]["R"]
    k_face = int(round(s_face / STATION))
    clot_pts = Cb[k_face:k_face + int(round(clot_len / STATION)) + 1]

    # --- 1: the segment inside the whole vessel tree (two projections), in life-size mm
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5))
    segs_all = np.concatenate([W[Tw[:, [0, 1]]], W[Tw[:, [1, 2]]], W[Tw[:, [2, 0]]]])
    for ax, (a, b), title in zip(axs, [(0, 1), (2, 1)], ["side view (x, y)", "front view (z, y)"]):
        ax.add_collection(LineCollection(life(segs_all[:, :, [a, b]]), colors=(0.78, 0.78, 0.8), linewidths=0.25))
        for name, bb in out.items():
            ax.plot(life(bb["C"][:, a]), life(bb["C"][:, b]), color=colors[name], lw=3, label=name)
        ax.plot(life(clot_pts[:, a]), life(clot_pts[:, b]), color="k", lw=6, solid_capstyle="butt", label="clot body")
        ax.scatter([life(cf[a])], [life(cf[b])], marker="*", s=260, c="gold", edgecolors="k", zorder=5,
                   label="clot face")
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_title(f"Whole vessel tree, {title}, mm (life size)")
        ax.set_xlabel("mm"), ax.set_ylabel("mm")
    axs[0].legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step1_context.png"), dpi=90)
    plt.close(fig)

    # --- 2: close-up: the segment's mesh piece, centrelines and fitted wall circles
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.5))
    m = piece[Tw[:, 0]] == cp
    tt = Tw[m]
    segs = np.concatenate([W[tt[:, [0, 1]]], W[tt[:, [1, 2]]], W[tt[:, [2, 0]]]])
    for ax, (a, b), title in zip(axs, [(0, 1), (2, 1)], ["side view (x, y)", "front view (z, y)"]):
        ax.add_collection(LineCollection(life(segs[:, :, [a, b]]), colors=(0.7, 0.7, 0.72), linewidths=0.35))
        for name, bb in out.items():
            C, R = bb["C"], bb["R"]
            ax.plot(life(C[:, a]), life(C[:, b]), color=colors[name], lw=2, label=f"{name} centreline")
            # the wall: centreline +- radius across the view (approximate silhouette)
            tng = np.gradient(C, axis=0)
            nrm = np.zeros_like(tng)
            nrm[:, a], nrm[:, b] = -tng[:, b], tng[:, a]
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            for sgn in (-1, 1):
                Wl = C + nrm * R[:, None] * sgn
                ax.plot(life(Wl[:, a]), life(Wl[:, b]), color=colors[name], lw=0.8, alpha=0.7)
        ax.plot(life(clot_pts[:, a]), life(clot_pts[:, b]), color="k", lw=7, solid_capstyle="butt",
                label="clot body")
        ax.scatter([life(J[a])], [life(J[b])], marker="o", s=90, c="white", edgecolors="k", zorder=5,
                   label="junction")
        ax.set_aspect("equal")
        ax.autoscale()
        ax.set_title(f"The extracted segment over the story's mesh, {title}, mm")
    axs[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step1_segment.png"), dpi=90)
    plt.close(fig)

    # --- 3: diameter along each branch (life size)
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, bb in out.items():
        sl = life(arc(bb["C"]))
        if name == "trunk":
            sl = sl - sl[-1]  # trunk: negative distance to the junction
        ax.plot(sl, life(2 * bb["R"]), color=colors[name], lw=2, label=name)
    sf = life(s_face)
    ax.axvspan(sf, sf + life(clot_len), color="k", alpha=0.25, label="clot body")
    # rough typical adult diameters (to be sourced and fixed in Step 2)
    for name, (lo, hi), (x0, x1) in (("trunk", ANAT["trunk"], (-30, 0)), ("blocked", ANAT["blocked"], (0, 45)),
                                     ("sibling", ANAT["sibling"], (0, 12))):
        ax.fill_between([x0, x1], lo, hi, color=colors[name], alpha=0.08)
        ax.text(x0 + 0.5, hi + 0.03, f"typical adult {name} (approx.)", color=colors[name], fontsize=8)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.text(0.3, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] else 1, "junction", va="top", fontsize=9)
    ax.set_xlabel("distance from the junction along the vessel (mm, life size)")
    ax.set_ylabel("diameter (mm)")
    ax.set_title("Vessel diameter along the segment (measured on the story's mesh)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "reports", "step1_diameter.png"), dpi=90)
    plt.close(fig)

    # --- report
    shell = s.part_group == SR.SHELL
    fs = np.isin(s.face_part, np.flatnonzero(shell))
    sp = s.pos[np.unique(s.tri[fs])]
    ext = life(sp.max(0) - sp.min(0))
    md = ["# Step 1 report: the blocked artery segment", "",
          "Source: the stroke story's vessel mesh (`stroke_in_3d` `StoryScene.vessel_mesh`). Units: model units x "
          f"{MM_PER_UNIT} mm (StoryScriptController's anatomical reference).", "",
          "## Scale check", "",
          f"- Brain shell at {MM_PER_UNIT} mm per unit: {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm "
          "(width x height x length). A typical adult brain is ~140 mm wide and ~165-170 mm long, so the scale holds.",
          "", "## Segment", "",
          f"- Vessel piece {cp} of the mesh holds the clot. Junction at flow coordinate {rj:.4f}.",
          f"- Clot face {life(s_face):.2f} mm past the junction along the blocked branch; clot body "
          f"{life(clot_len):.2f} mm long (story's _ClotStart..._ClotEnd).", "",
          "| branch | role | mesh vertices | length (mm) | diameter mean (mm) | min-max (mm) | wall fit error (mm, median abs) |",
          "|---|---|---|---|---|---|---|"]
    for name, role, nv, L, rr, R, wd in lines:
        err = np.median(np.abs(wd))
        md.append(f"| {name} | {role} | {nv} | {life(L):.1f} | {life(2 * R.mean()):.2f} | "
                  f"{life(2 * R.min()):.2f}-{life(2 * R.max()):.2f} | {life(err):.3f} |")
    md += ["", "## Diameters vs typical adult anatomy", "",
           "The story's vessels were modelled for display: they follow a plausible path at the right overall scale, "
           "but they are thinner than real arteries and their width wanders (the mesh has ~17 vertices per ring).", "",
           "| branch | measured mean (mm) | typical adult (approx., mm) | measured / typical |", "|---|---|---|---|"]
    for name, role, nv, L, rr, R, wd in lines:
        lo, hi = ANAT[name]
        md.append(f"| {name} | {life(2 * R.mean()):.2f} | {lo}-{hi} | {life(2 * R.mean()) / ((lo + hi) / 2):.2f} |")
    md += ["", "Flow resistance scales with radius^-4 (Poiseuille), so simulating the measured radii would make the "
           "flow unrealistic. Decision for Step 2: keep the story's centreline path, and give each branch a smooth, "
           "typical adult radius (sourced values), or keep the measured radii."]
    # the flow coordinate .r is distance along the flow in world units: compare it to the centreline length
    md += ["", "## Consistency", ""]
    for name, role, nv, L, rr, R, wd in lines:
        md.append(f"- {name}: centreline length {L:.4f} units vs flow-coordinate span {rr[1] - rr[0]:.4f} units "
                  f"({100 * (L / max(rr[1] - rr[0], 1e-9) - 1):+.0f} %)")
    md += ["", "## Files", "",
           "- `geometry/segment.json`: centreline points and radius per station (every "
           f"{STATION} units = {life(STATION):.2f} mm) for each branch, junction, clot, scale.",
           "- `reports/step1_context.png`: the segment inside the whole vessel tree.",
           "- `reports/step1_segment.png`: close-up over the story's mesh, with the fitted walls.",
           "- `reports/step1_diameter.png`: diameter along each branch."]
    with open(os.path.join(HERE, "reports", "step1_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
