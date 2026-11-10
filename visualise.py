# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "matplotlib", "pillow", "pyyaml", "warp-lang>=1.18"]
# ///
"""Step 5: show the simulated flow (NVIDIA Warp ray tracing of Step 4's results).

What is drawn, per case (healthy / 70 % narrowing / full occlusion), at any time in the heartbeat:
  - the vessel wall (marching cubes of Step 2's signed distance, Warp) as glass or solid, coloured by the blood
    pressure just inside it, or by the pressure DIFFERENCE from the healthy case;
  - blood particles carried by the simulated velocity (RK2 on the GPU, respawned at the inlet when they leave),
    drawn as small spheres with short trails, coloured by speed;
  - a cut plane through the clot: the near half of the vessel removed, optionally the section filled with the speed
    or pressure map;
  - the context: the story's brain (stroke renderer) with the simulated segment marked;
  - on-screen numbers and colour bars (PIL, the story's Arial).
Fields: the last heartbeat of each case, 40 frames on a 0.1 mm grid (`results/<case>/fields.npz`), interpolated in
time and trilinearly in space (velocity -> 0 at the wall).

  uv run visualise.py            # reports/step5_*.png + step5_report.md (stills of each view + checks vs the data)

The `Scene` class is what Step 6's sequence drives.
"""
import json
import math
import os
import re
import sys
import time

import numpy as np
import warp as wp
from PIL import Image, ImageDraw

import patient as PT
import vessel as V

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
REPORTS = os.path.join(HERE, "reports")
STROKE = os.path.join(os.path.dirname(HERE), "stroke_in_3d")

CASES = ("healthy", "narrow70", "occluded")
LABEL = {"healthy": "Healthy", "narrow30": "30 % narrowed", "narrow50": "50 % narrowed", "narrow70": "70 % narrowed",
         "narrow90": "90 % narrowed", "occluded": "Fully blocked"}
LABEL.update({f"{p}_{k}": f"{p}: {v}" for p, c in PT.PATIENTS.items() for k, v in c["scans"].items()})


def is_patient(case):
    return case.split("_")[0] in PT.PATIENTS


def clot_of(case):
    """The clot case of a results folder (narrow70_flow130 -> narrow70: same vessel, another inflow level)."""
    return re.split(r"_(?:flow|map)\d", case)[0]
PERIOD = V.HEART_PERIOD
SPEED_MAX = 4.0            # m/s, colour scale (shared by all cases; the 70 % jet's core saturates)
P_RANGE = (0.0, 150.0)     # mmHg, colour scale
DP_RANGE = 90.0            # mmHg, +- for the pressure difference
WSS_RANGE = (0.1, 100.0)   # Pa, wall shear stress colour scale (log; wss.py), above saturates
WALL_INSET = 0.15          # mm: wall pressure is read this far inside the lumen
N_PARTICLES = 16000
TRAIL = 5                  # spheres per particle (the particle and 4 older positions)
TRAIL_EVERY_MS = 0.25      # sim time between trail positions
PARTICLE_R = 0.07          # mm
SUBSTEP_MS = 0.02          # advection step (5.5 m/s -> 0.11 mm, about one bin)
SEED_DEPTH = (2.5, 2.7)    # mm inside the inlet where particles are released
EXIT_DEPTH = 0.3           # mm: particles leave this far inside an open end (the data stop at the cap)
GLASS = 0.22               # front wall opacity (glass look), more at grazing angles
BACKGROUND = (0.0, 0.0, 0.0)
SDF_DX = 0.1               # mm, signed distance grid (inside tests, wall surface)
KEEP_R = 9.0               # mm: close-ups show the artery within this distance of the clot
PEAK_T = 0.125             # s: peak systole in the last beat (Step 4 inflow maximum)


# ---------------------------------------------------------------------------------------------------------------
# GPU data: signed distance, fields, colour tables
# ---------------------------------------------------------------------------------------------------------------


@wp.struct
class Grid:
    a: wp.array3d(dtype=float)
    origin: wp.vec3
    dx: float


@wp.struct
class Field:
    grid: wp.array3d(dtype=int)     # bin index or -1 (wall / outside)
    u: wp.array(dtype=wp.vec3)      # frames x bins, m/s (= mm/ms)
    p: wp.array(dtype=float)        # frames x bins, mmHg
    n: int
    origin: wp.vec3
    dx: float
    f0: int
    f1: int
    w: float


WALL_INSET_K = wp.constant(WALL_INSET)
EXIT_DEPTH_K = wp.constant(EXIT_DEPTH)
STUCK_STEPS = wp.constant(25)   # advection substeps (0.5 ms) held at the wall before a particle is recycled


@wp.func
def sdf_at(G: Grid, x: wp.vec3) -> float:
    g = (x - G.origin) / G.dx
    i = int(wp.floor(g[0]))
    j = int(wp.floor(g[1]))
    k = int(wp.floor(g[2]))
    if i < 0 or j < 0 or k < 0 or i >= G.a.shape[0] - 1 or j >= G.a.shape[1] - 1 or k >= G.a.shape[2] - 1:
        return 1.0
    fx = g[0] - float(i)
    fy = g[1] - float(j)
    fz = g[2] - float(k)
    c00 = G.a[i, j, k] * (1.0 - fx) + G.a[i + 1, j, k] * fx
    c10 = G.a[i, j + 1, k] * (1.0 - fx) + G.a[i + 1, j + 1, k] * fx
    c01 = G.a[i, j, k + 1] * (1.0 - fx) + G.a[i + 1, j, k + 1] * fx
    c11 = G.a[i, j + 1, k + 1] * (1.0 - fx) + G.a[i + 1, j + 1, k + 1] * fx
    return (c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz) + (c01 * (1.0 - fy) + c11 * fy) * fz


@wp.func
def field_at(F: Field, x: wp.vec3) -> wp.vec4:
    """(velocity m/s, pressure mmHg) at x; velocity counts missing bins as 0 (no slip); pressure = -1e3 if none."""
    g = (x - F.origin) / F.dx
    i0 = int(wp.floor(g[0]))
    j0 = int(wp.floor(g[1]))
    k0 = int(wp.floor(g[2]))
    fx = g[0] - float(i0)
    fy = g[1] - float(j0)
    fz = g[2] - float(k0)
    ua = wp.vec3(0.0)
    pa = float(0.0)
    pw = float(0.0)
    o0 = F.f0 * F.n
    o1 = F.f1 * F.n
    for c in range(8):
        di = c & 1
        dj = (c >> 1) & 1
        dk = (c >> 2) & 1
        i = i0 + di
        j = j0 + dj
        k = k0 + dk
        if i >= 0 and j >= 0 and k >= 0 and i < F.grid.shape[0] and j < F.grid.shape[1] and k < F.grid.shape[2]:
            b = F.grid[i, j, k]
            if b >= 0:
                w = wp.where(di == 1, fx, 1.0 - fx) * wp.where(dj == 1, fy, 1.0 - fy) * wp.where(dk == 1, fz, 1.0 - fz)
                ua += w * (F.u[o0 + b] * (1.0 - F.w) + F.u[o1 + b] * F.w)
                pa += w * (F.p[o0 + b] * (1.0 - F.w) + F.p[o1 + b] * F.w)
                pw += w
    p = float(-1.0e3)
    if pw > 1.0e-6:
        p = pa / pw
    return wp.vec4(ua[0], ua[1], ua[2], p)


@wp.func
def lut(tab: wp.array(dtype=wp.vec3), x: float) -> wp.vec3:
    n = tab.shape[0]
    s = wp.clamp(x, 0.0, 1.0) * float(n - 1)
    i = wp.min(int(wp.floor(s)), n - 2)
    f = s - float(i)
    return tab[i] * (1.0 - f) + tab[i + 1] * f


def colour_table(name, n=256):
    import matplotlib
    c = matplotlib.colormaps[name](np.linspace(0, 1, n))[:, :3]
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)  # sRGB -> linear


def to_srgb8(c):
    c = np.clip(c, 0, 1)
    return (np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------
# particles
# ---------------------------------------------------------------------------------------------------------------


@wp.func
def respawn(seeds: wp.array(dtype=wp.vec3), cdf: wp.array(dtype=float), st: wp.uint32) -> wp.vec3:
    r = wp.randf(st)
    lo = int(0)
    hi = cdf.shape[0] - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < r:
            lo = mid + 1
        else:
            hi = mid
    j = wp.vec3(wp.randf(st) - 0.5, wp.randf(st) - 0.5, wp.randf(st) - 0.5)
    return seeds[lo] + j * 0.1


@wp.kernel
def k_advect(F: Field, G: Grid, pos: wp.array(dtype=wp.vec3), fresh: wp.array(dtype=int), stuck: wp.array(dtype=int),
             seeds: wp.array(dtype=wp.vec3), cdf: wp.array(dtype=float),
             ends_c: wp.array(dtype=wp.vec3), ends_n: wp.array(dtype=wp.vec3), ends_r: wp.array(dtype=float),
             dt: float, substeps: int, seed: int, exits: wp.array(dtype=int)):
    i = wp.tid()
    x = pos[i]
    st = wp.rand_init(seed, i)
    for s in range(substeps):
        a = field_at(F, x)
        u = wp.vec3(a[0], a[1], a[2])
        b = field_at(F, x + u * (0.5 * dt))
        step = wp.vec3(b[0], b[1], b[2]) * dt
        xn = x + step
        if sdf_at(G, xn) >= 0.0:         # a step into the wall (grazing particles): slide along it instead
            h = 0.5 * G.dx
            nrm = wp.vec3(sdf_at(G, x + wp.vec3(h, 0.0, 0.0)) - sdf_at(G, x - wp.vec3(h, 0.0, 0.0)),
                          sdf_at(G, x + wp.vec3(0.0, h, 0.0)) - sdf_at(G, x - wp.vec3(0.0, h, 0.0)),
                          sdf_at(G, x + wp.vec3(0.0, 0.0, h)) - sdf_at(G, x - wp.vec3(0.0, 0.0, h)))
            if wp.length(nrm) > 0.0:
                nrm = wp.normalize(nrm)
                xn = x + step - nrm * (wp.max(wp.dot(step, nrm), 0.0) + 0.002)
        if sdf_at(G, xn) < 0.0:
            x = xn
            stuck[i] = 0
        else:                            # held at the wall: after STUCK_STEPS in a row it is recycled at the inlet
            stuck[i] = stuck[i] + 1
            if stuck[i] > STUCK_STEPS:
                x = respawn(seeds, cdf, st)
                fresh[i] = 1
                stuck[i] = 0
        for e in range(ends_c.shape[0]):
            d = x - ends_c[e]
            if wp.dot(d, ends_n[e]) > -EXIT_DEPTH_K and wp.length(d) < 2.0 * ends_r[e]:
                wp.atomic_add(exits, e, 1)
                x = respawn(seeds, cdf, st)
                fresh[i] = 1
    pos[i] = x


@wp.kernel
def k_trail(F: Field, pos: wp.array(dtype=wp.vec3), fresh: wp.array(dtype=int), K: int, push: int,
            cen: wp.array(dtype=wp.vec3), spd: wp.array(dtype=float)):
    i = wp.tid()
    x = pos[i]
    a = field_at(F, x)
    s = wp.length(wp.vec3(a[0], a[1], a[2]))
    if fresh[i] == 1:
        for k in range(K):
            cen[i * K + k] = x
            spd[i * K + k] = s
        fresh[i] = 0
        return
    if push == 1:
        for k in range(K - 1, 0, -1):
            cen[i * K + k] = cen[i * K + k - 1]
            spd[i * K + k] = spd[i * K + k - 1]
    cen[i * K] = x
    spd[i * K] = s


@wp.func
def sphere_r(j: int, K: int, r0: float) -> float:
    return r0 * (1.0 - 0.55 * float(j % K) / float(K))


@wp.kernel
def k_spheres(cen: wp.array(dtype=wp.vec3), K: int, r0: float, ico: wp.array(dtype=wp.vec3),
              pts: wp.array(dtype=wp.vec3)):
    j, v = wp.tid()
    pts[j * 12 + v] = cen[j] + ico[v] * (sphere_r(j, K, r0) / 0.7558)  # icosahedron around the sphere


def icosahedron():
    p = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0], [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
                  [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], np.int32)
    return v.astype(np.float32), f


# ---------------------------------------------------------------------------------------------------------------
# ray tracing
# ---------------------------------------------------------------------------------------------------------------


@wp.struct
class Shade:
    lut_s: wp.array(dtype=wp.vec3)
    lut_p: wp.array(dtype=wp.vec3)
    lut_d: wp.array(dtype=wp.vec3)
    s_max: float
    p_lo: float
    p_hi: float
    d_range: float
    wall_mode: int        # 0 pressure, 1 pressure - healthy, 2 plain, 3 wall shear stress
    lut_w: wp.array(dtype=wp.vec3)
    wss: wp.array2d(dtype=float)   # frame x wall vertex, Pa (< 0: no wall there, e.g. an open end)
    tri: wp.array(dtype=int)       # the wall mesh's vertex indices
    has_wss: int
    w_lo: float           # log10 Pa
    w_hi: float
    glass: float          # front wall opacity (1 = solid)
    plane: wp.vec4        # n.x + d > 0 is cut away
    plane_mode: int       # 0 off, 1 cut away, 2 cut away + section filled
    section: int          # 0 speed, 1 pressure, 2 pressure - healthy
    particles: int
    light: wp.vec3
    keep: wp.vec4         # only what lies within this sphere is drawn (centre, radius; radius <= 0: all)


@wp.func
def kept(S: Shade, x: wp.vec3) -> bool:
    if S.keep[3] <= 0.0:
        return True
    return wp.length(x - wp.vec3(S.keep[0], S.keep[1], S.keep[2])) < S.keep[3]


@wp.func
def lit(c: wp.vec3, n: wp.vec3, rd: wp.vec3, l: wp.vec3) -> wp.vec3:
    d = wp.max(wp.dot(n, l), 0.0)
    h = wp.normalize(l - rd)
    sp = wp.pow(wp.max(wp.dot(n, h), 0.0), 40.0) * 0.25
    return c * (0.38 + 0.75 * d) + wp.vec3(sp)


@wp.func
def pressure_colour(S: Shade, A: Field, B: Field, x: wp.vec3, mode: int) -> wp.vec3:
    a = field_at(A, x)
    if a[3] < -500.0:
        return wp.vec3(0.25)
    if mode == 0:
        return lut(S.lut_p, (a[3] - S.p_lo) / (S.p_hi - S.p_lo))
    if mode == 1:
        b = field_at(B, x)
        if b[3] < -500.0:
            return wp.vec3(0.25)
        return lut(S.lut_d, 0.5 + 0.5 * (a[3] - b[3]) / S.d_range)
    return wp.vec3(0.62, 0.10, 0.09)


@wp.func
def wss_at(S: Shade, f: int, face: int, u: float, v: float) -> float:
    a = S.wss[f, S.tri[face * 3]]
    b = S.wss[f, S.tri[face * 3 + 1]]
    c = S.wss[f, S.tri[face * 3 + 2]]
    if a < 0.0 or b < 0.0 or c < 0.0:
        return -1.0
    return a * u + b * v + c * (1.0 - u - v)      # Warp's barycentrics (mesh_eval_position)


@wp.func
def wss_colour(S: Shade, A: Field, face: int, u: float, v: float) -> wp.vec3:
    if S.has_wss == 0:
        return wp.vec3(0.25)
    a = wss_at(S, A.f0, face, u, v)
    b = wss_at(S, A.f1, face, u, v)
    if a < 0.0 or b < 0.0:
        return wp.vec3(0.25)
    w = (1.0 - A.w) * a + A.w * b
    return lut(S.lut_w, (wp.log10(wp.max(w, 1.0e-4)) - S.w_lo) / (S.w_hi - S.w_lo))


@wp.kernel
def k_render(wall: wp.uint64, parts: wp.uint64, cen: wp.array(dtype=wp.vec3), spd: wp.array(dtype=float), K: int,
             r0: float, G: Grid, A: Field, B: Field, S: Shade,
             cam_pos: wp.vec3, cam_fwd: wp.vec3, cam_right: wp.vec3, cam_up: wp.vec3, tan_half: float,
             width: int, height: int, ss: int, bg: wp.vec3, out: wp.array2d(dtype=wp.vec3)):
    py, px = wp.tid()
    acc = wp.vec3(0.0)
    aspect = float(width) / float(height)
    pn = wp.vec3(S.plane[0], S.plane[1], S.plane[2])
    big = float(1.0e9)
    inset = float(WALL_INSET_K)
    for sy in range(ss):
        for sx in range(ss):
            u = (2.0 * (float(px) + (float(sx) + 0.5) / float(ss)) / float(width) - 1.0) * tan_half * aspect
            v = (1.0 - 2.0 * (float(py) + (float(sy) + 0.5) / float(ss)) / float(height)) * tan_half
            rd = wp.normalize(cam_fwd + cam_right * u + cam_up * v)
            ro = cam_pos
            col = wp.vec3(0.0)
            T = float(1.0)
            t = float(0.0)
            t_end = big
            done = int(0)
            if S.plane_mode > 0:
                side = wp.dot(pn, ro) + S.plane[3]
                den = wp.dot(pn, rd)
                if side > 0.0:
                    if den >= 0.0:
                        done = 1
                    else:
                        t = -side / den
                        if S.plane_mode == 2 and sdf_at(G, ro + rd * t) < 0.0 and kept(S, ro + rd * t):
                            xs = ro + rd * t
                            c = wp.vec3(0.0)
                            if S.section == 0:
                                a = field_at(A, xs)
                                c = lut(S.lut_s, wp.length(wp.vec3(a[0], a[1], a[2])) / S.s_max)
                            else:
                                c = pressure_colour(S, A, B, xs, S.section - 1)
                            col = c * 0.95
                            T = 0.0
                            done = 1
                elif den > 0.0:
                    t_end = -side / den
            for it in range(20):
                if done == 1:
                    break
                tp = big
                jp = int(-1)
                if S.particles == 1:
                    tq = t
                    for m in range(4):
                        if tq >= t_end:
                            break
                        q = wp.mesh_query_ray(parts, ro + rd * tq, rd, t_end - tq)
                        if not q.result:
                            break
                        j = q.face // 20
                        c = cen[j]
                        r = sphere_r(j, K, r0)
                        oc = ro - c
                        b = wp.dot(oc, rd)
                        disc = b * b - (wp.dot(oc, oc) - r * r)
                        if disc > 0.0:
                            ts = -b - wp.sqrt(disc)
                            if ts >= t and kept(S, ro + rd * ts):
                                tp = ts
                                jp = j
                                break
                        tq = tq + q.t + 1.0e-4
                qw = wp.mesh_query_ray(wall, ro + rd * t, rd, t_end - t)
                tw = big
                if qw.result:
                    tw = t + qw.t
                if jp >= 0 and tp < tw:
                    x = ro + rd * tp
                    n = wp.normalize(x - cen[jp])
                    c = lut(S.lut_s, spd[jp] / S.s_max)
                    col += T * lit(c, n, rd, S.light)
                    T = 0.0
                    break
                if tw >= big:
                    break
                x = ro + rd * tw
                if not kept(S, x):
                    t = tw + 0.01
                    continue
                n = wp.normalize(qw.normal)
                if wp.dot(n, rd) > 0.0:
                    n = -n
                entering = sdf_at(G, x - rd * 0.03) > 0.0
                xin = x + n * inset
                if entering:
                    xin = x - n * inset
                c = wp.vec3(0.0)
                if S.wall_mode == 3:
                    c = wss_colour(S, A, qw.face, qw.u, qw.v)
                else:
                    c = pressure_colour(S, A, B, xin, S.wall_mode)
                sh = lit(c, n, rd, S.light)
                if entering and S.glass < 1.0:
                    al = S.glass + (1.0 - S.glass) * 0.85 * wp.pow(1.0 - wp.abs(wp.dot(n, rd)), 3.0)
                    col += T * al * sh
                    T = T * (1.0 - al)
                    t = tw + 0.02
                else:
                    col += T * sh
                    T = 0.0
                    break
            acc += col + T * bg
    out[py, px] = acc / float(ss * ss)


# ---------------------------------------------------------------------------------------------------------------
# a case on the GPU
# ---------------------------------------------------------------------------------------------------------------


class Case:
    """One case's geometry (signed distance, wall mesh) and last-beat fields on the GPU."""

    def __init__(self, case, device):
        self.case, self.dev = case, device
        d = np.load(os.path.join(RESULTS, case, "fields.npz"))
        meta = json.load(open(os.path.join(RESULTS, case, "meta.json")))
        self.meta = meta
        c = d["coords"].astype(np.int64)
        dx = float(d["dx_mm"])
        shift = 0.5 * (meta["dx_mm"] * round(dx / meta["dx_mm"]) - meta["dx_mm"])  # bin centre (0.025 mm at 0.05)
        lo = c.min(0) - 1
        shape = c.max(0) - lo + 2
        grid = np.full(shape, -1, np.int32)
        grid[tuple((c - lo).T)] = np.arange(len(c), dtype=np.int32)
        self.n = len(c)
        self.t = d["t"]
        self.nf = len(self.t)
        self.origin = d["origin"] + lo * dx + shift
        self.dx = dx
        self.pos = self.origin + (c - lo) * dx            # bin centres (mm), for checks
        self.grid = wp.array(grid, dtype=int, device=device)
        self.u = wp.array(d["u_m_s"].astype(np.float32).reshape(-1, 3), dtype=wp.vec3, device=device)
        self.p = wp.array(d["p_mmhg"].astype(np.float32).reshape(-1), dtype=float, device=device)
        self.p_host = d["p_mmhg"].astype(np.float32)
        self.u_host = d["u_m_s"]
        # geometry: signed distance at 0.1 mm and the wall surface
        if is_patient(case):
            g = PT.geom(clot_of(case))
            phi, org = PT.build(clot_of(case), SDF_DX, device=device)[:2]
        else:
            # an AI estimate (Step 15, ai/estimate.py) is for any clot size
            g = V.build("healthy", severity=meta["severity"]) if meta.get("ai") else V.build(clot_of(case))
            phi, org = V.sdf_grid(g, SDF_DX, device=device)
        self.geom = g
        self.sdf = Grid()
        self.sdf.a = wp.array(phi, dtype=float, device=device)
        self.sdf.origin = wp.vec3(*map(float, org))
        self.sdf.dx = SDF_DX
        nx, ny, nz = phi.shape
        mc = wp.MarchingCubes(nx, ny, nz, device=device, domain_bounds_lower_corner=wp.vec3(*map(float, org)),
                              domain_bounds_upper_corner=wp.vec3(*map(float, org + (np.array(phi.shape) - 1) * SDF_DX)))
        mc.surface(self.sdf.a, 0.0)
        verts = mc.verts.numpy()
        self.wall = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device=device),
                            indices=wp.array(mc.indices.numpy().reshape(-1), dtype=int, device=device))
        self.verts = verts
        self.ends = meta["open_ends"]
        self.tri = wp.array(mc.indices.numpy().reshape(-1), dtype=int, device=device)
        self.load_wss()

    def load_wss(self):
        """Wall shear stress per wall vertex and frame (wss.py), if computed for this same mesh."""
        self.has_wss = 0
        self.wss = wp.zeros((1, 1), dtype=float, device=self.dev)
        f = os.path.join(RESULTS, self.case, "wss.npz")
        if not os.path.exists(f):
            return
        w = np.load(f)
        if w["verts"].shape != self.verts.shape or not np.allclose(w["verts"], self.verts, atol=1e-4):
            print(f"{self.case}: wss.npz is for another wall mesh; not shown")
            return
        mag = np.linalg.norm(w["tau"].astype(np.float32), axis=2)
        mag[:, ~w["wall"]] = -1.0
        self.wss = wp.array(mag, dtype=float, device=self.dev)
        self.has_wss = 1

    def field(self, t):
        """The Field struct at time t (s, wraps over the beat)."""
        s = (t % PERIOD) / (PERIOD / self.nf) - 1.0
        if s < 0:
            s += self.nf
        f0 = int(math.floor(s)) % self.nf
        F = Field()
        F.grid, F.u, F.p, F.n = self.grid, self.u, self.p, self.n
        F.origin = wp.vec3(*map(float, self.origin))
        F.dx = self.dx
        F.f0, F.f1, F.w = f0, (f0 + 1) % self.nf, float(s - math.floor(s))
        return F

    def frame_index(self, t):
        return int(round((t % PERIOD) / (PERIOD / self.nf) - 1.0)) % self.nf


# ---------------------------------------------------------------------------------------------------------------
# the scene: one case's particles + rendering
# ---------------------------------------------------------------------------------------------------------------


class Scene:
    def __init__(self, case, healthy=None, n=N_PARTICLES, seed=1, device=None):
        wp.init()
        self.dev = dev = device or wp.get_preferred_device()
        self.c = case if isinstance(case, Case) else Case(case, dev)
        self.h = healthy or self.c
        self.N, self.K = n, TRAIL
        self.t = 0.0
        self.step = 0
        self.seed = seed
        self._since_push = 0.0
        c = self.c
        # release points: bins in a thin slab inside the inlet, weighted by the mean inflow through them
        e = c.ends[0]
        cen, nrm = np.array(e["centre"]), np.array(e["normal"])
        depth = (cen - c.pos) @ nrm
        m = (depth > SEED_DEPTH[0]) & (depth < SEED_DEPTH[1])
        ubar = c.u_host[:, m].astype(np.float32).mean(0)
        wgt = np.clip(-(ubar @ nrm), 0, None)
        self.seeds = wp.array(c.pos[m].astype(np.float32), dtype=wp.vec3, device=dev)
        cdf = np.cumsum(wgt) / wgt.sum()
        self.cdf = wp.array(cdf.astype(np.float32), dtype=float, device=dev)
        self.ends_c = wp.array(np.array([x["centre"] for x in c.ends], np.float32), dtype=wp.vec3, device=dev)
        self.ends_n = wp.array(np.array([x["normal"] for x in c.ends], np.float32), dtype=wp.vec3, device=dev)
        self.ends_r = wp.array(np.array([x["radius"] for x in c.ends], np.float32), dtype=float, device=dev)
        self.exits = wp.zeros(len(c.ends), dtype=int, device=dev)
        # start: particles spread through the whole lumen (uniform over the bins)
        rng = np.random.default_rng(seed)
        p0 = c.pos[rng.integers(0, c.n, n)] + rng.uniform(-0.05, 0.05, (n, 3))
        self.pos = wp.array(p0.astype(np.float32), dtype=wp.vec3, device=dev)
        self.fresh = wp.ones(n, dtype=int, device=dev)
        self.stuck = wp.zeros(n, dtype=int, device=dev)
        self.cen = wp.zeros(n * self.K, dtype=wp.vec3, device=dev)
        self.spd = wp.zeros(n * self.K, dtype=float, device=dev)
        ico_v, ico_f = icosahedron()
        self.ico = wp.array(ico_v, dtype=wp.vec3, device=dev)
        idx = (np.arange(n * self.K, dtype=np.int32)[:, None, None] * 12 + ico_f[None]).reshape(-1)
        self.sph_idx = wp.array(idx, dtype=int, device=dev)
        self.sph_pts = wp.zeros(n * self.K * 12, dtype=wp.vec3, device=dev)
        self._trail(push=True)
        self.luts = {k: wp.array(colour_table(v), dtype=wp.vec3, device=dev)
                     for k, v in (("s", "turbo"), ("p", "RdYlBu_r"), ("d", "RdBu_r"), ("w", "viridis"))}

    def _trail(self, push):
        wp.launch(k_trail, dim=self.N, device=self.dev, inputs=[self.c.field(self.t), self.pos, self.fresh, self.K,
                                                                 int(push), self.cen, self.spd])

    def advance(self, dt_ms):
        """Move the particles dt_ms of simulated time (and the clock)."""
        if dt_ms <= 0:
            return
        sub = max(1, int(math.ceil(dt_ms / SUBSTEP_MS)))
        h = dt_ms / sub
        done = 0
        while done < sub:  # in chunks, so the field's time follows the particles
            k = min(sub - done, max(1, int(TRAIL_EVERY_MS / h)))
            self.step += 1
            wp.launch(k_advect, dim=self.N, device=self.dev, inputs=[
                self.c.field(self.t), self.c.sdf, self.pos, self.fresh, self.stuck, self.seeds, self.cdf, self.ends_c,
                self.ends_n, self.ends_r, h, k, self.seed * 100003 + self.step, self.exits])
            self.t += k * h * 1e-3
            self._since_push += k * h
            done += k
            push = self._since_push >= TRAIL_EVERY_MS
            if push:
                self._since_push = 0.0
            self._trail(push)

    def warm(self, beats=1.0, dt_ms=0.5):
        for _ in range(int(beats * PERIOD * 1e3 / dt_ms)):
            self.advance(dt_ms)

    def render(self, cam, width=1280, height=720, ss=2, wall_mode=0, glass=GLASS, plane=None, plane_mode=0,
               section=0, particles=True, fov=40.0, light=(-0.4, 0.8, -0.45), radius=PARTICLE_R, keep=None):
        dev = self.dev
        wp.launch(k_spheres, dim=(self.N * self.K, 12), device=dev,
                  inputs=[self.cen, self.K, float(radius), self.ico], outputs=[self.sph_pts])
        parts = wp.Mesh(points=self.sph_pts, indices=self.sph_idx)  # rebuilt every frame (never refit)
        S = Shade()
        S.lut_s, S.lut_p, S.lut_d = self.luts["s"], self.luts["p"], self.luts["d"]
        S.s_max, S.p_lo, S.p_hi, S.d_range = SPEED_MAX, P_RANGE[0], P_RANGE[1], DP_RANGE
        S.wall_mode, S.glass = wall_mode, float(glass)
        S.lut_w, S.wss, S.tri, S.has_wss = self.luts["w"], self.c.wss, self.c.tri, self.c.has_wss
        S.w_lo, S.w_hi = math.log10(WSS_RANGE[0]), math.log10(WSS_RANGE[1])
        S.plane = wp.vec4(*map(float, plane)) if plane is not None else wp.vec4(0.0)
        S.plane_mode, S.section, S.particles = (plane_mode if plane is not None else 0), section, int(particles)
        S.keep = wp.vec4(*map(float, keep)) if keep is not None else wp.vec4(0.0)
        l = np.array(light, float)
        S.light = wp.vec3(*map(float, l / np.linalg.norm(l)))
        pos, fwd, right, up = cam
        out = wp.empty((height, width), dtype=wp.vec3, device=dev)
        wp.launch(k_render, dim=(height, width), device=dev, inputs=[
            self.c.wall.id, parts.id, self.cen, self.spd, self.K, float(radius), self.c.sdf, self.c.field(self.t),
            self.h.field(self.t), S, wp.vec3(*map(float, pos)), wp.vec3(*map(float, fwd)), wp.vec3(*map(float, right)),
            wp.vec3(*map(float, up)), math.tan(math.radians(fov) / 2), width, height, ss, wp.vec3(*BACKGROUND)],
            outputs=[out])
        return to_srgb8(out.numpy())


def look_at(pos, target, up=(0.0, 1.0, 0.0)):
    """Camera basis, Unity's left-handed convention (right = up x forward), as the stroke renderer."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f /= np.linalg.norm(f)
    r = np.cross(np.asarray(up, float), f)
    r /= np.linalg.norm(r)
    return pos, f, r, np.cross(f, r)


def project(cam, X, width, height, fov):
    pos, f, r, u = cam
    d = np.atleast_2d(X) - pos
    z = d @ f
    th = math.tan(math.radians(fov) / 2)
    x = (d @ r) / (z * th * width / height)
    y = (d @ u) / (z * th)
    return np.stack([(x + 1) / 2 * width, (1 - y) / 2 * height], 1), z


# ---------------------------------------------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------------------------------------------


def segment_frame(g):
    """Viewing frame for the segment: the best-fit plane of its centrelines (normal = least spread)."""
    P = np.vstack([b["C"][b["segment_from"]:b["segment_to"]] for b in g["branches"].values()])
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[2]
    if n[2] > 0:   # look from the same side as Step 4's maps (towards -z)
        n = -n
    if "view_dir" in g:   # a patient: the scan's own front view (as in 3D Slicer)
        n = np.asarray(g["view_dir"], float)
    return c, n, P


def clot_point(g):
    b = g["branches"]["blocked"]
    s = V._arc(b["C"])
    k = np.argmin(np.abs(s - (g["clot"]["face_arc"] + 0.5 * g["clot"]["body_length"])))
    return b["C"][k], b["C"][min(k + 3, len(s) - 1)] - b["C"][max(k - 3, 0)]


def overview_cam(g, fov=40.0, aspect=16 / 9, margin=1.12):
    c, n, P = segment_frame(g)
    # fit the projected extent
    up0 = np.array(g.get("up", (0.0, 1.0, 0.0)), float)
    up0 = up0 - n * (up0 @ n)
    cam = look_at(c - n * 100, c, up0)
    _, f, r, u = cam
    xr, yr = (P - c) @ r, (P - c) @ u
    cx, cy = (xr.max() + xr.min()) / 2, (yr.max() + yr.min()) / 2
    centre = c + r * cx + u * cy
    th = math.tan(math.radians(fov) / 2)
    dist = margin * max((xr.max() - xr.min()) / 2 / (th * aspect), (yr.max() - yr.min()) / 2 / th) + 4.0
    return look_at(centre - f * dist, centre, up0)


def clot_cam(g, dist=17.0, tilt=(0.25, 0.2), ahead=1.5):
    p, tng = clot_point(g)
    c, n, _ = segment_frame(g)
    up0 = np.array(g.get("up", (0.0, 1.0, 0.0)), float)
    tng /= np.linalg.norm(tng)
    p = p + tng * ahead           # a little downstream, so the jet / the stagnant branch is in view
    side = np.cross(tng, n)
    d = -n + side * tilt[0] + up0 * tilt[1]
    d /= np.linalg.norm(d)
    return look_at(p + d * dist, p, up0 - n * (up0 @ n))


def clot_plane(g, cam):
    """Plane through the clot's centreline, containing the vessel axis, facing the camera (its camera side is cut)."""
    p, tng = clot_point(g)
    pos = cam[0]
    tng = tng / np.linalg.norm(tng)
    v = pos - p
    nrm = v - tng * (v @ tng)
    nrm /= np.linalg.norm(nrm)
    return np.array([*nrm, -nrm @ p])


# ---------------------------------------------------------------------------------------------------------------
# overlays (PIL)
# ---------------------------------------------------------------------------------------------------------------


def font(size, bold=True):
    from PIL import ImageFont
    for f in (("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def colour_bar(im, xy, size, cmap, lo, hi, label, ticks, scale=1.0):
    import matplotlib
    x0, y0 = xy
    w, h = size
    c = (matplotlib.colormaps[cmap](np.linspace(0, 1, w))[:, :3] * 255).astype(np.uint8)
    im.paste(Image.fromarray(np.repeat(c[None], h, 0)), (x0, y0))
    d = ImageDraw.Draw(im)
    d.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], outline=(200, 200, 200))
    f = font(int(15 * scale), False)
    for v in ticks:
        x = x0 + (v - lo) / (hi - lo) * (w - 1)
        d.line([x, y0 + h, x, y0 + h + 4], fill=(200, 200, 200))
        d.text((x, y0 + h + 6), f"{v:g}", fill=(220, 220, 220), font=f, anchor="mt")
    d.text((x0, y0 - 6), label, fill=(235, 235, 235), font=font(int(16 * scale)), anchor="lb")


def caption(im, text, xy, size=24, colour=(255, 255, 255), anchor="la", bold=True):
    ImageDraw.Draw(im).text(xy, text, fill=colour, font=font(size, bold), anchor=anchor)


# ---------------------------------------------------------------------------------------------------------------
# context: the story's brain with the segment marked (stroke renderer)
# ---------------------------------------------------------------------------------------------------------------


def context_still(width=1280, height=720):
    sys.path.insert(0, STROKE)
    import dataclasses
    import renderer as SR
    seg = json.load(open(os.path.join(HERE, "geometry", "segment.json")))
    r = SR.Renderer()
    J = np.array(seg["junction_point"])
    target = 0.5 * (J + r.s.pivot)
    # from the left, a little in front and above (the segment sits at the base of the left hemisphere)
    d = np.array([-1.0, 0.35, -0.55])
    d /= np.linalg.norm(d)
    pos = target + d * 1.25
    fwd = target - pos
    view = SR.View(cam_pos=tuple(pos), cam_fwd=tuple(fwd), shell=0.72, regions=1.0, vessels=0.0, lesion=1.0, fov=45.0)
    look = dataclasses.replace(SR.LOOKS["unity"], transparency="alpha", ss=3)
    img = Image.fromarray(r.render(view, look, width, height))
    cam_pos, cam_fwd, cam_right, cam_up, _ = r.camera(view)
    cam = (cam_pos, cam_fwd, cam_right, cam_up)
    pts = np.vstack([np.array(b["points"]) for b in seg["branches"].values()])
    xy, _ = project(cam, pts, width, height, 45.0)
    clot, _ = project(cam, np.array(seg["clot"]["face_point"]), width, height, 45.0)
    d = ImageDraw.Draw(img)
    lo, hi = xy.min(0) - 22, xy.max(0) + 22
    d.rounded_rectangle([*lo, *hi], radius=14, outline=(255, 214, 90), width=3)
    cx, cy = clot[0]
    d.ellipse([cx - 9, cy - 9, cx + 9, cy + 9], outline=(255, 255, 255), width=3)
    caption(img, "The simulated artery", (hi[0] + 16, lo[1] + 4), 26, (255, 214, 90))
    caption(img, "end of the internal carotid, its two branches,", (hi[0] + 16, lo[1] + 38), 19, bold=False)
    caption(img, "and the clot site in the middle cerebral artery (ring)", (hi[0] + 16, lo[1] + 62), 19, bold=False)
    del r
    return img, (cam, xy)


# ---------------------------------------------------------------------------------------------------------------
# stills + checks (Step 5 report)
# ---------------------------------------------------------------------------------------------------------------


def stats(case):
    rep = json.load(open(os.path.join(RESULTS, case, "meta.json")))
    s = np.load(os.path.join(RESULTS, case, "series.npz"))
    t = s["t"]
    m = t > t[-1] - PERIOD + 1e-9
    Q = s["Q"][m]
    names = [p["name"] for p in rep["probes"]]
    pp = s["p_probe"][m]
    before, after = around(rep)
    drop = pp[:, names.index(before)] - pp[:, names.index(after)]
    return {"Q_mca": float(Q[:, 1].mean()), "Q_in": float(Q[:, 0].mean()),
            "Q_aca": float(Q[:, 2].mean()) if Q.shape[1] > 2 else None,   # a patient may have no ACA outlet
            "umax_probes": float(s["umax"][m].max()),
            "umax": float(np.linalg.norm(np.load(os.path.join(RESULTS, case, "fields.npz"))["u_m_s"].astype(np.float32),
                                         axis=2).max()), "drop": float(drop.mean()), "names": names, "t": t[m] - t[m][0],
            "p_probe": pp, "series_t": t[m]}


def around(meta):
    """The probes either side of the clot / the patient's lesion (their difference: the pressure drop)."""
    return PT.PATIENTS[meta["patient"]["patient"]]["around"] if "patient" in meta else ("MCA before clot", "MCA after clot")


def numbers_panel(im, case, st, x, y, scale=1.0):
    lines = [(f"Middle cerebral artery flow", f"{max(st['Q_mca'], 0.0) + 0.0:.0f} mL/min"),
             (f"Pressure drop across the clot", f"{st['drop']:.0f} mmHg"),
             (f"Top speed", f"{st['umax']:.1f} m/s")]
    caption(im, LABEL[case], (x, y), int(30 * scale), (255, 214, 90))
    for i, (a, b) in enumerate(lines):
        yy = y + int((44 + 30 * i) * scale)
        caption(im, a, (x, yy), int(19 * scale), (210, 210, 210), bold=False)
        caption(im, b, (x + int(300 * scale), yy), int(19 * scale), (255, 255, 255))


def main():
    wp.init()
    dev = wp.get_preferred_device()
    os.makedirs(REPORTS, exist_ok=True)
    T0 = time.perf_counter()
    lines = ["# Step 5 report: showing the flow", "",
             "Generated by `uv run visualise.py`. Everything is ray traced with NVIDIA Warp from Step 4's results "
             "(the last simulated heartbeat of each case). Stills at peak systole (t = "
             f"{PEAK_T:.3f} s in the beat) unless stated.", ""]
    # --- context
    t0 = time.perf_counter()
    ctx, _ = context_still()
    ctx.save(os.path.join(REPORTS, "step5_context.png"))
    print(f"context still: {time.perf_counter() - t0:.1f} s", flush=True)
    # --- cases
    cases = {}
    healthy = Case("healthy", dev)
    cases["healthy"] = healthy
    for c in CASES[1:]:
        cases[c] = Case(c, dev)
    st = {c: stats(c) for c in CASES}
    scenes = {}
    checks = []
    for c in CASES:
        t0 = time.perf_counter()
        sc = Scene(cases[c], healthy, device=dev)
        sc.warm(beats=1.0)                     # particles reach their flow-weighted spread
        sc.exits.zero_()
        sc.warm(beats=1.0)
        ex = sc.exits.numpy()
        # land on peak systole of the following beat
        dt = ((PEAK_T - sc.t) % PERIOD) * 1e3
        sc.advance(dt)
        scenes[c] = sc
        stats_peak[c] = float(sc.spd.numpy()[::sc.K].max())
        split = ex[1] / max(ex[1] + ex[2], 1)
        q = st[c]
        checks.append((c, ex, split, q["Q_mca"] / max(q["Q_mca"] + q["Q_aca"], 1e-9)))
        print(f"{c}: particles ready in {time.perf_counter() - t0:.1f} s; exits over one beat MCA {ex[1]}, ACA {ex[2]}"
              f" (MCA share {split:.0%} vs flow {checks[-1][3]:.0%})", flush=True)
    g = healthy.geom
    W, H = 1280, 720
    cam_o = overview_cam(g, fov=40.0)
    cam_c = clot_cam(g)
    plane = clot_plane(g, cam_c)
    keep = (*clot_point(g)[0], KEEP_R)
    # --- 1. overview per case: glass wall coloured by pressure + particles coloured by speed
    t0 = time.perf_counter()
    tiles = {}
    for c in CASES:
        im = Image.fromarray(scenes[c].render(cam_o, W, H, ss=2, wall_mode=0, fov=40.0, radius=0.11))
        tiles[c] = im
    print(f"overview renders: {(time.perf_counter() - t0) / 3:.2f} s each (1280x720, 2x2 samples)", flush=True)
    # three tiles across at half size, numbers below
    sheet = Image.new("RGB", (3 * 640, 360 + 190), (0, 0, 0))
    for i, c in enumerate(CASES):
        sheet.paste(tiles[c].resize((640, 360), Image.LANCZOS), (640 * i, 0))
        numbers_panel(sheet, c, st[c], 640 * i + 24, 372, 0.9)
    colour_bar(sheet, (24, 360 + 160), (260, 12), "turbo", 0, SPEED_MAX, "blood speed (m/s)", [0, 1, 2, 3, 4], 0.85)
    colour_bar(sheet, (660, 360 + 160), (260, 12), "RdYlBu_r", *P_RANGE, "wall pressure (mmHg)", [0, 50, 100, 150],
               0.85)
    sheet = pad_bottom(sheet, 40)
    sheet.save(os.path.join(REPORTS, "step5_compare.png"))
    for c in CASES:
        im = tiles[c]
        caption(im, LABEL[c], (28, 24), 34, (255, 214, 90))
        caption(im, "particles: blood speed   wall: blood pressure", (28, 66), 19, (210, 210, 210), bold=False)
        colour_bar(im, (28, H - 70), (300, 14), "turbo", 0, SPEED_MAX, "blood speed (m/s)", [0, 1, 2, 3, 4])
        colour_bar(im, (370, H - 70), (300, 14), "RdYlBu_r", *P_RANGE, "wall pressure (mmHg)", [0, 50, 100, 150])
        numbers_panel(im, c, st[c], W - 470, H - 150, 0.85)
        im.save(os.path.join(REPORTS, f"step5_overview_{c}.png"))
    # --- 2. close-up at the clot: cut-away (near half removed), particles inside, back wall by pressure
    rows = []
    for c in CASES:
        a = Image.fromarray(scenes[c].render(cam_c, 640, 400, ss=3, wall_mode=0, glass=1.0, plane=plane,
                                             plane_mode=1, fov=40.0, keep=keep))
        b = Image.fromarray(scenes[c].render(cam_c, 640, 400, ss=3, wall_mode=0, glass=1.0, plane=plane,
                                             plane_mode=2, section=0, particles=False, fov=40.0, keep=keep))
        p = Image.fromarray(scenes[c].render(cam_c, 640, 400, ss=3, wall_mode=0, glass=1.0, plane=plane,
                                             plane_mode=2, section=1, particles=False, fov=40.0, keep=keep))
        caption(a, LABEL[c], (16, 14), 26, (255, 214, 90))
        rows.append((a, b, p))
    cut = Image.new("RGB", (3 * 640, 3 * 400 + 110), (0, 0, 0))
    for i, (a, b, p) in enumerate(rows):
        for j, im in enumerate((a, b, p)):
            cut.paste(im, (640 * j, 400 * i))
    for j, s in enumerate(("cut open: particles by speed", "cut plane: blood speed", "cut plane: blood pressure")):
        caption(cut, s, (640 * j + 16, 3 * 400 + 12), 20, (230, 230, 230))
    colour_bar(cut, (656, 3 * 400 + 64), (300, 13), "turbo", 0, SPEED_MAX, "speed (m/s)", [0, 1, 2, 3, 4])
    colour_bar(cut, (1296, 3 * 400 + 64), (300, 13), "RdYlBu_r", *P_RANGE, "pressure (mmHg)", [0, 50, 100, 150])
    cut = pad_bottom(cut, 30)
    cut.save(os.path.join(REPORTS, "step5_clot_cut.png"))
    # --- 3. difference maps: wall pressure minus healthy (solid wall), no particles
    diff = Image.new("RGB", (2 * 960, 540 + 110), (0, 0, 0))
    for i, c in enumerate(CASES[1:]):
        im = Image.fromarray(scenes[c].render(cam_o, 960, 540, ss=2, wall_mode=1, glass=1.0, particles=False,
                                              fov=40.0))
        caption(im, f"{LABEL[c]} minus healthy", (20, 18), 28, (255, 214, 90))
        caption(im, "wall pressure difference at peak systole", (20, 56), 18, (210, 210, 210), bold=False)
        diff.paste(im, (960 * i, 0))
    colour_bar(diff, (660, 540 + 50), (600, 14), "RdBu_r", -DP_RANGE, DP_RANGE,
               "pressure change vs healthy (mmHg): blue = lower, red = higher", [-90, -60, -30, 0, 30, 60, 90])
    diff = pad_bottom(diff, 20)
    diff.save(os.path.join(REPORTS, "step5_difference.png"))
    # --- 4. the beat: healthy vs 70 % at four times (particle motion through the cycle)
    times = (0.05, PEAK_T, 0.3, 0.7)
    beat = Image.new("RGB", (4 * 480, 2 * 270 + 40), (0, 0, 0))
    for i, c in enumerate(("healthy", "narrow70")):
        sc = scenes[c]
        for j, tb in enumerate(times):
            sc.advance(((tb - sc.t) % PERIOD) * 1e3)
            im = Image.fromarray(sc.render(cam_c, 480, 270, ss=2, wall_mode=0, glass=1.0, plane=plane,
                                           plane_mode=1, fov=40.0, keep=keep))
            caption(im, f"{LABEL[c]}, t = {tb:.2f} s", (10, 8), 17, (255, 214, 90))
            beat.paste(im, (480 * j, 270 * i))
    caption(beat, "through one heartbeat (peak systole at 0.12 s); particles by speed, wall by pressure",
            (10, 2 * 270 + 10), 18, (220, 220, 220), bold=False)
    beat.save(os.path.join(REPORTS, "step5_beat.png"))
    # --- checks vs the simulation data
    lines += ["## Views (stills)", "",
              "| file | what |", "|---|---|",
              "| `step5_context.png` | the story's brain (stroke renderer), the simulated segment boxed, the clot ringed |",
              "| `step5_overview_<case>.png` | the whole segment: glass wall coloured by pressure, particles by speed, "
              "numbers |",
              "| `step5_compare.png` | the three cases side by side with their numbers |",
              "| `step5_clot_cut.png` | close-up at the clot, near half cut away: particles; cut plane speed; cut plane "
              "pressure |",
              "| `step5_difference.png` | wall pressure minus the healthy case (70 % and full occlusion) |",
              "| `step5_beat.png` | healthy vs 70 % at four times in the beat |", ""]
    lines += ["## Checks against the simulation data", "",
              "1. Particles: share of particles leaving through the MCA over one beat vs the simulated flow split "
              "(MCA / (MCA + ACA)).", "",
              "| case | particles out MCA / ACA | MCA share (particles) | MCA share (flow) |", "|---|---|---|---|"]
    for c, ex, split, qs in checks:
        lines.append(f"| {c} | {ex[1]} / {ex[2]} | {split:.0%} | {qs:.0%} |")
    lines += ["", "2. Field sampling: pressure the renderer reads at each Step 4 probe point (trilinear, 0.1 mm bins) "
              "vs the probe's own record (sphere average on the simulation grid), mean over the 40 frames.", "",
              "| probe | " + " | ".join(CASES) + " |", "|---|" + "---|" * len(CASES)]
    probe_rows = {}
    for c in CASES:
        cs = cases[c]
        meta = cs.meta
        s = np.load(os.path.join(RESULTS, c, "series.npz"))
        tt = s["t"]
        for k, pr in enumerate(meta["probes"]):
            vals = []
            for f in range(cs.nf):
                tb = cs.t[f]
                F = cs.field(tb)
                out = wp.zeros(1, dtype=wp.vec4, device=dev)
                wp.launch(k_probe, dim=1, device=dev, inputs=[F, wp.vec3(*map(float, pr["point"]))], outputs=[out])
                vals.append(out.numpy()[0])
            vals = np.array(vals)
            # the record over the last beat at the frame times
            tl = tt[-1] - PERIOD + cs.t
            rec = np.interp(tl, tt, s["p_probe"][:, k])
            ok = vals[:, 3] > -500
            r_mean = float(rec[ok].mean()) if ok.any() else float("nan")
            v_mean = float(vals[ok, 3].mean()) if ok.any() else float("nan")
            probe_rows.setdefault(pr["name"], []).append((v_mean, r_mean))
    for name, vals in probe_rows.items():
        lines.append(f"| {name} | " + " | ".join("(no lumen)" if math.isnan(a) else f"{a:.1f} vs {b:.1f}"
                                                 for a, b in vals) + " |")
    lines += ["", "3. Speed: fastest particle in the stills vs the fastest point of the stored fields over the beat "
              "(whole artery, 0.1 mm bins; the on-screen 'top speed'), and Step 4's recorded maximum, which only "
              "covered the probe regions (correction to the Step 4 report).",
              "", "| case | particles at peak systole (max) | stored fields, whole artery (max over beat) | "
              "Step 4 record (probe regions only) |", "|---|---|---|---|"]
    for c in CASES:
        cs = cases[c]
        lines.append(f"| {c} | {stats_peak[c]:.2f} m/s | {st[c]['umax']:.2f} m/s | {st[c]['umax_probes']:.2f} m/s |")
    lines += ["", f"Run time: {(time.perf_counter() - T0) / 60:.1f} min.", ""]
    open(os.path.join(REPORTS, "step5_report.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


@wp.kernel
def k_probe(F: Field, x: wp.vec3, out: wp.array(dtype=wp.vec4)):
    out[0] = field_at(F, x)


def pad_bottom(im, h):
    out = Image.new("RGB", (im.width, im.height + h), (0, 0, 0))
    out.paste(im, (0, 0))
    return out


stats_peak = {}

if __name__ == "__main__":
    main()
