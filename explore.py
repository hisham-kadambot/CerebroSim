# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "matplotlib", "pillow", "warp-lang>=1.18,<1.19"]
# ///
"""Step 8: the interactive explorer - a live, browser-controlled view of the simulated artery, in the spirit of
NVIDIA's "Digital Twins for Fluid Simulation" blueprint (its Explore panel), built on this project's Warp renderer.

  blueprint (car in a wind tunnel)         here (blood in the artery)
  velocity / spoiler / mirrors / rims  ->  clot: healthy / 30 / 50 / 70 / 90 % narrowed / fully blocked, and the
                                           blood pressure: normal / stroke patient / severe (Step 10: one
                                           simulation per pair, run_bp.sh; a pair shows up once its run is finished)
  flow / streamlines / slice / volume  ->  blood particles, wall colour, lengthwise cut or cross-section
                                           coloured by speed / pressure / change from healthy
  comparison view                      ->  side by side with healthy; pressure change against healthy
  camera presets + orbit               ->  whole artery / clot / junction, orbit, drag to turn, wheel to zoom
  cached results (lite profile)        ->  the finished simulations (Step 4), switched instantly
  (Step 11d) artery: the story's MCA, or a real patient's own vessel (patient.py) with its scans as the cases
  (Step 15) AI estimate: any clot size and blood pressure, from the trained AI model (ai/estimate.py, a local
            service on port 8004 that this starts), shown next to the nearest simulation
Plus: click the artery to read the pressure and speed there, the heartbeat (play / pause / slow motion / phase)
and the case's numbers with its pressure-over-the-heartbeat chart.

Everything is rendered live on the GPU (visualise.Scene: particles advected through the stored last heartbeat,
ray traced) and streamed to the browser as MJPEG, like the stroke viewer.

  uv run explore.py        then open http://127.0.0.1:8003 (other computers: the address it prints; port 8003
                           needs a firewall rule, see README)
"""
import io
import json
import math
import os
import queue
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import warp as wp
from PIL import Image
from scipy.ndimage import map_coordinates

import patient as PT
import vessel as V
import visualise as VZ

HERE = os.path.dirname(os.path.abspath(__file__))
HOST, PORT = "0.0.0.0", 8003       # 8001 stroke viewer, 8002 artery demo video
MAX_FPS = 30.0
RESOLUTIONS = {"540p": (960, 540), "720p": (1280, 720), "1080p": (1920, 1080)}
FOV = 40.0
N_LIVE = 12000                     # particles per case (the demo uses 16,000)
MAX_STEP_MS = 10.0                 # simulated ms the particles may advance per frame; the clock never runs ahead of them
SECTION_KEEP = 6.0                 # mm of artery kept around a cross-section
SECTION_ANGLE = 50.0               # deg between the view and the vessel axis at a cross-section
SECTION_DIST = 9.0                 # mm
CLOTS = ("healthy", "narrow30", "narrow50", "narrow70", "narrow90", "occluded")
# arteries: the story's MCA (clot sizes) and the patients' own vessels (their scans); the first case is the default
VESSELS = {"story": CLOTS, **{p: tuple(f"{p}_{k}" for k in c["scans"]) for p, c in PT.PATIENTS.items()}}
DEFAULT = {"story": "narrow70", **{p: v[0] for p, v in VESSELS.items() if p != "story"}}
# blood-pressure levels: the inlet's mean pressure, mmHg (simulate.py --map). normal = the calibrated healthy one;
# stroke / severe: the 7 emergency stroke patients' average and highest on arrival (Sen et al. 2014, WJEM 15:345)
FLOWS = {"normal": 95, "stroke": 118, "severe": 146}
FLOW_LABEL = {"normal": "", "stroke": "stroke-patient blood pressure (118 mmHg)", "severe": "severe high blood pressure (146 mmHg)"}
# brain tissue blood flow, mL/100 g/min (CT perfusion, Ionita et al. 2009, Endovascular Today): normal, penumbra
# (starved, savable) and infarct core. The MCA's territory is sized so the healthy flow at normal pressure gives 70.8.
CBF = {"normal": 70.8, "penumbra": 64.8, "core": 24.6}
AVAIL_EVERY = 20.0                 # s between looks for newly finished simulations
# Step 15: the AI estimate service (ai/estimate.py --serve, in the PhysicsNeMo environment), this computer only
AI_URL = "http://127.0.0.1:8004"
AI_DIR = os.path.join(HERE, "ai")
AI_CLOT = (0, 100)                 # %, the clot-size slider
AI_MAP = (70, 145)                 # mmHg, the blood-pressure slider (the model was trained on 66-146)
ai_state = {"status": "starting", "info": None}

CHOICES = {
    "vessel": tuple(VESSELS), "case": tuple(c for v in VESSELS.values() for c in v), "flow": tuple(FLOWS), "compare": ("off", "side", "change"), "wall": ("pressure", "shear", "change", "plain"),
    "slice": ("off", "along", "across"), "field": ("speed", "pressure", "change"),
    "view": ("overview", "clot", "junction"), "resolution": tuple(RESOLUTIONS),
}
RANGES = {"slowmo": (0.01, 0.25), "across_at": (0.0, 1.0), "yaw": (-720.0, 720.0), "pitch": (-85.0, 85.0),
          "zoom": (0.25, 4.0), "quality": (1, 2)}
params = {"vessel": "story", "case": "narrow70", "flow": "normal", "compare": "off", "wall": "pressure", "particles": True, "slice": "off",
          "field": "speed", "across_at": 0.42, "view": "overview", "orbit": False, "yaw": 0.0, "pitch": 0.0,
          "zoom": 1.0, "playing": True, "slowmo": 0.05, "resolution": "540p", "quality": 1}
params_lock = threading.Lock()
clock = {"t": 0.0, "seek": None}   # simulated time (s); seek = a phase (s in the beat) to jump to
frame_cond = threading.Condition()
latest = {"jpeg": None, "id": 0, "fps": 0.0, "actual": 0.0, "render_ms": 0.0, "status": "loading", "cam": None, "size": (960, 540)}
probes = queue.Queue()
watchers = [0]                     # open /stream connections; with none, nothing is rendered (the GPU goes to the simulations)


def vessel_of(case):
    return case.split("_")[0] if VZ.is_patient(case) else "story"


def tag(clot, flow):
    """Results folder of a clot at a flow level (as simulate.result_name)."""
    return clot if flow == "normal" else f"{clot}_map{FLOWS[flow]}"


_avail = {"at": -1e9, "set": frozenset()}


def available():
    """Results folders whose simulation is finished (re-read every AVAIL_EVERY s, so new runs appear)."""
    if time.time() - _avail["at"] > AVAIL_EVERY:
        out = set()
        for c in CHOICES["case"]:
            for f in FLOWS:
                d = os.path.join(VZ.RESULTS, tag(c, f))
                try:
                    with open(os.path.join(d, "meta.json")) as fh:
                        done = json.load(fh).get("done")
                    if done and os.path.exists(os.path.join(d, "fields.npz")):
                        out.add(tag(c, f))
                except (OSError, ValueError):
                    pass
        out |= set(ai_estimates())
        _avail.update(at=time.time(), set=frozenset(out))
    return _avail["set"]


def is_ai(t):
    return t.startswith("ai_")


def ai_estimates():
    """The AI estimates on disk (ai/estimate.py writes fields.npz first, then meta.json)."""
    out = []
    for n in os.listdir(VZ.RESULTS):
        if is_ai(n) and os.path.exists(os.path.join(VZ.RESULTS, n, "fields.npz")):
            if ai_meta(n):
                out.append(n)
    return out


_ai_meta = {}


def ai_meta(t):
    """An AI estimate's meta.json (cached until the estimate is redone)."""
    f = os.path.join(VZ.RESULTS, t, "meta.json")
    try:
        m = os.path.getmtime(f)
        if t not in _ai_meta or _ai_meta[t][0] != m:
            with open(f) as fh:
                _ai_meta[t] = (m, json.load(fh))
        return _ai_meta[t][1]
    except (OSError, ValueError):
        return None


def nearest_sim(t, av):
    """The finished story simulation closest to an AI estimate: 10 % of clot size counts as much as 25 mmHg."""
    m = ai_meta(t)
    best = min(((abs(V.CASES[c] - m["severity"]) / 0.1 + abs(FLOWS[f] - m["map_mmhg"]) / 25.0, tag(c, f))
                for c in CLOTS for f in FLOWS if tag(c, f) in av), default=(0, "healthy"))
    return best[1]


def stamp(t):
    """When a results folder's fields were written (a re-run changes it)."""
    try:
        return os.path.getmtime(os.path.join(VZ.RESULTS, t, "fields.npz"))
    except OSError:
        return None


def reference(case):
    """What a case is compared with: the healthy artery; for a patient, the first scan (pre: no healthy scan)."""
    return "healthy" if vessel_of(case) == "story" else VESSELS[vessel_of(case)][0]


def pick(p):
    """(shown case, its reference): the reference at the same blood pressure when that exists."""
    av = available()
    if is_ai(p["case"]) and p["case"] in av:     # an AI estimate: next to the nearest simulation
        return p["case"], nearest_sim(p["case"], av)
    t = tag(p["case"], p["flow"])
    if t not in av:
        t = p["case"] if p["case"] in av else "narrow70"
    r = reference(VZ.clot_of(t))
    ref = tag(r, p["flow"])
    return t, ref if ref in av else (r if r in av else t)


def short_label(t):
    """Clot size and blood pressure, e.g. "50 %, 118 mmHg" (a story run or an AI estimate)."""
    if is_ai(t):
        m = ai_meta(t)
        return f"{round(100 * m['severity'])} %, {m['map_mmhg']:.0f} mmHg"
    c = VZ.clot_of(t)
    mp = next((v for f, v in FLOWS.items() if t == tag(c, f)), FLOWS["normal"])
    return f"{round(100 * V.CASES[c])} %, {mp} mmHg"


def label_of(t):
    if is_ai(t):
        m = ai_meta(t)
        return f"AI estimate: {round(100 * m['severity'])} % clot, {m['map_mmhg']:.0f} mmHg"
    flow = next((f for f in FLOWS if f != "normal" and t == tag(VZ.clot_of(t), f)), "normal")
    return VZ.LABEL[VZ.clot_of(t)] + (f", {FLOW_LABEL[flow]}" if FLOW_LABEL[flow] else "")


# ---------------------------------------------------------------------------------------------------------------
# scene state
# ---------------------------------------------------------------------------------------------------------------


class World:
    def __init__(self):
        wp.init()
        self.dev = wp.get_preferred_device()
        self.cases, self.sc, self.phi = {}, {}, {}
        self.probe_out = wp.zeros(1, dtype=wp.vec4, device=self.dev)
        for c in ("healthy", "narrow70", "occluded"):    # the rest load the first time they are picked
            self.load(c)
        self.views = {"story": ArteryView(self.cases["healthy"].geom)}

    def view(self, p):
        v = p["vessel"]
        if v not in self.views:
            self.views[v] = ArteryView(PT.geom(VESSELS[v][0]))
        return self.views[v]

    def load(self, t, ref="healthy"):
        """A results folder on the GPU, with its particles (about 10-20 s the first time)."""
        for k in (t, ref):          # re-simulated since it was loaded: drop it and every scene using it
            if k in self.cases and self.cases[k].stamp != stamp(k):
                old = self.cases.pop(k)
                self.sc = {n: sc for n, sc in self.sc.items() if sc.c is not old and sc.h is not old}
        if t in self.sc and self.sc[t].h is self.cases.get(ref):
            return
        if ref != t:
            self.load(ref, ref)
        latest["status"] = f"loading {label_of(t)}"
        if t not in self.cases:
            v = VZ.Case(t, self.dev)
            v.stamp = stamp(t)
            self.cases[t] = v
            self.phi[t] = (v.sdf.a.numpy(), np.array([v.sdf.origin[0], v.sdf.origin[1], v.sdf.origin[2]]), v.sdf.dx)
        self.sc[t] = VZ.Scene(self.cases[t], self.cases[ref], n=N_LIVE, device=self.dev)
        self.sc[t].warm(beats=0.3)
        latest["status"] = "ok"

    def section(self, p):
        return self.view(p).section(p["across_at"])

    def keep(self, p):
        if p["slice"] == "across":
            return (*self.section(p)[0], SECTION_KEEP)
        return (*self.view(p).clot_p, VZ.KEEP_R) if p["view"] == "clot" else None

    def camera(self, p, orbit_deg):
        tgt, off, up = self.view(p).presets[p["view"]]
        if p["slice"] == "across":  # look into the cut, obliquely from downstream
            x, tng = self.section(p)
            side = off / np.linalg.norm(off)
            side = side - tng * (side @ tng)
            side /= np.linalg.norm(side)
            a = math.radians(SECTION_ANGLE)
            tgt, off = x, (tng * math.cos(a) + side * math.sin(a)) * SECTION_DIST
        tgt, off, up = np.asarray(tgt, float), np.asarray(off, float), np.asarray(up, float)
        up = up / np.linalg.norm(up)
        yaw = math.radians(p["yaw"] + orbit_deg)
        v = off * math.cos(yaw) + np.cross(up, off) * math.sin(yaw) + up * (up @ off) * (1 - math.cos(yaw))
        right = np.cross(up, v)
        right /= np.linalg.norm(right)
        a = math.radians(p["pitch"])
        v = v * math.cos(a) + np.cross(right, v) * math.sin(a) + right * (right @ v) * (1 - math.cos(a))
        return VZ.look_at(tgt + v / p["zoom"], tgt, up)

    def plane(self, p, cam):
        """(plane vec4, plane_mode) for the slice choice; the camera's side is cut away."""
        if p["slice"] == "off":
            return None, 0
        pos = cam[0]
        if p["slice"] == "along":
            g = self.view(p).g
            if p["view"] == "clot":
                return VZ.clot_plane(g, cam), 2
            c, n, _ = VZ.segment_frame(g)
            n = n if n @ (pos - c) > 0 else -n
            return np.array([*n, -n @ c]), 2
        x, tng = self.section(p)
        tng = tng if tng @ (pos - x) > 0 else -tng
        return np.array([*tng, -tng @ x]), 2

    def render(self, case, p, cam, w, h, ss):
        plane, mode = self.plane(p, cam)
        wall = {"pressure": 0, "change": 1, "plain": 2, "shear": 3}[p["wall"]]
        section = {"speed": 0, "pressure": 1, "change": 2}[p["field"]]
        keep = self.keep(p)
        return self.sc[case].render(cam, w, h, ss=ss, wall_mode=wall, glass=VZ.GLASS if p["particles"] else 1.0,
                                    plane=plane, plane_mode=mode, section=section, particles=p["particles"],
                                    fov=FOV, keep=keep)

    def sync(self, case, t, dt_ms):
        s = self.sc[case]
        if abs(s.t - t) > 0.05:     # not shown for a while, or a phase jump: the particles carry on at the new time
            s.t = t
        else:
            s.advance(min((t - s.t) * 1e3, MAX_STEP_MS))
            s.t = t

    # --- click to read: the first visible lumen point along the pixel's ray
    def probe(self, case, ref, p, cam, x, y, w, h, t):
        pos, fwd, right, up = cam
        th = math.tan(math.radians(FOV) / 2)
        d = fwd + right * (2 * x - 1) * th * (w / h) + up * (1 - 2 * y) * th
        d /= np.linalg.norm(d)
        phi, org, dx = self.phi[case]
        steps = np.arange(0.0, 160.0, 0.03)
        P = pos[None] + d[None] * steps[:, None]
        val = map_coordinates(phi, ((P - org) / dx).T, order=1, mode="constant", cval=1.0)
        inside = val < 0
        plane, _ = self.plane(p, cam)
        if plane is not None:
            inside &= (P @ plane[:3] + plane[3]) <= 0
        keep = self.keep(p)
        if keep is not None:
            inside &= np.linalg.norm(P - np.array(keep[:3]), axis=1) < keep[3]
        idx = np.flatnonzero(inside)
        if not len(idx):
            return None
        i0 = idx[0]
        if plane is not None and i0 > 0 and val[i0 - 1] < 0:
            X = P[i0]                                   # on the cut face: read exactly what is shown
        else:
            run = i0
            while run + 1 < len(inside) and inside[run + 1]:
                run += 1
            X = 0.5 * (P[i0] + P[run])                  # through the wall: the middle of the vessel along the ray
        out = {}
        for c, key in ((case, "this"), (ref, "healthy")):
            wp.launch(VZ.k_probe, dim=1, device=self.dev, inputs=[self.cases[c].field(t), wp.vec3(*map(float, X))],
                      outputs=[self.probe_out])
            v = self.probe_out.numpy()[0]
            out[key] = {"p": float(v[3]), "speed": float(np.linalg.norm(v[:3]))} if v[3] > -500 else None
        return {"point": X.round(2).tolist(), "branch": self.view(p).branch(X), "values": out}


class ArteryView:
    """One artery's camera presets (target, offset from target, up), close-up point and centreline (the MCA for the
    story; the whole stretch for a patient), from a story-shaped geometry dict (vessel.build / patient.geom)."""

    def __init__(self, g):
        self.g = g
        c, n, _ = VZ.segment_frame(g)
        o = VZ.overview_cam(g, fov=FOV)
        cc = VZ.clot_cam(g)
        clot_t = cc[0] + cc[1] * 17.0
        b = g["branches"]["blocked"]
        self.mca = b["C"][b["segment_from"]:b["segment_to"]]
        self.mca_s = V._arc(self.mca)
        # junction: the ICA's split (story: the origin; patient: the carotid terminus)
        J = np.zeros(3) if "patient" not in g else self.mca[int(np.argmin(np.abs(self.mca_s - g["terminus"])))]
        if "patient" in g:   # aim at the fitted middle (a patient's hooked vessel is lopsided about its centroid)
            c = o[0] + o[1] * ((c - o[0]) @ o[1])
        self.presets = {
            "overview": (c, o[0] - c, o[3]),
            "clot": (clot_t, cc[0] - clot_t, cc[3]),
            "junction": (J, (o[0] - c) / np.linalg.norm(o[0] - c) * 16.0, o[3]),
        }
        self.clot_p, _ = VZ.clot_point(g)

    def section(self, f):
        """Cross-section point at fraction f along the centreline and its downstream tangent."""
        k = int(np.argmin(np.abs(self.mca_s - f * self.mca_s[-1])))
        tng = self.mca[min(k + 2, len(self.mca) - 1)] - self.mca[max(k - 2, 0)]
        return self.mca[k], tng / np.linalg.norm(tng)

    def branch(self, X):
        if "patient" in self.g:   # the nearest path; on any path, the stretch before the terminus is the carotid
            d, s, name = min((float(np.linalg.norm(br["C"] - X, axis=1).min()),
                              V._arc(br["C"])[int(np.argmin(np.linalg.norm(br["C"] - X, axis=1)))], br["name"])
                             for br in self.g["branches"].values())
            return "carotid (ICA)" if s < self.g["terminus"] else name
        b = self.g["branches"]
        near = min(((np.linalg.norm(br["C"] - X, axis=1).min(), name) for name, br in b.items()))[1]
        return {"trunk": "carotid (ICA)", "blocked": "middle cerebral (MCA)", "sibling": "anterior cerebral (ACA)"}[near]


# ---------------------------------------------------------------------------------------------------------------
# render loop
# ---------------------------------------------------------------------------------------------------------------


def label(img, text, xy):
    VZ.caption(img, text, xy, 26, (255, 214, 90))


def render_loop():
    try:
        world = World()
    except Exception as e:  # noqa: BLE001
        latest["status"] = f"failed: {e}"
        raise
    W = world
    latest["status"] = "compiling GPU kernels (first run only)"
    last = time.perf_counter()
    orbit = 0.0
    fps = 0.0
    last_frame = 0.0
    prev_key = None
    while True:
        with params_lock:
            p = dict(params)
            seek, clock["seek"] = clock["seek"], None
        now = time.perf_counter()
        if watchers[0] == 0 and probes.empty():
            time.sleep(0.2)
            last = now
            continue
        dt = min(now - last, 0.5)
        last = now
        if seek is not None:
            clock["t"] = math.floor(clock["t"] / VZ.PERIOD) * VZ.PERIOD + seek
        moving = p["playing"] or p["orbit"]
        step = min(dt * p["slowmo"], MAX_STEP_MS * 1e-3) if p["playing"] else 0.0
        clock["t"] += step     # at a low frame rate the particles set the pace, so the clock follows them
        latest["actual"] = 0.8 * latest["actual"] + 0.2 * (step / max(dt, 1e-6))
        if p["orbit"]:
            orbit = (orbit + dt * 12.0) % 360.0
        t = clock["t"]
        w, h = RESOLUTIONS[p["resolution"]]
        cam = W.camera(p, orbit)
        # probe requests (GPU work stays on this thread)
        while not probes.empty():
            req, box = probes.get_nowait()
            lc = latest["cam"]
            box["result"] = W.probe(*pick(p), p, lc[0], req["x"], req["y"], *lc[1], t) if lc else None
            box["done"].set()
        key = (t, tuple(sorted(p.items())), orbit)
        if key == prev_key and seek is None:
            time.sleep(0.02)
            continue
        prev_key = key
        wait = last_frame + 1.0 / MAX_FPS - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        last_frame = time.perf_counter()
        t0 = time.perf_counter()
        ss = int(p["quality"])
        cur, ref = pick(p)
        W.load(cur, ref)
        side = p["compare"] == "side" and cur != ref
        shown = [cur] + ([ref] if side else [])
        for c in shown:
            W.sync(c, t, step * 1e3)
        if side:
            hw = w // 2
            img = Image.new("RGB", (w, h))
            img.paste(Image.fromarray(W.render(ref, p, cam, hw, h, ss)), (0, 0))
            img.paste(Image.fromarray(W.render(cur, p, cam, w - hw, h, ss)), (hw, 0))
            if is_ai(cur):              # short: each half is narrow
                label(img, "Simulation: " + short_label(ref), (24, 18))
                label(img, "AI estimate: " + short_label(cur), (hw + 24, 18))
            else:
                label(img, label_of(ref), (24, 18))
                label(img, label_of(cur), (hw + 24, 18))
            pcam = (cam, (w - hw, h))   # picking uses the right half's camera
        else:
            q = dict(p)
            if p["compare"] == "change":
                q.update(wall="change", field="change" if p["slice"] != "off" else p["field"])
            img = Image.fromarray(W.render(cur, q, cam, w, h, ss))
            if is_ai(cur):              # never mistaken for a simulation, also in a screenshot
                label(img, label_of(cur), (24, 18))
            pcam = (cam, (w, h))
        render_ms = (time.perf_counter() - t0) * 1000
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        fps = 0.9 * fps + 0.1 / max(time.perf_counter() - now, 1e-6)
        with frame_cond:
            latest.update(jpeg=buf.getvalue(), id=latest["id"] + 1, fps=fps, render_ms=render_ms, status="ok",
                          cam=pcam, phase=t % VZ.PERIOD, moving=moving,
                          split=side, shown=cur, ref=ref)
            frame_cond.notify_all()


# ---------------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------------


ICON_TYPES = {".svg": "image/svg+xml", ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg",
              ".jpeg": "image/jpeg", ".ico": "image/x-icon"}


def icon_file(side):
    """The user's sidebar icon: icons/panel-<side>.<svg|png|...>, or None (the page then draws its own)."""
    for ext in ICON_TYPES:
        f = os.path.join(HERE, "icons", f"panel-{side}{ext}")
        if os.path.isfile(f):
            return f
    return None


_static = {}


def info():
    """Data for the page: per case numbers and the heartbeat series (finished runs), colour bars."""
    if not _static:
        _static.update(static_info())
    out = dict(_static, cases={c: case_info(c) for c in sorted(available())})
    if "healthy" in out["cases"]:   # g of brain the MCA feeds: healthy flow at normal pressure = normal tissue flow
        out["territory_g"] = 100.0 * out["cases"]["healthy"]["Q_mca"] / CBF["normal"]
    out["available"] = sorted(available())
    out["ai"] = dict(ai_state, clot=AI_CLOT, map=AI_MAP)
    return out


def static_info():
    import matplotlib
    out = {"period": VZ.PERIOD, "choices": CHOICES, "flows": FLOWS, "cbf": CBF,
           "ranges": {"speed": [0, VZ.SPEED_MAX], "pressure": list(VZ.P_RANGE), "change": [-VZ.DP_RANGE, VZ.DP_RANGE],
                      "shear": list(VZ.WSS_RANGE)},
           "log": {"shear": [0.1, 1, 10, 100]},      # ticks of the log-scaled colour bars
           "colours": {}}
    for name, cmap in (("speed", "turbo"), ("pressure", "RdYlBu_r"), ("change", "RdBu_r"), ("shear", "viridis")):
        cs = matplotlib.colormaps[cmap](np.linspace(0, 1, 11))[:, :3]
        out["colours"][name] = ["#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c) for c in cs]
    # per artery: its cases, the centreline the cross-section slider runs along (with the clot / lesion on it)
    out["vessels"] = {}
    for v, cases in VESSELS.items():
        g = V.build("healthy") if v == "story" else PT.geom(cases[0])
        b = g["branches"]["blocked"]
        mca = b["C"][b["segment_from"]:b["segment_to"]]
        cl = g["clot"]
        mid = cl["face_arc"] + 0.5 * cl["body_length"]
        if v != "story":   # a patient's narrowest point: its probe
            mid = {p[0]: p[1] for p in PT.PATIENTS[v]["probes"]}.get("narrowing", mid)
        out["vessels"][v] = {"cases": cases, "default": DEFAULT[v], "reference": reference(cases[0]),
                             "mca": {"length": float(V._arc(mca)[-1]), "clot_start": cl["face_arc"] - cl["ramp"],
                                     "clot_mid": mid, "clot_end": cl["face_arc"] + cl["body_length"] + cl["ramp"]}}
        if v != "story":
            c = PT.PATIENTS[v]
            out["vessels"][v].update(side=c["side"], vessel=c["vessel"], terminus=c["terminus"], title=c["title"],
                                     scans={f"{v}_{k}": t for k, t in c["scans"].items()}, note=c.get("note", ""))
    return out


_case_info = {}


def case_info(c):
    """The page's numbers and heartbeat series for one results folder (cached until it is re-simulated)."""
    if is_ai(c):
        m = ai_meta(c)
        near = nearest_sim(c, available())
        return dict(m["summary"], Q_mca=max(m["summary"]["Q_mca"], 0.0), label=label_of(c),
                    ai=True, severity=m["severity"], map_mmhg=m["map_mmhg"], secs=m.get("secs"), model=m["model"],
                    trained_on=len(m["trained_on"]), nearest=near, nearest_label=label_of(near),
                    trained_clots=sorted({round(100 * V.CASES[VZ.clot_of(r)]) for r in m["trained_on"]
                                          if VZ.clot_of(r) in V.CASES}))
    if c not in _case_info or _case_info[c]["stamp"] != stamp(c):
        st = VZ.stats(c)
        names = st["names"]
        pp = st["p_probe"]
        before, after = VZ.around(json.load(open(os.path.join(VZ.RESULTS, c, "meta.json"))))
        s = np.load(os.path.join(VZ.RESULTS, c, "series.npz"))
        m = s["t"] > s["t"][-1] - VZ.PERIOD + 1e-9
        Q = s["Q"][m]
        k = max(1, len(pp) // 200)
        _case_info[c] = {"stamp": stamp(c),
            "label": label_of(c), "Q_mca": max(st["Q_mca"], 0.0), "Q_aca": st["Q_aca"], "Q_in": st["Q_in"],
            "drop": st["drop"], "umax": st["umax"],
            "p_before": float(np.nanmean(pp[:, names.index(before)])),
            "p_after": float(np.nanmean(pp[:, names.index(after)])),
            "series": {"t": [round(float(x), 4) for x in st["t"][::k]],
                       "before": [round(float(x), 2) for x in pp[::k, names.index(before)]],
                       "after": [round(float(x), 2) for x in pp[::k, names.index(after)]],
                       "q_mca": [round(float(x), 2) for x in Q[::k, 1]]},
        }
    return dict(_case_info[c], **wss_info(c))


def wss_info(c):
    """Wall shear stress numbers (wss.py's summary; read each time, as it is filled in after the runs)."""
    try:
        s = json.load(open(os.path.join(VZ.RESULTS, "wss_summary.json"))).get(c)
    except (OSError, ValueError):
        s = None
    if not s:
        return {}
    g = (s.get("regions") or {}).get("MCA at the clot") or s["all"]
    return {"wss_clot": g["tawss_mean_pa"], "wss_peak": s["all"]["peak_pa"], "wss_low": s["all"]["low_tawss_area_pct"],
            "wss_where": "at the clot" if "regions" in s else "whole vessel"}


def ai_call(path, timeout):
    with urllib.request.urlopen(AI_URL + path, timeout=timeout) as r:
        return json.loads(r.read())


def ai_service():
    """Make sure the AI estimate service runs: start it (in the PhysicsNeMo environment, ai/) when nothing answers on
    its port; it is a child of this server, so stop.bat stops both."""
    with ai_lock:       # one start at a time (the page may ask while it is still starting)
        _ai_service()


ai_lock = threading.Lock()


def _ai_service():
    try:
        ai_state.update(status="ready", info=ai_call("/health", 2))
        return
    except (urllib.error.URLError, OSError, ValueError):
        pass
    if not os.path.exists(os.path.join(AI_DIR, "runs", "mlp", "model.pt")):
        ai_state["status"] = "unavailable: no trained model (ai/train.py)"
        return
    ai_state["status"] = "starting"
    log = open(os.path.join(AI_DIR, "estimate.log"), "a")
    subprocess.Popen(["uv", "run", "python", "estimate.py", "--serve"], cwd=AI_DIR, stdout=log, stderr=subprocess.STDOUT,
                     env=dict(os.environ, PYTHONUNBUFFERED="1"), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(180):
        time.sleep(1.0)
        try:
            ai_state.update(status="ready", info=ai_call("/health", 2))
            return
        except (urllib.error.URLError, OSError, ValueError):
            pass
    ai_state["status"] = "failed to start (see ai/estimate.log)"


def estimate(body):
    """Page request {clot: %, map: mmHg} -> an AI estimate, then shown (the render loop loads it like a run)."""
    try:
        clot = min(max(float(body["clot"]), AI_CLOT[0]), AI_CLOT[1])
        mp = min(max(float(body["map"]), AI_MAP[0]), AI_MAP[1])
    except (KeyError, TypeError, ValueError):
        return {"error": "need a clot size and a blood pressure"}
    if ai_state["status"] != "ready":
        ai_service()
    if ai_state["status"] != "ready":
        return {"error": f"AI model {ai_state['status']}"}
    try:
        r = ai_call(f"/estimate?severity={clot / 100:.4f}&map={mp:.1f}", 300)
    except (urllib.error.URLError, OSError, ValueError) as e:
        ai_state["status"] = "starting"
        threading.Thread(target=ai_service, daemon=True).start()
        return {"error": f"AI model not answering ({e}); restarting it, try again in a minute"}
    if "error" in r:
        return r
    _avail["at"] = -1e9                       # the new estimate joins the available results now
    available()
    with params_lock:
        params.update(case=r["tag"], flow="normal", vessel="story")
        s = dict(params)
    return {"tag": r["tag"], "secs": r["secs"], "state": s}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_body(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "explore.html"), "rb") as f:
                self.send_body(f.read(), "text/html; charset=utf-8")
        elif self.path.startswith("/stream"):
            self.stream()
        elif self.path == "/state":
            with params_lock:
                s = dict(params)
            s.update(ai=ai_state["status"])
            s.update(fps=latest["fps"], render_ms=latest["render_ms"], status=latest["status"], actual=latest["actual"],
                     available=sorted(available()), shown=latest.get("shown"), ref=latest.get("ref"),
                     phase=latest.get("phase", 0.0),
                     page=os.path.getmtime(os.path.join(HERE, "explore.html")))   # open pages reload when it changes
            self.send_body(json.dumps(s).encode(), "application/json")
        elif self.path in ("/icon/left", "/icon/right"):
            f = icon_file(self.path.rsplit("/", 1)[1])
            if f:
                with open(f, "rb") as fh:
                    self.send_body(fh.read(), ICON_TYPES[os.path.splitext(f)[1].lower()])
            else:
                self.send_body(b"no icon", "text/plain", 404)
        elif self.path == "/info":
            self.send_body(json.dumps(info()).encode(), "application/json")
        else:
            self.send_body(b"not found", "text/plain", 404)

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except json.JSONDecodeError:
            self.send_body(b"bad json", "text/plain", 400)
            return
        if self.path == "/probe":
            try:
                req = {"x": float(body["x"]), "y": float(body["y"])}
            except (KeyError, TypeError, ValueError):
                self.send_body(b"bad probe", "text/plain", 400)
                return
            if latest.get("split"):     # side by side: the click must be on the right half (the selected case)
                if req["x"] < 0.5:
                    self.send_body(json.dumps({"error": "click the right-hand artery"}).encode(), "application/json")
                    return
                req["x"] = (req["x"] - 0.5) * 2
            box = {"done": threading.Event(), "result": None}
            probes.put((req, box))
            box["done"].wait(5.0)
            self.send_body(json.dumps(box["result"] or {"error": "no artery under the cursor"}).encode(),
                           "application/json")
            return
        if self.path == "/estimate":
            self.send_body(json.dumps(estimate(body)).encode(), "application/json")
            return
        if self.path != "/params":
            self.send_body(b"not found", "text/plain", 404)
            return
        with params_lock:
            before = dict(params)
            for k, opts in CHOICES.items():
                if body.get(k) in opts:
                    params[k] = body[k]
            if isinstance(body.get("case"), str) and is_ai(body["case"]) and body["case"] in available():
                params.update(case=body["case"], flow="normal")
            # picking an artery picks its default case and the whole view; a case brings its artery
            if body.get("vessel") in VESSELS and body["vessel"] != before["vessel"] and "case" not in body:
                params["case"] = DEFAULT[body["vessel"]]
                params.update(yaw=0.0, pitch=0.0, zoom=1.0, view="overview")
            if params["case"] != before["case"]:
                params["vessel"] = vessel_of(params["case"])
            if tag(params["case"], params["flow"]) not in available():   # not simulated (yet): normal flow,
                params["flow"] = "normal"
                if params["case"] not in available():                    # or the clot as it was
                    params["case"], params["flow"] = before["case"], before["flow"]
                    params["vessel"] = before["vessel"]
            for k in ("particles", "orbit", "playing"):
                if k in body:
                    params[k] = bool(body[k])
            for k, (lo, hi) in RANGES.items():
                if k in body:
                    try:
                        v = float(body[k])
                    except (TypeError, ValueError):
                        continue
                    params[k] = type(lo)(min(max(v, lo), hi)) if not isinstance(lo, int) else int(min(max(v, lo), hi))
            if "phase" in body:
                try:
                    clock["seek"] = float(body["phase"]) % VZ.PERIOD
                except (TypeError, ValueError):
                    pass
            if body.get("reset_view"):
                params.update(yaw=0.0, pitch=0.0, zoom=1.0)
            s = dict(params)
        self.send_body(json.dumps(s).encode(), "application/json")

    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        seen = -1
        watchers[0] += 1
        try:
            while True:
                with frame_cond:
                    frame_cond.wait_for(lambda: latest["id"] != seen and latest["jpeg"] is not None)
                    seen, body = latest["id"], latest["jpeg"]
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(body)).encode()
                                 + b"\r\n\r\n" + body + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            watchers[0] -= 1


def lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def main():
    threading.Thread(target=render_loop, daemon=True).start()
    threading.Thread(target=ai_service, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"Explorer at http://127.0.0.1:{PORT}, on the network http://{lan_ip()}:{PORT}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
