"""Step 15: the AI estimate - pressure and velocity over a heartbeat for ANY clot size and blood pressure in the story
artery, from the trained model (Step 14; the plain MLP, the user's choice), in about a second instead of hours.

For a clot size (0-100 %) and a mean blood pressure (mmHg): the vessel with that clot (vessel.build), its lumen cells on
the saved fields' 0.1 mm grid (as a 0.1 mm simulation's), the model's inputs at every cell (dataset.py) and its
prediction at the 40 frames of the beat. Written like a simulation's results, to results/ai_s<clot %>_m<mmHg>/
(fields.npz + meta.json with "ai": true and the numbers the explorer shows: flows in the same slabs inside the open
ends and pressures in the same probes as the simulations), so the explorer shows it with the same renderer.
The newest KEEP estimates are kept; an estimate made by an older model is redone.

    uv run estimate.py --serve                     a local service for the explorer: http://127.0.0.1:8004
    uv run estimate.py 45 110                      one estimate (clot 45 %, 110 mmHg), printed
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)
import dataset as D  # noqa: E402
import train as T  # noqa: E402
import vessel as V  # noqa: E402

HOST, PORT = "127.0.0.1", 8004       # this computer only: the explorer (8003) asks it
MODEL = "mlp"
DX_MM = 0.1
CHUNK = 1 << 18                      # cells per forward pass (the MLP sees each cell on its own)
KEEP = 12                            # estimates kept on disk
MAP_RANGE = (66.0, 146.0)            # mmHg: the trained pressure levels (flow 70 % ... 146 mmHg)
FLUX_SLAB_MM = (2.5, 3.5)            # as simulate.py: flows measured this deep inside each open end
MMHG = 133.322


class Estimator:
    def __init__(self, model=MODEL):
        import warp as wp
        wp.config.quiet = True
        wp.init()
        self.model = model
        self.ck = os.path.join(T.RUNS, model, "model.pt")
        state = torch.load(self.ck, map_location=T.dev)
        cfg = state["cfg"]
        kw = {"hidden": cfg["layer_size"], "layers": cfg["num_layers"]} if model == "mlp" else \
             {"hidden": cfg["n_hidden"], "layers": cfg["n_layers"], "slices": cfg["slice_num"]}
        self.net, _ = T.make(model, **kw)
        self.net.load_state_dict(state["net"])
        self.net.eval()
        self.norm = T.Norm(state["norm"]["centre"], state["norm"]["scale"])
        self.stamp = os.path.getmtime(self.ck)
        self.steps = state.get("step")
        index = json.load(open(os.path.join(D.OUT, "index.json")))
        self.trained_on = sorted(e["run"] for e in index["runs"] if e["split"] == "train")
        # the simulations' shared set-up: the healthy run's calibration, open ends, probes and frame times
        h = os.path.join(RESULTS, "healthy")
        self.meta_h = json.load(open(os.path.join(h, "meta.json")))
        self.t = np.load(os.path.join(h, "fields.npz"))["t"]
        self.map_normal = self.meta_h["calibration"]["P_in_coef"][0] / MMHG
        self.gh = V.build("healthy")
        self.lo, self.hi = V.bounds(self.gh)
        self.lock = threading.Lock()

    def info(self):
        return {"model": self.model, "steps": self.steps, "trained_on": self.trained_on, "map_range": MAP_RANGE,
                "map_normal": self.map_normal}

    def run(self, severity, map_mmhg):
        """(results folder, its meta) for a clot size (0-1) and a mean blood pressure (mmHg); cached."""
        severity = float(np.clip(severity, 0.0, 1.0))
        map_mmhg = float(np.clip(map_mmhg, *MAP_RANGE))
        tag = f"ai_s{round(severity * 100)}_m{round(map_mmhg)}"
        d = os.path.join(RESULTS, tag)
        with self.lock:
            try:
                meta = json.load(open(os.path.join(d, "meta.json")))
                if meta.get("model_stamp") == self.stamp and os.path.exists(os.path.join(d, "fields.npz")):
                    os.utime(os.path.join(d, "meta.json"))         # recently used: kept longest
                    return tag, meta
            except (OSError, ValueError):
                pass
            t0 = time.time()
            meta = self._estimate(d, tag, round(severity * 100) / 100, round(map_mmhg))
            meta["secs"] = round(time.time() - t0, 2)
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
            self._tidy()
            return tag, meta

    def _estimate(self, d, tag, severity, map_mmhg):
        g = V.build("healthy", severity=severity)
        phi, origin = V.sdf_grid(g, DX_MM, self.lo, self.hi)
        coords = np.argwhere(phi < 0)
        pos = (origin + coords * DX_MM).astype(np.float32)
        wall = -phi[tuple(coords.T)]
        inf, _ = D.along(g, pos)
        onehot = np.eye(len(D.BRANCHES), dtype=np.float32)[inf[:, 0].astype(int)]
        local = np.column_stack([wall, inf[:, 1], inf[:, 2], inf[:, 3], inf[:, 4:7], onehot]).astype(np.float32)
        level = map_mmhg / self.map_normal
        t = np.asarray(self.t, float)
        frames = np.column_stack([t, (t % 1.0) / 1.0, D.drive(self.meta_h, level, t)]).astype(np.float32)
        P, U = self.predict(pos, local, severity, level, frames)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "fields.npz"), coords=coords.astype(np.int16), origin=origin, dx_mm=DX_MM,
                 wall_mm=wall.astype(np.float32), t=t, p_mmhg=P.astype(np.float16), u_m_s=U.astype(np.float16))
        return {"case": tag, "done": True, "ai": True, "model": self.model, "model_stamp": self.stamp,
                "model_steps": self.steps, "severity": severity, "map_mmhg": map_mmhg, "level": level, "dx_mm": DX_MM,
                "cells": int(len(pos)), "calibration": self.meta_h["calibration"], "open_ends": self.meta_h["open_ends"],
                "probes": self.meta_h["probes"], "trained_on": self.trained_on,
                "summary": self.summary(pos, P, U)}

    def predict(self, pos, local, severity, level, frames):
        cells = torch.from_numpy(self.norm.cell_inputs(pos, local)).to(T.dev)
        n = len(pos)
        P, U = np.zeros((len(frames), n), np.float32), np.zeros((len(frames), n, 3), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for j, fr in enumerate(frames):
                G = torch.from_numpy(self.norm.globals_(severity, level, fr)).to(T.dev)[None, None]
                y = torch.cat([T.forward(self.model, self.net, cells[i:i + CHUNK][None], G)[0].float()
                               for i in range(0, n, CHUNK)]).cpu().numpy()
                P[j] = y[:, 0] * T.P_SCALE + T.P_REF
                U[j] = y[:, 1:] * T.U_SCALE
        return P, U

    def summary(self, pos, P, U):
        """The explorer's numbers, measured as the simulations measure them (simulate.py): flows through slabs inside
        the open ends, pressures in the probe spheres; plus the per-frame series for its chart."""
        q = {}
        thick = FLUX_SLAB_MM[1] - FLUX_SLAB_MM[0]
        for k, e in enumerate(self.meta_h["open_ends"]):
            c, nrm = np.asarray(e["centre"]), np.asarray(e["normal"])
            rel = pos - c
            depth = -(rel @ nrm)
            side = np.linalg.norm(rel + depth[:, None] * nrm, axis=1) < 1.3 * e["radius"]
            m = side & (depth >= FLUX_SLAB_MM[0]) & (depth < FLUX_SLAB_MM[1])
            sign = -1.0 if k == 0 else 1.0
            q[e["label"]] = sign * (U[:, m] @ nrm).sum(1) * (DX_MM * 1e-3) ** 3 / (thick * 1e-3) * 6e7   # mL/min
        pr = {}
        for p in self.meta_h["probes"]:
            m = np.linalg.norm(pos - np.asarray(p["point"]), axis=1) < p["radius"]
            pr[p["name"]] = P[:, m].mean(1) if m.any() else np.full(len(P), np.nan)   # the throat may be shut
        before, after = pr["MCA before clot"], pr["MCA after clot"]
        ph = self.t % 1.0
        o = np.argsort(ph)                                  # the chart runs over one beat from its start
        r2 = lambda a: [round(float(x), 2) for x in a[o]]
        return {"Q_in": float(q["inlet (ICA)"].mean()), "Q_mca": float(q["MCA outlet"].mean()),
                "Q_aca": float(q["ACA outlet"].mean()), "drop": float(np.nanmean(before - after)),
                "p_before": float(np.nanmean(before)), "p_after": float(np.nanmean(after)),
                "umax": float(np.linalg.norm(U, axis=2).max()),
                "series": {"t": r2(ph), "before": r2(before),
                           "after": r2(after), "q_mca": r2(q["MCA outlet"])}}

    def _tidy(self):
        ai = sorted((os.path.getmtime(os.path.join(RESULTS, n, "meta.json")), n) for n in os.listdir(RESULTS)
                    if n.startswith("ai_") and os.path.exists(os.path.join(RESULTS, n, "meta.json")))
        for _, n in ai[:-KEEP]:
            shutil.rmtree(os.path.join(RESULTS, n), ignore_errors=True)


def serve(est):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/health":
                self.reply(est.info())
            elif u.path == "/estimate":
                q = parse_qs(u.query)
                try:
                    sev, mp = float(q["severity"][0]), float(q["map"][0])
                except (KeyError, ValueError):
                    self.reply({"error": "need severity (0-1) and map (mmHg)"}, 400)
                    return
                try:
                    tag, meta = est.run(sev, mp)
                except Exception as e:  # noqa: BLE001
                    self.reply({"error": f"{type(e).__name__}: {e}"}, 500)
                    return
                self.reply({"tag": tag, "secs": meta.get("secs"), "summary": {k: v for k, v in meta["summary"].items()
                                                                               if k != "series"}})
            else:
                self.reply({"error": "not found"}, 404)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"AI estimate service ({est.model}, {len(est.trained_on)} training runs) at http://{HOST}:{PORT}", flush=True)
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clot", nargs="?", type=float, help="clot size, % narrowing")
    ap.add_argument("map", nargs="?", type=float, help="mean blood pressure, mmHg")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--model", default=MODEL, choices=["mlp", "geotransolver"])
    a = ap.parse_args()
    est = Estimator(a.model)
    if a.serve:
        serve(est)
        return
    tag, meta = est.run(a.clot / 100, a.map)
    s = meta["summary"]
    print(f"{tag}: {meta.get('secs', 'cached')} s  MCA {s['Q_mca']:.1f} mL/min, ACA {s['Q_aca']:.1f}, in {s['Q_in']:.1f};"
          f"  drop {s['drop']:.1f} mmHg (before {s['p_before']:.1f}, after {s['p_after']:.1f});  umax {s['umax']:.2f} m/s")


if __name__ == "__main__":
    main()
