"""Step 13: the training set for the AI model, from the story artery's finished simulations.

Per run (one clot size x one pressure level), at every saved lumen cell (0.1 mm grid):
  inputs   position (mm), distance to the wall, distance along the artery and which branch, the healthy and the
           clotted radius there; per frame: clot size, pressure level, time in the beat, inlet pressure now
  targets  pressure (mmHg) and velocity (m/s) at the 40 frames of the last heartbeat
Written to ai/data/<run>/ (arrays as .npy, memory-mappable; fields as float16) plus ai/data/index.json.
Held out (never trained on): the whole 50 % clot and healthy at 118 mmHg.

    uv run dataset.py          (from artery_flow/ai)
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(HERE, "data")
sys.path.insert(0, ROOT)
import vessel  # noqa: E402

MMHG = 133.322
LEVELS = {"": "flow 100 %", "_flow70": "flow 70 %", "_flow130": "flow 130 %", "_map118": "118 mmHg", "_map146": "146 mmHg"}
HELD_OUT = {"narrow50", "narrow50_flow70", "narrow50_flow130", "narrow50_map118", "narrow50_map146", "healthy_map118"}
BRANCHES = ("trunk", "blocked", "sibling")


def level(run, meta):
    """The pressure level as the factor on the sourced inlet pressure waveform (simulate.run's `flow`)."""
    mean = meta["calibration"]["P_in_coef"][0] / MMHG
    suffix = run[len(run.split("_")[0]):]
    if suffix.startswith("_map"):
        return float(suffix[4:]) / mean
    if suffix.startswith("_flow"):
        return float(suffix[5:]) / 100
    return 1.0


def drive(meta, k, t, period=1.0):
    """The prescribed driving pressure (mmHg) at times t: the sourced Fourier waveform x the level (lbm set_drive).
    Known before any simulation, unlike the pressure measured at the inlet, which depends on the clot."""
    c = np.asarray(meta["calibration"]["P_in_coef"])
    ph = 2 * np.pi * np.asarray(t) / period
    d = c[0] + sum(c[2 * j - 1] * np.cos(j * ph) + c[2 * j] * np.sin(j * ph) for j in range(1, (len(c) - 1) // 2 + 1))
    return k * d / MMHG


def along(geo, pos):
    """Per cell: the nearest centreline point's branch, arc length (mm, from the junction; the trunk negative),
    healthy radius, clotted radius and the flow direction (unit tangent, inlet -> outlets)."""
    pts, info = [], []
    for b, name in enumerate(BRANCHES):
        B = geo["branches"][name]
        C = B["C"]
        s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))])
        s = s - s[-1] if name == "trunk" else s        # the trunk ends at the junction
        t = np.gradient(C, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        pts.append(C)
        info.append(np.column_stack([np.full(len(C), b), s, B["R0"], B["R"], t]))
    pts, info = np.vstack(pts), np.vstack(info)
    d, i = cKDTree(pts).query(pos)
    return info[i], d


def one(run):
    d = os.path.join(RESULTS, run)
    meta = json.load(open(os.path.join(d, "meta.json")))
    f = np.load(os.path.join(d, "fields.npz"))
    s = np.load(os.path.join(d, "series.npz"))
    case = run.split("_")[0]
    geo = vessel.build(case)
    pos = (f["origin"] * (1e3 if np.abs(f["origin"]).max() < 1 else 1) + f["coords"] * float(f["dx_mm"])).astype(np.float32)
    inf, dist = along(geo, pos)
    if np.percentile(dist, 99) > 4.0:      # the cells must sit around the vessel's centreline
        raise RuntimeError(f"{run}: cells are {np.percentile(dist, 99):.1f} mm from the centreline; frame mismatch")
    o = os.path.join(OUT, run)
    os.makedirs(o, exist_ok=True)
    onehot = np.eye(len(BRANCHES), dtype=np.float32)[inf[:, 0].astype(int)]
    local = np.column_stack([f["wall_mm"], inf[:, 1], inf[:, 2], inf[:, 3], inf[:, 4:7], onehot]).astype(np.float32)
    np.save(os.path.join(o, "pos.npy"), pos)
    np.save(os.path.join(o, "local.npy"), local)
    np.save(os.path.join(o, "p.npy"), f["p_mmhg"].astype(np.float16))
    np.save(os.path.join(o, "u.npy"), f["u_m_s"].astype(np.float16))
    t = np.asarray(f["t"], float)
    period = 1.0
    frames = np.column_stack([t, (t % period) / period, drive(meta, level(run, meta), t)]).astype(np.float32)
    np.save(os.path.join(o, "frames.npy"), frames)
    return {"run": run, "case": case, "severity": geo["severity"], "level": level(run, meta),
            "level_name": LEVELS[run[len(case):]], "split": "test" if run in HELD_OUT else "train", "cells": len(pos),
            "p_range": [float(f["p_mmhg"].min()), float(f["p_mmhg"].max())],
            "speed_max": float(np.linalg.norm(f["u_m_s"], axis=2).max()), "centreline_dist_p99": float(np.percentile(dist, 99))}


def main():
    runs = [c + sfx for c in vessel.CASES for sfx in LEVELS
            if os.path.exists(os.path.join(RESULTS, c + sfx, "fields.npz"))]
    os.makedirs(OUT, exist_ok=True)
    index = []
    for r in runs:
        e = one(r)
        index.append(e)
        print(f"{r:18s} {e['split']:5s} severity {e['severity']:.1f} level {e['level']:.3f}  {e['cells']} cells  "
              f"p {e['p_range'][0]:.0f}..{e['p_range'][1]:.0f} mmHg  max {e['speed_max']:.2f} m/s", flush=True)
    local_names = ["wall_mm", "s_mm", "R0_mm", "R_mm", "tx", "ty", "tz", "trunk", "blocked", "sibling"]
    json.dump({"runs": index, "local": local_names, "frames": ["t_s", "phase", "p_in_mmhg"],
               "held_out": sorted(HELD_OUT)}, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    n = {k: sum(e["split"] == k for e in index) for k in ("train", "test")}
    print(f"{len(index)} runs: {n['train']} train, {n['test']} test -> {OUT}")


if __name__ == "__main__":
    main()
