"""Step 14: train the AI model (PhysicsNeMo GeoTransolver, or a plain MLP as the baseline) and test it.

Learns, for the story artery: (cell position + local geometry, clot size, pressure level, time in the beat) ->
pressure and velocity. Trains on random subsets of cells of random (run, frame) samples from ai/data (dataset.py);
tests on the held-out runs (the whole 50 % clot, healthy at 118 mmHg) against the simulation: field errors, MCA flow,
pressure drop across the clot and top speed.

    uv run train.py --model geotransolver --hours 3       (from artery_flow/ai)
    uv run train.py --model mlp --hours 1
    uv run train.py --model geotransolver --eval          (test a trained model only)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, os.path.dirname(HERE))
import vessel  # noqa: E402

DX_M = 1e-4                    # the saved fields' grid (0.1 mm)
P_REF, P_SCALE, U_SCALE = 100.0, 50.0, 2.0      # mmHg, mmHg, m/s
dev = "cuda"


# ----------------------------------------------------------------------------------------------- data
def load(split):
    index = json.load(open(os.path.join(DATA, "index.json")))
    out = []
    for e in index["runs"]:
        if split != "all" and e["split"] != split:
            continue
        d = os.path.join(DATA, e["run"])
        r = dict(e, pos=np.load(os.path.join(d, "pos.npy")), local=np.load(os.path.join(d, "local.npy")),
                 frames=np.load(os.path.join(d, "frames.npy")),
                 p=np.load(os.path.join(d, "p.npy")), u=np.load(os.path.join(d, "u.npy")))
        out.append(r)
    return out


class Norm:
    """Fixed input scalings (kept with the model, so the explorer can use it on a new clot size)."""

    def __init__(self, centre=None, scale=None):
        self.centre, self.scale = centre, scale

    def fit(self, runs):
        P = np.vstack([r["pos"] for r in runs])
        self.centre, self.scale = P.mean(0).tolist(), float(P.std(0).max())
        return self

    def cell_inputs(self, pos, local):
        """Per cell: position, wall distance, arc length, healthy and clotted radius, tangent, branch."""
        x = (pos - np.asarray(self.centre, np.float32)) / self.scale
        L = local.copy()
        L[:, 0] /= 2.0          # wall distance, mm
        L[:, 1] /= 40.0         # arc length, mm
        L[:, 2:4] /= 2.0        # radii, mm
        return np.concatenate([x, L], 1).astype(np.float32)

    @staticmethod
    def globals_(severity, level, frame):
        """Per sample: clot size, pressure level, time in the beat (sin, cos), driving pressure now."""
        ph = 2 * math.pi * frame[1]
        return np.array([severity, level - 1.0, math.sin(ph), math.cos(ph), (frame[2] - P_REF) / P_SCALE], np.float32)

    def dict(self):
        return {"centre": self.centre, "scale": self.scale, "P_REF": P_REF, "P_SCALE": P_SCALE, "U_SCALE": U_SCALE}


def targets(p, u):
    return np.concatenate([(p[:, None].astype(np.float32) - P_REF) / P_SCALE, u.astype(np.float32) / U_SCALE], 1)


class Sampler:
    def __init__(self, runs, norm, n_points, batch, seed=0):
        self.runs, self.n, self.b = runs, n_points, batch
        self.rng = np.random.default_rng(seed)
        self.cells = [norm.cell_inputs(r["pos"], r["local"]) for r in runs]
        self.norm = norm

    def batch(self):
        X, G, Y = [], [], []
        for _ in range(self.b):
            k = self.rng.integers(len(self.runs))
            r = self.runs[k]
            f = self.rng.integers(len(r["frames"]))
            i = self.rng.choice(len(r["pos"]), self.n, replace=False)
            X.append(self.cells[k][i])
            G.append(self.norm.globals_(r["severity"], r["level"], r["frames"][f]))
            Y.append(targets(r["p"][f, i], r["u"][f, i]))
        t = lambda a: torch.from_numpy(np.stack(a)).to(dev, non_blocking=True)
        return t(X), t(G)[:, None], t(Y)


# ----------------------------------------------------------------------------------------------- models
N_CELL, N_GLOBAL, N_OUT = 13, 5, 4


def make(model, **kw):
    if model == "geotransolver":
        from physicsnemo.models.geotransolver import GeoTransolver
        cfg = dict(functional_dim=N_CELL, out_dim=N_OUT, geometry_dim=4, global_dim=N_GLOBAL, n_layers=kw.get("layers", 6),
                   n_hidden=kw.get("hidden", 256), n_head=8, slice_num=kw.get("slices", 64))
        net = GeoTransolver(**cfg)
    else:
        from physicsnemo.models.mlp import FullyConnected
        cfg = dict(in_features=N_CELL + N_GLOBAL, out_features=N_OUT, layer_size=kw.get("hidden", 512),
                   num_layers=kw.get("layers", 6))
        net = FullyConnected(**cfg)
    return net.to(dev), cfg


def forward(model, net, X, G):
    if model == "geotransolver":
        return net(X, global_embedding=G, geometry=X[..., :4])      # geometry: position + wall distance
    return net(torch.cat([X, G.expand(-1, X.shape[1], -1)], -1))


# ----------------------------------------------------------------------------------------------- testing
def predict(model, net, norm, r, n_points, frames=None, seed=1):
    """Pressure and velocity at every cell of run r: the cells are split at random into groups of n_points (the
    training sample size) and each group is predicted together."""
    rng = np.random.default_rng(seed)
    cells = torch.from_numpy(norm.cell_inputs(r["pos"], r["local"])).to(dev)
    n = len(r["pos"])
    frames = range(len(r["frames"])) if frames is None else frames
    P, U = np.zeros((len(frames), n), np.float32), np.zeros((len(frames), n, 3), np.float32)
    perm = rng.permutation(n)
    groups = np.array_split(perm, max(1, round(n / n_points)))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for j, f in enumerate(frames):
            G = torch.from_numpy(norm.globals_(r["severity"], r["level"], r["frames"][f])).to(dev)[None, None]
            for gi in groups:
                y = forward(model, net, cells[torch.from_numpy(gi).to(dev)][None], G)[0].float().cpu().numpy()
                P[j, gi] = y[:, 0] * P_SCALE + P_REF
                U[j, gi] = y[:, 1:] * U_SCALE
    return P, U


def summary(r, P, U, clot):
    """Beat-mean MCA flow through a slab past the clot, pressure drop across the clot, top speed."""
    L = r["local"]
    s, mca, tan = L[:, 1], L[:, 8] > 0.5, L[:, 4:7]
    def slab(s0, half=0.25):
        return mca & (np.abs(s - s0) < half)
    before = slab(clot["face_arc"] - 3.0)
    after = slab(clot["face_arc"] + clot["body_length"] + clot["ramp"] + 3.0)
    thick = len(np.unique(np.round(s[after], 3))) * vessel.RESAMPLE * 1e-3     # arc length comes in centreline steps
    q = (U[:, after] * tan[after]).sum(-1).sum(-1) * DX_M ** 3 / thick * 6e7   # m^3/s -> mL/min
    return {"Q_mca": float(q.mean()), "drop": float((P[:, before].mean(1) - P[:, after].mean(1)).mean()),
            "p_before": float(P[:, before].mean()), "umax": float(np.linalg.norm(U, axis=2).max())}


def evaluate(model, net, norm, runs, n_points, out):
    clot = vessel.build("healthy")["clot"]
    res = []
    for r in runs:
        t0 = time.time()
        P, U = predict(model, net, norm, r, n_points)
        Ps, Us = r["p"].astype(np.float32), r["u"].astype(np.float32)
        e = {"run": r["run"], "split": r["split"], "secs": round(time.time() - t0, 1),
             "p_err": float(np.sqrt(((P - Ps) ** 2).mean()) / np.sqrt(((Ps - Ps.mean()) ** 2).mean())),
             "p_rmse_mmhg": float(np.sqrt(((P - Ps) ** 2).mean())),
             "u_err": float(np.sqrt(((U - Us) ** 2).sum(-1).mean() / (Us ** 2).sum(-1).mean())),
             "sim": summary(r, Ps, Us, clot), "ai": summary(r, P, U, clot)}
        res.append(e)
        a, b = e["sim"], e["ai"]
        print(f"{r['run']:18s} {r['split']:5s}  p err {e['p_err']:.3f} ({e['p_rmse_mmhg']:.1f} mmHg)  u err {e['u_err']:.3f}  "
              f"MCA {a['Q_mca']:6.1f} -> {b['Q_mca']:6.1f} mL/min  drop {a['drop']:5.1f} -> {b['drop']:5.1f} mmHg  "
              f"umax {a['umax']:.2f} -> {b['umax']:.2f} m/s  ({e['secs']} s)", flush=True)
    json.dump(res, open(os.path.join(out, "eval.json"), "w"), indent=1)
    return res


# ----------------------------------------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["geotransolver", "mlp"], default="geotransolver")
    ap.add_argument("--hours", type=float, default=3.0, help="training time budget")
    ap.add_argument("--points", type=int, default=16384, help="cells per sample")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--slices", type=int, default=64)
    ap.add_argument("--eval", action="store_true", help="only test the saved model")
    ap.add_argument("--tag", default=None, help="output folder name (default: the model)")
    a = ap.parse_args()
    out = os.path.join(RUNS, a.tag or a.model)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)

    train, test = load("train"), load("test")
    print(f"{len(train)} training runs, {len(test)} test runs", flush=True)
    kw = dict(layers=a.layers, hidden=a.hidden, slices=a.slices)
    if a.model == "mlp" and a.hidden == 256:
        kw["hidden"] = 512
    net, cfg = make(a.model, **kw)
    ck = os.path.join(out, "model.pt")
    if a.eval:
        state = torch.load(ck, map_location=dev)
        net.load_state_dict(state["net"])
        norm = Norm(state["norm"]["centre"], state["norm"]["scale"])
        evaluate(a.model, net, norm, test + [r for r in train if r["run"] in ("narrow70_map118", "healthy")], a.points, out)
        return

    norm = Norm().fit(train)
    sampler = Sampler(train, norm, a.points, a.batch)
    print(f"{a.model}: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters, {cfg}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    budget, t0, step, hist = a.hours * 3600, time.time(), 0, []
    log = open(os.path.join(out, "train.log"), "a")
    run_loss = None
    while True:
        el = time.time() - t0
        if el > budget:
            break
        lr = 1e-5 + 0.5 * (a.lr - 1e-5) * (1 + math.cos(math.pi * min(1.0, el / budget)))     # cosine over the budget
        if step < 500:
            lr *= (step + 1) / 500                                                           # warm-up
        for g in opt.param_groups:
            g["lr"] = lr
        X, G, Y = sampler.batch()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = forward(a.model, net, X, G)
        loss = ((pred.float() - Y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        run_loss = loss.item() if run_loss is None else 0.98 * run_loss + 0.02 * loss.item()
        if step % 200 == 0:
            msg = f"step {step:6d}  {el / 60:6.1f} min  lr {lr:.2e}  loss {run_loss:.5f}  mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
            print(msg, flush=True)
            log.write(msg + "\n"); log.flush()
            hist.append([step, el, run_loss])
        if step % 5000 == 0:
            torch.save({"net": net.state_dict(), "cfg": cfg, "model": a.model, "norm": norm.dict(), "step": step}, ck)
    torch.save({"net": net.state_dict(), "cfg": cfg, "model": a.model, "norm": norm.dict(), "step": step}, ck)
    json.dump({"history": hist, "steps": step, "minutes": (time.time() - t0) / 60, "args": vars(a), "cfg": cfg},
              open(os.path.join(out, "train.json"), "w"), indent=1)
    print(f"trained {step} steps in {(time.time() - t0) / 60:.0f} min -> {ck}", flush=True)
    evaluate(a.model, net, norm, test + [r for r in train if r["run"] in ("narrow70_map118", "healthy")], a.points, out)


if __name__ == "__main__":
    main()
