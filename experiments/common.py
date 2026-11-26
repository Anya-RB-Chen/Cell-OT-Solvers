"""Helpers shared by the experiment scripts: paths, instances, warm-up, result IO."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import instances as I  # noqa: E402
import otsolvers as O  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
DEV = O.device(os.environ.get("OT_DEVICE"))          # e.g. OT_DEVICE=cpu
torch.set_num_threads(int(os.environ.get("OT_THREADS", "4")))

# The representative instance used for E1-E3: sci-Plex3 MCF7 (CVCL_0031), vehicle vs. 10 uM of an
# HDAC inhibitor -- a strong, partly cytotoxic response.
REP = dict(line="CVCL_0031", drug="Belinostat (PXD101)", replicate="rep1", n_ctrl=1000)


def sciplex_instance(line=REP["line"], drug=REP["drug"], replicate=REP["replicate"], dose_idx=3,
                     n_ctrl=REP["n_ctrl"], seed=0, dev=DEV):
    path = next(p for p in I.sciplex_dose_paths(line) if p["drug"] == drug and p["replicate"] == replicate)
    X = I.control_cells("sciplex3", line, n=n_ctrl, seed=seed, replicate=replicate)
    Y = I.cells_of("sciplex3", line, path["group_ids"][dose_idx], seed=seed)
    return O.Problem.from_points(X, Y, normalize=None, dev=dev), path


def random_sciplex_instances(k, seed=0, n_ctrl=1000, dev=DEV):
    """k random (line, drug, replicate, top dose) instances for statistics."""
    rng = np.random.default_rng(seed)
    paths = [(line, p) for line in ("CVCL_0023", "CVCL_0004", "CVCL_0031") for p in I.sciplex_dose_paths(line)]
    out = []
    for j in rng.choice(len(paths), k, replace=False):
        line, p = paths[j]
        X = I.control_cells("sciplex3", line, n=n_ctrl, seed=int(j), replicate=p["replicate"])
        Y = I.cells_of("sciplex3", line, p["group_ids"][3], seed=int(j))
        out.append((f"{line}|{p['drug']}|{p['replicate']}", O.Problem.from_points(X, Y, normalize=None, dev=dev)))
    return out


def warmup(dev=DEV):
    """One tiny solve so CUDA context creation / kernel compilation is not timed."""
    p = O.Problem.from_points(np.random.randn(64, 5), np.random.randn(64, 5), dev=dev)
    if dev.type == "cpu":
        return
    for s in ("sinkhorn", "anderson", "lbfgs", "apdagd", "eps_scaling", "sinkhorn_newton"):
        O.SOLVERS[s](p, 0.1, tol=1e-6, max_iter=20)


def hist_record(res, **meta):
    return {**meta, **res.summary(), "hist": {k: list(map(float, v)) for k, v in res.hist.items()}}


def save_json(obj, name):
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=1, default=float))
    print("saved", path)
    return path


def load_json(name):
    return json.loads((RESULTS / name).read_text())


def to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
