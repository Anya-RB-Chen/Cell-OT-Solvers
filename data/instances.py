"""OT instances from the cached PCA embeddings (see build_embeddings.py).

Source measure = control (vehicle / DMSO) cells, target = treated cells of one condition.
Coordinates are divided by sqrt(median squared control-control distance) of the cell line, so
the cost scale is identical for every problem of a line -- a requirement for warm starts.

* sci-Plex3: the 4 doses of a drug (per replicate, per time point) are compared with the *same*
  vehicle cells of that replicate -> genuine shared-source dose paths.
* Tahoe-100M: doses of a drug sit on different plates, so we use a fixed, plate-stratified pooled
  DMSO sample per cell line as the shared source of all ~1,340 problems of that line.

Viability rho (target mass for unbalanced OT):
* Tahoe: n_cells(well) / mean n_cells of the DMSO wells on the same plate (pooled multiplexed
  lines make well counts directly comparable).
* sci-Plex3: vehicle cells are pooled per replicate, so we normalise by the median count of the
  lowest-dose (10 nM) groups of the same line/replicate/time instead.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent / "cache"


@lru_cache(maxsize=8)
def load_line(dataset: str, line: str):
    z = np.load(CACHE / dataset / f"{line}.npz")
    groups = pd.read_parquet(CACHE / dataset / f"{line}_groups.parquet").set_index("group_id")
    Z, group = z["Z"], z["group"]
    order = np.argsort(group, kind="stable")
    starts = np.searchsorted(group[order], groups.index.to_numpy())
    ends = np.searchsorted(group[order], groups.index.to_numpy(), side="right")
    groups["start"], groups["end"] = starts, ends
    # store row of every embedded cell (to look up obs annotations such as cell-cycle scores)
    zpos = np.empty(len(group), np.int64)
    zpos[order] = np.arange(len(group))
    first = np.repeat(starts, ends - starts)            # first sorted position of each cell's group
    store_row = groups.row_start.to_numpy()[np.repeat(np.arange(len(groups)), ends - starts)]
    srow = np.empty(len(group), np.int64)
    srow[order] = store_row + (np.arange(len(group)) - first)
    groups.attrs = {}
    ctrl = order[np.concatenate([np.arange(s, e) for s, e in zip(starts[groups.is_control.values], ends[groups.is_control.values])])]
    rng = np.random.default_rng(0)
    sub = rng.choice(ctrl, min(4000, len(ctrl)), replace=False)
    d2 = ((Z[sub[:2000], None, :] - Z[None, sub[2000:4000], :]) ** 2).sum(-1)
    scale = float(np.sqrt(np.median(d2)))
    if dataset == "sciplex3":
        groups["rho"] = _sciplex_rho(groups)
    info = {"scale": scale, "evr": float((z["S"] ** 2).sum() / z["total_var"]), "store_row": srow}
    return (Z / scale).astype(np.float32), order, groups, info


def _sciplex_rho(g: pd.DataFrame) -> pd.Series:
    t = g[~g.is_control]
    lo = t[t.dose == t.dose.min()].groupby(["replicate", "time_h"]).n_cells.median()
    ref = lo.reindex(pd.MultiIndex.from_arrays([g.replicate, g.time_h])).to_numpy()
    rho = g.n_cells.to_numpy() / ref
    rho[g.is_control.to_numpy()] = 1.0
    return pd.Series(rho, index=g.index)


def cells_of(dataset, line, group_ids, n=None, seed=0, return_rows=False):
    """Embedded cells of the given groups (random subset of n if given); optionally their store rows."""
    Z, order, groups, info = load_line(dataset, line)
    idx = np.concatenate([order[groups.at[gid, "start"]:groups.at[gid, "end"]] for gid in np.atleast_1d(group_ids)])
    if n is not None and len(idx) > n:
        idx = np.random.default_rng(seed).choice(idx, n, replace=False)
    return (Z[idx], info["store_row"][idx]) if return_rows else Z[idx]


def control_cells(dataset, line, n=1024, seed=0, replicate=None, time_h=24.0, return_rows=False):
    """Control sample: one replicate (sci-Plex3) or plate-stratified pooled DMSO (Tahoe)."""
    _, _, groups, _ = load_line(dataset, line)
    c = groups[groups.is_control & (groups.time_h == time_h)]
    if replicate is not None:
        c = c[c.replicate == replicate]
    if dataset == "tahoe":                      # stratify: equal share from every plate
        per = int(np.ceil(n / c.plate.nunique()))
        parts = [cells_of(dataset, line, gg.index.to_numpy(), n=per, seed=seed, return_rows=True) for _, gg in c.groupby("plate")]
        X, rows = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
        keep = np.random.default_rng(seed).permutation(len(X))[:n]
        return (X[keep], rows[keep]) if return_rows else X[keep]
    return cells_of(dataset, line, c.index.to_numpy(), n=n, seed=seed, return_rows=return_rows)


def sciplex_dose_paths(line, min_cells=30, time_h=24.0):
    """All (drug, replicate) dose paths with >= min_cells cells at every dose."""
    _, _, g, _ = load_line("sciplex3", line)
    t = g[~g.is_control & (g.time_h == time_h)]
    out = []
    for (drug, rep), d in t.groupby(["drug", "replicate"]):
        d = d.sort_values("dose")
        if len(d) == 4 and (d.n_cells >= min_cells).all():
            out.append({"drug": drug, "replicate": rep, "group_ids": d.index.tolist(),
                        "doses": d.dose.tolist(), "rho": d.rho.tolist(), "n_cells": d.n_cells.tolist()})
    return out


def tahoe_conditions(line):
    """Treated conditions of a Tahoe line: condition_id, drug, dose, group ids, rho (well mean)."""
    _, _, g, _ = load_line("tahoe", line)
    t = g[~g.is_control]
    rows = []
    for cid, d in t.groupby("condition_id"):
        rows.append({"condition_id": cid, "drug": d.drug.iat[0], "dose": float(d.dose.iat[0]),
                     "group_ids": d.index.tolist(), "rho": float(d.rho.mean()), "n_cells": int(d.n_cells.sum())})
    return pd.DataFrame(rows)
