"""Embed single cells of Tahoe-100M / sci-Plex3 into a per-cell-line PCA-50 space.

For every cell line we read all stored cells (log1p CP10k on the 2,000 HVG panel), fit a
PCA-50 on them, and write one ``<dataset>/<cell_line>.npz`` with

    Z        float32 (n_cells, 50)   PCA coordinates
    group    int32   (n_cells,)      group id of each cell (index into groups.parquet)
    S        float32 (50,)           singular values (for explained-variance reporting)

plus ``<dataset>/<cell_line>_groups.parquet`` (one row per group: drug, dose, plate, counts,
viability rho = n_cells / mean n_cells of the plate-matched control groups).

Usage:  python data/build_embeddings.py --dataset tahoe --workers 10 [--lines CVCL_0023 ...]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

VCDATA = os.environ.get("VCDATA", "/filesdir/AI4Science/VirtCell/data/build")
OUT = Path(__file__).resolve().parent / "cache"
N_PC = 50


def viability(g: pd.DataFrame) -> pd.Series:
    """rho_g = n_cells(g) / mean n_cells of the control groups on the same batch (plate / replicate)."""
    ctrl = g[g.is_control].groupby(["cell_line", "batch"]).n_cells.mean()
    ref = ctrl.reindex(pd.MultiIndex.from_arrays([g.cell_line, g.batch])).to_numpy()
    return pd.Series(g.n_cells.to_numpy() / ref, index=g.index)


def embed_line(dataset: str, line: str) -> str:
    sys.path.insert(0, VCDATA)
    import torch
    from vcdata.api import PerturbData

    out = OUT / dataset
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"{line}.npz").exists():
        return f"{line}: cached"
    ds = PerturbData(dataset)
    g = ds.groups
    gl = g[g.cell_line == line].copy()
    gl["rho"] = viability(g).loc[gl.index]
    gids = gl.group_id.to_numpy()
    from vcdata.api import gene_panel
    panel = [e for e in gene_panel("hvg2000") if e in set(ds.genes.ensembl_id)]   # sci-Plex3 lacks a few
    X = ds.cells(gids, genes=panel, norm="log1p")
    sizes = (gl.row_end - gl.row_start).to_numpy()
    group = np.repeat(gids, sizes).astype(np.int32)
    assert len(group) == X.shape[0]

    torch.set_num_threads(16)
    D = torch.from_numpy(X.toarray().astype(np.float32))
    mu = D.mean(0, keepdim=True)
    torch.manual_seed(0)
    U, S, V = torch.pca_lowrank(D - mu, q=N_PC, center=False, niter=4)
    Z = ((D - mu) @ V).numpy().astype(np.float32)
    total_var = float(((D - mu) ** 2).sum())
    np.savez(out / f"{line}.npz", Z=Z, group=group, S=S.numpy(), total_var=total_var)
    gl.to_parquet(out / f"{line}_groups.parquet")
    return f"{line}: {Z.shape[0]} cells, {len(gids)} groups, PCA-50 explains {float((S**2).sum())/total_var:.1%}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tahoe", choices=["tahoe", "sciplex3"])
    ap.add_argument("--lines", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, VCDATA)
    from vcdata.api import PerturbData

    lines = a.lines or sorted(PerturbData(a.dataset).groups.cell_line.unique())
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(embed_line, [a.dataset] * len(lines), lines):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
