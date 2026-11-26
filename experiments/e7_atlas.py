"""E7 - atlas scale: every treated condition of Tahoe-100M (50 cell lines, ~56k OT problems).

For each line, one plate-stratified pooled DMSO sample (n = 512) is the shared source; every
condition (<= 256 stored cells) is a target with mass rho. All problems of a line are solved in
padded batches by the batched semi-relaxed TI-Sinkhorn (eps = 0.05, tau_a = 1). Per condition we
store iterations, OT effect size <C,P>/rho, survival heterogeneity CV(s), and the Spearman
correlation of s with the cell-cycle score of the control cells.
Throughput baselines on a random subset: sequential AA-TI (GPU) and POT's CPU unbalanced solver.
Output: results/e7_atlas.parquet, results/e7_throughput.json
"""
import argparse
import math
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import DEV, I, O, RESULTS, save_json
from otsolvers import batched as BT

EPS, TAU, N_CTRL, BATCH = 0.05, (1.0, math.inf), 512, 512


def obs_cycle():
    sys.path.insert(0, "/filesdir/AI4Science/VirtCell/data/build")
    from vcdata.api import PerturbData
    obs = PerturbData("tahoe").obs
    return np.maximum(obs.g2m_score.to_numpy(), obs.s_score.to_numpy())


def run_line(line, cyc, dtype=torch.float32):
    X, xrows = I.control_cells("tahoe", line, n=N_CTRL, seed=0, return_rows=True)
    cx = cyc[xrows]
    cond = I.tahoe_conditions(line)
    cond = cond[cond.n_cells >= 20].reset_index(drop=True)
    Ys = [I.cells_of("tahoe", line, gids, n=256) for gids in cond.group_ids]
    out = []
    t_total, it_total = 0.0, 0
    for s in range(0, len(cond), BATCH):
        sl = slice(s, s + BATCH)
        Xb, Yb, la, lb = BT.pad_batch([X] * len(Ys[sl]), Ys[sl], wb=cond.rho.values[sl], dtype=dtype, dev=DEV)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        C = BT.cost(Xb, Yb)
        f, g, iters, err = BT.sinkhorn_batched(C, la, lb, EPS, tau=TAU, tol=1e-5, max_iter=3000)
        P = BT.plan_batched(C, f, g, la, lb, EPS)
        surv = (P.sum(2) / la.exp())
        cost = (C * P).sum((1, 2)) / P.sum((1, 2))
        torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        surv = surv.cpu().numpy()
        for k in range(surv.shape[0]):
            sk = surv[k]
            out.append({"iters": int(iters[k]), "err": float(err[k]), "cost": float(cost[k]),
                        "s_cv": float(sk.std() / sk.mean()), "spearman_cycle": float(spearmanr(sk, cx).statistic)})
        del C, P, Xb, Yb
    df = pd.concat([cond.drop(columns=["group_ids"]), pd.DataFrame(out)], axis=1)
    df["line"] = line
    return df, t_total


def throughput(line="CVCL_0023", k=64):
    import ot
    X = I.control_cells("tahoe", line, n=N_CTRL, seed=0)
    cond = I.tahoe_conditions(line).sample(k, random_state=0)
    Ys = [I.cells_of("tahoe", line, gids, n=256) for gids in cond.group_ids]
    res = {"k": k}
    # batched GPU (float32 and float64)
    for dt, name in ((torch.float32, "batched_f32"), (torch.float64, "batched_f64")):
        Xb, Yb, la, lb = BT.pad_batch([X] * k, Ys, wb=cond.rho.values, dtype=dt, dev=DEV)
        BT.sinkhorn_batched(BT.cost(Xb, Yb), la, lb, EPS, tau=TAU, tol=1e-5, max_iter=10)   # warm-up
        torch.cuda.synchronize(); t0 = time.perf_counter()
        BT.sinkhorn_batched(BT.cost(Xb, Yb), la, lb, EPS, tau=TAU, tol=1e-5, max_iter=3000)
        torch.cuda.synchronize(); res[name] = (time.perf_counter() - t0) / k
    # sequential GPU solvers
    for name, fn in (("seq_aa_ti", O.aa_uot), ("seq_ti", O.ti_sinkhorn), ("seq_plain", O.uot_sinkhorn)):
        t0 = time.perf_counter()
        for Y, rho in zip(Ys, cond.rho.values):
            p = O.Problem.from_points(X, Y, normalize=None, b=np.full(len(Y), rho / len(Y)), dev=DEV)
            with O.time_limit(30):
                fn(p, EPS, TAU, tol=1e-5, max_iter=100_000)
        res[name] = (time.perf_counter() - t0) / k
    # POT on CPU: semi-relaxed via reg_m = (tau_a, inf), stabilised Sinkhorn
    t0 = time.perf_counter()
    for Y, rho in zip(Ys[:16], cond.rho.values[:16]):
        a = np.full(len(X), 1 / len(X)); b = np.full(len(Y), rho / len(Y))
        M = ot.dist(X.astype(np.float64), Y.astype(np.float64))
        ot.unbalanced.sinkhorn_unbalanced(a, b, M, EPS, (1.0, np.inf), method="sinkhorn_stabilized",
                                          stopThr=1e-5, numItermax=100_000)
    res["pot_cpu"] = (time.perf_counter() - t0) / 16
    print(res)
    save_json(res, "e7_throughput.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", nargs="*")
    ap.add_argument("--skip_throughput", action="store_true")
    a = ap.parse_args()
    if not a.skip_throughput:
        throughput()
    cyc = obs_cycle()
    lines = a.lines or sorted(p.name.split(".")[0] for p in (I.CACHE / "tahoe").glob("*.npz"))
    dfs, times = [], {}
    for line in lines:
        df, t = run_line(line, cyc)
        dfs.append(df)
        times[line] = t
        print(line, len(df), f"{t:.1f}s", f"median iters {df.iters.median():.0f}", flush=True)
        pd.concat(dfs).to_parquet(RESULTS / "e7_atlas.parquet")
    save_json({"solve_time_per_line": times, "n_problems": int(sum(len(d) for d in dfs)),
               "total_time": float(sum(times.values()))}, "e7_timing.json")


if __name__ == "__main__":
    main()
