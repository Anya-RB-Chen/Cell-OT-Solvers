"""E4 - computational complexity: wall time and memory vs. problem size n (= m).

Source: plate-stratified pooled DMSO cells of Tahoe A549 (CVCL_0023); target: pooled cells treated
at the top dose with any drug. n in {256, ..., 16384}. Entropic solvers at eps = 0.05 to 1e-6
(GPU, float64); LP solvers to optimality (CPU) until they exceed the time budget.
Output: results/e4_scaling.json
"""
import numpy as np
import torch

from common import DEV, I, O, save_json, warmup

LINE = "CVCL_0023"
SIZES = (256, 512, 1024, 2048, 4096, 8192, 16384)
ENTROPIC = ("sinkhorn", "anderson", "lbfgs", "sinkhorn_newton", "eps_scaling", "apdagd")
BUDGET = 600


def instance(n, seed=0):
    X = I.control_cells("tahoe", LINE, n=n, seed=seed)
    _, _, g, _ = I.load_line("tahoe", LINE)
    top = g[(~g.is_control) & (g.dose == g.dose.max())].index.to_numpy()
    Y = I.cells_of("tahoe", LINE, top, n=n, seed=seed + 1)
    return O.Problem.from_points(X, Y, normalize=None, dev=DEV)


def main():
    warmup()
    rows, dead = [], set()
    for n in SIZES:
        prob = instance(n)
        for name in ENTROPIC:
            if name in dead:
                continue
            torch.cuda.reset_peak_memory_stats()
            with O.time_limit(BUDGET):
                r = O.SOLVERS[name](prob, 0.05, tol=1e-6, max_iter=1_000_000)
            rows.append({"n": n, "solver": r.name, "time": r.time, "iters": r.n_iter, "converged": r.converged,
                         "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "family": "entropic"})
            print(rows[-1], flush=True)
            if not r.converged:
                dead.add(name)
        for name, fn in (("emd", O.emd), ("highs_ipm", O.highs_ipm), ("pdhg", O.pdhg)):
            if name in dead or (name == "highs_ipm" and n > 4096):
                continue
            with O.time_limit(BUDGET):
                r = fn(prob, tol=1e-6) if name == "pdhg" else fn(prob, time_limit=BUDGET) if name == "highs_ipm" else fn(prob)
            rows.append({"n": n, "solver": r.name, "time": r.time, "iters": r.n_iter, "converged": r.converged,
                         "cost": r.extra.get("cost"), "family": "lp"})
            print(rows[-1], flush=True)
            if r.time > BUDGET or not r.converged:
                dead.add(name)
        save_json(rows, "e4_scaling.json")


if __name__ == "__main__":
    main()
