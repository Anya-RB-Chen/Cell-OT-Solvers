"""E3 - convergence conditions and stability.

(a) eps sweep: textbook kernel-domain Sinkhorn vs. log-domain Sinkhorn -> divergence (under/overflow)
    and iteration counts; empirical linear rate vs. the Franklin-Lorenz bound;
(b) robustness over 10 random sub-samples (seeds) of the representative instance;
(c) hyper-parameter sensitivity: Anderson memory, SNS sparsity, eps-scaling decay q.
Output: results/e3_stability.json
"""
import math

import numpy as np

from common import O, save_json, sciplex_instance, warmup


def birkhoff_rate(C, eps):
    """Franklin-Lorenz: one Sinkhorn sweep contracts d_H by tanh(Delta/4)^2, with the projective
    diameter Delta = max (C_il + C_kj - C_ij - C_kl)/eps <= 2 osc(C)/eps (we use the bound)."""
    return math.tanh(float(C.max() - C.min()) / (2 * eps)) ** 2


def main():
    warmup()
    prob, path = sciplex_instance()
    sweep = []
    for eps in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        rk = O.sinkhorn_kernel(prob, eps, tol=1e-8, max_iter=50_000)
        with O.time_limit(300):
            rl = O.sinkhorn(prob, eps, tol=1e-8, max_iter=200_000)
        sweep.append({"eps": eps, "kernel_conv": rk.converged, "kernel_diverged": rk.extra["diverged"],
                      "kernel_iters": rk.n_iter, "log_conv": rl.converged, "log_iters": rl.n_iter,
                      "log_time": rl.time, "rate_emp": O.empirical_rate(rl.hist["err"], lo=1e-3, hi=1e-8),
                      "rate_bound": birkhoff_rate(prob.C, eps),
                      "kernel_underflow_frac": float((-prob.C / eps).exp().eq(0).double().mean())})
        print(sweep[-1])

    seeds = []
    for seed in range(10):
        p, _ = sciplex_instance(seed=seed)
        for name in ("sinkhorn", "anderson", "lbfgs", "sinkhorn_newton", "eps_scaling", "apdagd"):
            with O.time_limit(60):
                r = O.SOLVERS[name](p, 0.05, tol=1e-8, max_iter=100_000)
            seeds.append({"seed": seed, "solver": r.name, "iters": r.n_iter, "time": r.time,
                          "converged": r.converged, "work": r.hist["work"][-1]})
        print("seed", seed)

    sens = []
    for eps in (0.05, 0.01):
        for mem in (1, 2, 3, 5, 8, 12, 20):
            r = O.anderson(prob, eps, tol=1e-9, memory=mem, max_iter=50_000)
            sens.append({"eps": eps, "param": "anderson_memory", "value": mem, "iters": r.n_iter,
                         "time": r.time, "converged": r.converged, "rejected": r.extra["rejected"]})
        for keep in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0):
            r = O.sinkhorn_newton(prob, eps, tol=1e-9, keep=keep, max_iter=2_000)
            sens.append({"eps": eps, "param": "sns_keep", "value": keep, "iters": r.n_iter, "time": r.time,
                         "converged": r.converged, "work": r.hist["work"][-1]})
        for q in (0.9, 0.7, 0.5, 0.3, 0.1):
            r = O.eps_scaling(prob, eps, tol=1e-9, q=q, max_iter=100_000)
            sens.append({"eps": eps, "param": "epsscale_q", "value": q, "iters": r.n_iter, "time": r.time,
                         "converged": r.converged})
        print("sens", eps)
    save_json({"instance": path, "eps_sweep": sweep, "seeds": seeds, "sensitivity": sens}, "e3_stability.json")


if __name__ == "__main__":
    main()
