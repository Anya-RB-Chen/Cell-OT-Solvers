"""Dose-path warm start (DP-WS), the main contribution of the paper.

In a perturbation atlas, all doses of one drug on one plate are compared against the *same*
control cells. Their OT problems share the source measure (x_i, a_i) and differ only in the
target. The source potential f lives on those shared control cells, so the optimal f of dose k is
a valid -- and, when neighbouring doses are similar, very good -- initial point for dose k+1.
The target potential cannot be transferred (different cells) and is rebuilt for free by the soft
c-transform, which is the first half-iteration of Sinkhorn anyway.

Theory (Prop. 1 in the paper). Sinkhorn is a contraction in Hilbert's projective metric with a
linear rate lam < 1, and for scalings u = e^{f/eps},  d_H(u, u*) = osc(f - f*) / eps,
osc(h) = max h - min h. Hence the iterations to reach accuracy delta satisfy
K(f0) <= log(osc(f0 - f*) / (eps delta)) / log(1/lam), and DP-WS saves

    K_cold - K_warm  ~  log( osc(f*_{k+1}) / osc(f*_k - f*_{k+1}) ) / log(1/lam)  iterations.
"""
from __future__ import annotations

import torch


def osc(h: torch.Tensor) -> float:
    return float(h.max() - h.min())


def solve_path(problems, solver, warm=True, f_init=None, **kw):
    """Solve a list of problems that share their source, carrying f along the path."""
    out, f0 = [], f_init
    for prob in problems:
        res = solver(prob, f0=f0 if warm else None, **kw)
        out.append(res)
        f0 = res.f
    return out


def predicted_saving(f_prev, f_star, rate, center=True):
    """Iterations saved by starting at f_prev instead of 0 (Prop. 1), given the empirical rate."""
    cold, warm = osc(f_star), osc(f_star - f_prev)
    if warm <= 0 or rate <= 0 or rate >= 1:
        return float("nan")
    return (torch.log(torch.tensor(cold / warm)) / torch.log(torch.tensor(1.0 / rate))).item()


def empirical_rate(errs, lo=1e-2, hi=1e-7):
    """Linear rate lam fitted to the tail of an error curve between ``lo`` and ``hi``."""
    e = torch.tensor([x for x in errs if hi < x < lo], dtype=torch.float64)
    if len(e) < 3:
        return float("nan")
    k = torch.arange(len(e), dtype=torch.float64)
    slope = torch.linalg.lstsq(torch.stack([k, torch.ones_like(k)], 1), e.log()[:, None]).solution[0, 0]
    return float(torch.exp(slope))
