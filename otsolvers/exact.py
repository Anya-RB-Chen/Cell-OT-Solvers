"""Solvers for the exact Kantorovich LP (problem P1):

    min <C, P>  s.t.  P1 = a,  P^T 1 = b,  P >= 0.

| solver      | family                                   | where it runs |
|-------------|------------------------------------------|---------------|
| emd         | network simplex (POT, Bonneel et al.)    | CPU           |
| highs_ipm   | primal-dual interior point (HiGHS)       | CPU           |
| pdhg        | restarted primal-dual hybrid gradient    | GPU           |
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.optimize import linprog

from .core import Recorder, Result


def emd(prob, name="Network simplex (EMD)", **_):
    import ot
    C = prob.C.cpu().numpy()
    a, b = prob.a.cpu().numpy(), prob.b.cpu().numpy()
    b = b * (a.sum() / b.sum())
    t0 = time.perf_counter()
    P, log = ot.emd(a, b, C, numItermax=50_000_000, log=True)
    res = Result(name, P=torch.as_tensor(P))
    res.time = time.perf_counter() - t0
    res.converged = log["warning"] is None
    res.extra.update(cost=float(log["cost"]), f=log["u"], g=log["v"])
    res.f, res.g = torch.as_tensor(log["u"]), torch.as_tensor(log["v"])
    return res


def highs_ipm(prob, name="Interior point (HiGHS)", time_limit=600.0, **_):
    C = prob.C.cpu().numpy()
    a, b = prob.a.cpu().numpy(), prob.b.cpu().numpy()
    n, m = C.shape
    # A_eq vec(P) with row-major vec: rows of P sum to a, columns sum to b (drop one redundant row)
    rows = sp.kron(sp.eye(n), np.ones((1, m))).tocsr()
    cols = sp.kron(np.ones((1, n)), sp.eye(m)).tocsr()
    A = sp.vstack([rows, cols[:-1]]).tocsr()
    rhs = np.concatenate([a, b[:-1]])
    t0 = time.perf_counter()
    out = linprog(C.ravel(), A_eq=A, b_eq=rhs, bounds=(0, None), method="highs-ipm",
                  options={"time_limit": time_limit, "presolve": False})
    res = Result(name)
    res.time = time.perf_counter() - t0
    res.converged = out.status == 0
    res.n_iter = int(getattr(out, "nit", 0) or 0)
    if out.x is not None:
        res.P = torch.as_tensor(out.x.reshape(n, m))
        res.extra["cost"] = float(out.fun)
    return res


def pdhg(prob, tol=1e-6, max_iter=200_000, check_every=64, beta_suff=0.2, beta_nec=0.8,
         name="Restarted PDHG", **_):
    """PDLP-style restarted PDHG (Applegate et al. 2021; Lu & Yang 2024 for OT).

    Primal step  P+ = max(0, P - tau (C - f (+) g)),  dual step on the extrapolated P.
    Every ``check_every`` iterations the KKT error of the current and of the averaged iterate
    is evaluated; we restart to the better one when it decreased sufficiently (adaptive restart),
    and rebalance the primal weight omega = ||dual move|| / ||primal move||.
    """
    C, a, b = prob.C, prob.a, prob.b
    n, m = C.shape
    eta = 0.95 / math.sqrt(n + m)          # ||A||_2 = sqrt(n + m) for the OT constraint matrix
    omega = float(C.norm() / torch.cat([a, b]).norm())   # PDLP's initial primal weight ||c|| / ||q||

    P = a[:, None] * b[None, :]
    f = torch.zeros_like(a)
    g = torch.zeros_like(b)

    def kkt(P, f, g):
        pr = torch.cat([P.sum(1) - a, P.sum(0) - b]).norm()
        red = C - f[:, None] - g[None, :]
        du = torch.clamp(-red, min=0).norm()
        gap = (C * P).sum() - f @ a - g @ b
        rel = 1.0 + float(a.norm() + b.norm())
        return math.sqrt(float(pr) ** 2 + float(du) ** 2 + float(gap) ** 2) / rel

    res = Result(name)
    res.hist["kkt"] = []
    rec = Recorder(res, C.device)
    rec.start()
    Pa, fa, ga, wsum = torch.zeros_like(P), torch.zeros_like(f), torch.zeros_like(g), 0.0
    P0, f0, g0 = P.clone(), f.clone(), g.clone()
    kkt_restart = kkt(P, f, g)
    kkt_prev_cand = math.inf
    it, k_err = 0, math.inf
    for it in range(1, max_iter + 1):
        tau, sigma = eta / omega, eta * omega
        Pn = torch.clamp(P - tau * (C - f[:, None] - g[None, :]), min=0)
        Pb = 2 * Pn - P
        f = f + sigma * (a - Pb.sum(1))
        g = g + sigma * (b - Pb.sum(0))
        P = Pn
        Pa += P; fa += f; ga += g; wsum += 1
        if it % check_every == 0:
            k_cur = kkt(P, f, g)
            k_avg = kkt(Pa / wsum, fa / wsum, ga / wsum)
            cand = (P, f, g, k_cur) if k_cur <= k_avg else (Pa / wsum, fa / wsum, ga / wsum, k_avg)
            k_err = cand[3]
            err = float((cand[0].sum(1) - a).abs().sum() + (cand[0].sum(0) - b).abs().sum())
            res.hist["kkt"].append(k_err)
            rec(it, err, work=2 * check_every + 4)
            if k_err < tol or rec.timeout():
                P, f, g = cand[0].clone(), cand[1].clone(), cand[2].clone()
                break
            if k_err <= beta_suff * kkt_restart or (k_err <= beta_nec * kkt_restart and k_err > kkt_prev_cand):
                P, f, g = cand[0].clone(), cand[1].clone(), cand[2].clone()
                dx = float((P - P0).norm())
                dy = float(torch.cat([f - f0, g - g0]).norm())
                if dx > 1e-14 and dy > 1e-14:
                    omega = math.exp(0.5 * math.log(dy / dx) + 0.5 * math.log(omega))
                P0, f0, g0 = P.clone(), f.clone(), g.clone()
                Pa.zero_(); fa.zero_(); ga.zero_(); wsum = 0.0
                kkt_restart, kkt_prev_cand = k_err, math.inf
            else:
                kkt_prev_cand = k_err
    res.P, res.f, res.g = P, f, g
    res.extra["cost"] = float((C * P).sum())
    return rec.finish(it, k_err < tol)
