"""Solvers for entropic OT (problem P2). All share the signature

    solver(prob, eps, tol=1e-8, max_iter=..., f0=None, **kw) -> Result

and stop when the L1 marginal violation ||P1 - a||_1 + ||P^T 1 - b||_1 <= tol.

| solver              | view of the dual                                   | per-iteration work |
|---------------------|----------------------------------------------------|--------------------|
| sinkhorn            | exact block-coordinate ascent (Bregman projection) | 2 kernel passes    |
| sinkhorn_kernel     | same, in the scaling domain u = e^{f/eps}          | 2 passes, unstable |
| greenkhorn          | greedy single-coordinate ascent (Gauss-Southwell)  | 1 row / column     |
| eps_scaling         | continuation in eps with warm starts               | 2 passes           |
| anderson            | Anderson-type-II extrapolation of Sinkhorn          | 2 passes + tiny LS |
| apdagd              | accelerated gradient on the log-sum-exp dual       | ~3 passes          |
| lbfgs               | quasi-Newton on the semi-dual F(f)                 | ~2 passes / eval   |
| sinkhorn_newton     | Newton-CG on a sparsified Hessian (SNS)            | 1 pass + CG + LS   |
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import logsumexp

from .core import (Recorder, Result, dual_objective, log_plan, marginal_error, softmin_cols,
                   softmin_rows)


def _init(prob, f0):
    C, a, b = prob.C, prob.a, prob.b
    f = torch.zeros_like(a) if f0 is None else f0.clone().to(a)
    return C, a, b, a.log(), b.log(), f


def _row_err(a, f, f_new, eps):
    """Row-marginal L1 error of (f, g(f)): r_i = a_i exp((f_i - T(f)_i)/eps); columns are exact."""
    return float((a * torch.expm1((f - f_new) / eps)).abs().sum())


# ── 1. Sinkhorn (log domain) ───────────────────────────────────────────────────────────────
def sinkhorn(prob, eps, tol=1e-8, max_iter=10_000, f0=None, name="Sinkhorn", **_):
    C, a, b, loga, logb, f = _init(prob, f0)
    res = Result(name)
    rec = Recorder(res, C.device)
    rec.start()
    it, err = 0, math.inf
    for it in range(1, max_iter + 1):
        g = softmin_cols(C, f, loga, eps)
        f_new = softmin_rows(C, g, logb, eps)
        err = _row_err(a, f, f_new, eps)
        f = f_new
        rec(it, err, work=2)
        if err < tol or not math.isfinite(err) or rec.timeout():
            break
    res.f, res.g = f, softmin_cols(C, f, loga, eps)
    return rec.finish(it, err < tol)


# ── 2. Sinkhorn in the scaling domain (the textbook u/v iteration; underflows at small eps) ──
def sinkhorn_kernel(prob, eps, tol=1e-8, max_iter=10_000, f0=None, name="Sinkhorn (kernel)", **_):
    C, a, b = prob.C, prob.a, prob.b
    K = torch.exp(-C / eps)
    u = a.clone() if f0 is None else a * torch.exp(f0 / eps)
    res = Result(name)
    rec = Recorder(res, C.device)
    rec.start()
    it, err = 0, math.inf
    with np.errstate(all="ignore"):
        for it in range(1, max_iter + 1):
            v = b / (K.T @ u)
            Kv = K @ v
            err = float((u * Kv - a).abs().sum())
            u = a / Kv
            rec(it, err, work=2)
            if err < tol or not math.isfinite(err) or rec.timeout():
                break
    res.f, res.g = eps * (u / a).log(), eps * (v / b).log()     # P = diag(u) K diag(v) = ab^T e^{(f+g-C)/eps}
    res.extra["diverged"] = not math.isfinite(err)
    return rec.finish(it, err < tol)


# ── 3. Greenkhorn (Altschuler, Weed & Rigollet 2017) ────────────────────────────────────────
def greenkhorn(prob, eps, tol=1e-8, max_iter=200, f0=None, name="Greenkhorn", **_):
    """Greedy coordinate ascent: each step rescales the single row or column whose marginal
    deviates most, measured by rho(x, y) = y - x + x log(x/y). ``max_iter`` counts sweeps of
    n + m single-coordinate steps, so it is comparable with Sinkhorn iterations. Runs on CPU."""
    C = prob.C.double().cpu().numpy()
    a = prob.a.cpu().numpy()
    b = prob.b.cpu().numpy()
    n, m = C.shape
    f = np.zeros(n) if f0 is None else f0.cpu().numpy().copy()
    g = -eps * logsumexp((f[:, None] - C) / eps, b=a[:, None], axis=0)   # start from the exact g(f)
    P = a[:, None] * b[None, :] * np.exp((f[:, None] + g[None, :] - C) / eps)
    r, c = P.sum(1), P.sum(0)

    def rho(x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y > 0, y - x + x * np.log(x / y), np.inf)

    res = Result(name)
    rec = Recorder(res, "cpu")
    rec.start()
    err = np.abs(r - a).sum() + np.abs(c - b).sum()
    sweep, steps_per_sweep, work = 0, n + m, 0.0
    for sweep in range(1, max_iter + 1):
        dr, dc = rho(a, r), rho(b, c)
        for _ in range(steps_per_sweep):
            i, j = int(dr.argmax()), int(dc.argmax())
            if dr[i] >= dc[j]:
                s = a[i] / r[i]
                old = P[i].copy()
                P[i] *= s
                f[i] += eps * math.log(s)
                c += P[i] - old
                r[i] = a[i]
                dr[i] = 0.0
                dc = rho(b, c)
                work += 1.0 / n
            else:
                s = b[j] / c[j]
                old = P[:, j].copy()
                P[:, j] *= s
                g[j] += eps * math.log(s)
                r += P[:, j] - old
                c[j] = b[j]
                dc[j] = 0.0
                dr = rho(a, r)
                work += 1.0 / m
        err = np.abs(r - a).sum() + np.abs(c - b).sum()
        rec(sweep, err, work=work)
        work = 0.0
        if err < tol or not np.isfinite(err) or rec.timeout():
            break
    res.f, res.g = torch.as_tensor(f), torch.as_tensor(g)
    return rec.finish(sweep, err < tol)


# ── 4. eps-scaling / annealing (continuation in the regularisation) ─────────────────────────
def eps_scaling(prob, eps, tol=1e-8, max_iter=10_000, f0=None, eps0=1.0, q=0.5, inner=10,
                name="ε-scaling", **_):
    """Run ``inner`` Sinkhorn iterations at each eps_t = max(eps, eps0 q^t), warm-starting the
    potentials, then iterate at the target eps until the tolerance is met."""
    C, a, b, loga, logb, f = _init(prob, f0)
    res = Result(name)
    rec = Recorder(res, C.device)
    rec.start()
    it, err, e = 0, math.inf, max(eps0, eps)
    while it < max_iter:
        it += 1
        g = softmin_cols(C, f, loga, e)
        f_new = softmin_rows(C, g, logb, e)
        err_e = _row_err(a, f, f_new, e)
        f = f_new
        if e > eps:
            err = marginal_error(C, f, softmin_cols(C, f, loga, eps), a, b, eps)
            rec(it, err, work=3)
            if it % inner == 0:
                e = max(eps, e * q)
        else:
            err = err_e
            rec(it, err, work=2)
            if err < tol or not math.isfinite(err) or rec.timeout():
                break
    res.f, res.g = f, softmin_cols(C, f, loga, eps)
    return rec.finish(it, err < tol)


# ── 5. Anderson-accelerated Sinkhorn ────────────────────────────────────────────────────────
def anderson(prob, eps, tol=1e-8, max_iter=10_000, f0=None, memory=5, reg=1e-10,
             name="Anderson-Sinkhorn", **_):
    """Type-II Anderson acceleration of the fixed-point map T(f) = softmin_rows(softmin_cols(f)).
    Safeguard: if an extrapolated iterate increases the marginal error, fall back to the plain
    Sinkhorn image of the previous iterate and clear the memory."""
    C, a, b, loga, logb, f = _init(prob, f0)

    def T(x):
        return softmin_rows(C, softmin_cols(C, x, loga, eps), logb, eps)

    res = Result(name)
    rec = Recorder(res, C.device)
    rec.start()
    Fs, Rs = [], []                       # T(f_k) and residuals r_k = T(f_k) - f_k
    prev_err, prev_Tf = math.inf, None
    it, err, n_reject = 0, math.inf, 0
    for it in range(1, max_iter + 1):
        Tf = T(f)
        err = _row_err(a, f, Tf, eps)
        rec(it, err, work=2)
        if err < tol or not math.isfinite(err) or rec.timeout():
            break
        if err > prev_err and prev_Tf is not None:    # safeguard
            n_reject += 1
            Fs, Rs = [], []
            f, prev_err = prev_Tf, math.inf
            continue
        prev_err, prev_Tf = err, Tf
        r = Tf - f
        Fs.append(Tf)
        Rs.append(r)
        if len(Rs) > memory + 1:
            Fs.pop(0)
            Rs.pop(0)
        if len(Rs) == 1:
            f = Tf
            continue
        dR = torch.stack([Rs[k + 1] - Rs[k] for k in range(len(Rs) - 1)], 1)
        dF = torch.stack([Fs[k + 1] - Fs[k] for k in range(len(Fs) - 1)], 1)
        A = dR.T @ dR
        A += reg * torch.trace(A) * torch.eye(A.shape[0], dtype=A.dtype, device=A.device) + 1e-300
        gamma = torch.linalg.solve(A, dR.T @ r)
        f = Tf - dF @ gamma
    res.f, res.g = f, softmin_cols(C, f, loga, eps)
    res.extra["rejected"] = n_reject
    return rec.finish(it, err < tol)


# ── 6. APDAGD (Dvurechensky, Gasnikov & Kroshnin 2018) ──────────────────────────────────────
def apdagd(prob, eps, tol=1e-8, max_iter=10_000, f0=None, name="APDAGD", **_):
    """Adaptive accelerated gradient descent on phi(l) = eps logsumexp((y+z-C)/eps) - <y,a> - <z,b>,
    the dual of min <C,X> + eps sum X log X over X in the simplex with X1=a, X^T1=b. The primal
    iterate is the alpha-weighted average of X(lambda_k) (needed for the O(1/k^2) guarantee)."""
    C, a, b = prob.C, prob.a, prob.b
    n, m = C.shape

    def phi_grad(y, z, need_grad=True):
        S = (y[:, None] + z[None, :] - C) / eps
        lse = torch.logsumexp(S.reshape(-1), 0)
        val = eps * lse - y @ a - z @ b
        if not need_grad:
            return val, None, None, None
        X = torch.exp(S - lse)
        return val, X.sum(1) - a, X.sum(0) - b, X

    res = Result(name)
    rec = Recorder(res, C.device)
    rec.start()
    zeros = lambda k: torch.zeros(k, dtype=C.dtype, device=C.device)
    y_eta, z_eta = (zeros(n) if f0 is None else f0.clone()), zeros(m)
    y_zeta, z_zeta = y_eta.clone(), z_eta.clone()
    beta, M = 0.0, 1.0 / eps
    Xhat = torch.zeros_like(C)
    it, err = 0, math.inf
    for it in range(1, max_iter + 1):
        M = M / 2
        work = 0.0
        while True:
            M *= 2
            alpha = (1 + math.sqrt(1 + 4 * M * beta)) / (2 * M)
            beta_new = beta + alpha
            tau = alpha / beta_new
            y_l, z_l = tau * y_zeta + (1 - tau) * y_eta, tau * z_zeta + (1 - tau) * z_eta
            val_l, gy, gz, X_l = phi_grad(y_l, z_l)
            y_z2, z_z2 = y_zeta - alpha * gy, z_zeta - alpha * gz
            y_e2, z_e2 = tau * y_z2 + (1 - tau) * y_eta, tau * z_z2 + (1 - tau) * z_eta
            val_e, *_ = phi_grad(y_e2, z_e2, need_grad=False)
            work += 3
            dy, dz = y_e2 - y_l, z_e2 - z_l
            if val_e <= val_l + gy @ dy + gz @ dz + M / 2 * (dy @ dy + dz @ dz) + 1e-14 * abs(val_l):
                break
        Xhat = (alpha * X_l + beta * Xhat) / beta_new
        beta = beta_new
        y_zeta, z_zeta, y_eta, z_eta = y_z2, z_z2, y_e2, z_e2
        err = float((Xhat.sum(1) - a).abs().sum() + (Xhat.sum(0) - b).abs().sum())
        rec(it, err, work=work)
        if err < tol or not math.isfinite(err) or rec.timeout():
            break
    res.f, res.g, res.P = y_eta, z_eta, Xhat
    return rec.finish(it, err < tol)


# ── 7. L-BFGS on the semi-dual ──────────────────────────────────────────────────────────────
def lbfgs(prob, eps, tol=1e-8, max_iter=2_000, f0=None, history=20, name="L-BFGS (semi-dual)", **_):
    """Maximise the concave semi-dual F(f) = <f,a> + <g(f), b>, g(f) = soft c-transform, whose
    gradient is a - P(f)1 (the column marginal is exact by construction)."""
    C, a, b, loga, logb, f = _init(prob, f0)
    x = f.clone()
    res = Result(name)
    rec = Recorder(res, C.device)

    def closure_vals(x):
        g = softmin_cols(C, x, loga, eps)
        lp = log_plan(C, x, g, loga, logb, eps)
        r = torch.exp(lp).sum(1)
        F = x @ a + g @ b
        return -F, -(a - r), r

    # two-loop L-BFGS with Armijo backtracking on exact gradients
    S, Y = [], []
    val, grad, r = closure_vals(x)
    rec.start()
    it, err = 0, float((r - a).abs().sum())
    for it in range(1, max_iter + 1):
        q = grad.clone()
        alphas = []
        for s, y in reversed(list(zip(S, Y))):
            al = (s @ q) / (y @ s)
            alphas.append(al)
            q -= al * y
        if S:
            q *= (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1])
        else:                             # diagonal of the inverse Hessian at the solution: eps / a_i
            q *= eps / a
        for (s, y), al in zip(zip(S, Y), reversed(alphas)):
            be = (y @ q) / (y @ s)
            q += (al - be) * s
        d = -q
        t, work = 1.0, 0.0
        gd = float(grad @ d)
        if gd >= 0:                       # not a descent direction: reset memory
            d = -grad * eps / a
            S, Y, gd = [], [], float(grad @ d)
        while True:
            xn = x + t * d
            vn, gn, rn = closure_vals(xn)
            work += 2
            if vn <= val + 1e-4 * t * gd or t < 1e-12:
                break
            t *= 0.5
        s, y = xn - x, gn - grad
        if float(s @ y) > 1e-300:
            S.append(s)
            Y.append(y)
            if len(S) > history:
                S.pop(0)
                Y.pop(0)
        x, val, grad, r = xn, vn, gn, rn
        err = float((r - a).abs().sum())
        rec(it, err, work=work)
        if err < tol or not math.isfinite(err) or rec.timeout():
            break
    res.f, res.g = x, softmin_cols(C, x, loga, eps)
    return rec.finish(it, err < tol)


# ── 8. Sinkhorn-Newton-Sparse (Tang et al., ICLR 2024) ──────────────────────────────────────
def sinkhorn_newton(prob, eps, tol=1e-8, max_iter=500, f0=None, warm_iters=20, keep=0.05,
                    cg_iters=50, name="Sinkhorn-Newton-Sparse", **_):
    """``warm_iters`` Sinkhorn iterations, then Newton steps on the dual (f, g). The Hessian
    -(1/eps)[[diag(r), P], [P^T, diag(c)]] is sparsified by keeping the largest ``keep`` fraction
    of each row of P; the Newton system is solved by CG and globalised by Armijo backtracking."""
    C, a, b, loga, logb, f = _init(prob, f0)
    n, m = C.shape
    k = max(1, int(round(keep * m)))
    res = Result(name)
    rec = Recorder(res, C.device)
    rec.start()
    g = softmin_cols(C, f, loga, eps)
    it, err = 0, math.inf
    for it in range(1, warm_iters + 1):
        f_new = softmin_rows(C, g, logb, eps)
        err = _row_err(a, f, f_new, eps)
        f = f_new
        g = softmin_cols(C, f, loga, eps)
        rec(it, err, work=2)
        if err < tol:
            break

    def dual(f, g):
        return float(f @ a + g @ b) - eps * float(torch.exp(log_plan(C, f, g, loga, logb, eps)).sum() - 1.0)

    D = dual(f, g)
    while err >= tol and it < max_iter and not rec.timeout():
        it += 1
        P = torch.exp(log_plan(C, f, g, loga, logb, eps))
        r, c = P.sum(1), P.sum(0)
        gf, gg = a - r, b - c                       # ascent direction of the dual
        vals, idx = torch.topk(P, k, dim=1)         # sparse Hessian pattern

        def Hv(x, y):                               # (-H) [x; y] * eps, regularised for the gauge
            Px = torch.zeros(m, dtype=C.dtype, device=C.device).scatter_add_(0, idx.reshape(-1), (vals * x[:, None]).reshape(-1))
            Py = (vals * y[idx]).sum(1)
            return r * x + Py + 1e-10 * x, c * y + Px + 1e-10 * y

        # CG on (eps^{-1}) * system: solve (-H) d = grad  ->  Hv(d) = eps * grad
        bx, by = eps * gf, eps * gg
        dx, dy = torch.zeros_like(bx), torch.zeros_like(by)
        rx, ry = bx.clone(), by.clone()
        px, py = rx.clone(), ry.clone()
        rs = float(rx @ rx + ry @ ry)
        rs0 = rs
        for _ in range(cg_iters):
            qx, qy = Hv(px, py)
            alpha = rs / float(px @ qx + py @ qy)
            dx += alpha * px
            dy += alpha * py
            rx -= alpha * qx
            ry -= alpha * qy
            rs_new = float(rx @ rx + ry @ ry)
            if rs_new < 1e-20 * rs0:
                break
            px, py = rx + rs_new / rs * px, ry + rs_new / rs * py
            rs = rs_new
        slope = float(gf @ dx + gg @ dy)
        t, work = 1.0, 1.0
        if slope <= 0:                              # fall back to a Sinkhorn step
            f = softmin_rows(C, g, logb, eps)
            g = softmin_cols(C, f, loga, eps)
            work += 2
        else:
            while True:
                fn, gn = f + t * dx, g + t * dy
                Dn = dual(fn, gn)
                work += 1
                if Dn >= D + 1e-4 * t * slope or t < 1e-8:
                    break
                t *= 0.5
            f, g = fn, gn
        D = dual(f, g)
        err = marginal_error(C, f, g, a, b, eps)
        rec(it, err, work=work + 1)
    res.f, res.g = f, g
    return rec.finish(it, err < tol)


SOLVERS = {
    "sinkhorn": sinkhorn,
    "sinkhorn_kernel": sinkhorn_kernel,
    "greenkhorn": greenkhorn,
    "eps_scaling": eps_scaling,
    "anderson": anderson,
    "apdagd": apdagd,
    "lbfgs": lbfgs,
    "sinkhorn_newton": sinkhorn_newton,
}
