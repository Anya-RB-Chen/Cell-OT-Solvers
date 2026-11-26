"""Batched log-domain (unbalanced) Sinkhorn for thousands of padded OT problems on one GPU.

Problems of different sizes are padded to (n_max, m_max); padded points get log-weight -inf so
they drop out of every log-sum-exp. Each problem keeps iterating until all have converged, and we
record the first iteration at which each one met the tolerance (its own iteration count).
"""
from __future__ import annotations

import math

import torch


def pad_batch(Xs, Ys, wa=None, wb=None, dtype=torch.float32, dev="cuda"):
    """Stack point clouds into padded tensors. wa/wb: optional per-problem total masses."""
    B = len(Xs)
    n = max(len(x) for x in Xs)
    m = max(len(y) for y in Ys)
    d = Xs[0].shape[1]
    X = torch.zeros(B, n, d, dtype=dtype, device=dev)
    Y = torch.zeros(B, m, d, dtype=dtype, device=dev)
    loga = torch.full((B, n), -math.inf, dtype=dtype, device=dev)
    logb = torch.full((B, m), -math.inf, dtype=dtype, device=dev)
    for k, (x, y) in enumerate(zip(Xs, Ys)):
        X[k, :len(x)] = torch.as_tensor(x, dtype=dtype)
        Y[k, :len(y)] = torch.as_tensor(y, dtype=dtype)
        ma = 1.0 if wa is None else float(wa[k])
        mb = 1.0 if wb is None else float(wb[k])
        loga[k, :len(x)] = math.log(ma / len(x))
        logb[k, :len(y)] = math.log(mb / len(y))
    return X, Y, loga, logb


def cost(X, Y, scale=None):
    C = (X * X).sum(-1)[:, :, None] + (Y * Y).sum(-1)[:, None, :] - 2 * X @ Y.transpose(1, 2)
    C = C.clamp_min_(0)
    return C if scale is None else C / scale[:, None, None]


@torch.no_grad()
def sinkhorn_batched(C, loga, logb, eps, tau=None, tol=1e-6, max_iter=5000, f0=None, translate=True):
    """Batched balanced (tau=None) or unbalanced Sinkhorn; ``tau`` = scalar or (tau_a, tau_b),
    tau_b may be inf (semi-relaxed). With ``translate`` the exact optimal dual translation is applied
    after every half-iteration (TI-Sinkhorn). Returns f, g, per-problem iterations, final errors.

    Balanced error: L1 row-marginal violation; unbalanced: sup-norm change of f."""
    B, n, m = C.shape
    if tau is None:
        ta = tb = math.inf
    else:
        ta, tb = (tau, tau) if not isinstance(tau, (tuple, list)) else tau
    lam_a = 1.0 if math.isinf(ta) else ta / (ta + eps)
    lam_b = 1.0 if math.isinf(tb) else tb / (tb + eps)
    unbalanced = not (math.isinf(ta) and math.isinf(tb))
    translate = translate and unbalanced
    a = loga.exp()
    fin_a, fin_b = torch.isfinite(loga), torch.isfinite(logb)
    zero = lambda x, mask: torch.where(mask, x, torch.zeros_like(x))

    def shift(f, g):
        la = torch.logsumexp(loga - f / ta, 1) if not math.isinf(ta) else torch.logsumexp(loga, 1)
        lb = torch.logsumexp(logb - g / tb, 1) if not math.isinf(tb) else torch.logsumexp(logb, 1)
        t = (la - lb) / ((0.0 if math.isinf(ta) else 1 / ta) + (0.0 if math.isinf(tb) else 1 / tb))
        return zero(f + t[:, None], fin_a), zero(g - t[:, None], fin_b)

    def col(f):
        return zero(-lam_b * eps * torch.logsumexp((f[:, :, None] - C) / eps + loga[:, :, None], dim=1), fin_b)

    def row(g):
        return zero(-lam_a * eps * torch.logsumexp((g[:, None, :] - C) / eps + logb[:, None, :], dim=2), fin_a)

    f = torch.zeros(B, n, dtype=C.dtype, device=C.device) if f0 is None else zero(f0.clone(), fin_a)
    iters = torch.full((B,), max_iter, dtype=torch.long, device=C.device)
    done = torch.zeros(B, dtype=torch.bool, device=C.device)
    err = torch.full((B,), math.inf, dtype=C.dtype, device=C.device)
    for it in range(1, max_iter + 1):
        g = col(f)
        if translate:
            f, g = shift(f, g)
        f_new = row(g)
        if translate:
            f_new, g = shift(f_new, col(f_new))
        if unbalanced:
            err = (f_new - f).abs().amax(1)
        else:
            err = (a * torch.expm1((f - f_new) / eps)).abs().sum(1)
        f = f_new
        newly = (err < tol) & ~done
        iters[newly] = it
        done |= newly
        if bool(done.all()):
            break
    g = col(f)
    if translate:
        f, g = shift(f, g)
    return f, g, iters, err


def plan_batched(C, f, g, loga, logb, eps):
    return torch.exp((f[:, :, None] + g[:, None, :] - C) / eps + loga[:, :, None] + logb[:, None, :])
