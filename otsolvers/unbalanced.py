"""Unbalanced entropic OT (problem P3) with KL-relaxed marginals:

    min_{P >= 0} <C,P> + eps KL(P | a b^T) + tau_a KL(P1 | a) + tau_b KL(P^T 1 | b),

dual  max_{f,g} -tau_a<a, e^{-f/tau_a} - 1> - tau_b<b, e^{-g/tau_b} - 1> - eps<ab^T, e^{(f+g-C)/eps} - 1>.
``tau_b = inf`` gives the *semi-relaxed* problem with the exact constraint P^T 1 = b (dual term <g, b>):
every treated cell must be explained, while control cells may die or divide. With |b| = rho (the
measured viability) the transported mass is then exactly rho, and s_i = (P1)_i / a_i is the
survival score of control cell i (its a-weighted mean equals rho).

Exact block maximisation gives the scaling algorithm of Chizat et al. (2018):
f = lam_a * softmin_rows(g),  g = lam_b * softmin_cols(f),  lam = tau / (tau + eps)  (lam_b = 1 if tau_b = inf).

Translation mode. The entropic term only sees f + g, so moving (f + t, g - t) changes the dual
only through the marginal terms; Sinkhorn contracts this mode merely by lam_a * lam_b per sweep,
i.e. it needs ~ log(1/delta) / (1 - lam_a lam_b) ~ tau/eps * log(1/delta) iterations (Prop. 2 of
the paper). The translation-invariant variant (Séjourné, Vialard & Peyré 2022) removes the mode;
we maximise the dual over t exactly after every half-iteration. With A = <a, e^{-f/tau_a}>,
B = <b, e^{-g/tau_b}> (B = |b| if tau_b = inf), the optimal shift is

    t* = (log A - log B) / (1/tau_a + 1/tau_b).

aa_uot adds safeguarded Anderson acceleration on top (our solver, AA-TI-Sinkhorn).
"""
from __future__ import annotations

import math

import torch

from .core import Recorder, Result, log_plan, softmin_cols, softmin_rows


def _taus(tau):
    ta, tb = (tau, tau) if not isinstance(tau, (tuple, list)) else tau
    return float(ta), float(tb)


def _lam(t, eps):
    return 1.0 if math.isinf(t) else t / (t + eps)


def uot_dual(C, f, g, a, b, eps, tau) -> float:
    ta, tb = _taus(tau)
    P = torch.exp(log_plan(C, f, g, a.log(), b.log(), eps))
    term_a = -ta * (a @ torch.expm1(-f / ta)) if not math.isinf(ta) else f @ a
    term_b = -tb * (b @ torch.expm1(-g / tb)) if not math.isinf(tb) else g @ b
    return float(term_a + term_b - eps * (P.sum() - a.sum() * b.sum()))


def _translation(f, g, a, b, ta, tb):
    """Exact maximiser over t of the dual at (f + t, g - t); a 0-dim tensor (no host sync)."""
    la = torch.logsumexp(a.log() - f / ta, 0) if not math.isinf(ta) else a.sum().log()
    lb = torch.logsumexp(b.log() - g / tb, 0) if not math.isinf(tb) else b.sum().log()
    ia = 0.0 if math.isinf(ta) else 1.0 / ta
    ib = 0.0 if math.isinf(tb) else 1.0 / tb
    return (la - lb) / (ia + ib)


def _finish(res, C, f, g, a, b, eps, tau):
    P = torch.exp(log_plan(C, f, g, a.log(), b.log(), eps))
    res.f, res.g = f, g
    res.extra.update(mass=float(P.sum()), survival=(P.sum(1) / a), dual=uot_dual(C, f, g, a, b, eps, tau))


def uot_map(C, a, b, eps, tau, translate):
    """The UOT Sinkhorn fixed-point map f -> T(f) (one full sweep), returning (T(f), g)."""
    ta, tb = _taus(tau)
    la_, lb_ = _lam(ta, eps), _lam(tb, eps)
    loga, logb = a.log(), b.log()

    def T(f):
        g = lb_ * softmin_cols(C, f, loga, eps)
        if translate:
            t = _translation(f, g, a, b, ta, tb)
            f, g = f + t, g - t
        fn = la_ * softmin_rows(C, g, logb, eps)
        gn = lb_ * softmin_cols(C, fn, loga, eps)
        if translate:
            t = _translation(fn, gn, a, b, ta, tb)
            fn, gn = fn + t, gn - t
        return fn, gn

    return T


def uot_sinkhorn(prob, eps, tau, tol=1e-8, max_iter=20_000, f0=None, translate=False, name=None, **_):
    """Scaling algorithm (translate=False) or TI-Sinkhorn (translate=True). ``tau`` is a scalar or
    (tau_a, tau_b). Stops when the sup-norm fixed-point residual of f drops below ``tol``."""
    C, a, b = prob.C, prob.a, prob.b
    T = uot_map(C, a, b, eps, tau, translate)
    f = torch.zeros_like(a) if f0 is None else f0.clone().to(a)
    res = Result(name or ("TI-Sinkhorn (UOT)" if translate else "Sinkhorn (UOT)"))
    rec = Recorder(res, C.device)
    rec.start()
    it, err, g = 0, math.inf, None
    for it in range(1, max_iter + 1):
        fn, g = T(f)
        err = float((fn - f).abs().max())
        f = fn
        rec(it, err, work=2)
        if err < tol or not math.isfinite(err) or rec.timeout():
            break
    _finish(res, C, f, g, a, b, eps, tau)
    return rec.finish(it, err < tol)


def ti_sinkhorn(prob, eps, tau, **kw):
    return uot_sinkhorn(prob, eps, tau, translate=True, **kw)


def aa_uot(prob, eps, tau, tol=1e-8, max_iter=20_000, f0=None, translate=True, memory=5, reg=1e-10,
           name=None, **_):
    """AA-TI-Sinkhorn: safeguarded type-II Anderson acceleration of the translation-invariant UOT
    Sinkhorn map. An extrapolated step is rejected -- and the memory cleared -- whenever the
    fixed-point residual ||T(f) - f||_inf grows; the iterate then falls back to the plain TI image."""
    C, a, b = prob.C, prob.a, prob.b
    T = uot_map(C, a, b, eps, tau, translate)
    f = torch.zeros_like(a) if f0 is None else f0.clone().to(a)
    res = Result(name or ("AA-TI-Sinkhorn (UOT)" if translate else "AA-Sinkhorn (UOT)"))
    rec = Recorder(res, C.device)
    rec.start()
    Fs, Rs, prev_err, prev_T = [], [], math.inf, None
    it, err, n_reject = 0, math.inf, 0
    for it in range(1, max_iter + 1):
        Tf, g = T(f)
        err = float((Tf - f).abs().max())
        rec(it, err, work=2)
        if err < tol or not math.isfinite(err) or rec.timeout():
            f = Tf
            break
        if err > prev_err and prev_T is not None:
            n_reject += 1
            Fs, Rs, f, prev_err = [], [], prev_T, math.inf
            continue
        prev_err, prev_T = err, Tf
        Fs.append(Tf)
        Rs.append(Tf - f)
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
        f = Tf - dF @ torch.linalg.solve(A, dR.T @ Rs[-1])
    f, g = T(f)
    _finish(res, C, f, g, a, b, eps, tau)
    res.extra["rejected"] = n_reject
    return rec.finish(it, err < tol)
