"""Shared primitives: problem data, log-domain kernels, objectives, and the result container.

Conventions (used by every solver in this package)
-------------------------------------------------
Source weights a (n,), target weights b (m,), cost C (n, m). Entropic OT is

    min_{P >= 0, P1 = a, P^T 1 = b}  <C, P> + eps * KL(P | a b^T)

whose dual is the smooth concave program

    max_{f, g}  <f, a> + <g, b> - eps * sum_ij a_i b_j (exp((f_i + g_j - C_ij) / eps) - 1),

with primal-dual link  P_ij = a_i b_j exp((f_i + g_j - C_ij) / eps).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

DTYPE = torch.float64
TIME_LIMIT = [float("inf")]   # wall-clock budget per solve (seconds); set with ``time_limit``


class time_limit:
    """Context manager bounding every solve inside it, e.g. ``with time_limit(60): ...``."""

    def __init__(self, seconds):
        self.seconds, self.old = seconds, None

    def __enter__(self):
        self.old, TIME_LIMIT[0] = TIME_LIMIT[0], self.seconds

    def __exit__(self, *exc):
        TIME_LIMIT[0] = self.old


def device(name: str | None = None) -> torch.device:
    if name is not None:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def as_tensor(x, dev=None, dtype=DTYPE) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=dev or x.device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev or device())


def sqeuclidean(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean cost ||x_i - y_j||^2 (clamped at 0 against round-off)."""
    C = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return C.clamp_min_(0.0)


@dataclass
class Problem:
    """One OT instance. ``mass`` lets the target carry total mass != 1 (unbalanced OT)."""
    C: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor

    @classmethod
    def from_points(cls, X, Y, a=None, b=None, normalize="median", dev=None):
        dev = dev or device()
        X, Y = as_tensor(X, dev), as_tensor(Y, dev)
        C = sqeuclidean(X, Y)
        if normalize == "median":
            C = C / C.median()
        elif normalize == "max":
            C = C / C.max()
        n, m = C.shape
        a = torch.full((n,), 1.0 / n, dtype=DTYPE, device=dev) if a is None else as_tensor(a, dev)
        b = torch.full((m,), 1.0 / m, dtype=DTYPE, device=dev) if b is None else as_tensor(b, dev)
        return cls(C, a, b)

    @property
    def shape(self):
        return tuple(self.C.shape)


# ── log-domain building blocks ─────────────────────────────────────────────────────────────
def softmin_rows(C, g, logb, eps):
    """f_i = -eps log sum_j b_j exp((g_j - C_ij)/eps): the exact maximiser of the dual in f."""
    return -eps * torch.logsumexp((g[None, :] - C) / eps + logb[None, :], dim=1)


def softmin_cols(C, f, loga, eps):
    return -eps * torch.logsumexp((f[:, None] - C) / eps + loga[:, None], dim=0)


def log_plan(C, f, g, loga, logb, eps):
    return (f[:, None] + g[None, :] - C) / eps + loga[:, None] + logb[None, :]


def plan(C, f, g, a, b, eps):
    return torch.exp(log_plan(C, f, g, a.log(), b.log(), eps))


def marginals(C, f, g, a, b, eps):
    P = plan(C, f, g, a, b, eps)
    return P.sum(1), P.sum(0)


def marginal_error(C, f, g, a, b, eps) -> float:
    r, c = marginals(C, f, g, a, b, eps)
    return float((r - a).abs().sum() + (c - b).abs().sum())


def dual_objective(C, f, g, a, b, eps) -> float:
    P = plan(C, f, g, a, b, eps)
    return float(f @ a + g @ b - eps * (P.sum() - a.sum() * b.sum()))


def primal_objective(C, P, a, b, eps) -> float:
    """<C,P> + eps KL(P | ab^T) with KL(p|q) = sum p log(p/q) - p + q."""
    Q = a[:, None] * b[None, :]
    mask = P > 0
    kl = (P[mask] * (P[mask].log() - Q[mask].log())).sum() - P.sum() + Q.sum()
    return float((C * P).sum() + eps * kl)


# ── results ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Result:
    name: str
    f: torch.Tensor | None = None
    g: torch.Tensor | None = None
    P: torch.Tensor | None = None            # set by primal solvers (exact LP, APDAGD averages, PDHG)
    hist: dict = field(default_factory=lambda: {"it": [], "time": [], "err": [], "work": []})
    converged: bool = False
    n_iter: int = 0
    time: float = 0.0
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"solver": self.name, "converged": self.converged, "n_iter": self.n_iter,
                "time": self.time, "err": self.hist["err"][-1] if self.hist["err"] else float("nan")}


class Recorder:
    """Records (iteration, wall time, error, work) with GPU-synchronised timing.

    ``work`` counts matrix passes (one full sweep over the n x m kernel = 1), the
    hardware-independent cost unit we use to compare solvers with very different iterations.
    """

    def __init__(self, res: Result, dev):
        self.res, self.dev, self.t0 = res, dev, None
        self.work = 0.0

    def start(self):
        self._sync()
        self.t0 = time.perf_counter()

    def _sync(self):
        if isinstance(self.dev, torch.device) and self.dev.type == "cuda":
            torch.cuda.synchronize(self.dev)

    def __call__(self, it, err, work=0.0):
        self.work += work
        self._sync()
        h = self.res.hist
        h["it"].append(it)
        h["time"].append(time.perf_counter() - self.t0)
        h["err"].append(float(err))
        h["work"].append(self.work)

    def timeout(self) -> bool:
        return bool(self.res.hist["time"]) and self.res.hist["time"][-1] > TIME_LIMIT[0]

    def finish(self, it, converged):
        self.res.n_iter, self.res.converged = it, bool(converged)
        self.res.time = self.res.hist["time"][-1] if self.res.hist["time"] else 0.0
        return self.res
