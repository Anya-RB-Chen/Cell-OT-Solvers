"""E5 - dose-path warm start (DP-WS) on every sci-Plex3 dose path.

All drugs of one (cell line, replicate) are compared with the same vehicle sample, so a path
vehicle -> 10 nM -> 100 nM -> 1 uM -> 10 uM is a sequence of OT problems with a shared source.

Methods per dose k (balanced, Sinkhorn unless noted):
  cold        f0 = 0
  eps-scaling f0 = 0, continuation in eps
  anderson    f0 = 0, Anderson acceleration
  dpws        f0 = f*_{k-1}; dose 1 starts from the *anchor* f*_0 = OT(vehicle -> held-out vehicle),
              solved once per (line, replicate) and amortised over ~180 drugs
  dpws+aa     DP-WS initialisation + Anderson
  uot_cold / uot_dpws   the same for semi-relaxed UOT (AA-TI, tau_a = 1) with target mass = viability rho
Theory check: osc(f*_k), osc(f*_k - f*_{k-1}), empirical cold rate -> predicted saving (Prop. 1),
and the entropic OT cost between consecutive dose distributions.
Usage: python e5_warmstart.py --line CVCL_0031 --eps 0.05
Output: results/e5_<line>_<eps>.json
"""
import argparse
import math

import torch

from common import DEV, I, O, save_json, warmup

TOL = 1e-8
TAU = 1.0


def run_path(X, Ys, rhos, eps, f_anchor):
    probs = [O.Problem.from_points(X, Y, normalize=None, dev=DEV) for Y in Ys]
    out = {m: [] for m in ("cold", "eps_scaling", "anderson", "dpws", "dpws_aa", "uot_cold", "uot_dpws")}
    fstar, f_prev, f_prev_u = [], f_anchor, f_anchor
    for k, p in enumerate(probs):
        r_cold = O.sinkhorn(p, eps, tol=TOL, max_iter=50_000)
        r_eps = O.eps_scaling(p, eps, tol=TOL, max_iter=50_000)
        r_aa = O.anderson(p, eps, tol=TOL, max_iter=50_000)
        r_w = O.sinkhorn(p, eps, tol=TOL, max_iter=50_000, f0=f_prev)
        r_waa = O.anderson(p, eps, tol=TOL, max_iter=50_000, f0=f_prev)
        rate = O.empirical_rate(r_cold.hist["err"], lo=1e-3, hi=TOL * 10)
        fs = r_cold.f
        rec = {"rate": rate, "osc_star": O.osc(fs), "osc_gap": O.osc(fs - f_prev),
               "pred_saving": O.predicted_saving(f_prev, fs, rate)}
        for m, r in (("cold", r_cold), ("eps_scaling", r_eps), ("anderson", r_aa), ("dpws", r_w), ("dpws_aa", r_waa)):
            out[m].append({"iters": r.n_iter, "time": r.time, "conv": r.converged, **(rec if m == "dpws" else {})})
        # unbalanced, target mass = viability
        pu = O.Problem(p.C, p.a, p.b * rhos[k])
        u_cold = O.aa_uot(pu, eps, (TAU, math.inf), tol=TOL, max_iter=50_000)
        u_w = O.aa_uot(pu, eps, (TAU, math.inf), tol=TOL, max_iter=50_000, f0=f_prev_u)
        out["uot_cold"].append({"iters": u_cold.n_iter, "time": u_cold.time, "conv": u_cold.converged,
                                "mass": u_cold.extra["mass"], "rho": rhos[k]})
        out["uot_dpws"].append({"iters": u_w.n_iter, "time": u_w.time, "conv": u_w.converged})
        fstar.append(fs)
        f_prev, f_prev_u = fs, u_cold.f
    # distance between consecutive target distributions (entropic OT cost, same eps)
    dists = []
    Yprev = None
    for Y in Ys:
        if Yprev is None:
            dists.append(float("nan"))
        else:
            q = O.Problem.from_points(Yprev, Y, normalize=None, dev=DEV)
            r = O.sinkhorn(q, eps, tol=1e-6, max_iter=20_000)
            dists.append(float((q.C * O.plan(q.C, r.f, r.g, q.a, q.b, eps)).sum()))
        Yprev = Y
    return out, dists


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", default="CVCL_0031")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--n_ctrl", type=int, default=1000)
    ap.add_argument("--max_paths", type=int, default=10_000)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    warmup()
    paths = I.sciplex_dose_paths(a.line)[: a.max_paths][a.shard::a.nshards]
    tag = f"e5_{a.line}_{a.eps}_{a.shard}of{a.nshards}.json"
    results = []
    anchors = {}
    for j, p in enumerate(paths):
        rep = p["replicate"]
        X = I.control_cells("sciplex3", a.line, n=a.n_ctrl, seed=0, replicate=rep)
        if rep not in anchors:      # anchor: vehicle -> held-out vehicle cells (different seed)
            Xh = I.control_cells("sciplex3", a.line, n=256, seed=1, replicate=rep)
            ra = O.sinkhorn(O.Problem.from_points(X, Xh, normalize=None, dev=DEV), a.eps, tol=TOL, max_iter=50_000)
            anchors[rep] = ra.f
        Ys = [I.cells_of("sciplex3", a.line, gid, seed=0) for gid in p["group_ids"]]
        out, dists = run_path(X, Ys, p["rho"], a.eps, anchors[rep])
        results.append({"drug": p["drug"], "replicate": rep, "doses": p["doses"], "rho": p["rho"],
                        "n_cells": p["n_cells"], "dist_prev": dists, **out})
        if j % 20 == 0:
            print(j, len(paths), p["drug"], [r["iters"] for r in out["cold"]], [r["iters"] for r in out["dpws"]], flush=True)
            save_json({"line": a.line, "eps": a.eps, "paths": results}, tag)
    save_json({"line": a.line, "eps": a.eps, "paths": results}, tag)


if __name__ == "__main__":
    main()
