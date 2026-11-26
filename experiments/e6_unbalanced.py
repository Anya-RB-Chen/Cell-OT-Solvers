"""E6 - mass-aware (viability-constrained) unbalanced OT.

(a) solver sweep: plain scaling / TI / AA / AA-TI for symmetric (tau, tau) and semi-relaxed
    (tau, inf) UOT over tau in {0.1 ... 100}, eps in {0.05, 0.01}, 8 random sci-Plex3 instances;
    plain-Sinkhorn iterations vs. the translation-mode prediction log(e0/tol) / -log(lam_a lam_b);
(b) mass: symmetric UOT transports ~ sqrt(rho) mass, not the measured viability rho; the
    semi-relaxed problem transports exactly rho (all top-dose sci-Plex3 conditions, rep1);
(c) biology: Tahoe 5 uM conditions in 6 cell lines -> survival scores s_i of the pooled DMSO cells
    vs. their cell-cycle phase / G2M score (obs annotations of Tahoe-100M).
Output: results/e6_sweep.json, e6_mass.json, e6_biology.json
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from common import DEV, I, O, random_sciplex_instances, save_json, warmup

INF = math.inf
TAUS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def sweep():
    rows = []
    insts = random_sciplex_instances(8, seed=2)
    for iname, prob in insts:
        line, drug, rep = iname.split("|")
        path = next(p for p in I.sciplex_dose_paths(line) if p["drug"] == drug and p["replicate"] == rep)
        pu = O.Problem(prob.C, prob.a, prob.b * path["rho"][3])
        for eps in (0.05, 0.01):
            for tau in TAUS:
                for mode, tt in (("symmetric", (tau, tau)), ("semi", (tau, INF))):
                    for name, fn, kw in (("Sinkhorn", O.uot_sinkhorn, {}), ("TI-Sinkhorn", O.ti_sinkhorn, {}),
                                         ("AA-Sinkhorn", O.aa_uot, {"translate": False}),
                                         ("AA-TI-Sinkhorn", O.aa_uot, {})):
                        with O.time_limit(30):
                            r = fn(pu, eps, tt, tol=1e-8, max_iter=500_000, **kw)
                        lam = (tau / (tau + eps)) * (tau / (tau + eps) if mode == "symmetric" else 1.0)
                        e0 = r.hist["err"][0]
                        rows.append({"instance": iname, "eps": eps, "tau": tau, "mode": mode, "solver": name,
                                     "iters": r.n_iter, "time": r.time, "converged": r.converged,
                                     "mass": r.extra["mass"], "dual": r.extra["dual"], "rho": path["rho"][3],
                                     "pred_iters": math.log(e0 / 1e-8) / -math.log(lam)})
            print(iname, eps, flush=True)
        save_json(rows, "e6_sweep.json")


def mass():
    rows = []
    for line in ("CVCL_0023", "CVCL_0004", "CVCL_0031"):
        X = I.control_cells("sciplex3", line, n=1000, seed=0, replicate="rep1")
        for p in I.sciplex_dose_paths(line):
            if p["replicate"] != "rep1":
                continue
            Y = I.cells_of("sciplex3", line, p["group_ids"][3])
            prob = O.Problem.from_points(X, Y, normalize=None, b=np.full(len(Y), p["rho"][3] / len(Y)), dev=DEV)
            for mode, tt in (("symmetric", (1.0, 1.0)), ("semi", (1.0, INF))):
                r = O.aa_uot(prob, 0.05, tt, tol=1e-8, max_iter=20_000)
                s = r.extra["survival"]
                rows.append({"line": line, "drug": p["drug"], "rho": p["rho"][3], "mode": mode,
                             "mass": r.extra["mass"], "iters": r.n_iter, "s_cv": float(s.std() / s.mean())})
        print(line, flush=True)
    save_json(rows, "e6_mass.json")


def biology(lines):
    sys.path.insert(0, I.__file__.rsplit("/", 1)[0])
    sys.path.insert(0, "/filesdir/AI4Science/VirtCell/data/build")
    from vcdata.api import PerturbData
    obs = PerturbData("tahoe").obs[["g2m_score", "s_score", "phase"]]
    rows = []
    for line in lines:
        X, xrows = I.control_cells("tahoe", line, n=1024, seed=0, return_rows=True)
        ph = obs.phase.to_numpy()[xrows]
        g2m = obs.g2m_score.to_numpy()[xrows]
        sph = obs.s_score.to_numpy()[xrows]
        cond = I.tahoe_conditions(line)
        cond = cond[cond.dose == cond.dose.max()]
        for _, c in cond.iterrows():
            Y = I.cells_of("tahoe", line, c.group_ids, n=256)
            if len(Y) < 30:
                continue
            prob = O.Problem.from_points(X, Y, normalize=None, b=np.full(len(Y), c.rho / len(Y)), dev=DEV)
            r = O.aa_uot(prob, 0.05, (1.0, INF), tol=1e-7, max_iter=20_000)
            s = r.extra["survival"].cpu().numpy()
            cyc = np.maximum(g2m, sph)
            rows.append({"line": line, "drug": c.drug, "rho": c.rho, "n_cells": c.n_cells,
                         "spearman_cycle": float(spearmanr(s, cyc).statistic),
                         **{f"s_{k}": float(s[ph == k].mean()) for k in ("G1", "S", "G2M")},
                         "cost": float((prob.C * O.plan(prob.C, r.f, r.g, prob.a, prob.b, 0.05)).sum() / c.rho)})
        print(line, len(rows), flush=True)
        save_json(rows, "e6_biology.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["sweep", "mass", "biology", "all"])
    ap.add_argument("--lines", nargs="*", default=["CVCL_0023", "CVCL_0480", "CVCL_0366", "CVCL_0069", "CVCL_1098", "CVCL_0334"])
    a = ap.parse_args()
    warmup()
    if a.part in ("mass", "all"):
        mass()
    if a.part in ("biology", "all"):
        biology(a.lines)
    if a.part in ("sweep", "all"):
        sweep()


if __name__ == "__main__":
    main()
