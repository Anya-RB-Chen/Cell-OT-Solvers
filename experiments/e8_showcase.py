"""E8 - showcase data for Fig. 1 and the web demo.

Tahoe-100M A549 (CVCL_0023): pooled DMSO cells (n = 600) and the 5 uM cells of a handful of drugs
with different viability. One UMAP is fitted on all of them (from the PCA-50 coordinates). For each
drug we solve the balanced and the semi-relaxed (mass = rho) entropic OT and keep, per control cell,
the survival score s_i, its barycentric target, and the strongest couplings for drawing.
Output: results/e8_showcase.json  (also copied to docs/assets/ by make_demo_assets.py)
"""
import math
import sys

import numpy as np

from common import DEV, I, O, save_json

LINE, EPS, TAU = "CVCL_0023", 0.05, 1.0
N_CTRL, N_TRT, N_DRUGS = 600, 200, 8


def main():
    sys.path.insert(0, "/filesdir/AI4Science/VirtCell/data/build")
    from vcdata.api import PerturbData
    obs = PerturbData("tahoe").obs
    X, xrows = I.control_cells("tahoe", LINE, n=N_CTRL, seed=0, return_rows=True)
    phase = obs.phase.to_numpy()[xrows]
    cond = I.tahoe_conditions(LINE)
    top = cond[(cond.dose == cond.dose.max()) & (cond.n_cells >= 60)].copy()
    # drugs spanning the viability range, plus two well-known cytotoxic classes when present
    qs = np.quantile(top.rho, np.linspace(0.03, 0.97, N_DRUGS))
    picks = []
    for q in qs:
        cand = top[~top.drug.isin([p.drug for p in picks])]
        picks.append(cand.iloc[(cand.rho - q).abs().argmin()])
    Ys = [I.cells_of("tahoe", LINE, p.group_ids, n=N_TRT, seed=1) for p in picks]

    import umap
    allpts = np.concatenate([X] + Ys)
    emb = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(allpts)
    emb = (emb - emb.mean(0)) / emb.std(0)
    ex = emb[:N_CTRL]
    out = {"line": LINE, "cell_name": "A549", "eps": EPS, "tau": TAU,
           "control": {"xy": np.round(ex, 3).tolist(), "phase": phase.tolist()}, "drugs": []}
    off = N_CTRL
    for p, Y in zip(picks, Ys):
        ey = emb[off:off + len(Y)]
        off += len(Y)
        prob = O.Problem.from_points(X, Y, normalize=None, b=np.full(len(Y), p.rho / len(Y)), dev=DEV)
        bal = O.sinkhorn(O.Problem(prob.C, prob.a, prob.b / prob.b.sum()), EPS, tol=1e-9)
        uot = O.aa_uot(prob, EPS, (TAU, math.inf), tol=1e-9)
        Pb = O.plan(prob.C, bal.f, bal.g, prob.a, prob.b / prob.b.sum(), EPS).cpu().numpy()
        Pu = O.plan(prob.C, uot.f, uot.g, prob.a, prob.b, EPS).cpu().numpy()
        s = Pu.sum(1) / prob.a.cpu().numpy()
        # strongest coupling per control cell (for drawing arrows) and barycentric map in UMAP space
        j = Pu.argmax(1)
        bary = (Pu @ ey) / np.maximum(Pu.sum(1, keepdims=True), 1e-300)
        out["drugs"].append({
            "drug": p.drug, "rho": float(p.rho), "n_cells": int(p.n_cells),
            "xy": np.round(ey, 3).tolist(), "survival": np.round(s, 4).tolist(),
            "match": j.tolist(), "bary": np.round(bary, 3).tolist(),
            "cost_balanced": float((prob.C.cpu().numpy() * Pb).sum()),
            "cost_uot": float((prob.C.cpu().numpy() * Pu).sum() / Pu.sum()),
            "iters_aa_ti": uot.n_iter,
            "s_by_phase": {k: float(s[phase == k].mean()) for k in ("G1", "S", "G2M")},
        })
        print(p.drug, round(p.rho, 3), out["drugs"][-1]["s_by_phase"], flush=True)
    save_json(out, "e8_showcase.json")


if __name__ == "__main__":
    main()
