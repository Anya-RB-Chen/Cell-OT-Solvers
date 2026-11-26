"""E1 - correctness and accuracy of every solver on the representative sci-Plex3 instance.

* entropic solvers (P2) at eps = 0.05 vs. a Sinkhorn reference solved to 1e-13;
* exact LP solvers (P1): network simplex, HiGHS interior point, restarted PDHG;
* entropic bias <C, P_eps> - OT(a, b) as eps -> 0 (annealed Sinkhorn).
Output: results/e1_correctness.json
"""
from common import DEV, O, hist_record, save_json, sciplex_instance, warmup

EPS = 0.05


def main():
    warmup()
    prob, path = sciplex_instance()
    C, a, b = prob.C, prob.a, prob.b
    ref = O.sinkhorn(prob, EPS, tol=1e-13, max_iter=100_000)
    Pref = O.plan(C, ref.f, ref.g, a, b, EPS)
    obj_ref = O.primal_objective(C, Pref, a, b, EPS)
    rows = []
    for name, solver in O.SOLVERS.items():
        kw = dict(max_iter=300) if name == "greenkhorn" else dict(max_iter=20_000)
        with O.time_limit(300):
            r = solver(prob, EPS, tol=1e-9, **kw)
        P = r.P.to(C) if r.P is not None else O.plan(C, r.f.to(C), r.g.to(C), a, b, EPS)
        dual = O.dual_objective(C, r.f.to(C), r.g.to(C), a, b, EPS) if name not in ("apdagd",) else float("nan")
        rows.append(hist_record(r, eps=EPS, primal=O.primal_objective(C, P, a, b, EPS), dual=dual,
                                rel_obj_err=abs(O.primal_objective(C, P, a, b, EPS) - obj_ref) / obj_ref,
                                plan_l1=float((P - Pref).abs().sum()), transport_cost=float((C * P).sum()),
                                diverged=bool(r.extra.get("diverged", False))))
        print(f"{name:16s} conv={r.converged} it={r.n_iter} t={r.time:.3f}s relerr={rows[-1]['rel_obj_err']:.2e}")

    lp = []
    e = O.emd(prob)
    lp.append({"solver": e.name, "time": e.time, "cost": e.extra["cost"], "n_iter": None, "converged": e.converged})
    h = O.highs_ipm(prob)
    lp.append({"solver": h.name, "time": h.time, "cost": h.extra.get("cost"), "n_iter": h.n_iter, "converged": h.converged})
    with O.time_limit(600):
        d = O.pdhg(prob, tol=1e-6)
    lp.append({"solver": d.name, "time": d.time, "cost": d.extra["cost"], "n_iter": d.n_iter, "converged": d.converged,
               "hist": {k: list(map(float, v)) for k, v in d.hist.items()}})
    for r in lp:
        print(r["solver"], r["time"], r["cost"], r["n_iter"])

    bias = []
    for eps in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
        r = O.eps_scaling(prob, eps, tol=1e-9, max_iter=200_000)
        P = O.plan(C, r.f, r.g, a, b, eps)
        bias.append({"eps": eps, "cost": float((C * P).sum()), "iters": r.n_iter, "time": r.time,
                     "converged": r.converged, "nnz_frac": float((P > 1e-12).float().mean())})
        print("bias", eps, bias[-1])
    save_json({"instance": {**path, "n": prob.shape[0], "m": prob.shape[1], "eps": EPS},
               "reference_obj": obj_ref, "entropic": rows, "lp": lp, "bias": bias, "lp_cost": e.extra["cost"],
               "device": str(DEV)}, "e1_correctness.json")


if __name__ == "__main__":
    main()
