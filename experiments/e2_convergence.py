"""E2 - convergence rates of the entropic solvers, and time-to-accuracy statistics.

(a) curves: marginal error vs. iterations / kernel passes / seconds on the representative
    instance for eps in {0.1, 0.05, 0.01};
(b) statistics: 30 random sci-Plex3 instances x 3 eps -> time and work to reach 1e-8, used for
    the Dolan-More performance profile (E8).
Output: results/e2_curves.json, results/e2_stats.json
"""
import argparse

from common import O, hist_record, random_sciplex_instances, save_json, sciplex_instance, warmup

EPS_LIST = (0.1, 0.05, 0.01)
SOLVERS = ("sinkhorn", "greenkhorn", "eps_scaling", "anderson", "apdagd", "lbfgs", "sinkhorn_newton")


def run(prob, eps, name, tol, budget):
    kw = {"max_iter": 2_000} if name == "greenkhorn" else {"max_iter": 100_000}
    with O.time_limit(budget):
        return O.SOLVERS[name](prob, eps, tol=tol, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--part", default="both", choices=["curves", "stats", "both"])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    warmup()
    if args.part in ("curves", "both"):
        prob, path = sciplex_instance()
        curves = []
        for eps in EPS_LIST:
            for name in SOLVERS:
                r = run(prob, eps, name, tol=1e-10, budget=120)
                curves.append(hist_record(r, eps=eps))
                print(f"eps={eps} {name:16s} conv={r.converged} it={r.n_iter} t={r.time:.2f}s")
        save_json({"instance": path, "curves": curves}, "e2_curves.json")
    if args.part in ("stats", "both"):
        stats = []
        for iname, prob in random_sciplex_instances(args.k, seed=1)[args.shard::args.nshards]:
            for eps in EPS_LIST:
                for name in SOLVERS:
                    r = run(prob, eps, name, tol=1e-8, budget=60)
                    h = r.hist
                    stats.append({"instance": iname, "eps": eps, "solver": r.name, "converged": r.converged,
                                  "time": r.time, "iters": r.n_iter, "work": h["work"][-1] if h["work"] else None,
                                  "n": prob.shape[0], "m": prob.shape[1]})
            print(iname, "done", flush=True)
            save_json(stats, f"e2_stats_{args.shard}of{args.nshards}.json")


if __name__ == "__main__":
    main()
