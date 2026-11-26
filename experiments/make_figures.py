"""Paper figures from results/*.json. Usage: python make_figures.py [fig1 fig2 ...]
Writes figures/<name>.pdf (+ .png) and copies the PDFs to the paper repo if it exists."""
from __future__ import annotations

import glob
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
PAPER = ROOT.parent / "COMP6704_Advanced_Optimization_Indivisual_Assignment" / "figures"
FIG.mkdir(exist_ok=True)

# ── style: one fixed colour + marker per solver across every figure (colour follows the entity) ──
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df"
SOLVER = {   # name in results -> (label, colour, marker, linestyle)
    "Sinkhorn": ("Sinkhorn", "#2a78d6", "o", "-"),
    "Anderson-Sinkhorn": ("Anderson", "#eb6834", "s", "-"),
    "ε-scaling": ("ε-scaling", "#1baf7a", "^", "-"),
    "Greenkhorn": ("Greenkhorn", "#eda100", "D", "-"),
    "APDAGD": ("APDAGD", "#e87ba4", "v", "-"),
    "L-BFGS (semi-dual)": ("L-BFGS", "#008300", "P", "-"),
    "Sinkhorn-Newton-Sparse": ("SNS", "#6250d6", "X", "-"),
    "Sinkhorn (kernel)": ("Sinkhorn (kernel)", MUTED, "o", ":"),
    "Network simplex (EMD)": ("Net. simplex", INK2, "h", "--"),
    "Interior point (HiGHS)": ("IPM (HiGHS)", MUTED, "p", "--"),
    "Restarted PDHG": ("rPDHG", INK, "*", "--"),
}
UOT = {
    "Sinkhorn": ("UOT Sinkhorn", "#2a78d6", "o", "-"),
    "TI-Sinkhorn": ("TI-Sinkhorn", "#1baf7a", "^", "-"),
    "AA-Sinkhorn": ("AA (no TI)", "#eb6834", "s", "--"),
    "AA-TI-Sinkhorn": ("AA-TI (ours)", "#e34948", "*", "-"),
}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 3.5, "legend.frameon": False, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "pdf.fonttype": 42, "savefig.dpi": 300,
})
W = 5.5   # NeurIPS text width (in)


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    if PAPER.parent.exists():
        PAPER.mkdir(exist_ok=True)
        shutil.copy(FIG / f"{name}.pdf", PAPER / f"{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}")


def style_of(name, table=SOLVER):
    return table.get(name, (name, MUTED, "o", "-"))


# ── Fig. 2: convergence curves + entropic bias ──────────────────────────────────────────────
def fig_convergence():
    cur = load("e2_curves.json")["curves"]
    e1 = load("e1_correctness.json")
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.55), gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    for ax, eps, L in zip(axes[:3], (0.1, 0.05, 0.01), "abc"):
        for r in cur:
            if r["eps"] != eps:
                continue
            lab, col, mk, ls = style_of(r["solver"])
            w, e = np.array(r["hist"]["work"]), np.array(r["hist"]["err"])
            keep = e > 0
            w, e = w[keep], e[keep]
            ax.plot(w, e, color=col, ls=ls, lw=1.2, label=lab, marker=mk, markevery=max(1, len(w) // 6), ms=2.8)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_ylim(1e-10, 3)
        ax.set_xlabel("kernel passes")
        panel(ax, L, f"ε = {eps}")
    axes[0].set_ylabel("marginal violation")
    axes[0].legend(loc="lower left", ncol=1, handlelength=1.6, borderaxespad=0.1)
    # (d) entropic bias vs eps
    ax = axes[3]
    b = pd.DataFrame(e1["bias"])
    gap = (b.cost - e1["lp_cost"]) / e1["lp_cost"]
    ax.loglog(b.eps, gap, color="#2a78d6", marker="o", label="rel. bias  (⟨C,P_ε⟩−OT)/OT")
    ax2_vals = b.iters
    ax.loglog(b.eps, b.eps * np.log(1 / b.eps) * gap.iloc[0] / (b.eps.iloc[0] * np.log(1 / b.eps.iloc[0]) + 1e-12),
              color=MUTED, ls=":", lw=1, label="∝ ε log(1/ε)")
    for x, y, it in zip(b.eps, gap, ax2_vals):
        if x in (1.0, 0.1, 0.01, 0.002):
            ax.annotate(f"{it} it", (x, y), textcoords="offset points", xytext=(2, 4), fontsize=5.5, color=INK2)
    ax.set_xlabel("ε")
    ax.invert_xaxis()
    panel(ax, "d", "entropic bias → LP")
    ax.legend(loc="lower left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_convergence")


# ── Fig. 3: stability, complexity, performance profile ──────────────────────────────────────
def perf_profile(df, key="time"):
    piv = df.pivot_table(index=["instance", "eps"], columns="solver", values=key)
    conv = df.pivot_table(index=["instance", "eps"], columns="solver", values="converged")
    piv = piv.where(conv > 0.5)
    best = piv.min(1)
    ratio = piv.div(best, axis=0)
    return ratio


def fig_stability():
    e3 = load("e3_stability.json")
    stats = pd.concat([pd.DataFrame(json.loads(Path(f).read_text())) for f in glob.glob(str(RES / "e2_stats_*of10.json"))])
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.6))
    # (a) eps sweep: iterations + rates
    sw = pd.DataFrame(e3["eps_sweep"])
    ax = axes[0]
    ax.loglog(sw.eps, sw.log_iters, color="#2a78d6", marker="o", label="log-domain")
    ok = sw[~sw.kernel_diverged]
    ax.loglog(ok.eps, ok.kernel_iters, color=MUTED, ls=":", marker="o", mfc="white", label="kernel (u,v)")
    bad = sw[sw.kernel_diverged]
    if len(bad):
        ax.scatter(bad.eps, bad.log_iters, marker="x", color="#e34948", s=18, zorder=5, label="kernel → NaN")
    ax.invert_xaxis()
    ax.set_xlabel("ε")
    ax.set_ylabel("Sinkhorn iterations to 1e-8")
    panel(ax, "a", "stability vs ε")
    ax.legend(loc="upper left")
    # (b) empirical rate: 1 - lambda_hat ~ c * eps^p, while the Franklin-Lorenz bound is vacuous
    ax = axes[1]
    gap = 1 - sw.rate_emp
    ax.loglog(sw.eps, gap, color="#2a78d6", marker="o", label="measured 1−λ̂")
    p_, c_ = np.polyfit(np.log(sw.eps[sw.eps <= 0.1]), np.log(gap[sw.eps <= 0.1]), 1)
    xx = np.logspace(np.log10(sw.eps.min()), np.log10(0.1), 50)
    ax.loglog(xx, np.exp(c_) * xx ** p_, color=MUTED, ls="--", lw=0.9, label=f"fit ∝ ε^{p_:.2f}")
    vac = sw.eps[(1 - sw.rate_bound) < 1e-6].max()
    ax.text(0.04, 0.04, f"FL bound vacuous:\n1−λ<1e-6 (ε≤{vac:g})", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.4, color="#e34948")
    ax.invert_xaxis()
    ax.set_xlabel("ε")
    ax.set_ylabel("contraction gap 1−λ")
    panel(ax, "b", "linear rate")
    ax.set_ylim(top=40)
    ax.legend(loc="upper right", fontsize=5.6)
    # (c) performance profile on time
    ratio = perf_profile(stats, "time")
    ax = axes[2]
    taus = np.logspace(0, 3, 200)
    order = ["Anderson-Sinkhorn", "Sinkhorn", "ε-scaling", "L-BFGS (semi-dual)", "Sinkhorn-Newton-Sparse", "Greenkhorn", "APDAGD"]
    for s in order:
        if s not in ratio:
            continue
        lab, col, mk, ls = style_of(s)
        r = ratio[s].to_numpy()
        frac = [(np.nan_to_num(r, nan=np.inf) <= t).mean() for t in taus]
        ax.semilogx(taus, frac, color=col, ls=ls, lw=1.2, label=lab)
    ax.set_xlabel(f"time / best time  ({len(ratio)} runs)")
    ax.set_ylabel("fraction solved")
    ax.set_ylim(0, 1.02)
    panel(ax, "c", "perf. profile")
    # (d) robustness over seeds: median and range of kernel passes
    sd = pd.DataFrame(e3["seeds"])
    ax = axes[3]
    names = [s for s in order if s in set(sd.solver)]
    for k, s in enumerate(names):
        w = sd[sd.solver == s].work.to_numpy()
        lab, col, mk, ls = style_of(s)
        ax.plot([w.min(), w.max()], [k, k], color=col, lw=2.2, alpha=0.45, solid_capstyle="round")
        ax.plot(np.median(w), k, marker=mk, color=col, ms=5, mec="white", mew=0.6)
        conv = sd[sd.solver == s].converged.mean()
        if conv < 1:
            ax.text(w.max() * 1.3, k, f"{conv:.0%} conv.", va="center", fontsize=5.2, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)), [style_of(s)[0] for s in names])
    ax.invert_yaxis()
    ax.set_xlabel("kernel passes (10 seeds)")
    panel(ax, "d", "robustness")
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_stability")


def fig_scaling():
    rows = pd.DataFrame(load("e4_scaling.json"))
    fig, ax = plt.subplots(figsize=(2.0, 1.7))
    for s, d in rows.groupby("solver"):
        d = d[d.converged]
        lab, col, mk, ls = style_of(s)
        ax.loglog(d.n, d.time, color=col, ls=ls, marker=mk, label=lab)
    ax.set_xlabel("n = m")
    ax.set_ylabel("seconds")
    ax.legend(fontsize=5.5, ncol=2)
    save(fig, "fig_scaling")


# ── Fig. 4: unbalanced OT ───────────────────────────────────────────────────────────────────
def fig_uot():
    sw = pd.DataFrame(load("e6_sweep.json"))
    mass = pd.DataFrame(load("e6_mass.json"))
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.6))
    for ax, eps, L in zip(axes[:2], (0.05, 0.01), "ab"):
        d = sw[(sw["mode"] == "semi") & (sw.eps == eps)]
        for s in ("Sinkhorn", "TI-Sinkhorn", "AA-Sinkhorn", "AA-TI-Sinkhorn"):
            g = d[d.solver == s].groupby("tau").iters
            lab, col, mk, ls = UOT[s]
            med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
            ax.loglog(med.index, med.values, color=col, ls=ls, marker=mk, label=lab, ms=3.5 if s != "AA-TI-Sinkhorn" else 5)
            ax.fill_between(med.index, lo.values, hi.values, color=col, alpha=0.12, lw=0)
        ax.set_xlabel("τ (VC-UOT)")
        panel(ax, L, f"sweeps to 1e-8, ε = {eps}")
    axes[0].set_ylabel("iterations (median, 10–90%)")
    axes[0].legend(loc="upper left")
    # (c) prediction of Prop. 2
    ax = axes[2]
    d = sw[(sw.solver == "Sinkhorn") & sw.converged]
    for mode, mk, col in (("symmetric", "o", "#2a78d6"), ("semi", "^", "#e34948")):
        q = d[d["mode"] == mode]
        ax.loglog(q.pred_iters, q.iters, ls="none", marker=mk, ms=2.5, color=col, alpha=0.7,
                  label="symmetric (λ²)" if mode == "symmetric" else "VC-UOT (λ)")
    lim = [10, max(d.iters.max(), d.pred_iters.max()) * 1.5]
    ax.loglog(lim, lim, color=MUTED, ls="--", lw=0.8, label="y = x")
    ax.set_xlabel("predicted  log(e₀/δ) / log(1/λ_aλ_b)")
    ax.set_ylabel("measured iterations")
    panel(ax, "c", "Prop. 2: translation mode")
    ax.legend(loc="upper left")
    # (d) mass vs viability
    ax = axes[3]
    for mode, mk, col, lab in (("symmetric", "o", "#2a78d6", "symmetric UOT"), ("semi", "^", "#e34948", "VC-UOT (ours)")):
        q = mass[mass["mode"] == mode]
        ax.scatter(q.rho, q.mass, s=4, marker=mk, color=col, alpha=0.6, lw=0, label=lab)
    x = np.linspace(mass.rho.min(), mass.rho.max(), 100)
    ax.plot(x, x, color=INK2, lw=0.8, ls="--", label="mass = ρ")
    ax.plot(x, np.sqrt(x), color=MUTED, lw=0.8, ls=":", label="mass = √ρ")
    ax.set_xlabel("measured viability ρ")
    ax.set_ylabel("transported mass")
    panel(ax, "d", "Prop. 1: mass")
    ax.legend(loc="upper left", markerscale=2)
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig_uot")


# ── Fig. 5: warm start + atlas ──────────────────────────────────────────────────────────────
def load_e5(eps):
    rows = []
    for f in glob.glob(str(RES / f"e5_*_{eps}_*of8.json")):
        d = json.loads(Path(f).read_text())
        for p in d["paths"]:
            for k in range(4):
                r = {"line": d["line"], "drug": p["drug"], "dose": k, "rho": p["rho"][k], "dist": p["dist_prev"][k]}
                for m in ("cold", "eps_scaling", "anderson", "dpws", "dpws_aa", "uot_cold", "uot_dpws"):
                    r[m] = p[m][k]["iters"]
                    r[m + "_t"] = p[m][k]["time"]
                r.update({k2: p["dpws"][k][k2] for k2 in ("rate", "osc_star", "osc_gap", "pred_saving")})
                rows.append(r)
    return pd.DataFrame(rows)


def fig_warm_atlas():
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.65), gridspec_kw={"width_ratios": [1.15, 1, 1, 1.35]})
    ax = axes[0]
    labels = ["cold", "ε-scal.", "AA", "DP-WS", "DP-WS\n+AA", "VC-UOT\ncold", "VC-UOT\nDP-WS"]
    cols = ["cold", "eps_scaling", "anderson", "dpws", "dpws_aa", "uot_cold", "uot_dpws"]
    colors = ["#2a78d6", "#1baf7a", "#eb6834", "#2a78d6", "#eb6834", "#e34948", "#e34948"]
    for i, eps in enumerate((0.05, 0.01)):
        df = load_e5(eps)
        if df.empty:
            continue
        data = [df[c].to_numpy() for c in cols]
        pos = np.arange(len(cols)) * 2.4 + i * 1.0
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True, showfliers=False)
        for patch, c, h in zip(bp["boxes"], colors, ["", "", "", "////", "////", "", "////"]):
            patch.set_facecolor(c); patch.set_alpha(0.9 if i == 0 else 0.45); patch.set_edgecolor("white"); patch.set_hatch(h)
        for med in bp["medians"]:
            med.set_color(INK)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(cols)) * 2.4 + 0.5, labels, fontsize=5.2)
    ax.set_ylabel("iterations (dark ε=.05, light ε=.01)")
    panel(ax, "a", "shared-source warm start")
    # (b) theory: predicted vs measured saving
    ax = axes[1]
    df = load_e5(0.01)
    if not df.empty:
        d = df[df.dose > 0]
        ax.scatter(d.pred_saving, d.cold - d.dpws, s=2, color="#2a78d6", alpha=0.35, lw=0)
        lim = [min(d.pred_saving.min(), (d.cold - d.dpws).min()), max(d.pred_saving.max(), (d.cold - d.dpws).max())]
        ax.plot(lim, lim, color=MUTED, ls="--", lw=0.8)
        r = np.corrcoef(d.pred_saving, d.cold - d.dpws)[0, 1]
        ax.text(0.03, 0.95, f"r = {r:.2f}\nmedian osc ratio = {np.median(d.osc_gap / d.osc_star):.2f}",
                transform=ax.transAxes, va="top", fontsize=5.8, color=INK2)
    ax.set_xlabel("predicted saving, Eq. (4)")
    ax.set_ylabel("measured saving (iters)")
    panel(ax, "b", "why it helps little")
    # (c) atlas: iterations distribution + throughput
    at = pd.read_parquet(RES / "e7_atlas.parquet") if (RES / "e7_atlas.parquet").exists() else None
    ax = axes[2]
    if at is not None:
        ax.hist(at.iters.clip(upper=at.iters.quantile(0.995)), bins=40, color="#e34948", alpha=0.85, edgecolor="white", lw=0.3)
        ax.set_xlabel("TI-Sinkhorn sweeps per condition")
        ax.set_ylabel("conditions")
        panel(ax, "c", f"atlas: {len(at):,} problems")
    # (d) atlas heatmap of OT effect size
    ax = axes[3]
    if at is not None:
        top = at[at.dose == at.dose.max()]
        piv = top.pivot_table(index="drug", columns="line", values="cost")
        piv = piv.loc[piv.var(1).sort_values(ascending=False).index[:30]]
        im = ax.imshow(np.log(piv.values), aspect="auto", cmap=mpl.colors.LinearSegmentedColormap.from_list("b", SEQ))
        ax.set_yticks(range(len(piv)), [d[:14] for d in piv.index], fontsize=3.6)
        ax.set_xticks([])
        ax.set_xlabel("50 cell lines")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=5)
        cb.set_label("log OT effect", fontsize=5.5)
        panel(ax, "d", "OT effect at 5 µM")
    fig.tight_layout(w_pad=0.4)
    save(fig, "fig_warm_atlas")


# ── Fig. 1: teaser ─────────────────────────────────────────────────────────────────────────
def fig_teaser():
    sh = load("e8_showcase.json")
    sw = pd.DataFrame(load("e6_sweep.json")) if (RES / "e6_sweep.json").exists() else None
    d = min(sh["drugs"], key=lambda r: r["rho"])
    cx = np.array(sh["control"]["xy"])
    ty = np.array(d["xy"])
    s = np.array(d["survival"])
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.85), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ax = axes[0]
    ax.scatter(cx[:, 0], cx[:, 1], s=3, color=MUTED, lw=0, label=f"DMSO (n={len(cx)})")
    ax.scatter(ty[:, 0], ty[:, 1], s=4, color="#eb6834", lw=0, label=f"{d['drug']} 5 µM")
    rng = np.random.default_rng(0)
    for i in rng.choice(len(cx), 70, replace=False):
        j = d["match"][i]
        ax.annotate("", ty[j], cx[i], arrowprops=dict(arrowstyle="-|>", lw=0.35, color="#2a78d6", alpha=0.55, mutation_scale=4))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.legend(loc="lower left", markerscale=2.5, fontsize=5.5)
    panel(ax, "a", "OT coupling (A549, Tahoe-100M)")
    ax = axes[1]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("surv", ["#e34948", "#f0efec", "#2a78d6"])
    norm = mpl.colors.TwoSlopeNorm(vcenter=d["rho"], vmin=min(s.min(), d["rho"] * 0.5), vmax=max(s.max(), d["rho"] * 1.5))
    sc = ax.scatter(cx[:, 0], cx[:, 1], c=s, s=4, cmap=cmap, norm=norm, lw=0)
    ax.scatter(ty[:, 0], ty[:, 1], s=2, color="#d9d8d3", lw=0, zorder=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    cb.set_label(f"survival s_i  (mean = ρ = {d['rho']:.2f})", fontsize=5.5)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    panel(ax, "b", "VC-UOT survival scores")
    ax = axes[2]
    if sw is not None:
        q = sw[(sw["mode"] == "semi") & (sw.eps == 0.01)]
        for sname in ("Sinkhorn", "TI-Sinkhorn", "AA-TI-Sinkhorn"):
            g = q[q.solver == sname].groupby("tau").iters.median()
            lab, col, mk, ls = UOT[sname]
            ax.loglog(g.index, g.values, color=col, ls=ls, marker=mk, label=lab, ms=3.5 if "AA" not in sname else 5)
        ax.set_xlabel("marginal penalty τ")
        ax.set_ylabel("sweeps to 1e-8 (ε = 0.01)")
        ax.legend(loc="upper left")
    panel(ax, "c", "solver speed")
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig_teaser")


FIGS = {"teaser": fig_teaser, "convergence": fig_convergence, "stability": fig_stability,
        "scaling": fig_scaling, "uot": fig_uot, "warm_atlas": fig_warm_atlas}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGS:
        try:
            FIGS[name]()
        except FileNotFoundError as e:
            print("skip", name, "(missing", e.filename, ")")
