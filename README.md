# Cell-OT Solvers

**Who dies under a drug? Viability-constrained unbalanced optimal transport and fast Sinkhorn solvers for single-cell perturbation atlases.**


[Interactive demo](https://anya-rb-chen.github.io/Cell-OT-Solvers/) · [Report (PDF)](report/CHEN_Rubing_25110871R.pdf)

<p align="center"><img src="figures/fig_teaser.png" width="92%"></p>

## What is in here

A single-cell drug screen measures control cells and treated cells as two unpaired point clouds. Optimal transport (OT) recovers how the drug moved the population, and a perturbation atlas needs one OT problem per condition: 56,877 of them for Tahoe-100M. This repository contains:

* **`otsolvers/`**: 14 OT solvers in PyTorch from four algorithm families, all with the same interface and logging.

  | family | solvers | file |
  |---|---|---|
  | exact LP (P1) | network simplex (POT), interior point (HiGHS), restarted PDHG (PDLP-style) | `exact.py` |
  | entropic, coordinate (P2) | log-domain Sinkhorn, kernel Sinkhorn, Greenkhorn, ε-scaling | `entropic.py` |
  | entropic, gradient-type (P2) | Anderson-Sinkhorn, APDAGD, L-BFGS on the semi-dual, Sinkhorn–Newton–Sparse | `entropic.py` |
  | unbalanced (P3) | UOT scaling, TI-Sinkhorn, **AA-TI-Sinkhorn (ours)**, all supporting semi-relaxed τ_b = ∞ | `unbalanced.py` |
  | atlas scale | batched, padded TI-Sinkhorn over (B, n, m) tensors | `batched.py` |
  | warm starts | shared-source warm start, Prop.-based saving predictor | `warmstart.py` |

* **`data/`**: per-cell-line PCA-50 embeddings of sci-Plex3 and Tahoe-100M (`build_embeddings.py`), plus the OT instance factory (`instances.py`) with viability ρ from cell counts.
* **`experiments/`**: one script per experiment (E1–E8), `run_all.sh` to run them all, and `make_figures.py` to rebuild every figure in the report.
* **`docs/`**: the GitHub Pages demo, which runs Sinkhorn, TI-Sinkhorn and AA-TI live in the browser and includes a real-cell explorer.

## Main results

| | |
|---|---|
| **VC-UOT (formulation)** | Symmetric KL-UOT transports ρ^((τ+ε)/(2τ+ε)) → √ρ mass. The viability-constrained problem (τ_b = ∞, \|b\| = ρ) transports exactly ρ, with survival scores s_i = exp(−f_i/τ) (Prop. 1). |
| **Translation mode (theory)** | UOT Sinkhorn shrinks a dual translation by exactly λ_aλ_b per sweep, λ = τ/(τ+ε). This predicts the measured iteration counts and the 2× gap between VC-UOT and symmetric UOT (Prop. 2). |
| **AA-TI-Sinkhorn (algorithm)** | Its sweep count does not grow with τ, up to ~10³× fewer sweeps than UOT Sinkhorn and 2–3× fewer than TI-Sinkhorn. |
| **Balanced benchmark** | Anderson-Sinkhorn is the fastest entropic solver in the performance profile. APDAGD and Greenkhorn trail. Kernel-domain Sinkhorn underflows at ε = 10⁻³. The Franklin–Lorenz rate bound is vacuous on real costs, while measured 1−λ̂ ∝ ε^1.1. |
| **Warm starts (negative result)** | Warm-starting along shared-control dose paths saves little: osc(f*_k − f*_{k−1}) ≈ osc(f*_k), because target sampling noise dominates the potentials. Eq. (4) predicts this. |
| **Atlas scale** | Every Tahoe-100M condition (50 cell lines) is solved with batched TI-Sinkhorn on one GPU. |

## Reproduce

```bash
bash download.sh                      # large downloads only: CUDA PyTorch env + raw data (run by hand)
conda activate cellot
python data/build_embeddings.py --dataset sciplex3 --workers 3
python data/build_embeddings.py --dataset tahoe --workers 10
JOBS=40 bash experiments/run_all.sh   # E1–E7; CPU jobs in parallel, E4/E7 on a GPU if present
python experiments/e8_showcase.py
python experiments/make_figures.py    # figures/*.pdf, *.png
python experiments/make_demo_assets.py
```

`data/build_embeddings.py` reads the processed single-cell store of the VirtCell data pipeline through `vcdata.api.PerturbData`. Set `VCDATA` to its `build/` directory. Every experiment script is self-contained, writes a JSON (or parquet) file to `results/`, and prints progress. `OT_DEVICE=cpu|cuda` picks the device and `OT_THREADS` sets the number of CPU threads.

| script | question | output |
|---|---|---|
| `e1_correctness.py` | Do all solvers agree? LP optimum, entropic bias | `results/e1_correctness.json` |
| `e2_convergence.py` | Convergence curves; time to accuracy on 30 instances × 3 ε | `e2_curves.json`, `e2_stats_*.json` |
| `e3_stability.py` | ε sweep, underflow, linear rate, seeds, hyper-parameter sensitivity | `e3_stability.json` |
| `e4_scaling.py` | Wall time vs n up to 16,384 | `e4_scaling.json` |
| `e5_warmstart.py` | Shared-source warm start on 1,092 sci-Plex3 dose paths | `e5_<line>_<eps>_*.json` |
| `e6_unbalanced.py` | UOT solver sweep, mass vs viability, cell-cycle analysis | `e6_sweep.json`, `e6_mass.json`, `e6_biology.json` |
| `e7_atlas.py` | All Tahoe-100M conditions, batched; throughput | `e7_atlas.parquet`, `e7_throughput.json` |
| `e8_showcase.py` | UMAP + survival data for Fig. 1 and the demo | `e8_showcase.json` |

## Using the solvers

```python
import torch, otsolvers as O
prob = O.Problem.from_points(X, Y, b=torch.full((len(Y),), rho / len(Y)))   # target mass = viability
res = O.aa_uot(prob, eps=0.05, tau=(1.0, float("inf")), tol=1e-8)            # VC-UOT with AA-TI
survival = res.extra["survival"]                                              # s_i = (P1)_i / a_i
print(res.n_iter, res.extra["mass"])                                          # mass == rho
```

## Layout

```
otsolvers/   core.py exact.py entropic.py unbalanced.py batched.py warmstart.py
data/        build_embeddings.py instances.py  (cache/ is git-ignored)
experiments/ e1…e8, common.py, run_all.sh, make_figures.py, make_demo_assets.py
results/     JSON / parquet outputs of the experiments
figures/     figures of the report (PDF + PNG)
docs/        GitHub Pages demo (index.html + assets/)
```
