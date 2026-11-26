"""otsolvers: exact, entropic and unbalanced optimal-transport solvers in PyTorch."""
from .core import Problem, Result, device, time_limit, dual_objective, marginal_error, plan, primal_objective
from .entropic import (SOLVERS, anderson, apdagd, eps_scaling, greenkhorn, lbfgs, sinkhorn,
                       sinkhorn_kernel, sinkhorn_newton)
from .exact import emd, highs_ipm, pdhg
from .unbalanced import aa_uot, ti_sinkhorn, uot_dual, uot_sinkhorn
from .warmstart import empirical_rate, osc, predicted_saving, solve_path

__all__ = [
    "Problem", "Result", "device", "time_limit", "dual_objective", "marginal_error", "plan", "primal_objective",
    "SOLVERS", "anderson", "apdagd", "eps_scaling", "greenkhorn", "lbfgs", "sinkhorn",
    "sinkhorn_kernel", "sinkhorn_newton", "emd", "highs_ipm", "pdhg", "aa_uot", "ti_sinkhorn", "uot_dual",
    "uot_sinkhorn", "empirical_rate", "osc", "predicted_saving", "solve_path",
]
