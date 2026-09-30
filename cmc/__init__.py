"""Canonical microcircuits: cost-constrained multitask RNNs.

Modules
-------
``cmc.task``       Yang et al. (2019) cognitive task battery and trial generation.
``cmc.network``    LeakyRNN and DaleRNN (sign-constrained, E-only readout) cells.
``cmc.train_cog``  Objectives, regularizers, accuracy scoring, training loop, CLI.
``cmc.front``        Pareto dominance and the feasibility rule.
``cmc.runner``       Train + evaluate one network; the flags every script shares.
``cmc.lambda_pareto`` Lambda-grid sweeps (weighted sum) and their fronts.
``cmc.lambda_zoom``  Many seeds at chosen lambda points, every network saved.
``cmc.moo``          NSGA-III search over cost budgets (or lambdas).
``cmc.paths``        Repo-anchored ``runs/`` locations.

Submodules are imported explicitly (``from cmc.task import generate_trials``)
rather than re-exported here, so that importing ``cmc`` does not pull in torch
and matplotlib.
"""

__version__ = "0.1.0"
