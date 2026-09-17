"""Canonical microcircuits: cost-constrained multitask RNNs.

Modules
-------
``cmc.task``       Yang et al. (2019) cognitive task battery and trial generation.
``cmc.network``    LeakyRNN and DaleRNN (sign-constrained, E-only readout) cells.
``cmc.train_cog``  Objectives, regularizers, accuracy scoring, training loop, CLI.
``cmc.pareto``     Sweeps (lambda_rate, lambda_connectivity) and builds Pareto fronts.
``cmc.paths``      Repo-anchored ``checkpoints/`` and ``pareto_runs/`` locations.

Submodules are imported explicitly (``from cmc.task import generate_trials``)
rather than re-exported here, so that importing ``cmc`` does not pull in torch
and matplotlib.
"""

__version__ = "0.1.0"
