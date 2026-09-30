"""Repo-anchored output locations.

Everything is pinned to the repo root (not the launch directory), so a
notebook in ``notebooks/``, a slurm job and an interactive run all read and
write the same place: ``runs/{checkpoints,lambda_pareto,lambda_zoom,moo}``.

Set ``CMC_ROOT`` to override (useful on a cluster where scratch is elsewhere).
"""

import os
from pathlib import Path

#: Repo root: the parent of the ``cmc/`` package directory.
REPO_ROOT = Path(os.environ.get("CMC_ROOT", Path(__file__).resolve().parent.parent))

#: All experiment output lives under runs/, one subdirectory per kind.
RUNS_DIR = REPO_ROOT / "runs"

#: Trained weights written by ``cmc.train_cog.save_checkpoint``.
CHECKPOINTS_DIR = RUNS_DIR / "checkpoints"

#: One subdirectory per lambda-grid sweep, written by ``cmc.lambda_pareto``.
LAMBDA_PARETO_RUNS_DIR = RUNS_DIR / "lambda_pareto"

#: Saved networks at hand-picked lambda points, written by ``cmc.lambda_zoom``.
LAMBDA_ZOOM_RUNS_DIR = RUNS_DIR / "lambda_zoom"

#: NSGA-III searches, written by ``cmc.moo``.
MOO_RUNS_DIR = RUNS_DIR / "moo"

# Deprecated names (before the runs/ layout), kept so old notebooks still import.
PARETO_RUNS_DIR = LAMBDA_PARETO_RUNS_DIR
ZOOM_LAMBDA_DIR = LAMBDA_ZOOM_RUNS_DIR

__all__ = [
    "REPO_ROOT", "RUNS_DIR", "CHECKPOINTS_DIR", "LAMBDA_PARETO_RUNS_DIR",
    "LAMBDA_ZOOM_RUNS_DIR", "MOO_RUNS_DIR", "PARETO_RUNS_DIR", "ZOOM_LAMBDA_DIR",
]
