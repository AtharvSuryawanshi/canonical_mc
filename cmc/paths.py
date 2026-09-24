"""Repo-anchored output locations.

Checkpoints and Pareto runs used to be written to ``Path("checkpoints")`` /
``Path("pareto_runs")``, i.e. relative to whatever directory you happened to
launch from. Now that the notebooks live in ``notebooks/`` and the batch
scripts in ``slurm/``, that would scatter results into subdirectories. These
constants pin both to the repo root instead, so a notebook, a slurm job and an
interactive run all read and write the same place.

Set ``CMC_ROOT`` to override (useful on a cluster where scratch is elsewhere).
"""

import os
from pathlib import Path

#: Repo root: the parent of the ``cmc/`` package directory.
REPO_ROOT = Path(os.environ.get("CMC_ROOT", Path(__file__).resolve().parent.parent))

#: Trained weights written by ``cmc.train_cog.save_checkpoint``.
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"

#: One subdirectory per Pareto sweep, written by ``cmc.pareto``.
PARETO_RUNS_DIR = REPO_ROOT / "pareto_runs"

#: Saved networks at hand-picked lambda points, written by ``cmc.zoom_lambda``.
ZOOM_LAMBDA_DIR = REPO_ROOT / "zoom_lambda_runs"

__all__ = ["REPO_ROOT", "CHECKPOINTS_DIR", "PARETO_RUNS_DIR", "ZOOM_LAMBDA_DIR"]
