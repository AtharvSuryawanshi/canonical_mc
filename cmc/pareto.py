"""Deprecated: renamed to ``cmc.lambda_pareto`` (front helpers: ``cmc.front``,
per-network training: ``cmc.runner``). Kept so old imports and commands work."""

import warnings

warnings.warn(
    "cmc.pareto is deprecated: use cmc.lambda_pareto (python -m cmc.lambda_pareto); "
    "front helpers are in cmc.front, training in cmc.runner.",
    DeprecationWarning,
    stacklevel=2,
)

from cmc.lambda_pareto import *  # noqa: E402,F401,F403
from cmc.lambda_pareto import main  # noqa: E402,F401

if __name__ == "__main__":
    print("NOTE: python -m cmc.pareto is deprecated; use python -m cmc.lambda_pareto")
    main()
