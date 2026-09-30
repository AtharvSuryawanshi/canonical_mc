"""Deprecated: renamed to ``cmc.lambda_zoom``. Kept so old imports and commands work."""

import warnings

warnings.warn(
    "cmc.zoom_lambda is deprecated: use cmc.lambda_zoom (python -m cmc.lambda_zoom).",
    DeprecationWarning,
    stacklevel=2,
)

from cmc.lambda_zoom import *  # noqa: E402,F401,F403
from cmc.lambda_zoom import main  # noqa: E402,F401
from cmc.runner import resolve_run_settings, train_and_evaluate  # noqa: E402,F401

if __name__ == "__main__":
    print("NOTE: python -m cmc.zoom_lambda is deprecated; use python -m cmc.lambda_zoom")
    main()
