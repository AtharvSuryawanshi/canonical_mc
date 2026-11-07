"""Ground truth for cmc.selectivity: synthetic task variance with known breadth.

    python -m pytest tests/test_selectivity.py
"""

import numpy as np
import pandas as pd

from cmc.selectivity import active_units, breadth, network_summary, unit_frame

N_TASKS = 20


def unit_with_breadth(k, rng, scale=1.0):
    """A unit with exactly k tasks above 0.5 of its max: k values in (0.6, 1], the rest in [0, 0.4)."""
    v = rng.uniform(0.0, 0.4, N_TASKS)
    hi = rng.choice(N_TASKS, k, replace=False)
    v[hi] = rng.uniform(0.6, 1.0, k)
    v[hi[0]] = 1.0
    return scale * v


def test_known_breadth_recovered():
    rng = np.random.default_rng(0)
    ks = [1, 5, 20, 3, 12]
    tv = np.stack([unit_with_breadth(k, rng) for k in ks], axis=1)
    assert breadth(tv, 0.5).tolist() == ks


def test_scale_invariant():
    rng = np.random.default_rng(1)
    u = unit_with_breadth(7, rng)
    tv = np.stack([u, 100 * u, 0.03 * u], axis=1)
    assert breadth(tv, 0.5).tolist() == [7, 7, 7]


def test_threshold_is_strict_and_moves_breadth():
    # normalized values 1, 0.5, 0.3, 0.7: at thr 0.5 the 0.5 is not counted
    tv = np.array([[1.0], [0.5], [0.3], [0.7]])
    assert breadth(tv, 0.5).tolist() == [2]
    assert breadth(tv, 0.3).tolist() == [3]
    assert breadth(tv, 0.7).tolist() == [1]


def test_inactive_units_excluded():
    rng = np.random.default_rng(2)
    big, quiet = unit_with_breadth(2, rng, 10.0), unit_with_breadth(18, rng, 10.0 * 0.001)
    tv = np.stack([big, quiet], axis=1)
    assert active_units(tv).tolist() == [True, False]
    units = unit_frame(tv, is_e=[True, True])
    for k, v in dict(point="p", seed=0, thr=0.5, min_task_acc=0.9, is_feasible=True).items():
        units[k] = v
    row = network_summary(units)
    assert row["E_active"] == 1 and row["E_breadth"] == 2.0      # the quiet broad unit is not counted


def test_e_i_split_follows_sign():
    rng = np.random.default_rng(3)
    tv = np.stack([unit_with_breadth(k, rng) for k in (2, 4, 10, 6)], axis=1)
    units = unit_frame(tv, is_e=[True, False, True, False])
    assert units["pop"].tolist() == ["E", "I", "E", "I"]
    for k, v in dict(point="p", seed=0, thr=0.5, min_task_acc=0.9, is_feasible=True).items():
        units[k] = v
    row = network_summary(pd.DataFrame(units))
    assert row["E_breadth"] == 6.0 and row["I_breadth"] == 5.0
    assert row["E_active"] == 2 and row["I_active"] == 2
