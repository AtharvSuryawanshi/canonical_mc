"""Ground truth for cmc.selectivity: synthetic task variance with known breadth.

    python -m pytest tests/test_selectivity.py
"""

import numpy as np
import pandas as pd

from cmc.selectivity import active_units, breadth, network_summary, unit_frame, wrong_if

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


def _nets(e150, e50, acc=0.9, active=200, feasible=True):
    """Fake networks.csv frames: e150 / e50 = {point: [E breadth per seed]}, same at every thr."""
    def frame(d):
        return pd.DataFrame([dict(point=p, seed=i, E_breadth=b, min_task_acc=acc, E_active=active,
                                  is_feasible=feasible) for p, bs in d.items() for i, b in enumerate(bs)])
    return {**{("150k", t): frame(e150) for t in ("0.3", "0.5", "0.7")}, ("50k", "0.5"): frame(e50)}


def test_wrong_if_boundaries():
    # A3 max == control min: a tie counts against the claim (">="), rule 1 and 2 fail.
    # B0 max just below control min: rule 6 fails. A3 at control accuracy and breadth 4.0 = control
    # min: inside the (closed) control range, rule 3 fails.
    v = wrong_if(_nets({"control": [4.0, 5.0], "A3": [3.0, 4.0], "B0": [3.9, 3.8]},
                       {"control": [4.0], "rate_limited": [4.0]})).set_index("rule")["verdict"]
    assert v[1] == "fail" and v[2] == "fail" and v[3] == "fail" and v[6] == "fail"
    assert v[4] == "pass" and v[5] == "fail"                     # equal means: not below


def test_wrong_if_all_pass():
    v = wrong_if(_nets({"control": [4.5, 5.0], "A3": [2.5, 3.0], "B0": [4.6, 5.1]},
                       {"control": [4.5], "rate_limited": [3.0]})).set_index("rule")["verdict"]
    assert (v[[1, 2, 3, 4, 5, 6]] == "pass").all()
