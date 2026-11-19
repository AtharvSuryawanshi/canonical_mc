"""Task generators: the Yang et al. 2019 fixes (FIXED_ISSUES #15-#17) and the pinned old behaviour ("v1").

    python -m pytest tests/test_tasks.py -q
"""
import numpy as np
import pytest

from cmc.task import Trial, default_config, generate_trials

GO_ANTI = ["fdgo", "fdanti", "reactgo", "reactanti", "delaygo", "delayanti"]
N = 16                                   # units per ring
RING1, RING2 = slice(1, 1 + N), slice(1 + N, 1 + 2 * N)


def cfg(seed, version):
    c = default_config(n_eachring=N, seed=seed, easy_task=True)
    c["task_version"] = version
    return c


def ring_used(trial):
    x = np.abs(trial.x)
    return 1 if x[:, :, RING1].sum() > 0 and x[:, :, RING2].sum() == 0 else (
        2 if x[:, :, RING2].sum() > 0 and x[:, :, RING1].sum() == 0 else 0)


@pytest.mark.parametrize("rule", GO_ANTI)
def test_yang_stimulus_ring_is_random_per_batch(rule):
    rings = [ring_used(generate_trials(rule, cfg(s, "yang"), 8, noise_on=False)) for s in range(40)]
    assert set(rings) == {1, 2}                       # both rings, one per batch, never mixed (0)
    assert 10 <= rings.count(2) <= 30


@pytest.mark.parametrize("rule", ["reactgo", "reactanti"])
def test_yang_reaction_time_fixation_input_stays_on(rule):
    tr = generate_trials(rule, cfg(0, "yang"), 8, noise_on=False)
    assert (tr.x[-1, :, 0] == 1).all()                # fixation cue on at the last step
    assert (tr.y[-1, :, 0] < 0.5).all()               # but the fixation target is released


@pytest.mark.parametrize("rule", ["fdgo", "fdanti"])
def test_yang_go_stimulus_stays_on_through_response(rule):
    tr = generate_trials(rule, cfg(0, "yang"), 8, noise_on=False)
    assert np.abs(tr.x[-1, :, 1:1 + 2 * N]).sum(axis=1).min() > 0


@pytest.mark.parametrize("rule", GO_ANTI)
def test_v1_keeps_old_behaviour(rule):
    """Old checkpoints are evaluated with task_version 'v1': ring 1 only, old fix_in / stimulus offsets."""
    for s in range(10):
        assert ring_used(generate_trials(rule, cfg(s, "v1"), 8, noise_on=False)) == 1
    tr = generate_trials(rule, cfg(0, "v1"), 8, noise_on=False)
    if rule.startswith("react"):
        assert (tr.x[-1, :, 0] == 0).all()
    if rule.startswith("fd"):
        assert np.abs(tr.x[-1, :, 1:1 + 2 * N]).sum() == 0


def test_config_without_version_is_v1():
    c = default_config(n_eachring=N, seed=0)
    assert c["task_version"] == "yang"                 # new configs: fixed generators
    del c["task_version"]
    assert ring_used(generate_trials("fdgo", c, 8, noise_on=False)) == 1


def test_stim_requires_ring():
    tr = Trial(default_config(n_eachring=N), 10, 2)
    with pytest.raises(ValueError):
        tr.add("stim", np.zeros(2), ons=0)
