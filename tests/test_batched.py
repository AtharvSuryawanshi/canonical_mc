"""cmc.batched trains each network exactly like train_cog.train does.

P networks with different seeds and different cost settings (none / fixed lambdas /
rate budget / rate + wiring budget, with ramp and PI gain) are trained one by one
with ``train_cog.train`` and together with ``batched.train_batched`` on the same
trials, in float64, noise off. Losses, budget multipliers, costs and final weights
must agree to ~1e-9. Also: one network going NaN leaves the others unchanged.

    python -m pytest tests/test_batched.py      or      python tests/test_batched.py
"""

import numpy as np
import torch

import cmc.train_cog as tc
from cmc.batched import BatchedDaleRNN, PARAM_NAMES, train_batched
from cmc.task import default_config

TASKS = ("fdgo", "dm1", "dmsgo")
STEPS = 25
DATA_SEED = 7
GRAD_CLIP = 0.02  # below the ~0.06 gradient norm, so per-network clipping is exercised
BUDGET_KW = dict(budget_lr=0.05, budget_kp=0.5, budget_ramp=0.3, budget_rho=1.0,
                 budget_lambda_init=0.01, budget_ema=0.9)
SPECS = [
    {"seed": 0},
    {"seed": 1, "lambda_rate": 0.2, "lambda_connectivity": 2.0},
    {"seed": 2, "rate_budget": 0.01},
    {"seed": 3, "rate_budget": 0.02, "conn_budget": 0.004},
]


def _model(seed):
    cfg = default_config(n_eachring=8, seed=seed, easy_task=True)
    m = tc.make_dale_model(cfg, n_neurons=32, seed=seed).double()
    return m


def _single_runs():
    """train_cog.train per spec, all on the trial stream of config seed DATA_SEED."""
    recorded, out = [], []
    orig = tc.generate_mixed_trials

    def recording(*a, **k):
        trial = orig(*a, **k)
        recorded.append(trial)
        return trial

    for i, spec in enumerate(SPECS):
        tc.generate_mixed_trials = recording if i == 0 else orig
        model = _model(spec["seed"])
        cfg = default_config(n_eachring=8, seed=DATA_SEED, easy_task=True)
        kw = {k: v for k, v in spec.items() if k != "seed"}
        if "rate_budget" in kw or "conn_budget" in kw:
            kw.update(BUDGET_KW)
        try:
            hist = tc.train(model, cfg, active_tasks=TASKS, n_steps=STEPS, batch_size=12,
                            noise_level=0.0, grad_clip=GRAD_CLIP, log_every=5, plot_results=False, show_progress=False, **kw)
        finally:
            tc.generate_mixed_trials = orig
        out.append((model, hist))
    trials = [tc.trial_to_tensors(t, "cpu")[:3] for t in recorded]
    return out, trials


def _batched(trials, poison=None):
    models = [_model(s["seed"]) for s in SPECS]
    bmodel = BatchedDaleRNN(models)
    if poison is not None:
        with torch.no_grad():
            bmodel.w_out[poison] = float("nan")
    hists = train_batched(bmodel, None, TASKS, SPECS, n_steps=STEPS, noise_level=0.0, grad_clip=GRAD_CLIP, log_every=5,
                          trial_source=lambda step: trials[step - 1], **BUDGET_KW)
    return models, hists


def _close(a, b, what):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, (what, a.shape, b.shape)
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), (what, np.abs(a - b).max())


def test_batched_matches_single():
    singles, trials = _single_runs()
    assert len(trials) == STEPS
    models, hists = _batched(trials)
    for p, ((m_single, h_single), m_b, h_b) in enumerate(zip(singles, models, hists)):
        _close(h_b["loss"], h_single["loss"], f"loss net {p}")
        assert ("budget" in h_b) == ("budget" in h_single), p
        if "budget" in h_single:
            for key in ("rate", "conn"):
                for field in ("lambda", "cost", "budget_t"):
                    _close(h_b["budget"][key][field], h_single["budget"][key][field], f"{key} {field} net {p}")
        for name in PARAM_NAMES:
            _close(getattr(m_b, name).detach(), getattr(m_single, name).detach(), f"{name} net {p}")


def test_nan_network_is_isolated():
    _, trials = _single_runs()
    clean, _ = _batched(trials)
    poisoned, hists = _batched(trials, poison=1)
    assert hists[1]["diverged_step"] == 1
    for p in (0, 2, 3):
        assert hists[p]["diverged_step"] is None
        for name in PARAM_NAMES:
            _close(getattr(poisoned[p], name).detach(), getattr(clean[p], name).detach(), f"{name} net {p}")


if __name__ == "__main__":
    test_batched_matches_single()
    test_nan_network_is_isolated()
    print("ok")
