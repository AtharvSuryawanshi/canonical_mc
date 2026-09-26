"""Train Yang LeakyRNN or DaleRNN on fdgo / delaygo."""

import argparse
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from cmc.network import DaleRNN, LeakyRNN, _inv_softplus_torch
from cmc.paths import CHECKPOINTS_DIR
from cmc.task import default_config, generate_trials, generate_mixed_trials, rules_dict


# DaleRNN used to train with zero recurrent noise, so it never saw the
# perturbations that make redundant recurrent wiring worth paying for -- a
# noiseless objective reports wiring as nearly free.
DALE_DEFAULT_NOISE_LEVEL = 0.1


def trial_to_tensors(trial, device):
    """Convert a numpy Trial to model / loss tensors.

    Returns
    -------
    x_model : (batch, n_input, T)
    y : (batch, T, n_output)
    c_mask : (batch, T, n_output)
    y_loc : (batch, T)
    """
    x = torch.as_tensor(trial.x, device=device, dtype=torch.float32)
    y = torch.as_tensor(trial.y, device=device, dtype=torch.float32)
    c_mask = torch.as_tensor(trial.c_mask, device=device, dtype=torch.float32)
    y_loc = torch.as_tensor(trial.y_loc, device=device, dtype=torch.float32)
    x_model = x.permute(1, 2, 0).contiguous()
    y = y.permute(1, 0, 2).contiguous()
    c_mask = c_mask.permute(1, 0, 2).contiguous()
    y_loc = y_loc.permute(1, 0).contiguous()
    return x_model, y, c_mask, y_loc


def masked_mse(output, y, c_mask, per_trial=True):
    """Mask-weighted squared error on the ring readout.

    ``per_trial=True`` (default) divides each trial by its own total mask weight,
    so every trial contributes equally no matter how long it is. The old plain
    ``.mean()`` (``per_trial=False``) divided by ``B * T * n_output``, which makes
    a mixed batch weight each task in proportion to its trial duration: for
    sanity3 at batch 96, dmsgo received ~1.5x the gradient weight of fdgo purely
    because concat_trials right-pads the shorter tasks up to the longest tdim.

    Note the loss itself is already circular: the target is a Gaussian bump laid
    out on the ring with ``get_dist`` (circular distance), so a 1 deg error near
    180 deg costs the same as a 1 deg error near 0 deg. Only the *decoding* in
    batch_accuracy had a wrap bug.
    """
    err = c_mask * (output - y).square()
    if not per_trial:
        return err.mean()
    num = err.flatten(1).sum(dim=1)
    den = c_mask.flatten(1).sum(dim=1).clamp_min(1e-8)
    return (num / den).mean()


def _population_weighted(per_unit_cost, model, inh_scale):
    """Weight a per-unit cost by E/I population fraction.

    With ``inh_scale == 1`` this is exactly the plain per-unit mean, so it is
    invariant to frac_e and changes nothing by default. ``inh_scale < 1`` makes
    inhibitory activity cheaper, which is the knob for keeping a rate penalty
    from silencing the (unread-out, and therefore "free" to kill) I population
    first.
    """
    n_e = getattr(model, "n_e", None)
    n_i = getattr(model, "n_i", 0)
    if model is None or n_e is None or n_i == 0:
        return per_unit_cost.mean()
    n_rnn = n_e + n_i
    cost_e = per_unit_cost[..., :n_e].mean()
    cost_i = per_unit_cost[..., n_e:].mean()
    return (n_e / n_rnn) * cost_e + inh_scale * (n_i / n_rnn) * cost_i


def rate_reg(r_hist, model=None, kind="l2", inh_scale=1.0):
    """Metabolic / rate cost.

    ``kind="l1"`` is the closer analogue of metabolic cost: ATP per spike is
    roughly linear in firing rate. ``kind="l2"`` (the default, kept for
    continuity with earlier runs) barely penalizes a broadly active population
    and heavily penalizes a few high firers, which is a homeostatic rather than a
    metabolic cost.
    """
    if kind == "l1":
        per_unit = r_hist.abs()
    elif kind == "l2":
        per_unit = r_hist.square()
    else:
        raise ValueError(f"rate_reg: kind must be 'l1' or 'l2', got {kind!r}")
    return _population_weighted(per_unit, model, inh_scale)


def _w_rec_magnitude(model, inh_scale=1.0):
    """Normalized recurrent synaptic magnitudes plus the mask of counted synapses.

    Magnitudes are divided by ``ei_row_scale()`` so that E and I synapses feel
    equal *fractional* shrinkage pressure. Without this, _init_dale_w_rec's
    ``n_e / n_i`` balance scaling makes each inhibitory synapse ~3.9x larger than
    each excitatory one (measured at frac_e=0.8), so an unnormalized L1 spends
    half its budget on the 20% of rows that are inhibitory and prunes inhibition
    first -- which with ReLU units means runaway excitation.
    """
    w_rec = model.effective_w_rec()
    mag = w_rec.abs() / model.ei_row_scale()[:, None]
    # Autapses are structurally absent in DaleRNN, so they must not be counted.
    # LeakyRNN's fused kernel has a real, trainable diagonal, so it is counted.
    mask = getattr(model, "no_autapse", None)
    if mask is None:
        mask = torch.ones_like(mag)
    if inh_scale != 1.0:
        row_w = torch.ones(mag.shape[0], device=mag.device, dtype=mag.dtype)
        n_e = getattr(model, "n_e", None)
        if n_e is not None:
            row_w[n_e:] = inh_scale
        mask = mask * row_w[:, None]
    return mag, mask


def connectivity_reg(model, kind="l1", target="w_rec", inh_scale=1.0):
    """Wiring cost of the *recurrent* circuit.

    The hypothesis is about the cost of recurrent circuit wiring, so the penalty
    belongs on W_rec. It used to be applied to W_in (the input projection) with
    an L2 -- i.e. it penalized thalamocortical-style afferents and shrank them
    uniformly instead of pruning any. ``target="w_in"`` reproduces that legacy
    quantity for comparison with old sweeps.

    L1, not L2: L2 shrinks every synapse toward small-but-present, which is not
    what "fewer wires" means. L1 is the convex surrogate for a connection count.
    Note that for DaleRNN the magnitude is ``softplus(w_raw)``, which is strictly
    positive, so L1 alone can never reach exact zero -- see DaleRNN.prune_eps.

    Normalized per counted synapse, so lambda means "cost per synapse" and stays
    comparable if n_neurons changes.
    """
    if kind not in ("l1", "l2"):
        raise ValueError(f"connectivity_reg: kind must be 'l1' or 'l2', got {kind!r}")
    if target == "w_in":
        mag = model.w_in.abs()
        return mag.mean() if kind == "l1" else mag.square().mean()
    if target != "w_rec":
        raise ValueError(f"connectivity_reg: target must be 'w_rec' or 'w_in', got {target!r}")
    mag, mask = _w_rec_magnitude(model, inh_scale=inh_scale)
    term = mag if kind == "l1" else mag.square()
    return (term * mask).sum() / mask.sum().clamp_min(1.0)


def connection_fraction(model, thresh=1e-2):
    """Fraction of recurrent synapses that are effectively present.

    Reported, never trained on: L1 is a convex surrogate for a connection count,
    and this is the count itself. Magnitudes are E/I-normalized so the threshold
    means the same thing for both populations.
    """
    with torch.no_grad():
        mag, mask = _w_rec_magnitude(model)
        counted = mask > 0
        if not counted.any():
            return float("nan")
        return float(((mag > thresh) & counted).sum() / counted.sum())


def ring_prefs(config, device, dtype):
    n_eachring = config.get("n_eachring", config["n_neurons_per_ring"])
    return torch.arange(n_eachring, device=device, dtype=dtype) * (2 * math.pi / n_eachring)


def decode_ring(output, prefs):
    """Population-vector decode of ring channels. output: (B, T, 1+n_ring)."""
    ring = output[..., 1:]
    cos_c = (ring * torch.cos(prefs)).sum(dim=-1)
    sin_c = (ring * torch.sin(prefs)).sum(dim=-1)
    return torch.atan2(sin_c, cos_c)


def wrap_to_pi(delta):
    """Wrap an angle difference into (-pi, pi]: 181 deg vs 180 deg is a 1 deg error."""
    return torch.remainder(delta + math.pi, 2 * math.pi) - math.pi


# Backwards-compatible alias (used by exp.ipynb / exp_dale.ipynb).
circular_abs = wrap_to_pi


def circular_mean(angles, mask):
    """Mean direction over time: atan2(<sin>, <cos>) on the masked steps.

    An arithmetic mean of angles is wrong at the +-pi branch cut. decode_ring
    returns atan2 output in (-pi, pi], so for a target near pi the per-step
    decodes straddle the cut and average to ~0 -- i.e. 180 deg away. Verified:
    a network whose bump wobbled +-14 deg (well inside the 36 deg criterion) at
    *every* step scored acc=0.00 for targets in [3.00, 3.19] rad and acc=1.00
    just outside that band. That put a location-dependent floor on measured
    accuracy which grew with how jittery the network was, so it penalized
    high-lambda runs twice.

    angles, mask : (B, T). Returns (B,).
    """
    w = mask.to(angles.dtype)
    denom = w.sum(dim=1).clamp_min(1e-8)
    sin_m = (torch.sin(angles) * w).sum(dim=1) / denom
    cos_m = (torch.cos(angles) * w).sum(dim=1) / denom
    return torch.atan2(sin_m, cos_m)


def response_mask(trial, device):
    """(batch, T) bool mask of the graded response window.

    This is ``c_mask``'s post_ons region: every task sets
    ``check_ons = go_onset + 100 ms`` and ``add_c_mask(post_ons=check_ons)``, so
    the 100 ms reaction-time transient after go onset is deliberately ungraded by
    the loss. Accuracy must use the same window; scoring every step with
    y_loc >= 0 (the old behaviour) graded the network on exactly the transient
    the loss told it to ignore.
    """
    tdim = trial.tdim
    if getattr(trial, "post_ons", None) is None:
        mask = torch.zeros(trial.batch_size, tdim, dtype=torch.bool, device=device)
        mask[:, -1] = True
        return mask
    post = torch.as_tensor(np.asarray(trial.post_ons), device=device).long()
    steps = torch.arange(tdim, device=device)
    return steps[None, :] >= post[:, None]


def batch_accuracy(
    output, y_loc, prefs, resp_mask=None, ang_thresh=math.pi / 5, fix_thresh=0.5
):
    """Yang-style trial performance, scored on the graded response window.

    Criterion, matching ``network.get_perf``:
      * fixation-hold trials (no go target): correct iff still fixating;
      * go trials: correct iff the decoded location is within ``ang_thresh``
        **and** fixation has been released.

    The release requirement was previously missing -- fixation was only checked
    over the pre-go period, so a network that produced the right bump and held
    its fixation unit at 0.95 forever scored acc=1.0 (verified). That is a real
    loophole for dmsgo/dmsnogo/dmcgo/dmcnogo, where go-vs-nogo *is* the task.

    resp_mask : (batch, T) bool, from ``response_mask``. If None, falls back to
        the final time step alone, which is what ``network.get_perf`` uses.
    """
    batch_size, time_steps, _ = output.shape
    if resp_mask is None:
        resp_mask = torch.zeros(
            batch_size, time_steps, dtype=torch.bool, device=output.device
        )
        resp_mask[:, -1] = True

    pred_loc = decode_ring(output, prefs)
    go = (y_loc >= 0) & resp_mask
    has_go = go.any(dim=1)
    has_fix = ~has_go

    pred_go = circular_mean(pred_loc, go)
    target_loc = circular_mean(y_loc, go)
    ang_ok = wrap_to_pi(pred_go - target_loc).abs() < ang_thresh

    fix_out = output[..., 0]
    resp_counts = resp_mask.sum(dim=1).clamp_min(1)
    fix_resp = (fix_out * resp_mask).sum(dim=1) / resp_counts
    fixating = fix_resp > fix_thresh

    # go: right place AND fixation released. hold: still fixating.
    trial_ok = torch.where(has_go, ang_ok & ~fixating, fixating)

    def _mean(flags, sel):
        return flags[sel].float().mean().item() if sel.any() else float("nan")

    return {
        "acc": trial_ok.float().mean().item(),
        "go_acc": _mean(ang_ok & ~fixating, has_go),
        "fix_acc": _mean(fixating, has_fix),
        # Diagnostics: separates "saccaded to the wrong place" from "never let go".
        "loc_acc": _mean(ang_ok, has_go),
        "release_acc": _mean(~fixating, has_go),
    }


# Default regularizer shape. Kept in one place so training and the reported
# Pareto costs can never drift apart -- the front must plot the same functional
# that the gradient saw.
DEFAULT_REG = {
    "rate_kind": "l2",
    "rate_inh_scale": 1.0,
    "conn_kind": "l1",
    "conn_target": "w_rec",
    "conn_inh_scale": 1.0,
}


def _reg_opts(**overrides):
    opts = dict(DEFAULT_REG)
    unknown = set(overrides) - set(opts)
    if unknown:
        raise TypeError(f"unknown regularizer options: {sorted(unknown)}")
    opts.update({k: v for k, v in overrides.items() if v is not None})
    return opts


def eval_config_from(train_config, seed, easy_task=None):
    """Eval config that inherits the training config, with a fresh seeded RNG.

    Previously this was rebuilt from ``default_config()``, which silently
    discarded any non-default dt / tau / sigma_x / ruleset the model was actually
    trained with. Copying the training config keeps train and eval on the same
    task definition; only the RNG (and optionally easy_task) is swapped.
    """
    config = dict(train_config)
    if easy_task is not None:
        config["easy_task"] = bool(easy_task)
    config["rng"] = np.random.RandomState(int(seed))
    return config


def evaluate_task(model, config, rule, batch_size, device, noise_level=0.0):
    trial = generate_trials(rule, config, batch_size, noise_on=True)
    x, y, c_mask, y_loc = trial_to_tensors(trial, device)
    with torch.no_grad():
        r_hist, x_hist, output = model.simulate(x, noise_level=noise_level)
        loss = masked_mse(output, y, c_mask).item()
        prefs = ring_prefs(config, output.device, output.dtype)
        acc = batch_accuracy(output, y_loc, prefs, resp_mask=response_mask(trial, device))
        activity = model.activity_stats(r_hist, x_hist)
    return {"loss": loss, "activity": activity, **acc}


def evaluate_pareto_metrics(
    model,
    train_config,
    active_tasks,
    device,
    *,
    eval_seeds,
    batch_size,
    noise_level=0.0,
    input_noise=False,
    easy_task=True,
    n_eachring=None,
    conn_thresh=1e-2,
    **reg_overrides,
):
    """Seeded eval: accuracy (mean + min over tasks) and the two cost objectives.

    Costs use the same regularizer options as training, so the Pareto axes plot
    the quantity the gradient actually minimized.
    """
    device = torch.device(device)
    reg = _reg_opts(**reg_overrides)
    active_tasks = tuple(active_tasks)
    if n_eachring is not None:
        trained_ring = train_config.get("n_eachring")
        if trained_ring is not None and int(n_eachring) != int(trained_ring):
            raise ValueError(
                f"n_eachring={n_eachring} does not match the trained config "
                f"({trained_ring}); the model cannot consume that input size."
            )
    task_accs = {rule: [] for rule in active_tasks}
    task_losses = {rule: [] for rule in active_tasks}
    metabolic_costs = []

    with torch.no_grad():
        for seed_k in eval_seeds:
            eval_config = eval_config_from(train_config, seed_k, easy_task=easy_task)
            for rule in active_tasks:
                trial = generate_trials(
                    rule, eval_config, batch_size, noise_on=input_noise
                )
                x, y, c_mask, y_loc = trial_to_tensors(trial, device)
                r_hist, x_hist, output = model.simulate(x, noise_level=noise_level)
                task_losses[rule].append(masked_mse(output, y, c_mask).item())
                metabolic_costs.append(
                    rate_reg(
                        r_hist,
                        model=model,
                        kind=reg["rate_kind"],
                        inh_scale=reg["rate_inh_scale"],
                    ).item()
                )
                prefs = ring_prefs(eval_config, output.device, output.dtype)
                task_accs[rule].append(
                    batch_accuracy(
                        output, y_loc, prefs, resp_mask=response_mask(trial, device)
                    )["acc"]
                )

    per_task_mean_acc = {
        rule: float(np.mean(task_accs[rule])) for rule in active_tasks
    }
    mean_acc = float(np.mean(list(per_task_mean_acc.values())))
    min_task_acc = float(min(per_task_mean_acc.values()))
    task_loss = float(
        np.mean([np.mean(task_losses[rule]) for rule in active_tasks])
    )

    out = {
        "mean_acc": mean_acc,
        "min_task_acc": min_task_acc,
        "task_loss": task_loss,
        "metabolic_cost": float(np.mean(metabolic_costs)),
        "wiring_cost": float(
            connectivity_reg(
                model,
                kind=reg["conn_kind"],
                target=reg["conn_target"],
                inh_scale=reg["conn_inh_scale"],
            ).item()
        ),
        "conn_frac": connection_fraction(model, thresh=conn_thresh),
        "wiring_cost_w_in_l2": float(
            connectivity_reg(model, kind="l2", target="w_in").item()
        ),
    }
    for rule, acc in per_task_mean_acc.items():
        out[f"acc_{rule}"] = acc
    return out


def evaluate_objectives(model, config, active_tasks, batch_size, device, noise_level=0.0):
    """Single-pass eval (legacy). Prefer evaluate_pareto_metrics for Pareto sweeps."""
    device = torch.device(device)
    task_losses = []
    metabolic_costs = []
    accs = []
    with torch.no_grad():
        for rule in active_tasks:
            trial = generate_trials(rule, config, batch_size, noise_on=False)
            x, y, c_mask, y_loc = trial_to_tensors(trial, device)
            r_hist, x_hist, output = model.simulate(x, noise_level=noise_level)
            task_losses.append(masked_mse(output, y, c_mask).item())
            metabolic_costs.append(rate_reg(r_hist, model=model).item())
            prefs = ring_prefs(config, output.device, output.dtype)
            accs.append(
                batch_accuracy(
                    output, y_loc, prefs, resp_mask=response_mask(trial, device)
                )["acc"]
            )
    return {
        "task_loss": float(np.mean(task_losses)),
        "metabolic_cost": float(np.mean(metabolic_costs)),
        "wiring_cost": float(connectivity_reg(model).item()),
        "mean_acc": float(np.mean(accs)),
        "min_task_acc": float(min(accs)) if accs else float("nan"),
    }


def penalty_scales(
    model,
    config,
    active_tasks,
    batch_size=64,
    device="cpu",
    noise_level=0.0,
    loss_per_trial=True,
    **reg_overrides,
):
    """Magnitude of each loss term at the current weights, plus break-even lambdas.

    lambda_x is meaningful only relative to the task loss: the interesting range
    brackets lambda_x * cost_x == task_loss. Both of these moved in this round of
    fixes -- masked_mse now normalizes per trial (task loss ~2x smaller) and the
    wiring cost is now L1 on W_rec instead of L2 on W_in (a different quantity
    entirely) -- so the old calibrated ranges no longer apply. Run this on a
    freshly initialized model to pick new ones.
    """
    device = torch.device(device)
    reg = _reg_opts(**reg_overrides)
    task_losses, rate_costs = [], []
    with torch.no_grad():
        for rule in active_tasks:
            trial = generate_trials(rule, config, batch_size, noise_on=True)
            x, y, c_mask, _ = trial_to_tensors(trial, device)
            r_hist, _, output = model.simulate(x, noise_level=noise_level)
            task_losses.append(
                masked_mse(output, y, c_mask, per_trial=loss_per_trial).item()
            )
            rate_costs.append(
                rate_reg(
                    r_hist,
                    model=model,
                    kind=reg["rate_kind"],
                    inh_scale=reg["rate_inh_scale"],
                ).item()
            )
        wiring = connectivity_reg(
            model,
            kind=reg["conn_kind"],
            target=reg["conn_target"],
            inh_scale=reg["conn_inh_scale"],
        ).item()

    task_loss = float(np.mean(task_losses))
    rate_cost = float(np.mean(rate_costs))
    out = {
        "task_loss": task_loss,
        "metabolic_cost": rate_cost,
        "wiring_cost": float(wiring),
        "conn_frac": connection_fraction(model),
        "lambda_rate_breakeven": task_loss / rate_cost if rate_cost > 0 else float("inf"),
        "lambda_connectivity_breakeven": (
            task_loss / wiring if wiring > 0 else float("inf")
        ),
        "reg": dict(reg),
    }
    return out


def make_yang_model(
    config,
    n_rnn=256,
    activation="relu",
    w_rec_init="randortho",
    sigma_rec=0.05,
    seed=0,
    device="cpu",
):
    # `seed` alone must determine the network. numpy's RandomState(seed) covers
    # w_in/w_rec, but w_out and the per-step training noise come from torch's
    # global generator, which nothing seeded: the same command gave different
    # networks on every run, and each one depended on what was trained before it.
    torch.manual_seed(int(seed))
    model = LeakyRNN(
        n_input=config["n_input"],
        n_rnn=n_rnn,
        n_output=config["n_output"],
        alpha=config["alpha"],
        activation=activation,
        w_rec_init=w_rec_init,
        sigma_rec=sigma_rec,
        seed=seed,
    )
    print(f"Yang LeakyRNN: n_rnn={n_rnn}  alpha={config['alpha']:.3f}  activation={activation}")
    return model.to(device)


def make_dale_model(
    config,
    n_neurons=128,
    frac_e=0.8,
    g=1.0,
    activation="relu",
    w_rec_init="randortho",
    sigma_rec=0.05,
    prune_eps=0.0,
    seed=0,
    device="cpu",
):
    # `seed` alone must determine the network. numpy's RandomState(seed) covers
    # w_in/w_rec, but w_out and the per-step training noise come from torch's
    # global generator, which nothing seeded: the same command gave different
    # networks on every run, and each one depended on what was trained before it.
    torch.manual_seed(int(seed))
    model = DaleRNN(
        n_input=config["n_input"],
        n_rnn=n_neurons,
        n_output=config["n_output"],
        alpha=config["alpha"],
        activation=activation,
        w_rec_init=w_rec_init,
        sigma_rec=sigma_rec,
        frac_e=frac_e,
        target_rho=g,
        prune_eps=prune_eps,
        seed=seed,
    )
    rho = model.recurrent_spectral_radius()
    print(f"DaleRNN: N={n_neurons}  frac_e={frac_e}  rho(W)={rho:.3f}  activation={activation}")
    return model.to(device)


def make_model(config, model_type="yang", device="cpu", **kwargs):
    if model_type == "yang":
        return make_yang_model(config, device=device, **kwargs)
    if model_type == "dale":
        return make_dale_model(config, device=device, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")


def _serialize_config(config):
    """Task/config dict without the non-serializable RNG."""
    return {k: v for k, v in config.items() if k != "rng"}


def _restore_config(config_dict, seed=0):
    """Rebuild config for trial generation, including a fresh RNG."""
    config = dict(config_dict)
    config["rng"] = np.random.RandomState(seed)
    return config


def save_checkpoint(
    path,
    model,
    config,
    model_type,
    active_tasks,
    *,
    model_kwargs=None,
    seed=0,
    train_steps=None,
    history=None,
    extra=None,
    verbose=True,
):
    """Save trained weights plus metadata needed to reload and evaluate.

    ``extra`` is merged into the checkpoint dict (e.g. lambdas, eval metrics).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": 1,
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "config": _serialize_config(config),
        "active_tasks": list(active_tasks),
        "model_kwargs": dict(model_kwargs or {}),
        "seed": seed,
    }
    if train_steps is not None:
        checkpoint["train_steps"] = train_steps
    if history is not None:
        checkpoint["history"] = history
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    if verbose:
        print(f"Saved checkpoint -> {path.resolve()}")
    return path


def load_checkpoint(path, device="cpu", eval_mode=True):
    """Reload a model saved with ``save_checkpoint``.

    Returns
    -------
    model, config, checkpoint
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    device = torch.device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    seed = checkpoint.get("seed", 0)
    config = _restore_config(checkpoint["config"], seed=seed)
    model = make_model(
        config,
        model_type=checkpoint["model_type"],
        device=str(device),
        **checkpoint.get("model_kwargs", {}),
    )
    model.load_state_dict(checkpoint["state_dict"])
    if eval_mode:
        model.eval()
    print(
        f"Loaded checkpoint <- {path.resolve()}  "
        f"model={checkpoint['model_type']}  tasks={checkpoint.get('active_tasks')}"
    )
    return model, config, checkpoint


def sample_input_drive(model, config, task="fdgo", batch_size=8, device="cpu", t_step=None):
    """W_in @ x + b at one timestep (pre-recurrence drive) for Dale model diagnostics."""
    device = torch.device(device)
    trial = generate_trials(task, config, batch_size, noise_on=False)
    x, _, _, _ = trial_to_tensors(trial, device)
    if t_step is None:
        stim = trial.epochs.get("stim1", (0, trial.tdim // 2))
        t_step = int((stim[0] or 0) + max(1, ((stim[1] or trial.tdim) - (stim[0] or 0)) // 2))
        t_step = min(t_step, trial.tdim - 1)
    with torch.no_grad():
        x_t = x[:, :, t_step]
        # w_in is (n_input, n_rnn) for both models; the old code used a
        # nonexistent model.W_in / model.b and a transpose that would not align.
        drive = x_t @ model.w_in + model.bias
    return drive.detach().cpu().numpy().ravel(), t_step


def plot_preactivation_at_init(model, config, device="cpu", task="fdgo"):
    """Histogram of input drive at init and the rates it produces (both models).

    The old guard tested ``hasattr(model, "firing_rate")`` and referenced
    ``model.rate_max``; neither exists on DaleRNN or LeakyRNN, so this always
    printed "skipped" and never ran.
    """
    drive, t_step = sample_input_drive(model, config, task=task, device=device)
    param_device = next(model.parameters()).device
    with torch.no_grad():
        rates = (
            model._cell_act(torch.as_tensor(drive, device=param_device))
            .detach()
            .cpu()
            .numpy()
        )
    silent_thresh = 0.05  # matches activity_stats' frac_silent threshold

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(drive, bins=40, color="C0", alpha=0.85, edgecolor="white")
    axes[0].axvline(0.0, color="k", ls="--", lw=1)
    axes[0].set_xlabel("W_in @ x + b")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Input drive at t={t_step} ({task})")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(rates, bins=40, color="C1", alpha=0.85, edgecolor="white")
    axes[1].axvline(silent_thresh, color="k", ls="--", lw=1, label=f"silent<{silent_thresh:.1f}")
    axes[1].set_xlabel("firing rate")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"Rates from drive (frac silent={float((rates < silent_thresh).mean()):.2f})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def _project_l1_ball(z, radius):
    """Euclidean projection of nonnegative ``z`` onto {z >= 0, sum(z) <= radius}.

    Soft-thresholding by the one tau that makes the sum hit the radius (Duchi et
    al. 2008, sort-based, exact). No-op if already inside.
    """
    total = z.sum()
    if total <= radius * (1.0 + 1e-5):  # inside, up to softplus round-off
        return z
    u, _ = torch.sort(z.flatten(), descending=True)
    css = torch.cumsum(u, dim=0) - radius
    j = torch.arange(1, u.numel() + 1, device=z.device, dtype=z.dtype)
    rho = int(torch.nonzero(u - css / j > 0).max()) + 1
    tau = css[rho - 1] / rho
    return torch.clamp(z - tau, min=0.0)


def project_w_rec_to_budget(model, budget, inh_scale=1.0):
    """Shrink W_rec in place so that ``connectivity_reg(model, "l1") <= budget``.

    The wiring cost is ``sum(mask * |W| / row_scale) / sum(mask)``, so the
    budget is an L1 ball in the normalized coordinates z = mask * |W| / row_scale.
    Projecting there soft-thresholds every synapse by the same *normalized*
    amount -- the same equal-fractional pressure on E and I rows the penalty
    applies. Synapses thresholded to zero become softplus(w_raw) ~ 1e-8, i.e.
    pruned. Only for DaleRNN with prune_eps == 0 (``w_raw`` -> magnitude is then
    exactly softplus).
    """
    if not hasattr(model, "w_raw"):
        raise ValueError("project_w_rec_to_budget: needs DaleRNN (softplus magnitudes)")
    if getattr(model, "prune_eps", 0.0) > 0.0:
        raise ValueError("project_w_rec_to_budget: prune_eps must be 0")
    with torch.no_grad():
        mag, mask = _w_rec_magnitude(model, inh_scale=inh_scale)
        z = mag * mask
        z_new = _project_l1_ball(z, budget * mask.sum().clamp_min(1.0))
        if z_new is z:
            return
        counted = mask > 0
        new_abs = torch.where(counted, z_new / mask.clamp_min(1e-12), mag) * model.ei_row_scale()[:, None]
        model.w_raw.copy_(_inv_softplus_torch(new_abs).to(model.w_raw.dtype))


BUDGET_DEFAULTS = {
    "budget_lr": 0.01,
    "budget_kp": 0.0,
    "budget_ramp": 0.0,
    "budget_rho": 1.0,
    "budget_lambda_init": 0.01,
    "budget_ema": 0.9,
    "conn_budget_mode": "projection",
}


class BudgetConstraint:
    """Keep a cost at or below a budget: ``cost <= budget``.

    Instead of a hand-set weight, the weight ``lam`` is learned during training
    (augmented Lagrangian, i.e. a multiplier plus a quadratic penalty). With
    the relative violation ``v = cost / budget_t - 1`` the loss term is

        lam * cost  +  (rho / 2) * budget_t * relu(v)**2

    and ``lam`` is a PI controller on log(lam), stepped after every optimizer step:

        integral += lr * clip(ema(v), -1, 1)
        log(lam)  = integral + kp * clip(ema(v), -1, 1)

    Over budget the weight grows, under budget it shrinks, so the cost is not
    driven below what the budget asks for. Log space because the weights that
    matter span decades (the weighted-sum sweeps used 0.02 - 40); the clip and
    the EMA tame the large violation at init and the per-batch noise.

    Lag is the failure mode: wiring cost falls only as fast as the weights
    shrink, so a pure integral (kp=0) keeps raising lam while the cost is still
    catching up, overshoots by ~100x and prunes the network to death. Two
    remedies (Stooke et al. 2020, PID Lagrangian): the proportional term reacts
    to the current violation without accumulating it, and ``ramp`` anneals the
    budget geometrically from the cost at step 1 to the target over that
    fraction of training, so the violation never gets large.

    The quadratic term only acts above budget; it is what lets an augmented
    Lagrangian settle on non-convex parts of the front, where a fixed weight (a
    weighted sum) cannot.
    """

    LAM_MIN, LAM_MAX = 1e-6, 1e4

    def __init__(self, budget, n_steps=None, lr=0.01, kp=0.0, ramp=0.0, rho=1.0,
                 lambda_init=0.01, ema=0.9):
        budget = float(budget)
        if not budget > 0:
            raise ValueError(f"BudgetConstraint: budget must be > 0, got {budget}")
        self.budget = budget
        self.lr = float(lr)
        self.kp = float(kp)
        self.ramp_steps = int(round(float(ramp) * n_steps)) if ramp and n_steps else 0
        self.rho = float(rho)
        self.ema = float(ema)
        self.integral = float(np.log(np.clip(lambda_init, self.LAM_MIN, self.LAM_MAX)))
        self.log_lam = self.integral
        self.v_ema = None
        self.start_cost = None
        self.step = 0
        self.last_cost = float("nan")

    @property
    def lam(self):
        return float(np.exp(self.log_lam))

    @property
    def current_budget(self):
        """The target, or during the ramp a geometric step from the initial cost."""
        if self.step >= self.ramp_steps or self.start_cost is None or self.start_cost <= self.budget:
            return self.budget
        frac = self.step / self.ramp_steps
        return float(self.start_cost ** (1.0 - frac) * self.budget ** frac)

    def penalty(self, cost):
        """Loss term for the current batch; ``cost`` is a differentiable scalar."""
        self.last_cost = float(cost.detach())
        if self.start_cost is None:
            self.start_cost = self.last_cost
        budget_t = self.current_budget
        v = cost / budget_t - 1.0
        return self.lam * cost + 0.5 * self.rho * budget_t * torch.relu(v).square()

    def update(self):
        """Dual step; call once per optimizer step, after penalty()."""
        v = self.last_cost / self.current_budget - 1.0
        self.v_ema = v if self.v_ema is None else self.ema * self.v_ema + (1 - self.ema) * v
        e = float(np.clip(self.v_ema, -1.0, 1.0))
        lo, hi = np.log(self.LAM_MIN), np.log(self.LAM_MAX)
        self.integral = float(np.clip(self.integral + self.lr * e, lo, hi))
        self.log_lam = float(np.clip(self.integral + self.kp * e, lo, hi))
        self.step += 1


def train(
    model,
    config,
    active_tasks=("fdgo",),
    n_steps=200,
    batch_size=32,
    lr=1e-3,
    lambda_rate=0.0,
    lambda_connectivity=0.0,
    noise_level=0.0,
    grad_clip=1.0,
    log_every=10,
    eval_batch_size=64,
    mixed_batch=True,
    plot_results=True,
    show_progress=True,
    loss_per_trial=True,
    rate_budget=None,
    conn_budget=None,
    budget_lr=BUDGET_DEFAULTS["budget_lr"],
    budget_kp=BUDGET_DEFAULTS["budget_kp"],
    budget_ramp=BUDGET_DEFAULTS["budget_ramp"],
    budget_rho=BUDGET_DEFAULTS["budget_rho"],
    budget_lambda_init=BUDGET_DEFAULTS["budget_lambda_init"],
    budget_ema=BUDGET_DEFAULTS["budget_ema"],
    conn_budget_mode=BUDGET_DEFAULTS["conn_budget_mode"],
    **reg_overrides,
):
    """Multitask training. By default each batch mixes all active tasks.

    Cost terms come in two forms, per cost (rate / wiring), not both at once:

    * fixed weight: ``lambda_rate`` / ``lambda_connectivity`` (weighted sum);
    * budget: ``rate_budget`` / ``conn_budget``, a ceiling on the same quantity
      (``rate_reg`` / ``connectivity_reg``), enforced by an augmented Lagrangian
      whose multiplier adapts during training -- see ``BudgetConstraint``.
      The wiring budget is by default enforced exactly instead
      (``conn_budget_mode="projection"``): after every optimizer step W_rec is
      projected back onto the budget (``project_w_rec_to_budget``), so it has
      no multiplier and no lag. Rate depends on activity, not on the weights,
      so it can only use the multiplier.

    With both budgets ``None`` (the default) training is exactly the
    weighted-sum training it always was.
    """
    reg = _reg_opts(**reg_overrides)
    if rate_budget is not None and lambda_rate:
        raise ValueError("train: give lambda_rate or rate_budget, not both")
    if conn_budget is not None and lambda_connectivity:
        raise ValueError("train: give lambda_connectivity or conn_budget, not both")
    budget_kw = dict(lr=budget_lr, kp=budget_kp, ramp=budget_ramp, rho=budget_rho,
                     lambda_init=budget_lambda_init, ema=budget_ema)
    rate_con = BudgetConstraint(rate_budget, n_steps, **budget_kw) if rate_budget is not None else None
    if conn_budget_mode not in ("projection", "lagrangian"):
        raise ValueError(f"train: conn_budget_mode must be 'projection' or 'lagrangian', got {conn_budget_mode!r}")
    conn_con = conn_proj = None
    if conn_budget is not None and conn_budget_mode == "lagrangian":
        conn_con = BudgetConstraint(conn_budget, n_steps, **budget_kw)
    elif conn_budget is not None:
        if reg["conn_kind"] != "l1" or reg["conn_target"] != "w_rec":
            raise ValueError("train: projection needs conn_kind='l1', conn_target='w_rec'")
        # Only its ramp / budget bookkeeping is used; there is no multiplier.
        conn_proj = BudgetConstraint(conn_budget, n_steps, **budget_kw)
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "loss": [],
        "step": [],
        "eval_step": [],
        "frac_silent": [],
        "frac_saturated": [],
        "per_task": {
            task: {
                "loss": [],
                "acc": [],
                "go_acc": [],
                "fix_acc": [],
                "loc_acc": [],
                "release_acc": [],
            }
            for task in active_tasks
        },
        "reg": dict(reg),
        "loss_per_trial": bool(loss_per_trial),
    }
    if rate_con is not None or conn_budget is not None:
        history["budget"] = {
            "rate_budget": rate_budget,
            "conn_budget": conn_budget,
            "conn_budget_mode": conn_budget_mode,
            **budget_kw,
            "rate": {"lambda": [], "cost": [], "budget_t": []},
            "conn": {"lambda": [], "cost": [], "budget_t": []},
        }

    steps = tqdm(range(1, n_steps + 1)) if show_progress else range(1, n_steps + 1)
    for step in steps:
        if mixed_batch and len(active_tasks) > 1:
            trial = generate_mixed_trials(active_tasks, config, batch_size, noise_on=True)
        else:
            rule = str(config["rng"].choice(active_tasks))
            trial = generate_trials(rule, config, batch_size, noise_on=True)
        x, y, c_mask, y_loc = trial_to_tensors(trial, device)

        def loss_fn(r_hist, x_hist, output_matrix, y=y, c_mask=c_mask):
            total = masked_mse(output_matrix, y, c_mask, per_trial=loss_per_trial)
            if lambda_rate:
                total = total + lambda_rate * rate_reg(
                    r_hist,
                    model=model,
                    kind=reg["rate_kind"],
                    inh_scale=reg["rate_inh_scale"],
                )
            if lambda_connectivity:
                total = total + lambda_connectivity * connectivity_reg(
                    model,
                    kind=reg["conn_kind"],
                    target=reg["conn_target"],
                    inh_scale=reg["conn_inh_scale"],
                )
            if rate_con is not None:
                total = total + rate_con.penalty(
                    rate_reg(
                        r_hist,
                        model=model,
                        kind=reg["rate_kind"],
                        inh_scale=reg["rate_inh_scale"],
                    )
                )
            if conn_con is not None:
                total = total + conn_con.penalty(
                    connectivity_reg(
                        model,
                        kind=reg["conn_kind"],
                        target=reg["conn_target"],
                        inh_scale=reg["conn_inh_scale"],
                    )
                )
            return total

        loss = model.train_step(
            optimizer,
            x,
            loss_fn,
            noise_level=noise_level,
            grad_clip=grad_clip,
        )
        history["loss"].append(loss.item())
        history["step"].append(step)
        if conn_proj is not None:
            if conn_proj.start_cost is None:
                with torch.no_grad():
                    conn_proj.start_cost = float(connectivity_reg(model, inh_scale=reg["conn_inh_scale"]))
            budget_t = conn_proj.current_budget
            project_w_rec_to_budget(model, budget_t, inh_scale=reg["conn_inh_scale"])
            with torch.no_grad():
                cost = float(connectivity_reg(model, inh_scale=reg["conn_inh_scale"]))
            conn_proj.step += 1
            history["budget"]["conn"]["budget_t"].append(budget_t)
            history["budget"]["conn"]["lambda"].append(float("nan"))
            history["budget"]["conn"]["cost"].append(cost)
        for key, con in (("rate", rate_con), ("conn", conn_con)):
            if con is not None:
                history["budget"][key]["budget_t"].append(con.current_budget)
                con.update()
                history["budget"][key]["lambda"].append(con.lam)
                history["budget"][key]["cost"].append(con.last_cost)

        if step == 1 or step % log_every == 0 or step == n_steps:
            history["eval_step"].append(step)
            parts = [f"step {step:4d} | train {loss.item():.4f}"]
            if mixed_batch and len(active_tasks) > 1:
                parts[-1] += " (mixed)"
            acts = []
            for task in active_tasks:
                metrics = evaluate_task(model, config, task, eval_batch_size, device)
                for key in ("loss", "acc", "go_acc", "fix_acc", "loc_acc", "release_acc"):
                    history["per_task"][task][key].append(metrics[key])
                parts.append(
                    f"{task} accuracy:{metrics['acc']:.3f} "
                    f"go:{metrics['go_acc']:.2f} fix:{metrics['fix_acc']:.2f} "
                    f"loss:{metrics['loss']:.4f}"
                )
                acts.append(metrics["activity"])
            # Average over tasks: this used to record only the last task's stats.
            frac_silent = float(np.mean([a["frac_silent"] for a in acts]))
            frac_saturated = float(np.mean([a["frac_saturated"] for a in acts]))
            history["frac_silent"].append(frac_silent)
            history["frac_saturated"].append(frac_saturated)
            parts.append(f"sat:{frac_saturated:.2f}; silent:{frac_silent:.2f}")
            # print(" | ".join(parts))

    if plot_results:
        plot_training(history, active_tasks)
    return history


def train_without_plots(*args, **kwargs):
    kwargs["plot_results"] = False
    return train(*args, **kwargs)


def plot_training(history, active_tasks):
    """1x2: per-task loss and accuracy vs training step."""
    xs = history["eval_step"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for task in active_tasks:
        metrics = history["per_task"][task]
        axes[0].plot(xs, metrics["loss"], marker="o", label=task)
        axes[1].plot(xs, metrics["acc"], marker="o", label=task)
    axes[0].set_xlabel("steps")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss")
    axes[1].set_xlabel("steps")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig


def _example_trial(model, config, task):
    """One noiseless trial: input channels and hidden rates."""
    device = next(model.parameters()).device
    trial = generate_trials(task, config, batch_size=1, noise_on=False)
    x, _, _, _ = trial_to_tensors(trial, device)
    with torch.no_grad():
        r_hist, _, _ = model.simulate(x, noise_level=0.0)
    x_np = x[0].detach().cpu().numpy()
    r_np = r_hist[0].detach().cpu().numpy().T
    return x_np, r_np


def plot_example_runs(model, config, active_tasks):
    """One column per task: input traces + hidden activity imshow."""
    n_tasks = len(active_tasks)
    fig, axes = plt.subplots(
        2,
        n_tasks,
        figsize=(4.8 * n_tasks, 6.8),
        sharex="col",
        layout="constrained",
        gridspec_kw={"height_ratios": [1.0, 2.6]},
        squeeze=False,
    )
    rule_start = config["rule_start"]
    examples = [_example_trial(model, config, task) for task in active_tasks]
    rate_max = max(r.max() for _, r in examples)
    rate_max = 1.0 if rate_max <= 0 else rate_max

    for col, (task, (x_np, r_np)) in enumerate(zip(active_tasks, examples)):
        t = np.arange(x_np.shape[1])
        cue = 1.0 - x_np[0]
        stim = x_np[1:rule_start]
        context = x_np[rule_start:]

        ax_in = axes[0, col]
        ax_r = axes[1, col]
        ax_in.plot(t, cue, color="k", lw=1.6, label="cue (go)")
        ax_in.plot(t, stim.max(axis=0), color="C0", lw=1.4, label="input")
        ax_in.plot(t, context.max(axis=0), color="C1", lw=1.4, label="context")
        ax_in.set_ylim(-0.05, 1.15)
        ax_in.set_title(task)
        ax_in.grid(True, alpha=0.3)
        if col == 0:
            ax_in.set_ylabel("inputs")
            ax_in.legend(loc="upper right", fontsize=8, framealpha=0.9)

        im = ax_r.imshow(
            r_np,
            aspect="auto",
            interpolation="nearest",
            cmap="viridis",
            origin="upper",
            vmin=0.0,
            vmax=rate_max,
            extent=(-0.5, r_np.shape[1] - 0.5, r_np.shape[0] - 0.5, -0.5),
        )
        if hasattr(model, "n_e"):
            ax_r.axhline(model.n_e - 0.5, color="w", ls="--", lw=0.8)
        ax_r.set_xlabel("steps")
        if col == 0:
            ax_r.set_ylabel("neuron")

    fig.colorbar(im, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.02, label="activity")
    plt.show()
    return fig


def default_save_path(model_type, tasks, n_steps):
    """checkpoints/{model}_{n_tasks}_{n_steps}_{YYYY_MM_DD_HH_MM_SS}.pt"""
    n_tasks = len(tasks)
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return CHECKPOINTS_DIR / f"{model_type}_{n_tasks}_{n_steps}_{stamp}.pt"


def resolve_active_tasks(task_battery, tasks):
    """Return task tuple from explicit --tasks or a preset battery."""
    if tasks is not None:
        return tuple(tasks)
    return tuple(rules_dict[task_battery])


def parse_args():
    parser = argparse.ArgumentParser(description="Train Yang LeakyRNN or DaleRNN on fdgo / delaygo")
    parser.add_argument(
        "--model",
        choices=["yang", "dale"],
        default="dale",
        help="Model backend: DaleRNN (default) or Yang LeakyRNN.",
    )
    parser.add_argument(
        "--task-battery",
        choices=["all", "core5", "sanity3"],
        default="all",
        help=(
            "Preset task subset: all (20 tasks, default), "
            "core5 (fdgo/fdanti/dm1/contextdm1/dmsgo, major demand types), "
            "sanity3 (fdgo/contextdm1/dmsgo, bare-minimum smoke test). "
            "Ignored when --tasks is set."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=rules_dict["all"],
        help="Explicit task list (overrides --task-battery).",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-rnn", type=int, default=256, help="Hidden units (Yang LeakyRNN).")
    parser.add_argument("--n-neurons", type=int, default=256, help="Hidden units (DaleRNN).")
    parser.add_argument("--n-eachring", type=int, default=16)
    parser.add_argument("--frac-e", type=float, default=0.8)
    parser.add_argument("--g", type=float, default=1.0, help="Target rho(W) for DaleRNN.")
    parser.add_argument("--sigma-rec", type=float, default=0.05, help="Recurrent noise scale (both models).")
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Recurrent noise multiplier on gate (actual std = noise_level * sigma_rec scaled). "
        "Default: 1.0 for yang, 0.1 for dale.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--lambda-rate", type=float, default=0.0)
    parser.add_argument("--lambda-connectivity", type=float, default=0.0)
    parser.add_argument(
        "--rate-kind",
        choices=["l1", "l2"],
        default="l2",
        help="Metabolic cost norm on firing rates. l1 is closer to ATP-per-spike; "
        "l2 (default) is kept for continuity with earlier runs.",
    )
    parser.add_argument(
        "--rate-inh-scale",
        type=float,
        default=1.0,
        help="Extra weight on the inhibitory population in the rate cost. "
        "1.0 (default) == plain per-unit mean; <1 protects I units from being "
        "silenced first (they carry no readout cost, so they are cheap to kill).",
    )
    parser.add_argument(
        "--conn-kind",
        choices=["l1", "l2"],
        default="l1",
        help="Wiring cost norm on W_rec. L1 (default) is the convex surrogate "
        "for a connection count; L2 only shrinks.",
    )
    parser.add_argument(
        "--conn-target",
        choices=["w_rec", "w_in"],
        default="w_rec",
        help="Which matrix the wiring cost penalizes. w_rec (default) is the "
        "recurrent circuit; w_in reproduces the old (incorrect) input-projection "
        "penalty for comparison.",
    )
    parser.add_argument(
        "--conn-inh-scale",
        type=float,
        default=1.0,
        help="Extra weight on inhibitory rows of the wiring cost, applied on top "
        "of the n_e/n_i magnitude normalization. 1.0 == equal cost per synapse.",
    )
    parser.add_argument(
        "--prune-eps",
        type=float,
        default=0.0,
        help="DaleRNN only: hard-threshold synaptic magnitudes at this value in "
        "the forward pass, giving exact zeros. 0.0 (default) = off. Pruning is "
        "irreversible -- a synapse below eps has zero gradient and never returns.",
    )
    parser.add_argument(
        "--loss-per-trial",
        dest="loss_per_trial",
        action="store_true",
        default=True,
        help="Normalize the task loss per trial so long tasks are not favoured "
        "(default).",
    )
    parser.add_argument(
        "--no-loss-per-trial",
        dest="loss_per_trial",
        action="store_false",
        help="Legacy global .mean() task loss (weights tasks by trial duration).",
    )
    parser.add_argument(
        "--report-scales",
        action="store_true",
        help="Print the magnitude of each loss term and the break-even lambdas at "
        "init, then exit without training.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--plot-results", type=bool, default=False)
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help=(
            "Path to save trained weights after training. "
            "Default: checkpoints/{model}_{n_tasks}_{n_steps}_{date-time}.pt"
        ),
    )
    parser.add_argument(
        "--load-path",
        type=str,
        default=None,
        help="Load weights from a checkpoint before training (optional resume / eval-only).",
    )
    return parser.parse_args()


def _model_kwargs_from_args(args):
    if args.model == "yang":
        return {"n_rnn": args.n_rnn, "sigma_rec": args.sigma_rec, "seed": args.seed}
    return {
        "n_neurons": args.n_neurons,
        "frac_e": args.frac_e,
        "g": args.g,
        "sigma_rec": args.sigma_rec,
        "prune_eps": getattr(args, "prune_eps", 0.0),
        "seed": args.seed,
    }


def main():
    args = parse_args()
    active_tasks = resolve_active_tasks(args.task_battery, args.tasks)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = _model_kwargs_from_args(args)

    if args.load_path:
        model, config, _ = load_checkpoint(args.load_path, device=device, eval_mode=False)
        size_str = (
            f"n_rnn={model.n_rnn}"
            if args.model == "yang"
            else f"N={model.n_rnn}"
        )
    else:
        config = default_config(n_eachring=args.n_eachring, seed=args.seed, easy_task=True)
        if args.model == "yang":
            model = make_yang_model(config, device=device, **model_kwargs)
            size_str = f"n_rnn={args.n_rnn}"
        else:
            model = make_dale_model(config, device=device, **model_kwargs)
            size_str = f"N={args.n_neurons}"

    noise_level = (
        args.noise_level
        if args.noise_level is not None
        else (1.0 if args.model == "yang" else DALE_DEFAULT_NOISE_LEVEL)
    )

    battery_label = args.task_battery if args.tasks is None else "custom"
    print(
        f"model={args.model}  device={device}  {size_str}  "
        f"n_in={config['n_input']}  n_out={config['n_output']}  "
        f"task_battery={battery_label}  tasks={active_tasks}  "
        f"noise_level={noise_level}"
    )
    reg_opts = {
        "rate_kind": args.rate_kind,
        "rate_inh_scale": args.rate_inh_scale,
        "conn_kind": args.conn_kind,
        "conn_target": args.conn_target,
        "conn_inh_scale": args.conn_inh_scale,
    }

    if args.report_scales:
        scales = penalty_scales(
            model,
            config,
            active_tasks,
            batch_size=args.batch_size,
            device=device,
            noise_level=noise_level,
            loss_per_trial=args.loss_per_trial,
            **reg_opts,
        )
        print()
        print("Loss-term magnitudes at init:")
        for key in ("task_loss", "metabolic_cost", "wiring_cost", "conn_frac"):
            print(f"  {key:28s} {scales[key]:.6g}")
        print("Break-even lambdas (penalty == task loss):")
        print(f"  lambda_rate                  {scales['lambda_rate_breakeven']:.4g}")
        print(f"  lambda_connectivity          {scales['lambda_connectivity_breakeven']:.4g}")
        return

    history = train(
        model,
        config,
        active_tasks=active_tasks,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_rate=args.lambda_rate,
        lambda_connectivity=args.lambda_connectivity,
        noise_level=noise_level,
        log_every=args.log_every,
        plot_results=args.plot_results,
        loss_per_trial=args.loss_per_trial,
        **reg_opts,
    )

    save_path = args.save_path or default_save_path(args.model, active_tasks, args.steps)
    save_checkpoint(
        save_path,
        model,
        config,
        args.model,
        active_tasks,
        model_kwargs=model_kwargs,
        seed=args.seed,
        train_steps=args.steps,
        history=history,
    )


if __name__ == "__main__":
    main()
