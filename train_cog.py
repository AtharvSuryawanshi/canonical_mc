"""Train RateRNN on the scoped Yang 3-task battery (fdgo, dm1, delaygo)."""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RateRNN
from task import default_config, generate_trials, generate_mixed_trials, rules_dict


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


def masked_mse(output, y, c_mask):
    """Yang lsq loss: mean of mask-weighted squared error."""
    return (c_mask * (output - y).square()).mean()


def rate_reg(r_hist):
    """L2 metabolic / rate cost, factored out so it can be reweighted later."""
    return r_hist.square().mean()

def connectivity_reg(model):
    return model.W_in.square().mean()

def ring_prefs(n_eachring, device, dtype):
    return torch.arange(n_eachring, device=device, dtype=dtype) * (2 * math.pi / n_eachring)


def decode_ring(output, prefs):
    """Population-vector decode of ring channels. output: (B, T, 1+n_ring)."""
    ring = output[..., 1:]
    cos_c = (ring * torch.cos(prefs)).sum(dim=-1)
    sin_c = (ring * torch.sin(prefs)).sum(dim=-1)
    return torch.atan2(sin_c, cos_c)


def circular_abs(delta):
    return torch.remainder(delta + math.pi, 2 * math.pi) - math.pi


def batch_accuracy(output, y_loc, prefs, ang_thresh=math.pi / 5, fix_thresh=0.5):
    """Per-trial metrics: hold fixation, then land within 36° after go.

    Thresholds (defaults match Yang-style eval):
    - ``ang_thresh``: π/5 rad ≈ 36° population-vector decode error on go period
    - ``fix_thresh``: mean predicted fixation channel > 0.5 during fix period

    Returns combined accuracy plus go-only / fix-only rates so early
    training is not hidden by a strict AND.
    """
    pred_loc = decode_ring(output, prefs)
    go = y_loc >= 0
    fix = y_loc < 0

    go_counts = go.sum(dim=1).clamp_min(1)
    target_loc = (y_loc.masked_fill(~go, 0.0).sum(dim=1) / go_counts)
    pred_go = pred_loc.masked_fill(~go, 0.0).sum(dim=1) / go_counts
    ang_ok = circular_abs(pred_go - target_loc).abs() < ang_thresh
    has_go = go.any(dim=1)

    fix_out = output[..., 0]
    fix_counts = fix.sum(dim=1).clamp_min(1)
    fix_mean = fix_out.masked_fill(~fix, 0.0).sum(dim=1) / fix_counts
    fix_ok = fix_mean > fix_thresh
    has_fix = fix.any(dim=1)

    trial_ok = torch.ones_like(has_go)
    trial_ok = torch.where(has_go, ang_ok, trial_ok)
    trial_ok = torch.where(has_fix, trial_ok & fix_ok, trial_ok)

    go_acc = ang_ok[has_go].float().mean().item() if has_go.any() else float("nan")
    fix_acc = fix_ok[has_fix].float().mean().item() if has_fix.any() else float("nan")
    return {
        "acc": trial_ok.float().mean().item(),
        "go_acc": go_acc,
        "fix_acc": fix_acc,
    }


def evaluate_task(model, config, rule, batch_size, device, noise_level=0.0):
    trial = generate_trials(rule, config, batch_size, noise_on=True)
    x, y, c_mask, y_loc = trial_to_tensors(trial, device)
    with torch.no_grad():
        r_hist, x_hist, output = model.simulate(x, noise_level=noise_level)
        loss = masked_mse(output, y, c_mask).item()
        prefs = ring_prefs(config["n_eachring"], output.device, output.dtype)
        acc = batch_accuracy(output, y_loc, prefs)
        activity = model.activity_stats(r_hist, x_hist)
    return {"loss": loss, "activity": activity, **acc}


def make_model(config, n_neurons=64, frac_e=0.8, g=1.5, bias_init=1.0, device="cpu"):
    model = RateRNN(
        n_neurons=n_neurons,
        frac_e=frac_e,
        tau=config["tau"],
        dt=config["dt"],
        g=g,
        input_dim=config["n_input"],
        output_dim=config["n_output"],
        circuit_type="basic_dale",
        bias_init=bias_init,
        readout_nonlinearity="tanh",
    )
    rho = model.scale_recurrent_to_rho(g)
    # Ring inputs are O(1); scale W_in so ReLU^2 units sit in a responsive regime
    # without saturating (bias_init=2 + scale 8 drove ~95% saturation).
    with torch.no_grad():
        model.W_in.mul_(4.0)
        model.W_out[:, model.n_e:] = 0.0
        # Seed fixation readout positive so fix_acc is learnable from step 1.
        model.W_out[0, : model.n_e] = 0.15
    print(f"recurrent init: g={g:.2f}  rho(W)={rho:.3f}")
    return model.to(device)


def sample_input_drive(model, config, task="fdgo", batch_size=8, device="cpu", t_step=None):
    """W_in @ x + b at one timestep (pre-recurrence drive) for init diagnostics."""
    device = torch.device(device)
    trial = generate_trials(task, config, batch_size, noise_on=False)
    x, _, _, _ = trial_to_tensors(trial, device)
    if t_step is None:
        stim = trial.epochs.get("stim1", (0, trial.tdim // 2))
        t_step = int((stim[0] or 0) + max(1, ((stim[1] or trial.tdim) - (stim[0] or 0)) // 2))
        t_step = min(t_step, trial.tdim - 1)
    with torch.no_grad():
        x_t = x[:, :, t_step]
        drive = torch.matmul(x_t, model.W_in.t()) + model.b
    return drive.detach().cpu().numpy().ravel(), t_step


def plot_preactivation_at_init(model, config, device="cpu", task="fdgo"):
    """Histogram of input drive (W_in @ x + b) at init — flags sub-threshold units."""
    drive, t_step = sample_input_drive(model, config, task=task, device=device)
    rates = model.firing_rate(torch.as_tensor(drive, device=next(model.parameters()).device)).detach().cpu().numpy()
    silent_thresh = 0.05 * model.rate_max

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
    frac_neg = float((drive < 0).mean())
    print(f"pre-act: frac negative drive={frac_neg:.2f}, frac silent from drive alone={float((rates < silent_thresh).mean()):.2f}")
    return fig


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
):
    """Multitask training. By default each batch mixes all active tasks."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "loss": [],
        "step": [],
        "eval_step": [],
        "frac_silent": [],
        "frac_saturated": [],
        "per_task": {
            task: {"loss": [], "acc": [], "go_acc": [], "fix_acc": []}
            for task in active_tasks
        },
    }

    for step in range(1, n_steps + 1):
        if mixed_batch and len(active_tasks) > 1:
            trial = generate_mixed_trials(active_tasks, config, batch_size, noise_on=True)
        else:
            rule = str(config["rng"].choice(active_tasks))
            trial = generate_trials(rule, config, batch_size, noise_on=True)
        x, y, c_mask, y_loc = trial_to_tensors(trial, device)

        def loss_fn(r_hist, x_hist, output_matrix, y=y, c_mask=c_mask):
            task_loss = masked_mse(output_matrix, y, c_mask)
            reg = task_loss
            if lambda_rate:
                reg = reg + lambda_rate * rate_reg(r_hist)
            if lambda_connectivity:
                reg = reg + lambda_connectivity * connectivity_reg(model)
            return reg

        loss = model.train_step(
            optimizer,
            x,
            loss_fn,
            noise_level=noise_level,
            grad_clip=grad_clip,
        )
        history["loss"].append(loss.item())
        history["step"].append(step)

        if step == 1 or step % log_every == 0 or step == n_steps:
            history["eval_step"].append(step)
            parts = [f"step {step:4d} | train {loss.item():.4f}"]
            if mixed_batch and len(active_tasks) > 1:
                parts[-1] += " (mixed)"
            act = None
            for task in active_tasks:
                metrics = evaluate_task(model, config, task, eval_batch_size, device)
                history["per_task"][task]["loss"].append(metrics["loss"])
                history["per_task"][task]["acc"].append(metrics["acc"])
                history["per_task"][task]["go_acc"].append(metrics["go_acc"])
                history["per_task"][task]["fix_acc"].append(metrics["fix_acc"])
                parts.append(
                    f"{task} accuracy:{metrics['acc']:.3f} "
                    f"go:{metrics['go_acc']:.2f} fix:{metrics['fix_acc']:.2f} "
                    f"loss:{metrics['loss']:.4f}"
                )
                act = metrics["activity"]
            history["frac_silent"].append(act["frac_silent"])
            history["frac_saturated"].append(act["frac_saturated"])
            parts.append(f"sat:{act['frac_saturated']:.2f}; silent:{act['frac_silent']:.2f}")
            print(" | ".join(parts))

    if plot_results:
        plot_training(history, active_tasks)
        plot_example_runs(model, config, active_tasks)
    return history


def train_without_plots(*args, **kwargs):
    kwargs["plot_results"] = False
    return train(*args, **kwargs)


def plot_training(history, active_tasks):
    """2x2: per-task loss/acc and population silent/saturation fractions vs step."""
    xs = history["eval_step"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for task in active_tasks:
        metrics = history["per_task"][task]
        axes[0, 0].plot(xs, metrics["loss"], marker="o", label=task)
        axes[0, 1].plot(xs, metrics["acc"], marker="o", label=task)
    axes[0, 0].set_xlabel("steps")
    axes[0, 0].set_ylabel("loss")
    axes[0, 0].set_title("Loss")
    axes[0, 1].set_xlabel("steps")
    axes[0, 1].set_ylabel("accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_ylim(-0.05, 1.05)

    axes[1, 0].plot(xs, history["frac_silent"], marker="o", color="C3", label="frac silent")
    axes[1, 0].set_xlabel("steps")
    axes[1, 0].set_ylabel("fraction")
    axes[1, 0].set_title("Silent units (r < 5% rate_max)")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 1].plot(xs, history["frac_saturated"], marker="o", color="C2", label="frac saturated")
    axes[1, 1].set_xlabel("steps")
    axes[1, 1].set_ylabel("fraction")
    axes[1, 1].set_title("Saturated units (r > 95% rate_max)")
    axes[1, 1].set_ylim(-0.05, 1.05)

    for ax in axes.ravel():
        if ax.get_legend_handles_labels()[1]:
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
    """One column per task: input traces + neuron-rate imshow, shared time axis.

    Input layout (task.py): ch0 = fixation / go cue, ch1:rule_start = stimulus
    ring, remaining = rule context. Cue is plotted as the go signal (1 - fix).
    """
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
        ax_r.axhline(model.n_e - 0.5, color="w", ls="--", lw=0.8)
        ax_r.set_xlabel("steps")
        if col == 0:
            ax_r.set_ylabel("neuron")

    fig.colorbar(im, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.02, label="rate")
    plt.show()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description="Train RateRNN on fdgo / dm1 / delaygo")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["fdgo"],
        choices=rules_dict["all"],
        help="Tasks to train. Default: fdgo-only smoke run.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-neurons", type=int, default=128)
    parser.add_argument("--n-eachring", type=int, default=16)
    parser.add_argument("--frac-e", type=float, default=0.8)
    parser.add_argument(
        "--g",
        type=float,
        default=1.5,
        help="Target spectral radius of W_rec at init (near-critical ~1.0–1.5).",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda-rate", type=float, default=0.0)
    parser.add_argument("--lambda-connectivity", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = default_config(n_eachring=args.n_eachring, seed=args.seed, easy_task=True)
    model = make_model(
        config,
        n_neurons=args.n_neurons,
        frac_e=args.frac_e,
        g=args.g,
        device=device,
    )
    print(
        f"device={device}  N={args.n_neurons}  frac_e={args.frac_e}  g={args.g}  "
        f"n_in={config['n_input']}  n_out={config['n_output']}  tasks={args.tasks}"
    )
    train(
        model,
        config,
        active_tasks=tuple(args.tasks),
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_rate=args.lambda_rate,
        lambda_connectivity=args.lambda_connectivity,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
