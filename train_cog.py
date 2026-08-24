"""Train RateRNN on the scoped Yang 3-task battery (fdgo, dm1, delaygo)."""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import RateRNN
from task import default_config, generate_trials, rules_dict


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


def make_model(config, n_neurons=64, frac_e=0.8, g=0.4, device="cpu"):
    model = RateRNN(
        n_neurons=n_neurons,
        frac_e=frac_e,
        tau=config["tau"],
        dt=config["dt"],
        g=g,
        input_dim=config["n_input"],
        output_dim=config["n_output"],
        circuit_type="basic_dale",
    )
    # Decision-task init is tiny (W_in ~ 0.05). Ring inputs are O(1); scale up
    # here so the cognitive tasks can drive rates without changing RateRNN defaults.
    with torch.no_grad():
        model.W_in.mul_(8.0)
    return model.to(device)


def train(
    model,
    config,
    active_tasks=("fdgo",),
    n_steps=200,
    batch_size=32,
    lr=1e-3,
    lambda_rate=0.0,
    noise_level=0.0,
    grad_clip=1.0,
    log_every=10,
    eval_batch_size=32,
):
    """Homogeneous-batch multitask training. Samples one rule per step."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {
        "loss": [],
        "step": [],
        "eval_step": [],
        "rule": [],
        "per_task": {
            task: {"loss": [], "acc": [], "go_acc": [], "fix_acc": []}
            for task in active_tasks
        },
    }

    for step in range(1, n_steps + 1):
        rule = str(config["rng"].choice(active_tasks))
        trial = generate_trials(rule, config, batch_size, noise_on=True)
        x, y, c_mask, y_loc = trial_to_tensors(trial, device)

        def loss_fn(r_hist, x_hist, output_matrix, y=y, c_mask=c_mask):
            task_loss = masked_mse(output_matrix, y, c_mask)
            if lambda_rate:
                return task_loss + lambda_rate * rate_reg(r_hist)
            return task_loss

        loss = model.train_step(
            optimizer,
            x,
            loss_fn,
            noise_level=noise_level,
            grad_clip=grad_clip,
        )
        history["loss"].append(loss.item())
        history["step"].append(step)
        history["rule"].append(rule)

        if step == 1 or step % log_every == 0 or step == n_steps:
            history["eval_step"].append(step)
            parts = [f"step {step:4d} | train {loss.item():.4f} ({rule})"]
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
            parts.append(f"sat:{act['frac_saturated']:.2f}; silent:{act['frac_silent']:.2f}")
            print(" | ".join(parts))

    plot_training(history, active_tasks)
    return history


def plot_training(history, active_tasks):
    """1x2: per-task eval loss and combined accuracy vs training step."""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train RateRNN on fdgo / dm1 / delaygo")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["fdgo"],
        choices=rules_dict["all"],
        help="Tasks to train. Default: fdgo-only smoke run.",
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-neurons", type=int, default=100)
    parser.add_argument("--n-eachring", type=int, default=16)
    parser.add_argument("--frac-e", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-rate", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = default_config(n_eachring=args.n_eachring, seed=args.seed, easy_task=True)
    model = make_model(config, n_neurons=args.n_neurons, frac_e=args.frac_e, device=device)
    print(
        f"device={device}  N={args.n_neurons}  frac_e={args.frac_e}  "
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
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
