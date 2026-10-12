"""Network input / output schematic for one trial (Yang 2019 Fig. 1-style).

Left column: fixation input, stimulus modality 1 and 2 rings, rule inputs.
Right column: recurrent unit activity (E above the line, I below), fixation
output, response ring. One trial of one task on a trained network.

    python results/scripts/fig_task_io.py
    python results/scripts/fig_task_io.py --task delaydm1 --point both --trial-seed 3

Writes results/task_io/<run>/<point>_<task>_{inputs,outputs}.png.
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cmc.task import generate_trials, rule_name
from cmc.train_cog import eval_config_from, load_checkpoint, trial_to_tensors

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "runs" / "moo_zoom" / "dale_all_curves4_4seed_50k"
OUT_ROOT = REPO / "results" / "task_io"
LINE = "#3b4cc0"
CMAP = "Purples"


def run_trial(path, task, trial_seed, input_noise):
    with contextlib.redirect_stdout(io.StringIO()):
        model, config, ckpt = load_checkpoint(path, device="cpu")
    cfg = eval_config_from(config, trial_seed, easy_task=True)
    trial = generate_trials(task, cfg, 1, noise_on=input_noise)
    x, _, _, _ = trial_to_tensors(trial, "cpu")
    with torch.no_grad():
        r, _, y_hat = model.simulate(x, noise_level=0.0)
    n_ring = config["n_eachring"]
    t_ms = np.arange(trial.x.shape[0]) * config["dt"]
    return dict(
        t=t_ms, x=trial.x[:, 0, :], r=r[0].numpy(), y=y_hat[0].numpy(), n_ring=n_ring,
        rule_start=config["rule_start"], n_e=int(model.E_mask.sum()) if hasattr(model, "E_mask") else None,
        seed=ckpt["seed"],
    )


def _ring(ax, t, m, label, vmax=None):
    ax.imshow(m.T, aspect="auto", origin="lower", cmap=CMAP, vmin=0, vmax=vmax,
              extent=[t[0], t[-1], 0, 360], interpolation="nearest")
    ax.set_yticks([0, 360])
    ax.set_yticklabels(["0°", "360°"])
    ax.set_title(label, fontsize=10, pad=3)


def _trace(ax, t, v, label):
    ax.plot(t, v, color=LINE, lw=2)
    ax.set_ylim(-0.1, 1.2)
    ax.set_yticks([])
    ax.set_title(label, fontsize=10, pad=3)


def plot_inputs(d, out):
    t, x, n = d["t"], d["x"], d["n_ring"]
    fig, axes = plt.subplots(4, 1, figsize=(3.0, 5.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[0.5, 1.4, 1.4, 1.4]))
    _trace(axes[0], t, x[:, 0], "Fixation input")
    vmax = max(x[:, 1:1 + 2 * n].max(), 1e-6)
    _ring(axes[1], t, x[:, 1:1 + n], "Stimulus mod 1", vmax)
    _ring(axes[2], t, x[:, 1 + n:1 + 2 * n], "Stimulus mod 2", vmax)
    rules = x[:, d["rule_start"]:]
    axes[3].imshow(rules.T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1,
                   extent=[t[0], t[-1], 0.5, rules.shape[1] + 0.5], interpolation="nearest")
    axes[3].set_yticks([1, rules.shape[1]])
    axes[3].set_title("Rule inputs", fontsize=10, pad=3)
    axes[3].set_xlabel("Time (ms)")
    axes[3].set_xticks([0, int(t[-1])])
    _finish(fig, axes, out)


def plot_outputs(d, out):
    t, r, y = d["t"], d["r"], d["y"]
    fig, axes = plt.subplots(3, 1, figsize=(3.0, 4.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 0.5, 1.4]))
    axes[0].imshow(r.T, aspect="auto", origin="lower", cmap=CMAP, vmin=0,
                   vmax=np.percentile(r, 99.5), extent=[t[0], t[-1], 1, r.shape[1]],
                   interpolation="nearest")
    if d["n_e"] is not None:
        axes[0].axhline(d["n_e"] + 0.5, color="k", lw=0.8)
        axes[0].text(t[-1], d["n_e"] / 2, " E", va="center", fontsize=8)
        axes[0].text(t[-1], (d["n_e"] + r.shape[1]) / 2, " I", va="center", fontsize=8)
    axes[0].set_yticks([1, r.shape[1]])
    axes[0].set_title("Recurrent units", fontsize=10, pad=3)
    _trace(axes[1], t, y[:, 0], "Fixation output")
    _ring(axes[2], t, y[:, 1:], "Response", vmax=1)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_xticks([0, int(t[-1])])
    _finish(fig, axes, out)


def _finish(fig, axes, out):
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(h_pad=0.6)
    for ext in ("png",):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--point", default="control")
    ap.add_argument("--seed", type=int, default=None, help="Default: best seed of the point.")
    ap.add_argument("--task", default="dm1")
    ap.add_argument("--trial-seed", type=int, default=0)
    ap.add_argument("--no-input-noise", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.run / "runs.csv")
    df = df[df.point == args.point]
    row = df.loc[df.min_task_acc.idxmax()] if args.seed is None else df[df.seed == args.seed].iloc[0]
    d = run_trial(args.run / row.checkpoint, args.task, args.trial_seed, not args.no_input_noise)
    out_dir = OUT_ROOT / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{args.point}_{args.task}"
    plot_inputs(d, f"{stem}_inputs")
    plot_outputs(d, f"{stem}_outputs")
    print(f"{args.point} seed {d['seed']}, {rule_name.get(args.task, args.task)} -> {stem}_inputs/_outputs.png")


if __name__ == "__main__":
    main()
