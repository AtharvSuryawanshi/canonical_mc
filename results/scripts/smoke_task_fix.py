"""Smoke test of the 2026-11-19 task fix (FIXED_ISSUES #15-#18): short training on the six changed tasks.

Trains the same Dale network (256 units, fanin W_out, g = 1, same seed, same settings as the 150k runs
otherwise) twice for --steps steps on Go, RT Go, Dly Go, Anti, RT Anti, Dly Anti: once on the fixed
generators ("yang") and once on the old ones ("v1"). Both use the new squared loss mask. No budgets.

    python results/scripts/smoke_task_fix.py [--steps 1000]

Writes results/front/smoke_task_fix_curves.png (accuracy and loss vs step, per task, yang vs v1) and
smoke_task_fix_trials.png (one trial per task of the "yang" network: inputs on both rings, fixation input and
target, output ring).
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cmc.task import default_config, generate_trials, rule_name
from cmc.train_cog import DALE_DEFAULT_NOISE_LEVEL, make_dale_model, train, trial_to_tensors

REPO = Path(__file__).resolve().parents[2]
TASKS = ["fdgo", "reactgo", "delaygo", "fdanti", "reactanti", "delayanti"]
N_RING = 16


def run(version, steps, seed):
    cfg = default_config(n_eachring=N_RING, seed=seed, easy_task=True)
    cfg["task_version"] = version
    with contextlib.redirect_stdout(io.StringIO()):
        model = make_dale_model(cfg, n_neurons=256, frac_e=0.8, g=1.0, seed=seed, w_out_init="fanin")
        hist = train(model, cfg, active_tasks=TASKS, n_steps=steps, batch_size=32, lr=1e-3,
                     noise_level=DALE_DEFAULT_NOISE_LEVEL, log_every=50, plot_results=False, show_progress=False)
    return model, cfg, hist


def main(steps, seed):
    runs = {v: run(v, steps, seed) for v in ("yang", "v1")}
    for v, (_, _, h) in runs.items():
        print(v, {t: round(h["per_task"][t]["acc"][-1], 2) for t in TASKS})

    fig, axes = plt.subplots(2, 6, figsize=(18, 5.5), sharex=True)
    for j, t in enumerate(TASKS):
        for v, ls in (("yang", "-"), ("v1", "--")):
            h = runs[v][2]
            axes[0, j].plot(h["eval_step"], h["per_task"][t]["acc"], ls, label=v)
            axes[1, j].plot(h["eval_step"], h["per_task"][t]["loss"], ls, label=v)
        axes[0, j].set_title(rule_name[t])
        axes[0, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("step")
    axes[0, 0].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("task loss (squared mask)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Smoke test, {steps} steps, seed {seed}: fixed generators (yang, solid) vs old (v1, dashed)")
    fig.tight_layout()
    out = REPO / "results/front/smoke_task_fix_curves.png"
    fig.savefig(out, dpi=150)
    print("->", out)

    model, cfg, _ = runs["yang"]
    fig, axes = plt.subplots(4, 6, figsize=(18, 9), sharex="col",
                             gridspec_kw={"height_ratios": [1, 1, 0.7, 1]})
    for j, t in enumerate(TASKS):
        c = dict(cfg)
        c["rng"] = np.random.RandomState(1)        # draws ring 2 for most of these tasks
        tr = generate_trials(t, c, 4, noise_on=False)
        x, _, _, _ = trial_to_tensors(tr, "cpu")
        with torch.no_grad():
            _, _, y_hat = model.simulate(x, noise_level=0.0)
        x0, y0, yh = tr.x[:, 0], tr.y[:, 0], y_hat[0].numpy()
        axes[0, j].imshow(x0[:, 1:1 + N_RING].T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
        axes[1, j].imshow(x0[:, 1 + N_RING:1 + 2 * N_RING].T, aspect="auto", cmap="Oranges", vmin=0, vmax=1)
        axes[2, j].plot(x0[:, 0], "k", label="fixation input")
        axes[2, j].plot(y0[:, 0], "g--", label="fixation target")
        axes[2, j].plot(yh[:, 0], "g", label="fixation output")
        axes[2, j].set_ylim(-0.1, 1.1)
        axes[3, j].imshow(yh[:, 1:].T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        axes[3, j].plot(np.where(tr.y_loc[:, 0] >= 0, tr.y_loc[:, 0] / (2 * np.pi) * N_RING, np.nan), "w--",
                        lw=1)
        axes[0, j].set_title(rule_name[t])
        axes[3, j].set_xlabel("time step (20 ms)")
    for ax, lab in zip(axes[:, 0], ["ring 1 input", "ring 2 input", "fixation", "output ring\n(white: target)"]):
        ax.set_ylabel(lab)
    axes[2, 0].legend(fontsize=6)
    fig.suptitle("Fixed generators, one trial per task, network after the smoke run")
    fig.tight_layout()
    out = REPO / "results/front/smoke_task_fix_trials.png"
    fig.savefig(out, dpi=150)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.steps, a.seed)
