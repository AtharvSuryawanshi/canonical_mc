"""Single-unit activity and task-variance profile (Yang 2019 Fig. 3a,b).

For the best seed of each budget point: (a) one unit's activity over time on
several trials of one task, (b) that unit's task variance across the 20 tasks.
Default unit = the excitatory unit with the largest single-task variance, and
default task = the task where that unit's variance peaks; override with
--unit / --task.

    python results/scripts/fig_unit_task_variance.py
    python results/scripts/fig_unit_task_variance.py --point control --unit 145 --task fdgo

Task variance is the one stored in each checkpoint (runner.activity_summary).
Writes results/unit_task_variance/<run>/<point>_unit<k>_<task>.png.
"""
import argparse
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
OUT_ROOT = REPO / "results" / "unit_task_variance"
POINT_TITLES = {
    "control": "Unconstrained",
    "rate_limited": "Rate-limited",
    "wiring_limited": "Wiring-limited",
    "both": "Rate + wiring limited",
}


def best_seed_rows(run):
    df = pd.read_csv(run / "runs.csv")
    return df.loc[df.groupby("point").min_task_acc.idxmax()].set_index("point")


def simulate_unit(model, config, task, unit, n_trials, seed):
    cfg = eval_config_from(config, seed, easy_task=True)
    trial = generate_trials(task, cfg, n_trials, noise_on=False)
    x, _, _, _ = trial_to_tensors(trial, "cpu")
    with torch.no_grad():
        r_hist, _, _ = model.simulate(x, noise_level=0.0)
    t = np.arange(r_hist.shape[1]) * config["dt"] / 1000.0
    return t, r_hist[:, :, unit].numpy()


def plot_unit(ckpt, model, config, unit, task, point, out_dir, n_trials, seed):
    tasks = list(ckpt["active_tasks"])
    tv = np.asarray(ckpt["task_variance"])[:, unit]
    is_e = bool(model.E_mask[unit]) if hasattr(model, "E_mask") else None
    cell = "" if is_e is None else (" (E)" if is_e else " (I)")
    t, act = simulate_unit(model, config, task, unit, n_trials, seed)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.6, 5.6))
    ax_a.plot(t, act.T, color="0.45", lw=1.2)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Activity (a.u.)")
    ax_a.set_title(f"Unit {unit}{cell}  {rule_name.get(task, task)}", fontsize=10)
    ax_a.set_xlim(t[0], t[-1])
    ax_a.set_ylim(bottom=0)

    ax_b.plot(range(len(tasks)), tv/np.max(tv), "-o", color="k", ms=4, lw=1.4)
    ax_b.set_xticks(range(len(tasks)))
    ax_b.set_xticklabels([rule_name.get(k, k) for k in tasks], rotation=90, fontsize=7)
    ax_b.set_ylabel("Task variance")
    ax_b.set_title(f"Unit {unit}", fontsize=10)
    ax_b.set_ylim(bottom=0)
    for ax in (ax_a, ax_b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{POINT_TITLES.get(point, point)} (seed {ckpt['seed']})", fontsize=10)
    fig.tight_layout()
    stem = f"{point}_unit{unit}_{task}"
    for ext in ("png",):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_dir / f"{stem}.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--point", nargs="*", default=None, help="Default: every point in the run.")
    ap.add_argument("--unit", type=int, default=None)
    ap.add_argument("--task", type=str, default=None, help="Internal task name, e.g. fdgo.")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--trial-seed", type=int, default=0)
    args = ap.parse_args()

    best = best_seed_rows(args.run)
    points = args.point or [p for p in POINT_TITLES if p in best.index] + \
        [p for p in best.index if p not in POINT_TITLES]
    out_dir = OUT_ROOT / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for point in points:
        path = args.run / best.loc[point, "checkpoint"]
        model, config, ckpt = load_checkpoint(path, device="cpu")
        tv = np.asarray(ckpt["task_variance"])          # (n_tasks, N)
        unit = args.unit
        if unit is None:
            peak = tv.max(axis=0)
            if hasattr(model, "E_mask"):
                peak = np.where(model.E_mask.cpu().numpy() > 0, peak, -np.inf)
            unit = int(np.argmax(peak))
        task = args.task or ckpt["active_tasks"][int(np.argmax(tv[:, unit]))]
        out = plot_unit(ckpt, model, config, unit, task, point, out_dir, args.n_trials, args.trial_seed)
        print(f"{point}: seed {ckpt['seed']}, unit {unit}, task {task} -> {out}")


if __name__ == "__main__":
    main()
