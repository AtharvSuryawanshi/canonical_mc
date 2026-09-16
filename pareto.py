"""Pareto sweep over (lambda_rate, lambda_connectivity) for multitask RNN training."""

import argparse
import csv
import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from task import default_config, rules_dict
from train_cog import (
    _model_kwargs_from_args,
    evaluate_objectives,
    make_dale_model,
    make_yang_model,
    resolve_active_tasks,
    train_without_plots,
)


def _lambda_axis(min_val, max_val, n_lambda, scale):
    if scale == "log":
        if min_val <= 0:
            raise ValueError("log scale requires strictly positive lambda min values")
        return np.logspace(np.log10(min_val), np.log10(max_val), n_lambda)
    return np.linspace(min_val, max_val, n_lambda)


def build_lambda_grid(lr_min, lr_max, lc_min, lc_max, n_lambda, scale):
    """1D grids for lambda_rate and lambda_connectivity."""
    lr_vals = _lambda_axis(lr_min, lr_max, n_lambda, scale)
    lc_vals = _lambda_axis(lc_min, lc_max, n_lambda, scale)
    return lr_vals, lc_vals


def build_grid_pairs(args):
    """Cartesian 2D grid, or 1D sweep when one lambda is fixed."""
    fix_lr = args.fix_lambda_rate
    fix_lc = args.fix_lambda_connectivity
    if fix_lr is not None and fix_lc is not None:
        raise ValueError("Set at most one of --fix-lambda-rate and --fix-lambda-connectivity")

    if fix_lr is not None:
        lc_vals = _lambda_axis(
            args.lambda_connectivity_min,
            args.lambda_connectivity_max,
            args.n_lambda,
            args.lambda_scale,
        )
        lr_vals = np.array([fix_lr], dtype=float)
        pairs = [(float(fix_lr), float(lc)) for lc in lc_vals]
        sweep = "connectivity"
        pareto_objectives = ("task_loss", "wiring_cost")
        return pairs, lr_vals, lc_vals, sweep, pareto_objectives

    if fix_lc is not None:
        lr_vals = _lambda_axis(
            args.lambda_rate_min,
            args.lambda_rate_max,
            args.n_lambda,
            args.lambda_scale,
        )
        lc_vals = np.array([fix_lc], dtype=float)
        pairs = [(float(lr), float(fix_lc)) for lr in lr_vals]
        sweep = "rate"
        pareto_objectives = ("task_loss", "metabolic_cost")
        return pairs, lr_vals, lc_vals, sweep, pareto_objectives

    lr_vals, lc_vals = build_lambda_grid(
        args.lambda_rate_min,
        args.lambda_rate_max,
        args.lambda_connectivity_min,
        args.lambda_connectivity_max,
        args.n_lambda,
        args.lambda_scale,
    )
    pairs = list(product(lr_vals, lc_vals))
    return pairs, lr_vals, lc_vals, "both", ("task_loss", "metabolic_cost", "wiring_cost")


def make_fresh_model(args, config, device):
    model_kwargs = _model_kwargs_from_args(args)
    if args.model == "yang":
        return make_yang_model(config, device=device, **model_kwargs)
    return make_dale_model(config, device=device, **model_kwargs)


def default_output_dir(model, battery_label, n_lambda, sweep="both"):
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if sweep == "connectivity":
        tag = f"1d_lc_{n_lambda}"
    elif sweep == "rate":
        tag = f"1d_lr_{n_lambda}"
    else:
        tag = f"{n_lambda}x{n_lambda}"
    return Path("pareto_runs") / f"{model}_{battery_label}_{tag}_{stamp}"


def pareto_mask(objectives):
    """Mark nondominated rows (minimize all three objectives). objectives: (n, 3)."""
    n = objectives.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                mask[i] = False
                break
    return mask


def append_csv_row(csv_path, row, write_header=False):
    fieldnames = list(row.keys())
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _objective_label(name):
    return {
        "task_loss": "task loss",
        "metabolic_cost": "metabolic cost",
        "wiring_cost": "wiring cost",
    }[name]


def plot_pareto_front(rows, out_path, pareto_objectives):
    is_pareto = np.array([r["is_pareto"] for r in rows], dtype=bool)
    names = pareto_objectives
    objectives = np.array([[r[k] for k in names] for r in rows])

    if len(names) == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(objectives[~is_pareto, 0], objectives[~is_pareto, 1], c="0.7", s=36, label="grid")
        ax.scatter(objectives[is_pareto, 0], objectives[is_pareto, 1], c="C1", s=64, label="Pareto")
        ax.set_xlabel(_objective_label(names[0]))
        ax.set_ylabel(_objective_label(names[1]))
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.suptitle("Pareto front (2 objectives)")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        idx = {"task_loss": 0, "metabolic_cost": 1, "wiring_cost": 2}
        pairs = [
            ("task_loss", "metabolic_cost"),
            ("task_loss", "wiring_cost"),
            ("metabolic_cost", "wiring_cost"),
        ]
        for ax, (a, b) in zip(axes, pairs):
            i, j = idx[a], idx[b]
            ax.scatter(objectives[~is_pareto, i], objectives[~is_pareto, j], c="0.7", s=36, label="grid")
            ax.scatter(objectives[is_pareto, i], objectives[is_pareto, j], c="C1", s=64, label="Pareto")
            ax.set_xlabel(_objective_label(a))
            ax.set_ylabel(_objective_label(b))
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("Pareto front (3 objectives)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pareto sweep over lambda_rate and lambda_connectivity."
    )
    parser.add_argument("--model", choices=["yang", "dale"], default="dale")
    parser.add_argument(
        "--task-battery",
        choices=["all", "core5", "sanity3"],
        default="all",
        help="Preset task subset (ignored when --tasks is set).",
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
    parser.add_argument("--n-rnn", type=int, default=256)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--n-eachring", type=int, default=16)
    parser.add_argument("--frac-e", type=float, default=0.8)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument("--sigma-rec", type=float, default=0.05)
    parser.add_argument(
        "--noise-level",
        type=float,
        default=None,
        help="Default: 1.0 for yang, 0.0 for dale.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=64)

    parser.add_argument("--lambda-rate-min", type=float, default=0.0)
    parser.add_argument("--lambda-rate-max", type=float, default=1e-2)
    parser.add_argument("--lambda-connectivity-min", type=float, default=0.0)
    parser.add_argument("--lambda-connectivity-max", type=float, default=1e-2)
    parser.add_argument("--n-lambda", type=int, default=5)
    parser.add_argument(
        "--lambda-scale",
        choices=["linear", "log"],
        default="linear",
        help="Spacing for lambda grids (linear or log).",
    )
    parser.add_argument(
        "--fix-lambda-rate",
        type=float,
        default=None,
        help="Hold lambda_rate fixed; sweep lambda_connectivity only (2D Pareto: task vs wiring).",
    )
    parser.add_argument(
        "--fix-lambda-connectivity",
        type=float,
        default=None,
        help="Hold lambda_connectivity fixed; sweep lambda_rate only (2D Pareto: task vs metabolic).",
    )

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save pareto_front.png at end (default: off, for headless/Slurm).",
    )
    return parser.parse_args()


def finalize_results(rows, pareto_objectives):
    objectives = np.array([[r[k] for k in pareto_objectives] for r in rows])
    mask = pareto_mask(objectives)
    for row, is_p in zip(rows, mask):
        row["is_pareto"] = bool(is_p)
    return rows


def main():
    args = parse_args()
    active_tasks = resolve_active_tasks(args.task_battery, args.tasks)
    battery_label = args.task_battery if args.tasks is None else "custom"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    noise_level = (
        args.noise_level
        if args.noise_level is not None
        else (1.0 if args.model == "yang" else 0.0)
    )

    grid_pairs, lr_vals, lc_vals, sweep, pareto_objectives = build_grid_pairs(args)

    out_dir = Path(
        args.output_dir
        or default_output_dir(args.model, battery_label, args.n_lambda, sweep=sweep)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"

    if args.plot:
        plt.switch_backend("Agg")

    if sweep == "both":
        grid_desc = f"{args.n_lambda}x{args.n_lambda}={len(grid_pairs)}"
    else:
        grid_desc = f"1D {sweep} n={len(grid_pairs)}  pareto={pareto_objectives}"
    print(
        f"Pareto sweep: model={args.model}  device={device}  "
        f"task_battery={battery_label}  tasks={len(active_tasks)}  "
        f"grid={grid_desc}  output={out_dir.resolve()}"
    )

    rows = []
    for run_idx, (lr, lc) in enumerate(
        tqdm(grid_pairs, desc="Pareto sweep", unit="run"), start=1
    ):
        t0 = time.perf_counter()
        config = default_config(n_eachring=args.n_eachring, seed=args.seed, easy_task=True)
        model = make_fresh_model(args, config, device)

        train_without_plots(
            model,
            config,
            active_tasks=active_tasks,
            n_steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_rate=float(lr),
            lambda_connectivity=float(lc),
            noise_level=noise_level,
            log_every=args.log_every,
            show_progress=False,
        )

        obj = evaluate_objectives(
            model,
            config,
            active_tasks,
            args.eval_batch_size,
            device,
            noise_level=0.0,
        )
        train_time_s = time.perf_counter() - t0

        row = {
            "run_idx": run_idx,
            "lambda_rate": float(lr),
            "lambda_connectivity": float(lc),
            "task_loss": obj["task_loss"],
            "metabolic_cost": obj["metabolic_cost"],
            "wiring_cost": obj["wiring_cost"],
            "mean_acc": obj["mean_acc"],
            "train_time_s": train_time_s,
            "is_pareto": False,
        }
        rows.append(row)
        append_csv_row(csv_path, row, write_header=(run_idx == 1))

        tqdm.write(
            f"run {run_idx}/{len(grid_pairs)}  "
            f"lr={lr:.2e}  lc={lc:.2e}  "
            f"task={obj['task_loss']:.4f}  meta={obj['metabolic_cost']:.4f}  "
            f"wire={obj['wiring_cost']:.4f}  acc={obj['mean_acc']:.3f}  "
            f"time={train_time_s:.1f}s"
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = finalize_results(rows, pareto_objectives)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model": args.model,
        "task_battery": battery_label,
        "active_tasks": list(active_tasks),
        "steps": args.steps,
        "seed": args.seed,
        "noise_level": noise_level,
        "sweep_mode": sweep,
        "pareto_objectives": list(pareto_objectives),
        "fix_lambda_rate": args.fix_lambda_rate,
        "fix_lambda_connectivity": args.fix_lambda_connectivity,
        "lambda_grid": {
            "rate_min": args.lambda_rate_min,
            "rate_max": args.lambda_rate_max,
            "connectivity_min": args.lambda_connectivity_min,
            "connectivity_max": args.lambda_connectivity_max,
            "n_lambda": args.n_lambda,
            "scale": args.lambda_scale,
            "rate_values": lr_vals.tolist(),
            "connectivity_values": lc_vals.tolist(),
        },
        "n_runs": len(rows),
        "n_pareto": int(sum(r["is_pareto"] for r in rows)),
        "rows": rows,
    }
    json_path = out_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    n_pareto = summary["n_pareto"]
    print(f"Done. {n_pareto}/{len(rows)} Pareto-optimal points.")
    print(f"Saved -> {csv_path.resolve()}")
    print(f"Saved -> {json_path.resolve()}")

    if args.plot:
        plot_path = out_dir / "pareto_front.png"
        plot_pareto_front(rows, plot_path, pareto_objectives)
        print(f"Saved -> {plot_path.resolve()}")


if __name__ == "__main__":
    main()
