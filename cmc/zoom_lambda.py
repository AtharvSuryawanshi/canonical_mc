"""Zoom into a few (lambda_rate, lambda_connectivity) points and SAVE every network.

``cmc.pareto`` answers *where* the cost/accuracy trade-off is: it trains a grid
with few seeds and keeps only metrics. This script answers *what the networks
look like there*: it trains many seeds at a handful of hand-picked lambda points
and keeps every trained network, so connectivity motifs and functional
populations can be inspected afterwards.

Training and evaluation are the exact functions ``cmc.pareto`` uses, and the
training/eval flags (and their defaults) are inherited from its parser. A network
is fully determined by (lambda, seed): zoom_lambda and a pareto sweep produce the
identical network, whatever order or process it was trained in.

Output, under ``zoom_lambda_runs/<run>/``:

    <point>/seed_00.pt ...   one checkpoint per network (loadable with
                             cmc.train_cog.load_checkpoint), additionally holding
                             lambdas, eval metrics, feasibility and per-neuron
                             task variance / mean rate (n_tasks x N)
    runs.csv                 one row per network: point, seed, metrics, feasible
    summary.json             arguments, the points, feasible count per point

Full activity is deliberately NOT saved (~65 MB per network): it is reproduced
exactly by re-simulating a checkpoint on the same eval seeds.

    python -m cmc.zoom_lambda                      # default 4 points x 10 seeds
    python -m cmc.zoom_lambda --points "knee=0.27,4.37; 0.02,9.15" --n-seeds 5
"""

import argparse
import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from cmc.pareto import (
    append_csv_row,
    build_parser as build_pareto_parser,
    make_fresh_model,
    parse_eval_seeds,
)
from cmc.paths import ZOOM_LAMBDA_DIR
from cmc.task import default_config, generate_trials
from cmc.train_cog import (
    DALE_DEFAULT_NOISE_LEVEL,
    _model_kwargs_from_args,
    eval_config_from,
    evaluate_pareto_metrics,
    resolve_active_tasks,
    save_checkpoint,
    train_without_plots,
    trial_to_tensors,
)

# A 2x2 design over the two constraints, picked from the corrected front of
# pareto_runs/dale_core5_6x6_2026_09_23_05_45_12_5802320 (task axis min_task_acc,
# networks with min_task_acc < 0.6 excluded), at the exact grid lambdas. They are
# not bit-for-bit the sweep's networks: that sweep predates seeding torch, so its
# readout init and training noise were unseeded (see FIXED_ISSUES.md). Each
# constrained point is the most constrained network along its direction that
# still did every task (worst-task accuracy, conn_frac, 3-seed mean):
#
#   control  (0, 0)           no budget: structure from the tasks alone
#                             (min_task_acc 0.85, conn_frac 0.77; dominated,
#                             kept as the reference condition)
#   rate     (0.633, 1.0)     metabolic pressure, weak wiring pressure
#                             (0.68, conn_frac 0.39)
#   wiring   (0.02, 9.15)     wiring pressure, weak metabolic pressure
#                             (0.68, conn_frac 0.19; 1/3 seeds infeasible)
#   both     (0.267, 4.37)    both constraints at once
#                             (0.64, conn_frac 0.16)
#
# Reading it: a motif already present in `control` comes from the tasks, one
# only in `wiring` from the wiring cost, one only in `both` needs the
# constraints to combine.
DEFAULT_POINTS = (
    ("control", 0.0, 0.0),
    ("rate", 0.6325269095141253, 1.0),
    ("wiring", 0.020000000000000004, 9.14610103854653),
    ("both", 0.2667268608396602, 4.373448295773113),
)

# Flags taken over from cmc.pareto's parser. Everything else there (lambda grid,
# front construction, plotting) has no meaning for a fixed list of points.
SHARED_FLAGS = {
    "model", "task_battery", "tasks", "steps", "batch_size", "n_rnn",
    "n_neurons", "n_eachring", "frac_e", "g", "sigma_rec", "noise_level", "lr",
    "log_every", "seed", "device", "eval_batch_size", "eval_seeds",
    "feasible_min_task_acc", "n_seeds", "rate_kind", "rate_inh_scale",
    "conn_kind", "conn_target", "conn_inh_scale", "prune_eps", "loss_per_trial",
    "eval_noise_level", "eval_input_noise", "output_dir",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train many seeds at chosen lambda points and save every network."
    )
    for action in build_pareto_parser()._actions:
        if action.dest in SHARED_FLAGS:
            parser._add_action(copy.copy(action))
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help='Semicolon-separated "name=lambda_rate,lambda_conn" (name optional), '
        "e.g. \"knee=0.27,4.37; 0.02,9.15\". Default: the control/rate/wiring/both "
        "2x2 design from the core5 6x6 front.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain networks whose checkpoint already exists (default: skip them, "
        "so a job that hit its time limit can simply be resubmitted).",
    )
    parser.set_defaults(task_battery="core5", steps=4000, n_seeds=10)
    return parser


def parse_points(text):
    if text is None or not text.strip():
        return list(DEFAULT_POINTS)
    points = []
    for i, chunk in enumerate(c.strip() for c in text.split(";") if c.strip()):
        name, _, vals = chunk.rpartition("=")
        lr, lc = (float(v) for v in vals.split(","))
        points.append((name.strip() or f"p{i}", lr, lc))
    names = [p[0] for p in points]
    if len(set(names)) != len(names):
        raise ValueError(f"--points: duplicate names {names}")
    return points


def activity_summary(model, train_config, active_tasks, device, eval_seeds, batch_size):
    """Per-neuron task variance and mean rate, shape (n_tasks, N) each.

    Task variance follows notebooks/analysis_of_network.ipynb (Yang et al. 2019):
    variance across trials at each time step after fixation onset, averaged over
    time, then over eval seeds. Noise-free, on the same eval trials as the metrics.
    """
    tvs, rates = [], []
    with torch.no_grad():
        for rule in active_tasks:
            per_seed_tv, per_seed_rate = [], []
            for seed_k in eval_seeds:
                cfg = eval_config_from(train_config, seed_k, easy_task=True)
                trial = generate_trials(rule, cfg, batch_size, noise_on=False)
                x, _, _, _ = trial_to_tensors(trial, device)
                r_hist, _, _ = model.simulate(x, noise_level=0.0)
                h = r_hist[:, trial.epochs["fix1"][1]:, :]      # (B, T, N)
                per_seed_tv.append(h.var(dim=0).mean(dim=0).cpu().numpy())
                per_seed_rate.append(h.mean(dim=(0, 1)).cpu().numpy())
            tvs.append(np.mean(per_seed_tv, axis=0))
            rates.append(np.mean(per_seed_rate, axis=0))
    return np.stack(tvs).astype(np.float32), np.stack(rates).astype(np.float32)


def default_output_dir(model, battery_label, n_points, n_seeds):
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return ZOOM_LAMBDA_DIR / f"{model}_{battery_label}_{n_points}pt_{n_seeds}seed_{stamp}"


def resolve_run_settings(args):
    """(active_tasks, battery_label, device, noise_level, eval_seeds, reg_opts)."""
    active_tasks = resolve_active_tasks(args.task_battery, args.tasks)
    battery_label = args.task_battery if args.tasks is None else "custom"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    noise_level = (
        args.noise_level
        if args.noise_level is not None
        else (1.0 if args.model == "yang" else DALE_DEFAULT_NOISE_LEVEL)
    )
    reg_opts = {
        "rate_kind": args.rate_kind,
        "rate_inh_scale": args.rate_inh_scale,
        "conn_kind": args.conn_kind,
        "conn_target": args.conn_target,
        "conn_inh_scale": args.conn_inh_scale,
    }
    return active_tasks, battery_label, device, noise_level, parse_eval_seeds(args.eval_seeds), reg_opts


def train_and_evaluate(args, lambda_rate, lambda_connectivity, seed, active_tasks,
                       device, noise_level, eval_seeds, reg_opts):
    """Train one network exactly as ``cmc.pareto`` does and evaluate it.

    Returns (model, config, history, metrics). Shared by zoom_lambda and cmc.moo,
    so every script produces the identical network for a given (lambda, seed).
    """
    config = default_config(n_eachring=args.n_eachring, seed=seed, easy_task=True)
    model = make_fresh_model(args, config, device, seed=seed)
    history = train_without_plots(
        model,
        config,
        active_tasks=active_tasks,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_rate=float(lambda_rate),
        lambda_connectivity=float(lambda_connectivity),
        noise_level=noise_level,
        log_every=args.log_every,
        show_progress=False,
        loss_per_trial=args.loss_per_trial,
        **reg_opts,
    )
    model.eval()
    metrics = evaluate_pareto_metrics(
        model,
        config,
        active_tasks,
        device,
        eval_seeds=eval_seeds,
        batch_size=args.eval_batch_size,
        noise_level=args.eval_noise_level,
        input_noise=args.eval_input_noise,
        easy_task=True,
        **reg_opts,
    )
    return model, config, history, metrics


def main(argv=None):
    args = build_parser().parse_args(argv)
    points = parse_points(args.points)
    active_tasks, battery_label, device, noise_level, eval_seeds, reg_opts = (
        resolve_run_settings(args)
    )
    seeds = [int(args.seed) + k for k in range(max(1, int(args.n_seeds)))]

    out_dir = Path(args.output_dir or default_output_dir(args.model, battery_label, len(points), len(seeds)))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    if args.overwrite and runs_csv.exists():
        runs_csv.unlink()
    done_rows = runs_csv.exists()

    print(
        f"zoom_lambda: {len(points)} points x {len(seeds)} seeds = "
        f"{len(points) * len(seeds)} networks  tasks={active_tasks}  "
        f"steps={args.steps}  device={device}\n  output={out_dir.resolve()}"
    )
    for name, lr, lc in points:
        print(f"  {name:10s} lambda_rate={lr:.4g}  lambda_conn={lc:.4g}")

    rows = []
    jobs = [(p, s) for p in points for s in seeds]
    for (name, lr, lc), seed in tqdm(jobs, desc="zoom_lambda", unit="net"):
        ckpt_path = out_dir / name / f"seed_{seed:02d}.pt"
        if ckpt_path.exists() and not args.overwrite:
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            rows.append(ck["run_row"])
            continue

        t0 = time.perf_counter()
        model, config, history, metrics = train_and_evaluate(
            args, lr, lc, seed, active_tasks, device, noise_level, eval_seeds, reg_opts
        )
        task_var, mean_rate = activity_summary(
            model, config, active_tasks, device, eval_seeds, args.eval_batch_size
        )
        row = {
            "point": name,
            "seed": seed,
            "lambda_rate": float(lr),
            "lambda_connectivity": float(lc),
            **metrics,
            "is_feasible": bool(metrics["min_task_acc"] >= args.feasible_min_task_acc),
            "checkpoint": ckpt_path.relative_to(out_dir).as_posix(),
            "train_time_s": time.perf_counter() - t0,
        }
        model_kwargs = _model_kwargs_from_args(args)
        model_kwargs["seed"] = int(seed)
        save_checkpoint(
            ckpt_path,
            model,
            config,
            args.model,
            active_tasks,
            model_kwargs=model_kwargs,
            seed=seed,
            train_steps=args.steps,
            history=history,
            verbose=False,
            extra={
                "lambda_rate": float(lr),
                "lambda_connectivity": float(lc),
                "point": name,
                "metrics": metrics,
                "is_feasible": row["is_feasible"],
                "reg": reg_opts,
                "noise_level": noise_level,
                "eval_seeds": list(eval_seeds),
                "task_variance": task_var,
                "mean_rate": mean_rate,
                "run_row": row,
            },
        )
        append_csv_row(runs_csv, row, write_header=not done_rows)
        done_rows = True
        rows.append(row)
        tqdm.write(
            f"{name:10s} seed {seed:2d}  min_task_acc={metrics['min_task_acc']:.3f}  "
            f"mean_acc={metrics['mean_acc']:.3f}  conn_frac={metrics['conn_frac']:.3f}  "
            f"{'ok' if row['is_feasible'] else 'FAILS A TASK'}  "
            f"({row['train_time_s']:.0f}s)"
        )

    per_point = {
        name: {
            "lambda_rate": lr,
            "lambda_connectivity": lc,
            "n_trained": sum(r["point"] == name for r in rows),
            "n_feasible": sum(r["point"] == name and r["is_feasible"] for r in rows),
        }
        for name, lr, lc in points
    }
    summary = {
        "model": args.model,
        "task_battery": battery_label,
        "active_tasks": list(active_tasks),
        "steps": args.steps,
        "seeds": seeds,
        "points": per_point,
        "feasible_min_task_acc": args.feasible_min_task_acc,
        "noise_level": noise_level,
        "eval_seeds": list(eval_seeds),
        "eval_noise_level": args.eval_noise_level,
        "eval_input_noise": bool(args.eval_input_noise),
        "reg": reg_opts,
        "loss_per_trial": bool(args.loss_per_trial),
        "args": vars(args),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nFeasible networks per point (min_task_acc >= "
          f"{args.feasible_min_task_acc}):")
    for name, info in per_point.items():
        print(f"  {name:10s} {info['n_feasible']}/{info['n_trained']}")
    print(f"Saved -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
