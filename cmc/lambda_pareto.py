"""Pareto sweep over a grid of (lambda_rate, lambda_connectivity) -- weighted-sum training.

The front machinery lives in ``cmc.front``, the per-network training in ``cmc.runner``.
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from cmc.front import (  # noqa: F401  (re-exported: notebooks import these from here)
    FEASIBLE_MIN_TASK_ACC,
    OBJECTIVE_MAXIMIZE,
    compute_front,
    feasible_mask,
    finalize_results,
    pareto_mask,
    pareto_maximize_flags,
)
from cmc.paths import LAMBDA_PARETO_RUNS_DIR
from cmc.runner import (  # noqa: F401
    DEFAULT_EVAL_SEEDS,
    add_common_args,
    append_csv_row,
    make_fresh_model,
    parse_eval_seeds,
)
from cmc.task import default_config
from cmc.train_cog import (
    DALE_DEFAULT_NOISE_LEVEL,
    evaluate_pareto_metrics,
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
        if args.include_lambda_zero:
            pairs.insert(0, (float(fix_lr), 0.0))
        sweep = "connectivity"
        pareto_objectives = (args.pareto_task_objective, "wiring_cost")
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
        if args.include_lambda_zero:
            pairs.insert(0, (0.0, float(fix_lc)))
        sweep = "rate"
        pareto_objectives = (args.pareto_task_objective, "metabolic_cost")
        return pairs, lr_vals, lc_vals, sweep, pareto_objectives

    lr_vals, lc_vals = build_lambda_grid(
        args.lambda_rate_min,
        args.lambda_rate_max,
        args.lambda_connectivity_min,
        args.lambda_connectivity_max,
        args.n_lambda,
        args.lambda_scale,
    )
    pairs = [(float(a), float(b)) for a, b in product(lr_vals, lc_vals)]
    if args.include_lambda_zero:
        pairs.insert(0, (0.0, 0.0))
    return (
        pairs,
        lr_vals,
        lc_vals,
        "both",
        (args.pareto_task_objective, "metabolic_cost", "wiring_cost"),
    )


def default_output_dir(model, battery_label, n_lambda, sweep="both"):
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if sweep == "connectivity":
        tag = f"1d_lc_{n_lambda}"
    elif sweep == "rate":
        tag = f"1d_lr_{n_lambda}"
    else:
        tag = f"{n_lambda}x{n_lambda}"
    # Two jobs submitted in the same second would otherwise collide on the
    # timestamp; the Slurm job id keeps them distinct.
    job_id = os.environ.get("SLURM_JOB_ID", "").strip()
    suffix = f"_{job_id}" if job_id else ""
    return LAMBDA_PARETO_RUNS_DIR / f"{model}_{battery_label}_{tag}_{stamp}{suffix}"


AGG_METRIC_KEYS = (
    "mean_acc",
    "min_task_acc",
    "task_loss",
    "metabolic_cost",
    "wiring_cost",
    "conn_frac",
    "wiring_cost_w_in_l2",
)


def aggregate_seed_runs(lr, lc, objs, train_time_s):
    """Collapse the per-seed evals at one lambda point into a single row.

    Primary column names hold the across-seed mean (so downstream analysis keeps
    working), with a matching ``*_std`` column next to each.
    """
    row = {
        "lambda_rate": float(lr),
        "lambda_connectivity": float(lc),
        "n_seeds": len(objs),
    }
    keys = [k for k in AGG_METRIC_KEYS if k in objs[0]]
    keys += sorted(k for k in objs[0] if k.startswith("acc_"))
    for key in keys:
        vals = np.array([o[key] for o in objs], dtype=float)
        row[key] = float(np.mean(vals))
        row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    row["train_time_s"] = float(train_time_s)
    row["is_pareto"] = False
    return row


def _objective_label(name):
    return {
        "mean_acc": "mean task accuracy",
        "min_task_acc": "min task accuracy (worst task)",
        "conn_frac": "connection fraction",
        "task_loss": "task loss",
        "metabolic_cost": "metabolic cost",
        "wiring_cost": "wiring cost",
    }.get(name, name)


COST_OBJECTIVES = {"task_loss", "metabolic_cost", "wiring_cost", "conn_frac"}


def _maybe_log(ax, values, name, which):
    """Log-scale an axis when a cost objective spans more than a decade.

    The lambda=0 anchor is an unconstrained network -- nothing bounds its firing
    rates -- so its metabolic cost lands ~60x above every penalised point. On a
    linear axis that single point flattens the rest of the front onto zero.
    """
    if name not in COST_OBJECTIVES:
        return
    v = values[values > 0]
    if v.size and v.max() / v.min() > 10:
        (ax.set_xscale if which == "x" else ax.set_yscale)("log")


def _front_panel(ax, objectives, is_pareto, is_anchor, i, j, names, is_feasible=None):
    if is_feasible is None:
        is_feasible = np.ones(len(objectives), dtype=bool)
    grid = ~is_pareto & ~is_anchor & is_feasible
    front = is_pareto & ~is_anchor
    dead = ~is_feasible & ~is_anchor
    ax.scatter(objectives[grid, i], objectives[grid, j], c="0.7", s=36, label="dominated")
    if dead.any():
        ax.scatter(
            objectives[dead, i], objectives[dead, j],
            marker="x", c="0.55", s=36, label="fails a task",
        )
    ax.scatter(objectives[front, i], objectives[front, j], c="C1", s=64, label="Pareto")
    if is_anchor.any():
        ax.scatter(
            objectives[is_anchor, i],
            objectives[is_anchor, j],
            marker="*",
            s=260,
            c="C3",
            zorder=5,
            label=r"$\lambda=0$ anchor",
        )
    ax.set_xlabel(_objective_label(names[i]))
    ax.set_ylabel(_objective_label(names[j]))
    ax.grid(True, alpha=0.3)
    _maybe_log(ax, objectives[:, i], names[i], "x")
    _maybe_log(ax, objectives[:, j], names[j], "y")


def plot_pareto_front(rows, out_path, pareto_objectives):
    is_pareto = np.array([r["is_pareto"] for r in rows], dtype=bool)
    names = list(pareto_objectives)
    objectives = np.array([[r[k] for k in names] for r in rows])
    is_anchor = np.array(
        [r["lambda_rate"] == 0 and r["lambda_connectivity"] == 0 for r in rows],
        dtype=bool,
    )
    is_feasible = np.array([r.get("is_feasible", True) for r in rows], dtype=bool)

    if len(names) == 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        _front_panel(ax, objectives, is_pareto, is_anchor, 0, 1, names, is_feasible)
        ax.legend(loc="best", fontsize=8)
        fig.suptitle("Pareto front (2 objectives)")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            _front_panel(ax, objectives, is_pareto, is_anchor, i, j, names, is_feasible)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle("Pareto front (3 objectives)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    """CLI for the sweep: the common flags (cmc.runner) plus the lambda grid and front options."""
    parser = argparse.ArgumentParser(
        description="Pareto sweep over lambda_rate and lambda_connectivity."
    )
    add_common_args(parser)
    parser.add_argument(
        "--pareto-task-objective",
        choices=["min_task_acc", "mean_acc"],
        default="min_task_acc",
        help="Task axis of the front. mean_acc lets a network abandon one task "
        "and still score ~0.9; min_task_acc is only high if every task is done.",
    )
    parser.add_argument(
        "--include-infeasible",
        dest="exclude_infeasible",
        action="store_false",
        help="Let networks that fail a task (min_task_acc below "
        "--feasible-min-task-acc) compete for the front. Default: excluded.",
    )
    parser.set_defaults(exclude_infeasible=True)

    # Lambda ranges recalibrated against the CURRENT objectives: L1 on W_rec for
    # wiring, per-trial task loss, noise_level=0.1. The pre-fix ranges were
    # calibrated against the OLD task loss (global .mean()) and the OLD wiring
    # cost (L2 on W_in) -- a different quantity -- so they are not reusable.
    #
    # Measured at a *trained* lambda=0 DaleRNN (N=256, n_eachring=16, easy_task):
    #
    #                        task_loss   metabolic   wiring   break-even lr / lc
    #   sanity3 (1k steps)     0.0111      1.247      0.0234     0.0089 / 0.47
    #   core5   (2k steps)     0.0057      0.781      0.0236     0.0072 / 0.24
    #
    # Two traps in reading those numbers:
    #
    # 1. `train_cog --report-scales` measures at INIT, where the rate cost is 26x
    #    smaller and the task loss 30x larger than at the solution. Its break-even
    #    lambdas (~4 rate, ~8 wiring) overshoot by ~500x / ~20x. Use them to check
    #    the terms are finite, not to centre the grid.
    # 2. Break-even at the *unregularized* solution is only a lower anchor. The
    #    knee sits above it by however compressible the penalized term is, since
    #    the network shrinks that term before it sacrifices accuracy -- and the
    #    two costs differ enormously in compressibility:
    #
    #      rate cost   ~46x compressible (0.52 -> 0.011 at acc 0.95), so the
    #                  lambda_rate knee lands near 0.3-1, ~100x break-even.
    #      wiring cost only ~2.3x compressible. Measured (sanity3, 1k steps,
    #                  lambda_rate=0), wiring / conn_frac / mean_acc:
    #                    lc=0     0.0234 / 0.729 / 0.833
    #                    lc=0.1   0.0233 / 0.728 / 0.856   <- inert
    #                    lc=1     0.0219 / 0.710 / 0.858   <- inert
    #                    lc=10    0.0149 / 0.577 / 0.835   <- onset
    #                    lc=100   0.0106 / 0.448 / 0.649   <- knee
    #                    lc=1000  0.0103 / 0.436 / 0.653   <- saturated
    #                  so the lambda_conn knee is 10-100, ~200x break-even, and
    #                  beyond ~100 the cost stops falling: softplus magnitudes
    #                  are strictly positive, so with prune_eps=0 an L1 shrinks
    #                  synapses toward a floor instead of pruning them. The
    #                  wiring axis therefore spans only ~2.3x no matter how large
    #                  lambda_conn gets -- see DaleRNN.prune_eps to lift that.
    #
    # These defaults are sized for --n-lambda 5: they start where the cost first
    # moves and end at saturation / collapse, because the inert low end is already
    # covered by the --include-lambda-zero anchor. For --n-lambda 8-10, extend the
    # min down (--lambda-rate-min 1e-3, --lambda-connectivity-min 3e-1) to resolve
    # the flat "free lunch" arm where cost falls at no accuracy cost.
    #
    # Battery caveat: the knee scales with task_loss (lambda ~ task_loss / cost),
    # so a battery or step count with a lower task loss shifts it DOWN. core5 at
    # 2k steps has half sanity3's task loss, so expect its knee at ~0.5x these
    # values; that is why the connectivity range starts at 1 rather than 10.
    parser.add_argument("--lambda-rate-min", type=float, default=1e-1)
    parser.add_argument("--lambda-rate-max", type=float, default=5.0)
    parser.add_argument("--lambda-connectivity-min", type=float, default=1e0)
    parser.add_argument("--lambda-connectivity-max", type=float, default=60)
    parser.add_argument("--n-lambda", type=int, default=5)
    parser.add_argument(
        "--lambda-scale",
        choices=["linear", "log"],
        default="log",
        help="Spacing for lambda grids. Log requires strictly positive min values.",
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

    parser.add_argument(
        "--include-lambda-zero",
        dest="include_lambda_zero",
        action="store_true",
        default=True,
        help="Prepend an unregularized (swept lambda = 0) anchor run (default). "
        "A log-spaced axis cannot contain 0, so the reference point the whole "
        "claim is relative to was missing.",
    )
    parser.add_argument(
        "--no-include-lambda-zero",
        dest="include_lambda_zero",
        action="store_false",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save pareto_front.png at end (default: off, for headless/Slurm).",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def main():
    args = parse_args()
    active_tasks = resolve_active_tasks(args.task_battery, args.tasks)
    battery_label = args.task_battery if args.tasks is None else "custom"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    noise_level = (
        args.noise_level
        if args.noise_level is not None
        else (1.0 if args.model == "yang" else DALE_DEFAULT_NOISE_LEVEL)
    )

    grid_pairs, lr_vals, lc_vals, sweep, pareto_objectives = build_grid_pairs(args)
    eval_seeds = parse_eval_seeds(args.eval_seeds)

    out_dir = Path(
        args.output_dir
        or default_output_dir(args.model, battery_label, args.n_lambda, sweep=sweep)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"

    if args.plot:
        plt.switch_backend("Agg")

    if sweep == "both":
        grid_desc = f"{args.n_lambda}x{args.n_lambda} (+anchor) = {len(grid_pairs)}"
    else:
        grid_desc = f"1D {sweep} n={len(grid_pairs)}  pareto={pareto_objectives}"
    print(
        f"Pareto sweep: model={args.model}  device={device}  "
        f"task_battery={battery_label}  tasks={len(active_tasks)}  "
        f"grid={grid_desc}  n_seeds={max(1, int(args.n_seeds))}  "
        f"output={out_dir.resolve()}"
    )

    reg_opts = {
        "rate_kind": args.rate_kind,
        "rate_inh_scale": args.rate_inh_scale,
        "conn_kind": args.conn_kind,
        "conn_target": args.conn_target,
        "conn_inh_scale": args.conn_inh_scale,
    }
    seeds = [int(args.seed) + k for k in range(max(1, int(args.n_seeds)))]
    runs_csv = out_dir / "runs.csv"

    rows = []
    runs = []
    for run_idx, (lr, lc) in enumerate(
        tqdm(grid_pairs, desc="Pareto sweep", unit="point"), start=1
    ):
        t0 = time.perf_counter()
        objs = []
        for seed in seeds:
            config = default_config(
                n_eachring=args.n_eachring, seed=seed, easy_task=True
            )
            model = make_fresh_model(args, config, device, seed=seed)

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
                loss_per_trial=args.loss_per_trial,
                freeze=args.freeze,
                **reg_opts,
            )

            obj = evaluate_pareto_metrics(
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
            objs.append(obj)

            run_row = {
                "point_idx": run_idx,
                "seed": seed,
                "lambda_rate": float(lr),
                "lambda_connectivity": float(lc),
                **{k: v for k, v in obj.items()},
            }
            runs.append(run_row)
            append_csv_row(runs_csv, run_row, write_header=(len(runs) == 1))

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_time_s = time.perf_counter() - t0
        row = aggregate_seed_runs(lr, lc, objs, train_time_s)
        row["run_idx"] = run_idx
        rows.append(row)

        spread = f" +-{row['mean_acc_std']:.3f}" if len(seeds) > 1 else ""
        tqdm.write(
            f"point {run_idx}/{len(grid_pairs)}  "
            f"lr={lr:.2e}  lc={lc:.2e}  "
            f"acc={row['mean_acc']:.3f}{spread}  min_acc={row['min_task_acc']:.3f}  "
            f"task={row['task_loss']:.4f}  meta={row['metabolic_cost']:.4f}  "
            f"wire={row['wiring_cost']:.4f}  conn={row['conn_frac']:.3f}  "
            f"time={train_time_s:.1f}s"
        )

    rows = finalize_results(
        rows,
        pareto_objectives,
        exclude_infeasible=args.exclude_infeasible,
        min_task_acc=args.feasible_min_task_acc,
    )

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
        "n_seeds": len(seeds),
        "seeds": seeds,
        "noise_level": noise_level,
        "eval_noise_level": args.eval_noise_level,
        "eval_input_noise": bool(args.eval_input_noise),
        "loss_per_trial": bool(args.loss_per_trial),
        "reg": reg_opts,
        "prune_eps": args.prune_eps,
        "lambda_zero_anchor": bool(args.include_lambda_zero),
        "sweep_mode": sweep,
        "pareto_objectives": list(pareto_objectives),
        "pareto_task_objective": args.pareto_task_objective,
        "exclude_infeasible": bool(args.exclude_infeasible),
        "feasible_min_task_acc": args.feasible_min_task_acc,
        "n_feasible": int(sum(r["is_feasible"] for r in rows)),
        "eval_seeds": list(eval_seeds),
        "eval_easy_task": True,
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
        "n_points": len(rows),
        "n_runs": len(runs),
        "n_pareto": int(sum(r["is_pareto"] for r in rows)),
        "rows": rows,
        "per_seed_rows": runs,
    }
    json_path = out_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(summary, f, indent=2)

    n_pareto = summary["n_pareto"]
    if args.exclude_infeasible and summary["n_feasible"] == 0:
        print(
            f"WARNING: no network reached min_task_acc >= {args.feasible_min_task_acc}, "
            "so the front is empty. Train longer, lower --feasible-min-task-acc, "
            "or pass --include-infeasible."
        )
    print(
        f"Done. {n_pareto}/{len(rows)} Pareto-optimal points "
        f"({len(runs)} training runs, {len(seeds)} seed(s) per point)."
    )
    print(f"Saved -> {csv_path.resolve()}")
    if len(seeds) > 1:
        print(f"Saved -> {runs_csv.resolve()}  (per-seed rows)")
    print(f"Saved -> {json_path.resolve()}")

    if args.plot:
        plot_path = out_dir / "pareto_front.png"
        plot_pareto_front(rows, plot_path, pareto_objectives)
        print(f"Saved -> {plot_path.resolve()}")


if __name__ == "__main__":
    main()
