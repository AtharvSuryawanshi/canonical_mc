"""Zoom into a few cost-budget points of the NSGA-III front and SAVE every network.

``cmc.moo`` finds *where* the budget front is, with one seed per budget and
metrics only. This script trains many seeds at a handful of chosen budget
points and keeps every network, so connectivity motifs can be inspected. It is
the budget counterpart of ``cmc.lambda_zoom``.

Training is ``runner.train_and_evaluate`` with the same budget enforcement as
``cmc.moo`` (``--budget-*`` flags, same defaults), so seed 0 at a point is the
network ``cmc.moo`` trained there -- bit-identical on CPU; on GPU only up to
CUDA nondeterminism (seen: |d acc| <= 0.3 seed std). A point with no budgets (``control``)
is the unconstrained network, identical to the lambda grid's lambda=0 anchor.

Output, under ``runs/moo_zoom/<run>/``:

    <point>/seed_00.pt ...   one checkpoint per network (loadable with
                             cmc.train_cog.load_checkpoint): weights, budgets,
                             the training history incl. the learned rate
                             multiplier and cost per step, eval metrics,
                             feasibility, per-neuron task variance / mean rate
    runs.csv                 one row per network: point, seed, budgets, metrics
    summary.json             arguments, the points, feasible count per point

    python -m cmc.moo_zoom                                  # default 5 points x 10 seeds
    python -m cmc.moo_zoom --points "a=0.01,0.005; control=none" --n-seeds 5 --workers 4
    python -m cmc.moo_zoom --steps 32000 --save-at 4000,8000,16000   # training-length series

With ``--save-at`` the network is also saved at those steps, under
``<point>/step_XXXXXX/seed_XX.pt``, with one row per saved step in
``runs_by_step.csv``. A run stopped mid-training restarts that network from
step 0 when resumed (only finished networks are skipped).

``--batched`` trains up to ``--batch-pop`` networks together in one process
(``cmc.batched``), much faster on a GPU; same output layout. The networks of a
batch share one trial stream, so they are not bit-identical to the default
runs (and seed 0 no longer reproduces the ``cmc.moo`` network).

    python -m cmc.moo_zoom --batched --batch-pop 50 --steps 40000 --task-battery all
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cmc.batched import add_batched_args
from cmc.moo import add_budget_args, budget_opts_from_args
from cmc.paths import MOO_ZOOM_RUNS_DIR
from cmc.runner import (
    activity_summary,
    add_common_args,
    append_csv_row,
    resolve_run_settings,
    train_and_evaluate,
)
from cmc.train_cog import _model_kwargs_from_args, evaluate_pareto_metrics, save_checkpoint

# A walk along the iso-accuracy curve (min_task_acc ~0.65-0.72) of
# runs/moo/dale_core5_nsga3_budget_p24g10 (notebooks/moo_pareto_analysis.ipynb
# section 2.2), plus the unconstrained control and the knee (section 2.7).
# Along the walk the networks are about equally competent and differ in *which*
# cost binds, so a structural difference between them is attributable to the
# firing-vs-wiring exchange rather than to accuracy. Budgets are the exact
# values cmc.moo trained with (runs.csv rows 87, 39, 38, 101), acc = its seed 0.
#
#   control         no budget                  acc ~0.85  structure from the tasks alone
#   rate_limited    rate 0.00213, wire 0.0228  acc 0.66   cheap firing, dense wiring
#   middle          rate 0.00612, wire 0.0083  acc 0.67   both moderately tight
#   wiring_limited  rate 0.0322,  wire 0.0027  acc 0.65   sparse wiring, high firing
#   knee            rate 0.0199,  wire 0.0045  acc 0.72   best compromise
DEFAULT_POINTS = (
    ("control", None, None),
    ("rate_limited", 0.0021304559660999, 0.0228159943899826),
    ("middle", 0.0061202096772596, 0.0083127308558332),
    ("wiring_limited", 0.0322022890202443, 0.0027038098412398),
    ("knee", 0.0199479117560131, 0.0045022532148749),
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train many seeds at chosen cost-budget points and save every network."
    )
    add_common_args(parser)
    add_budget_args(parser)
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help='Semicolon-separated "name=rate_budget,conn_budget" (name optional); '
        '"name=none" trains without budgets. Default: control / rate_limited / '
        "middle / wiring_limited / knee from the core5 budget front.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain networks whose checkpoint already exists (default: skip them, "
        "so a job that hit its time limit can simply be resubmitted).",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Networks trained in parallel (one process each). With "
                        "--batched: batches trained in parallel (one process each).")
    add_batched_args(parser)
    parser.add_argument(
        "--save-at",
        type=str,
        default=None,
        help='Comma-separated training steps at which to also save the network, e.g. '
        '"4000,8000,16000" with --steps 32000: one run gives the whole training-length '
        "series. Saved as <point>/step_XXXXXX/seed_XX.pt, one row each in runs_by_step.csv "
        "(the final step included). Training is unaffected (eval uses its own RNGs).",
    )
    parser.set_defaults(task_battery="core5", steps=4000, n_seeds=10)
    return parser


def parse_points(text):
    if text is None or not text.strip():
        return list(DEFAULT_POINTS)
    points = []
    for i, chunk in enumerate(c.strip() for c in text.split(";") if c.strip()):
        name, _, vals = chunk.rpartition("=")
        name = name.strip() or f"p{i}"
        if vals.strip().lower() == "none":
            points.append((name, None, None))
        else:
            rb, cb = (float(v) for v in vals.split(","))
            points.append((name, rb, cb))
    names = [p[0] for p in points]
    if len(set(names)) != len(names):
        raise ValueError(f"--points: duplicate names {names}")
    return points


def default_output_dir(model, battery_label, n_points, n_seeds):
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return MOO_ZOOM_RUNS_DIR / f"{model}_{battery_label}_{n_points}pt_{n_seeds}seed_{stamp}"


def _summarize(args, name, rate_budget, conn_budget, seed, history, metrics, budget_opts, ckpt_rel, t0):
    """runs.csv row for a network: point, budgets, learned multiplier, metrics."""
    row = {
        "point": name,
        "seed": seed,
        "rate_budget": rate_budget if rate_budget is not None else float("nan"),
        "conn_budget": conn_budget if conn_budget is not None else float("nan"),
    }
    if budget_opts is not None:
        bh = history["budget"]
        tail = max(1, len(bh["rate"]["cost"]) // 10)
        row.update({
            "final_lambda_rate": bh["rate"]["lambda"][-1],
            "train_rate_cost_tail": float(np.mean(bh["rate"]["cost"][-tail:])),
            "train_conn_cost_tail": float(np.mean(bh["conn"]["cost"][-tail:])),
        })
    else:
        row.update({"final_lambda_rate": 0.0, "train_rate_cost_tail": float("nan"),
                    "train_conn_cost_tail": float("nan")})
    row.update({
        **metrics,
        "is_feasible": bool(metrics["min_task_acc"] >= args.feasible_min_task_acc),
        "checkpoint": ckpt_rel,
        "train_time_s": time.perf_counter() - t0,
    })
    return row


def _save(args, ckpt_path, model, config, active_tasks, seed, steps, history, name, rate_budget,
          conn_budget, budget_opts, metrics, row, reg_opts, noise_level, eval_seeds, task_var, mean_rate):
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
        train_steps=steps,
        history=history,  # includes history["budget"]: learned multiplier + cost per step
        verbose=False,
        extra={
            "point": name,
            "rate_budget": rate_budget,
            "conn_budget": conn_budget,
            "budget_opts": budget_opts,
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


def step_checkpoint_path(out_dir, name, seed, step):
    """Intermediate network saved by --save-at: <point>/step_XXXXXX/seed_XX.pt."""
    return Path(out_dir) / name / f"step_{step:06d}" / f"seed_{seed:02d}.pt"


def train_point(args, name, rate_budget, conn_budget, seed, out_dir):
    """Train, evaluate and save one network.

    Returns (row, step_rows): its runs.csv row and one row per saved step
    (``--save-at`` steps plus the final one, with a ``step`` column).
    Top-level so it can run in a worker process.
    """
    settings = resolve_run_settings(args)
    active_tasks, _, device, noise_level, eval_seeds, reg_opts = settings
    out_dir = Path(out_dir)
    ckpt_path = out_dir / name / f"seed_{seed:02d}.pt"
    budget_opts = None
    if rate_budget is not None:
        budget_opts = budget_opts_from_args(args, rate_budget, conn_budget)
    save_at = set(parse_save_at(args.save_at, args.steps))
    step_rows = []
    t0 = time.perf_counter()
    job = (name, rate_budget, conn_budget, seed, budget_opts)

    def on_step(step, history, model, config):
        if step in save_at:
            step_rows.append(_evaluate_and_save(args, settings, out_dir, job, model, config, history, step,
                                                step_checkpoint_path(out_dir, name, seed, step), t0))

    model, config, history, metrics = train_and_evaluate(
        args, 0.0, 0.0, seed, active_tasks, device, noise_level, eval_seeds, reg_opts, budget_opts,
        step_callback=on_step if save_at else None,
    )
    step_rows.append(_evaluate_and_save(args, settings, out_dir, job, model, config, history, args.steps,
                                        ckpt_path, t0, metrics))
    return {k: v for k, v in step_rows[-1].items() if k != "step"}, step_rows


def train_points_batched(args, jobs, out_dir, device=None):
    """``train_point`` for several (name, rate_budget, conn_budget, seed) jobs trained
    together (``--batched``). Returns one (row, step_rows) per job; ``train_time_s``
    counts from the start of the batch. Top-level so it can run in a worker process.
    """
    from cmc.batched import args_on_device, train_and_evaluate_batched

    if device is not None:
        args = args_on_device(args, device)
    settings = resolve_run_settings(args)
    active_tasks, _, device, noise_level, eval_seeds, reg_opts = settings
    out_dir = Path(out_dir)
    save_at = parse_save_at(args.save_at, args.steps)
    full_jobs, specs = [], []
    for name, rb, cb, seed in jobs:
        budget_opts = None if rb is None else budget_opts_from_args(args, rb, cb)
        full_jobs.append((name, rb, cb, seed, budget_opts))
        specs.append({"seed": seed, "rate_budget": rb, "conn_budget": cb})
    step_rows = [[] for _ in jobs]
    t0 = time.perf_counter()

    def on_step(step, models, configs, histories):
        for p, job in enumerate(full_jobs):
            path = step_checkpoint_path(out_dir, job[0], job[3], step)
            step_rows[p].append(_evaluate_and_save(args, settings, out_dir, job, models[p], configs[p],
                                                   histories[p], step, path, t0))

    results = train_and_evaluate_batched(args, specs, active_tasks, device, noise_level, eval_seeds, reg_opts,
                                         callback_steps=save_at, step_callback=on_step if save_at else None)
    out = []
    for p, (job, (model, config, history, metrics)) in enumerate(zip(full_jobs, results)):
        path = out_dir / job[0] / f"seed_{job[3]:02d}.pt"
        step_rows[p].append(_evaluate_and_save(args, settings, out_dir, job, model, config, history,
                                               args.steps, path, t0, metrics))
        out.append(({k: v for k, v in step_rows[p][-1].items() if k != "step"}, step_rows[p]))
    return out


def _evaluate_and_save(args, settings, out_dir, job, model, config, history, step, path, t0, metrics=None):
    """Noise-free eval on the fixed eval seeds (own RNGs, so training is unaffected),
    task variance, then save. Returns the runs_by_step row (runs.csv row + ``step``)."""
    active_tasks, _, device, noise_level, eval_seeds, reg_opts = settings
    name, rate_budget, conn_budget, seed, budget_opts = job
    if metrics is None:
        metrics = evaluate_pareto_metrics(
            model, config, active_tasks, device, eval_seeds=eval_seeds,
            batch_size=args.eval_batch_size, noise_level=args.eval_noise_level,
            input_noise=args.eval_input_noise, easy_task=True, **reg_opts,
        )
    task_var, mean_rate = activity_summary(
        model, config, active_tasks, device, eval_seeds, args.eval_batch_size
    )
    row = _summarize(args, name, rate_budget, conn_budget, seed, history, metrics, budget_opts,
                     path.relative_to(out_dir).as_posix(), t0)
    _save(args, path, model, config, active_tasks, seed, step, history, name, rate_budget,
          conn_budget, budget_opts, metrics, row, reg_opts, noise_level, eval_seeds, task_var, mean_rate)
    return {"step": step, **row}


def parse_save_at(text, steps):
    """'4000,8000' -> [4000, 8000]; steps must lie strictly before the final step."""
    if not text:
        return []
    vals = sorted({int(v) for v in str(text).split(",") if v.strip()})
    bad = [v for v in vals if not 0 < v < steps]
    if bad:
        raise ValueError(f"--save-at {bad}: must be between 0 and --steps ({steps}), exclusive")
    return vals


def main(argv=None):
    args = build_parser().parse_args(argv)
    points = parse_points(args.points)
    active_tasks, battery_label, device, _, _, _ = resolve_run_settings(args)
    seeds = [int(args.seed) + k for k in range(max(1, int(args.n_seeds)))]

    out_dir = Path(args.output_dir or default_output_dir(args.model, battery_label, len(points), len(seeds)))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    steps_csv = out_dir / "runs_by_step.csv"
    parse_save_at(args.save_at, args.steps)  # fail fast on bad steps
    if args.overwrite:
        for f in (runs_csv, steps_csv):
            if f.exists():
                f.unlink()

    print(
        f"moo_zoom: {len(points)} points x {len(seeds)} seeds = "
        f"{len(points) * len(seeds)} networks  tasks={active_tasks}  "
        f"steps={args.steps}  device={device}  workers={args.workers}\n"
        f"  output={out_dir.resolve()}"
    )
    for name, rb, cb in points:
        budgets = "no budget" if rb is None else f"rate_budget={rb:.4g}  conn_budget={cb:.4g}"
        print(f"  {name:15s} {budgets}")

    rows, todo = [], []
    for name, rb, cb in points:
        for seed in seeds:
            ckpt_path = out_dir / name / f"seed_{seed:02d}.pt"
            if ckpt_path.exists() and not args.overwrite:
                ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                rows.append(ck["run_row"])
            else:
                todo.append((name, rb, cb, seed))
    print(f"  {len(rows)} already trained, {len(todo)} to train")

    def record(result):
        row, step_rows = result
        append_csv_row(runs_csv, row, write_header=not runs_csv.exists())
        if args.save_at:
            for sr in step_rows:
                append_csv_row(steps_csv, sr, write_header=not steps_csv.exists())
        rows.append(row)
        print(
            f"{row['point']:15s} seed {row['seed']:2d}  min_task_acc={row['min_task_acc']:.3f}  "
            f"metabolic={row['metabolic_cost']:.4g}  wiring={row['wiring_cost']:.4g}  "
            f"conn_frac={row['conn_frac']:.3f}  "
            f"{'ok' if row['is_feasible'] else 'FAILS A TASK'}  ({row['train_time_s']:.0f}s)",
            flush=True,
        )

    if args.batched and todo:
        from cmc.batched import chunked, parse_devices

        chunks = chunked(todo, args.batch_pop)
        devices = parse_devices(args.devices) or [args.device]
        if args.workers > 1:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
                futures = [pool.submit(train_points_batched, args, chunk, str(out_dir), devices[k % len(devices)])
                           for k, chunk in enumerate(chunks)]
                for fut in as_completed(futures):
                    for result in fut.result():
                        record(result)
        else:
            for chunk in chunks:
                for result in train_points_batched(args, chunk, str(out_dir)):
                    record(result)
    elif args.workers > 1 and todo:
        # spawn, not fork: CUDA cannot be re-initialised in a forked child.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            futures = [pool.submit(train_point, args, *job, str(out_dir)) for job in todo]
            for fut in as_completed(futures):
                record(fut.result())
    else:
        for job in todo:
            record(train_point(args, *job, str(out_dir)))

    per_point = {
        name: {
            "rate_budget": rb,
            "conn_budget": cb,
            "n_trained": sum(r["point"] == name for r in rows),
            "n_feasible": sum(r["point"] == name and r["is_feasible"] for r in rows),
        }
        for name, rb, cb in points
    }
    summary = {
        "model": args.model,
        "task_battery": battery_label,
        "active_tasks": list(active_tasks),
        "steps": args.steps,
        "seeds": seeds,
        "save_at": parse_save_at(args.save_at, args.steps),
        "points": per_point,
        "feasible_min_task_acc": args.feasible_min_task_acc,
        "args": vars(args),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nFeasible networks per point (min_task_acc >= {args.feasible_min_task_acc}):")
    for name, info in per_point.items():
        print(f"  {name:15s} {info['n_feasible']}/{info['n_trained']}")
    print(f"Saved -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
