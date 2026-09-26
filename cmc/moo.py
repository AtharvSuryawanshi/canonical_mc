"""Multi-objective search of the cost/accuracy trade-off with NSGA-III (pymoo).

Instead of a fixed lambda grid (``cmc.pareto``), NSGA-III decides which networks
to train next. Every evaluation is one full training + evaluation, the exact
functions ``cmc.pareto`` and ``cmc.zoom_lambda`` use.

Objectives (all minimised):

    1 - min_task_acc          worst-task error
    log10 metabolic_cost      log: the costs span orders of magnitude, and
    log10 wiring_cost         NSGA-III's niching works in objective space
                              (monotone, so the Pareto set is unchanged)

Constraint: min_task_acc >= --feasible-min-task-acc (pymoo ``G <= 0``), the same
feasibility rule the Pareto front uses.

Genomes (``--genome``):

    budget   x = (log10 rate budget, log10 wiring budget). Each network is
             trained under cost ceilings, no hand-set lambdas: the wiring budget
             is enforced exactly by projecting W_rec after every step
             (``train_cog.project_w_rec_to_budget``), the rate budget by a
             learned multiplier (``train_cog.BudgetConstraint``). This is the
             epsilon-constraint method, so it can reach non-convex parts of the
             front. Budgets are on the quantities the objectives measure
             (rate_reg, connectivity_reg); eval metabolic cost comes out ~10%
             above the rate budget (eval trials differ from training batches).
    lambda   x = (log10 lambda_rate, log10 lambda_connectivity), weighted-sum
             training. Only the sampling is adaptive; kept to validate the loop
             against the 6x6 grid (moo_runs/dale_core5_nsga3_lambda_p8g4_*).

Output, under ``moo_runs/<run>/``:

    runs.csv          one row per evaluated network: generation, genome, lambdas
                      or budgets (and the learned final lambdas), metrics,
                      objectives, feasibility
    summary.json      arguments and the final non-dominated set
    moo_vs_reference.png / reference check in summary.json (with --reference-run)

Resuming: pymoo's ask() is deterministic given the seed and what was told, so a
rerun with the same --output-dir replays the search and takes finished networks
from runs.csv instead of retraining them.

``--workers N`` trains up to N networks of a generation at once (one process
each, sharing the GPU); results are identical to --workers 1.

    python -m cmc.moo --steps 200 --pop-size 6 --n-gen 2 --device cpu    # smoke test
    python -m cmc.moo --points "0.0087,0.0045; 0.0144,0.0056"            # fixed budgets only
    python -m cmc.moo --reference-run pareto_runs/<6x6 run> --workers 4  # NSGA-III search
"""

import argparse
import copy
import csv
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from cmc.pareto import (
    append_csv_row,
    build_parser as build_pareto_parser,
    compute_front,
    pareto_mask,
)
from cmc.paths import MOO_RUNS_DIR
from cmc.train_cog import BUDGET_DEFAULTS
from cmc.zoom_lambda import SHARED_FLAGS, resolve_run_settings, train_and_evaluate

OBJECTIVE_NAMES = ("task_error", "log10_metabolic_cost", "log10_wiring_cost")

# Per-genome defaults, applied where the flag was left unset. The lambda genome
# keeps exactly the settings its validation run used, so that run still resumes.
GENOME_DEFAULTS = {
    "lambda": {"pop_size": 8, "n_gen": 4, "n_partitions": 2, "dedup_eps": 0.0},
    "budget": {"pop_size": 12, "n_gen": 6, "n_partitions": 3, "dedup_eps": 0.02},
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="NSGA-III search of the accuracy / metabolic / wiring trade-off."
    )
    for action in build_pareto_parser()._actions:
        if action.dest in SHARED_FLAGS - {"n_seeds"}:
            parser._add_action(copy.copy(action))
    parser.add_argument("--genome", choices=sorted(GENOME_DEFAULTS), default="budget")
    # lambda genome: the core5 6x6 sweep's box, so the two are directly comparable.
    parser.add_argument("--lambda-rate-min", type=float, default=0.02)
    parser.add_argument("--lambda-rate-max", type=float, default=1.5)
    parser.add_argument("--lambda-connectivity-min", type=float, default=1.0)
    parser.add_argument("--lambda-connectivity-max", type=float, default=40.0)
    # budget genome: brackets the costs of every network in the core5 6x6 sweep
    # and the lambda-genome run that did all tasks (metabolic 0.0047-0.055,
    # wiring 0.0045-0.0165), with room on both sides.
    parser.add_argument("--rate-budget-min", type=float, default=0.002)
    parser.add_argument("--rate-budget-max", type=float, default=0.08)
    parser.add_argument("--conn-budget-min", type=float, default=0.002)
    parser.add_argument("--conn-budget-max", type=float, default=0.025)
    parser.add_argument("--budget-lr", type=float, default=BUDGET_DEFAULTS["budget_lr"],
                        help="Step size of the log-multiplier update.")
    parser.add_argument("--budget-kp", type=float, default=BUDGET_DEFAULTS["budget_kp"],
                        help="Proportional gain of the log-multiplier controller.")
    parser.add_argument("--budget-ramp", type=float, default=BUDGET_DEFAULTS["budget_ramp"],
                        help="Fraction of training over which each budget is annealed "
                        "from the initial cost down to its target.")
    parser.add_argument("--conn-budget-mode", choices=["projection", "lagrangian"],
                        default=BUDGET_DEFAULTS["conn_budget_mode"],
                        help="Enforce the wiring budget exactly by projecting W_rec after "
                        "every step (default), or with a learned multiplier.")
    parser.add_argument("--budget-rho", type=float, default=BUDGET_DEFAULTS["budget_rho"],
                        help="Weight of the quadratic over-budget penalty.")
    parser.add_argument("--budget-lambda-init", type=float,
                        default=BUDGET_DEFAULTS["budget_lambda_init"],
                        help="Initial multiplier of each budget.")
    parser.add_argument("--budget-ema", type=float, default=BUDGET_DEFAULTS["budget_ema"],
                        help="EMA factor smoothing the violation fed to the multiplier.")

    parser.add_argument("--pop-size", type=int, default=None,
                        help="Default: 12 (budget), 8 (lambda).")
    parser.add_argument("--n-gen", type=int, default=None,
                        help="Default: 6 (budget), 4 (lambda).")
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=None,
        help="Das-Dennis partitions of the 3-objective simplex (must give no more "
        "reference directions than --pop-size). Default: 3 -> 10 directions "
        "(budget), 2 -> 6 (lambda).",
    )
    parser.add_argument(
        "--dedup-eps",
        type=float,
        default=None,
        help="Offspring closer than this (genome units, i.e. log10) to an existing "
        "candidate are discarded; 0 = exact duplicates only. Default: 0.02 "
        "(budget), 0 (lambda).",
    )
    parser.add_argument("--moo-seed", type=int, default=1, help="Seed of NSGA-III itself.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Networks trained in parallel (one process each).")
    parser.add_argument(
        "--points",
        type=str,
        default=None,
        help='Evaluate only these genomes, no search: "a,b; c,d" in the genome\'s '
        "natural units (rate_budget,conn_budget or lambda_rate,lambda_conn).",
    )
    parser.add_argument(
        "--reference-run",
        type=str,
        default=None,
        help="A cmc.pareto run dir; compare the found points against its per-seed "
        "feasible front.",
    )
    # One training seed for every genome: the objective is then a deterministic
    # function of x, and seed 0 rows exist in every reference sweep.
    parser.set_defaults(task_battery="core5", steps=4000, seed=0)
    return parser


def parse_args(argv=None):
    args = build_parser().parse_args(argv)
    for key, value in GENOME_DEFAULTS[args.genome].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def default_output_dir(args, battery_label):
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    tag = "points" if args.points else f"p{args.pop_size}g{args.n_gen}"
    return MOO_RUNS_DIR / f"{args.model}_{battery_label}_nsga3_{args.genome}_{tag}_{stamp}"


def genome_bounds(args):
    if args.genome == "lambda":
        lo = [args.lambda_rate_min, args.lambda_connectivity_min]
        hi = [args.lambda_rate_max, args.lambda_connectivity_max]
    else:
        lo = [args.rate_budget_min, args.conn_budget_min]
        hi = [args.rate_budget_max, args.conn_budget_max]
    return np.log10(lo), np.log10(hi)


def parse_points(text):
    points = []
    for chunk in (c.strip() for c in text.split(";") if c.strip()):
        a, b = (float(v) for v in chunk.split(","))
        points.append(np.log10([a, b]))
    return points


def objectives_from_metrics(metrics, feasible_min_task_acc):
    F = [
        1.0 - metrics["min_task_acc"],
        np.log10(max(metrics["metabolic_cost"], 1e-12)),
        np.log10(max(metrics["wiring_cost"], 1e-12)),
    ]
    G = [feasible_min_task_acc - metrics["min_task_acc"]]
    return F, G


def genome_key(x):
    return tuple(round(float(v), 10) for v in x)


def load_cache(runs_csv):
    """genome_key -> row, for resuming."""
    if not runs_csv.exists():
        return {}
    with runs_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
        r["is_feasible"] = r["is_feasible"] in ("True", True, 1.0)
        out[genome_key((r["x0"], r["x1"]))] = r
    return out


def evaluate_genome(args, x, generation, index):
    """Train + evaluate one network for genome ``x``; returns its runs.csv row.

    Top-level so it can run in a worker process.
    """
    active_tasks, _, device, noise_level, eval_seeds, reg_opts = resolve_run_settings(args)
    key = genome_key(x)
    a, b = (float(v) for v in 10.0 ** np.asarray(x))
    if args.genome == "lambda":
        lr, lc, budget_opts = a, b, None
        genome_cols = {"lambda_rate": lr, "lambda_connectivity": lc}
    else:
        lr = lc = 0.0
        budget_opts = {
            "rate_budget": a,
            "conn_budget": b,
            "budget_lr": args.budget_lr,
            "budget_kp": args.budget_kp,
            "budget_ramp": args.budget_ramp,
            "budget_rho": args.budget_rho,
            "budget_lambda_init": args.budget_lambda_init,
            "budget_ema": args.budget_ema,
            "conn_budget_mode": args.conn_budget_mode,
        }
        genome_cols = {"rate_budget": a, "conn_budget": b}

    # Training draws from numpy's global RNG in places; keep it from shifting
    # NSGA-III's own draws when run in-process, or a resumed run would diverge.
    rng_state = np.random.get_state()
    t0 = time.perf_counter()
    _, _, history, metrics = train_and_evaluate(
        args, lr, lc, args.seed, active_tasks, device, noise_level, eval_seeds,
        reg_opts, budget_opts,
    )
    np.random.set_state(rng_state)

    f, g = objectives_from_metrics(metrics, args.feasible_min_task_acc)
    row = {"generation": generation, "index": index, "x0": key[0], "x1": key[1],
           "seed": args.seed, **genome_cols}
    if budget_opts is not None:
        bh = history["budget"]
        tail = max(1, len(bh["rate"]["cost"]) // 10)
        row.update({
            # The multipliers the budgets settled on: directly comparable to the
            # fixed lambdas of the weighted-sum sweeps.
            "final_lambda_rate": bh["rate"]["lambda"][-1],
            "final_lambda_connectivity": bh["conn"]["lambda"][-1],
            # Training-time costs over the last 10% of steps vs. their budgets.
            "train_rate_cost_tail": float(np.mean(bh["rate"]["cost"][-tail:])),
            "train_conn_cost_tail": float(np.mean(bh["conn"]["cost"][-tail:])),
        })
    row.update({
        **metrics,
        **dict(zip(OBJECTIVE_NAMES, f)),
        "constraint_g": g[0],
        "is_feasible": bool(g[0] <= 0),
        "train_time_s": time.perf_counter() - t0,
    })
    return row


def describe(row, genome):
    if genome == "lambda":
        head = f"lambda_rate={row['lambda_rate']:.4g} lambda_conn={row['lambda_connectivity']:.4g}"
    else:
        head = (f"budget rate={row['rate_budget']:.4g} conn={row['conn_budget']:.4g} "
                f"-> lambda {row['final_lambda_rate']:.3g},{row['final_lambda_connectivity']:.3g}")
    return (
        f"gen {int(row['generation'])} #{int(row['index'])}  {head}  "
        f"min_task_acc={row['min_task_acc']:.3f}  metabolic={row['metabolic_cost']:.4g}  "
        f"wiring={row['wiring_cost']:.4g}  "
        f"{'ok' if row['is_feasible'] else 'FAILS A TASK'}  ({row['train_time_s']:.0f}s)"
    )


def evaluate_all(args, X, generation, cache, runs_csv, pool):
    """Rows for every genome in X: from the cache, or trained (in parallel)."""
    todo = [(i, x) for i, x in enumerate(X) if genome_key(x) not in cache]
    done = {}

    def record(row):
        append_csv_row(runs_csv, row, write_header=not runs_csv.exists())
        cache[genome_key((row["x0"], row["x1"]))] = row
        print(describe(row, args.genome), flush=True)

    if pool is None:
        for i, x in todo:
            record(evaluate_genome(args, x, generation, i))
    else:
        futures = [pool.submit(evaluate_genome, args, x, generation, i) for i, x in todo]
        for fut in as_completed(futures):
            record(fut.result())
    return [cache[genome_key(x)] for x in X]


def reference_check(rows, reference_dir, feasible_min_task_acc, out_png):
    """Is each feasible found network dominated by a reference network (same seed)?"""
    ref_csv = Path(reference_dir) / "runs.csv"
    with ref_csv.open(newline="") as f:
        ref = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    objs = ("min_task_acc", "metabolic_cost", "wiring_cost")
    maximize = [True, False, False]

    def to_min(r):
        return np.array([-r[objs[0]], r[objs[1]], r[objs[2]]])

    ref = [r for r in ref if r["min_task_acc"] >= feasible_min_task_acc]
    ref_seed = [r for r in ref if int(r["seed"]) == 0]
    found = [r for r in rows if r["is_feasible"]]
    ref_front = [r for r, p in zip(ref_seed, pareto_mask(np.array([[r[k] for k in objs] for r in ref_seed]), maximize=maximize)) if p] if ref_seed else []

    dominated = 0
    for r in found:
        a = to_min(r)
        if any(np.all(to_min(q) <= a) and np.any(to_min(q) < a) for q in ref_seed):
            dominated += 1
    result = {
        "reference_run": str(reference_dir),
        "n_found_feasible": len(found),
        "n_reference_feasible_seed0": len(ref_seed),
        "n_found_dominated_by_reference_seed0": dominated,
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [("metabolic_cost", "wiring_cost"), ("metabolic_cost", "min_task_acc"), ("wiring_cost", "min_task_acc")]
    for ax, (a, b) in zip(axes, pairs):
        ax.scatter([r[a] for r in ref_seed], [r[b] for r in ref_seed], s=25, c="lightgrey", label="6x6 seed 0, feasible")
        ax.scatter([r[a] for r in ref_front], [r[b] for r in ref_front], s=45, facecolors="none", edgecolors="k", label="6x6 seed 0 front")
        sc = ax.scatter([r[a] for r in found], [r[b] for r in found], s=35, c=[r["generation"] for r in found], cmap="viridis", label="NSGA-III, feasible")
        ax.set_xscale("log")
        if b != "min_task_acc":
            ax.set_yscale("log")
        ax.set_xlabel(a)
        ax.set_ylabel(b)
    fig.colorbar(sc, ax=axes, label="generation")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"NSGA-III vs reference: {dominated}/{len(found)} found points dominated by a seed-0 reference network")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return result


def run_search(args, problem, cache, runs_csv, pool):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.duplicate import DefaultDuplicateElimination
    from pymoo.core.evaluator import Evaluator
    from pymoo.problems.static import StaticProblem
    from pymoo.util.ref_dirs import get_reference_directions

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=args.n_partitions)
    if len(ref_dirs) > args.pop_size:
        raise SystemExit(f"--pop-size {args.pop_size} < {len(ref_dirs)} reference directions")
    extra = {}
    if args.dedup_eps > 0:
        extra["eliminate_duplicates"] = DefaultDuplicateElimination(epsilon=args.dedup_eps)
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=args.pop_size, **extra)
    algorithm.setup(problem, termination=("n_gen", args.n_gen), seed=args.moo_seed, verbose=False)
    print(f"  NSGA-III pop={args.pop_size} gens={args.n_gen} ref_dirs={len(ref_dirs)} "
          f"dedup_eps={args.dedup_eps} workers={args.workers}")

    rows, gen = [], 0
    while algorithm.has_next():
        pop = algorithm.ask()
        gen_rows = evaluate_all(args, pop.get("X"), gen, cache, runs_csv, pool)
        rows.extend(gen_rows)
        F = np.array([[r[k] for k in OBJECTIVE_NAMES] for r in gen_rows])
        G = np.array([[r["constraint_g"]] for r in gen_rows])
        Evaluator().eval(StaticProblem(problem, F=F, G=G), pop)
        algorithm.tell(infills=pop)
        gen += 1
    return rows


def main(argv=None):
    from pymoo.core.problem import Problem

    args = parse_args(argv)
    active_tasks, battery_label, device, _, _, _ = resolve_run_settings(args)
    out_dir = Path(args.output_dir or default_output_dir(args, battery_label))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    cache = load_cache(runs_csv)

    xl, xu = genome_bounds(args)
    problem = Problem(n_var=2, n_obj=3, n_ieq_constr=1, xl=xl, xu=xu)
    print(
        f"cmc.moo: genome={args.genome} tasks={active_tasks} steps={args.steps} "
        f"device={device}\n  output={out_dir.resolve()}  cached={len(cache)}"
    )

    pool = None
    if args.workers > 1:
        # spawn, not fork: CUDA cannot be re-initialised in a forked child.
        pool = ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        )
    try:
        if args.points:
            rows = evaluate_all(args, parse_points(args.points), -1, cache, runs_csv, pool)
        else:
            rows = run_search(args, problem, cache, runs_csv, pool)
    finally:
        if pool is not None:
            pool.shutdown()

    # Final front over everything evaluated (not just the last population).
    unique = list({genome_key((r["x0"], r["x1"])): r for r in rows}.values())
    is_pareto, _ = compute_front(
        unique, ("min_task_acc", "metabolic_cost", "wiring_cost"),
        exclude_infeasible=True, min_task_acc=args.feasible_min_task_acc,
    )
    front = [r for r, p in zip(unique, is_pareto) if p]
    front_keys = ["lambda_rate", "lambda_connectivity", "rate_budget", "conn_budget",
                  "final_lambda_rate", "final_lambda_connectivity", "min_task_acc",
                  "metabolic_cost", "wiring_cost", "conn_frac", "generation"]
    summary = {
        "genome": args.genome,
        "model": args.model,
        "task_battery": battery_label,
        "active_tasks": list(active_tasks),
        "n_evaluated": len(unique),
        "n_feasible": sum(r["is_feasible"] for r in unique),
        "front": [{k: r[k] for k in front_keys if k in r} for r in front],
        "args": vars(args),
    }
    if args.reference_run:
        summary["reference_check"] = reference_check(
            unique, args.reference_run, args.feasible_min_task_acc, out_dir / "moo_vs_reference.png"
        )
        print(f"Reference check: {summary['reference_check']}")
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"{summary['n_feasible']}/{len(unique)} feasible, {len(front)} on the front. "
          f"Saved -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()
