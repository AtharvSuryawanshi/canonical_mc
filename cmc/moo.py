"""Multi-objective search of the cost/accuracy trade-off with NSGA-III (pymoo).

Instead of a fixed lambda grid (``cmc.pareto``), NSGA-III decides which networks
to train next. Every evaluation is one full training + evaluation, the exact
functions ``cmc.pareto`` and ``cmc.zoom_lambda`` use, so a point found here is
the same network a sweep would train at that (lambda, seed).

Objectives (all minimised):

    1 - min_task_acc          worst-task error
    log10 metabolic_cost      log: the costs span orders of magnitude, and
    log10 wiring_cost         NSGA-III's niching works in objective space
                              (monotone, so the Pareto set is unchanged)

Constraint: min_task_acc >= --feasible-min-task-acc (pymoo ``G <= 0``), the same
feasibility rule the Pareto front uses.

Genomes (``--genome``):

    lambda   x = (log10 lambda_rate, log10 lambda_connectivity). Still a weighted
             sum per training -- only the *sampling* is adaptive, so this cannot
             reach non-convex parts of the front. Its purpose is to validate the
             loop against the known 6x6 front (``--reference-run``).
    budget   (planned) x = (rate budget, wiring budget), constrained training.

Output, under ``moo_runs/<run>/``:

    runs.csv          one row per evaluated network: generation, genome, lambdas,
                      metrics, objectives, feasibility
    summary.json      arguments and the final non-dominated set
    moo_vs_reference.png / reference check in summary.json (with --reference-run)

Resuming: pymoo's ask() is deterministic given the seed and what was told, so a
rerun with the same --output-dir replays the search and takes finished networks
from runs.csv instead of retraining them.

    python -m cmc.moo --steps 200 --pop-size 4 --n-gen 2 --device cpu   # smoke test
    python -m cmc.moo --reference-run pareto_runs/<6x6 run>             # validation
"""

import argparse
import copy
import csv
import json
import time
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
from cmc.zoom_lambda import SHARED_FLAGS, resolve_run_settings, train_and_evaluate

OBJECTIVE_NAMES = ("task_error", "log10_metabolic_cost", "log10_wiring_cost")


def build_parser():
    parser = argparse.ArgumentParser(
        description="NSGA-III search of the accuracy / metabolic / wiring trade-off."
    )
    for action in build_pareto_parser()._actions:
        if action.dest in SHARED_FLAGS - {"n_seeds"}:
            parser._add_action(copy.copy(action))
    parser.add_argument("--genome", choices=["lambda"], default="lambda")
    # Same search box as the core5 6x6 sweep, so the two are directly comparable.
    parser.add_argument("--lambda-rate-min", type=float, default=0.02)
    parser.add_argument("--lambda-rate-max", type=float, default=1.5)
    parser.add_argument("--lambda-connectivity-min", type=float, default=1.0)
    parser.add_argument("--lambda-connectivity-max", type=float, default=40.0)
    parser.add_argument("--pop-size", type=int, default=8)
    parser.add_argument("--n-gen", type=int, default=4)
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=2,
        help="Das-Dennis partitions of the 3-objective simplex; 2 -> 6 reference "
        "directions (must not exceed --pop-size).",
    )
    parser.add_argument("--moo-seed", type=int, default=1, help="Seed of NSGA-III itself.")
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


def default_output_dir(model, battery_label, genome, pop_size, n_gen):
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return MOO_RUNS_DIR / f"{model}_{battery_label}_nsga3_{genome}_p{pop_size}g{n_gen}_{stamp}"


def genome_bounds(args):
    xl = np.log10([args.lambda_rate_min, args.lambda_connectivity_min])
    xu = np.log10([args.lambda_rate_max, args.lambda_connectivity_max])
    return xl, xu


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
            if k not in ("genome",):
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
        r["is_feasible"] = r["is_feasible"] in ("True", True, 1.0)
        out[genome_key((r["x0"], r["x1"]))] = r
    return out


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


def main(argv=None):
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.core.evaluator import Evaluator
    from pymoo.core.problem import Problem
    from pymoo.problems.static import StaticProblem
    from pymoo.util.ref_dirs import get_reference_directions

    args = build_parser().parse_args(argv)
    active_tasks, battery_label, device, noise_level, eval_seeds, reg_opts = (
        resolve_run_settings(args)
    )
    out_dir = Path(args.output_dir or default_output_dir(
        args.model, battery_label, args.genome, args.pop_size, args.n_gen
    ))
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = out_dir / "runs.csv"
    cache = load_cache(runs_csv)
    done_rows = runs_csv.exists()

    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=args.n_partitions)
    if len(ref_dirs) > args.pop_size:
        raise SystemExit(f"--pop-size {args.pop_size} < {len(ref_dirs)} reference directions")
    xl, xu = genome_bounds(args)
    problem = Problem(n_var=2, n_obj=3, n_ieq_constr=1, xl=xl, xu=xu)
    algorithm = NSGA3(ref_dirs=ref_dirs, pop_size=args.pop_size)
    algorithm.setup(problem, termination=("n_gen", args.n_gen), seed=args.moo_seed, verbose=False)

    print(
        f"cmc.moo: NSGA-III genome={args.genome} pop={args.pop_size} gens={args.n_gen} "
        f"ref_dirs={len(ref_dirs)} tasks={active_tasks} steps={args.steps} "
        f"device={device}\n  output={out_dir.resolve()}  cached={len(cache)}"
    )

    rows, gen = [], 0
    while algorithm.has_next():
        pop = algorithm.ask()
        X = pop.get("X")
        F, G = [], []
        for i, x in enumerate(X):
            key = genome_key(x)
            if key in cache:
                row = cache[key]
            else:
                lr, lc = 10.0 ** x
                # Training draws from numpy's global RNG in places; keep it from
                # shifting NSGA-III's own draws, or a resumed run would diverge.
                rng_state = np.random.get_state()
                t0 = time.perf_counter()
                _, _, _, metrics = train_and_evaluate(
                    args, lr, lc, args.seed, active_tasks, device, noise_level, eval_seeds, reg_opts
                )
                np.random.set_state(rng_state)
                f, g = objectives_from_metrics(metrics, args.feasible_min_task_acc)
                row = {
                    "generation": gen,
                    "index": i,
                    "x0": key[0],
                    "x1": key[1],
                    "seed": args.seed,
                    "lambda_rate": float(lr),
                    "lambda_connectivity": float(lc),
                    **metrics,
                    **dict(zip(OBJECTIVE_NAMES, f)),
                    "constraint_g": g[0],
                    "is_feasible": bool(g[0] <= 0),
                    "train_time_s": time.perf_counter() - t0,
                }
                append_csv_row(runs_csv, row, write_header=not done_rows)
                done_rows = True
                cache[key] = row
                print(
                    f"gen {gen} #{i}  lambda_rate={lr:.4g} lambda_conn={lc:.4g}  "
                    f"min_task_acc={metrics['min_task_acc']:.3f}  "
                    f"metabolic={metrics['metabolic_cost']:.4g}  wiring={metrics['wiring_cost']:.4g}  "
                    f"{'ok' if row['is_feasible'] else 'FAILS A TASK'}  ({row['train_time_s']:.0f}s)",
                    flush=True,
                )
            rows.append(row)
            F.append([row[k] for k in OBJECTIVE_NAMES])
            G.append([row["constraint_g"]])
        Evaluator().eval(StaticProblem(problem, F=np.array(F), G=np.array(G)), pop)
        algorithm.tell(infills=pop)
        gen += 1

    # Final front over everything evaluated (not just the last population).
    unique = list({genome_key((r["x0"], r["x1"])): r for r in rows}.values())
    is_pareto, _ = compute_front(
        unique, ("min_task_acc", "metabolic_cost", "wiring_cost"),
        exclude_infeasible=True, min_task_acc=args.feasible_min_task_acc,
    )
    front = [r for r, p in zip(unique, is_pareto) if p]
    summary = {
        "genome": args.genome,
        "model": args.model,
        "task_battery": battery_label,
        "active_tasks": list(active_tasks),
        "n_evaluated": len(unique),
        "n_feasible": sum(r["is_feasible"] for r in unique),
        "front": [
            {k: r[k] for k in ("lambda_rate", "lambda_connectivity", "min_task_acc",
                               "metabolic_cost", "wiring_cost", "conn_frac", "generation")}
            for r in front
        ],
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
