"""Pick zoom points from a finished ``cmc.moo`` budget front, for ``cmc.moo_zoom --points``.

Points (the design of ``cmc.moo_zoom``'s defaults, picked automatically):

    control          no budget
    rate_limited     on the front, inside the accuracy band: the lowest metabolic cost
    wiring_limited   on the front, inside the accuracy band: the lowest wiring cost
    middle           (--middle) inside the band: the most balanced, i.e. the smallest
                     max(rank by rate, rank by wiring)
    knee             pymoo HighTradeoffPoints on the feasible front (best compromise)

The accuracy band ("iso-accuracy", default 0.60-0.70 min_task_acc) keeps the
rate- and wiring-limited networks equally competent, so a difference between them
is about *which* cost binds, not about accuracy.

    python -m cmc.moo_pick runs/moo/<run>                      # prints the --points string
    python -m cmc.moo_pick runs/moo/<run> --acc-band 0.65,0.75 --middle
    python -m cmc.moo_pick runs/moo/<run> --all-front          # every front network, to sample the front
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cmc.front import FEASIBLE_MIN_TASK_ACC, compute_front

OBJ = ("min_task_acc", "metabolic_cost", "wiring_cost")


def load_front(run_dir, feasible_min_task_acc=FEASIBLE_MIN_TASK_ACC):
    df = pd.read_csv(Path(run_dir) / "runs.csv")
    if "rate_budget" not in df:
        raise SystemExit("not a budget-genome run (no rate_budget column)")
    df = df.drop_duplicates(["x0", "x1"]).reset_index(drop=True)
    df["is_feasible"] = df.min_task_acc >= feasible_min_task_acc
    df["is_pareto"], _ = compute_front(df.to_dict("records"), OBJ, exclude_infeasible=True,
                                       min_task_acc=feasible_min_task_acc)
    return df


def knee_index(front):
    from pymoo.mcdm.high_tradeoff import HighTradeoffPoints

    F = np.column_stack([1 - front.min_task_acc, np.log10(front.metabolic_cost), np.log10(front.wiring_cost)])
    F = (F - F.min(0)) / np.maximum(F.max(0) - F.min(0), 1e-12)
    idx = HighTradeoffPoints()(F)
    if idx is None or len(idx) == 0:
        # fall back to the point closest to the ideal corner
        return int(np.argmin(np.linalg.norm(F, axis=1)))
    # several candidates: the first, as moo_pareto_analysis.ipynb section 2.7 ("knee0")
    return int(np.atleast_1d(idx)[0])


def pick(df, acc_band=(0.60, 0.70), middle=False):
    front = df[df.is_pareto].reset_index(drop=True)
    lo, hi = acc_band
    band = front[(front.min_task_acc >= lo) & (front.min_task_acc <= hi)]
    if len(band) < 2:
        raise SystemExit(f"only {len(band)} front points with {lo} <= min_task_acc <= {hi}; widen --acc-band "
                         f"(front accuracies: {sorted(front.min_task_acc.round(3))})")
    picks = {"rate_limited": band.metabolic_cost.idxmin(), "wiring_limited": band.wiring_cost.idxmin()}
    if middle:
        balance = np.maximum(band.metabolic_cost.rank(), band.wiring_cost.rank())
        rest = balance.drop(list(picks.values()), errors="ignore")
        if len(rest):
            picks["middle"] = rest.idxmin()
    picks["knee"] = knee_index(front)
    rows = {name: front.loc[i] for name, i in picks.items()}
    return front, rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("--acc-band", default="0.60,0.70", help="min_task_acc band for the iso-accuracy points")
    p.add_argument("--middle", action="store_true", help="also pick a balanced point inside the band")
    p.add_argument("--all-front", action="store_true",
                   help="instead: every front network, named f00, f01, ... from sparsest to densest wiring "
                   "(to sample the whole front)")
    p.add_argument("--feasible-min-task-acc", type=float, default=FEASIBLE_MIN_TASK_ACC)
    args = p.parse_args(argv)
    lo, hi = (float(v) for v in args.acc_band.split(","))

    df = load_front(args.run_dir, args.feasible_min_task_acc)
    if args.all_front:
        front = df[df.is_pareto].sort_values(["conn_budget", "rate_budget"]).reset_index(drop=True)
        rows = {f"f{i:02d}": r for i, r in front.iterrows()}
    else:
        front, rows = pick(df, (lo, hi), args.middle)
    print(f"{len(df)} networks, {int(df.is_feasible.sum())} feasible, {len(front)} on the front")
    print(f"{'point':15s} {'rate_budget':>11s} {'conn_budget':>11s} {'min_acc':>8s} {'metabolic':>10s} "
          f"{'wiring':>8s} {'gen':>4s}")
    for name, r in rows.items():
        print(f"{name:15s} {r.rate_budget:11.4g} {r.conn_budget:11.4g} {r.min_task_acc:8.3f} "
              f"{r.metabolic_cost:10.4g} {r.wiring_cost:8.4g} {int(r.generation):4d}")
    points = ["control=none"] + [f"{n}={float(r.rate_budget)!r},{float(r.conn_budget)!r}" for n, r in rows.items()]
    print("\n--points \"" + "; ".join(points) + "\"")


if __name__ == "__main__":
    main()
