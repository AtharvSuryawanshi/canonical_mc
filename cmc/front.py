"""Pareto dominance and the feasibility rule, shared by every sweep and search.

Nothing here knows how a network was obtained (lambda grid, NSGA-III, ...): it
takes rows of metrics and says which rows are feasible and which are on the front.
"""

import numpy as np


# pareto_maximize_flags() falls back to False (minimize) for unknown names, so a
# missing entry here silently inverts the front. min_task_acc in particular is
# the neuroscience-faithful task objective (mean_acc lets the network abandon a
# hard task and still look good), and it was absent.
OBJECTIVE_MAXIMIZE = {
    "mean_acc": True,
    "min_task_acc": True,
    "task_loss": False,
    "metabolic_cost": False,
    "wiring_cost": False,
    "conn_frac": False,
    "wiring_cost_w_in_l2": False,
}


def pareto_maximize_flags(pareto_objectives):
    return tuple(OBJECTIVE_MAXIMIZE.get(name, False) for name in pareto_objectives)


def pareto_mask(objectives, maximize=None):
    """Mark nondominated rows. objectives: (n, k); maximize per column where True."""
    n, k = objectives.shape
    if maximize is None:
        maximize = (False,) * k
    else:
        maximize = tuple(maximize)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            better_or_equal = True
            strictly_better = False
            for d in range(k):
                if maximize[d]:
                    if objectives[j, d] < objectives[i, d]:
                        better_or_equal = False
                        break
                    if objectives[j, d] > objectives[i, d]:
                        strictly_better = True
                else:
                    if objectives[j, d] > objectives[i, d]:
                        better_or_equal = False
                        break
                    if objectives[j, d] < objectives[i, d]:
                        strictly_better = True
            if better_or_equal and strictly_better:
                mask[i] = False
                break
    return mask


# A network only counts as a candidate solution if it is still doing EVERY task.
# Without this, networks at chance sit on the front: they are the cheapest in the
# grid, so nothing can dominate them on cost (35/37 points were "Pareto" in the
# core5 6x6 run, 8 of them at chance). 0.6 is above always-fixate on the go/no-go
# tasks (~0.5 on dmsgo) and far above the dead-network floor on the rest (~0.2-0.3).
FEASIBLE_MIN_TASK_ACC = 0.6


def feasible_mask(rows, min_task_acc=FEASIBLE_MIN_TASK_ACC):
    """True where the network clears ``min_task_acc`` on its worst task."""
    return np.array([r["min_task_acc"] >= min_task_acc for r in rows], dtype=bool)


def compute_front(
    rows,
    pareto_objectives,
    exclude_infeasible=True,
    min_task_acc=FEASIBLE_MIN_TASK_ACC,
):
    """Return (is_pareto, is_feasible) boolean arrays aligned with ``rows``.

    With ``exclude_infeasible`` the front is computed among feasible rows only,
    and infeasible rows are never on it. Otherwise every row competes, as before.
    """
    rows = list(rows)
    is_feasible = feasible_mask(rows, min_task_acc)
    pool = is_feasible if exclude_infeasible else np.ones(len(rows), dtype=bool)
    is_pareto = np.zeros(len(rows), dtype=bool)
    if pool.any():
        objectives = np.array(
            [[r[k] for k in pareto_objectives] for r, keep in zip(rows, pool) if keep]
        )
        is_pareto[pool] = pareto_mask(
            objectives, maximize=pareto_maximize_flags(pareto_objectives)
        )
    return is_pareto, is_feasible


def finalize_results(
    rows,
    pareto_objectives,
    exclude_infeasible=True,
    min_task_acc=FEASIBLE_MIN_TASK_ACC,
):
    is_pareto, is_feasible = compute_front(
        rows, pareto_objectives, exclude_infeasible, min_task_acc
    )
    for row, p, f in zip(rows, is_pareto, is_feasible):
        row["is_pareto"] = bool(p)
        row["is_feasible"] = bool(f)
    return rows
