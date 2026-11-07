"""Task selectivity per unit: how many tasks a unit's activity varies with ("breadth").

Works on the ``task_variance`` stored in every checkpoint (tasks x units, computed by
``cmc.runner.activity_summary`` as in Yang et al. 2019), so nothing is re-simulated and
no model is built. Spec: notes/idea.md, section "Metric".

- Active unit: total task variance > 1% of the network's largest total (over all units).
- Each unit is divided by its own max over tasks, so a quiet unit is not "more selective".
- Breadth: number of tasks with normalized variance strictly above ``thr`` (1..n_tasks).
"""

import numpy as np
import pandas as pd
import torch

ACTIVE_FRAC = 1e-2


def active_units(tv, frac=ACTIVE_FRAC):
    """tv: (tasks, units). True where the unit's total variance > frac x the largest total."""
    total = np.asarray(tv, dtype=float).sum(0)
    return total > frac * total.max()


def normalized(tv):
    """Each unit (column) divided by its own max over tasks; all-zero units stay 0."""
    tv = np.asarray(tv, dtype=float)
    return tv / np.maximum(tv.max(0, keepdims=True), 1e-30)


def breadth(tv, thr=0.5):
    """Per unit: number of tasks with normalized variance > thr."""
    return (normalized(tv) > thr).sum(0)


def unit_frame(tv, is_e, thr=0.5):
    """One row per unit: unit index, pop (E/I), active, total task variance, breadth."""
    tv = np.asarray(tv, dtype=float)
    return pd.DataFrame({
        "unit": np.arange(tv.shape[1]),
        "pop": np.where(np.asarray(is_e, bool), "E", "I"),
        "active": active_units(tv),
        "total_tv": tv.sum(0),
        "breadth": breadth(tv, thr),
    })


def load_checkpoint_tv(ckpt):
    """(task_variance, is_e, info) from a saved checkpoint; E/I from the fixed Dale sign vector."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    is_e = ck["state_dict"]["sign_vector"].numpy() > 0
    m = ck["metrics"]
    info = dict(point=ck.get("point"), seed=int(ck["seed"]), min_task_acc=float(m["min_task_acc"]),
                is_feasible=bool(ck["is_feasible"]))
    return np.asarray(ck["task_variance"], dtype=float), is_e, info


def unit_table(ckpt, thr=0.5):
    """Per-unit table for one checkpoint, with its point, seed, min_task_acc and feasibility."""
    tv, is_e, info = load_checkpoint_tv(ckpt)
    df = unit_frame(tv, is_e, thr)
    for k, v in reversed(info.items()):
        df.insert(0, k, v)
    df.insert(len(info), "thr", thr)
    return df


def network_summary(units):
    """One row from a per-unit table: mean breadth over active units and active count, E and I."""
    row = {k: units[k].iloc[0] for k in ("point", "seed", "thr", "min_task_acc", "is_feasible")}
    for pop in ("E", "I"):
        act = units[(units["pop"] == pop) & units["active"]]
        row[f"{pop}_breadth"] = act["breadth"].mean()
        row[f"{pop}_active"] = len(act)
    return row
