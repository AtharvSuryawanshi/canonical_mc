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


# ---- "Wrong if" verdicts (notes/idea.md, "## Plan" step 4) --------------------------------------
# verdict "fail" = the Wrong-if condition holds (the claim is wrong on that rule), "pass" = it does not.

def _b(nets, point, col="E_breadth"):
    return nets[nets["point"] == point][col]


def wrong_if(nets):
    """nets: {(set, thr): networks.csv frame}, thr as a string ("0.5"). One row per rule."""
    from scipy.stats import spearmanr

    n5 = nets["150k", "0.5"]
    rows = []

    def add(rule, name, set_, thr, fail, numbers, note=""):
        verdict = "untestable" if fail is None else ("fail" if fail else "pass")
        rows.append(dict(rule=rule, name=name, set=set_, thr=thr, verdict=verdict,
                         numbers=numbers, note=note))

    # 1. No rate effect: fail if any A3 seed >= the lowest control seed (thr 0.5).
    a3, ctl = _b(n5, "A3"), _b(n5, "control")
    add(1, "no rate effect", "150k", "0.5", bool(a3.max() >= ctl.min()),
        f"A3 max {a3.max():.3f} vs control min {ctl.min():.3f}")

    # 2. Threshold artefact: rule 1's separation breaks at 0.3 OR at 0.7.
    parts, fail2 = [], False
    for thr in ("0.3", "0.7"):
        a3t, ctlt = _b(nets["150k", thr], "A3"), _b(nets["150k", thr], "control")
        fail2 |= bool(a3t.max() >= ctlt.min())
        parts.append(f"thr {thr}: A3 max {a3t.max():.3f} vs control min {ctlt.min():.3f}")
    add(2, "threshold artefact", "150k", "0.3,0.7", fail2, "; ".join(parts))

    # 3. Accuracy, not budget: A3 seeds as accurate as the least accurate control seed must sit
    #    outside (below) the control breadth range.
    acc_floor = _b(n5, "control", "min_task_acc").min()
    top = n5[(n5["point"] == "A3") & (n5["min_task_acc"] >= acc_floor)]
    inside = (top["E_breadth"] >= ctl.min()) & (top["E_breadth"] <= ctl.max())
    rho, p = spearmanr(n5["E_breadth"], n5["min_task_acc"])
    add(3, "accuracy, not budget", "150k", "0.5", None if top.empty else bool(inside.any()),
        f"control acc min {acc_floor:.3f}; A3 seeds at/above it: "
        + (", ".join(f"seed {int(r.seed)} acc {r.min_task_acc:.3f} breadth {r.E_breadth:.3f}"
                     for r in top.itertuples()) or "none")
        + f"; control range [{ctl.min():.3f}, {ctl.max():.3f}]",
        f"Spearman(E breadth, min_task_acc), all 20: rho {rho:.3f}, p {p:.3g}")

    # 4. Population artefact: A3 mean active E count below half of control's.
    a3n, ctln = _b(n5, "A3", "E_active").mean(), _b(n5, "control", "E_active").mean()
    add(4, "population artefact", "150k", "0.5", bool(a3n < 0.5 * ctln),
        f"A3 mean active E {a3n:.1f} vs 0.5 x control {0.5 * ctln:.1f}")

    # 5. Does not replicate: 50k, rate_limited mean not below control mean. Decided on feasible
    #    networks; all-12 reported as confounded. Weak either way (3/12 rate_limited feasible).
    n50 = nets["50k", "0.5"]
    for label, df in (("feasible only", n50[n50["is_feasible"]]), ("all 12, confounded by infeasibility", n50)):
        rl, c = _b(df, "rate_limited"), _b(df, "control")
        fail = None if rl.empty or c.empty else bool(rl.mean() >= c.mean())
        add(5 if label == "feasible only" else "5 (all)", "does not replicate", "50k", "0.5", fail,
            f"{label}: rate_limited mean {rl.mean():.3f} (n={len(rl)}) vs control mean {c.mean():.3f} (n={len(c)})",
            f"decides rule 5; weak ({len(rl)}/{len(_b(n50, 'rate_limited'))} rate_limited feasible)"
            if label == "feasible only" else "reported only, not decisive")

    # 6. Wiring does specialize: every B0 seed below the lowest control seed.
    b0 = _b(n5, "B0")
    add(6, "wiring does specialize", "150k", "0.5", bool(b0.max() < ctl.min()),
        f"B0 max {b0.max():.3f} vs control min {ctl.min():.3f}")

    return pd.DataFrame(rows)


# ---- Figure ---------------------------------------------------------------------------------------
# One colour per budget role, same in both sets; control is a dark neutral (it is the baseline).
# Palette validated with the dataviz validator (all-pairs, light surface).
ROLE_COLOR = {"control": "#3a3936", "rate": "#eb6834", "wiring": "#2a78d6", "both": "#1baf7a"}
ROLE = {"control": "control", "A3": "rate", "B0": "wiring", "C0": "both",
        "rate_limited": "rate", "wiring_limited": "wiring", "knee": "both"}
LABEL = {"control": "control", "A3": "A3\nrate", "B0": "B0\nwiring", "C0": "C0\nboth",
         "rate_limited": "rate-\nlimited", "wiring_limited": "wiring-\nlimited", "knee": "knee\nboth"}


def plot_breadth(nets, points, out=None, feasible_min=0.6):
    """nets: {(set, thr): networks.csv frame}; points: {set: [point order]}. Writes a PNG to out,
    or returns the figure if out is None (used to check the plotted values)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "axes.spines.top": False,
                         "axes.spines.right": False})
    strips = [("150k", "0.3"), ("150k", "0.5"), ("150k", "0.7"), ("50k", "0.5")]
    fig = plt.figure(figsize=(11, 6.6), facecolor="#fcfcfb")
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.55, wspace=0.32)
    def dot(ax, x, y, color, filled):
        ax.scatter(x, y, s=26, zorder=3, linewidths=1.2, edgecolors=color,
                   facecolors=color if filled else "#fcfcfb")

    for i, (s, thr) in enumerate(strips):
        ax = fig.add_subplot(gs[0, i], facecolor="#fcfcfb")
        df = nets[s, thr]
        ctl_min = df[df["point"] == "control"]["E_breadth"].min()
        ax.axhline(ctl_min, color=muted, lw=0.8, ls=(0, (3, 3)), zorder=1)
        for j, p in enumerate(points[s]):
            d = df[df["point"] == p].sort_values("seed")
            c = ROLE_COLOR[ROLE[p]]
            xs = j + np.linspace(-0.18, 0.18, len(d))
            for x, r in zip(xs, d.itertuples()):
                dot(ax, x, r.E_breadth, c, r.is_feasible)
            ax.hlines(d["E_breadth"].mean(), j - 0.28, j + 0.28, color=c, lw=2, zorder=2)
        ax.set_xticks(range(len(points[s])), [LABEL[p] for p in points[s]])
        ax.set_xlim(-0.6, len(points[s]) - 0.4)
        ax.grid(axis="y", color=grid, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(f"{s} set, threshold {thr}", fontsize=9, color=ink, loc="left")
        if i == 0:
            ax.set_ylabel("E breadth (tasks per active E unit)")
        ax.text(len(points[s]) - 0.45, ctl_min, "lowest control seed", fontsize=6.5, color=muted,
                ha="right", va="bottom")

    for k, s in enumerate(("150k", "50k")):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2], facecolor="#fcfcfb")
        df = nets[s, "0.5"]
        for p in points[s]:
            d = df[df["point"] == p]
            c = ROLE_COLOR[ROLE[p]]
            for r in d.itertuples():
                dot(ax, r.min_task_acc, r.E_breadth, c, r.is_feasible)
            ax.scatter([], [], s=26, color=c, label=LABEL[p].replace("-\n", "-").replace("\n", " "))
        if s == "50k":
            ax.axvline(feasible_min, color=muted, lw=0.8, ls=(0, (3, 3)))
            ax.text(feasible_min, ax.get_ylim()[1], " feasibility 0.6", fontsize=6.5, color=muted, va="top")
        ax.grid(color=grid, lw=0.6); ax.set_axisbelow(True)
        ax.set_xlabel("min task accuracy (worst of 20 tasks)")
        if k == 0:
            ax.set_ylabel("E breadth, threshold 0.5")
        ax.set_title(f"{s} set: breadth vs accuracy, one dot per network", fontsize=9, color=ink, loc="left")
        ax.legend(frameon=False, fontsize=7, loc="best")

    fig.suptitle("Task breadth of excitatory units by cost budget. Dot = one network (150k: 5 seeds per "
                 "condition, 50k: 12); bar = condition mean; hollow = infeasible (min task acc < 0.6)",
                 fontsize=9, color=ink, x=0.06, ha="left")
    if out is None:
        return fig
    fig.savefig(out, dpi=200, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
