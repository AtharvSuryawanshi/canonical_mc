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
LABEL = {"control": "control", "A3": "rate-\nonly", "B0": "wiring-\nonly", "C0": "both\nbudgets",
         "rate_limited": "rate-\nlimited", "wiring_limited": "wiring-\nlimited", "knee": "both\nbudgets"}
SET_NAME = {"150k": "main set", "50k": "second set"}


def label(point):
    """One-line condition name for legends and tables."""
    return LABEL[point].replace("-\n", "-").replace("\n", " ")


def _strip(ax, df, pts, feasible_min, ref=True, ticklabels=True, size=22):
    """One condition per column: a dot per network (hollow = infeasible), a bar at the mean, and
    optionally a dashed line at the lowest control network."""
    if ref:
        ax.axhline(df[df["point"] == "control"]["E_breadth"].min(), color=_MUTED, lw=0.8,
                   ls=(0, (3, 3)), zorder=1)
    for j, p in enumerate(pts):
        d = df[df["point"] == p].sort_values("seed")
        c = ROLE_COLOR[ROLE[p]]
        xs = j + (np.linspace(-0.2, 0.2, len(d)) if len(d) > 1 else np.zeros(1))
        feas = d["is_feasible"].astype(bool).to_numpy()
        ax.scatter(xs, d["E_breadth"], s=size, zorder=3, linewidths=1.0, edgecolors=c,
                   facecolors=[c if f else _BG for f in feas])
        ax.hlines(d["E_breadth"].mean(), j - 0.32, j + 0.32, color=c, lw=2.2, zorder=2)
    ax.set_xlim(-0.6, len(pts) - 0.4)
    ax.set_xticks(range(len(pts)), [LABEL[p] for p in pts] if ticklabels else [])
    ax.tick_params(axis="x", length=0)


_BG, _INK, _MUTED = "white", "#0b0b0b", "#52514e"


def _panel_label(fig, ax, letter, dx=-0.09):
    bb = ax.get_position()
    fig.text(bb.x0 + dx, bb.y1 + 0.012, letter, fontsize=12, fontweight="bold", color=_INK,
             ha="left", va="bottom")


def plot_breadth(nets, points, out=None, feasible_min=0.6):
    """nets: {(set, thr): networks.csv frame}; points: {set: [point order]}. Writes a PNG to out,
    or returns the figure if out is None (used to check the plotted values).
    Panels: a main set, thr 0.5; b main set, thr 0.3 and 0.7; c main set, breadth vs worst-task
    accuracy (thr 0.5); d second set, thr 0.5."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "legend.fontsize": 9, "axes.edgecolor": _MUTED,
                         "axes.labelcolor": _INK, "xtick.color": _INK, "ytick.color": _INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(6.7, 5.6), facecolor=_BG)   # 17 cm wide
    outer = fig.add_gridspec(2, 2, left=0.1, right=0.985, top=0.95, bottom=0.09,
                             hspace=0.42, wspace=0.36, height_ratios=[1, 1])
    ylab = "E breadth (tasks per active E unit)"
    main, sec = "150k", "50k"

    ax_a = fig.add_subplot(outer[0, 0])
    _strip(ax_a, nets[main, "0.5"], points[main], feasible_min)
    ax_a.set_ylabel(ylab)
    ax_a.set_title(f"{SET_NAME[main]}, threshold 0.5", fontsize=9, color=_INK, loc="left")

    inner = outer[0, 1].subgridspec(2, 1, hspace=0.32)
    ax_b = []
    for k, thr in enumerate(("0.3", "0.7")):
        ax = fig.add_subplot(inner[k])
        _strip(ax, nets[main, thr], points[main], feasible_min, ticklabels=(k == 1), size=14)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.set_title(f"{SET_NAME[main]}, threshold {thr}", fontsize=9, color=_INK, loc="left",
                     pad=2)
        ax_b.append(ax)
    ax_b[0].set_ylabel(ylab, y=-0.16)

    ax_c = fig.add_subplot(outer[1, 0])
    df = nets[main, "0.5"]
    for p in points[main]:
        d = df[df["point"] == p].sort_values("seed")
        c = ROLE_COLOR[ROLE[p]]
        feas = d["is_feasible"].astype(bool).to_numpy()
        ax_c.scatter(d["min_task_acc"], d["E_breadth"], s=22, zorder=3, linewidths=1.0,
                     edgecolors=c, facecolors=[c if f else _BG for f in feas])
    ax_c.set_xlabel("worst-task accuracy")
    ax_c.set_ylabel(ylab)
    ax_c.set_title(f"{SET_NAME[main]}, threshold 0.5", fontsize=9, color=_INK, loc="left")
    ax_c.legend(handles=[Line2D([], [], ls="", marker="o", ms=5, color=ROLE_COLOR[ROLE[p]],
                                label=label(p)) for p in points[main]],
                frameon=False, loc="upper left", bbox_to_anchor=(0.17, 1.0), handletextpad=0.1,
                borderaxespad=0.1,
                labelspacing=0.3)

    ax_d = fig.add_subplot(outer[1, 1])
    _strip(ax_d, nets[sec, "0.5"], points[sec], feasible_min)
    ax_d.set_ylabel(ylab)
    ax_d.set_title(f"{SET_NAME[sec]}, threshold 0.5", fontsize=9, color=_INK, loc="left")

    for ax in (ax_a, *ax_b, ax_c, ax_d):
        ax.set_facecolor(_BG)
    for ax, letter in ((ax_a, "a"), (ax_b[0], "b"), (ax_c, "c"), (ax_d, "d")):
        _panel_label(fig, ax, letter)

    if out is None:
        return fig
    fig.savefig(out, dpi=300, facecolor=_BG, metadata={"Software": None})
    plt.close(fig)


# ---- Report tables (markdown, included by reports/study.qmd) --------------------------------------

def _rng(s, fmt="{:.2f}"):
    lo, hi = s.min(), s.max()
    return fmt.format(lo) if fmt.format(lo) == fmt.format(hi) else f"{fmt.format(lo)}–{fmt.format(hi)}"


def condition_table(nets, runs, names):
    """Per condition, threshold 0.5. nets / runs: {set: frame}; names: {set: display name}.
    Metabolic cost (eval, noise-free) and the training-time rate cost come from the run's runs.csv."""
    rows = []
    for s, df in nets.items():
        df = df.merge(runs[s][["point", "seed", "metabolic_cost", "train_rate_cost_tail"]],
                      on=["point", "seed"], validate="1:1")
        for p, g in df.groupby("point", sort=False):
            rows.append({
                "set": names[s], "condition": label(p),
                "networks (feasible)": f"{len(g)} ({int(g['is_feasible'].sum())})",
                "E breadth, mean (range)": f"{g['E_breadth'].mean():.2f} ({_rng(g['E_breadth'])})",
                "active E (of 204)": f"{g['E_active'].mean():.0f}",
                "worst-task accuracy": _rng(g["min_task_acc"]),
                "metabolic cost": _rng(g["metabolic_cost"], "{:#.2g}"),
            })
    return pd.DataFrame(rows)


def verdict_table(verdicts, names):
    """wrong_if.csv as a report table, in the report's names; names: {set: display name}."""
    v = verdicts.copy()
    v["set"] = v["set"].map(names)
    v["rule"] = v["rule"].astype(str).replace({"5 (all)": "5, all 12 (supplementary)"})
    v["verdict"] = [f"{x} (weak)" if r == "5" else x for r, x in zip(v["rule"], v["verdict"])]
    words = [("control acc min", "lowest control worst-task accuracy"), ("A3 seeds at/above it", "rate-only networks at or above it"),
             ("thr ", "threshold ")]
    words += [(code, label(code)) for code in ("A3", "B0", "rate_limited")]
    for a, b in words:
        v["numbers"] = v["numbers"].str.replace(a, b, regex=False)
    v["numbers"] = v["numbers"].str.replace(r"seed (\d+) acc ([\d.]+) breadth", r"seed \1: worst-task accuracy \2, E breadth",
                                            regex=True)
    # the registered Spearman for rule 3 lives in the note column; rule 5's notes are said in the text
    v["note"] = v["note"].fillna("").where(v["rule"] == "3", "")
    v["note"] = v["note"].str.replace(r"Spearman\(E breadth, min_task_acc\), all 20: rho ([-\d.]+), p ([\d.e-]+)",
                                      r"Spearman, E breadth vs worst-task accuracy, all 20 networks: ρ \1, p \2", regex=True)
    v = v.rename(columns={"name": "registered name", "thr": "threshold",
                          "numbers": "E breadth (unless stated)"})
    return v[["rule", "registered name", "set", "threshold", "verdict", "E breadth (unless stated)", "note"]]
