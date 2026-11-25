"""One front from the two cog20 150k NSGA-III searches on the Yang-fixed battery (fanin, nothing frozen).

Pools runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_ms{1,2} (same settings, NSGA seeds 1 and 2,
training seed 0, batched): 288 networks. The front is recomputed over all of them (feasible = worst-task accuracy
>= 0.6), so a network from one search can drop off the front because of one from the other. Background: worst-task
accuracy contours, thin-plate-spline fit (accuracy_surface from fig_front_top_fanin150k.py), inside the convex hull.

    python results/scripts/fig_front_yang_ms12.py          # ms1 + ms2 -> ..._ms12
    python results/scripts/fig_front_yang_ms12.py ext      # extended box (rate 0.001-0.5, wiring 0.002-0.06),
                                                           # ext_ms3 + ext_ms4 -> ..._ext_ms34

Writes to runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_ms12/: runs.csv (both searches, with a `run`
column, the pooled `is_pareto` and `is_pareto_all`), front.csv (the pooled front), front_all.csv (the front over
every network, no feasibility cut: its rows with min_task_acc >= t are exactly the front at threshold t, for any t),
front_top.png.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmc.front import FEASIBLE_MIN_TASK_ACC, pareto_mask

_spec = importlib.util.spec_from_file_location("top150k", Path(__file__).with_name("fig_front_top_fanin150k.py"))
top = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(top)

REPO = Path(__file__).resolve().parents[2]
BASE = "runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_"
# set name -> (pooled searches, output run, NSGA seeds for the title)
SETS = {"ms12": (["ms1", "ms2"], "ms12", "1+2"),
        "ext": (["ext_ms3", "ext_ms4"], "ext_ms34", "3+4, extended box")}
SET = sys.argv[1] if len(sys.argv) > 1 else "ms12"
RUNS = [BASE + t for t in SETS[SET][0]]
OUT = REPO / (BASE + SETS[SET][1])


def load():
    frames = []
    for run in RUNS:
        d = pd.read_csv(REPO / run / "runs.csv")
        d["run"] = Path(run).name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["x"] = np.log10(d.metabolic_cost)
    d["y"] = np.log10(d.wiring_cost)
    d["z"] = d.min_task_acc
    d["feasible"] = d.z >= FEASIBLE_MIN_TASK_ACC
    d["is_pareto"] = False
    f = d.feasible
    d.loc[f, "is_pareto"] = pareto_mask(d.loc[f, ["z", "x", "y"]].to_numpy(), maximize=(True, False, False))
    d["is_pareto_all"] = pareto_mask(d[["z", "x", "y"]].to_numpy(), maximize=(True, False, False))
    return d


def main():
    d = load()
    OUT.mkdir(parents=True, exist_ok=True)
    xx, yy, zz = top.accuracy_surface(d[["x", "y", "z"]])
    fig, ax = plt.subplots(figsize=(7, 5.4))
    cf = ax.contourf(xx, yy, zz, levels=np.arange(0.0, 1.0001, 0.05), cmap="RdYlGn", vmin=0.0, vmax=1.0, zorder=0)
    cl = ax.contour(xx, yy, zz, levels=[0.7, 0.8, 0.9], colors="0.15", linewidths=0.7, zorder=1)
    for t in ax.clabel(cl, fmt="%.1f", fontsize=7.5, inline_spacing=3):
        t.set_bbox(dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.contour(xx, yy, zz, levels=[FEASIBLE_MIN_TASK_ACC], colors="k", linewidths=1.4, linestyles="--", zorder=1)
    inf = d[~d.feasible]
    ax.scatter(inf.x, inf.y, marker="x", s=22, color="0.35", linewidths=1, zorder=2,
               label=f"infeasible (< {FEASIBLE_MIN_TASK_ACC})")
    for (run, g), m in zip(d[d.feasible].groupby("run"), ["o", "s"]):
        tag = run.rsplit("_", 1)[-1]
        rest, par = g[~g.is_pareto], g[g.is_pareto]
        ax.scatter(rest.x, rest.y, marker=m, s=10, color="0.2", edgecolors="none", zorder=2, label=f"feasible, {tag}")
        ax.scatter(par.x, par.y, marker=m, s=36, color="white", edgecolors="k", linewidths=0.9, zorder=3,
                   label=f"Pareto front, {tag} ({len(par)})")
    fig.colorbar(cf, ax=ax, ticks=np.arange(0.0, 1.01, 0.1)).set_label("Worst-task accuracy (interpolated)")
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
    ax.set_xlabel("log(Firing cost)")
    ax.set_ylabel("log(Connectivity cost)")
    ax.set_title(f"cog20, 150k, Yang tasks, NSGA seeds {SETS[SET][2]}: {len(d)} networks, "
                 f"{int(d.feasible.sum())} feasible, {int(d.is_pareto.sum())} on the front", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "front_top.png", dpi=200, bbox_inches="tight")
    d = d.drop(columns=["x", "y", "z", "feasible"])
    d.to_csv(OUT / "runs.csv", index=False)
    d[d.is_pareto].sort_values("min_task_acc", ascending=False).to_csv(OUT / "front.csv", index=False)
    d[d.is_pareto_all].sort_values("min_task_acc", ascending=False).to_csv(OUT / "front_all.csv", index=False)
    print(d.groupby("run").agg(n=("is_pareto", "size"), feasible=("is_feasible", "sum"), pareto=("is_pareto", "sum")))
    print(f"{len(d)} networks, {int(d.is_feasible.sum())} feasible, {int(d.is_pareto.sum())} Pareto, "
          f"{int(d.is_pareto_all.sum())} Pareto without the cut -> {OUT}")


if __name__ == "__main__":
    main()
