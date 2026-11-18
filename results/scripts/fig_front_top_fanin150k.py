"""Top view of the cog20 150k fanin front (W_in and W_out trained, nothing frozen).

Pools every such network: the 29-point budget grid (3 seeds, runs/moo/dale_all_grid29_fanin_150k_*)
and the 4-point zoom runs (control / A3 / B0 / C0; 5 seeds + 1 save-at seed). Each network is
plotted at the costs it reached; the front is over feasible networks (min_task_acc >= 0.6).

    python results/scripts/fig_front_top_fanin150k.py

Writes results/front/cog20_150k_fanin_front_top.png, front_networks_150k_fanin.csv and the rotating
3D view cog20_150k_fanin_front.gif (drawn with fig_front_gif_150k.draw).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import importlib.util

from matplotlib.animation import FuncAnimation, PillowWriter

from cmc.front import FEASIBLE_MIN_TASK_ACC, pareto_mask

_spec = importlib.util.spec_from_file_location("front150k", Path(__file__).with_name("fig_front_gif_150k.py"))
gif = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gif)

REPO = Path(__file__).resolve().parents[2]
RUN_GLOBS = ["runs/moo/dale_all_grid29_fanin_150k_*",
             "runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k",
             "runs/moo_zoom/dale_all_woutfree_fanin_4pt_1seed_150k_saveat"]
OUT = REPO / "results/front"
EXAMPLE = ("dale_all_woutfree_fanin_4pt_5seed_150k", 4)   # (run, seed): the example networks ringed on the top view


def load():
    frames = []
    for g in RUN_GLOBS:
        for run in sorted(REPO.glob(g)):
            d = pd.read_csv(run / "runs.csv")
            d["run"] = run.name
            frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["x"] = np.log10(d.metabolic_cost)
    d["y"] = np.log10(d.wiring_cost)
    d["z"] = d.min_task_acc
    d["feasible"] = d.z >= FEASIBLE_MIN_TASK_ACC
    d["pareto"] = False
    f = d.feasible
    d.loc[f, "pareto"] = pareto_mask(d.loc[f, ["z", "x", "y"]].to_numpy(), maximize=(True, False, False))
    return d


def main():
    d = load()
    fig, ax = plt.subplots(figsize=(7, 5.4))
    kw = dict(cmap="RdYlGn", vmin=0.0, vmax=1)
    inf, feas, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    ax.scatter(inf.x, inf.y, c=inf.z, marker="X", s=30, edgecolors="0.3", linewidths=0.4, **kw)
    ax.scatter(feas.x, feas.y, c=feas.z, s=24, edgecolors="none", alpha=0.85, **kw)
    sc = ax.scatter(par.x, par.y, c=par.z, s=50, edgecolors="k", linewidths=0.9, zorder=3, **kw)
    z = d[d.point.isin(["control", "A3", "B0", "C0"])] if "point" in d else d.iloc[:0]
    for p, g in z.groupby("point"):
        ax.annotate({"control": "Unconstrained", "A3": "A3 rate", "B0": "B0 wiring", "C0": "C0 both"}[p],
                    (g.x.mean(), g.y.mean()), xytext=(0, 10), textcoords="offset points", ha="center",
                    fontsize=8, color="#1f3b73")
    ex = d[(d.run == EXAMPLE[0]) & (d.seed == EXAMPLE[1])]
    ax.scatter(ex.x, ex.y, s=230, facecolors="none", edgecolors="#1f3b73", linewidths=1.8, zorder=4)
    fig.colorbar(sc, ax=ax).set_label("Min Task Accuracy")
    ax.scatter([], [], marker="X", c="0.6", label=f"infeasible (< {FEASIBLE_MIN_TASK_ACC})")
    ax.scatter([], [], c="0.6", label="feasible")
    ax.scatter([], [], c="0.6", edgecolors="k", label="Pareto front")
    ax.scatter([], [], s=120, facecolors="none", edgecolors="#1f3b73", linewidths=1.8,
               label=f"example network (seed {EXAMPLE[1]})")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_xlabel("log(Firing cost)")
    ax.set_ylabel("log(Connectivity cost)")
    ax.set_title(f"cog20, 150k steps, fanin, nothing frozen: {len(d)} networks, "
                 f"{int(d.feasible.sum())} feasible, {int(d.pareto.sum())} on the front", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "cog20_150k_fanin_front_top.png", dpi=200, bbox_inches="tight")
    d[d.pareto].sort_values("z", ascending=False).drop(columns=["x", "y", "z"]).to_csv(
        OUT / "front_networks_150k_fanin.csv", index=False)
    write_gif(d)
    print(d.groupby("run").size().to_string())
    print(f"{len(d)} networks, {int(d.feasible.sum())} feasible, {int(d.pareto.sum())} Pareto")


def write_gif(d):
    z = d[d.point.isin(list(gif.HIGHLIGHT_NAMES))]
    hl = z.loc[z.groupby("point").min_task_acc.idxmax()]   # best seed per labelled point
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    gif.draw(ax, d, hl)
    ax.set_xlim(d.x.min() - 0.1, d.x.max() + 0.1)   # autoscale leaves the cloud in a corner
    ax.set_ylim(d.y.min() - 0.1, d.y.max() + 0.1)

    def update(i):
        ax.view_init(gif.ELEV, -60 + 360 * i / gif.N_FRAMES)
        return []

    FuncAnimation(fig, update, frames=gif.N_FRAMES).save(OUT / "cog20_150k_fanin_front.gif",
                                                         writer=PillowWriter(fps=gif.FPS), dpi=gif.DPI)
    plt.close(fig)


if __name__ == "__main__":
    main()
