"""Rotating 3D GIF of the cog20 50k Pareto front.

Pools every cog20 (20-task) 50k-step batched network: the NSGA-III search and
all moo_zoom re-trainings (408 networks). Each network is its own point at the
costs it actually reached (not its budget); seeds are not averaged. The front
is computed over the feasible networks (min_task_acc >= 0.6) with
cmc.front.pareto_mask: maximise min_task_acc, minimise both costs.

    python results/scripts/fig_front_gif.py

Writes results/front/cog20_50k_front.gif, a still front view
(cog20_50k_front_pic.png), a 2D top view coloured by min task accuracy
(cog20_50k_front_top.png) and front_networks.csv (the Pareto-optimal networks).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import Delaunay

from cmc.front import FEASIBLE_MIN_TASK_ACC, pareto_mask

REPO = Path(__file__).resolve().parents[2]
RUNS = [
    "runs/moo/dale_all_nsga3_budget_p48g3_50k",
    "runs/moo_zoom/dale_all_4pt_12seed_50k",
    "runs/moo_zoom/dale_all_front27_2seed_50k",
    "runs/moo_zoom/dale_all_gap12_2seed_50k",
    "runs/moo_zoom/dale_all_lines5_4seed_50k",
    "runs/moo_zoom/dale_all_curves4_4seed_50k",
]
HIGHLIGHT_RUN = "runs/moo_zoom/dale_all_curves4_4seed_50k"   # best seed per point gets a label
HIGHLIGHT_NAMES = {"control": "Unconstrained", "rate_limited": "Rate-limited",
                   "wiring_limited": "Wiring-limited", "both": "Both"}
OUT_DIR = REPO / "results" / "front"
N_FRAMES, FPS, DPI = 120, 10, 110
ELEV = 22


def load():
    frames = []
    for run in RUNS:
        d = pd.read_csv(REPO / run / "runs.csv")
        d["run"] = run
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["x"] = np.log10(d.metabolic_cost)      # firing
    d["y"] = np.log10(d.wiring_cost)         # connectivity
    d["z"] = d.min_task_acc
    d["feasible"] = d.min_task_acc >= FEASIBLE_MIN_TASK_ACC
    d["pareto"] = False
    f = d.feasible
    d.loc[f, "pareto"] = pareto_mask(d.loc[f, ["z", "x", "y"]].to_numpy(), maximize=(True, False, False))
    return d


def draw(ax, d, hl):
    feas, inf, par = d[d.feasible & ~d.pareto], d[~d.feasible], d[d.pareto]
    ax.scatter(inf.x, inf.y, inf.z, marker="x", s=12, c="0.6", alpha=0.6, depthshade=False)
    ax.scatter(feas.x, feas.y, feas.z, s=9, c="0.45", alpha=0.35, depthshade=False)
    if len(par) >= 3:
        tri = Delaunay(par[["x", "y"]].to_numpy())
        ax.plot_trisurf(par.x, par.y, par.z, triangles=tri.simplices, cmap="RdYlGn", vmin=0.5, vmax=1,
                        alpha=0.35, linewidth=0.2, edgecolor="0.5")
    ax.scatter(par.x, par.y, par.z, s=30, c=par.z, cmap="RdYlGn", vmin=0.5, vmax=1,
               edgecolors="k", linewidths=0.5, depthshade=False)
    for _, r in hl.iterrows():
        ax.scatter([r.x], [r.y], [r.z], s=140, facecolors="none", edgecolors="#1f3b73", linewidths=2, zorder=10,
                   depthshade=False)
        ax.text(r.x, r.y, r.z + 0.05, HIGHLIGHT_NAMES.get(r.point, r.point), fontsize=8, color="#1f3b73", zorder=11,
                ha="center")
    xx, yy = np.meshgrid([d.x.min(), d.x.max()], [d.y.min(), d.y.max()])
    ax.plot_surface(xx, yy, np.full_like(xx, FEASIBLE_MIN_TASK_ACC), color="#ffd8a8", alpha=0.25,
                    linewidth=0)
    ax.set_xlabel("log(Firing cost)", labelpad=6)
    ax.set_ylabel("log(Connectivity cost)", labelpad=6)
    ax.set_zlabel("Min Task Accuracy", labelpad=6)
    ax.set_zlim(0.0, 1.0)


def draw_top(d, hl, out):
    """Top view: every network at its reached costs, coloured by min task accuracy."""
    fig, ax = plt.subplots(figsize=(6.4, 5))
    kw = dict(cmap="RdYlGn", vmin=0.0, vmax=1)
    inf, feas, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    ax.scatter(inf.x, inf.y, c=inf.z, marker="X", s=28, edgecolors="0.3", linewidths=0.4, **kw)
    ax.scatter(feas.x, feas.y, c=feas.z, s=22, edgecolors="none", alpha=0.85, **kw)
    sc = ax.scatter(par.x, par.y, c=par.z, s=48, edgecolors="k", linewidths=0.9, zorder=3, **kw)
    for _, r in hl.iterrows():
        ax.scatter([r.x], [r.y], s=260, facecolors="none", edgecolors="#1f3b73", linewidths=2, zorder=4)
        ax.annotate(HIGHLIGHT_NAMES.get(r.point, r.point), (r.x, r.y), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=8, color="#1f3b73", zorder=5)
    fig.colorbar(sc, ax=ax).set_label("Min Task Accuracy")
    ax.set_xlabel("log(Firing cost)")
    ax.set_ylabel("log(Connectivity cost)")
    ax.scatter([], [], marker="X", c="0.6", label=f"infeasible (< {FEASIBLE_MIN_TASK_ACC})")
    ax.scatter([], [], c="0.6", label="feasible")
    ax.scatter([], [], c="0.6", edgecolors="k", label="Pareto front")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    d = load()
    c = d[d.run == HIGHLIGHT_RUN]
    hl = c.loc[c.groupby("point").min_task_acc.idxmax()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d[d.pareto].sort_values("z", ascending=False).drop(columns=["x", "y", "z"]).to_csv(
        OUT_DIR / "front_networks.csv", index=False)

    draw_top(d, hl, OUT_DIR / "cog20_50k_front_top.png")
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    draw(ax, d, hl)
    # ax.set_title(f"cog20, 50k steps: {len(d)} networks, {int(d.feasible.sum())} feasible, "
                #  f"{int(d.pareto.sum())} on the front", fontsize=10)
    ax.view_init(ELEV, -60)
    fig.savefig(OUT_DIR / "cog20_50k_front_pic.png", dpi=200, bbox_inches="tight")

    def update(i):
        ax.view_init(ELEV, -60 + 360 * i / N_FRAMES)
        return []

    FuncAnimation(fig, update, frames=N_FRAMES).save(OUT_DIR / "cog20_50k_front.gif",
                                                     writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"{len(d)} networks, {int(d.feasible.sum())} feasible, {int(d.pareto.sum())} Pareto -> "
          f"{OUT_DIR / 'cog20_50k_front.gif'}")


if __name__ == "__main__":
    main()
