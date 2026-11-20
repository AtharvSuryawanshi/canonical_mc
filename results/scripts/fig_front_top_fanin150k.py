"""Top view of the cog20 150k fanin front (W_in and W_out trained, nothing frozen).

Pools every such network: the 29-point budget grid (3 seeds, runs/moo/dale_all_grid29_fanin_150k_*)
and the 4-point zoom runs (control / A3 / B0 / C0; 5 seeds + 1 save-at seed). Each network is
plotted at the costs it reached; the front is over feasible networks (worst-task accuracy >= 0.6). Background:
filled contours of worst-task accuracy, a thin-plate-spline fit of the per-budget seed means (budget_means,
accuracy_surface), only inside their convex hull.

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


def budget_means(d):
    """One point per budget pair (grid point or zoom point): mean reached costs and mean worst-task accuracy over its
    seeds. The contours interpolate between these, not between single networks."""
    key = [d.rate_budget.round(6).fillna(-1), d.conn_budget.round(6).fillna(-1)]
    return d.groupby(key)[["x", "y", "z"]].mean().reset_index(drop=True)


def accuracy_surface(m, n=300, smoothing=1e-3):
    """Thin-plate-spline fit of the per-budget means (scipy RBFInterpolator, small smoothing so it stays close to
    them), on a grid, blanked outside their convex hull (no extrapolation)."""
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import Delaunay
    P = m[["x", "y"]].to_numpy()
    xx, yy = np.meshgrid(np.linspace(P[:, 0].min(), P[:, 0].max(), n), np.linspace(P[:, 1].min(), P[:, 1].max(), n))
    G = np.c_[xx.ravel(), yy.ravel()]
    zz = RBFInterpolator(P, m.z.to_numpy(), kernel="thin_plate_spline", smoothing=smoothing)(G)
    zz[Delaunay(P).find_simplex(G) < 0] = np.nan
    return xx, yy, np.clip(zz, 0, 1).reshape(xx.shape)


def main():
    d = load()
    m = budget_means(d)
    xx, yy, zz = accuracy_surface(m)
    fig, ax = plt.subplots(figsize=(7, 5.4))
    levels = np.arange(0.0, 1.0001, 0.05)
    cf = ax.contourf(xx, yy, zz, levels=levels, cmap="RdYlGn", vmin=0.0, vmax=1.0, zorder=0)
    cl = ax.contour(xx, yy, zz, levels=[0.7, 0.8, 0.9], colors="0.15", linewidths=0.7, zorder=1)
    for t in ax.clabel(cl, fmt="%.1f", fontsize=7.5, inline_spacing=3):
        t.set_bbox(dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.contour(xx, yy, zz, levels=[FEASIBLE_MIN_TASK_ACC], colors="k", linewidths=1.4, linestyles="--", zorder=1)
    inf, feas, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    ax.scatter(inf.x, inf.y, marker="x", s=22, color="0.35", linewidths=1, zorder=2)
    ax.scatter(feas.x, feas.y, s=10, color="0.2", edgecolors="none", zorder=2)
    ax.scatter(par.x, par.y, s=34, color="white", edgecolors="k", linewidths=0.9, zorder=3)
    z = d[d.point.isin(["control", "A3", "B0", "C0"])] if "point" in d else d.iloc[:0]
    for p, g in z.groupby("point"):
        ax.annotate({"control": "Control", "A3": "Firing constraint (A3)", "B0": "Wiring constraint (B0)", "C0": "Both constraints (C0)"}[p],
                    (g.x.mean(), g.y.mean()), xytext=(0, 11), textcoords="offset points", ha="center",
                    fontsize=8, color="#1f3b73", weight="bold")
    ex = d[(d.run == EXAMPLE[0]) & (d.seed == EXAMPLE[1])]
    ax.scatter(ex.x, ex.y, s=230, facecolors="none", edgecolors="#1f3b73", linewidths=1.8, zorder=4)
    fig.colorbar(cf, ax=ax, ticks=np.arange(0.0, 1.01, 0.1)).set_label(
        "Worst-task accuracy (mean over seeds, interpolated)")
    ax.scatter([], [], marker="x", color="0.35", label=f"infeasible network (< {FEASIBLE_MIN_TASK_ACC})")
    ax.scatter([], [], s=10, color="0.2", label="feasible network")
    ax.scatter([], [], s=34, color="white", edgecolors="k", label="Pareto front")
    ax.scatter([], [], s=120, facecolors="none", edgecolors="#1f3b73", linewidths=1.8,
               label=f"example network (seed {EXAMPLE[1]})")
    ax.plot([], [], "k--", lw=1.4, label=f"worst task = {FEASIBLE_MIN_TASK_ACC}")
    ax.legend(fontsize=7, frameon=True, framealpha=0.9, edgecolor="none", loc="lower right")
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
