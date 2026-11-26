"""Talk figures of the Yang-battery front (slides 11-12): the 4 NSGA-III searches pooled, with the 4 points of the
front run (control, rate, synaptic, both constraints; 5 seeds each) on it.

Front: runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_{ms1,ms2,ext_ms3,ext_ms4} (573 networks; feasible
= worst-task accuracy >= 0.6; Pareto over feasible on (accuracy max, log rate cost min, log synaptic cost min)).
Points: fig_tv_ei_fanin.RUNS["front"].

    python results/scripts/front_yang.py [frames top summary]      (default: all)

Writes presentation/figs/talk/frames_yang/front_000..059.png (rotating 3D front, slide 11), front_top_front.pdf
(top view with the points) and points_summary_front.pdf (accuracy and costs per point, slide 12).
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmc.front import FEASIBLE_MIN_TASK_ACC, pareto_mask
from fig_tv_ei_fanin import COLORS, RUNS
from talk_style import LABEL, OUT, REPO, save

SEARCHES = [f"runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_{t}"
            for t in ("ms1", "ms2", "ext_ms3", "ext_ms4")]
POINTS = ["control", "firing", "wiring", "both"]
ACC = plt.get_cmap("RdYlGn")                       # accuracy colour, always 0 (red) - 1 (green)
X_LAB, Y_LAB = "rate cost (log$_{10}$)", "synaptic cost (log$_{10}$)"


def load_front():
    d = pd.concat([pd.read_csv(REPO / r / "runs.csv").assign(run=r.rsplit("_", 1)[-1]) for r in SEARCHES],
                  ignore_index=True)
    d["x"], d["y"], d["z"] = np.log10(d.metabolic_cost), np.log10(d.wiring_cost), d.min_task_acc
    d["feasible"] = d.z >= FEASIBLE_MIN_TASK_ACC
    d["pareto"] = False
    d.loc[d.feasible, "pareto"] = pareto_mask(d.loc[d.feasible, ["z", "x", "y"]].to_numpy(),
                                              maximize=(True, False, False))
    return d


def load_points():
    z = pd.read_csv(RUNS["front"] / "runs.csv")
    z = z[z.point.isin(POINTS)].copy()
    z["x"], z["y"], z["z"] = np.log10(z.metabolic_cost), np.log10(z.wiring_cost), z.min_task_acc
    return z


def surface(d, n=300, smoothing=1.0):
    """Thin-plate fit of worst-task accuracy over the feasible networks, inside their convex hull (smoothing 1.0: single
    NSGA networks are noisy; the same fit the points were picked on)."""
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import Delaunay
    f = d[d.feasible]
    P = f[["x", "y"]].to_numpy()
    xx, yy = np.meshgrid(np.linspace(P[:, 0].min(), P[:, 0].max(), n), np.linspace(P[:, 1].min(), P[:, 1].max(), n))
    G = np.c_[xx.ravel(), yy.ravel()]
    zz = RBFInterpolator(P, f.z.to_numpy(), kernel="thin_plate_spline", smoothing=smoothing)(G)
    zz[Delaunay(P).find_simplex(G) < 0] = np.nan
    return xx, yy, np.clip(zz, 0, 1).reshape(xx.shape)


def frames(n_frames=60, elev=22):
    from scipy.spatial import Delaunay
    d, z = load_front(), load_points()
    inf, rest, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    fig = plt.figure(figsize=(4.6, 3.7))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.scatter(inf.x, inf.y, np.full(len(inf), 0.6), marker="x", s=12, c="0.55", depthshade=False)  # on the floor
    ax.scatter(rest.x, rest.y, rest.z, s=5, c="0.4", alpha=0.4, depthshade=False)
    tri = Delaunay(par[["x", "y"]].to_numpy())
    ax.plot_trisurf(par.x, par.y, par.z, triangles=tri.simplices, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.45,
                    linewidth=0.2, edgecolor="0.5")
    ax.scatter(par.x, par.y, par.z, s=16, c=par.z, cmap="RdYlGn", vmin=0, vmax=1, edgecolors="k", linewidths=0.4,
               depthshade=False)
    for p in POINTS:
        g = z[z.point == p]
        ax.scatter([g.x.mean()], [g.y.mean()], [g.z.mean()], s=110, marker="*", color=COLORS[p], edgecolors="k",
                   linewidths=0.6, zorder=10, depthshade=False)
    ax.set_xlabel(X_LAB, labelpad=3, fontsize=9)
    ax.set_ylabel(Y_LAB, labelpad=3, fontsize=9)
    ax.set_zlabel("worst-task accuracy", labelpad=1, fontsize=9)
    ax.set_zticks([0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8, pad=0)
    ax.set_zlim(0.6, 1)
    h = [plt.Line2D([], [], marker="*", ls="", color=COLORS[p], mec="k", ms=10) for p in POINTS]
    fig.legend(h, [LABEL[p] for p in POINTS], loc="upper left", fontsize=8, frameon=False, ncol=2)
    fig.subplots_adjust(left=0, right=0.94, bottom=0.02, top=0.92)
    out = OUT / "frames_yang"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        ax.view_init(elev, -60 + 360 * i / n_frames)
        fig.savefig(out / f"front_{i:03d}.png", dpi=220)
    plt.close(fig)
    print(f"-> {out} ({n_frames} frames); {len(d)} networks, {int(d.feasible.sum())} feasible, "
          f"{int(d.pareto.sum())} on the front")


def top():
    d, z = load_front(), load_points()
    xx, yy, zz = surface(d)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    cf = ax.contourf(xx, yy, zz, levels=np.arange(0.6, 1.0001, 0.025), cmap="RdYlGn", vmin=0, vmax=1, zorder=0)
    cl = ax.contour(xx, yy, zz, levels=[0.9, 0.95], colors="0.15", linewidths=0.6, zorder=1)
    for t in ax.clabel(cl, fmt="%.2f", fontsize=7, inline_spacing=2):
        t.set_bbox(dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))
    inf, rest, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    ax.scatter(inf.x, inf.y, marker="x", s=10, color="0.35", linewidths=0.7, zorder=2, label="fails a task")
    ax.scatter(rest.x, rest.y, s=4, color="0.2", edgecolors="none", zorder=2, label="network")
    ax.scatter(par.x, par.y, s=14, color="white", edgecolors="k", linewidths=0.6, zorder=3, label="on the front")
    for p in POINTS:
        g = z[z.point == p]
        ax.scatter(g.x, g.y, s=10, color=COLORS[p], edgecolors="none", alpha=0.7, zorder=4)
        ax.scatter(g.x.mean(), g.y.mean(), s=150, marker="*", color=COLORS[p], edgecolors="k", linewidths=0.7,
                   zorder=5)
    cb = fig.colorbar(cf, ax=ax, ticks=[0.6, 0.7, 0.8, 0.9, 1.0], fraction=0.05, pad=0.03)
    cb.set_label("worst-task accuracy", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.85, edgecolor="none", handletextpad=0.2,
              borderpad=0.3, labelspacing=0.25)
    ax.set_xlabel(X_LAB, fontsize=9)
    ax.set_ylabel(Y_LAB, fontsize=9)
    ax.tick_params(labelsize=8)
    save(fig, "front_top", "front")


def summary():
    """One row per point: worst-task accuracy of each seed (dots in the accuracy colour, bar = mean), rate cost and
    synaptic cost (bars = mean on a log axis, dots = seeds), fraction of synapses kept (text)."""
    z = load_points()
    fig, axes = plt.subplots(1, 3, figsize=(4.2, 2.6), sharey=True,
                             gridspec_kw=dict(width_ratios=[1.2, 1, 1], wspace=0.12))
    ys = np.arange(len(POINTS))[::-1]
    for y, p in zip(ys, POINTS):
        g = z[z.point == p]
        jit = np.linspace(-0.18, 0.18, len(g))
        ax = axes[0]
        ax.scatter(g.z, y + jit, s=26, c=ACC(g.z.to_numpy()), edgecolors="k", linewidths=0.4, zorder=3)
        ax.plot([g.z.mean()] * 2, [y - 0.3, y + 0.3], color="k", lw=1.6, zorder=2)
        ax.text(g.z.mean(), y + 0.33, f"{g.z.mean():.2f}", ha="center", va="bottom", fontsize=7.5)
        for ax, col in ((axes[1], "metabolic_cost"), (axes[2], "wiring_cost")):
            ax.barh(y, g[col].mean(), height=0.55, color=COLORS[p], alpha=0.75)
            ax.scatter(g[col], y + jit, s=8, color="k", zorder=3)
        axes[2].text(g.wiring_cost.mean() * 1.25, y, f"{100 * g.conn_frac.mean():.0f}% kept" if g.conn_frac.mean()
                     > 0.1 else f"{100 * g.conn_frac.mean():.1f}% kept", va="center", fontsize=7.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([LABEL[p] for p in POINTS], fontsize=9)
    for t, p in zip(axes[0].get_yticklabels(), POINTS):
        t.set_color(COLORS[p])
    axes[0].set_xlim(0.85, 1.0)
    axes[0].set_xticks([0.85, 0.9, 0.95, 1.0])
    axes[0].set_xlabel("worst-task accuracy", fontsize=9)
    for ax, lab in ((axes[1], "rate cost"), (axes[2], "synaptic cost")):
        ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=9)
    axes[2].set_xlim(right=0.4)
    for ax in axes:
        ax.tick_params(labelsize=8)
    save(fig, "points_summary", "front")
    print(z.groupby("point")[["z", "metabolic_cost", "wiring_cost", "conn_frac"]].mean().loc[POINTS].round(4))


if __name__ == "__main__":
    which = sys.argv[1:] or ["frames", "top", "summary"]
    for w in which:
        {"frames": frames, "top": top, "summary": summary}[w]()
