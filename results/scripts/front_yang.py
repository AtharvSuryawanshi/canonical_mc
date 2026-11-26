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

import matplotlib.patheffects as pe
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


SHORT = {"control": "Control", "firing": "Rate", "wiring": "Synaptic", "both": "Both"}


def top():
    """Top view of the front: smoothed worst-task accuracy (filled bands), every network as a light mark, the front
    ringed, and the 4 points as stars. Few ticks; the legend sits above the map."""
    d, z = load_front(), load_points()
    xx, yy, zz = surface(d)
    fig, ax = plt.subplots(figsize=(3.5, 3.35))
    cf = ax.contourf(xx, yy, zz, levels=np.arange(0.6, 1.0001, 0.05), cmap="RdYlGn", vmin=0, vmax=1, zorder=0)
    cl = ax.contour(xx, yy, zz, levels=[0.9, 0.95], colors="0.2", linewidths=0.6, zorder=1)
    for t in ax.clabel(cl, fmt="%.2f", fontsize=7, inline_spacing=3, manual=[(-2.2, -1.62), (-2.35, -1.95)]):
        t.set_bbox(dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    inf, rest, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    ax.scatter(inf.x, inf.y, marker="x", s=9, color="0.45", linewidths=0.5, alpha=0.6, zorder=2)
    ax.scatter(rest.x, rest.y, s=3, color="0.15", edgecolors="none", alpha=0.28, zorder=2)
    ax.scatter(par.x, par.y, s=10, facecolors="none", edgecolors="0.1", linewidths=0.5, alpha=0.75, zorder=3)
    for p in POINTS:
        g = z[z.point == p]
        ax.scatter(g.x.mean(), g.y.mean(), s=210, marker="*", color=COLORS[p], edgecolors="white", linewidths=1.6,
                   zorder=5)
        ax.scatter(g.x.mean(), g.y.mean(), s=210, marker="*", facecolors="none", edgecolors="0.1", linewidths=0.9,
                   zorder=6)
        dx, ha = (0.1, "left") if g.x.mean() < -0.8 else (-0.1, "right")
        ax.text(g.x.mean() + dx, g.y.mean() - 0.065, SHORT[p], ha=ha, va="center", fontsize=7.5, color=COLORS[p],
                zorder=7, path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xlim(d.x.min() - 0.2, d.x.max() + 0.2)
    ax.set_ylim(d.y.min() - 0.1, d.y.max() + 0.12)
    ax.set_xticks([-2.5, -1.5, -0.5])
    ax.set_yticks([-2.5, -2.0, -1.5])
    ax.set_xlabel(X_LAB, fontsize=9)
    ax.set_ylabel(Y_LAB, fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values():
        s.set_visible(False)
    h = [plt.Line2D([], [], marker="x", ls="", color="0.45", ms=4, mew=0.7),
         plt.Line2D([], [], marker="o", ls="", color="0.15", alpha=0.4, ms=2.5),
         plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.1", ms=4, mew=0.6)]
    ax.legend(h, ["fails a task", "network", "on the front"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3,
              fontsize=7.5, frameon=False, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.2)
    cb = fig.colorbar(cf, ax=ax, ticks=[0.6, 0.7, 0.8, 0.9, 1.0], fraction=0.045, pad=0.03, aspect=25)
    cb.set_label("worst-task accuracy", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_visible(False)
    save(fig, "front_top", "front")


TASK = {"fdgo": "Go", "reactgo": "RT Go", "delaygo": "Dly Go", "fdanti": "Anti", "reactanti": "RT Anti",
        "delayanti": "Dly Anti", "dm1": "DM 1", "dm2": "DM 2", "contextdm1": "Ctx DM 1", "contextdm2": "Ctx DM 2",
        "multidm": "MultSen DM", "delaydm1": "Dly DM 1", "delaydm2": "Dly DM 2", "contextdelaydm1": "Ctx Dly DM 1",
        "contextdelaydm2": "Ctx Dly DM 2", "multidelaydm": "MultSen Dly DM", "dmsgo": "DMS", "dmsnogo": "DNMS",
        "dmcgo": "DMC", "dmcnogo": "DNMC"}


XLIM = {"metabolic_cost": (0.001, 1.0), "wiring_cost": (0.002, 0.35)}


def summary():
    """One group of 5 lines per point, one line per network (seed). Accuracy: its 20 tasks (grey) and its worst task
    (coloured), so worst-task accuracy is visibly the lowest of the 20; the vertical tick and number are the mean over
    seeds, the name the most common worst task. Rate and synaptic cost of each network on the same lines (log axes),
    bar = mean (log axes, bars start at the left axis); % of synapses kept."""
    from collections import Counter
    z = load_points()
    acc = [c for c in z.columns if c.startswith("acc_")]
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.9), sharey=True,
                             gridspec_kw=dict(width_ratios=[2.2, 1, 1], wspace=0.2))
    ys = np.arange(len(POINTS))[::-1].astype(float)
    sub = np.array([2, 1, 0, -1, -2]) * 0.14
    lo = np.floor(z[acc].to_numpy().min() * 50) / 50 - 0.01
    for y, p in zip(ys, POINTS):
        g = z[z.point == p].sort_values("seed")
        ax = axes[0]
        for yy, (_, r) in zip(y + sub, g.iterrows()):
            a = r[acc].to_numpy(float)
            ax.plot([lo, 1.0], [yy, yy], color="0.93", lw=0.6, zorder=0)
            ax.scatter(a, np.full(len(a), yy), s=5, color="0.6", edgecolors="none", zorder=2)
            ax.scatter(a.min(), yy, s=20, color=COLORS[p], edgecolors="white", linewidths=0.5, zorder=3)
        w = g.min_task_acc.mean()
        ax.plot([w, w], [y + sub[-1] - 0.1, y + sub[0] + 0.1], color=COLORS[p], lw=1.4, zorder=1)
        name, n = Counter(g[acc].idxmin(axis=1).str[4:]).most_common(1)[0]
        ax.text(lo - 0.064, y + 0.07, f"{w:.2f}", ha="left", va="bottom", fontsize=8.5, color=COLORS[p])
        ax.text(lo - 0.064, y + 0.02, f"{TASK[name]} ({n}/5)", ha="left", va="top", fontsize=6.5, color="0.4")
        for ax, col in ((axes[1], "metabolic_cost"), (axes[2], "wiring_cost")):
            lim = XLIM[col][0]
            ax.barh(y, g[col].mean() - lim, left=lim, height=0.78, color=COLORS[p], alpha=0.75, zorder=1)
            ax.scatter(g[col], y + sub, s=7, color="0.1", edgecolors="none", zorder=2)
        k = 100 * g.conn_frac.mean()
        axes[2].text(g.wiring_cost.max() * 1.5, y, f"{k:.0f}% kept" if k > 10 else f"{k:.1f}% kept",
                     va="center", fontsize=7.5, color="0.3")
    ax = axes[0]
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[p] for p in POINTS], fontsize=9)
    for t, p in zip(ax.get_yticklabels(), POINTS):
        t.set_color(COLORS[p])
    ax.set_ylim(-0.55, len(POINTS) - 0.45)
    ax.set_xlim(lo - 0.067, 1.008)
    ax.spines["bottom"].set_bounds(lo, 1.0)
    ax.set_xticks([t for t in (0.8, 0.85, 0.9, 0.95, 1.0) if t >= lo])
    ax.set_xlabel("accuracy", fontsize=9)
    h = [plt.Line2D([], [], marker="o", ls="", color="0.6", ms=2.5),
         plt.Line2D([], [], marker="o", ls="", color="0.3", ms=4.5),
         plt.Line2D([], [], marker="|", ls="", color="0.3", ms=8, mew=1.4)]
    ax.legend(h, ["20 tasks", "worst task", "mean of 5 networks"], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, fontsize=7.5, frameon=False, handletextpad=0.2, columnspacing=0.9, borderaxespad=0.2)
    for ax, lab in ((axes[1], "rate cost"), (axes[2], "synaptic cost")):
        ax.set_xscale("log")
        ax.set_xlabel(lab, fontsize=9)
    axes[1].set_xlim(*XLIM["metabolic_cost"])
    axes[2].set_xlim(*XLIM["wiring_cost"])
    for ax in axes:
        ax.tick_params(labelsize=8, length=2.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    for ax in axes[1:]:
        ax.spines["left"].set_visible(True)                         # bars start here
    for t in axes[0].get_xticks():
        axes[0].axvline(t, color="0.92", lw=0.6, zorder=0)
    save(fig, "points_summary", "front")
    print(z.groupby("point")[["z", "metabolic_cost", "wiring_cost", "conn_frac"]].mean().loc[POINTS].round(4))


if __name__ == "__main__":
    which = sys.argv[1:] or ["frames", "top", "summary"]
    for w in which:
        {"frames": frames, "top": top, "summary": summary}[w]()
