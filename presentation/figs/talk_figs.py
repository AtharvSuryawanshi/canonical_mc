"""Talk versions of the figures on slides 12-19: drawn at the size they appear on the slide (16:9 Beamer, text
width ~5.6 in), large labels, little text. The detailed analysis figures in results/front/ are unchanged; anything
precise that does not fit here goes in presentation/presentation.md.

    python presentation/figs/talk_figs.py [frames front convergence silhouette tv] [--run yang]   (default: all)

Writes presentation/figs/talk/*.pdf (the original figures are not touched). Data and analysis come from the results/scripts modules:
  frames      fig_front_top_fanin150k.load, rotating 3D front for slide 11         -> frames/front_000..059.png
  front       fig_front_top_fanin150k (load, budget_means, accuracy_surface)    -> front_top.pdf
  convergence save-at run runs_by_step.csv (as fig_acc_vs_steps_grid.py)          -> convergence.pdf
  (slides 15-16, like-to-like wiring: results/scripts/like2like.py, cached)
  silhouette  fig_tv_ei_fanin.analyse, 5-seed run, seed 4                         -> silhouette.pdf
              (--run yang: the fixed-task 5-seed run                              -> silhouette_yang.pdf)
  tv          same                                                                -> tv_control_A3.pdf,
                                                                                     tv_B0_C0.pdf
Accuracy colour: RdYlGn on 0-1 (0 red, 1 green), always.
"""
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "results/scripts"))

from fig_tv_ei_fanin import COLORS, RUNS, analyse            # noqa: E402
from cmc.front import FEASIBLE_MIN_TASK_ACC                  # noqa: E402
from cmc.task import rule_name                               # noqa: E402
from talk_style import E_COL, I_COL, NAMES, OUT, UNCONN, W, save   # noqa: E402,F401

SEED = 4


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "results/scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- slide 11: the front in 3D (animation frames)

def frames(n_frames=60, elev=22):
    from scipy.spatial import Delaunay
    fan = _module("front_top", "fig_front_top_fanin150k.py")
    d = fan.load()
    inf, rest, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    ex = d[(d.run == fan.EXAMPLE[0]) & (d.seed == fan.EXAMPLE[1])]
    fig = plt.figure(figsize=(4.4, 3.6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.scatter(inf.x, inf.y, inf.z, marker="x", s=12, c="0.55", depthshade=False)
    ax.scatter(rest.x, rest.y, rest.z, s=6, c="0.4", alpha=0.5, depthshade=False)
    tri = Delaunay(par[["x", "y"]].to_numpy())
    ax.plot_trisurf(par.x, par.y, par.z, triangles=tri.simplices, cmap="RdYlGn", vmin=0, vmax=1, alpha=0.45,
                    linewidth=0.2, edgecolor="0.5")
    ax.scatter(par.x, par.y, par.z, s=18, c=par.z, cmap="RdYlGn", vmin=0, vmax=1, edgecolors="k",
               linewidths=0.4, depthshade=False)
    xx, yy = np.meshgrid([d.x.min(), d.x.max()], [d.y.min(), d.y.max()])
    ax.plot_surface(xx, yy, np.full_like(xx, FEASIBLE_MIN_TASK_ACC), color="0.6", alpha=0.15, linewidth=0)
    for _, r in ex.iterrows():
        ax.scatter([r.x], [r.y], [r.z], s=90, facecolors="none", edgecolors="#1f3b73", linewidths=1.5, zorder=10,
                   depthshade=False)
        ax.text(r.x, r.y, r.z + {"C0": -0.16, "A3": 0.1}.get(r.point, 0.07), NAMES[r.point], fontsize=7,
                color="#1f3b73", zorder=11,
                ha="center")
    ax.set_xlabel("firing cost (log$_{10}$)", labelpad=2)
    ax.set_ylabel("wiring cost (log$_{10}$)", labelpad=2)
    ax.set_zlabel("worst-task accuracy", labelpad=0)
    ax.set_xticks([-3, -2, -1, 0]); ax.set_yticks([-2.6, -2.0, -1.4]); ax.set_zticks([0, 0.5, 1])
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlim(d.x.min() - 0.1, d.x.max() + 0.1); ax.set_ylim(d.y.min() - 0.1, d.y.max() + 0.1); ax.set_zlim(0, 1)
    fig.subplots_adjust(left=0, right=0.94, bottom=0.02, top=1.0)
    out = OUT / "frames"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        ax.view_init(elev, -60 + 360 * i / n_frames)
        fig.savefig(out / f"front_{i:03d}.png", dpi=220)
    plt.close(fig)
    print(f"-> {out} ({n_frames} frames)")


# ---------------------------------------------------------------- slide 12: the front, top view

def front():
    fan = _module("front_top", "fig_front_top_fanin150k.py")
    d = fan.load()
    xx, yy, zz = fan.accuracy_surface(fan.budget_means(d))
    fig, ax = plt.subplots(figsize=(3.0, 2.75))
    cf = ax.contourf(xx, yy, zz, levels=np.arange(0, 1.0001, 0.05), cmap="RdYlGn", vmin=0, vmax=1, zorder=0)
    cl = ax.contour(xx, yy, zz, levels=[0.8, 0.9], colors="0.15", linewidths=0.5, zorder=1)
    for t in ax.clabel(cl, fmt="%.1f", fontsize=6.5, inline_spacing=2):
        t.set_bbox(dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.8))
    ax.contour(xx, yy, zz, levels=[FEASIBLE_MIN_TASK_ACC], colors="k", linewidths=1, linestyles="--", zorder=1)
    inf, rest, par = d[~d.feasible], d[d.feasible & ~d.pareto], d[d.pareto]
    ax.scatter(inf.x, inf.y, marker="x", s=12, color="0.35", linewidths=0.8, zorder=2, label="infeasible")
    ax.scatter(rest.x, rest.y, s=5, color="0.2", edgecolors="none", zorder=2, label="feasible")
    ax.scatter(par.x, par.y, s=16, color="white", edgecolors="k", linewidths=0.6, zorder=3, label="Pareto front")
    ex = d[(d.run == fan.EXAMPLE[0]) & (d.seed == fan.EXAMPLE[1])]
    ax.scatter(ex.x, ex.y, s=110, facecolors="none", edgecolors="#1f3b73", linewidths=1.3, zorder=4)
    offset = {"control": (0, 9, "center"), "A3": (8, 6, "left"), "B0": (-4, 9, "right"), "C0": (8, -12, "left")}
    for p, g in d[d.point.isin(list(NAMES))].groupby("point"):
        dx, dy, ha = offset[p]
        ax.annotate(NAMES[p], (g.x.mean(), g.y.mean()), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=7, color="#1f3b73",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    cb = fig.colorbar(cf, ax=ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1], fraction=0.05, pad=0.03)
    cb.set_label("worst-task accuracy"); cb.ax.tick_params(labelsize=6.5)
    ax.legend(loc="lower right", fontsize=6, frameon=True, framealpha=0.85, edgecolor="none", handletextpad=0.2,
              borderpad=0.3, labelspacing=0.25)
    ax.set_xlabel("firing cost (log$_{10}$)")
    ax.set_ylabel("wiring cost (log$_{10}$)")
    save(fig, "front_top")


# ---------------------------------------------------------------- slide 13: convergence

def convergence():
    single = _module("acc_vs_steps", "fig_acc_vs_steps.py")
    d = pd.read_csv(single.REPO / single.RUNS["fanin"][0] / "runs_by_step.csv")
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, p in zip(axes, NAMES):
        m = d[d.point == p].groupby("step").mean(numeric_only=True)
        k = m.index / 1000
        for t in single.NAMES:
            ax.plot(k, m[f"acc_{t}"], color="0.75", lw=0.6)
        ax.plot(k, m["mean_acc"], color="k", lw=1.6, label="mean over tasks")
        ax.plot(k, m["min_task_acc"], color="crimson", lw=1.6, label="worst task")
        ax.text(k[-1] + 4, m["min_task_acc"].iloc[-1], f"{m['min_task_acc'].iloc[-1]:.2f}", color="crimson",
                fontsize=7, ha="left", va="center")
        ax.set_xlim(-5, 185)
        ax.axhline(FEASIBLE_MIN_TASK_ACC, color="0.4", ls=":", lw=0.8)
        ax.set_title(NAMES[p], color=COLORS[p])
        ax.set_ylim(0, 1.02); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([0, 50, 100, 150])
        ax.set_xlabel("training steps (k)")
    axes[0].set_ylabel("accuracy")
    axes[0].plot([], [], color="0.75", lw=0.6, label="each task")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    save(fig, "convergence")


# ---------------------------------------------------------------- slides 17-19: task variance groups

RUN = "5seed"          # which run silhouette / tv use: "5seed" (v1 tasks) or "yang" (--run yang); yang outputs get _yang


def _analyse_all():
    return {p: analyse(RUNS[RUN] / p / f"seed_{SEED:02d}.pt") for p in NAMES}


def silhouette(A=None):
    A = A or _analyse_all()
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True, gridspec_kw=dict(wspace=0.1,
                                                                                    width_ratios=[1.6, 1]))
    for ax, pop in zip(axes, "EI"):
        for p, R in A.items():
            sc = R[pop]["sil_curve"]
            ks = sorted(sc)
            ax.plot(ks, [sc[k] for k in ks], color=COLORS[p], lw=1.1, label=NAMES[p])
            ax.plot(R[pop]["K"], sc[R[pop]["K"]], "o", color=COLORS[p], ms=4.5, mec="k", mew=0.5, zorder=3)
        ax.set_title(f"{pop} neurons", color=E_COL if pop == "E" else I_COL)
        ax.set_xticks(range(2, max(ks) + 1, 4 if pop == "E" else 2))
    axes[0].set_ylabel("silhouette")
    fig.supxlabel("number of groups k", fontsize=8, y=-0.1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.98), fontsize=6.5)
    save(fig, "silhouette" + ("_yang" if RUN == "yang" else ""))


def tv(A=None):
    A = A or _analyse_all()
    for pair in (("control", "A3"), ("B0", "C0")):
        fig = plt.figure(figsize=(W, 2.75))
        n_units = {p: (len(A[p]["E"]["units"]), len(A[p]["I"]["units"])) for p in pair}
        gs = fig.add_gridspec(1, 5, width_ratios=[n_units[pair[0]][0], n_units[pair[0]][1] * 1.4, 18,
                                                  n_units[pair[1]][0], n_units[pair[1]][1] * 1.4],
                              wspace=0.04, left=0.11, right=0.995, top=0.82, bottom=0.04)
        cols = {pair[0]: (0, 1), pair[1]: (3, 4)}
        for p in pair:
            R = A[p]
            for pop, gi in zip("EI", cols[p]):
                ax = fig.add_subplot(gs[gi])
                P = R[pop]
                o = np.argsort(P["lab"], kind="stable")
                im = ax.imshow(R["norm"][:, P["units"][o]], cmap="hot", aspect="auto", interpolation="nearest",
                               vmin=0, vmax=1)
                labs = P["lab"][o]
                for c in range(1, P["K"]):
                    ax.axvline(np.flatnonzero(labs == c)[0] - 0.5, color="white", lw=0.35)
                ax.set_xticks([])
                n_t = len(R["tasks"])
                # group numbers under the columns; one moves down a row if it would touch its left neighbour
                gap = 0.11 * len(labs) / (ax.get_position().width * fig.get_figwidth())   # ~0.11 in, in units
                last = []                                # x of the last number in each row
                for c in range(P["K"]):
                    x = np.flatnonzero(labs == c).mean()
                    row = next((i for i, xl in enumerate(last) if x - xl >= gap), len(last))
                    if row == len(last):
                        last.append(x)
                    last[row] = x
                    ax.text(x, n_t - 0.2 + 1.1 * row, str(c + 1), ha="center", va="top", fontsize=6.5)
                ax.set_yticks(range(len(R["tasks"])))
                show_y = p == pair[0] and pop == "E"
                ax.set_yticklabels([rule_name[t] for t in R["tasks"]] if show_y else [], fontsize=6)
                ax.tick_params(length=0, pad=1.5)
                ax.spines[:].set_visible(False)
                ax.set_title(f"{pop}: {P['K']} groups", fontsize=7.5, color=E_COL if pop == "E" else I_COL,
                             pad=2)
                if pop == "E":
                    ax.text(0.0, 1.13, NAMES[p], transform=ax.transAxes, fontsize=9, color=COLORS[p])
        cax = fig.add_axes([0.80, -0.16, 0.19, 0.025])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1])
        cb.ax.tick_params(labelsize=6, length=2, pad=1)
        cax.set_title("task variance / neuron max", fontsize=6, pad=1.5, loc="left")
        save(fig, f"tv_{pair[0]}_{pair[1]}")


# ---------------------------------------------------------------- slide 18: groups of the 4 front points, one seed

def tv_front(seed=0, points=("control", "firing", "wiring", "both")):
    """2 x 2 figure, one panel per point (E and I heatmaps side by side), front run, k as on slide 17 (I up to 20)."""
    from talk_style import LABEL
    A = {p: analyse(RUNS["front"] / p / f"seed_{seed:02d}.pt", k_max_I=20) for p in points}
    fig = plt.figure(figsize=(W, 4.4))
    outer = fig.add_gridspec(2, 2, wspace=0.08, hspace=0.32, left=0.13, right=0.995, top=0.93, bottom=0.07)
    for n, p in enumerate(points):
        R = A[p]
        nE, nI = len(R["E"]["units"]), len(R["I"]["units"])
        inner = outer[n // 2, n % 2].subgridspec(1, 2, width_ratios=[nE, nI * 1.6], wspace=0.04)
        for pop, gi in zip("EI", (0, 1)):
            ax = fig.add_subplot(inner[gi])
            P = R[pop]
            o = np.argsort(P["lab"], kind="stable")
            im = ax.imshow(R["norm"][:, P["units"][o]], cmap="hot", aspect="auto", interpolation="nearest",
                           vmin=0, vmax=1)
            labs = P["lab"][o]
            for c in range(1, P["K"]):
                ax.axvline(np.flatnonzero(labs == c)[0] - 0.5, color="white", lw=0.35)
            ax.set_xticks([])
            ax.set_yticks(range(len(R["tasks"])))
            ax.set_yticklabels([rule_name[t] for t in R["tasks"]] if (n % 2 == 0 and pop == "E") else [],
                               fontsize=5.5)
            ax.tick_params(length=0, pad=1.5)
            ax.spines[:].set_visible(False)
            ax.set_title(f"{pop}: {P['K']} groups", fontsize=7.5, color=E_COL if pop == "E" else I_COL, pad=2)
            if pop == "E":
                ax.text(0.0, 1.16, LABEL[p], transform=ax.transAxes, fontsize=9.5, color=COLORS[p])
    cax = fig.add_axes([0.75, 0.015, 0.2, 0.012])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
    cax.set_title("task variance / neuron max", fontsize=6.5, pad=1.5, loc="left")
    save(fig, f"tv_s{seed}", "front")
    print({p: (A[p]["E"]["K"], A[p]["I"]["K"]) for p in points})


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--run" in args:
        i = args.index("--run")
        RUN = args[i + 1]
        del args[i:i + 2]
    which = args or ["frames", "front", "convergence", "silhouette", "tv"]
    A = None
    if {"silhouette", "tv"} & set(which):
        A = _analyse_all()
    for w in which:
        {"frames": frames, "front": front, "convergence": convergence,
         "silhouette": lambda: silhouette(A), "tv": lambda: tv(A), "tv_front": tv_front}[w]()
