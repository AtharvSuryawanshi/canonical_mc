"""Animation of NSGA-III on a real search (slide 10): frames for Beamer's \\animategraphics.

Data: runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_ms1 (48 networks per generation).
Axes: rate cost, synaptic cost (log10) and worst-task accuracy. Story, with the step highlighted in a row of boxes:
  1 initialise   generation 0: 48 budgets appear on the floor (accuracy not known yet)
  2 train        each network rises to its worst-task accuracy and moves to the costs it reached (gradient descent)
  3 check        networks below 0.6 on some task are dropped (red x)
  4 rank         non-dominated networks (front 1, all three objectives) are ringed; the rest fade
  5 breed        generation 1's budgets appear on the floor, then train; ranking parents + children gives the new front

    python results/scripts/anim_nsga3.py [--dpi 150]

Writes presentation/figs/talk/frames_nsga/nsga_000..NNN.png (number printed at the end).
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from cmc.front import FEASIBLE_MIN_TASK_ACC
from talk_style import OUT, REPO

RUN = REPO / "runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_ms1"
STEPS = ["initialise\nbudgets", "train each\nnetwork", "check: every\ntask ≥ 0.6", "rank:\nPareto front",
         "breed new\nbudgets"]
ACC = plt.get_cmap("RdYlGn")
FLOOR = 0.4                                     # z of the floor (accuracy axis starts here)


def ease(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def front1(df):
    """Mask of front 1 (non-dominated) among feasible rows, on (1 - acc, log rate, log synaptic)."""
    m = np.zeros(len(df), bool)
    f = (df.acc >= FEASIBLE_MIN_TASK_ACC).to_numpy()
    F = np.c_[1 - df.acc, df.rx, df.ry][f]
    m[np.flatnonzero(f)[NonDominatedSorting().do(F, only_non_dominated_front=True)]] = True
    return m


def load():
    d = pd.read_csv(RUN / "runs.csv")
    d = d[d.generation <= 1].copy()
    d["bx"], d["by"] = np.log10(d.rate_budget), np.log10(d.conn_budget)        # before training: the budget
    d["rx"], d["ry"] = np.log10(d.metabolic_cost), np.log10(d.wiring_cost)     # after training: reached cost
    d["acc"] = d.min_task_acc
    return d.reset_index(drop=True)


def timeline():
    """List of (stage, t, gen) per frame. stage in init/train/check/rank/breed; gen = generation being shown."""
    T = []
    T += [("init", t, 0) for t in np.linspace(0, 1, 12)] + [("init", 1, 0)] * 4
    T += [("train", t, 0) for t in np.linspace(0, 1, 26)] + [("train", 1, 0)] * 4
    T += [("check", t, 0) for t in np.linspace(0, 1, 10)] + [("check", 1, 0)] * 6
    T += [("rank", t, 0) for t in np.linspace(0, 1, 10)] + [("rank", 1, 0)] * 8
    T += [("breed", t, 1) for t in np.linspace(0, 1, 12)] + [("breed", 1, 1)] * 4
    T += [("train", t, 1) for t in np.linspace(0, 1, 22)] + [("train", 1, 1)] * 4
    T += [("check", t, 1) for t in np.linspace(0, 1, 8)]
    T += [("rank", t, 1) for t in np.linspace(0, 1, 10)] + [("rank", 1, 1)] * 16
    return T


def draw_boxes(ax, active):
    ax.set_xlim(0, len(STEPS)); ax.set_ylim(0, 1); ax.axis("off")
    for i, s in enumerate(STEPS):
        on = i == active
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.12), 0.8, 0.76, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc="#1F8AD0" if on else "white", ec="#1F8AD0" if on else "0.6", lw=1.2))
        ax.text(i + 0.46, 0.5, s, ha="center", va="center", fontsize=8.5, color="white" if on else "0.35")
        if i < len(STEPS) - 1:
            ax.annotate("", (i + 1.05, 0.5), (i + 0.88, 0.5), arrowprops=dict(arrowstyle="-|>", color="0.5", lw=1))
    ax.annotate("", (0.46, 0.06), (4.46, 0.06), arrowprops=dict(arrowstyle="-|>", color="0.6", lw=0.8,
                                                                connectionstyle="arc3,rad=-0.12"))
    ax.text(2.46, -0.2, "repeat for each generation", ha="center", va="top", fontsize=7, color="0.5")


def main(dpi):
    d = load()
    g0, g1 = d[d.generation == 0].reset_index(drop=True), d[d.generation == 1].reset_index(drop=True)
    f0 = front1(g0)
    both = pd.concat([g0, g1], ignore_index=True)
    fb = front1(both)
    lim = dict(x=(min(d.bx.min(), d.rx.min()) - 0.1, max(d.bx.max(), d.rx.max()) + 0.1),
               y=(min(d.by.min(), d.ry.min()) - 0.05, max(d.by.max(), d.ry.max()) + 0.05))
    out = OUT / "frames_nsga"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("nsga_*.png"):
        old.unlink()
    T = timeline()
    stage_idx = {"init": 0, "train": 1, "check": 2, "rank": 3, "breed": 4}
    for n, (stage, t, gen) in enumerate(T):
        fig = plt.figure(figsize=(6.4, 4.2))
        axb = fig.add_axes([0.02, 0.86, 0.96, 0.13])
        draw_boxes(axb, stage_idx[stage])
        ax = fig.add_axes([0.06, 0.07, 0.88, 0.76], projection="3d", computed_zorder=False)
        ax.set_box_aspect((1.25, 1, 0.8), zoom=1.12)
        ax.view_init(22, -62 + 25 * n / len(T))

        def plot_gen(g, progress, fade=1.0, ring=None, drop_bad=False):
            e = ease(progress)
            x = g.bx + (g.rx - g.bx) * e
            y = g.by + (g.ry - g.by) * e
            z = FLOOR + (np.maximum(g.acc, FLOOR) - FLOOR) * e
            bad = (g.acc < FEASIBLE_MIN_TASK_ACC).to_numpy()
            if progress == 0:
                ax.scatter(x, y, z, s=14, color="0.45", depthshade=False, alpha=fade)
                return
            c = ACC(g.acc.to_numpy()) * np.array([1, 1, 1, 0]) + np.array([0, 0, 0, 1]) * fade
            c = c.copy()
            c[:, :3] = c[:, :3] * e + 0.45 * (1 - e)              # grey -> accuracy colour while training
            keep = ~bad if drop_bad else np.ones(len(g), bool)
            ax.scatter(x[keep], y[keep], z[keep], s=16, c=c[keep], edgecolors=(0, 0, 0, 0.6 * fade), linewidths=0.3,
                       depthshade=False)
            if drop_bad:
                ax.scatter(x[bad], y[bad], np.full(bad.sum(), FLOOR), marker="x", s=22, color="#E0393E",
                           depthshade=False)
            if ring is not None:
                ax.scatter(x[ring], y[ring], z[ring], s=60, facecolors="none", edgecolors="k", linewidths=1.0,
                           depthshade=False, zorder=10)

        if gen == 0:
            if stage == "init":
                k = int(round(ease(t) * len(g0)))
                plot_gen(g0.iloc[:k], 0)
            elif stage == "train":
                plot_gen(g0, t)
            elif stage == "check":
                plot_gen(g0, 1, drop_bad=t > 0.3)
            else:  # rank
                plot_gen(g0[~f0], 1, fade=1 - 0.65 * ease(t), drop_bad=True)
                plot_gen(g0[f0], 1, ring=np.ones(f0.sum(), bool) if t > 0.2 else None, drop_bad=True)
        else:
            if stage == "breed":
                plot_gen(g0[f0], 1, ring=np.ones(f0.sum(), bool), drop_bad=True)
                plot_gen(g0[~f0], 1, fade=0.35, drop_bad=True)
                k = int(round(ease(t) * len(g1)))
                ax.scatter(g1.bx[:k], g1.by[:k], np.full(k, FLOOR), s=14, color="#1F8AD0", depthshade=False)
            elif stage in ("train", "check"):
                plot_gen(g0[f0], 1, ring=np.ones(f0.sum(), bool), drop_bad=True)
                plot_gen(g0[~f0], 1, fade=0.35, drop_bad=True)
                plot_gen(g1, t if stage == "train" else 1, drop_bad=stage == "check" and t > 0.3)
            else:  # rank parents + children together
                fb0, fb1 = fb[:len(g0)], fb[len(g0):]
                fade = 1 - 0.65 * ease(t)
                plot_gen(g0[~fb0], 1, fade=0.35, drop_bad=True)
                plot_gen(g1[~fb1], 1, fade=fade, drop_bad=True)
                plot_gen(g0[fb0], 1, ring=np.ones(fb0.sum(), bool) if t > 0.2 else None, drop_bad=True)
                plot_gen(g1[fb1], 1, ring=np.ones(fb1.sum(), bool) if t > 0.2 else None, drop_bad=True)
        xx, yy = np.meshgrid(lim["x"], lim["y"])
        if stage in ("check", "rank") or (stage == "breed" and gen == 1):
            ax.plot_surface(xx, yy, np.full_like(xx, FEASIBLE_MIN_TASK_ACC), color="#E0393E", alpha=0.08,
                            linewidth=0)
        ax.set_xlim(*lim["x"]); ax.set_ylim(*lim["y"]); ax.set_zlim(FLOOR, 1)
        ax.set_xlabel("rate cost (log$_{10}$)", fontsize=8.5, labelpad=2)
        ax.set_ylabel("synaptic cost (log$_{10}$)", fontsize=8.5, labelpad=2)
        ax.set_zlabel("worst-task accuracy", fontsize=8.5, labelpad=0)
        ax.set_zticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xticks([-2.5, -2, -1.5, -1]); ax.set_yticks([-2.6, -2.2, -1.8])
        ax.tick_params(labelsize=7, pad=0)
        label = {0: "generation 0: 48 networks", 1: "generation 1: 48 new networks"}[gen]
        if stage == "rank" and gen == 1:
            label = f"front of generations 0 + 1: {int(fb.sum())} networks"
        elif stage == "rank":
            label = f"front of generation 0: {int(f0.sum())} networks"
        fig.text(0.03, 0.8, label, fontsize=9, color="0.25")
        fig.savefig(out / f"nsga_{n:03d}.png", dpi=dpi)
        plt.close(fig)
    print(f"-> {out} ({len(T)} frames, nsga_000..nsga_{len(T) - 1:03d})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=150)
    main(ap.parse_args().dpi)
