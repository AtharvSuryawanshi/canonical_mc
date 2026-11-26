"""NSGA-III, one generation, drawn from a real search (slide 10). Draft for discussion.

Data: runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_ms1 (48 networks per generation).
Panels, left to right:
  1 choose budgets       generation 0: 48 (rate budget, synaptic budget) pairs, spread over the box
  2 train each network   reached rate and synaptic cost, colour = worst-task accuracy (0 red, 1 green)
  3 rank                 non-dominated sorting on all three objectives (accuracy, rate, synaptic): front 1, 2, 3+;
                         networks failing a task (< 0.6) ranked last
  4 next generation      parents (front 1-2, kept spread along reference directions) and the 48 new budgets
                         (generation 1) they produce

    python results/scripts/fig_nsga3_scheme.py

Writes presentation/figs/talk/nsga3_scheme_front.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from cmc.front import FEASIBLE_MIN_TASK_ACC
from talk_style import REPO, save

RUN = REPO / "runs/moo/dale_all_nsga3_budget_p48g3_fanin_150k_yang_fullgpu_ms1"
RANK_COL = {0: "k", 1: "0.45", 2: "0.8"}


def main():
    d = pd.read_csv(RUN / "runs.csv")
    g0, g1 = d[d.generation == 0].copy(), d[d.generation == 1]
    g0["x"], g0["y"] = np.log10(g0.metabolic_cost), np.log10(g0.wiring_cost)
    feas = g0.min_task_acc >= FEASIBLE_MIN_TASK_ACC
    F = np.c_[1 - g0.min_task_acc, g0.x, g0.y]
    rank = np.full(len(g0), 3)
    fronts = NonDominatedSorting().do(F[feas.to_numpy()])
    idx = np.flatnonzero(feas.to_numpy())
    for r, f in enumerate(fronts):
        rank[idx[f]] = min(r, 2)
    g0["rank"] = rank

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6), gridspec_kw=dict(wspace=0.5))
    bx, by = np.log10(g0.rate_budget), np.log10(g0.conn_budget)
    ax = axes[0]
    ax.scatter(bx, by, s=14, color="0.3")
    ax.set_title("1  choose budgets", loc="left")
    ax.set_xlabel("rate budget (log$_{10}$)"); ax.set_ylabel("synaptic budget (log$_{10}$)")

    ax = axes[1]
    sc = ax.scatter(g0.x, g0.y, s=18, c=g0.min_task_acc, cmap="RdYlGn", vmin=0, vmax=1, edgecolors="k", lw=0.3)
    ax.set_title("→ 2  train each", loc="left")
    ax.set_xlabel("rate cost (log$_{10}$)"); ax.set_ylabel("synaptic cost (log$_{10}$)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03, ticks=[0, 0.5, 1])
    cb.set_label("worst-task acc.", fontsize=7)

    ax = axes[2]
    for r, lab in ((2, "front 3+"), (1, "front 2"), (0, "front 1")):
        m = g0["rank"] == r
        ax.scatter(g0.x[m], g0.y[m], s=26 if r == 0 else 14, color=RANK_COL[r], label=lab, zorder=3 - r)
    m = g0["rank"] == 3
    ax.scatter(g0.x[m], g0.y[m], marker="x", s=16, color="crimson", label="fails a task")
    ax.set_title("→ 3  rank (3 objectives)", loc="left")
    ax.set_xlabel("rate cost (log$_{10}$)")
    ax.legend(fontsize=7, frameon=False, loc="lower left", handletextpad=0.1, labelspacing=0.2)

    ax = axes[3]
    par = g0["rank"] <= 1
    ax.scatter(bx[par], by[par], s=26, facecolors="none", edgecolors="k", lw=0.9, label="parents (gen 0)")
    ax.scatter(np.log10(g1.rate_budget), np.log10(g1.conn_budget), s=12, color="#1F8AD0", label="children (gen 1)")
    ax.set_title("→ 4  next generation", loc="left")
    ax.set_xlabel("rate budget (log$_{10}$)")
    ax.legend(fontsize=7, frameon=False, loc="lower left", handletextpad=0.1, labelspacing=0.2)
    for a in axes:
        a.tick_params(labelsize=7)
    fig.text(0.5, -0.13, "repeat for every generation (here 3 x 48 networks per search)", ha="center", fontsize=8,
             color="0.3")
    save(fig, "nsga3_scheme", "front")
    print({f"front {r + 1}": int((g0['rank'] == r).sum()) for r in range(3)}, "fails:", int((~feas).sum()))


if __name__ == "__main__":
    main()
