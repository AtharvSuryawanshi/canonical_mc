"""Does a rate budget make neurons more task-selective? (slide 20). Front run, 5 points x 5 seeds (control, loose,
rate, synaptic, both constraints), one dot per network, x = rate cost (log):

  left    tasks per neuron: per active neuron, the number of tasks whose normalized task variance is > 0.5 (as
          fig_tv_ei_fanin.analyse "breadth"), mean over the population
  right   silhouette averaged over k = 3-20 (k-means on the normalized task-variance profiles; silhouette_seeds cache)

Rows: E and I neurons. Lines join the point means within a synaptic level: unconstrained synapses (control, loose, rate;
wiring cost ~0.03-0.05) and sparse synapses (synaptic, both; ~0.005), so the rate effect is read at two synaptic levels.
Check printed, not drawn: tasks per neuron with only strongly active neurons (total task variance > 10% of the max).

    python results/scripts/selectivity_rate.py      (computes loose silhouettes into the cache if missing)

Writes presentation/figs/talk/selectivity_front.pdf.
"""
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import COLORS, RUNS
from silhouette_seeds import compute
from talk_style import E_COL, I_COL, LABEL, save

RUN = "front"
POINTS = ["control", "loose", "firing", "wiring", "both"]
SEEDS = range(5)
LEVELS = {"unconstrained synapses": ["control", "loose", "firing"], "sparse synapses": ["wiring", "both"]}
NAME = {**LABEL, "loose": "Loose budgets"}


def breadth(point, seed, active_frac):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, ck = load_checkpoint(RUNS[RUN] / point / f"seed_{seed:02d}.pt")
    tv = np.asarray(ck["task_variance"], float)
    total = tv.sum(0)
    active = total > active_frac * total.max()
    norm = tv / np.maximum(tv.max(0, keepdims=True), 1e-30)
    n = np.arange(tv.shape[1])
    return {pop: float((norm[:, active & sel] > 0.5).sum(0).mean())
            for pop, sel in (("E", n < model.n_e), ("I", n >= model.n_e))}


def table():
    runs = pd.read_csv(RUNS[RUN] / "runs.csv").set_index(["point", "seed"])
    sil = compute(RUN, POINTS, list(SEEDS), "task", False)
    sil = sil[(sil.k >= 3) & sil.point.isin(POINTS)].groupby(["point", "seed", "pop"]).silhouette.mean()
    rows = []
    for p in POINTS:
        for s in SEEDS:
            b, b10 = breadth(p, s, 1e-2), breadth(p, s, 1e-1)
            for pop in "EI":
                rows.append(dict(point=p, seed=s, pop=pop, rate=runs.loc[(p, s), "metabolic_cost"],
                                 tasks=b[pop], tasks_strong=b10[pop], sil=sil.loc[(p, s, pop)]))
    return pd.DataFrame(rows)


def main():
    d = table()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), sharex=True, gridspec_kw=dict(hspace=0.16, wspace=0.28))
    for r, pop in enumerate("EI"):
        for c, (col, ylab) in enumerate((("tasks", "tasks per neuron"), ("sil", "silhouette (mean, k = 3–20)"))):
            ax = axes[r, c]
            g = d[d["pop"] == pop]
            for (level, pts), ls in zip(LEVELS.items(), ("-", "--")):
                m = g[g.point.isin(pts)].groupby("point").agg(rate=("rate", lambda x: np.exp(np.log(x).mean())),
                                                              y=(col, "mean")).loc[pts]
                ax.plot(m.rate, m.y, color="0.55", ls=ls, lw=1.2, zorder=1, label=level)
            for p in POINTS:
                q = g[g.point == p]
                ax.scatter(q.rate, q[col], s=34, color=COLORS[p], edgecolors="white", lw=0.6, zorder=3,
                           label=NAME[p])
            ax.set_xscale("log")
            ax.tick_params(labelsize=9)
            ax.set_ylabel(ylab, fontsize=10)
            if r == 0:
                ax.set_title(("Selectivity" if c == 0 else "Distinct groups"), fontsize=11.5, pad=6)
            if r == 1:
                ax.set_xlabel("rate cost", fontsize=10.5)
        axes[r, 0].text(-0.3, 0.5, f"{pop}\nneurons", transform=axes[r, 0].transAxes, ha="center", va="center",
                        fontsize=12, color=E_COL if pop == "E" else I_COL)
    h, lab = axes[0, 0].get_legend_handles_labels()
    order = [2, 3, 4, 5, 6, 0, 1]                              # points first, then the two line styles
    fig.legend([h[i] for i in order], [lab[i] for i in order], loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.52, 1.05), fontsize=9.5, columnspacing=1.2, handletextpad=0.3)
    save(fig, "selectivity", RUN)

    print(d.groupby(["pop", "point"])[["rate", "tasks", "tasks_strong", "sil"]].mean().round(3).to_string())
    for pop in "EI":
        for level, pts in LEVELS.items():
            g = d[(d["pop"] == pop) & d.point.isin(pts)]
            for col in ("tasks", "tasks_strong", "sil"):
                rho, pv = spearmanr(g.rate, g[col])
                print(f"{pop} {level:23s} {col:12s} Spearman with rate cost {rho:+.2f} (p = {pv:.2g}, n = {len(g)})")


if __name__ == "__main__":
    main()
