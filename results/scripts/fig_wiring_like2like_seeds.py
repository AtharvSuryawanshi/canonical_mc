"""Like-to-like wiring AUC for all 20 networks of the 5-seed run (4 points x 5 seeds), fanin, 150k.

Same measure as fig_wiring_like2like.py (per type E->E, E->I, I->E, I->I: AUC = P(a connected pair has more similar
period fingerprints than an unconnected one); p from 500 shuffles of the fingerprints within E and within I). Per-period
task variance is computed and cached by fig_tv_period.load (results/front/tv_period_<point>_s<seed>.npz).

    python results/scripts/fig_wiring_like2like_seeds.py

Writes results/front/wiring_like2like_5seed.png and wiring_like2like_5seed.csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_tv_ei_fanin import COLORS, POINTS
from fig_wiring_like2like import QUADS, TYPES, network

REPO = Path(__file__).resolve().parents[2]
SEEDS = range(5)
EXAMPLE_SEED = 4


def main():
    rng = np.random.default_rng(0)
    rows, dirs = [], {}
    for p in POINTS:
        for s in SEEDS:
            res = network(p, s, rng)
            dirs[(p, s)] = res["dir_ratio"]
            for q in QUADS:
                rows.append(dict(point=p, seed=s, type=q, auc=res[q]["auc"], p=res[q]["auc_p"],
                                 n_connected=res["n_conn"][q], strength_rho=res[q]["rho"]))
            print(p, s, {q: round(res[q]["auc"], 3) for q in QUADS})
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/front" / "wiring_like2like_5seed.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.05, right=0.99, top=0.82, bottom=0.12))
    xs = np.arange(len(POINTS))
    for ax, q in zip(axes, QUADS):
        d = df[df.type == q]
        for s in SEEDS:
            y = [d[(d.point == p) & (d.seed == s)].auc.item() for p in POINTS]
            ax.plot(xs, y, color="0.8", lw=0.8, zorder=1)
        for i, p in enumerate(POINTS):
            dp = d[d.point == p].sort_values("seed")
            jit = np.linspace(-0.12, 0.12, len(dp))
            sig = dp.p.values < 0.05
            ax.scatter(xs[i] + jit[sig], dp.auc.values[sig], color=COLORS[p], s=34, zorder=3)
            ax.scatter(xs[i] + jit[~sig], dp.auc.values[~sig], facecolors="white", edgecolors=COLORS[p], s=34, zorder=3)
            ex = dp.seed.values == EXAMPLE_SEED
            ax.scatter(xs[i] + jit[ex], dp.auc.values[ex], facecolors="none", edgecolors="k", s=110, lw=1, zorder=4)
            ax.plot([xs[i] - 0.2, xs[i] + 0.2], [dp.auc.mean()] * 2, color=COLORS[p], lw=2.2, zorder=2)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels([POINTS[p].replace(", ", "\n") for p in POINTS], fontsize=8)
        ax.set_title(q, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUC  (0.5 = no relation, > 0.5 = like-to-like)")
    fig.suptitle("Are neurons that work in the same task periods more often connected? 5-seed run, fanin, 150k\n"
                 "dot = one network (filled: p < 0.05 vs 500 fingerprint shuffles; open: n.s.), bar = mean, "
                 "grey lines join the same seed, ring = example seed 4", fontsize=10)
    out = REPO / "results/front" / "wiring_like2like_5seed.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("->", out)

    print("\nmean AUC (sd) [networks with p < 0.05]")
    for q in QUADS:
        print(q, "  ".join(f"{p}: {g.auc.mean():.2f} ({g.auc.std():.2f}) [{(g.p < 0.05).sum()}/5]"
                           for p, g in df[df.type == q].groupby("point", sort=False)))
    print("\nE->E direction obs/shuffled, rows pre / cols post = stimulus, memory, response; mean (min-max) over seeds")
    for p in POINTS:
        M = np.array([dirs[(p, s)] for s in SEEDS])
        print(p)
        for a, na in enumerate(TYPES):
            print(f"  {na:9s}", "  ".join(f"{np.nanmean(M[:, a, b]):.2f} ({np.nanmin(M[:, a, b]):.2f}-{np.nanmax(M[:, a, b]):.2f})"
                                          for b in range(3)))


if __name__ == "__main__":
    main()
