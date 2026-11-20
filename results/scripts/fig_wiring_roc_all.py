"""Like-to-like wiring at all four points: similarity distributions + ROC, one seed, and the 5-seed summary.

Measure as in fig_wiring_like2like.py: per type (E->E, E->I, I->E, I->I), fingerprint similarity of connected vs
unconnected pairs (connected = |W|/c > 0.01); sliding a threshold over similarity gives the ROC, its area the AUC.

Figure 1 (one seed): rows = types, columns = points; each panel shows the two similarity distributions as
histograms (20 bins of width 0.1 on [-1, 1], same bins for both, each scaled to area 1; no smoothing, which blurred C0's few connected pairs), with the
ROC in a small panel to the right (an inset hid the right end of the histograms).
Figure 2 (all 5 seeds): top row = mean ROC per point (vertical average over seeds, band = min-max), bottom row = AUC
per network (filled = p < 0.05 against 500 fingerprint shuffles), grey lines join the same seed.

    python results/scripts/fig_wiring_roc_all.py [--seed 4] [--bins 40 20 sqrt]

Writes results/front/wiring_roc_all_s<seed>.png and wiring_roc_all_5seed.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig_tv_ei_fanin import COLORS, POINTS
from fig_wiring_like2like import QUADS, network
from fig_wiring_roc import roc

REPO = Path(__file__).resolve().parents[2]
SEEDS = range(5)
E_COL, I_COL, UNCONN = "#1F8AD0", "#E0393E", "#a8a8a8"
GRID = np.linspace(0, 1, 101)
BIN_RULES = ("40", "20", "sqrt")


def bins_for(rule, n_small):
    """Bin edges on [-1, 1], one width for both histograms of a panel: 40 or 20 bins, or (sqrt) sqrt(n) bins of the
    smaller group (connected or unconnected), 5..40."""
    k = int(np.clip(np.ceil(np.sqrt(n_small)), 5, 40)) if rule == "sqrt" else int(rule)
    return np.linspace(-1, 1, k + 1)


def one_seed(R, seed, rule="20"):
    fig = plt.figure(figsize=(18, 13))
    outer = fig.add_gridspec(len(QUADS), len(POINTS), hspace=0.42, wspace=0.14, left=0.06, right=0.99, top=0.86,
                             bottom=0.05)
    for r, q in enumerate(QUADS):
        col = E_COL if q[0] == "E" else I_COL
        for j, p in enumerate(POINTS):
            res = R[(p, seed)]
            mk = res["masks"][q]
            s, c = res["S"][mk], res["C"][mk]
            inner = outer[r, j].subgridspec(1, 2, width_ratios=[2.6, 1], wspace=0.08)
            ax, ax_roc = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            bins = bins_for(rule, min(c.sum(), (~c).sum()))
            hu = ax.hist(s[~c], bins, density=True, color=UNCONN, alpha=0.5, label=f"unconnected ({(~c).sum()})")[0]
            hc = ax.hist(s[c], bins, density=True, histtype="step", color=col, lw=1.8, label=f"connected ({c.sum()})")[0]
            ax.text(0.99, 0.97, f"{len(bins) - 1} bins", transform=ax.transAxes, ha="right", va="top", fontsize=6,
                    color="0.4")
            ax.axvline(np.median(s[~c]), color="0.45", ls=":", lw=1.2)
            ax.axvline(np.median(s[c]), color=col, ls=":", lw=1.2)
            ax.set_xlim(-1, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_ylim(0, max(hu.max(), hc.max()) * 1.3)
            ax.set_yticks([]); ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6.5, frameon=False, loc="upper left")
            if r == 0:
                ax.text(0.68, 1.25, POINTS[p], transform=ax.transAxes, ha="center", fontsize=12, color=COLORS[p],
                        weight="bold")
            if j == 0:
                ax.set_ylabel(q, fontsize=13, rotation=0, ha="right", va="center", labelpad=12)
            if r == len(QUADS) - 1:
                ax.set_xlabel("fingerprint similarity (pre, post)", fontsize=8)
            f, t = roc(s, c)
            ax_roc.fill_between(f, t, color=col, alpha=0.15); ax_roc.plot(f, t, color=col, lw=1.5)
            ax_roc.plot([0, 1], [0, 1], "k--", lw=0.6)
            ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1); ax_roc.set_xticks([]); ax_roc.set_yticks([])
            ax_roc.set_box_aspect(1)
            ax_roc.set_title(f"AUC {res[q]['auc']:.2f}" + ("" if res[q]["auc_p"] < 0.05 else "\n(n.s.)"),
                             fontsize=9, pad=3)
            if r == len(QUADS) - 1:
                ax_roc.set_xlabel("ROC", fontsize=8)
    fig.suptitle(f"Are neurons that work in the same task periods more often connected? Seed {seed} (5-seed run, fanin, "
                 f"150k)\nfingerprint similarity of connected (colour) vs unconnected (grey) pairs, histograms scaled "
                 f"to area 1, dotted = medians; right: ROC, its area = AUC (0.5 = same distributions)", fontsize=11,
                 y=0.985)
    out = REPO / "results/front" / f"wiring_roc_all_s{seed}{'' if rule == '20' else '_bins' + rule}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("->", out)


def all_seeds(R):
    fig, axes = plt.subplots(2, len(QUADS), figsize=(16, 8.4), gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.35,
                             wspace=0.15, left=0.05, right=0.99, top=0.88, bottom=0.08))
    xs = np.arange(len(POINTS))
    for k, q in enumerate(QUADS):
        ax = axes[0, k]
        for p in POINTS:
            T = []
            for s in SEEDS:
                res = R[(p, s)]
                mk = res["masks"][q]
                f, t = roc(res["S"][mk], res["C"][mk])
                T.append(np.interp(GRID, f, t))
            T = np.array(T)
            a = [R[(p, s)][q]["auc"] for s in SEEDS]
            ax.fill_between(GRID, T.min(0), T.max(0), color=COLORS[p], alpha=0.18, lw=0)
            ax.plot(GRID, T.mean(0), color=COLORS[p], lw=2, label=f"{POINTS[p]}: {np.mean(a):.2f} ± {np.std(a):.2f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("FPR (unconnected above threshold)", fontsize=8)
        if k == 0:
            ax.set_ylabel("TPR (connected above threshold)", fontsize=8)
        ax.set_title(q, fontsize=13)
        ax.legend(fontsize=7, frameon=False, loc="lower right", title="mean AUC ± sd", title_fontsize=7)
        ax.tick_params(labelsize=7)
        ax = axes[1, k]
        for s in SEEDS:
            ax.plot(xs, [R[(p, s)][q]["auc"] for p in POINTS], color="0.82", lw=0.8, zorder=1)
        for i, p in enumerate(POINTS):
            a = np.array([R[(p, s)][q]["auc"] for s in SEEDS])
            sig = np.array([R[(p, s)][q]["auc_p"] < 0.05 for s in SEEDS])
            jit = np.linspace(-0.12, 0.12, len(a))
            ax.scatter(xs[i] + jit[sig], a[sig], color=COLORS[p], s=30, zorder=3)
            ax.scatter(xs[i] + jit[~sig], a[~sig], facecolors="white", edgecolors=COLORS[p], s=30, zorder=3)
            ax.plot([xs[i] - 0.2, xs[i] + 0.2], [a.mean()] * 2, color=COLORS[p], lw=2.2, zorder=2)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_ylim(0.4, 0.85)
        ax.set_xticks(xs); ax.set_xticklabels([POINTS[p].replace(", ", "\n") for p in POINTS], fontsize=8)
        if k == 0:
            ax.set_ylabel("AUC per network")
        else:
            ax.set_yticklabels([])
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Like-to-like wiring over all 20 networks (4 points x 5 seeds, fanin, 150k). Top: mean ROC per point, "
                 "band = min-max over seeds.\nBottom: AUC per network (filled = p < 0.05 vs 500 fingerprint shuffles), "
                 "bar = mean, grey lines join the same seed", fontsize=11)
    out = REPO / "results/front" / "wiring_roc_all_5seed.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("->", out)


def main(seed=4, rules=("20",)):
    rng = np.random.default_rng(0)
    R = {(p, s): network(p, s, rng) for p in POINTS for s in SEEDS}
    for rule in rules:
        one_seed(R, seed, rule)
    all_seeds(R)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--bins", nargs="+", choices=BIN_RULES, default=["20"],
                    help="histogram bins of figure 1: 20 (default, no suffix), 40, or sqrt(n) of the smaller group (suffix _bins<rule>)")
    a = ap.parse_args()
    main(a.seed, a.bins)
