"""The like-to-like AUC as an ROC picture, seed 4: control, B0 and C0, for E->E and I->E.

For each network and type: the distribution of fingerprint similarity for connected pairs vs unconnected pairs
(fig_wiring_like2like.py), and the ROC curve obtained by sliding a similarity threshold t:
    TPR(t) = fraction of connected pairs with similarity > t,  FPR(t) = fraction of unconnected pairs with similarity > t.
The area under it is the AUC (0.5 = the two distributions coincide = the diagonal).

    python results/scripts/fig_wiring_roc.py [--seed 4]

Writes results/front/wiring_roc_s<seed>.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fig_tv_ei_fanin import POINTS
from fig_wiring_like2like import network

REPO = Path(__file__).resolve().parents[2]
NETS, TYPES = ("control", "B0", "C0"), ("E→E", "I→E")
CONN, UNCONN = "#1F8AD0", "#9a9a9a"


def roc(s, c):
    """TPR, FPR over all thresholds (descending similarity)."""
    o = np.argsort(-s, kind="stable")
    cc = c[o]
    tpr = np.r_[0, np.cumsum(cc) / cc.sum()]
    fpr = np.r_[0, np.cumsum(~cc) / (~cc).sum()]
    return fpr, tpr


def main(seed=4):
    rng = np.random.default_rng(0)
    R = {p: network(p, seed, rng) for p in NETS}
    fig, axes = plt.subplots(len(TYPES), 2 * len(NETS), figsize=(18, 6.2),
                             gridspec_kw=dict(width_ratios=[1.35, 1] * len(NETS), wspace=0.35, hspace=0.5,
                                              left=0.04, right=0.99, top=0.86, bottom=0.09))
    bins = np.linspace(-1, 1, 41)
    for r, q in enumerate(TYPES):
        for j, p in enumerate(NETS):
            res = R[p]
            mk = res["masks"][q]
            s, c = res["S"][mk], res["C"][mk]
            ax = axes[r, 2 * j]
            ax.hist(s[~c], bins, density=True, color=UNCONN, alpha=0.55, label=f"unconnected ({(~c).sum()})")
            ax.hist(s[c], bins, density=True, histtype="step", lw=2, color=CONN, label=f"connected ({c.sum()})")
            ax.axvline(np.median(s[~c]), color=UNCONN, ls=":", lw=1.2)
            ax.axvline(np.median(s[c]), color=CONN, ls=":", lw=1.2)
            ax.set_xlabel("fingerprint similarity (pre, post)", fontsize=8)
            ax.set_title(f"{POINTS[p]}, {q}", fontsize=10, loc="left")
            ax.legend(fontsize=7, frameon=False, loc="upper left")
            ax.tick_params(labelsize=7); ax.set_yticks([])
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax = axes[r, 2 * j + 1]
            fpr, tpr = roc(s, c)
            ax.fill_between(fpr, tpr, color=CONN, alpha=0.12)
            ax.plot(fpr, tpr, color=CONN, lw=2)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("FPR: unconnected above t", fontsize=7); ax.set_ylabel("TPR: connected above t", fontsize=7)
            ax.set_title(f"ROC, AUC {res[q]['auc']:.2f}", fontsize=10)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Like-to-like wiring as two distributions, seed {seed}: similarity of connected (blue) vs unconnected "
                 f"(grey) pairs, dotted = medians.\nSliding a threshold t over similarity traces the ROC; its area is the "
                 f"AUC (diagonal = the two distributions are the same)", fontsize=10)
    out = REPO / "results/front" / f"wiring_roc_s{seed}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    main(ap.parse_args().seed)
