"""Driscoll et al. 2024 Fig. 3a style variance matrix for one network (default: 5-seed run, seed 4, control and C0).

Uses the per-period task variance cached by fig_tv_period.py (cmc.runner.period_variance). As in Driscoll's code
(stepnet/analysis/clustering.py, tools_lnd.py::make_dendro): each active unit divided by its own max over task
periods, Ward hierarchical clustering of the units (columns) and of the task periods (rows), the number of clusters
on each axis = the global silhouette maximum over 3..39. One population (no E/I split in the clustering); a strip under
the top dendrogram marks E / I. Differences from Driscoll: our active rule (total > 1% of max, not > 1e-3), random
eval trials instead of a stimulus grid, and k by silhouette instead of the hand-set distance cut of their figure.

    python results/scripts/fig_tv_period_driscoll.py [--points control C0] [--seed 4]

Writes results/front/tv_period_driscoll_<point>_s<seed>.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score

from cmc.task import rule_name
from fig_tv_ei_fanin import POINTS
from fig_tv_period import load

REPO = Path(__file__).resolve().parents[2]
PERIOD_BG = {"stim1": "#c9c9ea", "stim2": "#a9e4e4", "delay1": "#d8f3d0", "delay2": "#f3e9c8", "go1": "#e3d3cf"}
PERIOD_LEG = {"stim1": "Stimulus 1", "stim2": "Stimulus 2", "delay1": "Memory 1", "delay2": "Memory 2", "go1": "Response"}
PALETTE = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628", "#984ea3", "#e41a1c", "#dede00"]
E_COL, I_COL = "#1F8AD0", "#E0393E"


def cname(n):
    """a..z, then aa, ab, ... (Driscoll labels unit clusters with letters)."""
    return chr(97 + n) if n < 26 else chr(97 + n // 26 - 1) + chr(97 + n % 26)


def best_k(X, Y, k_max=39):
    ks = range(3, min(k_max, len(X) - 1) + 1)
    sil = {k: silhouette_score(X, sch.fcluster(Y, k, criterion="maxclust")) for k in ks}
    k = max(sil, key=sil.get)
    return k, sil[k]


def cut_height(Y, k):
    """A height between the merges that leave k and k-1 clusters, for colouring the dendrogram."""
    return (Y[-k, 2] + Y[-(k - 1), 2]) / 2


def main(point, seed):
    model, ck, tv_t, tv_p, keys = load(point, seed)
    total = tv_t.sum(0)
    units = np.flatnonzero(total > 1e-2 * total.max())
    X = tv_p[:, units] / tv_p[:, units].max(0, keepdims=True)      # (rows = task periods, cols = units)
    isE = units < model.n_e

    Yu, Yr = sch.linkage(X.T, method="ward"), sch.linkage(X, method="ward")
    ku, su = best_k(X.T, Yu)
    kr, sr = best_k(X, Yr)
    lab_u = sch.fcluster(Yu, ku, criterion="maxclust")
    lab_r = sch.fcluster(Yr, kr, criterion="maxclust")

    nR, nU = X.shape
    fig = plt.figure(figsize=(10, 12.5))
    L, W, B, H = 0.20, 0.66, 0.06, 0.74
    ax_top = fig.add_axes([L, B + H + 0.045, W, 0.08])
    ax_ei = fig.add_axes([L, B + H + 0.03, W, 0.012])
    ax_m = fig.add_axes([L, B, W, H])
    ax_r = fig.add_axes([L + W + 0.005, B, 0.09, H])
    ax_cb = fig.add_axes([L + W * 0.55, B - 0.035, W * 0.4, 0.01])

    sch.set_link_color_palette(PALETTE)
    Zu = sch.dendrogram(Yu, ax=ax_top, color_threshold=cut_height(Yu, ku), above_threshold_color="#888888",
                        no_labels=True)
    Zr = sch.dendrogram(Yr, ax=ax_r, orientation="right", color_threshold=cut_height(Yr, kr),
                        above_threshold_color="#888888", no_labels=True)
    ou, orow = Zu["leaves"], Zr["leaves"]
    for ax in (ax_top, ax_r):
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_visible(False)
    ax_r.set_ylim(10 * nR, 0)                                        # first leaf on top, as the matrix

    im = ax_m.imshow(X[np.ix_(orow, ou)], cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                     extent=(0, 10 * nU, 10 * nR, 0))
    ax_m.set_xlim(0, 10 * nU); ax_top.set_xlim(0, 10 * nU)
    ax_m.set_xticks([]); ax_m.set_yticks([])

    # unit cluster letters (dendrogram leaf colours) and E/I strip
    lu = lab_u[ou]
    starts = np.r_[0, np.flatnonzero(np.diff(lu)) + 1]
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else nU
        c = Zu["leaves_color_list"][s]
        ax_top.text(10 * (s + e) / 2, -0.04, cname(n), transform=ax_top.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color=c if c != "#888888" else "k")
        if s:
            ax_m.axvline(10 * s, color="white", lw=0.4)
    ax_ei.imshow(np.where(isE[ou], 1, 0)[None], cmap=matplotlib.colors.ListedColormap([I_COL, E_COL]), aspect="auto",
                 extent=(0, 10 * nU, 0, 1), interpolation="nearest")
    ax_ei.set_xlim(0, 10 * nU); ax_ei.set_xticks([]); ax_ei.set_yticks([])
    ax_ei.set_ylabel("E/I", rotation=0, fontsize=6, ha="right", va="center")

    # row labels with period-type background, row cluster numbers
    for j, i in enumerate(orow):
        t, p = keys[i]
        ax_m.text(-0.01, 1 - (j + 0.5) / nR, rule_name[t], transform=ax_m.transAxes, ha="right", va="center", fontsize=6,
                  bbox=dict(boxstyle="square,pad=0.12", fc=PERIOD_BG[p], ec="none"))
    lr = lab_r[orow]
    rs = np.r_[0, np.flatnonzero(np.diff(lr)) + 1]
    for n, s in enumerate(rs):
        e = rs[n + 1] if n + 1 < len(rs) else nR
        c = Zr["leaves_color_list"][s]
        ax_r.text(1.02, 1 - (s + e) / 2 / nR, str(n + 1), transform=ax_r.transAxes, ha="left", va="center",
                  fontsize=9, color=c if c != "#888888" else "k")
        if s:
            ax_m.axhline(10 * s, color="white", lw=0.6)

    cb = fig.colorbar(im, cax=ax_cb, orientation="horizontal")
    cb.set_label("normalized variance", fontsize=8); cb.ax.tick_params(labelsize=7)
    for n, (p, name) in enumerate(PERIOD_LEG.items()):
        fig.text(0.03 + 0.085 * n, B - 0.03, name, fontsize=7, va="center",
                 bbox=dict(boxstyle="square,pad=0.25", fc=PERIOD_BG[p], ec="none"))
    nE, nI = int(isE.sum()), int((~isE).sum())
    fig.suptitle(f"{POINTS[point]}, seed {seed} (5-seed run, fanin, 150k): {nU} active units ({nE} E, {nI} I)\n"
                 f"Ward on units: {ku} clusters (sil {su:.2f}); Ward on task periods: {kr} clusters (sil {sr:.2f})",
                 fontsize=10, y=0.985)
    out = REPO / "results/front" / f"tv_period_driscoll_{point}_s{seed}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("->", out)

    print(f"{point}: units {ku} clusters (sil {su:.2f}), task periods {kr} clusters (sil {sr:.2f})")
    for n, s in enumerate(rs):
        e = rs[n + 1] if n + 1 < len(rs) else nR
        members = [keys[i] for i in orow[s:e]]
        types = {}
        for _, p in members:
            types[PERIOD_LEG[p]] = types.get(PERIOD_LEG[p], 0) + 1
        print(f"  row cluster {n + 1}: {len(members)} periods {types}  e.g. "
              + ", ".join(f"{rule_name[t]}·{PERIOD_LEG[p]}" for t, p in members[:4]))
    comp = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else nU
        u = np.array(ou[s:e])
        comp.append(f"{cname(n)}:{int(isE[u].sum())}E/{int((~isE[u]).sum())}I")
    print("  unit clusters:", " ".join(comp))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", nargs="+", choices=sorted(POINTS), default=["control", "C0"])
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    for p in a.points:
        main(p, a.seed)
