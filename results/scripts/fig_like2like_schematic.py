"""Like-to-like connectivity, explained with real neurons (slide 14). One network (default: front run, both constraints,
seed 0), E->E pairs, per-task profiles (as like2like.py --basis task):

  left    a connected pair with similar task profiles: two E neurons, synapse arrow, both profiles (20 tasks), r
  middle  an unconnected pair with dissimilar profiles: no arrow, r
  right   all E->E pairs of the network: similarity of connected vs unconnected pairs (the two example pairs marked),
          and the AUC

Profiles: task variance per task (checkpoint), each active neuron divided by its maximum. Connected: |W|/c > 0.01.

    python results/scripts/fig_like2like_schematic.py [--point both] [--seed 0] [--run front]

Writes presentation/figs/talk/like2like_schematic_<run>.pdf and prints the chosen pairs.
"""
import argparse
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from cmc.task import rule_name
from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import RUNS
from fig_wiring_like2like import THRESH, auc
from fig_wiring_roc import roc
from talk_style import E_COL, LABEL, UNCONN, W, save

LIGHT = "#9CCBEA"                          # second neuron of a pair (lighter E blue)


def network(run, point, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, ck = load_checkpoint(RUNS[run] / point / f"seed_{seed:02d}.pt")
    tv = np.asarray(ck["task_variance"], float)                        # (tasks, N)
    total = tv.sum(0)
    units = np.flatnonzero(total > 1e-2 * total.max())
    E = units[units < model.n_e]
    X = tv[:, E] / tv[:, E].max(0, keepdims=True)                      # profiles, (tasks, E units)
    S = np.corrcoef(X.T)
    Wn = np.abs(model.effective_w_rec().detach().numpy()) / model.ei_row_scale().detach().numpy()[:, None]
    C = Wn[np.ix_(E, E)] > THRESH                                       # C[pre, post]
    return list(ck["active_tasks"]), X, S, C, E


def pick_pairs(S, C, X):
    """Connected pair: highest similarity among connected pairs whose profiles both vary (not flat); unconnected
    pair: similarity closest to -0.1 among unconnected pairs with varied profiles (a clearly different pair)."""
    n = len(S)
    off = ~np.eye(n, dtype=bool)
    varied = (X.std(0) > 0.2)
    ok = off & np.outer(varied, varied)
    target = np.quantile(S[ok & C], 0.9)                   # a strong but typical connected pair, not the extreme
    i, j = np.unravel_index(np.where(ok & C, -np.abs(S - target), -np.inf).argmax(), S.shape)
    a, b = np.unravel_index(np.where(ok & ~C & ~C.T, -np.abs(S + 0.1), -np.inf).argmax(), S.shape)
    return (i, j), (a, b)


def pair_panel(fig, spec, tasks, X, pair, S, connected):
    i, j = pair
    sub = spec.subgridspec(3, 1, height_ratios=[0.9, 1, 1], hspace=0.15)
    ax = fig.add_subplot(sub[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for x, col, name in ((0.15, E_COL, "a"), (0.85, LIGHT, "b")):
        ax.scatter([x], [0.4], marker="^", s=330, color=col, edgecolors="k", linewidths=0.6, zorder=3)
        ax.text(x, 0.33, name, ha="center", va="center", fontsize=8, color="white" if col == E_COL else "k",
                zorder=4)
    if connected:
        ax.add_patch(FancyArrowPatch((0.25, 0.4), (0.75, 0.4), arrowstyle="-|>", mutation_scale=10, lw=1.4,
                                     color="k"))
        ax.text(0.5, 0.55, "synapse", ha="center", fontsize=7.5)
    else:
        ax.plot([0.25, 0.75], [0.4, 0.4], color="0.6", lw=1, ls=":")
        ax.text(0.5, 0.55, "no synapse", ha="center", fontsize=7.5, color="0.4")
    ax.set_title(("connected" if connected else "unconnected") + f": r = {S[i, j]:+.2f}", fontsize=8.5, pad=2)
    x = np.arange(len(tasks))
    for k, (u, col, name) in enumerate(((i, E_COL, "a"), (j, LIGHT, "b"))):
        axp = fig.add_subplot(sub[k + 1])
        axp.bar(x, X[:, u], color=col, ec="k" if col == LIGHT else col, lw=0.3, width=0.8)
        axp.set_ylim(0, 1.25); axp.set_xlim(-0.6, len(tasks) - 0.4)
        axp.set_yticks([0, 1]); axp.tick_params(labelsize=6.5, length=2, pad=1)
        axp.text(0.01, 0.98, f"neuron {name}", transform=axp.transAxes, ha="left", va="top", fontsize=7)
        axp.set_xticks([])
        if k == 1:
            axp.set_xlabel("20 tasks", fontsize=7.5, labelpad=1)
    if connected:                                           # shared y label for the two profiles (first pair only)
        fig.add_subplot(sub[1:]).set_axis_off()
        fig.axes[-1].text(-0.13, 0.5, "task variance / its max", transform=fig.axes[-1].transAxes, rotation=90,
                          ha="center", va="center", fontsize=7)
    return sub


def main(run, point, seed):
    tasks, X, S, C, E = network(run, point, seed)
    (pc, pu) = pick_pairs(S, C, X)
    fig = plt.figure(figsize=(W * 1.15, 2.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1.1, 0.75], wspace=0.38, left=0.05, right=0.99, top=0.86,
                          bottom=0.17)
    pair_panel(fig, gs[0], tasks, X, pc, S, connected=True)
    pair_panel(fig, gs[1], tasks, X, pu, S, connected=False)

    ax = fig.add_subplot(gs[2])
    off = ~np.eye(len(S), dtype=bool)
    s, c = S[off], C[off]
    bins = np.linspace(-1, 1, 21)
    ax.hist(s[~c], bins, density=True, color=UNCONN, lw=0, label="unconnected")
    ax.hist(s[c], bins, density=True, histtype="step", color=E_COL, lw=1.3, label="connected")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.45)                              # room above the bars for the example labels
    for (i, j), col, lab, ha, rel in ((pc, "k", "connected\nexample", "center", (0.5, 0)),
                                      (pu, "0.4", "unconnected\nexample", "right", (1, 0))):
        ax.annotate(lab, (S[i, j], 0), (S[i, j], top * 1.08), ha=ha, va="bottom", fontsize=6.5, color=col,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, relpos=rel))
    ax.set_xlim(-1, 1); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("similarity r of the two profiles", fontsize=8)
    ax.set_ylabel("fraction of pairs", fontsize=7.5, labelpad=2)
    ax.tick_params(labelsize=7)
    ax.set_title(f"all E→E pairs: AUC = {auc(s, c):.2f}", fontsize=9, pad=12)
    ax.text(0.5, 1.01, "grey: unconnected, line: connected", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=6.5, color="0.35")

    ax = fig.add_subplot(gs[3])                             # ROC: slide a threshold over r
    f, t = roc(s, c)
    ax.fill_between(f, t, color=E_COL, alpha=0.18, lw=0)
    ax.plot(f, t, color=E_COL, lw=1.3)
    ax.plot([0, 1], [0, 1], color="0.3", ls=":", lw=0.7)
    ax.text(0.62, 0.3, f"area =\nAUC = {auc(s, c):.2f}", ha="center", fontsize=7)
    ax.text(0.86, 0.74, "chance", rotation=45, fontsize=6.5, color="0.4", ha="center", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_box_aspect(1)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.tick_params(labelsize=7)
    ax.set_xlabel("false positive rate\n(unconnected pairs above r)", fontsize=6.5, labelpad=1)
    ax.set_ylabel("true positive rate\n(connected pairs above r)", fontsize=6.5, labelpad=1)
    ax.set_title("ROC", fontsize=9, pad=12)
    save(fig, "like2like_schematic", run)
    for name, (i, j) in (("connected", pc), ("unconnected", pu)):
        top = lambda u: [rule_name[tasks[t]] for t in np.argsort(-X[:, u])[:3]]
        print(f"{name}: units {E[i]} -> {E[j]}, r = {S[i, j]:+.2f}, top tasks {top(i)} / {top(j)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="front")
    ap.add_argument("--point", default="both")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.run, a.point, a.seed)
