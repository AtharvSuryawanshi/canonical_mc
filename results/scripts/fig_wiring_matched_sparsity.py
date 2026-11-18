"""Is C0's like-to-like wiring just sparsity? Same AUC as fig_wiring_like2like_seeds.py, at matched sparsity.

For every network of the 5-seed run (4 points x 5 seeds), 'connected' = its strongest synapses (E/I-normalised |W|),
keeping the same fraction of active off-diagonal pairs as the C0 network with the same seed is connected
(|W|/c > 0.01). C0 is therefore unchanged; control, A3 and B0 are cut down to C0's sparsity after training.
If their AUC rises to C0's level, sparsity explains it; if it stays near 0.5, it does not. Post-hoc cutting is not the
same as training under that budget, so this is a hint, not proof.

    python results/scripts/fig_wiring_matched_sparsity.py

Writes results/front/wiring_like2like_matched_5seed.png and .csv.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_tv_ei_fanin import COLORS, POINTS
from fig_wiring_like2like import QUADS, network

REPO = Path(__file__).resolve().parents[2]
SEEDS = range(5)


def main():
    rng = np.random.default_rng(0)
    orig = pd.read_csv(REPO / "results/front" / "wiring_like2like_5seed.csv")
    rows = []
    for s in SEEDS:
        keep = network("C0", s, rng)["conn"]                                # C0's connected fraction, this seed
        for p in POINTS:
            res = network(p, s, rng, keep_frac=keep)
            for q in QUADS:
                rows.append(dict(point=p, seed=s, type=q, keep_frac=keep, auc=res[q]["auc"], p=res[q]["auc_p"],
                                 n_connected=res["n_conn"][q]))
            print(p, s, f"keep {keep:.4f}", {q: round(res[q]["auc"], 3) for q in QUADS})
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/front" / "wiring_like2like_matched_5seed.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True,
                             gridspec_kw=dict(wspace=0.08, left=0.05, right=0.99, top=0.8, bottom=0.12))
    xs = np.arange(len(POINTS))
    for ax, q in zip(axes, QUADS):
        for i, p in enumerate(POINTS):
            for d, dx, filled, lab in ((orig, -0.15, False, "as trained (|W|/c > 0.01)"),
                                       (df, 0.15, True, "strongest synapses, C0's sparsity")):
                v = d[(d.type == q) & (d.point == p)].auc.values
                jit = np.linspace(-0.06, 0.06, len(v))
                ax.scatter(xs[i] + dx + jit, v, s=30, edgecolors=COLORS[p], linewidths=1.2,
                           facecolors=COLORS[p] if filled else "white", zorder=3,
                           label=lab if (i == 0 and q == QUADS[0]) else None)
                ax.plot([xs[i] + dx - 0.1, xs[i] + dx + 0.1], [v.mean()] * 2, color=COLORS[p], lw=2)
        ax.axhline(0.5, color="k", ls="--", lw=0.8)
        ax.set_xticks(xs); ax.set_xticklabels([POINTS[p].replace(", ", "\n") for p in POINTS], fontsize=8)
        ax.set_title(q, fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("AUC  (0.5 = no relation)")
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Is C0's like-to-like wiring just sparsity? Open: connected as trained. Filled: only each network's "
                 "strongest synapses,\nas many as the same-seed C0 network has (~0.7-0.9%). 5 seeds per point, bar = mean",
                 fontsize=10)
    out = REPO / "results/front" / "wiring_like2like_matched_5seed.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("->", out)
    print("\nmean AUC: as trained -> matched sparsity")
    for q in QUADS:
        print(q, "  ".join(f"{p}: {orig[(orig.type == q) & (orig.point == p)].auc.mean():.2f} -> "
                           f"{df[(df.type == q) & (df.point == p)].auc.mean():.2f}" for p in POINTS))


if __name__ == "__main__":
    main()
