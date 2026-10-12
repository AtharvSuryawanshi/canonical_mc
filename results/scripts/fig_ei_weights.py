"""E/I recurrent weight histograms: E->E, E->I, I->E, I->I, one curve per budget point.

Best seed per point (highest final min_task_acc). W[pre, post] from
model.effective_w_rec(); |w| is divided by the E/I row scale (I rows are
n_e/n_i larger at init by construction), so E and I blocks share one axis.
Only present synapses (normalised |w| > 1e-2, the conn_frac threshold, as in
notebooks/cog20_front_explorer.py) are histogrammed; the legend gives each
block's connection density.

    python results/scripts/fig_ei_weights.py

Writes results/ei_weights/<run>/ei_weight_hist.png.
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmc.train_cog import load_checkpoint

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "runs" / "moo_zoom" / "dale_all_curves4_4seed_50k"
OUT_ROOT = REPO / "results" / "ei_weights"
W_THRESH = 1e-2
BLOCKS = {"E→E": ("E", "E"), "E→I": ("E", "I"), "I→E": ("I", "E"), "I→I": ("I", "I")}
POINTS = {  # name, colour
    "control": ("Unconstrained", "#1a9850"),
    "rate_limited": ("Rate-limited", "#4575b4"),
    "wiring_limited": ("Wiring-limited", "#d73027"),
    "both": ("Both", "#984ea3"),
}


def block_weights(path):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _ = load_checkpoint(path, device="cpu")
    W = np.abs(model.effective_w_rec().detach().numpy()) / model.ei_row_scale().detach().numpy()[:, None]
    mask = model.no_autapse.numpy() > 0
    present = (W > W_THRESH) & mask
    sl = {"E": slice(0, model.n_e), "I": slice(model.n_e, None)}
    out = {}
    for b, (pre, post) in BLOCKS.items():
        p = present[sl[pre], sl[post]]
        out[b] = (W[sl[pre], sl[post]][p], p.sum() / mask[sl[pre], sl[post]].sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()

    df = pd.read_csv(args.run / "runs.csv")
    best = df.loc[df.groupby("point").min_task_acc.idxmax()].set_index("point")
    data = {p: block_weights(args.run / best.loc[p, "checkpoint"]) for p in POINTS if p in best.index}
    allv = np.concatenate([v for d in data.values() for v, _ in d.values()])
    bins = np.logspace(np.log10(W_THRESH), np.log10(allv.max()), 30)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4), sharex=True)
    for ax, b in zip(axes, BLOCKS):
        for p, d in data.items():
            vals, dens = d[b]
            name, color = POINTS[p]
            ax.hist(vals, bins=bins, weights=np.full(len(vals), 1 / max(len(vals), 1)), histtype="step", lw=1.8,
                    color=color,
                    label=f"{name} ({dens:.1%})")
        ax.set_xscale("log")
        ax.set_xlabel(f"{b}  |w| (E/I-normalised)")
        ax.legend(fontsize=7, frameon=False, title="connection density", title_fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Fraction of synapses")
    fig.tight_layout()
    out_dir = OUT_ROOT / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "ei_weight_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    for p, d in data.items():
        print(p, f"seed {int(best.loc[p, 'seed'])}", {b: f"{dens:.1%}, median {np.median(v):.3g}" for b, (v, dens) in d.items()})
    print("->", out_dir / "ei_weight_hist.png")


if __name__ == "__main__":
    main()
