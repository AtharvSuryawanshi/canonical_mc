"""The four module-contrast heatmaps of fig_lesion_period.py side by side, one shared colour scale.

Reads results/front/lesion_period_<point>_s<seed><tag>.csv and _modules_contrast.csv (no re-running).

    python results/scripts/fig_lesion_contrast_grid.py [--seed 4] [--tag _ward]

Writes results/front/lesion_period_s<seed><tag>_contrast_grid.png.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
POINTS = {"control": "Unconstrained", "A3": "A3, rate only", "B0": "B0, wiring only", "C0": "C0, both"}
REF = {"Anti": "not Anti", "Delay": "not Delay", "Modality1": "Modality2", "Modality2": "Modality1",
       "Cat Mem.": "Contin Mem.", "Contin Mem.": "Cat Mem."}


def main(seed, tag):
    out_dir = REPO / "results/front"
    data = {}
    for p in POINTS:
        R = pd.read_csv(out_dir / f"lesion_period_{p}_s{seed}{tag}.csv")
        C = pd.read_csv(out_dir / f"lesion_period_{p}_s{seed}{tag}_modules_contrast.csv", index_col=0)
        data[p] = (R, C.loc[R.cluster])
    lim = max(float(np.abs(C.values).max()) for _, C in data.values())
    tasks = [c for c in next(iter(data.values()))[0].columns if c.startswith("task_")]

    fig, axes = plt.subplots(1, 4, figsize=(17, 8.6), sharey=False)
    for ax, (p, (R, C)) in zip(axes, data.items()):
        im = ax.imshow(C.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for (i, j), v in np.ndenumerate(C.values):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.6 * lim else "k")
        ax.set_xticks(range(C.shape[1]))
        ax.set_xticklabels([f"{m}\nvs {REF[m]}" for m in C.columns], rotation=90, fontsize=7)
        ax.set_yticks(range(len(C)))
        ax.set_yticklabels([f"{c} ({n})" for c, n in zip(R.cluster, R.n)], fontsize=7)
        ax.axhline((R["pop"] == "E").sum() - 0.5, color="k", lw=2)
        n_strong = int((C.abs().max(axis=1) >= 0.2).sum())
        ax.set_title(f"{POINTS[p]}\nmean change per lesion {R[tasks].values.mean():+.2f}; "
                     f"|contrast| ≥ 0.2: {n_strong}/{len(C)}", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01).set_label(
        "mean change, module tasks − reference tasks (red = module hurt more)", fontsize=8)
    method = "Ward" if "ward" in tag else "k-means"
    k = R.groupby("pop").k.first()
    fig.suptitle(f"Seed {seed}: lesion of each per-period cluster ({method}, E and I separate, "
                 f"k = {k.get('E')} E / {k.get('I')} I), Driscoll's task modules", fontsize=11)
    out = out_dir / f"lesion_period_s{seed}{tag}_contrast_grid.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--tag", default="_ward")
    a = ap.parse_args()
    main(a.seed, a.tag)
