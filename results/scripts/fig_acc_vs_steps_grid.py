"""Accuracy vs training step at the four fanin points, 2 x 2 with one shared legend (for the slides).

Same data as fig_acc_vs_steps.py ... fanin: the save-at run, 1 seed, accuracy of every task saved along
training. Thin: per task; thick: mean over tasks (black) and worst task (red); dotted: feasibility (0.6).

    python results/scripts/fig_acc_vs_steps_grid.py

Writes results/front/acc_vs_steps_fanin_grid.pdf.
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("acc_vs_steps", HERE / "fig_acc_vs_steps.py")
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)

REPO = single.REPO
RUN = REPO / single.RUNS["fanin"][0]
POINTS = {"control": "Unconstrained", "A3": "A3, rate only", "B0": "B0, wiring only", "C0": "C0, both"}


def main():
    d = pd.read_csv(RUN / "runs_by_step.csv")
    tasks = list(single.NAMES)
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, (p, title) in zip(axes.flat, POINTS.items()):
        m = d[d.point == p].groupby("step").mean(numeric_only=True)
        k = m.index / 1000
        for i, t in enumerate(tasks):
            ax.plot(k, m[f"acc_{t}"], color=cmap(i), lw=0.9, alpha=0.8, label=single.NAMES[t])
        ax.plot(k, m["mean_acc"], color="k", lw=2.4, label="mean over tasks")
        ax.plot(k, m["min_task_acc"], color="crimson", lw=2.4, label="worst task")
        ax.axhline(0.6, color="0.5", ls=":", lw=1)
        ax.set_title(f"{title}  (worst task at 150k: {m['min_task_acc'].iloc[-1]:.2f})", fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("training step (×1000)")
    for ax in axes[:, 0]:
        ax.set_ylabel("accuracy")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7.5, ncol=1, frameon=False, loc="center left", bbox_to_anchor=(0.995, 0.5))
    fig.tight_layout()
    out = REPO / "results/front/acc_vs_steps_fanin_grid.pdf"
    fig.savefig(out, bbox_inches="tight")
    print("->", out)


if __name__ == "__main__":
    main()
