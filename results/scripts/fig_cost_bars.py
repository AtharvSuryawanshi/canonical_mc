"""Small bar chart: final accuracy, connectivity cost (L1) and firing cost (L2).

One figure per budget point, best seed (highest final min_task_acc). Accuracy
is on its own 0-1 scale; each cost is scaled to its maximum over the points
in the run, so the bars compare across points (the unconstrained network sets
the full-length bar). Raw values are written next to the bars.

    python results/scripts/fig_cost_bars.py
    python results/scripts/fig_cost_bars.py --point control --acc mean_acc

Writes results/cost_bars/<run>/<point>.png.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "runs" / "moo_zoom" / "dale_all_curves4_4seed_50k"
OUT_ROOT = REPO / "results" / "cost_bars"
POINT_ORDER = ["control", "rate_limited", "wiring_limited", "both"]


def plot(row, acc_col, cost_max, out):
    labels = ["Min Task\n Accuracy", "Connectivity\ncost (L1)", "Firing\ncost (L2)"]
    raw = [row[acc_col], row.wiring_cost, row.metabolic_cost]
    scaled = [raw[0], raw[1] / cost_max["wiring_cost"], raw[2] / cost_max["metabolic_cost"]]
    fig, ax = plt.subplots(figsize=(2.6, 1.3))
    y = range(len(labels))[::-1]
    ax.barh(y, scaled, height=0.6, color="#e0e0e0", edgecolor="k", lw=0.8)
    for yi, s, v in zip(y, scaled, raw):
        ax.text(s + 0.03, yi, f"{v:.2g}", va="center", fontsize=7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--point", nargs="*", default=None, help="Default: every point in the run.")
    ap.add_argument("--acc", default="min_task_acc", choices=["min_task_acc", "mean_acc"])
    args = ap.parse_args()

    df = pd.read_csv(args.run / "runs.csv")
    best = df.loc[df.groupby("point").min_task_acc.idxmax()].set_index("point")
    cost_max = best[["wiring_cost", "metabolic_cost"]].max()
    points = args.point or [p for p in POINT_ORDER if p in best.index] + \
        [p for p in best.index if p not in POINT_ORDER]
    out_dir = OUT_ROOT / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for point in points:
        row = best.loc[point]
        plot(row, args.acc, cost_max, out_dir / point)
        print(f"{point} seed {int(row.seed)}: {args.acc} {row[args.acc]:.3f}, "
              f"wiring {row.wiring_cost:.4f}, firing {row.metabolic_cost:.4f} -> {out_dir / point}.png")


if __name__ == "__main__":
    main()
