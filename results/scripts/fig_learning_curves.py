"""Per-task accuracy over training, best seed of each budget point.

One figure per point (control / rate_limited / both / wiring_limited), the 20
tasks as lines coloured by task family. "Best seed" = highest final
min_task_acc. Data: runs_by_step.csv of a ``cmc.moo_zoom --save-at`` run.

    python results/scripts/fig_learning_curves.py
    python results/scripts/fig_learning_curves.py --run runs/moo_zoom/<other_run>

Writes results/learning_curves/<run>/<point>.png, plus best_seeds.csv.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from cmc.task import rule_name

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "runs" / "moo_zoom" / "dale_all_curves4_4seed_50k"
OUT_ROOT = REPO / "results" / "learning_curves"
FEASIBLE = 0.6

POINT_TITLES = {
    "control": "Unconstrained",
    "rate_limited": "Rate-limited",
    "wiring_limited": "Wiring-limited",
    "both": "Rate + wiring limited",
}
# Task families as in Yang 2019, one colour each.
FAMILIES = [
    ("Go", ["fdgo", "reactgo", "delaygo"], "#1f77b4"),
    ("Anti", ["fdanti", "reactanti", "delayanti"], "#ff7f0e"),
    ("DM", ["dm1", "dm2", "contextdm1", "contextdm2", "multidm"], "#2ca02c"),
    ("Dly DM", ["delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm"], "#9467bd"),
    ("Match", ["dmsgo", "dmsnogo", "dmcgo", "dmcnogo"], "#d62728"),
]
LINESTYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1))]


def best_seeds(df):
    final = df[df.step == df.step.max()]
    return final.loc[final.groupby("point").min_task_acc.idxmax()].set_index("point")


def plot_point(df_point, point, row, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for family, tasks, color in FAMILIES:
        for task, ls in zip(tasks, LINESTYLES):
            col = f"acc_{task}"
            if col in df_point:
                ax.plot(df_point.step, df_point[col], color=color, ls=ls, lw=1.6,
                        marker="o", ms=2.5, label=rule_name.get(task, task))
    ax.axhline(FEASIBLE, color="0.6", lw=0.8, ls="--", zorder=0)
    # ax.set_xscale("log")
    ax.set_xticks([0, 25000, 50000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x >= 1000 else f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Training step (batches of 32 trials)")
    ax.set_ylabel("Accuracy")
    budgets = ("no budget" if pd.isna(row.rate_budget)
               else f"rate budget {row.rate_budget:.3g}, wiring budget {row.conn_budget:.3g}")
    ax.set_title(f"{POINT_TITLES.get(point, point)}  (seed {int(row.seed)})\n"
                 f"{budgets}; final min acc {row.min_task_acc:.2f}, mean {row.mean_acc:.2f}",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=1, fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(out_dir / f"{point}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()

    df = pd.read_csv(args.run / "runs_by_step.csv")
    best = best_seeds(df)
    out_dir = OUT_ROOT / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best[["seed", "rate_budget", "conn_budget", "min_task_acc", "mean_acc"]].to_csv(out_dir / "best_seeds.csv")

    order = [p for p in POINT_TITLES if p in best.index] + [p for p in best.index if p not in POINT_TITLES]
    for point in order:
        row = best.loc[point]
        d = df[(df.point == point) & (df.seed == row.seed)].sort_values("step")
        plot_point(d, point, row, out_dir)
        print(f"{point}: seed {int(row.seed)}, min acc {row.min_task_acc:.3f} -> {out_dir / point}.png")


if __name__ == "__main__":
    main()
