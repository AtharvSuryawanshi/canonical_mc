"""Accuracy vs training step for one point of a cog20 150k run with runs_by_step.csv.

Per-task accuracy (mean over seeds), plus worst-task and mean accuracy (mean ± sd over seeds).

    python results/scripts/fig_acc_vs_steps.py [point] [run]
        point: default control
        run:   rays3 (small W_out init, 5 seeds; default) or fanin (fanin init, 1 seed, save-at run)

Writes results/front/acc_vs_steps_<point>.png (rays3) or acc_vs_steps_<point>_fanin.png.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = {"rays3": ("runs/moo_zoom/dale_all_rays3_5seed_150k", "small", ""),
        "fanin": ("runs/moo_zoom/dale_all_woutfree_fanin_4pt_1seed_150k_saveat", "fanin", "_fanin")}
NAMES = {"fdgo": "Go", "reactgo": "RT Go", "delaygo": "Dly Go", "fdanti": "Anti", "reactanti": "RT Anti",
         "delayanti": "Dly Anti", "dm1": "DM 1", "dm2": "DM 2", "contextdm1": "Ctx DM 1",
         "contextdm2": "Ctx DM 2", "multidm": "MultSen DM", "delaydm1": "Dly DM 1", "delaydm2": "Dly DM 2",
         "contextdelaydm1": "Ctx Dly DM 1", "contextdelaydm2": "Ctx Dly DM 2",
         "multidelaydm": "MultSen Dly DM", "dmsgo": "DMS", "dmsnogo": "DNMS", "dmcgo": "DMC", "dmcnogo": "DNMC"}


def main(point="control", run="rays3"):
    path, init, suffix = RUNS[run]
    d = pd.read_csv(REPO / path / "runs_by_step.csv")
    d = d[d.point == point]
    tasks = list(NAMES)
    g = d.groupby("step")
    m = g[[f"acc_{t}" for t in tasks] + ["min_task_acc", "mean_acc"]].mean()
    s = g[["min_task_acc", "mean_acc"]].std()
    s = s.fillna(0)   # one seed: no spread
    k = m.index / 1000

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(tasks):
        ax.plot(k, m[f"acc_{t}"], color=cmap(i), lw=1, alpha=0.8, marker=".", ms=3, label=NAMES[t])
    for col, c, lab in [("mean_acc", "k", "mean over tasks"), ("min_task_acc", "crimson", "worst task")]:
        ax.plot(k, m[col], color=c, lw=2.6, label=lab)
        ax.fill_between(k, m[col] - s[col], m[col] + s[col], color=c, alpha=0.15, lw=0)
    ax.axhline(0.6, color="0.5", ls=":", lw=1)
    ax.set_xlabel("training step (×1000)")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    n = d.seed.nunique()
    ax.set_title(f"{point}, cog20, W_out {init} init, {n} seed{'s' if n > 1 else ''} "
                 + ("(thin: per task, seed mean; thick: ± sd over seeds)" if n > 1 else "(thin: per task)"),
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = REPO / "results/front" / f"acc_vs_steps_{point}{suffix}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(m[["min_task_acc", "mean_acc"]].round(3).to_string())
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
