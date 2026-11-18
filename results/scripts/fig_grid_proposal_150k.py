"""Proposed budget grid for a cog20 150k fanin front, drawn on the 150k top view.

Rate budget levels (log10): free, -1.5, -2.0, -2.5, -3.0, -3.5.
Wiring budget levels (log10): free, -1.75, -2.0, -2.25, -2.5, -2.75.
Points far past the feasibility cliff (log rate + log wiring < -5.25, with
"free" counted as the unconstrained cost: rate -0.3, wiring -1.5) are dropped.

Proposed points are placed at the cost they are expected to reach: wiring =
budget (exact projection); firing = budget + 0.24 in log10 (the 150k runs reach
~1.7x their rate budget); "free" = where the existing networks with that axis
free landed.

    python results/scripts/fig_grid_proposal_150k.py

Writes results/front/cog20_150k_grid_proposal.png and grid_proposal_150k.csv.
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("front150k", HERE / "fig_front_gif_150k.py")
front = importlib.util.module_from_spec(spec)
spec.loader.exec_module(front)

RATE_LEVELS = [None, -1.5, -2.0, -2.5, -3.0, -3.5]     # None = free
WIRE_LEVELS = [None, -1.75, -2.0, -2.25, -2.5, -2.75]
FREE_RATE, FREE_WIRE = -0.3, -1.5                       # unconstrained costs, for the cut
CUT = -5.25
RATE_OFFSET = 0.24                                      # reached log firing - log budget


def proposal():
    rows = []
    for lr in RATE_LEVELS:
        for lw in WIRE_LEVELS:
            s = (FREE_RATE if lr is None else lr) + (FREE_WIRE if lw is None else lw)
            if s < CUT:
                continue
            # expected reached costs (for plotting only)
            x = -0.3 if lr is None else lr + RATE_OFFSET
            if lw is not None:
                y = lw
            elif lr is None:
                y = -1.52
            else:   # rate-only: wiring drifts up as the rate budget tightens (A0..A4: -1.45..-1.29)
                y = np.interp(lr, [-3.35, -1.76], [-1.29, -1.45])
            name = f"r{'free' if lr is None else lr}_w{'free' if lw is None else lw}"
            rows.append(dict(name=name,
                             rate_budget=None if lr is None else round(10 ** lr, 6),
                             conn_budget=None if lw is None else round(10 ** lw, 6),
                             x=x, y=y))
    return pd.DataFrame(rows)


def main():
    d = front.load()
    c = d[d.run == front.HIGHLIGHT_RUN]
    hl = c.loc[c.groupby("point").min_task_acc.idxmax()]
    p = proposal()

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    kw = dict(cmap="RdYlGn", vmin=0.0, vmax=1)
    inf, feas = d[~d.feasible], d[d.feasible]
    ax.scatter(inf.x, inf.y, c=inf.z, marker="X", s=26, edgecolors="0.3", linewidths=0.4, alpha=0.6, **kw)
    sc = ax.scatter(feas.x, feas.y, c=feas.z, s=20, edgecolors="none", alpha=0.6, **kw)
    for _, r in hl.iterrows():
        ax.annotate(front.HIGHLIGHT_NAMES.get(r.point, r.point), (r.x, r.y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=7, color="#1f3b73")
    ax.scatter(p.x, p.y, marker="s", s=70, facecolors="none", edgecolors="k", linewidths=1.2, zorder=5)
    fig.colorbar(sc, ax=ax).set_label("Min Task Accuracy (existing networks)")
    ax.scatter([], [], marker="X", c="0.6", label="existing, infeasible")
    ax.scatter([], [], c="0.6", label="existing, feasible")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors="k", label=f"proposed ({len(p)} points)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_xlabel("log(Firing cost)")
    ax.set_ylabel("log(Connectivity cost)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = front.OUT_DIR / "cog20_150k_grid_proposal.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    p.drop(columns=["x", "y"]).to_csv(front.OUT_DIR / "grid_proposal_150k.csv", index=False)
    print(f"{len(p)} proposed points -> {out}")


if __name__ == "__main__":
    main()
