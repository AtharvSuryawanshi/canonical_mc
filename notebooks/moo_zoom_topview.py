"""Top view: feasibility map of the NSGA-III budget run, with the moo_zoom points on it.

Background = min_task_acc over the (rate budget, wiring budget) plane, interpolated
between the NSGA-III networks (same as moo_pareto_analysis 2.4). Bold line = 0.6.
Big markers = zoom points at their budgets, filled by their seed-mean accuracy.

Run:  python notebooks/moo_zoom_topview.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ PARAMETERS ============================
FRONT_CSV = "runs/moo/dale_core5_nsga3_budget_p24g10/runs.csv"
ZOOM_CSV = "runs/moo_zoom/dale_core5_5pt_10seed/runs.csv"
OUT_PNG = "runs/moo_zoom/dale_core5_5pt_10seed/zoom_topview.png"

FEAS = 0.6
CMAP = "RdYlGn"
LEVELS = np.linspace(0.1, 0.9, 17)
SHOW_SEARCH_POINTS = True   # small dots / crosses for the NSGA-III networks
POINTS = None               # subset of zoom points; None = all
POINT_MARKERS = {"control": "o", "rate_limited": "s", "middle": "D",
                 "wiring_limited": "^", "knee": "P"}
MARKER_SIZE = 260
FILL_BY_ACC = True          # True: big marker filled with seed-mean acc; False: white
SHOW_ACC_TEXT = True        # "0.66 ± 0.03" under the label
LABEL_OFFSET = (0.04, 0.03) # default (dx, dy) label offset, log10 units
LABEL_OFFSETS = {"rate_limited": (0.06, -0.08)}  # per-point overrides
SHOW_UNBUDGETED = True      # note in the corner for points without budgets (control)
FIGSIZE = (7.5, 5.8)
DPI = 200
TITLE = "zoom points on the feasibility map (bold: min_task_acc = 0.6)"
# ====================================================================

f = pd.read_csv(FRONT_CSV)
z = pd.read_csv(ZOOM_CSV)
if POINTS:
    z = z[z["point"].isin(POINTS)]
g = z.groupby("point").agg(rate=("rate_budget", "first"), conn=("conn_budget", "first"),
                           acc=("min_task_acc", "mean"), sd=("min_task_acc", "std"))
g = g.reindex([p for p in POINT_MARKERS if p in g.index] +
              [p for p in g.index if p not in POINT_MARKERS])

x, y = np.log10(f.rate_budget), np.log10(f.conn_budget)
fig, ax = plt.subplots(figsize=FIGSIZE)
tc = ax.tricontourf(x, y, f.min_task_acc, levels=LEVELS, cmap=CMAP)
ax.tricontour(x, y, f.min_task_acc, levels=[FEAS], colors="k", linewidths=2.5)
if SHOW_SEARCH_POINTS:
    ok = f.min_task_acc >= FEAS
    ax.scatter(x[ok], y[ok], s=8, c="k", alpha=.3)
    ax.scatter(x[~ok], y[~ok], s=30, marker="x", c="k", alpha=.6, label="search: fails a task")

cmap, norm = plt.get_cmap(CMAP), plt.Normalize(LEVELS[0], LEVELS[-1])
for p, r in g.iterrows():
    if np.isnan(r.rate) or np.isnan(r.conn):
        if SHOW_UNBUDGETED:
            ax.text(0.02, 0.02, f"{p}: no budget, acc {r.acc:.2f} ± {r.sd:.2f}",
                    transform=ax.transAxes, fontsize=8, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.85))
        continue
    px, py = np.log10(r.rate), np.log10(r.conn)
    ax.scatter(px, py, s=MARKER_SIZE, clip_on=False, marker=POINT_MARKERS.get(p, "o"),
               c=[cmap(norm(r.acc))] if FILL_BY_ACC else "white",
               edgecolor="k", linewidth=2, zorder=5, label=p)
    txt = p + (f"\n{r.acc:.2f} ± {r.sd:.2f}" if SHOW_ACC_TEXT else "")
    ax.annotate(txt, (px, py), tuple(np.add((px, py), LABEL_OFFSETS.get(p, LABEL_OFFSET))),
                fontsize=8, weight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

fig.colorbar(tc, ax=ax, label="min_task_acc")
ax.set_xlabel("log10 rate_budget"); ax.set_ylabel("log10 conn_budget")
ax.legend(loc="upper right", fontsize=7, markerscale=0.5)
ax.set_title(TITLE)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=DPI)
print("wrote", OUT_PNG)
