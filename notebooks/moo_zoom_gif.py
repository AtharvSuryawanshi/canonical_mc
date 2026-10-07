"""Rotating 3D GIF of the moo_zoom seeds placed on the NSGA-III budget front.

Each zoom point (control / rate_limited / middle / wiring_limited / knee) is a
cloud of seeds, drawn over the faint full front for context.

Run:  python notebooks/moo_zoom_gif.py
Tweak everything in the PARAMETERS block.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial import Delaunay

# ============================ PARAMETERS ============================
ZOOM_CSV = "runs/moo_zoom/dale_core5_5pt_10seed/runs.csv"
FRONT_CSV = "runs/moo/dale_core5_nsga3_budget_p24g10/runs.csv"   # context; None to skip
OUT_GIF = "runs/moo_zoom/dale_core5_5pt_10seed/zoom_rotate.gif"

# what to show
SHOW_FRONT_POINTS = True   # small Pareto-front points, coloured by accuracy (CMAP)
FRONT_INCLUDE_INFEASIBLE = True  # front over ALL evals, so < 0.6 networks show too
SHOW_FEAS_PLANE = True     # horizontal plane at the feasibility threshold
FEAS_PLANE_Z = 0.6
FEAS_PLANE_COLOR = "#ffd8a8"   # very light orange
FEAS_PLANE_ALPHA = 0.35
SHOW_FRONT_SURFACE = True  # translucent surface through the front
SHOW_BUDGET = True         # star at each point's budget ceilings (x, y), at the seed-mean accuracy
SHOW_MEAN = True           # big marker at each point's seed mean
SHOW_LABELS = True         # point name next to the mean
SHOW_DROPLINES = True      # vertical lines from seed means to the floor
COLOR_BY = "point"         # "point" | "min_task_acc" (one colour per zoom point)
POINTS = None              # e.g. ["control", "rate_limited", "wiring_limited"]; None = all
AXES = {
    "x": ("log10_metabolic_cost", "log$_{10}$ metabolic cost"),
    "y": ("log10_wiring_cost", "log$_{10}$ wiring cost"),
    "z": ("min_task_acc", "min task accuracy"),
}
ACC_LIM = (0, 1)            # task-accuracy axis limits; None = auto

# camera / animation
ELEV_START, ELEV_END = 22, 22
AZIM_START, AZIM_SWEEP = -135, 360
ROCK = False
N_FRAMES = 120
FPS = 15
DPI = 120
FIGSIZE = (7, 6)

# style
CMAP = "RdYlGn"             # background front colours
ACC_RANGE = (0.1, 0.9)      # colour scale for accuracy
POINT_COLORS = {"control": "#4c4c4c", "rate_limited": "#d1495b", "middle": "#edae49",
                "wiring_limited": "#00798c", "knee": "#6a4c93"}
POINT_MARKERS = {"control": "o", "rate_limited": "s", "middle": "D",
                 "wiring_limited": "^", "knee": "P"}
SEED_SIZE = 40
MEAN_SIZE = 260
ZOOM_EDGE = 1.5             # outline width of zoom markers
BG_SIZE = 14               # size of background front points
FRONT_ALPHA = 0.55
SURFACE_ALPHA = 0.12
SURFACE_COLOR = "0.5"
TITLE = "moo_zoom: 10 seeds at 5 budget points"
BACKGROUND = "white"
# ====================================================================


def pareto_mask(F):
    keep = np.ones(len(F), bool)
    for i in range(len(F)):
        keep[i] = not (np.all(F <= F[i], 1) & np.any(F < F[i], 1)).any()
    return keep


def add_logs(d):
    d["log10_metabolic_cost"] = np.log10(d["metabolic_cost"])
    d["log10_wiring_cost"] = np.log10(d["wiring_cost"])
    return d


z = add_logs(pd.read_csv(ZOOM_CSV))
if POINTS:
    z = z[z["point"].isin(POINTS)]
order = [p for p in POINT_COLORS if p in set(z["point"])] + \
        sorted(set(z["point"]) - set(POINT_COLORS))
cx, cy, cz = (AXES[k][0] for k in "xyz")

fig = plt.figure(figsize=FIGSIZE, facecolor=BACKGROUND)
ax = fig.add_subplot(projection="3d", facecolor=BACKGROUND)

if FRONT_CSV:
    f = pd.read_csv(FRONT_CSV)
    feas = (f["min_task_acc"] >= 0.6).values
    F = np.c_[1 - f["min_task_acc"], f["log10_metabolic_cost"], f["log10_wiring_cost"]]
    m = np.zeros(len(f), bool)
    if FRONT_INCLUDE_INFEASIBLE:
        m[:] = pareto_mask(F)
    else:
        m[feas] = pareto_mask(F[feas])
    fx, fy, fz = f.loc[m, cx].values, f.loc[m, cy].values, f.loc[m, cz].values
    if SHOW_FRONT_POINTS:
        sc_bg = ax.scatter(fx, fy, fz, c=fz, cmap=CMAP, vmin=ACC_RANGE[0], vmax=ACC_RANGE[1],
                           s=BG_SIZE, alpha=FRONT_ALPHA, edgecolor="none",
                           depthshade=False, label="NSGA-III front")
        fig.colorbar(sc_bg, ax=ax, shrink=0.55, pad=0.1, label="min_task_acc")
    if SHOW_FEAS_PLANE:
        xx, yy = np.meshgrid(np.linspace(fx.min(), fx.max(), 2), np.linspace(fy.min(), fy.max(), 2))
        ax.plot_surface(xx, yy, np.full_like(xx, FEAS_PLANE_Z), color=FEAS_PLANE_COLOR,
                        alpha=FEAS_PLANE_ALPHA, linewidth=0, shade=False)
    if SHOW_FRONT_SURFACE and len(fx) >= 3:
        ax.plot_trisurf(fx, fy, fz, triangles=Delaunay(np.c_[fx, fy]).simplices,
                        color=SURFACE_COLOR, alpha=SURFACE_ALPHA, linewidth=0)

vmin, vmax = ACC_RANGE
sc = None
z0 = ACC_LIM[0] if ACC_LIM else z[cz].min() if not FRONT_CSV else min(z[cz].min(), fz.min())
for p in order:
    d = z[z["point"] == p]
    col = POINT_COLORS.get(p, "k")
    mk = POINT_MARKERS.get(p, "o")
    if COLOR_BY == "point":
        ax.scatter(d[cx], d[cy], d[cz], c=col, marker=mk, s=SEED_SIZE,
                   edgecolor="k", linewidth=0.6, depthshade=False, label=p)
    else:
        sc = ax.scatter(d[cx], d[cy], d[cz], c=d[COLOR_BY], cmap=CMAP, vmin=vmin,
                        vmax=vmax, marker=mk, s=SEED_SIZE, edgecolor="k",
                        linewidth=0.3, depthshade=False, label=p)
    mx, my, mz = d[cx].mean(), d[cy].mean(), d[cz].mean()
    if SHOW_MEAN:
        ax.scatter([mx], [my], [mz], c=col, marker=mk, s=MEAN_SIZE, edgecolor="k",
                   linewidth=ZOOM_EDGE * 1.5, depthshade=False, zorder=10)
    if SHOW_BUDGET and cx == "log10_metabolic_cost" and cy == "log10_wiring_cost":
        bx, by = np.log10(d["rate_budget"].iloc[0]), np.log10(d["conn_budget"].iloc[0])
        ax.scatter([bx], [by], [mz], c=col, marker="*", s=MEAN_SIZE, edgecolor="k",
                   linewidth=0.6, depthshade=False)
    if SHOW_DROPLINES:
        ax.plot([mx, mx], [my, my], [z0, mz], c=col, lw=1, alpha=0.7)
    if SHOW_LABELS:
        ax.text(mx, my, mz + 0.02, p, fontsize=8, color=col, weight="bold")

if sc is not None and not (FRONT_CSV and SHOW_FRONT_POINTS):
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1, label=COLOR_BY)

ax.set_xlabel(AXES["x"][1]); ax.set_ylabel(AXES["y"][1]); ax.set_zlabel(AXES["z"][1])
if ACC_LIM:
    ax.set_zlim(*ACC_LIM)
ax.set_title(TITLE)
ax.legend(loc="upper left", fontsize=7, frameon=False)
fig.tight_layout()


def update(i):
    t = i / (N_FRAMES - 1)
    if ROCK:
        t = 0.5 - 0.5 * np.cos(2 * np.pi * t)
    ax.view_init(elev=ELEV_START + (ELEV_END - ELEV_START) * t,
                 azim=AZIM_START + AZIM_SWEEP * t)
    return []


anim = FuncAnimation(fig, update, frames=N_FRAMES)
anim.save(OUT_GIF, writer=PillowWriter(fps=FPS), dpi=DPI)
print("wrote", OUT_GIF)
