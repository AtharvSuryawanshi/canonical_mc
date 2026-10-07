"""Rotating 3D GIF of the NSGA-III budget front (task error x metabolic x wiring).

Run:  python notebooks/moo_front_gif.py
Tweak everything in the PARAMETERS block.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
from scipy.spatial import Delaunay

# ============================ PARAMETERS ============================
RUN_CSV = "runs/moo/dale_core5_nsga3_budget_p24g10/runs.csv"
OUT_GIF = "runs/moo/dale_core5_nsga3_budget_p24g10/front_rotate.gif"

# what to show
SHOW_INFEASIBLE = True     # grey x's for min_task_acc < 0.6
SHOW_DOMINATED = True      # faint feasible-but-dominated points
SHOW_SURFACE = True        # triangulated surface through the front
SHOW_DROPLINES = True      # vertical lines from front points to the floor
FRONT_INCLUDE_INFEASIBLE = True  # front over ALL evals, so < 0.6 networks get coloured too
SHOW_FEAS_PLANE = True     # horizontal plane at the feasibility threshold
FEAS_PLANE_Z = 0.6
FEAS_PLANE_COLOR = "#ffd8a8"   # very light orange
FEAS_PLANE_ALPHA = 0.35
COLOR_BY = "min_task_acc"    # "generation" | "min_task_acc" | None (front = one colour)
AXES = {
    "x": ("log10_metabolic_cost", "log$_{10}$ metabolic cost"),
    "y": ("log10_wiring_cost", "log$_{10}$ wiring cost"),
    "z": ("min_task_acc", "min task accuracy"),
}
ACC_LIM = (0, 1)            # task-accuracy axis limits; None = auto

# camera / animation
ELEV_START, ELEV_END = 22, 22   # set different to add a slow tilt
AZIM_START, AZIM_SWEEP = -135, 360   # degrees; sweep 360 = full turn, e.g. 90 = rock
ROCK = False                    # True: sweep back and forth instead of looping
N_FRAMES = 120
FPS = 15
DPI = 120
FIGSIZE = (7, 6)

# style
FRONT_COLOR = "#d1495b"
CMAP = "RdYlGn"
POINT_SIZE = 45
SURFACE_ALPHA = 0.25
SURFACE_COLOR = "#d1495b"
TITLE = "NSGA-III budget front (core5, p24 g10)"
BACKGROUND = "white"
# ====================================================================


def pareto_mask(F):
    """F: objectives to MINIMISE, shape (n, k)."""
    n = len(F)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = np.all(F <= F[i], 1) & np.any(F < F[i], 1)
        keep[i] = not dom.any()
    return keep


df = pd.read_csv(RUN_CSV)
feas = df["min_task_acc"] >= 0.6
F = np.c_[1 - df["min_task_acc"], df["log10_metabolic_cost"], df["log10_wiring_cost"]]
front = np.zeros(len(df), bool)
if FRONT_INCLUDE_INFEASIBLE:
    front[:] = pareto_mask(F)
else:
    front[feas.values] = pareto_mask(F[feas.values])
print(f"{len(df)} evals, {feas.sum()} feasible, {front.sum()} on front "
      f"({(front & ~feas.values).sum()} of them < 0.6)")

cx, cy, cz = (AXES[k][0] for k in "xyz")
X, Y, Z = df[cx].values, df[cy].values, df[cz].values

fig = plt.figure(figsize=FIGSIZE, facecolor=BACKGROUND)
ax = fig.add_subplot(projection="3d", facecolor=BACKGROUND)

if SHOW_FEAS_PLANE:
    xx, yy = np.meshgrid(np.linspace(X.min(), X.max(), 2), np.linspace(Y.min(), Y.max(), 2))
    ax.plot_surface(xx, yy, np.full_like(xx, FEAS_PLANE_Z), color=FEAS_PLANE_COLOR,
                    alpha=FEAS_PLANE_ALPHA, linewidth=0, shade=False)
if SHOW_INFEASIBLE:
    m = ~feas.values & ~front
    ax.scatter(X[m], Y[m], Z[m], marker="x", c="0.6", s=POINT_SIZE * 0.5,
               alpha=0.5, label="infeasible (min acc < 0.6)")
if SHOW_DOMINATED:
    m = feas.values & ~front
    ax.scatter(X[m], Y[m], Z[m], c="0.45", s=POINT_SIZE * 0.4, alpha=0.35,
               label="feasible, dominated")

m = front
if COLOR_BY:
    sc = ax.scatter(X[m], Y[m], Z[m], c=df.loc[m, COLOR_BY], cmap=CMAP,
                    s=POINT_SIZE, edgecolor="k", linewidth=0.4, depthshade=False,
                    label="Pareto front")
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.1, label=COLOR_BY)
else:
    ax.scatter(X[m], Y[m], Z[m], c=FRONT_COLOR, s=POINT_SIZE, edgecolor="k",
               linewidth=0.4, depthshade=False, label="Pareto front")

if SHOW_SURFACE and m.sum() >= 3:
    tri = Delaunay(np.c_[X[m], Y[m]])
    ax.plot_trisurf(X[m], Y[m], Z[m], triangles=tri.simplices,
                    color=SURFACE_COLOR, alpha=SURFACE_ALPHA, linewidth=0)

if SHOW_DROPLINES:
    z0 = ACC_LIM[0] if ACC_LIM else Z.min()
    for x, y, z in zip(X[m], Y[m], Z[m]):
        ax.plot([x, x], [y, y], [z0, z], c="0.5", lw=0.5, alpha=0.5)

ax.set_xlabel(AXES["x"][1]); ax.set_ylabel(AXES["y"][1]); ax.set_zlabel(AXES["z"][1])
if ACC_LIM:
    ax.set_zlim(*ACC_LIM)
ax.set_title(TITLE)
ax.legend(loc="upper left", fontsize=8, frameon=False)
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
