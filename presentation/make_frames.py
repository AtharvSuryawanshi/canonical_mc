"""Render the rotating cog20 150k fanin front (W_in and W_out trained, nothing frozen) as PNG frames
for the animate package.

    python presentation/make_frames.py

Data: fig_front_top_fanin150k.load() (29-point grid x 3 seeds + the 4-point zoom runs); drawing:
fig_front_gif_150k.draw. Writes presentation/frames/front_000.png ... front_<N-1>.png.
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _module(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "results/scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


front = _module("front150k", "fig_front_gif_150k.py")
fanin = _module("front150k_fanin", "fig_front_top_fanin150k.py")

N_FRAMES, DPI = 60, 110
OUT = HERE / "frames"


def main():
    OUT.mkdir(exist_ok=True)
    d = fanin.load()
    z = d[d.point.isin(list(front.HIGHLIGHT_NAMES))]
    hl = z.loc[z.groupby("point").min_task_acc.idxmax()]   # best seed per labelled point
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    front.draw(ax, d, hl)
    ax.set_xlim(d.x.min() - 0.1, d.x.max() + 0.1)   # autoscale leaves the cloud in a corner
    ax.set_ylim(d.y.min() - 0.1, d.y.max() + 0.1)
    fig.subplots_adjust(left=-0.08, right=1.0, bottom=0.0, top=1.08)  # 3D axes leave wide margins
    for i in range(N_FRAMES):
        ax.view_init(front.ELEV, -60 + 360 * i / N_FRAMES)
        fig.savefig(OUT / f"front_{i:03d}.png", dpi=DPI)
    plt.close(fig)
    print(f"{len(d)} networks, {int(d.feasible.sum())} feasible, {int(d.pareto.sum())} Pareto")
    print(f"{N_FRAMES} frames -> {OUT}")


if __name__ == "__main__":
    main()
