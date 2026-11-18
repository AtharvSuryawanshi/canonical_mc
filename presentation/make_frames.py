"""Render the rotating cog20 150k front as PNG frames for the animate package.

    python presentation/make_frames.py

Writes presentation/frames/front_000.png ... front_<N-1>.png.
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
spec = importlib.util.spec_from_file_location("front150k", REPO / "results/scripts/fig_front_gif_150k.py")
front = importlib.util.module_from_spec(spec)
spec.loader.exec_module(front)

N_FRAMES, DPI = 60, 110
OUT = HERE / "frames"


def main():
    OUT.mkdir(exist_ok=True)
    d = front.load()
    c = d[d.run == front.HIGHLIGHT_RUN]
    hl = c.loc[c.groupby("point").min_task_acc.idxmax()]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    front.draw(ax, d, hl)
    fig.subplots_adjust(left=-0.08, right=1.0, bottom=0.0, top=1.08)  # 3D axes leave wide margins
    for i in range(N_FRAMES):
        ax.view_init(front.ELEV, -60 + 360 * i / N_FRAMES)
        fig.savefig(OUT / f"front_{i:03d}.png", dpi=DPI)
    plt.close(fig)
    print(f"{N_FRAMES} frames -> {OUT}")


if __name__ == "__main__":
    main()
