"""Split the stacked 4-point figures from results/scripts/fig_tv_ei_fanin.py into one image per point.

    python presentation/figs/crop_tv_rows.py [tag]     # default tag: 5seed_s4_150k (the example network, seed 4)

Writes figs/tv_ei_<point>.png (task variance, E | I) and figs/wiring_<point>.png (task similarity | wiring).
Row boxes are in pixels of the fixed figure sizes (1440 x 1920 and 1950 x 3720); re-check them if
fig_tv_ei_fanin.py changes its layout.
"""
import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
POINTS = ["control", "A3", "B0", "C0"]
TV_ROWS = {"control": (70, 460), "A3": (543, 930), "B0": (1015, 1398), "C0": (1488, 1876)}
WIRING_TOP = 110  # suptitle band (text ends at y ~97); the four rows below it are equal

tag = sys.argv[1] if len(sys.argv) > 1 else "5seed_s4_150k"
tv = Image.open(REPO / f"results/front/tv_ei_fanin_{tag}.png")
for p, (y0, y1) in TV_ROWS.items():
    tv.crop((0, y0, tv.width, y1)).save(OUT / f"tv_ei_{p}.png")
w = Image.open(REPO / f"results/front/tv_ei_fanin_{tag}_wiring.png")
h = (w.height - WIRING_TOP) / 4
for i, p in enumerate(POINTS):
    w.crop((0, int(WIRING_TOP + i * h), w.width, int(WIRING_TOP + (i + 1) * h))).save(OUT / f"wiring_{p}.png")
print(f"{tag}: 4 task-variance crops {tv.size}, 4 wiring crops {w.size}")
