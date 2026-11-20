"""Shared style of the talk figures (presentation/figs/talk/*.pdf): drawn at the size they appear on a 16:9 Beamer
slide (text width ~5.6 in), large labels, little text. Imported by presentation/figs/talk_figs.py, like2like.py and
silhouette_seeds.py."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "presentation/figs/talk"

NAMES = {"control": "Control", "A3": "Firing constraint", "B0": "Wiring constraint", "C0": "Both constraints"}
E_COL, I_COL, UNCONN = "#1F8AD0", "#E0393E", "#b0b0b0"
W = 5.6                                                       # Beamer 16:9 text width, inches

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})


def save(fig, name, run=None):
    """presentation/figs/talk/<name>[_<run>].pdf"""
    OUT.mkdir(exist_ok=True)
    out = OUT / (f"{name}_{run}.pdf" if run else f"{name}.pdf")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("->", out)
