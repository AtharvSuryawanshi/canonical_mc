"""Assets for the HTML deck (presentation/html/slides.qmd), from the same figures main.tex uses.

  vector figures  PDF -> SVG (pdftocairo, TeX Live), so they stay sharp; multi-page crops -> one SVG per page
  animations      frame folders -> GIF (Pillow): the rotating front (slide 11, loops) and NSGA-III (slide 10,
                  holds the last frame)
  bitmaps         copied as they are (experimental slides)

    python presentation/html/build_assets.py          (in cmc_env, from the repo root)
    quarto render presentation/html/slides.qmd

Writes presentation/html/assets/.
"""
import shutil
import subprocess
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
PRES = HERE.parent
REPO = PRES.parent
OUT = HERE / "assets"

SVG = {  # source pdf (relative to presentation/) -> pages to convert
    "figs/fig1_intro_question.pdf": [1], "figs/yang2019_crops.pdf": [1, 2], "figs/song2016_crops.pdf": [1],
    "figs/fakhar2026_crops.pdf": [1], "figs/fig_setup.pdf": [1], "figs/driscoll2024_crops.pdf": [1],
    "figs/talk/front_top_front.pdf": [1], "figs/talk/points_summary_front.pdf": [1],
    "figs/talk/convergence_front.pdf": [1], "figs/talk/like2like_schematic_front.pdf": [1],
    "figs/talk/like2like_example_s0_front_task.pdf": [1], "figs/talk/like2like_seeds_front_task.pdf": [1],
    "figs/talk/silhouette_seeds_task_front.pdf": [1], "figs/talk/tv_s0_front.pdf": [1],
    "figs/talk/selectivity_front.pdf": [1], "figs/talk/lesion_dose_front.pdf": [1],
}
PNG = ["figs/wiring_control.png", "figs/wiring_A3.png", "figs/wiring_B0.png", "figs/wiring_C0.png",
       "../results/front/tv_period_driscoll_control_s4.png", "../results/front/tv_period_driscoll_A3_s4.png",
       "../results/front/tv_period_driscoll_B0_s4.png", "../results/front/tv_period_driscoll_C0_s4.png",
       "../results/front/lesion_period_s4_ward_contrast_grid.png"]
GIF = {  # name: (frame glob, ms per frame, ms to hold the last frame, width)
    "front_yang.gif": ("figs/talk/frames_yang/front_*.png", 100, 100, 900),
    "nsga.gif": ("figs/talk/frames_nsga/nsga_*.png", 83, 3000, 1000),
}


def svg(src, page):
    """SVG of one page; if pdftocairo fails on it (it crashes on some paper crops), a 220-dpi PNG instead."""
    stem = Path(src).stem + (f"_p{page}" if len(SVG[src]) > 1 else "")
    r = subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), str(PRES / src),
                        str(OUT / f"{stem}.svg")], capture_output=True)
    if r.returncode == 0:
        return f"{stem}.svg"
    (OUT / f"{stem}.svg").unlink(missing_ok=True)
    subprocess.run(["pdftocairo", "-png", "-r", "220", "-singlefile", "-f", str(page), "-l", str(page),
                    str(PRES / src), str(OUT / stem)], check=True)
    return f"{stem}.png"


def gif(name, pattern, ms, hold, width):
    frames = []
    for f in sorted(PRES.glob(pattern)):
        im = Image.open(f).convert("RGB")
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
        frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    durations = [ms] * (len(frames) - 1) + [hold]
    frames[0].save(OUT / name, save_all=True, append_images=frames[1:], duration=durations, loop=0,
                   optimize=True, disposal=1)
    return name, len(frames)


def main():
    OUT.mkdir(exist_ok=True)
    for src, pages in SVG.items():
        for p in pages:
            print("svg", svg(src, p))
    for src in PNG:
        shutil.copy(PRES / src, OUT / Path(src).name)
        print("png", Path(src).name)
    for name, args in GIF.items():
        n, k = gif(name, *args)
        print(f"gif {n}: {k} frames, {(OUT / n).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
