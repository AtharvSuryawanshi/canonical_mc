"""Typed 2- and 3-neuron connectivity motifs on the front networks (4 points x 5 seeds), slides 23-24.

M[a, b] = 1 if a -> b (rows of W_rec are presynaptic). Counting: |W|/c > 0.01 (as like-to-like); weighted: |W|/c.
Each motif count is a sum of edge products (a matrix product). Null: for every target, its senders are shuffled
among the E neurons and, separately, among the I neurons (every neuron keeps its E and I in-degree and in-weights),
100x. Ratio = observed / mean over shuffles; p_up = fraction of shuffles >= observed.

    python motif_typed.py            (from results/scripts; ~60 s, caches results/cache/front/motif_typed.csv)
    python motif_typed.py --talk     talk figure presentation/figs/talk/motif_typed_front.pdf and the motif icons
                                     presentation/html/assets/motif_<key>.svg
"""
import argparse
import contextlib
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import ArrowStyle, Circle, FancyArrowPatch

from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import COLORS, RUNS
from talk_style import E_COL, I_COL, LABEL, REPO, save

POINTS, THRESH, NPERM = ["control", "firing", "wiring", "both"], 0.01, 100
CSV = REPO / "results/cache/front/motif_typed.csv"
ASSETS = REPO / "presentation/html/assets"

# key: (short name, nodes {name: (type, x, y)}, edges [(sender, receiver, curved)])
TRI = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (0.5, 0.82)}
MOTIFS = {
    "ee_recip": ("E↔E\nreciprocal", {"E1": ("E", 0, 0), "E2": ("E", 1, 0)}, [("E1", "E2", 1), ("E2", "E1", 1)]),
    "ii_recip": ("I↔I\nreciprocal", {"I1": ("I", 0, 0), "I2": ("I", 1, 0)}, [("I1", "I2", 1), ("I2", "I1", 1)]),
    "ei_recip": ("E↔I\nreciprocal", {"E": ("E", 0, 0), "I": ("I", 1, 0)}, [("E", "I", 1), ("I", "E", 1)]),
    "lateral": ("lateral\ninhibition", {"E1": ("E", *TRI["a"]), "E2": ("E", *TRI["b"]), "I": ("I", *TRI["c"])},
                [("E1", "I", 0), ("I", "E2", 0)]),
    "ffi": ("feedforward\ninhibition", {"E1": ("E", *TRI["a"]), "E2": ("E", *TRI["b"]), "I": ("I", *TRI["c"])},
            [("E1", "E2", 0), ("E1", "I", 0), ("I", "E2", 0)]),
    "disinh": ("disinhibition", {"I1": ("I", *TRI["a"]), "E": ("E", *TRI["b"]), "I2": ("I", *TRI["c"])},
               [("I1", "I2", 0), ("I2", "E", 0)]),
    "e_cycle": ("E cycle", {"E1": ("E", *TRI["a"]), "E2": ("E", *TRI["b"]), "E3": ("E", *TRI["c"])},
                [("E1", "E2", 0), ("E2", "E3", 0), ("E3", "E1", 0)]),
    "e_fftri": ("E feedforward\ntriangle", {"E1": ("E", *TRI["a"]), "E2": ("E", *TRI["c"]), "E3": ("E", *TRI["b"])},
                [("E1", "E2", 0), ("E2", "E3", 0), ("E1", "E3", 0)]),
    "loop_inh": ("inhibition\nonto sender", {"E1": ("E", *TRI["a"]), "E2": ("E", *TRI["b"]), "I": ("I", *TRI["c"])},
                 [("E1", "E2", 0), ("E2", "I", 0), ("I", "E1", 0)]),
}


def offsum(X):
    return X.sum() - np.trace(X)


def counts(M, ne, binary):
    EE, EI, IE, II = M[:ne, :ne], M[:ne, ne:], M[ne:, :ne], M[ne:, ne:]
    out = {"ee_recip": (EE * EE.T).sum() / 2, "ii_recip": (II * II.T).sum() / 2, "ei_recip": (EI * IE.T).sum(),
           "lateral": offsum(EI @ IE), "ffi": (EE * (EI @ IE)).sum(), "disinh": (II @ IE).sum(),
           "e_cycle": np.trace(EE @ EE @ EE) / 3, "e_fftri": ((EE @ EE) * EE).sum(),
           "loop_inh": np.trace(EE @ EI @ IE)}
    if binary:   # "pure" versions (the competing direct edge absent); printed only
        out["lateral_pure"] = offsum((EI * (1 - IE.T)) @ IE)
        out["disinh_pure"] = ((II @ IE) * (1 - IE)).sum()
    return out


def shuffle(M, ne, rng):
    """Per target (column), its E senders shuffled among E and its I senders among I; no self-connections."""
    n = M.shape[0]
    out = np.zeros_like(M)
    for lo, hi in ((0, ne), (ne, n)):
        B = M[lo:hi]
        for j in range(n):
            col = B[:, j].copy()
            idx = np.arange(hi - lo)
            if lo <= j < hi:
                idx = idx[idx != j - lo]
            col[idx] = col[idx][rng.permutation(len(idx))]
            out[lo:hi, j] = col
    return out


def compute():
    t0, rows = time.time(), []
    for p in POINTS:
        for s in range(5):
            with contextlib.redirect_stdout(io.StringIO()):
                model, _, _ = load_checkpoint(RUNS["front"] / p / f"seed_{s:02d}.pt")
            ne = model.n_e
            W = np.abs(model.effective_w_rec().detach().numpy()) / model.ei_row_scale().detach().numpy()[:, None]
            np.fill_diagonal(W, 0)
            rng = np.random.default_rng(s)
            for kind, M in (("count", (W > THRESH).astype(float)), ("weighted", W)):
                obs = counts(M, ne, kind == "count")
                null = [counts(shuffle(M, ne, rng), ne, kind == "count") for _ in range(NPERM)]
                for k, v in obs.items():
                    nk = np.array([d[k] for d in null])
                    rows.append(dict(point=p, seed=s, kind=kind, motif=k, obs=v, null=nk.mean(),
                                     ratio=v / max(nk.mean(), 1e-12), p_up=(nk >= v).mean()))
        print(f"{p} done {time.time() - t0:.0f} s", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(CSV, index=False)
    return d


def draw_motif(ax, key, r=0.17, lw=1.1, font=7, head=1.0):
    """Nodes coloured by type (E blue, I red); edges in the sender's colour, arrowhead from E, bar from I."""
    _, nodes, edges = MOTIFS[key]
    for name, (t, x, y) in nodes.items():
        ax.add_patch(Circle((x, y), r, color=E_COL if t == "E" else I_COL, zorder=3, lw=0))
        ax.text(x, y, name, ha="center", va="center", color="white", fontsize=font, zorder=4)
    for a, b, curved in edges:
        ta, xa, ya = nodes[a]
        _, xb, yb = nodes[b]
        A, B = np.array([xa, ya], float), np.array([xb, yb], float)
        rad = 0.3 * curved
        C = (A + B) / 2 + rad * np.array([B[1] - A[1], -(B[0] - A[0])])     # arc3 control point
        u0, u1 = (C - A) / np.linalg.norm(C - A), (B - C) / np.linalg.norm(B - C)
        P0, P1 = A + (r + 0.04) * u0, B - (r + 0.04) * u1                     # start / end on the circle edges
        style = ArrowStyle("-|>", head_length=4 * head, head_width=2.6 * head) if ta == "E" else \
            ArrowStyle("-[", widthB=3.2 * head, lengthB=0)
        ax.add_patch(FancyArrowPatch(P0, P1, arrowstyle=style, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                     color=E_COL if ta == "E" else I_COL, lw=lw, mutation_scale=1, zorder=2))
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.32, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")


def icons():
    for key in MOTIFS:
        fig, ax = plt.subplots(figsize=(1.0, 0.875))
        fig.subplots_adjust(0, 0, 1, 1)
        draw_motif(ax, key, lw=1.3, font=7.5, head=1.3)
        fig.savefig(ASSETS / f"motif_{key}.svg", transparent=True)
        plt.close(fig)
    print("->", ASSETS / "motif_<key>.svg")


def talk_fig(d):
    keys, floor = list(MOTIFS), 0.01
    fig = plt.figure(figsize=(9.4, 4.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.6], hspace=0.17)
    off = (np.arange(len(POINTS)) - 1.5) * 0.19
    for r, (kind, title) in enumerate((("count", "Counting synapses (|W|/c > 0.01)"), ("weighted", "Weighted (|W|/c)"))):
        ax = fig.add_subplot(gs[r])
        for i, p in enumerate(POINTS):
            q = d[(d.kind == kind) & (d.point == p)]
            for k, key in enumerate(keys):
                v = q[q.motif == key].ratio.clip(lower=floor).to_numpy()
                m = max(v.mean(), floor)
                ax.bar(k + off[i], m - 1, bottom=1, width=0.17, color=COLORS[p], lw=0,
                       label=LABEL[p] if k == 0 else None)
                ax.scatter(np.full(len(v), k + off[i]) + np.linspace(-0.04, 0.04, len(v)), v, s=5, color="black",
                           lw=0, zorder=3)
        ax.axhline(1, color="0.35", lw=0.7)
        ax.set_yscale("log")
        ax.set_ylim(floor * 0.8, 20)
        ax.set_yticks([0.01, 0.1, 1, 10])
        ax.set_yticklabels(["0.01", "0.1", "1", "10"])
        ax.minorticks_off()
        ax.set_xlim(-0.55, len(keys) - 0.45)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_ylabel("observed / shuffled", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left", pad=3)
        ax.tick_params(labelsize=8.5)
        if r == 0:
            ax.legend(ncol=4, frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.08),
                      handlelength=1.2, columnspacing=1.4)
    axi = fig.add_subplot(gs[2], sharex=ax)
    axi.axis("off")
    for k, key in enumerate(keys):
        sub = axi.inset_axes([k - 0.45, 0.34, 0.9, 0.7], transform=axi.get_xaxis_transform())
        draw_motif(sub, key, r=0.19, lw=1.0, font=6, head=0.95)
        axi.text(k, 0.3, MOTIFS[key][0], ha="center", va="top", fontsize=8.2, linespacing=1.1, transform=axi.get_xaxis_transform())
    save(fig, "motif_typed", run="front")


def report(d):
    for kind in ("count", "weighted"):
        print(f"\n{kind}: observed / shuffled, mean [min-max] over seeds; seeds above / below the null (5%)")
        for key in d[d.kind == kind].motif.unique():
            cells = []
            for p in POINTS:
                q = d[(d.kind == kind) & (d.point == p) & (d.motif == key)]
                cells.append(f"{q.ratio.mean():5.2f} [{q.ratio.min():4.2f}-{q.ratio.max():5.2f}] "
                             f"{(q.p_up < 0.05).sum()}/{(q.p_up > 0.95).sum()}")
            print(f"  {key:13s}" + "".join(f"{c:>24s}" for c in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--talk", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    d = pd.read_csv(CSV) if CSV.exists() and not a.recompute else compute()
    report(d)
    if a.talk:
        icons()
        talk_fig(d)
