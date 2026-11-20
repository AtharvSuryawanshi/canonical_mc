"""Like-to-like wiring (slides 15-16) and its matched-sparsity check, cached per network.

Per network (point x seed of a 4-point run): similarity = correlation of two neurons' per-period task-variance profiles,
connected = |W|/c > 0.01; AUC = P(connected pair more similar than unconnected), null = profiles shuffled within E and
within I (fig_wiring_like2like.network, unchanged). Two modes per network:
  trained   connected as trained (|W|/c > --thresh)
  matched   only the strongest synapses, as many as the same-seed C0 network has connected (sparsity matched to C0)

    python results/scripts/like2like.py [--run yang]             # compute missing caches, then all figures
    python results/scripts/like2like.py --run yang --plot-only   # figures from the cache only (seconds)
    python results/scripts/like2like.py --run yang --recompute   # redo every cache
    options: --seeds 0-4  --points control,A3,B0,C0  --example-seed 4  --n-shuffle 500  --thresh 0.01

Cache: results/cache/<run>/like2like_<point>_s<seed>.npz (similarity S, E/I labels, connected masks of both modes,
stats) and results/cache/<run>/like2like.csv (one row per point, seed, mode, type). The per-period task variance it
needs is cached by fig_tv_period.load (results/cache/<run>/tv_period_*.npz; run 5seed: results/front/).
Figures (talk size): presentation/figs/talk/like2like_example_<run>.pdf (slide 15, one seed),
like2like_seeds_<run>.pdf (slide 16, all networks), like2like_matched_<run>.pdf (as trained vs at C0's sparsity).
"""
import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig_tv_ei_fanin import COLORS, RUNS
from fig_wiring_like2like import N_SHUFFLE, QUADS, THRESH, network
from fig_wiring_roc import roc
from talk_style import E_COL, I_COL, NAMES, REPO, UNCONN, W, save

MODES = ("trained", "matched")
REF = "C0"                                          # matched mode: sparsity of the same-seed network at this point


def cache_dir(run):
    d = REPO / "results/cache" / run
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(run, point, seed):
    return cache_dir(run) / f"like2like_{point}_s{seed}.npz"


def read_cache(run, point, seed):
    z = np.load(cache_path(run, point, seed))
    return dict(S=z["S"].astype(float), isE=z["isE"], C={m: z[f"C_{m}"] for m in MODES},
                stats=json.loads(str(z["stats"])), thresh=float(z["thresh"]), n_shuffle=int(z["n_shuffle"]))


# ---------------------------------------------------------------- compute

def compute_one(run, point, seed, keep, thresh, n_shuffle):
    """keep: fraction of synapses kept in 'matched' mode; None = this network's own 'trained' fraction (the C0 case)."""
    C, stats = {}, []
    for m, mode in enumerate(MODES):
        rng = np.random.default_rng([seed, list(NAMES).index(point), m])      # per network: recompute one alone
        k = None if mode == "trained" else keep
        res = network(point, seed, rng, keep_frac=k, run=run, thresh=thresh, n_shuffle=n_shuffle)
        if mode == "trained" and keep is None:
            keep = float(res["conn"])
        C[mode] = res["C"]
        for q in QUADS:
            stats.append(dict(point=point, seed=seed, mode=mode, type=q, n_units=int(res["n_units"]),
                              n_connected=int(res["n_conn"][q]), conn_frac=float(res["conn"]),
                              keep_frac=keep if mode == "matched" else None, auc=float(res[q]["auc"]),
                              auc_null=float(res[q]["auc_null"]), p=float(res[q]["auc_p"]),
                              strength_rho=float(res[q]["rho"])))
        S, isE = res["S"], res["isE"]
    np.savez_compressed(cache_path(run, point, seed), S=S.astype(np.float32), isE=isE, C_trained=C["trained"],
                        C_matched=C["matched"], stats=json.dumps(stats), thresh=thresh, n_shuffle=n_shuffle)


def _stale(run, point, seed, thresh, n_shuffle, keep):
    if not cache_path(run, point, seed).exists():
        return True
    c = read_cache(run, point, seed)
    if c["thresh"] != thresh or c["n_shuffle"] != n_shuffle:
        return True
    if keep is not None:                                    # matched to a C0 that has since changed
        stored = next(r["keep_frac"] for r in c["stats"] if r["mode"] == "matched")
        return not np.isclose(stored, keep)
    return False


def compute(run, points, seeds, thresh, n_shuffle, recompute):
    order = [REF] + [p for p in points if p != REF]         # C0 first: it sets the matched sparsity
    for s in seeds:
        keep = None
        for p in order:
            if p not in points and p != REF:
                continue
            if recompute or _stale(run, p, s, thresh, n_shuffle, keep):
                t = time.time()
                compute_one(run, p, s, keep, thresh, n_shuffle)
                print(f"  {p} seed {s}: computed ({time.time() - t:.0f} s)")
            if p == REF:
                keep = read_cache(run, p, s)["stats"][0]["conn_frac"]


def load_all(run, points, seeds):
    R = {(p, s): read_cache(run, p, s) for p in points for s in seeds}
    df = pd.DataFrame([r for c in R.values() for r in c["stats"]])
    df.to_csv(cache_dir(run) / "like2like.csv", index=False)
    return R, df


# ---------------------------------------------------------------- plots (from cache only)

def masks(isE):
    off = ~np.eye(len(isE), dtype=bool)
    return {q: off & np.outer(isE if q[0] == "E" else ~isE, isE if q[-1] == "E" else ~isE) for q in QUADS}


def stat(df, p, s, q, mode="trained"):
    return df[(df.point == p) & (df.seed == s) & (df.type == q) & (df["mode"] == mode)].iloc[0]


def plot_example(run, R, df, points, seed):
    """Slide 15: one network per point, histogram of similarity (connected vs unconnected) + ROC per type."""
    bins = np.linspace(-1, 1, 21)
    fig = plt.figure(figsize=(W, 3.0))
    outer = fig.add_gridspec(4, len(points), hspace=0.3, wspace=0.12, left=0.06, right=0.995, top=0.9, bottom=0.1)
    for r, q in enumerate(QUADS):
        col = E_COL if q[0] == "E" else I_COL
        for j, p in enumerate(points):
            c = R[(p, seed)]
            mk = masks(c["isE"])[q]
            s, cc = c["S"][mk], c["C"]["trained"][mk]
            st = stat(df, p, seed, q)
            inner = outer[r, j].subgridspec(1, 2, width_ratios=[2.4, 1], wspace=0.06)
            ax, ax_roc = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            hu = ax.hist(s[~cc], bins, density=True, color=UNCONN, lw=0)[0]
            hc = ax.hist(s[cc], bins, density=True, histtype="step", color=col, lw=1.1)[0]
            ax.set_xlim(-1, 1); ax.set_ylim(0, max(hu.max(), hc.max()) * 1.1)
            ax.set_yticks([]); ax.spines["left"].set_visible(False)
            ax.set_xticks([-1, 0, 1]); ax.tick_params(labelsize=6, length=2, pad=1)
            if r < 3:
                ax.set_xticklabels([])
            f, t = roc(s, cc)
            sig = st.p < 0.05
            ax_roc.fill_between(f, t, color=col, alpha=0.18, lw=0); ax_roc.plot(f, t, color=col, lw=0.9)
            ax_roc.plot([0, 1], [0, 1], color="0.3", ls=":", lw=0.5)
            ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1); ax_roc.set_xticks([]); ax_roc.set_yticks([])
            ax_roc.set_box_aspect(1)
            ax_roc.spines[["top", "right"]].set_visible(True)
            ax_roc.set_title(f"{st.auc:.2f}" + ("" if sig else " n.s."), fontsize=7, pad=1.5,
                             color="k" if sig else "0.45")
            if r == 0:
                ax.text(0.7, 1.32, NAMES[p], transform=ax.transAxes, ha="center", fontsize=9, color=COLORS[p])
            if j == 0:
                ax.set_ylabel(q, rotation=0, ha="right", va="center", fontsize=9, labelpad=4)
    h = [plt.Rectangle((0, 0), 1, 1, color=UNCONN), plt.Line2D([], [], color="k", lw=1.1)]
    fig.legend(h, ["unconnected pairs", "connected pairs (E: blue, I: red)"], loc="lower right", ncol=2,
               frameon=False, bbox_to_anchor=(0.995, -0.03), fontsize=7)
    save(fig, f"like2like_example_s{seed}", run)


def _dots(ax, df, points, seeds, q, mode, xs, dx=0.0, open_all=False, jw=0.12):
    for i, p in enumerate(points):
        a = np.array([stat(df, p, s, q, mode).auc for s in seeds])
        sig = np.array([stat(df, p, s, q, mode).p < 0.05 for s in seeds]) & (not open_all)
        jit = np.linspace(-jw, jw, len(seeds)) if len(seeds) > 1 else np.zeros(1)
        x = xs[i] + dx + jit
        ax.scatter(x[sig], a[sig], color=COLORS[p], s=10, zorder=3, lw=0)
        ax.scatter(x[~sig], a[~sig], facecolors="white", edgecolors=COLORS[p], s=10, zorder=3, lw=0.7)
        ax.plot([xs[i] + dx - jw - 0.1, xs[i] + dx + jw + 0.1], [a.mean()] * 2, color=COLORS[p], lw=1.6, zorder=2)


def plot_seeds(run, R, df, points, seeds):
    """Slide 16: all networks; mean ROC (band: seed range) and AUC per network (filled: p < 0.05)."""
    grid = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.1), gridspec_kw=dict(height_ratios=[1, 0.9], hspace=0.45,
                                                                      wspace=0.25))
    xs = np.arange(len(points))
    for k, q in enumerate(QUADS):
        ax = axes[0, k]
        for p in points:
            T = []
            for s in seeds:
                c = R[(p, s)]
                mk = masks(c["isE"])[q]
                T.append(np.interp(grid, *roc(c["S"][mk], c["C"]["trained"][mk])))
            T = np.array(T)
            ax.fill_between(grid, T.min(0), T.max(0), color=COLORS[p], alpha=0.18, lw=0)
            ax.plot(grid, T.mean(0), color=COLORS[p], lw=1.3, label=NAMES[p])
        ax.plot([0, 1], [0, 1], color="0.3", ls=":", lw=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_box_aspect(1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_title(q, fontsize=10)
        ax.set_xlabel("FPR", labelpad=-6)
        if k == 0:
            ax.set_ylabel("TPR", labelpad=-4)
        ax = axes[1, k]
        for s in seeds:
            ax.plot(xs, [stat(df, p, s, q).auc for p in points], color="0.85", lw=0.6, zorder=1)
        _dots(ax, df, points, seeds, q, "trained", xs)
        ax.axhline(0.5, color="0.3", ls=":", lw=0.6)
        ax.set_ylim(0.4, 0.85); ax.set_yticks([0.5, 0.6, 0.7, 0.8])
        ax.set_xticks([])
        if k == 0:
            ax.set_ylabel("AUC")
        else:
            ax.set_yticklabels([])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.99))
    save(fig, "like2like_seeds", run)


def plot_matched(run, df, points, seeds):
    """AUC as trained (left of each pair) vs with only the strongest synapses at C0's sparsity (right)."""
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.9), sharey=True, gridspec_kw=dict(wspace=0.12))
    xs = np.arange(len(points))
    for ax, q in zip(axes, QUADS):
        for s in seeds:
            for i, p in enumerate(points):
                ax.plot([xs[i] - 0.18, xs[i] + 0.18], [stat(df, p, s, q, "trained").auc, stat(df, p, s, q, "matched").auc],
                        color="0.85", lw=0.5, zorder=1)
        _dots(ax, df, points, seeds, q, "trained", xs, dx=-0.18, open_all=True, jw=0.05)
        _dots(ax, df, points, seeds, q, "matched", xs, dx=+0.18, jw=0.05)
        ax.axhline(0.5, color="0.3", ls=":", lw=0.6)
        ax.set_xticks([]); ax.set_title(q, fontsize=10)
    axes[0].set_ylabel("AUC")
    h = [plt.Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", ms=4),
         plt.Line2D([], [], marker="o", ls="", color="0.3", ms=4)]
    h += [plt.Line2D([], [], color=COLORS[p], lw=1.6) for p in points]
    fig.legend(h, ["as trained", f"strongest synapses, {NAMES[REF].lower()} sparsity (filled: p < 0.05)"]
               + [NAMES[p] for p in points], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98),
               fontsize=6.5)
    save(fig, "like2like_matched", run)


def summary(df, points):
    g = df.groupby(["mode", "type", "point"]).auc.mean()
    keep = df[df["mode"] == "matched"].groupby("seed").keep_frac.first()
    print(f"\nmean AUC, as trained -> at {REF}'s sparsity (kept {keep.min():.2%}-{keep.max():.2%} of synapses):")
    for q in QUADS:
        print(f"  {q}  " + "   ".join(f"{p}: {g[('trained', q, p)]:.2f} -> {g[('matched', q, p)]:.2f}" for p in points))
    n_sig = df[df["mode"] == "trained"].assign(sig=lambda d: d.p < 0.05).groupby(["type", "point"]).sig.sum()
    print("networks with p < 0.05 (as trained):")
    for q in QUADS:
        print(f"  {q}  " + "   ".join(f"{p}: {n_sig[(q, p)]}/{df.seed.nunique()}" for p in points))


def seed_list(text):
    if "-" in text:
        a, b = map(int, text.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in text.split(",")]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yang", choices=sorted(RUNS))
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--points", default=",".join(NAMES))
    ap.add_argument("--example-seed", type=int, default=4)
    ap.add_argument("--n-shuffle", type=int, default=N_SHUFFLE)
    ap.add_argument("--thresh", type=float, default=THRESH)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true")
    g.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    points, seeds = a.points.split(","), seed_list(a.seeds)
    if not a.plot_only:
        compute(a.run, points, seeds, a.thresh, a.n_shuffle, a.recompute)
    R, df = load_all(a.run, points, seeds)
    if a.example_seed in seeds:
        plot_example(a.run, R, df, points, a.example_seed)
    plot_seeds(a.run, R, df, points, seeds)
    plot_matched(a.run, df, points, seeds)
    summary(df, points)
