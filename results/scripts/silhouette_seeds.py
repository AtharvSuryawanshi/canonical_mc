"""Silhouette vs number of groups k, all seeds of a 4-point run, E and I separately (slide 17, all seeds), cached.

Per network: active units (total task variance > 1% of max), each neuron divided by its own max, k-means for every k
(E and I: 2-20; I was 2-10 before 2026-11-26) and the chosen k (best local silhouette maximum with k > 2): fig_tv_ei_fanin.cluster_pop, unchanged.
Profile (--basis):
  task     per-task variance stored in the checkpoint (one row per task), as slide 17 and fig_tv_ei_fanin
  period   per-period task variance (one row per task period; fig_tv_period.load, cache shared with like2like.py)

    python results/scripts/silhouette_seeds.py [--run yang] [--basis task|period]
    python results/scripts/silhouette_seeds.py --run yang --plot-only     # figure from the cache only
    python results/scripts/silhouette_seeds.py --run yang --recompute
    options: --seeds 0-4  --points control,A3,B0,C0

Cache: results/cache/<run>/silhouette_<basis>.csv (point, seed, pop, k, silhouette, chosen, n_units); networks
already in it are not recomputed. Figure: presentation/figs/talk/silhouette_seeds_<basis>_<run>.pdf (per point: mean
over seeds with min-max band, each seed's chosen k as a dot).
"""
import argparse
import contextlib
import io
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import COLORS, RUNS, cluster_pop
from fig_tv_period import load as load_period
from like2like import cache_dir, seed_list
from talk_style import E_COL, I_COL, LABEL, REPO, run_points, save

K_MAX = {"E": 20, "I": 20}                  # I was 10 (as fig_tv_ei_fanin) until 2026-11-26


def profile(run, point, seed, basis):
    """(norm profile rows x N, total per-task variance per unit, n_e)."""
    with contextlib.redirect_stdout(io.StringIO()):
        if basis == "task":
            model, _, ck = load_checkpoint(RUNS[run] / point / f"seed_{seed:02d}.pt")
            tv = np.asarray(ck["task_variance"], float)
            total = tv.sum(0)
        else:
            model, _, tv_t, tv, _ = load_period(point, seed, run)
            total = tv_t.sum(0)                         # same active units as the per-task rule
    return tv, total, model.n_e, model.n_rnn


def compute_one(run, point, seed, basis):
    tv, total, n_e, n = profile(run, point, seed, basis)
    active = total > 1e-2 * total.max()
    rows = []
    for pop, sel in (("E", np.arange(n) < n_e), ("I", np.arange(n) >= n_e)):
        units = np.flatnonzero(active & sel)
        X = tv[:, units]
        norm = X / np.maximum(X.max(0, keepdims=True), 1e-30)
        _, K, sil = cluster_pop(norm, K_MAX[pop])
        rows += [dict(point=point, seed=seed, pop=pop, k=k, silhouette=v, chosen=k == K, n_units=len(units))
                 for k, v in sil.items()]
    return rows


def compute(run, points, seeds, basis, recompute):
    path = cache_dir(run) / f"silhouette_{basis}.csv"
    df = pd.read_csv(path) if path.exists() and not recompute else pd.DataFrame(columns=["point", "seed"])
    done = set(zip(df.point, df.seed))
    new = []
    for p in points:
        for s in seeds:
            if (p, s) not in done:
                t = time.time()
                new += compute_one(run, p, s, basis)
                print(f"  {p} seed {s}: computed ({time.time() - t:.0f} s)")
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True) if len(df) else pd.DataFrame(new)
        df.to_csv(path, index=False)
    return df


def plot(run, df, points, seeds, basis):
    df = df[df.point.isin(points) & df.seed.isin(seeds)]
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, pop in zip(axes, "EI"):
        d = df[df["pop"] == pop]
        for i, p in enumerate(points):
            t = d[d.point == p].pivot(index="k", columns="seed", values="silhouette")
            ax.fill_between(t.index, t.min(1), t.max(1), color=COLORS[p], alpha=0.15, lw=0)
            ax.plot(t.index, t.mean(1), color=COLORS[p], lw=1.1, label=LABEL[p])
            c = d[(d.point == p) & d.chosen.astype(bool)].sort_values("seed")
            jit = (i - (len(points) - 1) / 2) * 0.12
            ax.plot(c.k + jit, c.silhouette, "o", color=COLORS[p], ms=4, mec="k", mew=0.4, zorder=3)
        ax.set_title(f"{pop} neurons", color=E_COL if pop == "E" else I_COL, fontsize=11)
        ax.set_xticks(range(2, K_MAX[pop] + 1, 4))
        ax.tick_params(labelsize=9)
        ax.set_xlabel("number of groups k", fontsize=10)
    axes[0].set_ylabel("silhouette", fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.97), fontsize=8, handlelength=1.2, columnspacing=1)
    save(fig, f"silhouette_seeds_{basis}", run)

    print(f"chosen k ({basis} variance), seeds {seeds}:")
    ch = df[df.chosen.astype(bool)]
    for pop in "EI":
        print(f"  {pop}: " + "   ".join(
            f"{p} " + "/".join(str(int(v)) for v in ch[(ch["pop"] == pop) & (ch.point == p)].sort_values("seed").k)
            for p in points))


def pick_k(sil):
    """cluster_pop's rule on a silhouette curve {k: s}: the highest local peak with k > 2 (higher than both
    neighbours, so the largest k can't be one); no peak -> the overall maximum. Returns (k, reason)."""
    ks = sorted(sil)
    peaks = [k for k in ks[1:-1] if k > 2 and sil[k] > sil[k - 1] and sil[k] > sil[k + 1]]
    if peaks:
        k = max(peaks, key=sil.get)
        others = ", ".join(str(p) for p in sorted(peaks) if p != k)
        return k, "highest peak of the mean" + (f"\n(other peaks: {others})" if others else "\n(its only peak)")
    k = max(sil, key=sil.get)
    return k, "no peak with k > 2:\noverall maximum"


def plot_panels(run, df, points, seeds, basis):
    """One figure: a row per point, a column per population (E, I). Per panel: each seed's silhouette curve (light),
    the mean (dark), each seed's chosen k (dot), and the k the same rule picks on the mean curve (dashed, with why)."""
    df = df[df.point.isin(points) & df.seed.isin(seeds)]
    fig, axes = plt.subplots(len(points), 2, figsize=(10, 2.9 * len(points)), sharey="row",
                             gridspec_kw=dict(width_ratios=[1, 1], hspace=0.95, wspace=0.08))
    for r, p in enumerate(points):
        for c, pop in enumerate("EI"):
            ax = axes[r, c]
            d = df[(df["pop"] == pop) & (df.point == p)]
            t = d.pivot(index="k", columns="seed", values="silhouette")
            col = COLORS[p]
            ax.axvspan(1.5, 2.5, color="0.92", lw=0, zorder=0)
            for s_ in t.columns:
                ax.plot(t.index, t[s_], color=col, alpha=0.3, lw=1)
                ks = d[(d.seed == s_) & d.chosen.astype(bool)].k
                ax.plot(ks, t.loc[ks, s_], "o", color=col, alpha=0.6, ms=5, mec="none")
            mean = t.mean(1)
            ax.plot(mean.index, mean, color=col, lw=2.5)
            k, why = pick_k(mean.to_dict())
            ax.axvline(k, color="k", lw=1.2, ls="--")
            seed_k = d[d.chosen.astype(bool)].sort_values("seed").k.astype(int).tolist()
            ax.set_title(f"{pop} neurons: k = {k}, {why}\nseeds chose "
                         f"{', '.join(map(str, seed_k))}", fontsize=8.5, loc="left",
                         color=E_COL if pop == "E" else I_COL)
            ax.set_xticks(range(2, K_MAX[pop] + 1, 2))
            ax.set_xlim(1.5, K_MAX[pop] + 0.5)
            if r == len(points) - 1:
                ax.set_xlabel("number of groups k")
            if c == 0:
                ax.set_ylabel(f"{LABEL[p]}\nsilhouette", color=col, fontsize=10)
    axes[0, 0].text(2, 0.97, "k = 2\nexcluded", transform=axes[0, 0].get_xaxis_transform(), ha="center",
                    va="top", fontsize=7, color="0.4")
    h = [plt.Line2D([], [], color="0.5", alpha=0.4, lw=1), plt.Line2D([], [], color="0.3", lw=2.5),
         plt.Line2D([], [], marker="o", ls="", color="0.5", alpha=0.6), plt.Line2D([], [], color="k", ls="--")]
    fig.legend(h, ["one seed", f"mean of {len(seeds)} seeds", "k chosen by that seed", "k chosen on the mean"],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Silhouette vs number of groups k, {basis} variance ({run} run). Rule: the highest peak with "
                 f"k > 2\n(a peak = higher than both neighbours; no peak: the overall maximum)", fontsize=10.5)
    out = REPO / "results/front" / f"silhouette_k_{basis}_{run}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="yang", choices=sorted(RUNS))
    ap.add_argument("--basis", default="task", choices=["task", "period"])
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--points", default=None, help="default: the run's points (talk_style.RUN_POINTS)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plot-only", action="store_true")
    g.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    points, seeds = (a.points.split(",") if a.points else run_points(a.run)), seed_list(a.seeds)
    path = cache_dir(a.run) / f"silhouette_{a.basis}.csv"
    df = pd.read_csv(path) if a.plot_only else compute(a.run, points, seeds, a.basis, a.recompute)
    plot(a.run, df, points, seeds, a.basis)
    plot_panels(a.run, df, points, seeds, a.basis)
