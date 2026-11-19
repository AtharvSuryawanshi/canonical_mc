"""Cluster lesions on per-period task-variance groups (Driscoll et al. 2024, Fig. 5), E and I separately.

One network (default: 5-seed run, C0, seed 4).
1. Per-period task variance (59 task periods, cached by fig_tv_period.load), same active units as the per-task
   analysis, each unit divided by its own max.
2. k-means (n_init 10, random_state 0) on E and on I separately, with a fixed k grounded in the task structure:
   E k = 14 = the (family x period type) combinations that exist (5 families x stimulus/delay/response, DM has no
   delay), I k = 5 = the task families. The silhouette-best k (Driscoll's rule, k >= 3) is reported for reference.
3. Lesion: one cluster at a time, units' rates set to 0 at every time step of the trial (Driscoll: "setting the
   output of all units within the cluster to be zero throughout a given trial"). No size control (Driscoll has none).
4. Damage: fraction performance change per task, (acc_lesioned - acc_full) / acc_full, on the network's 10 eval
   seeds x 64 noise-free trials per task; then the mean over each family, and Anti vs non-Anti.

    python results/scripts/fig_lesion_period.py [--point C0] [--seed 4] [--kE 14] [--kI 5] [--k-range LO HI]
        [--method kmeans|ward] [--joint] [--driscoll]   (--driscoll = Ward, E+I together, k best in 3..39)

5. Driscoll Fig. 5 comparison for every cluster: their task modules (Anti, Delay, Modality1/2, Cat./Contin. Mem.)
   and their reference sets (complement, or the matched alternative), see MODULES.

Writes results/front/lesion_period_<point>_s<seed>.png and .csv, _modules.png (cluster x module panels) and
_modules_contrast.png/.csv (mean module change - mean reference change).
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cmc.task import generate_trials, rule_name
from cmc.train_cog import (batch_accuracy, eval_config_from, load_checkpoint, response_mask, ring_prefs,
                           trial_to_tensors)
from fig_tv_ei_fanin import RUNS
from fig_tv_period import BATCH, PERIOD_NAME, load

REPO = Path(__file__).resolve().parents[2]
FAMILY = {"fdgo": "Go", "reactgo": "Go", "delaygo": "Go", "fdanti": "Anti", "reactanti": "Anti",
          "delayanti": "Anti", "dm1": "DM", "dm2": "DM", "contextdm1": "DM", "contextdm2": "DM", "multidm": "DM",
          "delaydm1": "Dly DM", "delaydm2": "Dly DM", "contextdelaydm1": "Dly DM", "contextdelaydm2": "Dly DM",
          "multidelaydm": "Dly DM", "dmsgo": "Matching", "dmsnogo": "Matching", "dmcgo": "Matching",
          "dmcnogo": "Matching"}
FAMILIES = ["Go", "Anti", "DM", "Dly DM", "Matching"]
FAM_COLOR = dict(zip(FAMILIES, ["#E69F00", "#D55E00", "#0072B2", "#56B4E9", "#009E73"]))   # Okabe-Ito


def fit(X, k, method, Y=None):
    """Cluster rows of X into k groups: k-means, or Ward (scipy linkage on Euclidean distances, tree cut into
    k clusters with fcluster maxclust, as in Driscoll's make_dendro / clustering code)."""
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    lab = sch.fcluster(Y, k, criterion="maxclust")
    return np.unique(lab, return_inverse=True)[1]                  # 0..k-1 (fewer if the cut has ties)


def cluster_fixed(norm_units, k, k_ref_max, k_range=None, method="kmeans"):
    """Fixed k, or (k_range = (lo, hi)) the silhouette-best k within lo..hi; also the silhouette-best k over
    3..k_ref_max (Driscoll's rule) for reference."""
    X = norm_units.T
    Y = sch.linkage(X, method="ward") if method == "ward" else None
    sil = {kk: silhouette_score(X, fit(X, kk, method, Y))
           for kk in range(3, min(max(k_ref_max, k_range[1] if k_range else 0), len(X) - 1) + 1)}
    if k_range:
        k = max((kk for kk in sil if k_range[0] <= kk <= k_range[1]), key=sil.get)
    lab = fit(X, k, method, Y)
    k = lab.max() + 1
    k_best = max((kk for kk in sil if kk <= k_ref_max), key=sil.get)
    pref = np.array([np.argmax(norm_units[:, lab == c].mean(axis=1)) for c in range(k)])
    order = np.argsort(pref, kind="stable")                    # clusters ordered by their preferred task period
    return np.argsort(order)[lab], k, silhouette_score(X, lab), k_best, sil[k_best]


def clabel(r):
    """Cluster label: size, and the E/I split when E and I were clustered together."""
    return f"{r.cluster} ({r.n})" if r["pop"] != "U" else f"{r.cluster} ({r.nE}E {r.nI}I)"


def simulate(model, x, keep):
    """DaleRNN.simulate (noise-free) with the rates of silenced units (keep = 0) held at 0 at every step."""
    x = model._normalize_input(x)
    h = torch.zeros(x.shape[0], model.n_rnn)
    W = model.effective_w_rec()
    out = []
    for t in range(x.shape[2]):
        h = (1 - model.alpha) * h + model.alpha * model._cell_act(x[:, :, t] @ model.w_in + h @ W + model.bias)
        h = h * keep
        out.append(torch.sigmoid(h[:, :model.n_e] @ model.w_out + model.b_out))
    return torch.stack(out, dim=1)


def make_eval(model, cfg, ck):
    tasks = list(ck["active_tasks"])
    trials = {}
    for rule in tasks:
        for s in ck["eval_seeds"]:
            c = eval_config_from(cfg, s, easy_task=True)
            tr = generate_trials(rule, c, BATCH, noise_on=False)
            x, _, _, y_loc = trial_to_tensors(tr, "cpu")
            trials[rule, s] = (x, y_loc, response_mask(tr, "cpu"), ring_prefs(c, "cpu", torch.float32))

    @torch.no_grad()
    def acc(keep):
        return pd.Series({rule: np.mean([batch_accuracy(simulate(model, x, keep), y, p, resp_mask=m)["acc"]
                                         for x, y, m, p in (trials[rule, s] for s in ck["eval_seeds"])])
                          for rule in tasks})
    return acc


def main(point, seed, kE, kI, k_range=None, method="kmeans", joint=False):
    # outputs are tagged with the k choice unless it is the default (kE 14, kI 5), so runs don't overwrite each other
    tag = (f"_k{k_range[0]}-{k_range[1]}" if k_range else
           ("" if (kE, kI) == (14, 5) else f"_kE{kE}_kI{kI}"))
    tag += ("_ward" if method == "ward" else "") + ("_joint" if joint else "")
    model, ck, tv_task, tv_per, keys = load(point, seed)
    model.eval()
    N, N_E = model.n_rnn, model.n_e
    total = tv_task.sum(0)
    active = total > 1e-2 * total.max()
    norm = tv_per / np.maximum(tv_per.max(0, keepdims=True), 1e-30)
    with contextlib.redirect_stdout(io.StringIO()):
        _, cfg, _ = load_checkpoint(RUNS["5seed"] / point / f"seed_{seed:02d}.pt")   # load() does not return cfg
    acc = make_eval(model, cfg, ck)
    base = acc(torch.ones(N))
    print(f"full network: min {base.min():.3f}, mean {base.mean():.3f} "
          f"(stored: min {ck['metrics']['min_task_acc']:.3f}, mean {ck['metrics']['mean_acc']:.3f})")

    rows, profiles = [], {}
    pops = ((("U", np.ones(N, bool), kE, 39),) if joint else         # joint: E and I together, as Driscoll
            (("E", np.arange(N) < N_E, kE, 40), ("I", np.arange(N) >= N_E, kI, 20)))
    for pop, sel, k, k_ref in pops:
        units = np.flatnonzero(active & sel)
        lab, k, sil_k, k_best, sil_best = cluster_fixed(norm[:, units], k, k_ref, k_range, method)
        print(f"{pop}: {len(units)} active, k = {k} (silhouette {sil_k:.3f}); "
              f"silhouette-best k = {k_best} ({sil_best:.3f})")
        for c in range(k):
            members = units[lab == c]
            keep = torch.ones(N)
            keep[members] = 0
            frac = (acc(keep) - base) / base
            name = f"{pop}{c + 1}"
            profiles[name] = norm[:, members].mean(axis=1)
            pt, pp = keys[int(np.argmax(profiles[name]))]          # preferred task period -> the cluster's family
            rows.append(dict(cluster=name, pop=pop, n=len(members), nE=int((members < N_E).sum()),
                             nI=int((members >= N_E).sum()), k=k, sil=sil_k, k_best=k_best,
                             pref_task=pt, pref_period=PERIOD_NAME[pp], family=FAMILY[pt],
                             **{f"task_{t}": v for t, v in frac.items()}))
    R = pd.DataFrame(rows)
    out_csv = REPO / "results/front" / f"lesion_period_{point}_s{seed}{tag}.csv"
    R.to_csv(out_csv, index=False)
    plot(R, profiles, keys, list(base.index), point, seed, tag)
    plot_modules(R, list(base.index), point, seed, tag)
    print("->", out_csv)


def plot(R, profiles, keys, tasks, point, seed, tag=""):
    task_cols = [f"task_{t}" for t in tasks]
    T = R.set_index("cluster")[task_cols]
    T.columns = tasks
    fam = pd.DataFrame({f: T[[t for t in tasks if FAMILY[t] == f]].mean(axis=1) for f in FAMILIES})
    fam["Anti"], fam["non-Anti"] = fam["Anti"], T[[t for t in tasks if FAMILY[t] != "Anti"]].mean(axis=1)
    P = pd.DataFrame(profiles).T.loc[T.index]

    n = len(T)
    fig = plt.figure(figsize=(16, 0.32 * n + 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[59, 20, 7], wspace=0.04, left=0.07, right=0.94, top=0.86, bottom=0.2)
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
    ax0.imshow(P.values, cmap="hot", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax0.set_xticks(range(len(keys)))
    ax0.set_xticklabels([f"{rule_name[t]} · {PERIOD_NAME[p]}" for t, p in keys], rotation=90, fontsize=5)
    for i, (t, _) in enumerate(keys):
        ax0.get_xticklabels()[i].set_color(FAM_COLOR[FAMILY[t]])
    ax0.set_yticks(range(n))
    ax0.set_yticklabels([clabel(r) for _, r in R.iterrows()], fontsize=7)
    nE = (R["pop"] == "E").sum()
    for ax in (ax0, ax1, ax2):
        ax.axhline(nE - 0.5, color="k", lw=2)
    ax0.set_title("cluster mean task variance per period (unit-normalised)", fontsize=9)

    lim = max(0.2, float(np.nanmax(np.abs(T.values))))
    kw = dict(cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    im = ax1.imshow(T.values, **kw)
    ax1.set_xticks(range(len(tasks)))
    ax1.set_xticklabels([rule_name[t] for t in tasks], rotation=90, fontsize=7)
    for i, t in enumerate(tasks):
        ax1.get_xticklabels()[i].set_color(FAM_COLOR[FAMILY[t]])
    ax1.set_yticks([])
    ax1.set_title("lesion: fraction performance change per task", fontsize=9)
    ax2.imshow(fam.values, **kw)
    ax2.set_xticks(range(fam.shape[1]))
    ax2.set_xticklabels(fam.columns, rotation=90, fontsize=7)
    ax2.set_yticks([])
    ax2.axvline(4.5, color="k", lw=1)
    ax2.set_title("by family", fontsize=9)
    for (i, j), v in np.ndenumerate(fam.values):
        if abs(v) >= 0.1:
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(v) > 0.6 * lim else "k")
    cax = fig.add_axes([0.95, 0.2, 0.008, 0.66])
    fig.colorbar(im, cax=cax).set_label("(acc lesioned − acc full) / acc full", fontsize=8)
    r = R.groupby("pop").first()
    fig.suptitle(f"{point}, seed {seed}: lesion each per-period cluster (rates set to 0 for the whole trial). "
                 + ", ".join(f"{'E+I' if q == 'U' else q} k = {r.loc[q, 'k']} (silhouette-best {r.loc[q, 'k_best']})"
                             for q in r.index), fontsize=10)
    out = REPO / "results/front" / f"lesion_period_{point}_s{seed}{tag}.png"
    fig.savefig(out, dpi=170)
    print("->", out)


# Driscoll et al. 2024, Fig. 5 (flexible_multitask, analysis/fig5/fig5_ADGJMP.ipynb, make_label_sets): each module
# is a set of tasks, compared with a reference set: the complement for Anti and Delay, the matched alternative for
# the modality and memory-type modules. Same string rules, applied to our 20 task names.
MODULES = {
    "Anti": (lambda r: "anti" in r, lambda r: "anti" not in r),
    "Delay": (lambda r: "delay" in r or "fd" in r, lambda r: not ("delay" in r or "fd" in r)),
    "Modality1": (lambda r: "dm1" in r, lambda r: "dm2" in r),
    "Modality2": (lambda r: "dm2" in r, lambda r: "dm1" in r),
    "Cat Mem.": (lambda r: "dmc" in r, lambda r: "delay" in r),
    "Contin Mem.": (lambda r: "delay" in r, lambda r: "dmc" in r),
}
REF_NAME = {"Anti": "not Anti", "Delay": "not Delay", "Modality1": "Modality2", "Modality2": "Modality1",
            "Cat Mem.": "Contin Mem.", "Contin Mem.": "Cat Mem."}


def module_sets(tasks):
    return {m: ([t for t in tasks if f(t)], [t for t in tasks if g(t)]) for m, (f, g) in MODULES.items()}


def plot_modules(R, tasks, point, seed, tag=""):
    """Driscoll Fig. 5 style for every cluster x module: grey = one task, black = mean of the module tasks and
    of the reference tasks. Also writes the contrast (mean module - mean reference) per cluster and module."""
    sets = module_sets(tasks)
    for m, (a, b) in sets.items():
        print(f"{m}: {len(a)} tasks {[rule_name[t] for t in a]} vs {REF_NAME[m]}: {len(b)}")
    clusters = list(R.cluster)
    fig, axes = plt.subplots(len(sets), len(clusters), figsize=(0.95 * len(clusters) + 1.2, 1.45 * len(sets) + 1),
                             sharey=True)
    rng = np.random.default_rng(0)
    contrast = {}
    for j, (_, r) in enumerate(R.iterrows()):
        for i, (m, (own, ref)) in enumerate(sets.items()):
            ax = axes[i, j]
            means = []
            for x, ts in ((0, ref), (1, own)):
                v = np.array([r[f"task_{t}"] for t in ts])
                ax.scatter(x + rng.uniform(-0.1, 0.1, len(v)), v, s=6, color="k", alpha=0.3, lw=0, zorder=2)
                means.append(v.mean())
            ax.plot([0, 1], means, "-", color="k", lw=1.1, zorder=3)
            contrast[r.cluster, m] = means[1] - means[0]
            ax.axhline(0, color="0.85", lw=0.5, zorder=1)
            ax.set_xlim(-0.5, 1.5)
            ax.set_xticks([])
            ax.spines[["top", "right", "bottom"]].set_visible(False)
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(clabel(r).replace(" (", "\n("), fontsize=7,
                             color={"E": "#c0392b", "I": "#2c6fbb"}.get(r["pop"], "k"))
            if j == 0:
                ax.set_ylabel(f"{m}\n(right) vs\n{REF_NAME[m]}", fontsize=6.5)
            if j == (R["pop"] == "E").sum():                                  # first I cluster
                for a_ in [ax]:
                    a_.spines["left"].set_color("#2c6fbb")
    axes[0, 0].set_ylim(-1.05, 0.15)
    fig.suptitle(f"{point}, seed {seed}: perf. change after lesioning each cluster (columns), for Driscoll's task modules "
                 "(rows). In each panel: left = reference tasks, right = module tasks; dots = tasks, line = means",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.1, h_pad=0.4)
    out = REPO / "results/front" / f"lesion_period_{point}_s{seed}{tag}_modules.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("->", out)

    C = pd.Series(contrast).unstack()[list(sets)].loc[clusters]
    fig, ax = plt.subplots(figsize=(5.2, 0.3 * len(C) + 1.6))
    lim = max(0.2, float(np.abs(C.values).max()))
    im = ax.imshow(C.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(C.values):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(v) > 0.6 * lim else "k")
    ax.set_xticks(range(C.shape[1]))
    ax.set_xticklabels([f"{m}\nvs {REF_NAME[m]}" for m in C.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels([clabel(r) for _, r in R.iterrows()], fontsize=7)
    ax.axhline((R["pop"] == "E").sum() - 0.5, color="k", lw=2)
    fig.colorbar(im, ax=ax, fraction=0.05).set_label("mean change, module − reference", fontsize=8)
    ax.set_title(f"{point}, seed {seed}: module-specific damage (red = module tasks hurt more)", fontsize=8.5)
    fig.tight_layout()
    out = REPO / "results/front" / f"lesion_period_{point}_s{seed}{tag}_modules_contrast.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    C.to_csv(REPO / "results/front" / f"lesion_period_{point}_s{seed}{tag}_modules_contrast.csv")
    print("->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", default="C0")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--kE", type=int, default=14)
    ap.add_argument("--kI", type=int, default=5)
    ap.add_argument("--k-range", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="instead of fixed kE/kI: silhouette-best k within LO..HI, for E and for I")
    ap.add_argument("--method", choices=["kmeans", "ward"], default="kmeans")
    ap.add_argument("--joint", action="store_true", help="cluster E and I together (one population, as Driscoll)")
    ap.add_argument("--driscoll", action="store_true",
                    help="Driscoll's setup: Ward, E and I together, k = silhouette-best in 3..39")
    a = ap.parse_args()
    if a.driscoll:
        a.method, a.joint, a.k_range = "ward", True, (3, 39)
    main(a.point, a.seed, a.kE, a.kI, a.k_range, a.method, a.joint)
