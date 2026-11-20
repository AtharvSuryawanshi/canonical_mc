"""Task variance per task period (Driscoll et al. 2024) vs per task, for one network (default: 5-seed run, control, seed 4).

Per task: one row per task, variance across trials averaged over the whole post-fixation trial
(cmc.runner.activity_summary, the value stored in the checkpoint). Per period: one row per (task, period),
the same variance averaged over each non-fixation epoch (cmc.runner.period_variance). Rows are sorted
period first (stimulus, delay, response), as in Driscoll. Same active units for both (the per-task rule:
total > 1% of max), each unit divided by its own max, E and I clustered separately with fig_tv_ei_fanin's
cluster_pop (k-means, best local silhouette maximum with k > 2).

    python results/scripts/fig_tv_period.py [--point control] [--seed 4]

Writes results/front/tv_period_<point>_s<seed>.npz (cache), tv_period_<point>_s<seed>.png (heatmaps, per task
on top, per period below) and tv_period_<point>_s<seed>_compare.png (silhouette vs k, and which per-task group
each per-period group's neurons came from).
"""
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from cmc.runner import period_variance
from cmc.task import rule_name
from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import POINTS, RUNS, cluster_pop

REPO = Path(__file__).resolve().parents[2]
PERIOD_ORDER = ["stim1", "delay1", "stim2", "delay2", "go1"]
PERIOD_NAME = {"stim1": "stim", "delay1": "delay", "stim2": "stim 2", "delay2": "delay 2", "go1": "response"}
PERIOD_COLOR = {"stim1": "#1f4e9c", "delay1": "#4daf4a", "stim2": "#8fb8e8", "delay2": "#dede00", "go1": "#a65628"}
BATCH = 64                                    # eval batch size of the stored task variance (runner default)


def load(point, seed, run="5seed"):
    """Per-period task variance of one network of RUNS[run], cached. The 5seed (v1) caches stay in results/front;
    other runs cache in results/cache/<run>/."""
    ckpt = RUNS[run] / point / f"seed_{seed:02d}.pt"
    with contextlib.redirect_stdout(io.StringIO()):
        model, cfg, ck = load_checkpoint(ckpt)
    cache_dir = REPO / "results/front" if run == "5seed" else REPO / "results/cache" / run
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"tv_period_{point}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        tv_p, keys = z["tv"], [tuple(k) for k in z["keys"]]
    else:
        tv_p, keys, _, _ = period_variance(model, cfg, ck["active_tasks"], torch.device("cpu"), ck["eval_seeds"], BATCH)
        np.savez(cache, tv=tv_p, keys=np.array(keys))
        print("->", cache)
    order = sorted(range(len(keys)), key=lambda i: PERIOD_ORDER.index(keys[i][1]))      # period first, then task
    return model, ck, np.asarray(ck["task_variance"], float), tv_p[order].astype(float), [keys[i] for i in order]


def clusters(tv, active, model):
    norm = tv / np.maximum(tv.max(0, keepdims=True), 1e-30)
    N, N_E = model.n_rnn, model.n_e
    out = dict(norm=norm)
    for pop, sel, k_max in (("E", np.arange(N) < N_E, 20), ("I", np.arange(N) >= N_E, 10)):
        units = np.flatnonzero(active & sel)
        lab, K, sil = cluster_pop(norm[:, units], k_max)
        out[pop] = dict(units=units, lab=lab, K=K, sil=sil[K], sil_curve=sil, breadth=(norm[:, units] > 0.5).sum(0).mean())
    return out


def panel(ax, C, pop, labels, title, row_colors=None):
    P = C[pop]
    o = np.argsort(P["lab"], kind="stable")
    n = len(labels)
    ax.imshow(C["norm"][:, P["units"][o]], cmap="hot", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    labs = P["lab"][o]
    for c in range(P["K"]):
        pos = np.flatnonzero(labs == c)
        if c:
            ax.axvline(pos[0] - 0.5, color="white", lw=0.6)
        ax.text(pos.mean(), n - 0.5 + 0.012 * n, str(c + 1), ha="center", va="top", fontsize=5.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=4.3 if n > 30 else 5.5)
    if row_colors:
        for t, c in zip(ax.get_yticklabels(), row_colors):
            t.set_color(c)
    ax.set_xticks([])
    ax.set_ylim(n - 0.5 + 0.045 * n, -0.5)
    ax.set_title(title, fontsize=6.5, loc="left")
    ax.spines[:].set_visible(False)


def main(point="control", seed=4):
    model, ck, tv_t, tv_p, keys = load(point, seed)
    total = tv_t.sum(0)
    active = total > 1e-2 * total.max()
    T, Pd = clusters(tv_t, active, model), clusters(tv_p, active, model)
    who = f"{POINTS[point]}, seed {seed} (5-seed run, fanin, 150k)"

    fig = plt.figure(figsize=(7.4, 11))
    gs = fig.add_gridspec(2, 2, width_ratios=[204, 80], height_ratios=[20, 59], left=0.14, right=0.99,
                          top=0.95, bottom=0.02, hspace=0.12, wspace=0.05)
    task_labels = [rule_name[t] for t in ck["active_tasks"]]
    per_labels = [f"{rule_name[t]} · {PERIOD_NAME[e]}" for t, e in keys]
    per_colors = [PERIOD_COLOR[e] for _, e in keys]
    for col, pop in enumerate("EI"):
        for row, (C, labels, rc, what, unit) in enumerate(((T, task_labels, None, "per task", "tasks"),
                                                            (Pd, per_labels, per_colors, "per period", "periods"))):
            P = C[pop]
            panel(fig.add_subplot(gs[row, col]), C, pop, labels if col == 0 else [""] * len(labels),
                  f"{what}, {pop}: {len(P['units'])} units, {P['K']} groups\n"
                  f"sil {P['sil']:.2f}, {P['breadth']:.1f} {unit}/neuron", rc if col == 0 else None)
    fig.suptitle(f"Task variance per task (top) vs per task period (bottom): {who}", fontsize=8.5)
    out = REPO / "results/front" / f"tv_period_{point}_s{seed}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("->", out)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), gridspec_kw=dict(height_ratios=[1, 1.6]))
    for j, pop in enumerate("EI"):
        ax = axes[0, j]
        for C, name, c in ((T, "per task", "#444444"), (Pd, "per period", "#d62728")):
            s = C[pop]["sil_curve"]
            ks = sorted(s)
            ax.plot(ks, [s[k] for k in ks], "-o", ms=3, color=c, label=f"{name} (k = {C[pop]['K']})")
            ax.axvline(C[pop]["K"], color=c, ls="--", lw=1)
        ax.set_title(f"{pop}: silhouette vs k", fontsize=9, loc="left")
        ax.set_xlabel("k")
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax = axes[1, j]
        a, b = T[pop]["lab"], Pd[pop]["lab"]
        M = np.zeros((T[pop]["K"], Pd[pop]["K"]), int)
        np.add.at(M, (a, b), 1)
        ax.imshow(M, cmap="Blues", aspect="auto")
        for (i, k), v in np.ndenumerate(M):
            if v:
                ax.text(k, i, v, ha="center", va="center", fontsize=6, color="white" if v > M.max() / 2 else "black")
        ax.set_xticks(range(M.shape[1])); ax.set_xticklabels(range(1, M.shape[1] + 1), fontsize=7)
        ax.set_yticks(range(M.shape[0])); ax.set_yticklabels(range(1, M.shape[0] + 1), fontsize=7)
        ax.set_xlabel("per-period group"); ax.set_ylabel("per-task group")
        ax.set_title(f"{pop}: neurons per (per-task group, per-period group); ARI {adjusted_rand_score(a, b):.2f}",
                     fontsize=9, loc="left")
        print(pop, {"per task": (T[pop]["K"], round(T[pop]["sil"], 2)), "per period": (Pd[pop]["K"], round(Pd[pop]["sil"], 2)),
                    "ARI": round(adjusted_rand_score(a, b), 2)})
    fig.suptitle(f"Per task vs per period clustering: {who}", fontsize=10)
    fig.tight_layout()
    out = REPO / "results/front" / f"tv_period_{point}_s{seed}_compare.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", choices=sorted(POINTS), default="control")
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    main(a.point, a.seed)
