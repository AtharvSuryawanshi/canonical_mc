"""Task-variance clusters (Yang 2019 Fig. 3c) and silhouette score vs k.

For the best seed of each budget point, following reports/theory.md section 6:
active units = summed task variance above 1% of the network's max; each unit's
task-variance vector is normalised by its max over tasks; k-means for
k = 2..10, k chosen by silhouette score. Units are sorted by cluster, clusters
by their dominant task; strips under the heatmap give cluster and E/I identity.

    python results/scripts/fig_task_variance_clusters.py
    python results/scripts/fig_task_variance_clusters.py --point control --k 6

Writes results/unit_task_variance/<run>/<point>_tv_clusters.png and
<point>_silhouette.png, plus clusters_summary.csv.
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
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cmc.task import rule_name
from cmc.train_cog import load_checkpoint

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "runs" / "moo_zoom" / "dale_all_curves4_4seed_50k"
OUT_ROOT = REPO / "results" / "unit_task_variance"
K_RANGE = range(2, 20)
ACTIVE_FRAC = 0.01
POINT_TITLES = {
    "control": "Unconstrained",
    "rate_limited": "Rate-limited",
    "wiring_limited": "Wiring-limited",
    "both": "Rate + wiring limited",
}


def cluster_units(tv, k=None, seed=0):
    """tv (n_tasks, N) -> active mask, normalised X (n_active, n_tasks), labels, scores per k."""
    total = tv.sum(axis=0)
    active = total > ACTIVE_FRAC * total.max()
    X = (tv[:, active] / tv[:, active].max(axis=0)).T
    scores, labels = {}, {}
    for kk in K_RANGE:
        lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(X)
        scores[kk], labels[kk] = silhouette_score(X, lab), lab
    k = k or max(scores, key=scores.get)
    return active, X, labels[k], k, scores


def plot_heatmap(X, labels, is_e, tasks, title, out):
    n_k = labels.max() + 1
    dominant = np.array([X[labels == c].mean(axis=0).argmax() for c in range(n_k)])
    cluster_order = np.argsort(dominant, kind="stable")
    rank = np.empty(n_k, int)
    rank[cluster_order] = np.arange(n_k)
    new_lab = rank[labels]
    order = np.lexsort((-X.max(axis=1), new_lab))

    fig = plt.figure(figsize=(7.5, 5.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[20, 0.8, 0.8], width_ratios=[40, 1], hspace=0.08, wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(X[order].T, aspect="auto", cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([rule_name.get(t, t) for t in tasks], fontsize=8)
    ax.set_xticks([])
    ax.set_title(title, fontsize=10)
    cax = fig.add_subplot(gs[0, 1])
    fig.colorbar(im, cax=cax, ticks=[0, 1]).set_label("Normalized task variance", fontsize=8)

    cmap = ListedColormap(plt.get_cmap("tab20")(np.linspace(0, 1, 20))[:max(n_k, 1)])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.imshow(new_lab[order][None, :], aspect="auto", cmap=cmap, vmin=-0.5, vmax=n_k - 0.5,
                interpolation="nearest")
    ax_c.set_xticks([])
    ax_c.set_yticks([0])
    ax_c.set_yticklabels(["Cluster"], fontsize=8)
    for c in range(n_k):
        idx = np.where(new_lab[order] == c)[0]
        ax_c.text(idx.mean(), 0, str(c + 1), ha="center", va="center", fontsize=7)

    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.imshow(is_e[order][None, :].astype(float), aspect="auto", cmap=ListedColormap(["#2166ac", "#b2182b"]),
                vmin=0, vmax=1, interpolation="nearest")
    ax_e.set_yticks([0])
    ax_e.set_yticklabels(["E / I"], fontsize=8)
    ax_e.set_xticks([])
    ax_e.set_xlabel(f"Units (n = {len(order)} active; red = E, blue = I)", fontsize=8)
    for ext in ("png",):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette(scores, k, title, out):
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    ks = list(scores)
    ax.plot(ks, [scores[kk] for kk in ks], "-o", color="k", ms=4)
    ax.plot([k], [scores[k]], "o", color="#d62728", ms=7, zorder=3)
    ax.hlines(0.2, ks[0], ks[-1], color="#310808", lw=0.8, ls="--", zorder=2) 
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Silhouette score")
    ax.set_xticks(ks[::2])
    ax.set_title(f"{title}\nchosen k = {k}", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png",):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--point", nargs="*", default=None, help="Default: every point in the run.")
    ap.add_argument("--k", type=int, default=None, help="Force k instead of the silhouette choice.")
    args = ap.parse_args()

    df = pd.read_csv(args.run / "runs.csv")
    best = df.loc[df.groupby("point").min_task_acc.idxmax()].set_index("point")
    points = args.point or [p for p in POINT_TITLES if p in best.index] + \
        [p for p in best.index if p not in POINT_TITLES]
    out_dir = OUT_ROOT / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for point in points:
        with contextlib.redirect_stdout(io.StringIO()):
            model, _, ckpt = load_checkpoint(args.run / best.loc[point, "checkpoint"], device="cpu")
        tv = np.asarray(ckpt["task_variance"])
        e_mask = model.E_mask.cpu().numpy() > 0
        active, X, labels, k, scores = cluster_units(tv, args.k)
        title = f"{POINT_TITLES.get(point, point)} (seed {ckpt['seed']})"
        plot_heatmap(X, labels, e_mask[active], list(ckpt["active_tasks"]), title, out_dir / f"{point}_tv_clusters")
        plot_silhouette(scores, k, title, out_dir / f"{point}_silhouette")
        rows.append(dict(point=point, seed=ckpt["seed"], n_active=int(active.sum()), k=k,
                         silhouette=scores[k], **{f"sil_k{kk}": s for kk, s in scores.items()}))
        print(f"{point}: seed {ckpt['seed']}, {active.sum()} active units, k = {k}, silhouette {scores[k]:.3f}")
    pd.DataFrame(rows).to_csv(out_dir / "clusters_summary.csv", index=False)


if __name__ == "__main__":
    main()
