"""Recurrent connectivity matrices ordered by connectivity modules (Louvain, max Q), 1x4.

Best seed per budget point. As in notebooks/cog20_front_explorer.py
(conn_communities): present synapses = E/I-normalised |w| > 1e-2; Louvain
communities on |W| made symmetric; units ordered E then I, and by module
inside each (modules sorted by size). Q is compared with a null that shuffles
the weights inside each Dale block (E->E, E->I, I->E, I->I), keeping density,
weight distribution and E/I strengths.

    python results/scripts/fig_conn_modules.py

Second figure, conn_tv_clusters.png: the same matrices ordered by E/I, then by
task-variance cluster (fig_task_variance_clusters.cluster_units; inactive units
last), with the within/between-cluster mean |w| ratio against a label shuffle.

Writes results/conn_modules/<run>/conn_modules.png, conn_tv_clusters.png and
modules_summary.csv.
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from networkx.algorithms.community import louvain_communities, modularity

from cmc.train_cog import load_checkpoint

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_task_variance_clusters import cluster_units  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUN = REPO / "runs" / "moo_zoom" / "dale_all_curves4_4seed_50k"
OUT_ROOT = REPO / "results" / "conn_modules"
W_THRESH = 1e-2
N_NULL = 10
POINTS = {"control": "Unconstrained", "rate_limited": "Rate-limited",
          "wiring_limited": "Wiring-limited", "both": "Both"}


def q_of(A, seed=0):
    G = nx.from_numpy_array((A + A.T) / 2)
    comms = louvain_communities(G, weight="weight", seed=seed)
    return modularity(G, comms, weight="weight"), comms


def modules(path, seed=0):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, _ = load_checkpoint(path, device="cpu")
    W = model.effective_w_rec().detach().numpy()
    scale = model.ei_row_scale().detach().numpy()[:, None]
    present = (np.abs(W) / scale > W_THRESH) & (model.no_autapse.numpy() > 0)
    W = np.where(present, W, 0.0)
    A = np.abs(W)
    N, n_e = len(A), model.n_e

    Q, comms = q_of(A, seed)
    comms = sorted(comms, key=len, reverse=True)
    comm = np.empty(N, int)
    for i, c in enumerate(comms):
        comm[list(c)] = i

    rng = np.random.default_rng(seed)
    off = ~np.eye(N, dtype=bool)
    blocks = [(slice(0, n_e), slice(0, n_e)), (slice(0, n_e), slice(n_e, N)),
              (slice(n_e, N), slice(0, n_e)), (slice(n_e, N), slice(n_e, N))]
    q_null = []
    for _ in range(N_NULL):
        M = A.copy()
        for r, c in blocks:
            sub, m = M[r, c], off[r, c]
            sub[m] = rng.permutation(sub[m])
        q_null.append(q_of(M, seed)[0])

    is_i = np.arange(N) >= n_e
    order = np.lexsort((np.arange(N), comm, is_i))
    return dict(W=W, order=order, comm=comm, is_i=is_i, n_comm=len(comms), Q=Q, q_null=np.array(q_null))


def tv_groups(path, m, n_perm=200, seed=0):
    """Units ordered E then I, by task-variance cluster inside each (inactive units last).

    Statistic R = mean |w| between units of the same TV cluster / mean |w| between
    different clusters (active units, self-pairs excluded, pruned synapses count as 0).
    Null: cluster labels permuted among E units and, separately, among I units.
    """
    ckpt = __import__("torch").load(path, map_location="cpu", weights_only=False)
    active, _, lab_a, k, _ = cluster_units(np.asarray(ckpt["task_variance"]))
    N = len(m["W"])
    labels = np.full(N, -1)
    labels[active] = lab_a
    is_i = m["is_i"]
    order = np.lexsort((np.arange(N), labels < 0, labels, is_i))
    A = np.abs(m["W"])
    act = np.flatnonzero(labels >= 0)
    Aa, la, ia = A[np.ix_(act, act)], labels[act], is_i[act]
    off = ~np.eye(len(act), dtype=bool)

    def ratio(l):
        same = (l[:, None] == l[None, :]) & off
        return Aa[same].mean() / Aa[~same & off].mean()

    R = ratio(la)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        l = la.copy()
        for g in (~ia, ia):
            l[g] = rng.permutation(l[g])
        null.append(ratio(l))
    return dict(W=m["W"], order=order, comm=labels, is_i=is_i, k=k, R=R, R_null=np.array(null))


def draw(ax, m, name):
    o = m["order"]
    Wo = m["W"][np.ix_(o, o)]
    nz = np.abs(Wo[Wo != 0])
    vmax = np.percentile(nz, 99.5)
    im = ax.imshow(Wo, cmap="RdBu_r", interpolation="nearest",
                   norm=SymLogNorm(linthresh=np.percentile(nz, 5), vmin=-vmax, vmax=vmax))
    key = list(zip(m["is_i"][o], m["comm"][o]))
    for j in range(1, len(o)):
        if key[j] != key[j - 1]:
            ei = key[j][0] != key[j - 1][0]
            for f in (ax.axhline, ax.axvline):
                f(j - 0.5, color="k", lw=1.4 if ei else 0.5, alpha=1.0 if ei else 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("postsynaptic (to)", fontsize=8)
    if "Q" in m:
        ax.set_title(f"{name}\n{m['n_comm']} modules, Q = {m['Q']:.2f} "
                     f"(null {m['q_null'].mean():.2f} ± {m['q_null'].std():.2f})", fontsize=9)
    else:
        ax.set_title(f"{name}\n{m['k']} TV clusters, within/between |w| = {m['R']:.2f} "
                     f"(null {m['R_null'].mean():.2f} ± {m['R_null'].std():.2f})", fontsize=9)
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()

    df = pd.read_csv(args.run / "runs.csv")
    best = df.loc[df.groupby("point").min_task_acc.idxmax()].set_index("point")
    pts = [p for p in POINTS if p in best.index]
    fig, axes = plt.subplots(1, len(pts), figsize=(4.4 * len(pts), 4.8))
    fig2, axes2 = plt.subplots(1, len(pts), figsize=(4.4 * len(pts), 4.8))
    rows = []
    for ax, ax2, p in zip(axes, axes2, pts):
        m = modules(args.run / best.loc[p, "checkpoint"])
        im = draw(ax, m, POINTS[p])
        t = tv_groups(args.run / best.loc[p, "checkpoint"], m)
        im2 = draw(ax2, t, POINTS[p])
        fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
        rows.append(dict(point=p, seed=int(best.loc[p, "seed"]), n_modules=m["n_comm"], Q=m["Q"],
                         Q_null_mean=m["q_null"].mean(), Q_null_std=m["q_null"].std(),
                         tv_k=t["k"], tv_within_between=t["R"], tv_null_mean=t["R_null"].mean(),
                         tv_null_std=t["R_null"].std()))
        print(rows[-1])
    axes[0].set_ylabel("presynaptic (from)\nE above thick line, I below", fontsize=8)
    fig.tight_layout()
    out_dir = OUT_ROOT / args.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "conn_modules.png", dpi=200, bbox_inches="tight")
    axes2[0].set_ylabel("presynaptic (from)\nE above thick line, I below", fontsize=8)
    fig2.tight_layout()
    fig2.savefig(out_dir / "conn_tv_clusters.png", dpi=200, bbox_inches="tight")
    pd.DataFrame(rows).to_csv(out_dir / "modules_summary.csv", index=False)
    print("->", out_dir / "conn_modules.png")


if __name__ == "__main__":
    main()
