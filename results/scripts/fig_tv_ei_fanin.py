"""Task-variance clustering, E and I separately, for the four fanin save-at networks (150k, seed 0).

Same method as reports/ei_modularity/make_figures.py (cluster_pop and tv_panel copied from there):
active units (total task variance > 1% of max), each unit divided by its own max, E and I clustered
separately with k-means, k by the local-max silhouette rule (k > 2; E up to 20, I up to 10).

    python results/scripts/fig_tv_ei_fanin.py [step]    # default: final (150k); or e.g. 50000

Writes results/front/tv_ei_fanin_<step>.png, tv_ei_fanin_<step>_sil.png (silhouette vs k, E and I,
chosen k as a dashed line) and tv_ei_fanin_<step>_wiring.png: task similarity and wiring between all E and
I groups (as notebooks/ei_groups_figures.ipynb, Figure 2: standardised |w| = |W| / E-I row scale, mean over
group pairs without self-pairs, z-scored against 500 shuffles of group labels within E and within I; groups
smaller than 4 neurons greyed out).
"""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cmc.task import rule_name
from cmc.train_cog import load_checkpoint

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/moo_zoom/dale_all_woutfree_fanin_4pt_1seed_150k_saveat"
POINTS = {"control": "Unconstrained", "A3": "A3, rate only", "B0": "B0, wiring only", "C0": "C0, both"}
COLORS = {"control": "#444444", "A3": "#d62728", "B0": "#1f77b4", "C0": "#2ca02c"}   # as in the E/I report


def cluster_pop(norm, k_max):
    X = norm.T
    k_max = min(k_max, len(X) - 1)
    sil, labs = {}, {}
    for k in range(2, k_max + 1):
        labs[k] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        sil[k] = silhouette_score(X, labs[k])
    local = [k for k in range(3, k_max) if sil[k] > sil[k - 1] and sil[k] > sil[k + 1]]
    K = max(local, key=sil.get) if local else max(sil, key=sil.get)
    lab = labs[K]
    pref = {c: int(np.argmax(norm[:, lab == c].mean(axis=1))) for c in range(K)}
    order = sorted(range(K), key=lambda c: (pref[c], -norm[pref[c], lab == c].mean()))
    return np.array([order.index(c) for c in lab]), K, sil


def analyse(ckpt):
    with contextlib.redirect_stdout(io.StringIO()):
        model, _, ck = load_checkpoint(ckpt)
    tv = np.asarray(ck["task_variance"], dtype=float)
    total = tv.sum(0)
    active = total > 1e-2 * total.max()
    norm = tv / np.maximum(tv.max(0, keepdims=True), 1e-30)
    R = dict(tasks=list(ck["active_tasks"]), norm=norm, acc=ck["metrics"]["min_task_acc"],
             Wn=np.abs(model.effective_w_rec().detach().numpy())
             / model.ei_row_scale().detach().numpy()[:, None])          # standardised |w|, W[pre, post]
    N, N_E = model.n_rnn, model.n_e
    for pop, sel, k_max in (("E", np.arange(N) < N_E, 20), ("I", np.arange(N) >= N_E, 10)):
        units = np.flatnonzero(active & sel)
        lab, K, sil = cluster_pop(norm[:, units], k_max)
        R[pop] = dict(units=units, lab=lab, K=K, sil=sil[K], sil_curve=sil,
                      breadth=(norm[:, units] > 0.5).sum(0).mean())
    return R


def tv_panel(ax, R, pop, show_y, head=""):
    P = R[pop]
    o = np.argsort(P["lab"], kind="stable")
    ax.imshow(R["norm"][:, P["units"][o]], cmap="hot", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    labs = P["lab"][o]
    n_t = len(R["tasks"])
    for c in range(P["K"]):
        pos = np.flatnonzero(labs == c)
        if c:
            ax.axvline(pos[0] - 0.5, color="white", lw=0.6)
        ax.text(pos.mean(), n_t + 0.1, str(c + 1), ha="center", va="top", fontsize=5.5)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([rule_name[t] for t in R["tasks"]] if show_y else [], fontsize=5.5)
    ax.set_xticks([])
    ax.set_ylim(n_t + 0.9, -0.5)
    ax.set_title(f"{head}{pop}: {len(P['units'])} units, {P['K']} groups\n"
                 f"sil {P['sil']:.2f}, {P['breadth']:.1f} tasks/neuron", fontsize=6.5, loc="left")
    ax.spines[:].set_visible(False)


def main(step=None):
    sub = "" if step is None else f"step_{int(step):06d}/"
    tag = "150k" if step is None else f"{int(step) // 1000}k"
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(4, 2, width_ratios=[204, 80], left=0.12, right=0.99, top=0.93, bottom=0.03,
                          hspace=0.5, wspace=0.05)
    ALL = {}
    for i, (p, name) in enumerate(POINTS.items()):
        R = ALL[p] = analyse(RUN / p / f"{sub}seed_00.pt")
        tv_panel(fig.add_subplot(gs[i, 0]), R, "E", True, head=f"{name} (min task acc {R['acc']:.2f})   ")
        tv_panel(fig.add_subplot(gs[i, 1]), R, "I", False)
        print(p, {pop: (R[pop]["K"], round(R[pop]["sil"], 2), round(R[pop]["breadth"], 1)) for pop in "EI"})
    fig.suptitle(f"Task variance, E and I clustered separately: cog20, fanin, {tag} steps, seed 0", fontsize=9)
    out = REPO / "results/front" / f"tv_ei_fanin_{tag}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("->", out)
    plot_silhouette(ALL, tag)
    plot_wiring(ALL, tag)


N_SHUFFLE, MIN_GROUP = 500, 4


def wiring(R):
    E, I = R["E"], R["I"]
    KE, Kt = E["K"], E["K"] + I["K"]
    units = np.r_[E["units"], I["units"]]
    lab = np.r_[E["lab"], I["lab"] + KE]
    isE = np.r_[np.ones(len(E["units"]), bool), np.zeros(len(I["units"]), bool)]
    sizes = np.bincount(lab, minlength=Kt)
    names = [f"E{c + 1} ({n})" for c, n in enumerate(sizes[:KE])] +             [f"I{c + 1} ({n})" for c, n in enumerate(sizes[KE:])]
    prof = np.array([R["norm"][:, units[lab == g]].mean(axis=1) for g in range(Kt)])
    Wu = R["Wn"][np.ix_(units, units)].copy()
    np.fill_diagonal(Wu, 0.0)

    def blocks(l):
        Gm = np.eye(Kt)[l]
        n = Gm.sum(0)
        return (Gm.T @ Wu @ Gm) / (np.outer(n, n) - np.diag(n))      # mean over pairs, no self-pairs

    obs = blocks(lab)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(N_SHUFFLE):
        l = lab.copy()
        l[isE] = rng.permutation(l[isE])
        l[~isE] = rng.permutation(l[~isE])
        null.append(blocks(l))
    null = np.array(null)
    z = (obs - null.mean(0)) / null.std(0)
    small = sizes < MIN_GROUP
    z[small, :] = np.nan
    z[:, small] = np.nan
    sim = np.corrcoef(prof)
    # summary per quadrant: Spearman rho(similarity, z) over pairs of different groups, within / between for X->X
    from scipy.stats import spearmanr
    q = {}
    for name, rows, cols in (("E→E", slice(0, KE), slice(0, KE)), ("E→I", slice(0, KE), slice(KE, Kt)),
                             ("I→E", slice(KE, Kt), slice(0, KE)), ("I→I", slice(KE, Kt), slice(KE, Kt))):
        S, Z, O = sim[rows, cols], z[rows, cols], obs[rows, cols]
        off = ~np.eye(*S.shape, dtype=bool) if rows == cols else np.ones(S.shape, bool)
        ok = off & np.isfinite(Z)
        q[name] = dict(rho=spearmanr(S[ok], Z[ok])[0] if ok.sum() > 2 else np.nan)
        if rows == cols:
            q[name]["within/between"] = np.diag(O).mean() / O[off].mean()
    return dict(names=names, KE=KE, Kt=Kt, sim=sim, z=z, q=q)


def matrix_panel(ax, M, w, cmap, lim, cbar_label):
    cm = matplotlib.colormaps[cmap].copy()
    cm.set_bad("#bdbdbd")
    im = ax.imshow(M, cmap=cm, vmin=-lim, vmax=lim, interpolation="nearest")
    kE, Kt = w["KE"], w["Kt"]
    ax.axhline(kE - 0.5, color="k", lw=2)
    ax.axvline(kE - 0.5, color="k", lw=2)
    ax.set_xticks(range(Kt)); ax.set_xticklabels(w["names"], rotation=90, fontsize=6)
    ax.set_yticks(range(Kt)); ax.set_yticklabels(w["names"], fontsize=6)
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=6)


def plot_wiring(ALL, tag):
    fig, axes = plt.subplots(len(ALL), 2, figsize=(13, 6.2 * len(ALL)))
    for r, (p, R) in enumerate(ALL.items()):
        w = wiring(R)
        matrix_panel(axes[r, 0], w["sim"], w, "coolwarm", 1, "r of the two groups' task profiles")
        axes[r, 0].set_title(f"{POINTS[p]} (min task acc {R['acc']:.2f}): task similarity", loc="left",
                             fontsize=9, weight="bold")
        ax = axes[r, 1]
        matrix_panel(ax, w["z"], w, "RdBu_r", 4, "z vs shuffled groups (grey: group < 4 neurons)")
        for (a, b), v in np.ndenumerate(w["z"]):
            if np.isfinite(v) and abs(v) >= 3:
                ax.text(b, a, f"{v:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 6 else "black")
        kE = w["KE"]
        for y0, x0, qn in ((-0.5, -0.5, "E→E"), (-0.5, kE - 0.5, "E→I"), (kE - 0.5, -0.5, "I→E"),
                           (kE - 0.5, kE - 0.5, "I→I")):
            ax.text(x0 + 0.2, y0 + 0.2, qn, fontsize=7, weight="bold", ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        summ = "   ".join(f"{k}: ρ {v['rho']:+.2f}" + (f", w/b {v['within/between']:.2f}" if "within/between" in v
                                                        else "") for k, v in w["q"].items())
        ax.set_title(f"wiring, row → column group\n{summ}", loc="left", fontsize=7.5)
        ax.set_xlabel("to (post)", fontsize=7); ax.set_ylabel("from (pre)", fontsize=7)
        print(p, {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in w["q"].items()})
    fig.suptitle(f"Task similarity (left) and wiring (right) between all E and I groups: fanin, {tag} steps, seed 0",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = REPO / "results/front" / f"tv_ei_fanin_{tag}_wiring.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("->", out)


def plot_silhouette(ALL, tag):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, pop in zip(axes, "EI"):
        for p, R in ALL.items():
            sil = R[pop]["sil_curve"]
            ks = sorted(sil)
            ax.plot(ks, [sil[k] for k in ks], "-o", ms=3, color=COLORS[p], label=f"{POINTS[p]} (k = {R[pop]['K']})")
            ax.axvline(R[pop]["K"], color=COLORS[p], ls="--", lw=1)
        ax.set_title(f"{pop} neurons")
        ax.set_xlabel("number of clusters k")
        ax.set_xticks(range(2, max(max(R[pop]["sil_curve"]) for R in ALL.values()) + 1, 2))
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("silhouette score")
    fig.suptitle(f"Silhouette vs k (dashed: chosen k, best local maximum with k > 2): fanin, {tag} steps, seed 0",
                 fontsize=9)
    fig.tight_layout()
    out = REPO / "results/front" / f"tv_ei_fanin_{tag}_sil.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
