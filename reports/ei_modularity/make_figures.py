"""Figures for reports/ei_modularity/ei_modularity.tex (no lesions; clustering + wiring only).

Same analysis as notebooks/ei_modularity_150k.ipynb: active units (total task variance > 1% of max), each unit's
task variance divided by its own max, E and I clustered separately with k-means (local-max silhouette rule, k > 2),
I-group x E-group task-profile r, block mean |W| of present synapses z-scored against 500 label shuffles,
Spearman rho between the two. Run from the repo root: python reports/ei_modularity/make_figures.py
"""
import contextlib
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cmc.paths import MOO_ZOOM_RUNS_DIR
from cmc.task import rule_name
from cmc.train_cog import load_checkpoint

OUT = Path(__file__).parent / "fig"
OUT.mkdir(exist_ok=True)
RUN = MOO_ZOOM_RUNS_DIR / "dale_all_woutfree_fanin_4pt_5seed_150k"
POINTS = ["control", "A3", "B0", "C0"]
NAME = {"control": "control", "A3": "A3 (rate only)", "B0": "B0 (wiring only)", "C0": "C0 (both)"}
COL = dict(zip(POINTS, ["#444444", "#d62728", "#1f77b4", "#2ca02c"]))
N_SHUFFLE, PRESENT = 500, 1e-2

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


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
    tasks = list(ck["active_tasks"])
    N, N_E = model.n_rnn, model.n_e
    tv = np.asarray(ck["task_variance"], dtype=float)
    total = tv.sum(0)
    active = total > 1e-2 * total.max()
    norm = tv / np.maximum(tv.max(0, keepdims=True), 1e-30)
    R = dict(tasks=tasks, norm=norm)
    for pop, sel, k_max in (("E", np.arange(N) < N_E, 20), ("I", np.arange(N) >= N_E, 10)):
        units = np.flatnonzero(active & sel)
        lab, K, sil = cluster_pop(norm[:, units], k_max)
        prof = np.stack([norm[:, units[lab == c]].mean(axis=1) for c in range(K)])
        R[pop] = dict(units=units, lab=lab, K=K, sil=sil[K], prof=prof,
                      breadth=(norm[:, units] > 0.5).sum(0), sizes=np.bincount(lab, minlength=K))
    W = model.effective_w_rec().detach().numpy()
    scale = model.ei_row_scale().detach().numpy()[:, None]
    A = np.where(np.abs(W) / scale > PRESENT, np.abs(W), 0.0)
    E, I = R["E"], R["I"]

    def blocks(labE, labI):
        GE, GI = np.eye(E["K"])[labE], np.eye(I["K"])[labI]
        nE, nI = GE.sum(0), GI.sum(0)
        return ((GI.T @ A[np.ix_(I["units"], E["units"])] @ GE) / np.outer(nI, nE),
                (GE.T @ A[np.ix_(E["units"], I["units"])] @ GI) / np.outer(nE, nI))

    IE, EI = blocks(E["lab"], I["lab"])
    rng = np.random.default_rng(0)
    nulls = [blocks(rng.permutation(E["lab"]), rng.permutation(I["lab"])) for _ in range(N_SHUFFLE)]
    nIE, nEI = np.array([n[0] for n in nulls]), np.array([n[1] for n in nulls])
    R["IE_z"] = (IE - nIE.mean(0)) / np.maximum(nIE.std(0), 1e-12)
    R["EI_z"] = ((EI - nEI.mean(0)) / np.maximum(nEI.std(0), 1e-12)).T     # (KI, KE) like IE_z
    R["sim"] = np.corrcoef(np.vstack([I["prof"], E["prof"]]))[:I["K"], I["K"]:]
    R["rho_IE"] = spearmanr(R["sim"].ravel(), R["IE_z"].ravel())[0]
    R["rho_EI"] = spearmanr(R["sim"].ravel(), R["EI_z"].ravel())[0]
    R["match"] = R["sim"].max(axis=1).mean()
    R["present"] = float((np.abs(W) / scale > PRESENT).sum() / (N * (N - 1)))
    return R


runs = pd.read_csv(RUN / "runs.csv")
ALL = {(p, s): analyse(RUN / p / f"seed_{s:02d}.pt") for p in POINTS for s in range(5)}
rows = []
for (p, s), R in ALL.items():
    r = runs[(runs.point == p) & (runs.seed == s)].iloc[0]
    rows.append(dict(point=p, seed=s, min_task_acc=r.min_task_acc, mean_acc=r.mean_acc,
                     metabolic=r.metabolic_cost, wiring=r.wiring_cost, present=R["present"],
                     E_active=len(R["E"]["units"]), I_active=len(R["I"]["units"]), KE=R["E"]["K"], KI=R["I"]["K"],
                     E_tasks=R["E"]["breadth"].mean(), I_tasks=R["I"]["breadth"].mean(),
                     E_sil=R["E"]["sil"], I_sil=R["I"]["sil"], match=R["match"],
                     rho_IE=R["rho_IE"], rho_EI=R["rho_EI"]))
T = pd.DataFrame(rows)
T.to_csv(OUT / "summary.csv", index=False)
best = {p: int(T[T.point == p].sort_values("min_task_acc").seed.iloc[-1]) for p in POINTS}
json.dump(best, open(OUT / "best_seed.json", "w"))
print(T.groupby("point", sort=False).agg(["mean", "std"]).round(3).T.to_string())
print("best seeds", best)

# ---------------- Fig 1: the four points (budgets and what the networks actually spend)
budgets = runs.groupby("point").first()[["rate_budget", "conn_budget"]]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
ax = axes[0]
FREE_R, FREE_W = 0.3, 0.06                               # where "no budget" is drawn
for p in POINTS:
    rb, wb = budgets.loc[p]
    x = FREE_R if (np.isnan(rb) or rb >= 1) else rb
    y = FREE_W if (np.isnan(wb) or wb >= 1) else wb
    ax.scatter(x, y, s=60, color=COL[p], zorder=3)
    ax.annotate(p, (x, y), xytext=(6, 4), textcoords="offset points", color=COL[p], fontsize=8)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(5e-4, 0.6); ax.set_ylim(2e-3, 0.1)
ax.axvline(FREE_R / 1.6, color="grey", lw=0.5, ls=":"); ax.axhline(FREE_W / 1.4, color="grey", lw=0.5, ls=":")
ax.text(FREE_R, 2.3e-3, "no rate\nbudget", ha="center", fontsize=6.5, color="grey")
ax.text(0.012, FREE_W * 1.12, "no wiring budget", fontsize=6.5, color="grey", ha="center")
ax.set_xlabel("rate budget (ceiling on firing cost)"); ax.set_ylabel("wiring budget (ceiling on L1 of $W_{rec}$)")
ax.set_title("a   budgets the networks were trained under", loc="left")
ax = axes[1]
for p in POINTS:
    d = T[T.point == p]
    ax.scatter(d.metabolic, d.wiring, color=COL[p], s=18, label=NAME[p])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("firing cost after training"); ax.set_ylabel("wiring cost after training")
ax.set_title("b   what they spend (5 seeds each)", loc="left")
ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.45, 0.5))
plt.tight_layout()
fig.savefig(OUT / "fig1_points.pdf"); plt.close(fig)

# ---------------- Fig 2: the three types (schematic)
fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35))
tcol = ["#e6a100", "#7b3fb5", "#0f8a8a"]
titles = ["type 1: E modules,\nshared inhibition", "type 2: mixed E+I modules",
          "type 3: inhibition\nbetween modules"]
for t, ax in enumerate(axes):
    ax.set_xlim(0, 3); ax.set_ylim(-0.35, 2.3); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(titles[t], fontsize=8.5)
    xs = [0.5, 1.5, 2.5]
    for x, c, lab in zip(xs, tcol, ["task A", "task B", "task C"]):
        ax.add_patch(Circle((x, 1.75), 0.32, color=c, alpha=0.85))
        ax.text(x, 1.75, "E", ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(x, 2.17, lab, ha="center", fontsize=6.5, color=c)
    if t == 0:
        ax.add_patch(Circle((1.5, 0.35), 0.3, color="#888888"))
        ax.text(1.5, 0.35, "I", ha="center", va="center", color="white", fontsize=9, weight="bold")
        ax.text(1.5, -0.2, "broad: all tasks", ha="center", fontsize=6.5, color="#555555")
        for x in xs:
            ax.add_patch(FancyArrowPatch((1.5, 0.65), (x, 1.42), arrowstyle="-[", mutation_scale=5,
                                         color="#555555", lw=0.9))
    else:
        for i, (x, c) in enumerate(zip(xs, tcol)):
            ax.add_patch(Circle((x, 0.35), 0.22, color=c, alpha=0.85))
            ax.text(x, 0.35, "I", ha="center", va="center", color="white", fontsize=8, weight="bold")
            if t == 1:
                ax.add_patch(FancyArrowPatch((x, 0.58), (x, 1.42), arrowstyle="-[", mutation_scale=5,
                                             color=c, lw=1.2))
            else:
                for x2 in xs:
                    if x2 != x:
                        ax.add_patch(FancyArrowPatch((x, 0.58), (x2, 1.42), arrowstyle="-[", mutation_scale=4,
                                                     color=c, lw=0.8, connectionstyle="arc3,rad=0.1"))
        ax.text(1.5, -0.2, "each I group tuned like one E group", ha="center", fontsize=6.5, color="#555555")
plt.tight_layout()
fig.savefig(OUT / "fig2_types.pdf"); plt.close(fig)


# ---------------- per-network figure: E | I task variance, then the two scatters
def tv_panel(ax, R, pop, show_y):
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
    ax.set_title(f"{pop}: {len(P['units'])} units, {P['K']} groups", fontsize=7.5)
    ax.spines[:].set_visible(False)


def scatter_panel(ax, R, key, rho_key, title):
    ax.scatter(R["sim"].ravel(), R[key].ravel(), s=9, color="#333333", alpha=0.7, lw=0)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlim(-1, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("task similarity (r) of the I group and E group")
    ax.set_title(f"{title}\nSpearman ρ = {R[rho_key]:+.2f}", fontsize=7.5)


for p in POINTS:
    R = ALL[p, best[p]]
    fig = plt.figure(figsize=(6.8, 5.0))
    top = fig.add_gridspec(1, 2, width_ratios=[len(R["E"]["units"]), max(len(R["I"]["units"]), 62)],
                           left=0.13, right=0.99, top=0.95, bottom=0.53, wspace=0.05)
    tv_panel(fig.add_subplot(top[0]), R, "E", True)
    tv_panel(fig.add_subplot(top[1]), R, "I", False)
    bot = fig.add_gridspec(1, 2, left=0.13, right=0.97, top=0.42, bottom=0.09, wspace=0.3)
    ax = fig.add_subplot(bot[0]); scatter_panel(ax, R, "IE_z", "rho_IE", "I → E wiring")
    ax.set_ylabel("wiring strength\n(z vs shuffled groups)")
    ax = fig.add_subplot(bot[1]); scatter_panel(ax, R, "EI_z", "rho_EI", "E → I wiring")
    fig.savefig(OUT / f"fig_net_{p}.pdf"); plt.close(fig)

# ---------------- method figure: the two matrices and the scatter for C0, best seed
R = ALL["C0", best["C0"]]
fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
for ax, M, title, lim in [(axes[0], R["sim"], "a   task similarity r", 1),
                          (axes[1], R["IE_z"], "b   I → E wiring, z", 4)]:
    im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(R["E"]["K"])); ax.set_xticklabels([f"E{c + 1}" for c in range(R["E"]["K"])], fontsize=5.5,
                                                          rotation=90)
    ax.set_yticks(range(R["I"]["K"])); ax.set_yticklabels([f"I{c + 1}" for c in range(R["I"]["K"])], fontsize=6)
    ax.set_title(title, loc="left"); ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
scatter_panel(axes[2], R, "IE_z", "rho_IE", "c   one dot per (I, E) pair")
axes[2].set_xlabel("task similarity r"); axes[2].set_ylabel("I → E wiring z")
plt.tight_layout()
fig.savefig(OUT / "fig3_method.pdf"); plt.close(fig)

# ---------------- summary across seeds
metrics = [("E_tasks", "E: tasks per neuron"), ("I_tasks", "I: tasks per neuron"),
           ("match", "best E partner\nof an I group (r)"), ("rho_IE", "I → E wiring\nvs similarity (ρ)"),
           ("rho_EI", "E → I wiring\nvs similarity (ρ)")]
fig, axes = plt.subplots(1, len(metrics), figsize=(6.8, 2.3))
for ax, (m, title) in zip(axes, metrics):
    for i, p in enumerate(POINTS):
        v = T[T.point == p][m].values
        ax.scatter(np.full(5, i) + np.linspace(-0.14, 0.14, 5), v, color=COL[p], s=11, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=COL[p], lw=1.6)
    if m.startswith("rho"):
        ax.axhline(0, color="grey", lw=0.6); ax.set_ylim(-0.85, 0.85)
    ax.set_xticks(range(4)); ax.set_xticklabels(POINTS, rotation=45, fontsize=6.5)
    ax.set_title(title, fontsize=7.5)
plt.tight_layout()
fig.savefig(OUT / "fig4_summary.pdf"); plt.close(fig)
print("figures written to", OUT)
