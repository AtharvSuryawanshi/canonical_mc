"""Like-to-like wiring without clusters: are neurons that work in the same task periods more often connected?

Per network (5-seed run, fanin, 150k; default seed 4, all four points):
  fingerprint  = each active neuron's per-period task variance (fig_tv_period.py cache), divided by its max
  similarity   = Pearson correlation of the two fingerprints, for every ordered pair pre -> post (no self-pairs)
  connected    = E/I-normalised weight |W| / c_pre > 0.01 (the threshold of train_cog.connection_fraction)
  strength     = that normalised weight, among connected pairs
Per type (E->E, E->I, I->E, I->I): P(connected) in 10 similarity bins; AUC = P(a connected pair is more similar than
an unconnected one), 0.5 = no relation; Spearman(similarity, strength) among connected pairs. Null: fingerprints
shuffled within E and within I (wiring unchanged), 500 times. Direction: each neuron's main period type (stimulus,
memory = delay, response; largest mean normalised variance), P(connected) from type to type for E->E, as observed /
shuffled.

    python results/scripts/fig_wiring_like2like.py [--seed 4]

Writes results/front/wiring_like2like_s<seed>.png and prints the summary.
"""
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from fig_tv_ei_fanin import POINTS
from fig_tv_period import load

REPO = Path(__file__).resolve().parents[2]
THRESH, N_SHUFFLE, N_BINS = 1e-2, 500, 10
TYPES = {"stimulus": ("stim1", "stim2"), "memory": ("delay1", "delay2"), "response": ("go1",)}
QUADS = ("E→E", "E→I", "I→E", "I→I")


def auc(sim, conn):
    """Mann-Whitney AUC of similarity for connected vs unconnected pairs."""
    n1, n0 = conn.sum(), (~conn).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(sim)
    return (r[conn].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def network(point, seed, rng, keep_frac=None, run="5seed", thresh=THRESH, n_shuffle=N_SHUFFLE):
    """keep_frac: if set, 'connected' = the strongest keep_frac of the (active, off-diagonal) synapses instead of
    |W|/c > thresh, to compare networks at matched sparsity. run: key of fig_tv_ei_fanin.RUNS."""
    with contextlib.redirect_stdout(io.StringIO()):
        model, ck, tv_t, tv_p, keys = load(point, seed, run)
    total = tv_t.sum(0)
    units = np.flatnonzero(total > 1e-2 * total.max())
    F = tv_p[:, units] / tv_p[:, units].max(0, keepdims=True)              # (rows, units)
    S = np.corrcoef(F.T)                                                     # similarity, units x units
    Wn = np.abs(model.effective_w_rec().detach().numpy()) / model.ei_row_scale().detach().numpy()[:, None]
    W = Wn[np.ix_(units, units)]                                             # W[pre, post]
    isE = units < model.n_e
    off = ~np.eye(len(units), dtype=bool)
    C = W > (thresh if keep_frac is None else np.quantile(W[off], 1 - keep_frac))
    masks = {q: off & np.outer(isE if q[0] == "E" else ~isE, isE if q[-1] == "E" else ~isE) for q in QUADS}
    ptype = np.array([[F[[i for i, (_, p) in enumerate(keys) if p in ps]].mean(0) for ps in TYPES.values()]]).squeeze(0)
    ptype = ptype.argmax(0)                                                  # main period type per unit

    def stats(Sx, Tx):
        out = {}
        for q, mk in masks.items():
            s, c = Sx[mk], C[mk]
            out[q] = dict(auc=auc(s, c), rho=spearmanr(s[c], W[mk][c])[0] if c.sum() > 3 else np.nan)
        E = masks["E→E"]
        out["dir"] = np.array([[C[E & (Tx[:, None] == a) & (Tx[None, :] == b)].mean() if (E & (Tx[:, None] == a)
                                & (Tx[None, :] == b)).any() else np.nan for b in range(3)] for a in range(3)])
        return out

    obs = stats(S, ptype)
    edges = {q: np.quantile(S[mk], np.linspace(0, 1, N_BINS + 1)) for q, mk in masks.items()}
    curve = lambda Sx, q: np.array([C[masks[q]][(Sx[masks[q]] >= lo) & (Sx[masks[q]] <= hi)].mean()
                                    for lo, hi in zip(edges[q][:-1], edges[q][1:])])
    curves = {q: curve(S, q) for q in QUADS}
    null_c, null_s = {q: [] for q in QUADS}, []
    iE, iI = np.flatnonzero(isE), np.flatnonzero(~isE)
    for _ in range(n_shuffle):
        perm = np.arange(len(units))
        perm[iE] = rng.permutation(iE); perm[iI] = rng.permutation(iI)
        Sp, Tp = S[np.ix_(perm, perm)], ptype[perm]
        null_s.append(stats(Sp, Tp))
        for q in QUADS:
            null_c[q].append(curve(Sp, q))
    res = dict(obs=obs, curves=curves, edges=edges, n_units=len(units), isE=isE, conn=C[off].mean(), S=S, C=C, masks=masks,
               n_conn={q: int(C[masks[q]].sum()) for q in QUADS}, types=np.bincount(ptype[isE], minlength=3))
    for q in QUADS:
        nul = np.array([n[q]["auc"] for n in null_s])
        res[q] = dict(auc=obs[q]["auc"], auc_null=np.nanmean(nul), auc_p=(np.sum(np.abs(nul - 0.5) >= abs(obs[q]["auc"] - 0.5)) + 1) / (len(nul) + 1),
                      rho=obs[q]["rho"], rho_null=np.nanmean([n[q]["rho"] for n in null_s]),
                      band=np.nanpercentile(np.array(null_c[q]), [2.5, 50, 97.5], axis=0))
    res["dir_ratio"] = obs["dir"] / np.nanmean([n["dir"] for n in null_s], axis=0)
    return res


def main(seed=4):
    rng = np.random.default_rng(0)
    R = {p: network(p, seed, rng) for p in POINTS}
    fig, axes = plt.subplots(len(R), 5, figsize=(19, 3.4 * len(R)),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.9], hspace=0.55, wspace=0.35, left=0.05,
                                              right=0.98, top=0.92, bottom=0.05))
    print("point | type | n connected | AUC (null) p | strength rho (null)")
    for r, (p, res) in enumerate(R.items()):
        for c, q in enumerate(QUADS):
            ax = axes[r, c]
            e = res["edges"][q]
            x = (e[:-1] + e[1:]) / 2
            lo, md, hi = res[q]["band"]
            ax.fill_between(x, lo, hi, color="0.85", label="shuffled (95%)")
            ax.plot(x, md, color="0.5", lw=1)
            ax.plot(x, res["curves"][q], "-o", ms=3, color="#1F8AD0" if q[0] == "E" else "#E0393E", label="observed")
            ax.set_title(f"{q}: AUC {res[q]['auc']:.2f} (p {res[q]['auc_p']:.3f}), strength ρ {res[q]['rho']:+.2f}\n"
                         f"{res['n_conn'][q]} connected pairs", fontsize=8, loc="left")
            ax.set_xlabel("fingerprint similarity (pre, post)", fontsize=7)
            if c == 0:
                ax.set_ylabel(f"{POINTS[p]}\nP(connected)", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, frameon=False)
            print(f"{p:8s} {q} {res['n_conn'][q]:6d}   {res[q]['auc']:.3f} ({res[q]['auc_null']:.3f}) p={res[q]['auc_p']:.3f}"
                  f"   {res[q]['rho']:+.2f} ({res[q]['rho_null']:+.2f})")
        ax = axes[r, 4]
        M = np.log2(res["dir_ratio"])
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        for (i, j), v in np.ndenumerate(res["dir_ratio"]):
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "–", ha="center", va="center", fontsize=8)
        names = [f"{k}\n({n})" for k, n in zip(TYPES, res["types"])]
        ax.set_xticks(range(3)); ax.set_xticklabels(list(TYPES), fontsize=7)
        ax.set_yticks(range(3)); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("post (E)", fontsize=7); ax.set_ylabel("pre (E), n neurons", fontsize=7)
        ax.set_title("E→E P(connected), observed / shuffled", fontsize=8, loc="left")
        print(f"{p:8s} E->E direction (rows pre stim/mem/resp, cols post), obs/shuffled:\n", np.round(res["dir_ratio"], 2),
              " E neurons per type:", res["types"], f" overall connected {res['conn']:.3f}")
    fig.suptitle(f"Are neurons that work in the same periods more connected? Seed {seed} (5-seed run, fanin, 150k). "
                 f"Connected = |W|/c > {THRESH}; grey = fingerprints shuffled within E and within I", fontsize=10)
    out = REPO / "results/front" / f"wiring_like2like_s{seed}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("->", out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4)
    main(ap.parse_args().seed)
