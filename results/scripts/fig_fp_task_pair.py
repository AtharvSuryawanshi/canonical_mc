"""Rule-input interpolation between two tasks, per task period (Driscoll et al. 2024, Figs. 2 and 4), one network.

Pair 1 (shared motifs expected): Dly Go vs Dly Anti (= Driscoll's MemoryPro vs MemoryAnti). Matched inputs: one Dly Go
batch, and the same array with only the rule one-hot switched to Dly Anti (same stimuli, same timing).
Pair 2 (control, different motifs expected): Dly Anti vs RT Go (= Driscoll's DelayAnti vs ReactPro), stimulus period
only. Synthetic matched trials: a context period (fixation + rule) of the Dly Anti trial's length, then 25 steps of
stimulus; Dly Anti keeps fixation on, RT Go turns it off (its stimulus period is its response period).

Interpolated input x(a) = (1 - a) x_A + a x_B over the whole trial, a in 0..1 (21 steps). For each a the trial is
simulated; in each period the end-of-period state of every trial is relaxed (cmc.fixed_points.relax) to its fixed or
slow point under that period's input: the "relevant" point (Driscoll: the fixed point closest to the end of the
trajectory). A smooth path of the relevant point along a, with unchanged stability, = shared motif; a jump = a
bifurcation, separate motifs. Subspace overlap (Driscoll Fig. 4d,g): variance of task B's states in a period explained
by task A's top PCs vs B's own.

    python results/scripts/fig_fp_task_pair.py [--point C0] [--seed 4]

Writes results/front/fp_pair_<point>_s<seed>.png and a .npz cache.
"""
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cmc import fixed_points as fp
from cmc.task import generate_trials, get_rule_index
from cmc.train_cog import decode_ring, eval_config_from, load_checkpoint, ring_prefs
from fig_tv_ei_fanin import POINTS, RUNS

REPO = Path(__file__).resolve().parents[2]
ALPHAS = np.linspace(0, 1, 21)
N_TRIALS, N_STIM_ANGLES, STIM_STEPS = 16, 8, 25


def sim(model, x):
    """x (T, B, n_in) -> states (B, T, N), outputs (B, T, n_out)."""
    with torch.no_grad():
        h, _, y = model.simulate(torch.as_tensor(np.transpose(x, (1, 2, 0)), dtype=torch.float32), noise_level=0.0)
    return h.numpy(), y


def build(cfg):
    tr = generate_trials("delaygo", eval_config_from(cfg, 10000, easy_task=True), 512, noise_on=False)
    stim = np.mod(tr.y_loc[-1], 2 * np.pi)                                  # Dly Go target = stimulus angle
    pick = [int(np.argmin(np.abs(np.angle(np.exp(1j * (stim - t)))))) for t in np.linspace(0, 2 * np.pi, N_TRIALS,
                                                                                               endpoint=False)]
    xA = tr.x[:, pick].astype(float)
    stim = stim[pick]
    rA, rDA, rRT = (get_rule_index(r, cfg) for r in ("delaygo", "delayanti", "reactgo"))
    xB = xA.copy(); xB[:, :, rA] = 0; xB[:, :, rDA] = 1
    pair1 = dict(A="Dly Go", B="Dly Anti", xA=xA, xB=xB, stim=stim,
                 periods={"stimulus": tr.epochs["stim1"], "delay": tr.epochs["delay1"], "response": (tr.epochs["go1"][0],
                                                                                                    xA.shape[0])})
    s0 = tr.epochs["stim1"][0]
    cA = np.r_[xB[:s0], np.repeat(xB[s0:s0 + 1], STIM_STEPS, 0)]          # Dly Anti: context, then stimulus, fixation on
    cB = cA.copy(); cB[:, :, rDA] = 0; cB[:, :, rRT] = 1; cB[s0:, :, 0] = 0  # RT Go: same, fixation off at stimulus
    pair2 = dict(A="Dly Anti", B="RT Go", xA=cA, xB=cB, stim=stim, periods={"stimulus": (s0, s0 + STIM_STEPS)})
    return pair1, pair2


def analyse(model, P):
    """Relevant point per trial, per period, per a."""
    res = {k: dict(h=[], exact=[], speed=[], max_eig=[]) for k in P["periods"]}
    states = {}
    for a in ALPHAS:
        x = (1 - a) * P["xA"] + a * P["xB"]
        H, _ = sim(model, x)
        states[a] = H
        for k, (s, e) in P["periods"].items():
            ends = H[:, e - 1]
            same = np.allclose(x[s], x[s, :1])
            groups = [np.arange(len(ends))] if same else [[b] for b in range(0, len(ends), N_TRIALS // N_STIM_ANGLES)]
            out = dict(h=np.full_like(ends, np.nan), exact=np.zeros(len(ends), bool), speed=np.full(len(ends), np.nan),
                       max_eig=np.full(len(ends), np.nan))
            for g in groups:
                R = fp.relax(model, x[s, g[0]], ends[g])
                for j, b in enumerate(g):
                    for f in out:
                        out[f][b] = R[f][j]
            for f in out:
                res[k][f].append(out[f])
    return {k: {f: np.array(v) for f, v in r.items()} for k, r in res.items()}, states


def accuracy(model, cfg, x, target, t):
    _, y = sim(model, x)
    ang = decode_ring(y, ring_prefs(cfg, "cpu", torch.float32))[:, t].numpy()
    return (np.abs(np.angle(np.exp(1j * (ang - target)))) < np.pi / 5).mean()


def pcs(X, k=10):
    mu = X.mean(0)
    return mu, np.linalg.svd(X - mu, full_matrices=False)[2][:k]


def overlap(XA, XB, k=10):
    """Fraction of B's variance explained by A's top 1..k PCs and by B's own."""
    _, VA = pcs(XA, k)
    muB, VB = pcs(XB, k)
    tot = ((XB - muB) ** 2).sum()
    f = lambda V: [(((XB - muB) @ V[:i].T) ** 2).sum() / tot for i in range(1, k + 1)]
    return np.array(f(VA)), np.array(f(VB))


def main(point="C0", seed=4):
    with contextlib.redirect_stdout(io.StringIO()):
        model, cfg, ck = load_checkpoint(RUNS["5seed"] / point / f"seed_{seed:02d}.pt")
    pair1, pair2 = build(cfg)
    T = pair1["xA"].shape[0]
    print(f"{point}: accuracy Dly Go {accuracy(model, cfg, pair1['xA'], pair1['stim'], T - 1):.2f}, "
          f"Dly Anti (rule swapped) {accuracy(model, cfg, pair1['xB'], pair1['stim'] + np.pi, T - 1):.2f}, "
          f"RT Go (synthetic) {accuracy(model, cfg, pair2['xB'], pair2['stim'], pair2['xB'].shape[0] - 1):.2f}")
    cache = REPO / "results/front" / f"fp_pair_{point}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        R1, R2 = z["R1"].item(), z["R2"].item()
        S1 = {a: sim(model, (1 - a) * pair1["xA"] + a * pair1["xB"])[0] for a in (0.0, 1.0)}
        S2 = {a: sim(model, (1 - a) * pair2["xA"] + a * pair2["xB"])[0] for a in (0.0, 1.0)}
    else:
        R1, S1 = analyse(model, pair1)
        R2, S2 = analyse(model, pair2)
        np.savez(cache, R1=np.array(R1, dtype=object), R2=np.array(R2, dtype=object))

    rows = [(pair1, R1, S1, k) for k in pair1["periods"]] + [(pair2, R2, S2, "stimulus")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 4.1 * len(rows)),
                             gridspec_kw=dict(hspace=0.45, wspace=0.32, left=0.05, right=0.97, top=0.93, bottom=0.04))
    summary = []
    for r, (P, R, S, k) in enumerate(rows):
        s, e = P["periods"][k]
        rr = R[k]
        XA, XB = S[0.0][:, s:e].reshape(-1, S[0.0].shape[2]), S[1.0][:, s:e].reshape(-1, S[1.0].shape[2])
        mu, V = pcs(XA, 2)
        pr = lambda X: (X - mu) @ V.T
        col = plt.cm.hsv(np.mod(P["stim"], 2 * np.pi) / (2 * np.pi))
        valid = np.flatnonzero(np.isfinite(rr["speed"][0]))
        name = f"{P['A']} vs {P['B']}, {k}"

        ax = axes[r, 0]
        for b in valid:
            ax.plot(*pr(S[0.0][b, s:e]).T, color=col[b], lw=0.9)
            ax.plot(*pr(S[1.0][b, s:e]).T, color=col[b], lw=0.9, ls="--")
        for ai, mk in ((0, "o"), (-1, "s")):
            p = pr(rr["h"][ai][valid])
            st = rr["max_eig"][ai][valid] < 1
            ax.scatter(*p[st].T, marker=mk, c="k", s=30, zorder=3)
            ax.scatter(*p[~st].T, marker=mk, facecolors="white", edgecolors="k", s=30, zorder=3)
        ax.set_title(f"{name}\nsolid / o = {P['A']}, dashed / square = {P['B']} (filled = stable)", fontsize=8,
                     loc="left")
        ax.set_xlabel(f"PC1 ({P['A']})", fontsize=7); ax.set_ylabel("PC2", fontsize=7); ax.tick_params(labelsize=6)

        ax = axes[r, 1]
        for b in valid:
            y = pr(rr["h"][:, b])[:, 0]
            st = rr["max_eig"][:, b] < 1
            ax.plot(ALPHAS, y, color=col[b], lw=0.6, alpha=0.6)
            ax.scatter(ALPHAS[st], y[st], color=col[b], s=12)
            ax.scatter(ALPHAS[~st], y[~st], facecolors="white", edgecolors=col[b], s=12)
        ax.set_xlabel(f"a  (0 = {P['A']}, 1 = {P['B']})", fontsize=7); ax.set_ylabel(f"relevant point, PC1", fontsize=7)
        ax.set_title("relevant fixed / slow point along a (one line per stimulus)", fontsize=8, loc="left")
        ax.tick_params(labelsize=6)

        ax = axes[r, 2]
        d = np.linalg.norm(np.diff(rr["h"][:, valid], axis=0), axis=2)
        scale = np.median(np.linalg.norm(rr["h"][:, valid], axis=2))
        ax.plot(ALPHAS[1:], np.median(d, 1) / scale, "-o", ms=3, color="k", label="median over stimuli")
        ax.plot(ALPHAS[1:], d.max(1) / scale, "-", color="0.6", label="max over stimuli")
        ax.set_xlabel("a", fontsize=7); ax.set_ylabel("step between consecutive a / |h|", fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(ALPHAS, np.median(rr["max_eig"][:, valid], 1), "--", color="#d62728", label="median max|eig|")
        ax2.axhline(1, color="#d62728", lw=0.5, ls=":")
        ax2.tick_params(labelsize=6, colors="#d62728")
        ax.set_title(f"jumps and stability   (exact fixed points: {rr['exact'][:, valid].mean():.0%})", fontsize=8,
                     loc="left")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False); ax.tick_params(labelsize=6)

        ax = axes[r, 3]
        fA, fB = overlap(XA, XB)
        ks = np.arange(1, 11)
        ax.plot(ks, fB, "-o", ms=3, color="k", label=f"{P['B']}'s own PCs")
        ax.plot(ks, fA, "-o", ms=3, color="#7b3294", label=f"{P['A']}'s PCs")
        ax.set_ylim(0, 1.02); ax.set_xlabel("number of PCs", fontsize=7)
        ax.set_ylabel(f"fraction of {P['B']} variance", fontsize=7)
        ax.set_title("subspace overlap", fontsize=8, loc="left"); ax.legend(fontsize=6, frameon=False)
        ax.tick_params(labelsize=6)
        summary.append((name, float(np.median(d, 1).max() / scale), float(d.max() / scale), fA[1], fB[1],
                        float(np.median(rr["max_eig"][0, valid])), float(np.median(rr["max_eig"][-1, valid])),
                        float(rr["exact"][:, valid].mean())))
    fig.suptitle(f"Rule interpolation between tasks, per period: {POINTS[point]}, seed {seed} (5-seed run, fanin, 150k)"
                 f"   |   colour = stimulus angle", fontsize=11)
    out = REPO / "results/front" / f"fp_pair_{point}_s{seed}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("->", out)
    print("row | biggest median step/|h| | biggest single step/|h| | B var in A's top-2 PCs vs own | median max|eig| "
          "at a=0, a=1 | exact")
    for n, dm, dx, fa, fb, e0, e1, ex in summary:
        print(f"{n:32s} {dm:.3f}  {dx:.3f}   {fa:.2f} vs {fb:.2f}   {e0:.3f} {e1:.3f}   {ex:.0%}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", choices=sorted(POINTS), default="C0")
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    main(a.point, a.seed)
