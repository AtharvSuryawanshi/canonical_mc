"""Fixed points of the Dly Go task, delay -> response input interpolation (Driscoll et al. 2024, Fig. 1), one network.

Network: 5-seed run (fanin, 150k), seed 4, one point (default C0). Task: Dly Go (= Driscoll's MemoryPro), 128
noise-free trials (eval seed 10000). Inputs are constant within a period: u_delay = fixation on + rule, u_go =
rule only. Interpolated input u(a) = (1 - a) u_delay + a u_go, a in 0..1 (21 steps). At each a: start from states
of the end-of-delay trajectories run forward under u(a) plus the previous a's fixed points, find exact fixed points
(cmc.fixed_points: speed minimisation, then the exact ReLU linear solve, deduplicated), and their stability (max |eig|
of the exact Jacobian of the update; < 1 = stable). Readout of a state: fixation output and the ring "bump amplitude"
|sum_k y_k e^{i theta_k}| / sum_k y_k of the sigmoid ring outputs (0 = flat, 1 = all on one angle).
Memory check: continue the delay for extra time, then apply the go input for 25 steps and decode.

    python results/scripts/fig_fp_delaygo.py [--point C0] [--seed 4]

Writes results/front/fp_delaygo_<point>_s<seed>.png and a .npz cache of the fixed points.
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
from cmc.task import generate_trials
from cmc.train_cog import decode_ring, eval_config_from, load_checkpoint, ring_prefs, trial_to_tensors
from fig_tv_ei_fanin import POINTS, RUNS

REPO = Path(__file__).resolve().parents[2]
ALPHAS = np.linspace(0, 1, 21)
EXTRA_DELAY_MS = (0, 2000, 5000, 10000)


def setup(point, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        model, cfg, ck = load_checkpoint(RUNS["5seed"] / point / f"seed_{seed:02d}.pt")
    tr = generate_trials("delaygo", eval_config_from(cfg, 10000, easy_task=True), 128, noise_on=False)
    x, _, _, yloc = trial_to_tensors(tr, "cpu")
    d0, d1 = tr.epochs["delay1"]
    u_delay, u_go = tr.x[d0, 0].astype(float), tr.x[d1, 0].astype(float)
    assert np.allclose(tr.x[d0:d1], u_delay) and np.allclose(tr.x[d1:], u_go)
    with torch.no_grad():
        h, _, _ = model.simulate(x, noise_level=0.0)
    return model, cfg, ck, tr, h.numpy(), yloc[:, -1].numpy(), u_delay, u_go


def main(point="C0", seed=4):
    model, cfg, ck, tr, H, target, u_delay, u_go = setup(point, seed)
    s0, _ = tr.epochs["stim1"]
    d0, d1 = tr.epochs["delay1"]
    prefs = ring_prefs(cfg, "cpu", torch.float32)
    P = prefs.double().numpy()
    W_out, b_out, nE = model.w_out.detach().double().numpy(), model.b_out.detach().double().numpy(), model.n_e
    rng = np.random.default_rng(0)

    def run(h0, u, T):
        inp = torch.as_tensor(np.repeat(np.asarray(u)[None, :, None], len(h0), 0).repeat(T, 2), dtype=torch.float32)
        with torch.no_grad():
            hh, _, yy = model.simulate(inp, x_init=torch.as_tensor(h0, dtype=torch.float32))
        return hh.numpy(), yy

    def readout(Hs):
        y = 1 / (1 + np.exp(-(Hs[:, :nE] @ W_out + b_out)))
        z = (y[:, 1:] * np.exp(1j * P)).sum(1)
        return y[:, 0], np.abs(z) / y[:, 1:].sum(1).clip(1e-9), np.angle(z)

    end = H[:, d1 - 1]
    cache = REPO / "results/front" / f"fp_delaygo_{point}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        FPS, EIG = list(z["fps"]), list(z["eig"])
    else:
        FPS, EIG, prev = [], [], np.zeros((0, H.shape[2]))
        for a in ALPHAS:
            u = (1 - a) * u_delay + a * u_go
            hs = run(end, u, 60)[0][:, ::6].reshape(-1, H.shape[2])
            R = fp.find(model, u, np.r_[hs[rng.choice(len(hs), 400, replace=False)], prev])
            FPS.append(R["h"]); EIG.append(R["max_eig"]); prev = R["h"]
            print(f"a={a:.2f}: {R['n_found']} fixed points, {int((R['max_eig'] < 1).sum())} stable")
        np.savez(cache, fps=np.array(FPS, dtype=object), eig=np.array(EIG, dtype=object))

    # memory retention over an extended delay
    ret = []
    for ms in EXTRA_DELAY_MS:
        T = ms // int(cfg["dt"])
        hT = end if T == 0 else run(end, u_delay, T)[0][:, -1]
        ang = decode_ring(run(hT, u_go, 25)[1], prefs)[:, -1].numpy()
        ret.append((np.abs(np.angle(np.exp(1j * (ang - target)))) < np.pi / 5).mean())

    # axes: top 2 PCs of delay-period states, z = readout direction for cos(angle) (E units only, as Driscoll's W_out axis)
    Dst = H[:, d0:d1].reshape(-1, H.shape[2])
    mu = Dst.mean(0)
    pcs = np.linalg.svd(Dst - mu, full_matrices=False)[2][:2]
    vz = np.r_[W_out[:, 1:] @ np.cos(P), np.zeros(model.n_rnn - nE)]
    vz /= np.linalg.norm(vz)
    proj = lambda X: np.c_[(X - mu) @ pcs.T, X @ vz]
    col = plt.cm.hsv(np.mod(target, 2 * np.pi) / (2 * np.pi))

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.32, wspace=0.55, left=0.05, right=0.95, top=0.9,
                          bottom=0.07)
    go_traj = run(end, u_go, 25)[0]
    for j, (a_i, traj, ttl) in enumerate(((0, H[:, s0:d1], "delay input (a = 0)"),
                                           (len(ALPHAS) - 1, np.concatenate([end[:, None], go_traj], 1),
                                            "response input (a = 1)"))):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        for b in range(0, len(traj), 4):
            p = proj(traj[b])
            ax.plot(*p.T, color=col[b], lw=0.7, alpha=0.8)
            ax.scatter(*p[-1], color=col[b], marker="^", s=10)
        F, e = FPS[a_i], EIG[a_i]
        if len(F):
            p = proj(F)
            st = e < 1
            ax.scatter(*p[st].T, c="k", s=28, depthshade=False, label="stable")
            ax.scatter(*p[~st].T, facecolors="white", edgecolors="k", s=28, depthshade=False, label="unstable")
        ax.set_xlabel("PC1 (delay)", fontsize=7); ax.set_ylabel("PC2 (delay)", fontsize=7)
        ax.set_zlabel("readout axis (cos)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{ttl}: {len(F)} fixed points ({int((e < 1).sum())} stable)", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")

    # bifurcation diagram: angle of each fixed point in the delay PC plane vs a, colour = bump amplitude in the output
    ax = fig.add_subplot(gs[0, 2])
    for a, F, e in zip(ALPHAS, FPS, EIG):
        if not len(F):
            continue
        p = proj(F)
        th = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        amp = readout(F)[1]
        st = e < 1
        sc = ax.scatter(np.full(st.sum(), a), th[st], c=amp[st], cmap="viridis", vmin=0, vmax=1, s=22, edgecolors="k",
                        linewidths=0.4)
        ax.scatter(np.full((~st).sum(), a), th[~st], c=amp[~st], cmap="viridis", vmin=0, vmax=1, s=22, marker="o",
                   edgecolors="none", alpha=0.75)
    plt.colorbar(sc, ax=ax, fraction=0.05).set_label("output bump amplitude", fontsize=7)
    ax.set_xlabel("a   (0 = delay input, 1 = response input)")
    ax.set_ylabel("angle in delay PC plane (deg)")
    ax.set_title("fixed points along the interpolation\n(edged = stable)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 0])
    med = lambda f: [np.median(f(F)) if len(F) else np.nan for F in FPS]
    ax.plot(ALPHAS, med(lambda F: readout(F)[0]), "-o", ms=3, label="fixation output")
    ax.plot(ALPHAS, med(lambda F: readout(F)[1]), "-o", ms=3, label="ring bump amplitude")
    ax.set_xlabel("a"); ax.set_ylim(0, 1)
    ax.set_title("readout at the fixed points (median)", fontsize=9)
    ax.legend(fontsize=7, frameon=False); ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ALPHAS, [len(F) for F in FPS], "-o", ms=3, color="k", label="fixed points")
    ax.plot(ALPHAS, [int((e < 1).sum()) for e in EIG], "-o", ms=3, color="#2ca02c", label="stable")
    ax.set_xlabel("a"); ax.set_ylabel("count")
    ax2 = ax.twinx()
    ax2.plot(ALPHAS, [np.median(e) if len(e) else np.nan for e in EIG], "--", color="#d62728", label="median max|eig|")
    ax2.axhline(1, color="#d62728", lw=0.5, ls=":")
    ax2.set_ylabel("max |eig| (1 = marginal)", color="#d62728")
    ax.set_title("how many, and how stable", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(np.array(EXTRA_DELAY_MS) / 1000, ret, "-o", color="k")
    ax.set_ylim(0, 1.02); ax.set_xlabel("extra delay (s)"); ax.set_ylabel("fraction correct")
    ax.set_title("does the memory hold? (delay extended, then go)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    acc = ck["metrics"].get("acc_delaygo", np.nan) if isinstance(ck["metrics"], dict) else np.nan
    fig.suptitle(f"Dly Go fixed points, delay -> response: {POINTS[point]}, seed {seed} (5-seed run, fanin, 150k)"
                 f"   |   trajectories coloured by stimulus angle", fontsize=11)
    out = REPO / "results/front" / f"fp_delaygo_{point}_s{seed}.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("->", out, "| retention", np.round(ret, 2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", choices=sorted(POINTS), default="C0")
    ap.add_argument("--seed", type=int, default=4)
    a = ap.parse_args()
    main(a.point, a.seed)
