"""Animated GIF of one trial on a trained network: inputs -> RNN -> outputs.

Left: fixation input, stimulus mod 1 / mod 2 rings, rule inputs (time x unit).
Centre: network schematic (input rings, rule row, a sample of recurrent E/I
units with their strongest W_rec edges, output ring) whose nodes light up with
their activity at the current timestep, plus all recurrent units over time.
Right: fixation output and response ring of the network (y_hat).
A vertical playhead sweeps every time axis.

    python scripts/gifs/trial_gif.py
    python scripts/gifs/trial_gif.py --task dmsgo --trial-seed 3
    python scripts/gifs/trial_gif.py --all-tasks

Writes results/trials/<task>.gif.
"""
import argparse
import contextlib
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyArrowPatch

from cmc.task import generate_trials, rule_name
from cmc.train_cog import DALE_DEFAULT_NOISE_LEVEL, eval_config_from, load_checkpoint, trial_to_tensors

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO / "runs" / "moo_zoom" / "dale_all_rateray_woutfrozen_150k" / "control" / "seed_00.pt"
OUT_DIR = REPO / "results" / "trials"
LINE = "#3b4cc0"
CMAP = plt.get_cmap("Purples")
E_COL, I_COL = "#d62728", "#1f77b4"
NODE_EDGE = "#555555"
BG = "#f5f5f5"
EPOCH_NAMES = {"fix": "Fixation", "stim": "Stimulus", "delay": "Delay", "go": "Go / response"}


# --------------------------------------------------------------------------- data
def run_trial(path, task, trial_seed, noise_level, input_noise=True):
    with contextlib.redirect_stdout(io.StringIO()):
        model, config, ckpt = load_checkpoint(path, device="cpu")
    cfg = eval_config_from(config, trial_seed, easy_task=True)
    trial = generate_trials(task, cfg, 1, noise_on=input_noise)
    x, _, _, _ = trial_to_tensors(trial, "cpu")
    torch.manual_seed(trial_seed)
    with torch.no_grad():
        r, _, y_hat = model.simulate(x, noise_level=noise_level)
        w_rec = model.effective_w_rec().cpu().numpy()  # [pre, post]
    sign = model.sign_vector.cpu().numpy() if hasattr(model, "sign_vector") else np.ones(model.n_rnn)
    return dict(
        task=task, t=np.arange(trial.tdim) * config["dt"], x=trial.x[:, 0, :], r=r[0].numpy(),
        y=y_hat[0].numpy(), n_ring=config["n_eachring"], rule_start=config["rule_start"],
        epochs=trial.epochs, tdim=trial.tdim, w_rec=w_rec, sign=sign, dt=config["dt"],
        active_tasks=ckpt.get("active_tasks"),
    )


def epoch_at(epochs, step, tdim):
    for name, (on, off) in epochs.items():
        if (on or 0) <= step < (off if off is not None else tdim):
            base = name.rstrip("0123456789")
            return EPOCH_NAMES.get(base, base) + (f" {name[len(base):]}" if base in ("stim", "delay") else "")
    return ""


def response_onset(epochs):
    for name, (on, _) in epochs.items():
        if name.startswith("go"):
            return on
    return None


# --------------------------------------------------------------------------- side panels
def _style(ax):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


def _trace(ax, t, v, label):
    ax.plot(t, v, color=LINE, lw=1.8)
    ax.set_ylim(-0.1, 1.25)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(label, fontsize=10, pad=3)


def _heat(ax, t, m, label, vmax, ylim=(0, 360), yticks=((0, "0°"), (360, "360°")), cmap=CMAP):
    ax.imshow(m.T, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=vmax,
              extent=[t[0], t[-1], *ylim], interpolation="nearest")
    ax.set_yticks([v for v, _ in yticks])
    ax.set_yticklabels([s for _, s in yticks])
    ax.set_xticks([])
    ax.set_title(label, fontsize=10, pad=3)


# --------------------------------------------------------------------------- network schematic
def ring_xy(cx, cy, rad, n):
    a = np.pi / 2 - 2 * np.pi * np.arange(n) / n  # unit 0 at top, clockwise
    return np.c_[cx + rad * np.cos(a), cy + rad * np.sin(a)]


def blob_xy(cx, cy, rad, n, rng):
    """Sunflower layout inside a disc, lightly jittered."""
    k = np.arange(n) + 0.5
    rr = rad * np.sqrt(k / n)
    th = k * np.pi * (3 - np.sqrt(5)) + rng.uniform(0, 0.3, n)
    return np.c_[cx + rr * np.cos(th), cy + rr * np.sin(th)]


def pick_units(d, n_e_show, n_i_show):
    """Most active-varying E and I units: the ones worth watching."""
    var = d["r"].std(0)
    e_idx = np.where(d["sign"] > 0)[0]
    i_idx = np.where(d["sign"] < 0)[0]
    e = e_idx[np.argsort(var[e_idx])[::-1][:n_e_show]]
    i = i_idx[np.argsort(var[i_idx])[::-1][:n_i_show]]
    return np.r_[e, i]


def strongest_edges(w_rec, units, k):
    sub = np.abs(w_rec[np.ix_(units, units)])
    np.fill_diagonal(sub, 0)
    flat = np.argsort(sub.ravel())[::-1][:k]
    pre, post = np.unravel_index(flat, sub.shape)
    return [(a, b, sub[a, b]) for a, b in zip(pre, post) if sub[a, b] > 0]


def build_network(ax, d, args):
    ax.set_xlim(0, 100)
    ax.set_ylim(-6, 62)
    ax.set_aspect("equal")
    ax.axis("off")
    n = d["n_ring"]
    n_rule = d["x"].shape[1] - d["rule_start"]
    rng = np.random.default_rng(0)
    art = {}

    # --- input side
    fix_in = np.array([[17, 58]])
    m1c, m2c, ring_r, plate_r = (13, 43), (13, 24), 5.2, 7.2
    m1, m2 = ring_xy(*m1c, ring_r, n), ring_xy(*m2c, ring_r, n)
    n_col = int(np.ceil(n_rule / 2))
    rx = 2 + np.arange(n_col) * (24 / max(n_col - 1, 1))
    rule = np.r_[np.c_[rx, np.full(n_col, 9.0)], np.c_[rx, np.full(n_col, 6.3)]][:n_rule]

    # --- recurrent blob
    bc, br = np.array([52.0, 30.0]), 14.0
    units = pick_units(d, args.n_e_show, args.n_i_show)
    pos = blob_xy(*bc, br - 1.8, len(units), rng)
    is_e = d["sign"][units] > 0

    # --- output side
    fix_out = np.array([[86, 48]])
    oc = (88, 27)
    out = ring_xy(*oc, ring_r, n)

    for c in (m1c, m2c, oc):
        ax.add_patch(Circle(c, plate_r, fc="#e2e2e2", ec="#cfcfcf", lw=1, zorder=1))
    ax.add_patch(Circle(bc, br, fc="#dcdcdc", ec="k", lw=1.2, zorder=1))

    # projection lines (alpha/width follow source activity)
    def proj(p0, p1, rad):
        a = FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                            color="k", lw=1.0, alpha=0.4, zorder=0.5)
        ax.add_patch(a)
        return a

    edge_pt = lambda ang: tuple(bc + br * np.array([np.cos(ang), np.sin(ang)]))
    art["proj_in"] = [
        proj(tuple(fix_in[0] + [1.0, 0]), edge_pt(np.deg2rad(125)), -0.3),
        proj((m1c[0] + plate_r, m1c[1]), edge_pt(np.deg2rad(160)), -0.15),
        proj((m2c[0] + plate_r, m2c[1]), edge_pt(np.deg2rad(185)), 0.0),
        proj((26.5, 7.5), edge_pt(np.deg2rad(220)), 0.25),
    ]
    art["proj_out"] = [
        proj(edge_pt(np.deg2rad(20)), tuple(fix_out[0] - [1.0, 0]), 0.2),
        proj(edge_pt(np.deg2rad(-5)), (oc[0] - plate_r, oc[1]), 0.0),
    ]

    # recurrent edges among the shown units
    for a, b, w in strongest_edges(d["w_rec"], units, args.n_edges):
        wmax = max(np.abs(d["w_rec"][np.ix_(units, units)]).max(), 1e-9)
        ax.add_patch(FancyArrowPatch(pos[a], pos[b], connectionstyle="arc3,rad=0.35",
                                     arrowstyle="-|>,head_length=2.2,head_width=1.2",
                                     color=E_COL if is_e[a] else I_COL, lw=0.4 + 1.4 * w / wmax,
                                     alpha=0.25 + 0.45 * w / wmax, shrinkA=3, shrinkB=3, zorder=2))

    kw = dict(edgecolors=NODE_EDGE, linewidths=0.8, zorder=3)
    art["fix_in"] = ax.scatter(*fix_in.T, s=70, **kw)
    art["m1"] = ax.scatter(*m1.T, s=34, **kw)
    art["m2"] = ax.scatter(*m2.T, s=34, **kw)
    art["rule"] = ax.scatter(*rule.T, s=34, **kw)
    art["E"] = ax.scatter(*pos[is_e].T, s=95, marker="^", edgecolors=E_COL, linewidths=1.2, zorder=4)
    art["I"] = ax.scatter(*pos[~is_e].T, s=80, marker="o", edgecolors=I_COL, linewidths=1.2, zorder=4)
    art["fix_out"] = ax.scatter(*fix_out.T, s=70, **kw)
    art["out"] = ax.scatter(*out.T, s=34, **kw)
    art["units_E"], art["units_I"] = units[is_e], units[~is_e]

    txt = dict(ha="center", va="center", fontsize=9)
    for c in (m1c, m2c, oc):
        ax.text(*c, str(n), **txt)
    ax.text(fix_in[0, 0], fix_in[0, 1] + 2.6, "1", **txt)
    ax.text(fix_out[0, 0], fix_out[0, 1] + 2.6, "1", **txt)
    ax.text(14, 2.6, str(n_rule), **txt)
    ax.text(2, m1c[1] + plate_r + 1.5, "mod 1", fontsize=8, color="#666")
    ax.text(2, m2c[1] + plate_r + 1.5, "mod 2", fontsize=8, color="#666")
    ax.text(2, 11.6, "rule", fontsize=8, color="#666")
    ax.plot([2, 98], [-0.5, -0.5], color="k", lw=0.8)
    n_in, n_out, n_rnn = d["x"].shape[1], d["y"].shape[1], d["r"].shape[1]
    ax.text(14, -3.5, f"{n_in} input neurons", **txt)
    ax.text(52, -3.5, f"{n_rnn} recurrent neurons  (showing {len(units)})", **txt)
    ax.text(88, -3.5, f"{n_out} output neurons", **txt)
    ax.scatter([39], [12.5], marker="^", s=30, facecolors="none", edgecolors=E_COL, clip_on=False)
    ax.text(40.5, 12.5, "E", fontsize=8, va="center", color=E_COL)
    ax.scatter([44], [12.5], marker="o", s=26, facecolors="none", edgecolors=I_COL, clip_on=False)
    ax.text(45.5, 12.5, "I", fontsize=8, va="center", color=I_COL)
    return art


# --------------------------------------------------------------------------- figure
def make_gif(d, out, args):
    t, x, y, r, n = d["t"], d["x"], d["y"], d["r"], d["n_ring"]
    stim_max = max(x[:, 1:1 + 2 * n].max(), 1e-6)
    r_max = max(np.percentile(r, 99.5), 1e-6)
    unit_max = np.maximum(r.max(0), 1e-6)  # schematic units: each scaled to its own peak

    fig = plt.figure(figsize=(16, 7.2), facecolor=BG)
    # left column
    L, W = 0.045, 0.15
    ax_fi = fig.add_axes([L, 0.86, W, 0.08])
    ax_m1 = fig.add_axes([L, 0.60, W, 0.2])
    ax_m2 = fig.add_axes([L, 0.34, W, 0.2])
    ax_ru = fig.add_axes([L, 0.10, W, 0.18])
    # centre
    ax_net = fig.add_axes([0.21, 0.02, 0.6, 0.78])
    ax_r = fig.add_axes([0.43, 0.76, 0.16, 0.15])
    # right column
    R = 0.83
    ax_fo = fig.add_axes([R, 0.66, W, 0.08])
    ax_ro = fig.add_axes([R, 0.36, W, 0.24])
    time_axes = [ax_fi, ax_m1, ax_m2, ax_ru, ax_r, ax_fo, ax_ro]
    for ax in time_axes:
        _style(ax)

    _trace(ax_fi, t, x[:, 0], "Fixation input")
    _heat(ax_m1, t, x[:, 1:1 + n], "Stimulus mod 1", stim_max)
    _heat(ax_m2, t, x[:, 1 + n:1 + 2 * n], "Stimulus mod 2", stim_max)
    rules = x[:, d["rule_start"]:]
    _heat(ax_ru, t, rules, "Rule inputs", 1, ylim=(0.5, rules.shape[1] + 0.5),
          yticks=((1, "1"), (rules.shape[1], str(rules.shape[1]))), cmap=plt.get_cmap("Greys"))
    ax_ru.set_xticks([0, t[-1]])
    ax_ru.set_xlabel("Time (ms)", fontsize=9)

    # recurrent units: E then I, each sorted by time of peak activity
    e_idx, i_idx = np.where(d["sign"] > 0)[0], np.where(d["sign"] < 0)[0]
    peak = r.argmax(0)
    order = np.r_[e_idx[np.argsort(peak[e_idx])], i_idx[np.argsort(peak[i_idx])]]
    _heat(ax_r, t, r[:, order], "Recurrent units", r_max, ylim=(1, r.shape[1]),
          yticks=((1, "1"), (r.shape[1], str(r.shape[1]))))
    ax_r.axhline(len(e_idx) + 0.5, color="k", lw=0.8)
    ax_r.text(t[-1], len(e_idx) / 2, " E", va="center", fontsize=8, color=E_COL)
    ax_r.text(t[-1], (len(e_idx) + r.shape[1]) / 2, " I", va="center", fontsize=8, color=I_COL)

    _trace(ax_fo, t, y[:, 0], "Fixation output")
    _heat(ax_ro, t, y[:, 1:], "Response", 1)
    ax_ro.set_xticks([0, t[-1]])
    ax_ro.set_xlabel("Time (ms)", fontsize=9)
    go = response_onset(d["epochs"])
    if go is not None:
        for ax in (ax_fo, ax_ro):
            ax.axvspan(t[go], t[-1], color="#f0c040", alpha=0.12, lw=0)

    art = build_network(ax_net, d, args)
    title = fig.text(0.21, 0.975, "", ha="left", va="top", fontsize=13)
    fig.text(0.985, 0.04, f"{rule_name.get(d['task'], d['task'])}", ha="right", fontsize=11,
             color="#444")
    heads = [ax.axvline(t[0], color="#e4572e", lw=1.6, zorder=10) for ax in time_axes]

    col = lambda v, vmax: CMAP(np.clip(0.08 + 0.92 * np.asarray(v) / vmax, 0, 1))
    frames = list(range(0, d["tdim"], args.stride))
    if frames[-1] != d["tdim"] - 1:
        frames.append(d["tdim"] - 1)

    def update(k):
        xt, yt, rt = x[k], y[k], r[k]
        for h in heads:
            h.set_xdata([t[k], t[k]])
        art["fix_in"].set_facecolor(col([xt[0]], 1))
        art["m1"].set_facecolor(col(xt[1:1 + n], stim_max))
        art["m2"].set_facecolor(col(xt[1 + n:1 + 2 * n], stim_max))
        art["rule"].set_facecolor(col(xt[d["rule_start"]:], 1))
        art["E"].set_facecolor(col(rt[art["units_E"]] / unit_max[art["units_E"]], 1))
        art["I"].set_facecolor(col(rt[art["units_I"]] / unit_max[art["units_I"]], 1))
        art["fix_out"].set_facecolor(col([yt[0]], 1))
        art["out"].set_facecolor(col(yt[1:], 1))
        src_in = [xt[0], xt[1:1 + n].max() / stim_max, xt[1 + n:1 + 2 * n].max() / stim_max, 1.0]
        for p, s in zip(art["proj_in"], src_in):
            p.set_alpha(0.15 + 0.75 * np.clip(s, 0, 1))
            p.set_linewidth(0.6 + 1.6 * np.clip(s, 0, 1))
        for p, s in zip(art["proj_out"], [yt[0], yt[1:].max()]):
            p.set_alpha(0.15 + 0.75 * np.clip(s, 0, 1))
            p.set_linewidth(0.6 + 1.6 * np.clip(s, 0, 1))
        title.set_text(f"{rule_name.get(d['task'], d['task'])}   ·   t = {t[k]:.0f} ms   ·   "
                       f"{epoch_at(d['epochs'], k, d['tdim'])}")
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=args.fps), dpi=args.dpi,
              savefig_kwargs=dict(facecolor=BG))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--task", default="delaydm1")
    ap.add_argument("--all-tasks", action="store_true", help="One GIF per task the network was trained on.")
    ap.add_argument("--trial-seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=DALE_DEFAULT_NOISE_LEVEL,
                    help="Recurrent noise level during simulation.")
    ap.add_argument("--no-input-noise", action="store_true")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=1, help="Timesteps per frame.")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--n-e-show", type=int, default=16, help="E units drawn in the schematic.")
    ap.add_argument("--n-i-show", type=int, default=6, help="I units drawn in the schematic.")
    ap.add_argument("--n-edges", type=int, default=30, help="Strongest W_rec edges drawn among them.")
    ap.add_argument("--out", type=Path, default=None, help=f"Default: {OUT_DIR}/<task>.gif")
    args = ap.parse_args()

    tasks = [args.task]
    if args.all_tasks:
        tasks = run_trial(args.checkpoint, args.task, 0, 0.0)["active_tasks"]
    for task in tasks:
        d = run_trial(args.checkpoint, task, args.trial_seed, args.noise, not args.no_input_noise)
        out = args.out if (args.out and not args.all_tasks) else OUT_DIR / f"{task}.gif"
        make_gif(d, out, args)
        print(f"{task}: {d['tdim']} steps -> {out}")


if __name__ == "__main__":
    main()
