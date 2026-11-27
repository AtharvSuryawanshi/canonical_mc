"""Cluster lesions, step 1 (reports/IDEAS.md B1): Yang et al. 2019 Fig. 2e on the front points, plus a size control.

One network per point (front run, default seed 0). Groups = the per-task task-variance clusters of slide 19
(fig_tv_ei_fanin.analyse, k_max_I = 20: active units, normalized task variance, k-means, E and I separately).
Lesion a group = its rates held at 0 for the whole trial (fig_lesion_period.simulate; same as Yang's "projection
weights set to 0" and Driscoll's silencing). Outcome = accuracy change on each of the 20 tasks vs the intact network,
same trials (5 eval seeds x 64 noise-free trials per task). Control (Prince et al. 2024): for each group, N_RAND
random groups of the same size from the same population's active units.

    python results/scripts/lesion_clusters.py --point control [--seed 0]      (one network, cached)
    python results/scripts/lesion_clusters.py --plot [--seed 0]               (figure from the caches)

Cache: results/cache/front/lesion_clusters_<point>_s<seed>.npz. Figure: results/front/lesion_clusters_s<seed>.png.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cmc.task import rule_name
from cmc.train_cog import load_checkpoint
from fig_lesion_period import make_eval
from fig_tv_ei_fanin import COLORS, RUNS, analyse
from talk_style import E_COL, I_COL, LABEL

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "results/cache/front"
POINTS = ["control", "firing", "wiring", "both"]
N_EVAL, N_RAND = 5, 10


def path(point, seed):
    return CACHE / f"lesion_clusters_{point}_s{seed}.npz"


def compute(point, seed, threads=3):
    torch.set_num_threads(threads)
    ckpt = RUNS["front"] / point / f"seed_{seed:02d}.pt"
    R = analyse(ckpt, k_max_I=20)
    with contextlib.redirect_stdout(io.StringIO()):
        model, cfg, ck = load_checkpoint(ckpt)
    ck = dict(ck, eval_seeds=list(ck["eval_seeds"])[:N_EVAL])
    acc = make_eval(model, cfg, ck)
    tasks = list(ck["active_tasks"])
    keep = lambda units: torch.ones(model.n_rnn).index_fill_(0, torch.as_tensor(units, dtype=torch.long), 0.0)
    intact = acc(torch.ones(model.n_rnn))[tasks].to_numpy()
    rng = np.random.default_rng([seed, POINTS.index(point)])
    rows, d_les, d_rand = [], [], []
    for pop in "EI":
        P = R[pop]
        for c in range(P["K"]):
            units = P["units"][P["lab"] == c]
            d_les.append(acc(keep(units))[tasks].to_numpy() - intact)
            d_rand.append([acc(keep(rng.choice(P["units"], len(units), replace=False)))[tasks].to_numpy() - intact
                           for _ in range(N_RAND)])
            prof = R["norm"][:, units].mean(1)                                   # mean normalized TV per task
            rows.append(dict(pop=pop, cluster=c, n=len(units), pref=rule_name[tasks[int(prof.argmax())]],
                             profile=prof.round(4).tolist()))
            print(f"  {point} {pop}{c + 1} n={len(units)}: worst task change {d_les[-1].min():+.2f}", flush=True)
    np.savez_compressed(path(point, seed), intact=intact, d_les=np.array(d_les), d_rand=np.array(d_rand),
                        rows=json.dumps(rows), tasks=json.dumps(tasks),
                        n_active=json.dumps({pop: len(R[pop]["units"]) for pop in "EI"}))
    print("->", path(point, seed))


def load(point, seed):
    z = np.load(path(point, seed))
    return dict(intact=z["intact"], d_les=z["d_les"], d_rand=z["d_rand"], rows=json.loads(str(z["rows"])),
                tasks=json.loads(str(z["tasks"])), n_active=json.loads(str(z["n_active"])))


def plot(seed):
    L = {p: load(p, seed) for p in POINTS if path(p, seed).exists()}
    tasks = next(iter(L.values()))["tasks"]
    nmax = max(len(v["rows"]) for v in L.values())
    fig, axes = plt.subplots(1, len(L), figsize=(5.4 * len(L), 0.2 * nmax + 2.8), squeeze=False,
                             gridspec_kw=dict(wspace=0.85))
    for ax, (p, v) in zip(axes[0], L.items()):
        D = v["d_les"]
        # beyond chance: the lesion hurts the task more than every random group of the same size
        beyond = D < v["d_rand"].min(1)
        im = ax.imshow(D, cmap="RdBu", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
        yy, xx = np.nonzero(beyond & (D < -0.1))
        ax.scatter(xx, yy, s=6, c="k", marker="o", lw=0)
        ylab = [f"{r['pop']}{r['cluster'] + 1} (n={r['n']}) {r['pref']}" for r in v["rows"]]
        ax.set_yticks(range(len(ylab)))
        ax.set_yticklabels(ylab, fontsize=6.5)
        for t, r in zip(ax.get_yticklabels(), v["rows"]):
            t.set_color(E_COL if r["pop"] == "E" else I_COL)
        nE = sum(r["pop"] == "E" for r in v["rows"])
        ax.axhline(nE - 0.5, color="k", lw=0.8)
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([rule_name[t] for t in tasks], rotation=90, fontsize=6.5)
        rand = v["d_rand"].mean(1)
        ax.set_title(f"{LABEL[p]}, seed {seed}", fontsize=11, color=COLORS[p], loc="left", pad=30)
        ax.text(0, 1.01, f"{nE} E + {len(v['rows']) - nE} I groups\nmean change: groups {D.mean():+.3f}, "
                f"random same size {rand.mean():+.3f}", transform=ax.transAxes, fontsize=7.5, va="bottom")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.5, pad=0.01, label="accuracy change after lesion")
    fig.suptitle("Silencing each task-variance group (rows; label = size, task with the highest mean normalized TV). "
                 "Dot: drop > 0.1 and larger than all 10 random groups of the same size.", fontsize=8.5, y=0.995)
    out = REPO / "results/front" / f"lesion_clusters_s{seed}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("->", out)
    for p, v in L.items():
        D, Rm = v["d_les"], v["d_rand"]
        hit = (D < -0.1).sum(1)
        beyond = ((D < Rm.min(1)) & (D < -0.1)).sum(1)
        print(f"{p:8s} groups {len(D):2d} | tasks hurt > 0.1 per lesion: median {np.median(hit):.0f} "
              f"(beyond random: {np.median(beyond):.0f}) | lesions hurting no task: {(hit == 0).sum()} | "
              f"random same size, tasks hurt > 0.1: median {np.median((Rm < -0.1).sum(2)):.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", choices=POINTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    if a.point:
        compute(a.point, a.seed, a.threads)
    if a.plot:
        plot(a.seed)
