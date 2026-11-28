"""Task-group lesions at equal size, two lesion types (reports/IDEAS.md B1, step 1b).

One network per point (front run, default seed 0). For each task a, the group G_a = the M E neurons with the largest
raw task variance on a (checkpoint "task_variance", Yang et al. 2019 definition): the neurons most involved in a,
specialists or generalists. Same M in every network, no clustering, no k. Control: N_RAND random groups of M active
E neurons (shared by all tasks and both lesion types).

Two lesion types, the group's rates held at
  zero   0 at every time step (Yang 2019, Driscoll 2024): removes the neurons' information AND their mean drive
  mean   their intact trial-average at each time step of that task, r_i(t) averaged over the trials of the same
         evaluation batch: removes only the trial-to-trial variation (= what task variance measures), keeps the drive
Outcome: accuracy change on each of the 20 tasks vs the intact network, same trials (N_EVAL eval seeds x 64
noise-free trials per task; fig_lesion_period.make_eval's trials).

    python results/scripts/lesion_taskgroups.py --point control [--seed 0]    (one network, cached)
    python results/scripts/lesion_taskgroups.py --plot [--seed 0]             (figure + summary from the caches)
    --m 2,5,20,40 --modes mean       dose-response sizes; --dose [--seeds 0-4] for the figure
    --pop I --m 1,2,5,10             the same with the top-m I neurons per task (random groups from the active I);
                                     cache tag _I, figure lesion_dose_s<seed>_I.png

Cache: results/cache/front/lesion_taskgroups_<point>_s<seed>.npz. Figure: results/front/lesion_taskgroups_s<seed>.png.
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

from cmc.task import generate_trials, rule_name
from cmc.train_cog import batch_accuracy, eval_config_from, load_checkpoint, response_mask, ring_prefs, trial_to_tensors
from fig_tv_ei_fanin import COLORS, RUNS
from fig_tv_period import BATCH
from talk_style import E_COL, I_COL, LABEL

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "results/cache/front"
POINTS = ["control", "firing", "wiring", "both"]
M, N_RAND, N_EVAL = 10, 20, 5
MODES = ("zero", "mean")


def path(point, seed, m=M, pop="E"):
    tag = ("" if m == M else f"_m{m}") + ("" if pop == "E" else f"_{pop}")   # E, m = 10: the original caches
    return CACHE / f"lesion_taskgroups_{point}_s{seed}{tag}.npz"


def run(model, x, units=None, mode=None, mean=None):
    """Noise-free DaleRNN dynamics (as DaleRNN.simulate / fig_lesion_period.simulate); the rates of `units` are held
    at 0 (mode "zero") or at mean[t] (mode "mean") at every step. Returns outputs (B, T, n_out) and rates (B, T, N)."""
    x = model._normalize_input(x)
    h = torch.zeros(x.shape[0], model.n_rnn)
    W = model.effective_w_rec()
    keep = torch.ones(model.n_rnn)
    if units is not None:
        keep[torch.as_tensor(units, dtype=torch.long)] = 0.0
    ys, hs = [], []
    for t in range(x.shape[2]):
        h = (1 - model.alpha) * h + model.alpha * model._cell_act(x[:, :, t] @ model.w_in + h @ W + model.bias)
        if units is not None:
            h = h * keep + ((1 - keep) * mean[t] if mode == "mean" else 0.0)
        hs.append(h)
        ys.append(torch.sigmoid(h[:, :model.n_e] @ model.w_out + model.b_out))
    return torch.stack(ys, 1), torch.stack(hs, 1)


@torch.no_grad()
def compute(point, seed, threads=3, m=M, modes=MODES, pop="E"):
    torch.set_num_threads(threads)
    with contextlib.redirect_stdout(io.StringIO()):
        model, cfg, ck = load_checkpoint(RUNS["front"] / point / f"seed_{seed:02d}.pt")
    tasks = list(ck["active_tasks"])
    tv = np.asarray(ck["task_variance"], float)                                  # (tasks, N), raw
    ne = model.n_e
    lo = 0 if pop == "E" else ne                                                 # population's first index
    hi = ne if pop == "E" else model.n_rnn
    groups = [lo + np.argsort(-tv[a, lo:hi])[:m] for a in range(len(tasks))]     # top-m neurons of pop per task
    total = tv.sum(0)
    active = lo + np.flatnonzero((total > 1e-2 * total.max())[lo:hi])
    rng = np.random.default_rng([seed, POINTS.index(point)] + ([] if pop == "E" else [1]))
    rand = [rng.choice(active, m, replace=False) for _ in range(N_RAND)]
    # trials, intact outputs and intact trial-average rates per (task, eval batch)
    B = []
    for rule in tasks:
        for s in list(ck["eval_seeds"])[:N_EVAL]:
            c = eval_config_from(cfg, s, easy_task=True)
            tr = generate_trials(rule, c, BATCH, noise_on=False)
            x, _, _, y_loc = trial_to_tensors(tr, "cpu")
            y, h = run(model, x)
            B.append(dict(rule=rule, x=x, y_loc=y_loc, mask=response_mask(tr, "cpu"),
                          prefs=ring_prefs(c, "cpu", torch.float32), mean=h.mean(0)))

    def acc(units=None, mode=None):
        out = {}
        for b in B:
            y, _ = run(model, b["x"], units, mode, b["mean"])
            out.setdefault(b["rule"], []).append(batch_accuracy(y, b["y_loc"], b["prefs"], resp_mask=b["mask"])["acc"])
        return np.array([np.mean(out[r]) for r in tasks])

    intact = acc()
    res = {}
    for mode in modes:
        res[f"d_{mode}"] = np.array([acc(g, mode) - intact for g in groups])
        res[f"r_{mode}"] = np.array([acc(g, mode) - intact for g in rand])
        print(f"  {point} {mode}: mean change groups {res[f'd_{mode}'].mean():+.3f}, random {res[f'r_{mode}'].mean():+.3f}",
              flush=True)
    np.savez_compressed(path(point, seed, m, pop), intact=intact, groups=np.array(groups), rand=np.array(rand),
                        tasks=json.dumps(tasks), tv=tv, **res)
    print("->", path(point, seed, m, pop), flush=True)


def load(point, seed, m=M, pop="E"):
    z = np.load(path(point, seed, m, pop))
    return {k: (json.loads(str(z[k])) if k == "tasks" else z[k]) for k in z.files}


def summary(v, mode):
    """Damage d = max(0, -change). Per group a: own-task damage d_aa, mean damage on the 19 other tasks, number of
    tasks hurt > 0.1, whether the most-hurt task is a; beyond random = more than all N_RAND random groups."""
    D = np.clip(-v[f"d_{mode}"], 0, None)
    Rd = np.clip(-v[f"r_{mode}"], 0, None)
    n = len(D)
    off = ~np.eye(n, dtype=bool)
    own = np.diag(D)
    other = np.array([D[a][off[a]].mean() for a in range(n)])
    hurt = (D > 0.1).sum(1)
    top_own = (D.argmax(1) == np.arange(n)) & (D.max(1) > 0.1)
    beyond_own = own > Rd.max(0)                                                   # vs every random group, same task
    return dict(own=own.mean(), other=other.mean(), hurt=np.median(hurt), none=int((hurt == 0).sum()),
                top_own=int(top_own.sum()), beyond_own=int((beyond_own & (own > 0.1)).sum()),
                rand=Rd.mean(), rand_hurt=np.median((Rd > 0.1).sum(1)))


def plot(seed):
    L = {p: load(p, seed) for p in POINTS if path(p, seed).exists()}
    tasks = next(iter(L.values()))["tasks"]
    names = [rule_name[t] for t in tasks]
    fig, axes = plt.subplots(2, len(L), figsize=(4.3 * len(L), 9.4), squeeze=False,
                             gridspec_kw=dict(wspace=0.08, hspace=0.32))
    for c, (p, v) in enumerate(L.items()):
        for r, mode in enumerate(MODES):
            ax = axes[r, c]
            Dm = v[f"d_{mode}"]
            im = ax.imshow(Dm, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
            beyond = (Dm < v[f"r_{mode}"].min(0)) & (Dm < -0.1)
            yy, xx = np.nonzero(beyond)
            ax.scatter(xx, yy, s=7, c="k", lw=0)
            ax.plot([-0.5, len(tasks) - 0.5], [-0.5, len(tasks) - 0.5], color="0.6", lw=0.6, ls=":")
            ax.set_xticks(range(len(tasks)))
            ax.set_xticklabels(names, rotation=90, fontsize=6.5)
            ax.set_yticks(range(len(tasks)))
            ax.set_yticklabels([f"{n} group" for n in names] if c == 0 else [], fontsize=6.5)
            S = summary(v, mode)
            ax.set_title(f"{LABEL[p]}  ·  {'silenced (r = 0)' if mode == 'zero' else 'mean-clamped'}\n"
                         f"own task {-S['own']:+.2f}, other tasks {-S['other']:+.2f}, random {-S['rand']:+.2f}",
                         fontsize=8, color=COLORS[p], loc="left")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.4, pad=0.01, label="accuracy change after lesion")
    fig.suptitle(f"Lesioning the {M} E neurons with the largest task variance on each task (rows) — seed {seed}. "
                 f"Dot: drop > 0.1 and larger than all {N_RAND} random groups of {M}. Diagonal: the group's own task.",
                 fontsize=9, y=0.995)
    out = REPO / "results/front" / f"lesion_taskgroups_s{seed}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("->", out)
    print(f"{'':20s} own   other  random | tasks hurt>0.1 (median) | groups hurting nothing | most-hurt = own | "
          f"own beyond random | random: tasks hurt")
    for p, v in L.items():
        g = v["groups"]
        jac = np.mean([len(set(g[a]) & set(g[b])) / len(set(g[a]) | set(g[b]))
                       for a in range(len(g)) for b in range(a + 1, len(g))])
        for mode in MODES:
            S = summary(v, mode)
            print(f"{p:8s} {mode:5s} {S['own']:6.3f} {S['other']:6.3f} {S['rand']:6.3f} | {S['hurt']:4.0f} | "
                  f"{S['none']:2d}/20 | {S['top_own']:2d}/20 | {S['beyond_own']:2d}/20 | {S['rand_hurt']:.0f}")
        print(f"{p:8s} group overlap (mean Jaccard over task pairs) {jac:.3f}; "
              f"distinct neurons in the 20 groups {len(np.unique(g))}/{g.size}")


def seeds_summary(seeds):
    """Across seeds: per network and lesion type, own-task damage, other-task damage, their ratio (specificity,
    independent of the overall damage), random-group damage, tasks hurt > 0.1 per group, group overlap.
    Mann-Whitney of each constrained point vs control (n = seeds per point). Figure: one dot per network."""
    import pandas as pd
    from scipy.stats import mannwhitneyu
    rows = []
    for p in POINTS:
        for s in seeds:
            if not path(p, s).exists():
                continue
            v = load(p, s)
            g = v["groups"]
            jac = np.mean([len(set(g[a]) & set(g[b])) / len(set(g[a]) | set(g[b]))
                           for a in range(len(g)) for b in range(a + 1, len(g))])
            for mode in MODES:
                S = summary(v, mode)
                rows.append(dict(point=p, seed=s, mode=mode, own=S["own"], other=S["other"],
                                 ratio=S["own"] / max(S["other"], 1e-9), rand=S["rand"], hurt=S["hurt"],
                                 top_own=S["top_own"], overlap=jac))
    d = pd.DataFrame(rows)
    d.to_csv(CACHE / "lesion_taskgroups_seeds.csv", index=False)
    for mode in MODES:
        print(f"\n{mode}: mean +- sd over seeds (n per point: {d[d['mode'] == mode].groupby('point').size().to_dict()})")
        print(f"{'':9s} {'own':>12s} {'other':>12s} {'own/other':>12s} {'random':>12s} {'tasks hurt':>11s}"
              f" {'top = own':>10s} {'overlap':>8s}  vs control (own/other)")
        ctl = d[(d["mode"] == mode) & (d.point == "control")].ratio
        for p in POINTS:
            q = d[(d["mode"] == mode) & (d.point == p)]
            f = lambda c, k=3: f"{q[c].mean():.{k}f}+-{q[c].std():.{k}f}"
            pv = "" if p == "control" else f"Mann-Whitney p = {mannwhitneyu(q.ratio, ctl).pvalue:.3f}"
            print(f"{p:9s} {f('own'):>12s} {f('other'):>12s} {f('ratio', 2):>12s} {f('rand'):>12s} "
                  f"{f('hurt', 1):>11s} {f('top_own', 1):>10s} {q.overlap.mean():8.3f}  {pv}")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, mode in zip(axes, MODES):
        for i, p in enumerate(POINTS):
            q = d[(d["mode"] == mode) & (d.point == p)]
            ax.scatter(np.full(len(q), i) + np.linspace(-0.12, 0.12, len(q)), q.ratio, s=28, color=COLORS[p],
                       edgecolors="white", lw=0.5, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [q.ratio.mean()] * 2, color=COLORS[p], lw=2)
        ax.set_xticks(range(len(POINTS)))
        ax.set_xticklabels([LABEL[p].replace(" constraint", "\nconstraint") for p in POINTS], fontsize=8.5)
        ax.set_title("silenced (r = 0)" if mode == "zero" else "mean-clamped", fontsize=10)
        ax.axhline(1, color="0.6", ls=":", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("damage on own task / on other tasks")
    fig.suptitle(f"Task-group lesions ({M} E neurons per task), one dot per network (seeds {min(seeds)}-{max(seeds)})",
                 fontsize=10)
    out = REPO / "results/front" / "lesion_taskgroups_seeds.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("->", out)


FAMILY = {"Go": ["fdgo", "reactgo", "delaygo"], "Anti": ["fdanti", "reactanti", "delayanti"],
          "DM": ["dm1", "dm2", "contextdm1", "contextdm2", "multidm"],
          "Dly DM": ["delaydm1", "delaydm2", "contextdelaydm1", "contextdelaydm2", "multidelaydm"],
          "Matching": ["dmsgo", "dmsnogo", "dmcgo", "dmcnogo"]}                    # Yang et al. 2019 task families


def family_scores(v, mode):
    """Damage d = max(0, -change), averaged over the 20 groups: own task, the other tasks of the group's family,
    the other families; and the same for the random groups (all tasks)."""
    tasks = v["tasks"]
    fam = np.array([next(f for f, ts in FAMILY.items() if t in ts) for t in tasks])
    D = np.clip(-v[f"d_{mode}"], 0, None)
    n = np.arange(len(tasks))
    same = np.mean([D[a][(fam == fam[a]) & (n != a)].mean() for a in n])
    other = np.mean([D[a][fam != fam[a]].mean() for a in n])
    return dict(own=np.diag(D).mean(), same=same, other=other, ratio=same / max(other, 1e-9),
                rand=np.clip(-v[f"r_{mode}"], 0, None).mean())


def dose(seed, sizes=(2, 5, 10, 20, 40), mode="mean", pop="E"):
    """Dose-response: family specificity and damage against the number of neurons lesioned per task group."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(wspace=0.32))
    print(f"seed {seed}, {mode}: m | own task | same family | other families | family ratio | random")
    for p in POINTS:
        S = [(m, family_scores(load(p, seed, m, pop), mode)) for m in sizes if path(p, seed, m, pop).exists()]
        ms = [m for m, _ in S]
        for ax, k in zip(axes, ("ratio", "same", "other")):
            ax.plot(ms, [s[k] for _, s in S], "-o", color=COLORS[p], ms=4, label=LABEL[p])
        axes[2].plot(ms, [s["rand"] for _, s in S], ":", color=COLORS[p], lw=1)
        for m, s in S:
            print(f"  {p:8s} {m:3d} | {s['own']:.3f} | {s['same']:.3f} | {s['other']:.3f} | {s['ratio']:5.2f} | {s['rand']:.3f}")
    titles = ("family specificity\n(same family / other families)", "damage on the other tasks\nof the group's family",
              "damage on other families\n(dotted: random groups, all tasks)")
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(m) for m in sizes])
        ax.minorticks_off()
        ax.set_xlabel(f"{pop} neurons lesioned per task group")
        ax.set_title(t, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(1, color="0.6", ls=":", lw=0.8)
    axes[1].set_ylabel("accuracy drop")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Dose-response of {pop} task-group lesions ({'mean-clamped' if mode == 'mean' else 'silenced'}), "
                 f"seed {seed}", fontsize=10, y=1.04)
    out = REPO / "results/front" / f"lesion_dose_s{seed}{'' if pop == 'E' else '_' + pop}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("->", out)


def talk_fig(seeds=range(5), mode="mean"):
    """Talk figure (presentation/figs/talk/lesion_dose_front.pdf): columns E / I groups, rows family specificity and
    damage on other families (dotted: random groups); thin = one network, thick = mean over seeds. Reads the
    dose_seeds CSVs (lesion_dose_seeds_<mode>[_I].csv), so the per-network caches are not needed."""
    import pandas as pd
    from talk_style import W, save
    pops = {"E": ((2, 5, 10, 20, 40), 204), "I": ((1, 2, 5, 10), 52)}
    fig, axes = plt.subplots(2, 2, figsize=(W, 3.6), gridspec_kw=dict(hspace=0.25, wspace=0.22), sharex="col")
    for c, (pop, (sizes, n)) in enumerate(pops.items()):
        d = pd.read_csv(CACHE / f"lesion_dose_seeds_{mode}{'' if pop == 'E' else '_' + pop}.csv")
        for p in POINTS:
            S = {(r.seed, r.m): r._asdict() for r in d[d.point == p].itertuples() if r.seed in seeds and r.m in sizes}
            for r, k in enumerate(("ratio", "other")):
                ax = axes[r, c]
                for s in seeds:
                    ms = [m for m in sizes if (s, m) in S]
                    ax.plot(ms, [S[s, m][k] for m in ms], color=COLORS[p], lw=0.5, alpha=0.3)
                mean = [np.mean([S[s, m][k] for s in seeds if (s, m) in S]) for m in sizes]
                ax.plot(sizes, mean, "-o", color=COLORS[p], ms=3, lw=1.8, label=LABEL[p])
                if k == "other":
                    ax.plot(sizes, [np.mean([S[s, m]["rand"] for s in seeds if (s, m) in S]) for m in sizes], ":",
                            color=COLORS[p], lw=1)
        for r in range(2):
            ax = axes[r, c]
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels([f"{m}\n{100 * m / n:.0f}%" for m in sizes] if r else [])
            ax.minorticks_off()
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        axes[0, c].axhline(1, color="0.6", ls=":", lw=0.8)
        axes[0, c].set_title(f"{pop} groups", fontsize=9, color=E_COL if pop == "E" else I_COL)
        axes[1, c].set_xlabel(f"{pop} neurons lesioned per task (% of {n})", fontsize=7.5)
    axes[0, 0].set_ylabel("family specificity", fontsize=7.5)
    axes[1, 0].set_ylabel("damage on\nother families", fontsize=7.5)
    axes[0, 1].legend(fontsize=6.5, frameon=False, loc="upper right")
    save(fig, "lesion_dose", run="front")


def dose_seeds(seeds, sizes=(2, 5, 10, 20, 40), mode="mean", pop="E"):
    """Dose-response across seeds: thin line per network, thick = mean over seeds. CSV of every value, and per size
    a Mann-Whitney of each constrained point's family ratio vs control (n = seeds per point)."""
    import pandas as pd
    from scipy.stats import mannwhitneyu
    tag = "" if pop == "E" else f"_{pop}"
    d = pd.DataFrame([dict(point=p, seed=s, m=m, **family_scores(load(p, s, m, pop), mode))
                      for p in POINTS for s in seeds for m in sizes if path(p, s, m, pop).exists()])
    d.to_csv(CACHE / f"lesion_dose_seeds_{mode}{tag}.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw=dict(wspace=0.32))
    for p in POINTS:
        q = d[d.point == p]
        for ax, k in zip(axes, ("ratio", "same", "other")):
            for s in seeds:
                r = q[q.seed == s].sort_values("m")
                ax.plot(r.m, r[k], "-", color=COLORS[p], lw=0.6, alpha=0.35)
            g = q.groupby("m")[k].mean()
            ax.plot(g.index, g.values, "-o", color=COLORS[p], ms=4, lw=2, label=LABEL[p])
        g = q.groupby("m")["rand"].mean()
        axes[2].plot(g.index, g.values, ":", color=COLORS[p], lw=1.2)
    titles = ("family specificity\n(same family / other families)", "damage on the other tasks\nof the group's family",
              "damage on other families\n(dotted: random groups, all tasks)")
    for ax, t in zip(axes, titles):
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(m) for m in sizes])
        ax.minorticks_off()
        ax.set_xlabel(f"{pop} neurons lesioned per task group")
        ax.set_title(t, fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(1, color="0.6", ls=":", lw=0.8)
    axes[1].set_ylabel("accuracy drop")
    axes[0].legend(fontsize=8, frameon=False)
    n = d.groupby("point").seed.nunique().to_dict()
    fig.suptitle(f"Dose-response of {pop} task-group lesions ({'mean-clamped' if mode == 'mean' else 'silenced'}), "
                 f"seeds {min(seeds)}-{max(seeds)} (thin: one network, thick: mean)", fontsize=10, y=1.04)
    out = REPO / "results/front" / f"lesion_dose_seeds_{mode}{tag}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("->", out, "| seeds per point:", n)
    for k in ("ratio", "same", "other"):
        print(f"\n{k}: mean +- sd over seeds" + ("   [Mann-Whitney vs control]" if k == "ratio" else ""))
        print(f"{'':9s}" + "".join(f"{'m = ' + str(m):>16s}" for m in sizes))
        for p in POINTS:
            cells = []
            for m in sizes:
                q = d[(d.point == p) & (d.m == m)][k]
                cells.append(f"{q.mean():.2f}+-{q.std():.2f}" if k == "ratio" else f"{q.mean():.3f}+-{q.std():.3f}")
            print(f"{p:9s}" + "".join(f"{c:>16s}" for c in cells))
        if k == "ratio":
            for p in POINTS[1:]:
                ps = [mannwhitneyu(d[(d.point == p) & (d.m == m)].ratio,
                                   d[(d.point == "control") & (d.m == m)].ratio).pvalue for m in sizes]
                print(f"  {p} vs control, p by m: " + "  ".join(f"{m}: {v:.3f}" for m, v in zip(sizes, ps)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--point", choices=POINTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--seeds", default=None, help="e.g. 0-4: across-seed summary from the caches")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--m", default=str(M), help="group size(s), e.g. 2,5,10,20,40 (dose-response)")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--dose", action="store_true", help="dose-response figure for --seed")
    ap.add_argument("--talk", action="store_true", help="talk figure, E and I, seeds 0-4")
    ap.add_argument("--pop", default="E", choices=["E", "I"], help="lesion the top-m E or I neurons per task")
    a = ap.parse_args()
    if a.point:
        for m in map(int, a.m.split(",")):
            compute(a.point, a.seed, a.threads, m, a.modes.split(","), a.pop)
    if a.plot:
        plot(a.seed)
    if a.talk:
        talk_fig()
    if a.dose and not a.seeds:
        dose(a.seed, **(dict(sizes=(1, 2, 5, 10), pop="I") if a.pop == "I" else {}))
    if a.seeds:
        lo, hi = map(int, a.seeds.split("-"))
        if a.dose:
            dose_seeds(range(lo, hi + 1), **(dict(sizes=(1, 2, 5, 10), pop="I") if a.pop == "I" else {}))
        else:
            seeds_summary(range(lo, hi + 1))
