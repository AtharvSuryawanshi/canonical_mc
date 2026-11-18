import sys, contextlib, io, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, "/home/atharv/Documents/Github/canonical_mc/results/scripts")
import fig_fp_task_pair as fpp
from cmc.train_cog import load_checkpoint
from fig_tv_ei_fanin import RUNS
OUT = "/tmp/claude-1000/-home-atharv-Documents-Github-canonical-mc/739940c6-2167-4885-8e61-811b6968664a/scratchpad/"
fpp.N_TRIALS = 24
NETS = {"control": "Unconstrained", "C0": "C0, both budgets"}
D = {}
for pt in NETS:
    with contextlib.redirect_stdout(io.StringIO()):
        m, cfg, ck = load_checkpoint(RUNS["5seed"] / pt / "seed_04.pt")
    p1, _ = fpp.build(cfg)                      # same eval seed -> same stimuli / timing for both nets
    HA, _ = fpp.sim(m, p1["xA"]); HB, _ = fpp.sim(m, p1["xB"])
    per = p1["periods"]; s0 = per["stimulus"][0]
    X = np.r_[HA[:, s0:].reshape(-1, HA.shape[2]), HB[:, s0:].reshape(-1, HB.shape[2])]
    mu = X.mean(0); V = np.linalg.svd(X - mu, full_matrices=False)[2][:3]
    pA, pB = (HA - mu) @ V.T, (HB - mu) @ V.T
    lim = [(min(pA[:, s0:, i].min(), pB[:, s0:, i].min()) - .1, max(pA[:, s0:, i].max(), pB[:, s0:, i].max()) + .1) for i in range(3)]
    # numbers: where does each Anti trial sit relative to Go trials
    th = p1["stim"]; dd = lambda x: np.degrees(np.abs(np.angle(np.exp(1j * x))))
    nums = []
    for t in (per["stimulus"][1] - 1, per["delay"][1] - 1):
        Dm = np.linalg.norm(HB[:, t][:, None] - HA[:, t][None], axis=2); j = Dm.argmin(1)
        nums.append(np.median(dd(th[j] - th)))
    print(pt, "nearest-Go angle difference, end of stimulus / end of delay:", np.round(nums))
    D[pt] = dict(pA=pA, pB=pB, lim=lim, per=per, s0=s0, col=plt.cm.hsv(np.mod(th, 2 * np.pi) / (2 * np.pi)))
per, s0 = D["C0"]["per"], D["C0"]["s0"]
T = D["C0"]["pA"].shape[1]
def name(t):
    for k, (a, b) in per.items():
        if a <= t < b: return k.upper()
    return "before stimulus"
for tag, both in (("gif4_dlygo_compare", False), ("gif5_goanti_compare", True)):
    fig = plt.figure(figsize=(11.5, 5.6))
    axs = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    def frame(t):
        for ax, (pt, title) in zip(axs, NETS.items()):
            d = D[pt]; ax.cla()
            sets = ((d["pA"], "o", "-"), (d["pB"], "^", "--")) if both else ((d["pA"], "o", "-"),)
            for P, mk, ls in sets:
                for b in range(len(d["col"])):
                    ax.plot(*P[b, max(s0 - 1, t - 10):t + 1].T, color=d["col"][b], lw=0.8, alpha=0.5, ls=ls)
                    ax.scatter(*P[b, t], color=d["col"][b], marker=mk, s=26, depthshade=False, edgecolors="k", linewidths=0.3)
            ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.set_title(title, fontsize=11)
            ax.view_init(22, -60 + 0.3 * (t - s0))
        fig.suptitle(f"time {t*20} ms  |  {name(t)}" + ("      circles = Dly Go, triangles = Dly Anti" if both else "      Dly Go") +
                     "   (colour = stimulus angle)", fontsize=11)
    anim = FuncAnimation(fig, frame, frames=list(range(s0 - 3, T)) + [T - 1] * 12, interval=70)
    anim.save(OUT + tag + ".gif", writer=PillowWriter(fps=13)); plt.close(fig)
    from PIL import Image
    im = Image.open(OUT + tag + ".gif"); fr = []
    for k in (8, 40, 110):
        im.seek(min(k, im.n_frames - 1)); fr.append(im.convert("RGB").copy())
    out = Image.new("RGB", (fr[0].width, sum(f.height for f in fr)), "white"); y = 0
    for f in fr: out.paste(f, (0, y)); y += f.height
    out.save(OUT + tag + "_peek.png")
print("done")
