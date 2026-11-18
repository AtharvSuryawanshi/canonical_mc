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
with contextlib.redirect_stdout(io.StringIO()):
    m, cfg, ck = load_checkpoint(RUNS["5seed"] / "C0" / "seed_04.pt")
p1, _ = fpp.build(cfg)
HA, _ = fpp.sim(m, p1["xA"]); HB, _ = fpp.sim(m, p1["xB"])
per = p1["periods"]; s0 = per["stimulus"][0]
X = np.r_[HA[:, s0:].reshape(-1, HA.shape[2]), HB[:, s0:].reshape(-1, HB.shape[2])]
mu = X.mean(0); V = np.linalg.svd(X - mu, full_matrices=False)[2][:3]
pA, pB = (HA - mu) @ V.T, (HB - mu) @ V.T
col = plt.cm.hsv(np.mod(p1["stim"], 2 * np.pi) / (2 * np.pi))
lim = [(min(pA[:, s0:, i].min(), pB[:, s0:, i].min()) - .2, max(pA[:, s0:, i].max(), pB[:, s0:, i].max()) + .2) for i in range(3)]
def name(t):
    for k, (a, b) in per.items():
        if a <= t < b: return k.upper()
    return "before stimulus"
fig = plt.figure(figsize=(6.6, 5.8)); ax = fig.add_subplot(projection="3d")
def frame(t):
    ax.cla()
    for P, mk in ((pA, "o"), (pB, "^")):
        for b in range(len(col)):
            ax.plot(*P[b, max(s0 - 1, t - 10):t + 1].T, color=col[b], lw=0.8, alpha=0.5, ls="-" if mk == "o" else "--")
            ax.scatter(*P[b, t], color=col[b], marker=mk, s=30, depthshade=False, edgecolors="k", linewidths=0.3)
    ax.set_title(f"time {t*20} ms  |  {name(t)}\ncircles = Dly Go,  triangles = Dly Anti  (same stimuli; colour = stimulus angle)", fontsize=9)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.view_init(22, -60 + 0.3 * (t - s0))
T = HA.shape[1]
anim = FuncAnimation(fig, frame, frames=list(range(s0 - 3, T)) + [T - 1] * 12, interval=70)
anim.save(OUT + "gif3_go_vs_anti.gif", writer=PillowWriter(fps=13)); plt.close(fig)
# peek frames
from PIL import Image
im = Image.open(OUT + "gif3_go_vs_anti.gif"); fr = []
for k in (8, 30, 70, 110):
    im.seek(min(k, im.n_frames - 1)); fr.append(im.convert("RGB").copy())
out = Image.new("RGB", (sum(f.width for f in fr), fr[0].height), "white"); x = 0
for f in fr: out.paste(f, (x, 0)); x += f.width
out.save(OUT + "gif3_peek.png"); print(per, T, im.n_frames)
