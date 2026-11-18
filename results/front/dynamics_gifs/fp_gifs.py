import sys, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, "/home/atharv/Documents/Github/canonical_mc/results/scripts")
from fig_fp_delaygo import setup, ALPHAS
OUT = "/tmp/claude-1000/-home-atharv-Documents-Github-canonical-mc/739940c6-2167-4885-8e61-811b6968664a/scratchpad/"
m, cfg, ck, tr, H, tgt, ud, ug = setup("C0", 4)
z = np.load("/home/atharv/Documents/Github/canonical_mc/results/front/fp_delaygo_C0_s4.npz", allow_pickle=True)
FPS, EIG = list(z["fps"]), list(z["eig"])
s0, s1 = tr.epochs["stim1"]; d0, d1 = tr.epochs["delay1"]
nE = m.n_e; W_out = m.w_out.detach().double().numpy(); P = (np.arange(W_out.shape[1] - 1) * 2 * np.pi / (W_out.shape[1] - 1))
def run(h0, u, T):
    inp = torch.as_tensor(np.repeat(u[None, :, None], len(h0), 0).repeat(T, 2), dtype=torch.float32)
    with torch.no_grad():
        return m.simulate(inp, x_init=torch.as_tensor(h0, dtype=torch.float32))[0].numpy()
D = H[:, d0:d1].reshape(-1, H.shape[2]); mu = D.mean(0)
pcs = np.linalg.svd(D - mu, full_matrices=False)[2][:2]
vz = np.r_[W_out[:, 1:] @ np.cos(P), np.zeros(m.n_rnn - nE)]; vz /= np.linalg.norm(vz)
proj = lambda X: np.c_[(X - mu) @ pcs.T, X @ vz]
idx = np.argsort(np.mod(tgt, 2*np.pi))[::4]            # 32 trials
col = plt.cm.hsv(np.mod(tgt[idx], 2*np.pi) / (2*np.pi))
Pall = proj(H[idx].reshape(-1, H.shape[2])).reshape(len(idx), H.shape[1], 3)
lim = [(Pall[..., i].min() - .2, Pall[..., i].max() + .2) for i in range(3)]

# GIF 1: time
fig = plt.figure(figsize=(6.4, 5.6)); ax = fig.add_subplot(projection="3d")
def frame_t(t):
    ax.cla()
    for b in range(len(idx)):
        tr_ = Pall[b, max(0, t - 12):t + 1]
        ax.plot(*tr_.T, color=col[b], lw=1, alpha=0.6)
        ax.scatter(*Pall[b, t], color=col[b], s=25, depthshade=False)
    per = "before stimulus" if t < s0 else "STIMULUS on" if t < s1 else "DELAY (remember)" if t < d1 else "GO (respond)"
    ax.set_title(f"time {t*20} ms   |   {per}", fontsize=11)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("readout axis")
    ax.view_init(25, -60 + 0.25 * t)
anim = FuncAnimation(fig, frame_t, frames=range(s0 - 5, H.shape[1], 1), interval=60)
anim.save(OUT + "gif1_time.gif", writer=PillowWriter(fps=15)); plt.close(fig)

# GIF 2: alpha (input slid from delay to response); balls = end-of-delay states run 40 steps under u(alpha)
end = H[idx, d1 - 1]
AF = np.linspace(0, 1, 41)
settle = [proj(run(end, (1 - a) * ud + a * ug, 40)[:, -1]) for a in AF]
lim2 = [(min(lim[i][0], min(s[:, i].min() for s in settle) - .2), max(lim[i][1], max(s[:, i].max() for s in settle) + .2)) for i in range(3)]
fig = plt.figure(figsize=(6.4, 5.6)); ax = fig.add_subplot(projection="3d")
def frame_a(k):
    k = min(k, len(AF) - 1)
    a = AF[k]; ax.cla()
    j = int(np.argmin(np.abs(ALPHAS - a)))
    F, e = FPS[j], EIG[j]
    if len(F):
        p = proj(F); st = e < 1
        ax.scatter(*p[st].T, c="k", s=40, depthshade=False)
        ax.scatter(*p[~st].T, facecolors="white", edgecolors="k", s=40, depthshade=False)
    ax.scatter(*settle[k].T, c=col, s=28, depthshade=False)
    ax.set_title(f"alpha = {a:.2f}   (0 = delay input, 1 = response input)\ncoloured = where each trial settles; black/white = flat spots (fixed points)", fontsize=9)
    ax.set_xlim(*lim2[0]); ax.set_ylim(*lim2[1]); ax.set_zlim(*lim2[2])
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("readout axis")
    ax.view_init(25, -60 + 1.5 * k)
anim = FuncAnimation(fig, frame_a, frames=range(len(AF) + 10), interval=150)
anim.save(OUT + "gif2_alpha.gif", writer=PillowWriter(fps=6)); plt.close(fig)
print("done")
