import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
sys.path.insert(0, "/home/atharv/Documents/Github/canonical_mc/results/scripts")
from fig_wiring_like2like import network
from fig_wiring_roc import roc
from fig_tv_ei_fanin import POINTS
OUT = sys.argv[1]
rng = np.random.default_rng(0)
D = {}
for p in ("control", "C0"):
    r = network(p, 4, rng); mk = r["masks"]["E→E"]
    s, c = r["S"][mk], r["C"][mk]
    D[p] = dict(s=s, c=c, auc=r["E→E"]["auc"], roc=roc(s, c))
bins = np.linspace(-1, 1, 41)
T = np.r_[np.linspace(1.0, -0.6, 60), [-0.6] * 10]
fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), gridspec_kw=dict(width_ratios=[1.5, 1], hspace=0.45, wspace=0.25))
def frame(k):
    t = T[k]
    for r, p in enumerate(D):
        d = D[p]; s, c = d["s"], d["c"]
        ax = axes[r, 0]; ax.cla()
        hu, _ = np.histogram(s[~c], bins, density=True); hc, _ = np.histogram(s[c], bins, density=True)
        ctr = (bins[:-1] + bins[1:]) / 2; w = bins[1] - bins[0]
        ax.bar(ctr, hu, w, color=np.where(ctr > t, "#555555", "#d0d0d0"), label="unconnected")
        ax.step(ctr, hc, where="mid", color="#1F8AD0", lw=2, label="connected")
        ax.fill_between(ctr, hc, step="mid", where=ctr > t, color="#1F8AD0", alpha=0.35)
        ax.axvline(t, color="k", lw=1.5)
        tpr, fpr = (s[c] > t).mean(), (s[~c] > t).mean()
        ax.set_title(f"{POINTS[p]}, E→E   threshold t = {t:+.2f}\nconnected above t (TPR) {tpr:.2f}    unconnected above t (FPR) {fpr:.2f}", fontsize=9, loc="left")
        ax.set_xlim(-1, 1); ax.set_yticks([]); ax.set_xlabel("fingerprint similarity")
        ax.legend(fontsize=7, frameon=False, loc="upper left"); ax.spines[["top", "right", "left"]].set_visible(False)
        ax = axes[r, 1]; ax.cla()
        f, tp = d["roc"]
        sel = np.searchsorted(-np.sort(-s), t, side="left")
        o = np.sort(-s)  # descending as negatives
        n_above = (s > t).sum()
        ax.plot(f[:n_above + 1], tp[:n_above + 1], color="#1F8AD0", lw=2)
        ax.fill_between(f[:n_above + 1], tp[:n_above + 1], color="#1F8AD0", alpha=0.12)
        ax.plot(fpr, tpr, "o", color="k", ms=7)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC (final area = AUC {d['auc']:.2f})", fontsize=9)
    fig.suptitle("Slide a threshold t down the similarity axis: each t gives one (FPR, TPR) point", fontsize=11)
anim = FuncAnimation(fig, frame, frames=len(T), interval=120)
anim.save(OUT + "/roc_threshold.gif", writer=PillowWriter(fps=8)); plt.close(fig)
from PIL import Image
im = Image.open(OUT + "/roc_threshold.gif"); im.seek(35); im.convert("RGB").save(OUT + "/roc_peek.png"); print("ok")
