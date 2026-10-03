import matplotlib;matplotlib.use("Agg")
display=print
import contextlib, io, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from cmc.train_cog import load_checkpoint
from cmc.runner import activity_summary

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ZOOM_DIR = ROOT / "runs/lambda_zoom/dale_cog20_middle_3seed"
CKPT_DIR = ROOT / "runs/checkpoints"
zsum = json.loads((ZOOM_DIR / "summary.json").read_text())
TASKS = zsum["active_tasks"]
EVAL_SEEDS = zsum["eval_seeds"]
EVAL_BS = zsum["args"].get("eval_batch_size", 64)
zoom = pd.read_csv(ZOOM_DIR / "runs.csv")
zoom[["seed", "mean_acc", "min_task_acc", "metabolic_cost", "wiring_cost", "conn_frac", "is_feasible"]]
zoom.filter(like="acc_").T.set_axis([f"seed {s}" for s in zoom.seed], axis=1).round(3)
def load(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_checkpoint(path)

nets = {}   # name -> dict(tv, n_e, group)
for p in sorted((ZOOM_DIR / "middle").glob("seed_*.pt")):
    model, cfg, ck = load(p)
    nets[f"middle/{p.stem}"] = dict(tv=np.asarray(ck["task_variance"]), n_e=int(model.n_e), group="middle")

for p in sorted(CKPT_DIR.glob("dale_20_*.pt")):
    model, cfg, ck = load(p)
    if list(ck.get("active_tasks", [])) != TASKS:
        print("skip (task set differs):", p.name); continue
    tv, _ = activity_summary(model, cfg, TASKS, torch.device("cpu"), EVAL_SEEDS, EVAL_BS)
    nets[f"control/{p.stem}"] = dict(tv=tv, n_e=int(model.n_e), group="control")
print(len(nets), "networks:", list(nets))
K_RANGE = range(2, 21)

def tv_clusters(tv, k_range=K_RANGE, seed=0):
    total = tv.sum(axis=0)
    active = total > 1e-2 * total.max()
    norm = tv[:, active] / tv[:, active].max(axis=0, keepdims=True)
    X = norm.T
    scores, labs = {}, {}
    for k in k_range:
        if k >= len(X): break
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        scores[k], labs[k] = silhouette_score(X, lab), lab
    k = max(scores, key=scores.get)
    lab = labs[k]
    pref = {c: np.argmax(norm[:, lab == c].mean(axis=1)) for c in range(k)}
    order = sorted(range(k), key=lambda c: (pref[c], -norm[pref[c], lab == c].mean()))
    remap = {c: i for i, c in enumerate(order)}
    labels = np.full(tv.shape[1], -1); labels[active] = [remap[c] for c in lab]
    return dict(active=active, norm_full=tv / np.maximum(tv.max(axis=0, keepdims=True), 1e-30),
                labels=labels, k=k, silhouette=scores[k], scores=scores,
                pref_tasks=sorted({TASKS[pref[c]] for c in range(k)}))

for name, n in nets.items():
    n["cl"] = tv_clusters(n["tv"])

stats = pd.DataFrame([dict(net=name, group=n["group"], n_active=int(n["cl"]["active"].sum()),
                           n_active_E=int(n["cl"]["active"][:n["n_e"]].sum()),
                           n_active_I=int(n["cl"]["active"][n["n_e"]:].sum()),
                           k=n["cl"]["k"], silhouette=n["cl"]["silhouette"],
                           n_pref_tasks=len(n["cl"]["pref_tasks"]))
                      for name, n in nets.items()])
display(stats.round(3))
display(stats.groupby("group")[["n_active", "k", "silhouette", "n_pref_tasks"]].agg(["mean", "std"]).round(2))
fig, ax = plt.subplots(figsize=(7, 4))
for name, n in nets.items():
    s = n["cl"]["scores"]
    ax.plot(list(s), list(s.values()), "-o", ms=3, color="C3" if n["group"] == "middle" else "C0",
            alpha=0.8, label=name)
ax.set_xlabel("k"); ax.set_ylabel("silhouette"); ax.legend(fontsize=7)
ax.set_title("silhouette vs k (red = middle λ, blue = unconstrained)")
plt.close()
fig, axes = plt.subplots(1, len(nets), figsize=(4.2 * len(nets), 7))
axes = np.atleast_1d(axes)
for ax, (name, n) in zip(axes, nets.items()):
    cl = n["cl"]
    idx = np.flatnonzero(cl["active"])
    idx = idx[np.lexsort((idx >= n["n_e"], cl["labels"][idx]))]
    ax.imshow(cl["norm_full"][:, idx].T, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    for b in np.flatnonzero(np.diff(cl["labels"][idx])) + 0.5:
        ax.axhline(b, color="cyan", lw=0.6)
    ax.set_xticks(range(len(TASKS))); ax.set_xticklabels(TASKS, rotation=90, fontsize=7)
    ax.set_title(f"{name}\n{cl['active'].sum()} active, k={cl['k']}, sil={cl['silhouette']:.2f}", fontsize=8)
axes[0].set_ylabel("active neurons, sorted by cluster")
plt.tight_layout(); plt.close()
rows = []
for name, n in nets.items():
    norm = n["cl"]["norm_full"][:, n["cl"]["active"]]
    frac = np.bincount(norm.argmax(axis=0), minlength=len(TASKS)) / norm.shape[1]
    rows.append(pd.Series(frac, index=TASKS, name=name))
    print(f"{name:45s} cluster-preferred tasks: {n['cl']['pref_tasks']}")
pd.DataFrame(rows).T.round(2)
print(pd.DataFrame(rows).T.round(2).to_string())