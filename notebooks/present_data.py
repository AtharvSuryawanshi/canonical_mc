"""Per-seed numbers behind notebooks/moo_present.py, computed from the saved checkpoints.

For every feasible seed of every zoom point (g = 1 run):
  ei_ratio.csv      I/E input ratio: per neuron, total |inhibitory| input weight / total
                    excitatory input weight (column sums of W[pre, post]); median over neurons.
  w_vs_init.csv     corr(W, W0) and |dW| / |W|, W0 = the same seed's random init.
  delta_motifs.csv  overall motifs of dW = learned_change(W, W0) vs the target null
                    (cmc.motifs.delta_null_stats, no task groups).

Run:  conda run -n cmc_env python notebooks/present_data.py
"""
import contextlib
import io

import numpy as np
import pandas as pd

from cmc.motifs import delta_null_stats, learned_change
from cmc.paths import MOO_ZOOM_RUNS_DIR
from cmc.task import default_config
from cmc.train_cog import load_checkpoint, make_dale_model

# ============================ PARAMETERS ============================
ZOOM_DIR = MOO_ZOOM_RUNS_DIR / "dale_core5_5pt_10seed"
OUT_DIR = ZOOM_DIR / "present"
POINTS = ["control", "rate_limited", "middle", "wiring_limited", "knee"]
N_PERM = 200
# ====================================================================

OUT_DIR.mkdir(exist_ok=True)
zoom = pd.read_csv(ZOOM_DIR / "runs.csv")
ei, wi, dm = [], [], []
for p in POINTS:
    for seed in sorted(zoom[(zoom.point == p) & zoom.is_feasible].seed):
        seed = int(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            m, _, ck = load_checkpoint(ZOOM_DIR / p / f"seed_{seed:02d}.pt")
            g = ck.get("model_kwargs", {}).get("g", 1.0)
            m0 = make_dale_model(default_config(n_eachring=16, seed=seed), n_neurons=256,
                                 frac_e=0.8, g=g, seed=seed)
        n_e = int(m.n_e)
        W = m.effective_w_rec().detach().numpy()
        W0 = m0.effective_w_rec().detach().numpy()

        exc, inh = np.clip(W, 0, None).sum(0), -np.clip(W, None, 0).sum(0)
        ei.append({"point": p, "seed": seed,
                   "I/E": float(np.median(inh / np.maximum(exc, 1e-12)))})

        D = learned_change(W, W0, m.ei_row_scale().detach().numpy())
        wi.append({"point": p, "seed": seed,
                   "corr(W, W0)": np.corrcoef(W.ravel(), W0.ravel())[0, 1],
                   "|ΔW| / |W|": np.linalg.norm(D) / np.linalg.norm(np.abs(W))})

        for stat, r in delta_null_stats(D, n_e, None, n_perm=N_PERM, seed=seed).items():
            dm.append({"point": p, "seed": seed, "stat": stat, **r})
        print(p, seed, "done")

for name, rows in [("ei_ratio", ei), ("w_vs_init", wi), ("delta_motifs", dm)]:
    pd.DataFrame(rows).to_csv(OUT_DIR / f"{name}.csv", index=False)
print("wrote", OUT_DIR)
