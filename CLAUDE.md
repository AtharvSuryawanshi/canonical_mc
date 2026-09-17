# canonical_mc — fast context

Cost-constrained multitask RNN project. Hypothesis: canonical cortical
circuit motifs emerge when a network must solve many tasks under metabolic
(firing-rate) and wiring (recurrent-connectivity) budgets. Train Dale's-law
RNNs on the Yang et al. (2019) cognitive task battery, penalize rate and
wiring cost, and look at the resulting Pareto front (task accuracy vs.
metabolic cost vs. wiring cost).

## Layout

```
cmc/            installable package: task.py, network.py, train_cog.py,
                pareto.py, paths.py — see README.md for install/run commands
notebooks/      analysis, imports cmc.*
slurm/          LRZ batch scripts
archives/       legacy code, not imported by anything
checkpoints/    trained weights (small ones only — don't bloat this)
pareto_runs/    one dir per sweep: runs.csv (per-seed) + summary.csv/json
theory.md       the neuroscience the objectives are meant to encode
FIXED_ISSUES.md audit of past mismatches between that theory and the code
```

Env: `conda create -n cmc_env python=3.12 && pip install -r requirements.txt
&& pip install -e .`. Full install/run instructions are in README.md —
don't duplicate them here.

## Model

- `DaleRNN`: sign-constrained recurrent weights (`softplus(w_raw)` magnitude
  × fixed presynaptic E/I sign vector), zero diagonal (no autapses), E-only
  readout, `frac_e=0.8`. `LeakyRNN` is the unconstrained baseline.
- Leaky integration: `h = (1-alpha)h + alpha*act(gate)`, `alpha = dt/tau`.
- Loss = task loss (masked MSE via `c_mask`) + `lambda_rate * rate_reg` +
  `lambda_connectivity * connectivity_reg`.

## Neuroscience-faithfulness fixes already applied (see FIXED_ISSUES.md)

These are the load-bearing facts — later work should not accidentally
regress them:

1. **`connectivity_reg` penalizes `W_rec` with L1, not `W_in` with L2.**
   Wiring cost is about recurrent circuit connectivity, not input
   projections; L1 is the convex surrogate for connection count, L2 is
   shrinkage (doesn't prune). Population-normalized (`ei_row_scale`) so L1
   doesn't preferentially prune inhibition.
2. **Circular decoding fixed.** `batch_accuracy` now uses `atan2`-based
   circular mean and `wrap_to_pi` angular error (181° vs 180° is 1°, not
   179°). Task loss was already circular via `get_dist`; only the accuracy
   decoder was broken.
3. **Fixation release and response-window scoring fixed** using
   neuroscience-correct criteria (`response_mask`, `post_ons`), not just
   whatever window happened to be convenient in code.
4. **Easy-task coherence band** is `EASY_STIM_COH_RANGE = [0.10, 0.15,
   0.20]`, not an arbitrary multiplier on the hard-task range.
5. **`rate_reg` / `connectivity_reg` support L1 or L2**, default L2 for
   rate, L1 for connectivity (wiring), both population-weighted so E and I
   aren't penalized asymmetrically by accident.
6. Default eval noise level is `DALE_DEFAULT_NOISE_LEVEL = 0.1`.

**Deliberately not changed / open:** `rho(W_rec)` is only enforced at init
and can drift during training; L1 is a proxy for connection count, not
actual wire length (no spatial embedding); Pareto front currently
optimizes `mean_acc`, not `min_task_acc` (a network can abandon a hard task
and still look good) — deferred; L0 / connection-probability formulation
also deferred, would need a probability-of-connection parameterization.

**Important:** the fixes above changed what the loss functions numerically
mean (wiring quantity, loss normalization, task difficulty). Old
checkpoints/sweeps from before this fix are **not directly comparable** to
new ones. Use `python -m cmc.train_cog --report-scales` to recalibrate
lambda ranges before trusting a new sweep.
