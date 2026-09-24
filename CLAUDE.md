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
                pareto.py, zoom_lambda.py, moo.py, paths.py — see README.md for install/run commands
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
actual wire length (no spatial embedding); L0 / connection-probability
formulation deferred, would need a probability-of-connection parameterization.

**Pareto front rules** (`cmc/pareto.py` `compute_front`, mirrored in the
notebook): task axis is `min_task_acc` (worst task), not `mean_acc` -- a
network can abandon one task and still score ~0.9 mean. Networks with
`min_task_acc < 0.6` (`FEASIBLE_MIN_TASK_ACC`) are excluded from the front by
default, otherwise chance-level networks sit on it as the cheapest points
(35/37 "Pareto" in the core5 6x6 run -> 14 after the fix). Switches:
`--include-infeasible`, `--pareto-task-objective mean_acc`,
`--feasible-min-task-acc`; notebook `EXCLUDE_INFEASIBLE` / `TASK_OBJECTIVE`.

**Inspecting networks:** `cmc.pareto` keeps only metrics. `cmc.zoom_lambda`
trains many seeds (default 10) at a few chosen lambda points (default: a 2x2
control / rate / wiring / both design from the corrected core5 6x6 front) and
saves every network with its lambdas, metrics, feasibility and per-neuron task
variance; full activity is re-simulated, not stored. Output in
`zoom_lambda_runs/`, weights git-ignored. A network is fully determined by
(lambda, seed): torch is seeded in the model constructors, which it was not
before -- sweeps predating that are not bit-reproducible.

**Multi-objective (`cmc.moo`):** NSGA-III (pymoo ask/tell), where each evaluation
is one full training via `zoom_lambda.train_and_evaluate`. Objectives are
(1 - min_task_acc, log10 metabolic, log10 wiring), with min_task_acc >= 0.6 as
a constraint. Genome `lambda` = log lambdas: still a weighted sum, so it cannot
reach non-convex parts of the front; it only validates the loop against the 6x6
front. The goal is the planned `budget` genome: constrained training with cost
ceilings, no hand-set lambdas. Output in `moo_runs/`.

**Important:** the fixes above changed what the loss functions numerically
mean (wiring quantity, loss normalization, task difficulty). Old
checkpoints/sweeps from before this fix are **not directly comparable** to
new ones. Lambda ranges have been recalibrated for the new objectives --
the defaults in `cmc/pareto.py` carry both the measurements and the
reasoning. Note that `--report-scales` measures at *init*, where the rate
cost is 26x smaller and the task loss 30x larger than at a trained
solution, so its break-even lambdas overshoot by ~500x (rate) / ~20x
(wiring): it is a check that the terms are finite, not a grid centre.
