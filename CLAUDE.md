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
                front.py (dominance, feasibility), runner.py (train+eval one
                network, common flags), lambda_pareto.py, lambda_zoom.py, moo.py,
                moo_zoom.py, batched.py, paths.py. pareto.py / zoom_lambda.py are deprecated shims.
notebooks/      lambda_pareto_analysis, moo_pareto_analysis, ...; import cmc.*
slurm/          LRZ batch scripts
runs/           ALL output, committed incl. weights (git is the cluster transfer):
                checkpoints/, lambda_pareto/, lambda_zoom/, moo/, moo_zoom/
archives/       legacy code, not imported by anything
reports/        theory.md (neuroscience the objectives encode), FIXED_ISSUES.md (audit
                of theory/code mismatches), IMPLEMENTATION.md, PRESENTATION.md, ABSTRACT.md
```

Env: `conda create -n cmc_env python=3.12 && pip install -r requirements.txt
&& pip install -e .`. Full install/run instructions are in README.md —
don't duplicate them here.

Naming: `lambda_*` = anything driven by hand-set lambdas (weighted sum),
`moo_*` = the NSGA-III budget search. Shared, method-agnostic code lives in
`cmc.front` / `cmc.runner`, never in a `lambda_*` or `moo` module.

## Model

- `DaleRNN`: sign-constrained recurrent weights (`softplus(w_raw)` magnitude
  × fixed presynaptic E/I sign vector), zero diagonal (no autapses), E-only
  readout, `frac_e=0.8`. `LeakyRNN` is the unconstrained baseline.
- Leaky integration: `h = (1-alpha)h + alpha*act(gate)`, `alpha = dt/tau`.
- Loss = task loss (masked MSE via `c_mask`) + `lambda_rate * rate_reg` +
  `lambda_connectivity * connectivity_reg`.

## Neuroscience-faithfulness fixes already applied (see reports/FIXED_ISSUES.md)

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

**Pareto front rules** (`cmc/lambda_pareto.py` `compute_front`, mirrored in the
notebook): task axis is `min_task_acc` (worst task), not `mean_acc` -- a
network can abandon one task and still score ~0.9 mean. Networks with
`min_task_acc < 0.6` (`FEASIBLE_MIN_TASK_ACC`) are excluded from the front by
default, otherwise chance-level networks sit on it as the cheapest points
(35/37 "Pareto" in the core5 6x6 run -> 14 after the fix). Switches:
`--include-infeasible`, `--pareto-task-objective mean_acc`,
`--feasible-min-task-acc`; notebook `EXCLUDE_INFEASIBLE` / `TASK_OBJECTIVE`.

**Inspecting networks:** `cmc.lambda_pareto` keeps only metrics. `cmc.lambda_zoom`
trains many seeds (default 10) at a few chosen lambda points (default: a 2x2
control / rate / wiring / both design from the corrected core5 6x6 front) and
saves every network with its lambdas, metrics, feasibility and per-neuron task
variance; full activity is re-simulated, not stored. Output in
`runs/lambda_zoom/`, weights committed. A network is fully determined by
(lambda, seed): torch is seeded in the model constructors, which it was not
before -- sweeps predating that are not bit-reproducible.

**Multi-objective (`cmc.moo`):** NSGA-III (pymoo ask/tell), where each evaluation
is one full training via `runner.train_and_evaluate`. Objectives are
(1 - min_task_acc, log10 metabolic, log10 wiring), with min_task_acc >= 0.6 as
a constraint. Default genome `budget` = log cost ceilings (epsilon-constraint,
no hand-set lambdas). The wiring budget is enforced exactly by projecting W_rec
onto the L1 ball after every step (`project_w_rec_to_budget`). The rate budget
uses a learned log-multiplier (`BudgetConstraint`). Don't switch wiring to the
multiplier (`--conn-budget-mode lagrangian`): wiring responds to lambda with a
lag and a threshold, so the multiplier winds up to 50-500x the needed value and
prunes the network to death (tested; ramp and PI terms did not fix it). Checked
at 3 reference-front costs: accuracy matches the grid networks within seed
noise. Genome `lambda` (log lambdas, weighted sum) only validated the loop
against the 6x6. Budget training is opt-in in `train()`; with no budgets it is
bit-identical to before. Output in `runs/moo/`. Analysis:
`notebooks/moo_pareto_analysis.ipynb`. For the core5 p24g10 budget run: front
~50% larger than the 6x6 by hypervolume. Non-convex (unreachable by any
lambda) only at the feasibility cliff. The feasibility edge follows
rate x wiring ~ const. **dmsgo is the bottleneck in every network**, with the
other 4 tasks at ceiling, so on core5 min_task_acc == dmsgo accuracy.
`cmc.moo_zoom` saves 10 seeds at 5 budget points: control, then
rate_limited / middle / wiring_limited along the ~0.65 iso-accuracy curve
(equal competence, different binding cost: the key contrast), and
the knee. Seed 0 reproduces the cmc.moo network bit-for-bit on CPU; on GPU
(CUDA nondeterminism, different MIG slices) only to within ~0.3 seed std
(checked in notebooks/moo_zoom_analysis.ipynb). Same caveat for every
"network is determined by (lambda, seed)" statement here: exact on CPU only.

**Batched training (`cmc.batched`, `--batched` on moo / moo_zoom, opt-in):** P
DaleRNNs stacked into `[P, ...]` tensors and trained in one process; per-network
init, budgets (BudgetConstraint math, L1 projection), Adam and grad clipping are
exactly the single-network ones (`tests/test_batched.py`: equal to ~1e-15 in
float64 on shared trials). But the P networks share ONE trial stream and do no
per-task logging eval, so a batched network is not the single-run network with
that seed. Don't mix batched and single networks in one comparison, and don't
resume a run in the other mode. Default (no flag) code path is unchanged.

**Important:** the fixes above changed what the loss functions numerically
mean (wiring quantity, loss normalization, task difficulty). Old
runs/checkpoints/sweeps from before this fix are **not directly comparable** to
new ones. Lambda ranges have been recalibrated for the new objectives --
the defaults in `cmc/lambda_pareto.py` carry both the measurements and the
reasoning. Note that `--report-scales` measures at *init*, where the rate
cost is 26x smaller and the task loss 30x larger than at a trained
solution, so its break-even lambdas overshoot by ~500x (rate) / ~20x
(wiring): it is a check that the terms are finite, not a grid centre.
