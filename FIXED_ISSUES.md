# Fixed issues

Audit of `task.py` / `network.py` / `train_cog.py` / `pareto.py` for places where the
neuroscience intent and the code disagreed. Each entry states the issue, why it
matters, and what was changed.

> **Read this first if you have existing results.** Three changes make old runs
> non-comparable: the wiring cost is now a different quantity (#1), the task loss
> is normalized differently (#8), and the decision tasks got harder (#6).
> Accuracy numbers also drop because two scoring loopholes were closed (#3, #4).
> Recalibrate the lambda ranges before the next sweep:
>
> ```bash
> python train_cog.py --model dale --task-battery sanity3 --n-neurons 256 --report-scales
> ```

---

## 1. The wiring penalty was applied to W_in with an L2

**Files:** `train_cog.py` (`connectivity_reg`), `network.py`

**Issue.** `lambda_connectivity` penalized `w_in.square().mean()`. Verified on a
live `DaleRNN(N=128)`: the penalized tensor was `(53, 128)` = `W_in`, while
`W_rec` is `(128, 128)`.

**Why it matters.** The hypothesis is about the cost of *recurrent circuit*
wiring. `W_in` is the input projection -- a thalamocortical-style afferent, not
part of the local circuit. And L2 shrinks every synapse toward small-but-present,
which is not what "fewer wires" means. Every `wiring_cost` column in the existing
`pareto_runs/*` is that W_in quantity, so the reported front axis was wrong too,
not just the training signal.

**Fix.** `connectivity_reg(model, kind="l1", target="w_rec", inh_scale=1.0)`:

- penalizes the **effective W_rec** actually used in the forward pass, via the new
  `model.effective_w_rec()` (defined on both `DaleRNN` and `LeakyRNN`);
- **L1** by default, as the convex surrogate for a connection count;
- for Dale, uses `softplus(w_raw)` -- the actual synaptic magnitude. Penalizing
  `w_raw` would be meaningless, since a negative `w_raw` is a *weak* synapse, not
  a strong inhibitory one;
- excludes autapses from both the sum and the count for `DaleRNN` (structurally
  absent), but counts `LeakyRNN`'s diagonal (a real trainable weight there);
- normalizes per **counted synapse**, so lambda means "cost per synapse" and stays
  comparable if `n_neurons` changes;
- `target="w_in"` reproduces the legacy quantity for comparison, and
  `wiring_cost_w_in_l2` is still reported alongside the new cost.

**On exact sparsity.** `softplus` is strictly positive, so L1 alone can never
reach zero -- it only drives `w_raw` toward `-inf` (`softplus(-20) = 2e-9`).
Penalizing `softplus(|w_raw - t|)` does *not* fix this: `softplus(|z|) >= log 2`
and is *minimized at* `w_raw = t`, so it would pull every synapse toward a fixed
nonzero magnitude and actively prevent pruning. Instead:

- **training** uses plain L1 (smooth, no dead gradients);
- `DaleRNN(prune_eps=...)` / `--prune-eps` optionally applies
  `relu(softplus(w_raw) - eps)` in the forward pass, giving exact zeros. Default
  `0.0` = off, because pruning there is **irreversible**: below `eps` the gradient
  is zero and the synapse never comes back;
- `connection_fraction(model, thresh)` reports the fraction of synapses that are
  effectively present. Reported as `conn_frac`, never trained on -- L1 is the
  surrogate, this is the count itself.

---

## 2. An L1 on W_rec would preferentially prune inhibition

**Files:** `network.py` (`DaleRNN.ei_row_scale`), `train_cog.py` (`_w_rec_magnitude`)

**Issue.** `_init_dale_w_rec` multiplies inhibitory rows by `n_e / n_i` to balance
total E and I drive. Measured at `frac_e=0.8`:

```
mean |w| of E rows: 0.02809
mean |w| of I rows: 0.10994    ratio I/E = 3.91   (= n_e/n_i = 3.92)
L1 budget:  E rows 50%  |  I rows 50%   <- but I is only 20% of neurons
```

**Why it matters.** Each inhibitory synapse was ~4x more expensive than each
excitatory one, so the cheapest way to cut the penalty is to prune inhibition.
With ReLU units and no rate ceiling that means runaway excitation, and it destroys
exactly the E/I balance the canonical-microcircuit story is about.

**Fix.** Magnitudes are divided by `model.ei_row_scale()` (1 for E rows, `n_e/n_i`
for I rows) before the norm, equalizing *fractional* shrinkage pressure with a
single lambda. After the fix:

```
normalized mean |w|: E=0.02831  I=0.02824  ratio=0.998
L1 budget share:     E=80%  I=20%   (matches the row counts)
```

`--conn-inh-scale` reweights inhibition on top of this if you want to explore the
asymmetry deliberately. The init scaling itself was **not** weakened -- it *is* the
E/I balance, and weakening it would break the thing under study.

---

## 3. Accuracy arithmetically averaged circular angles

**File:** `train_cog.py` (`circular_mean`, `wrap_to_pi`, `batch_accuracy`)

**Issue.** `pred_go` was the plain mean over time of `atan2` output, which lives
on `(-pi, pi]`. For a target near `pi` the per-step decodes straddle the branch cut
and average to ~0 -- i.e. 180 deg away.

Verified with a network whose bump wobbled +-0.25 rad (14 deg, well inside the
36 deg criterion) at *every single* timestep:

```
theta= 2.80  worst per-step err= 14.3deg  ->  scored acc=1.00
theta= 3.00  worst per-step err= 14.3deg  ->  scored acc=0.00
theta= 3.14  worst per-step err= 14.3deg  ->  scored acc=0.00
theta= 3.19  worst per-step err= 14.3deg  ->  scored acc=0.00
theta= 3.40  worst per-step err= 14.3deg  ->  scored acc=1.00
```

**Why it matters.** A location-dependent accuracy floor whose width *grows with
how jittery the network is* -- so it penalized high-lambda runs twice, once for
real degradation and once for the wrap artifact. That directly bends the Pareto
front.

**Fix.** New `circular_mean(angles, mask)` = `atan2(<sin>, <cos>)` over the masked
steps, used for both the prediction and the target. Verified: all targets now
score 1.00, including `pi`.

**Note on the loss:** the task loss was already circular and needed no change.
The target is a Gaussian bump laid out on the ring with `get_dist` (circular
distance), so a 1 deg error at 180 deg already cost the same as 1 deg at 0 deg.
Only the *decoding* had the bug. `wrap_to_pi` (the wrap used for the final
comparison) was also already correct; `circular_abs` is kept as an alias since the
notebooks import it.

---

## 4. Accuracy never required the network to release fixation

**File:** `train_cog.py` (`batch_accuracy`)

**Issue.** `fix_ok` was checked only over steps where `y_loc < 0` (the pre-go
period). Nothing checked that the fixation output goes *down* after go onset.
Verified: a network with the correct ring bump that held its fixation unit at 0.95
forever scored **acc = 1.0**.

**Why it matters.** `network.get_perf` -- the correct implementation, already in the
repo but unused by the training path -- requires
`perf = should_fix*fixating + (1-should_fix)*corr_loc*(1-fixating)`: a go trial
must break fixation. Without that, a network that always saccades *and* always
holds fixation scores well on both trial types, which guts
`dmsgo`/`dmsnogo`/`dmcgo`/`dmcnogo`, where go-vs-nogo **is** the task.

**Fix.** `batch_accuracy` now applies the Yang criterion on the response window:

- fixation-hold trials (no go target): correct iff still fixating;
- go trials: correct iff location within `ang_thresh` **and** fixation released.

Two diagnostics were added so the two failure modes stay separable:
`loc_acc` (right place, ignoring release) and `release_acc` (let go at all).
Verified: the fixation-holding network now scores `acc=0.00` with
`loc_acc=1.00, release_acc=0.00`.

Fixation is also now read over the *response* window rather than the pre-go
window, which is where the nogo criterion actually lives.

---

## 5. Accuracy was scored on steps the loss refuses to grade

**Files:** `task.py` (`Trial.post_ons`), `train_cog.py` (`response_mask`)

**Issue.** Every task sets `check_ons = go_onset + 100 ms` and
`add_c_mask(post_ons=check_ons)`, so the 100 ms reaction-time transient after go
onset is deliberately ungraded by the loss. But `batch_accuracy` averaged over
*all* steps with `y_loc >= 0`, including that transient.

**Why it matters.** The metric penalized the network for exactly the window the
loss told it to ignore, smearing the response readout across the pre-response
transient and depressing measured accuracy.

**Fix.** `Trial.add_c_mask` now records `trial.post_ons`, and
`response_mask(trial, device)` builds the `(batch, T)` boolean response window
from it. All accuracy calls pass it. `batch_accuracy(resp_mask=None)` falls back to
the final timestep alone, which is what `network.get_perf` uses.

`trial_to_tensors` keeps its 4-tuple return so the notebooks are unaffected.

---

## 6. `easy_task` removed the integration from the integration tasks

**File:** `task.py` (`EASY_STIM_COH_RANGE`)

**Issue.** `easy_task=True` multiplied the coherence set by 10 (dm/contextdm) or 2
(delaydm/contextdelaydm), giving coherences up to 0.8 on a mean of 1.0:

```
dm1 coherence, easy=False: [0.01 0.02 0.04 0.08]
dm1 coherence, easy=True : [0.1  0.2  0.4  0.8 ]   -> per-step SNR up to ~20x
```

**Why it matters.** At that SNR `dm1`/`contextdm1` are solvable by a single-frame
comparison -- no accumulation, no recurrence needed. The "evidence integration" and
"context-gated integration" demand types in the `core5`/`sanity3` comments were
not being exercised, and accuracy saturated near ceiling, leaving the accuracy axis
of the Pareto front almost no dynamic range to trade against wiring cost.

**Fix.** One shared band, as requested, replacing both multipliers:

```python
EASY_STIM_COH_RANGE = np.array([0.10, 0.15, 0.20], dtype=np.float32)
```

Per-step SNR is now 2.5x-5.1x, so the tasks require genuine accumulation.
**Expect lower accuracy and possibly more training steps than before.**

---

## 7. Metabolic cost had no L1 option and no E/I knob

**File:** `train_cog.py` (`rate_reg`, `_population_weighted`)

**Issue.** The rate cost was hard-coded to `r_hist.square().mean()`.

**Why it matters.** ATP cost per spike is roughly linear in firing rate, which is
why the metabolic cost is conventionally L1. L2 barely penalizes a broadly active
population and heavily penalizes a few high firers -- that is a *homeostatic* cost,
not a metabolic one. It is the same L1-vs-L2 argument as #1, so it should be
deliberate rather than accidental.

**Fix.** `rate_reg(r_hist, model=None, kind="l2", inh_scale=1.0)` with
`--rate-kind {l1,l2}`. **L2 remains the default** for continuity with earlier runs.

**On the E/I knob -- being explicit about what it does.** Unlike the wiring cost,
there is no magnitude asymmetry to undo here: a plain `mean()` over units already
weights every neuron equally, which is per-unit fair. The E/I asymmetry for rates
is on the **benefit** side -- only E units are read out, so silencing an I unit is
"free" -- and no cost normalization fixes that. So `_population_weighted` weights E
and I by their population fractions, which at `inh_scale=1.0` reduces *exactly* to
the plain per-unit mean (verified: `0.49999` vs `0.49999`), and
`--rate-inh-scale < 1` is the knob that actually makes inhibitory activity cheaper.
The default is therefore a deliberate no-op; the mechanism and the knob are what is
new.

---

## 8. Mixed batches weighted tasks by trial duration

**File:** `train_cog.py` (`masked_mse`)

**Issue.** `concat_trials` right-pads short tasks up to the longest `tdim`, and
`masked_mse` took a plain `.mean()` over the whole padded tensor. Measured for
`sanity3` at batch 96:

```
merged tdim=165
  fdgo        frac graded=0.473   sum c_mask= 91008
  contextdm1  frac graded=0.442   sum c_mask= 88128
  dmsgo       frac graded=0.939   sum c_mask=135360
```

**Why it matters.** `dmsgo` got ~1.5x the gradient weight of the other two purely
because its trials are longer -- an accidental curriculum favouring long/delay
tasks. `generate_mixed_trials` equalized trial *counts* but not loss *weight*. And
because the penalties do not shrink with padding, the effective lambda relative to
the task loss varied with batch composition, adding noise to exactly the
monotonicity-over-lambda being chased.

**Fix.** `masked_mse(..., per_trial=True)` (default) divides each trial by its own
total mask weight, then averages over trials. Verified share of total loss:

```
             old      new
fdgo        33.8%    33.1%
contextdm1  35.1%    33.0%
dmsgo       31.1%    33.9%
```

`--no-loss-per-trial` restores the legacy behaviour. **This changes the loss scale**
(roughly `1/mean(c_mask)`, ~0.5-0.6x), which is one reason the lambda ranges need
recalibrating.

---

## 9. Dale trained and evaluated with zero recurrent noise

**Files:** `train_cog.py` (`DALE_DEFAULT_NOISE_LEVEL`), `pareto.py`

**Issue.** `noise_level` defaulted to `0.0` for `dale`, and
`evaluate_pareto_metrics` used `noise_on=False` plus `noise_level=0.0`. The Dale
network never saw recurrent noise at all.

**Why it matters.** Noise robustness is a principal reason biological circuits
carry redundant recurrent wiring. A noiseless objective rewards brittle,
hyper-tuned solutions, gives no credit to the redundancy a wiring penalty removes,
and therefore reports wiring as nearly free -- it removes the force on the other
side of the trade-off.

**Fix.** `DALE_DEFAULT_NOISE_LEVEL = 0.1` is now the training default (yang stays
at 1.0). Eval noise is exposed as `--eval-noise-level` (default `0.0`, keeping the
clean reproducible readout) and `--eval-input-noise`, so the front can be made to
reward robustness when you want that.

---

## 10. One seed per lambda, and no lambda = 0 anchor

**File:** `pareto.py`

**Issue.** Every grid point used `seed=args.seed` for both init and trial stream,
with no replication. And with log spacing from `1e-3` there was no `lambda = 0`
run.

**Why it matters.** A single unlucky init became a "Pareto point" with no error
bar, and the unregularized reference point that the whole claim is relative to was
missing (a log axis cannot contain 0).

**Fix.**

- `--n-seeds N` trains N models per lambda point with seeds `seed + k`. The front is
  computed on the across-seed **means**; `summary.csv` keeps the original column
  names holding those means and adds a `*_std` beside each, so existing analysis
  keeps working. Per-seed rows go to `runs.csv`, and `summary.json` carries them
  under `per_seed_rows`.
- `--include-lambda-zero` (**on by default**) prepends an unregularized anchor:
  `(fix_lr, 0.0)` for a connectivity sweep, `(0.0, fix_lc)` for a rate sweep,
  `(0, 0)` for the 2D grid. Disable with `--no-include-lambda-zero`.

---

## 11. `OBJECTIVE_MAXIMIZE` was missing `min_task_acc`

**File:** `pareto.py`

**Issue.** `pareto_maximize_flags` uses `OBJECTIVE_MAXIMIZE.get(name, False)`, and
`min_task_acc` had no entry -- so naming it as an objective would silently
**minimize** it and produce an inverted front. `pareto_analysis.ipynb` already
lists `min_task_acc` in `OBJECTIVE_LABELS`, so this was one edit from happening.

**Why it matters.** `min_task_acc` is the neuroscience-faithful task objective:
with `mean_acc`, the cheapest way to stay high as lambda rises is to abandon the one
hard task, so the front traces "how many tasks did it give up" rather than "same
competence, cheaper wiring."

**Fix.** Added `min_task_acc: True`, plus entries for `conn_frac` and
`wiring_cost_w_in_l2`. The default objective is **still `mean_acc`** -- switching it
is a separate analysis decision.

---

## 12. Minor / latent fixes

| Where | Issue | Fix |
|---|---|---|
| `train_cog.py` `connectivity_reg` | `getattr(m,"W_in",None) or getattr(m,"w_in",None)` worked only because `W_in` is absent; a model defining it would make `bool()` on a multi-element tensor raise | removed, replaced by explicit `target=` dispatch |
| `train_cog.py` `train` | `frac_silent` / `frac_saturated` in `history` came from the **last task in the loop only** (`act` was overwritten each iteration) | averaged across tasks |
| `train_cog.py` `sample_input_drive` | used `model.W_in.t()` and `model.b`; neither exists (`w_in`, `bias`), and `w_in` is already `(n_input, N)` so the transpose would not align | `x_t @ model.w_in + model.bias`, works for both models |
| `train_cog.py` `plot_preactivation_at_init` | guarded on `model.firing_rate` and used `model.rate_max`; no model has either, so it always printed "skipped" | uses `model._cell_act` and the 0.05 threshold from `activity_stats`; now runs for both backends |
| `task.py` `concat_trials` | kept `trials[0].epochs`, so a mixed trial's epoch dict silently described only the first task | `epochs = {}` plus `epochs_by_task` per rule (captured *before* mutating, since `merged is trials[0]`) |
| `train_cog.py` `evaluate_pareto_metrics` | rebuilt the eval config from `default_config()` defaults, silently discarding any non-default `dt`/`tau`/`sigma_x`/`ruleset` the model was trained with | new `eval_config_from()` copies the training config and swaps only the RNG; `n_eachring` mismatches now raise instead of passing silently |
| `train_cog.py` | no way to pick lambda ranges after the cost definitions changed | new `penalty_scales()` and `--report-scales`: prints each term's magnitude and the break-even lambdas |
| `train_cog.py` / `pareto.py` | training and the reported Pareto costs could drift apart | single `DEFAULT_REG` dict threaded through `train()` and `evaluate_pareto_metrics()`, so the front plots the functional the gradient saw; `summary.json` records it under `reg` |

---

## Still open (deliberately not changed)

- **`rho(W_rec)` is only set at init.** `network.py:scale_recurrent_to_rho` is
  applied once; the spectral radius then drifts freely, and a wiring penalty will
  systematically shrink it. Any "near-critical dynamics" claim holds at step 0 only.
- **L1 is not a wiring cost.** It penalizes synaptic *strength*; the biological cost
  is axonal volume, which scales with the *number* of connections and their
  *length*. There is no spatial embedding in the model, so nothing prefers *local*
  connections -- a weak, diffuse, globally-connected network has low L1 and zero
  locality. `conn_frac` is reported as the count-side metric; a genuine locality
  cost (`sum |w_ij| * d(i,j)` with neurons embedded in 1D/2D) and an L0 /
  connection-probability formulation are open.
- **The Pareto front still uses `mean_acc`** and has no feasibility filter, so a run
  where training collapsed to chance still enters as a legitimate low-cost point
  (see #11).
- **Task-variance normalization** in `analysis_of_network.ipynb` divides by each
  unit's peak after only an `ACTIVE_THRESH=1e-3` filter, so units just above
  threshold get their noise amplified to full scale and can form spurious clusters.
