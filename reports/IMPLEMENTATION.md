# Implementation summary (for the deck)

Every number below is read from the code (`cmc/network.py`, `cmc/task.py`, `cmc/train_cog.py`,
`cmc/runner.py`, `cmc/moo.py`) as of 2026-10-02, not from memory. "Default" means the value used
in every core5 run (lambda grid, moo, zoom).

## 0. History (from git)

| Phase | Dates | What |
|---|---|---|
| 0. Hand-built CMC | Aug 6-7 | `archives/model.py`: a rate RNN with **hand-designed** E / PV / SST / VIP blocks (fixed block strengths and connection probabilities). Abandoned: it builds the motif in instead of asking whether it emerges. |
| 1. Port of Yang | Aug 23-26 | PyTorch port of Yang et al. 2019 `LeakyRNN` + task battery. "well, it doesnt work" (Aug 24), then "leakyRNN from Yang 2019 works for fdgo and delaygo" (Aug 26). |
| 2. Dale's law | Aug 26 | `DaleRNN`: "dale with relu works for all 3 tasks". Then tasks scaled up to 10 and to all 20 (Aug 26-29). |
| 3. Long runs, noise | Aug 30 - Sep 5 | Task variance analysis, SLURM on LRZ, 50k-step Yang and Dale checkpoints, recurrent-noise levels for Dale. |
| 4. Costs + lambda grid | Sep 10-22 | core5 / sanity3 batteries, rate and wiring penalties, 6x6 lambda Pareto sweep. |
| 5. Neuroscience audit | Sep 17 | 14 code/theory mismatches fixed (`FIXED_ISSUES.md`): wiring penalty on W_rec not W_in, circular decoding, fixation release, response window, easy-task coherence, seeding, ... Old runs not comparable. |
| 6. Multi-objective | Sep 24-30 | NSGA-III (pymoo) over cost **budgets** instead of lambdas. |
| 7. Zoom | Sep 30 - Oct 1 | 5 budget points x 10 seeds, saved networks, wiring / E-I analysis. |

## 1. Network (`DaleRNN`)

| | Value |
|---|---|
| Units | N = 256: **204 excitatory, 52 inhibitory** (`frac_e = 0.8`, E first) |
| Dynamics | leaky rate RNN, Euler: `h_t = (1-α) h_{t-1} + α · ReLU(x_t W_in + h_{t-1} W_rec + b + noise)` |
| Time constants | dt = 20 ms, τ = 100 ms, **α = 0.2** |
| Activation | ReLU (rates ≥ 0; tanh is refused because Dale needs nonnegative rates) |
| Dale's law | `W_rec = softplus(w_raw) · sign_pre`: the network learns magnitudes, the sign of each **presynaptic** neuron is fixed (+1 E, -1 I) |
| Autapses | none (diagonal masked to 0 every step) |
| Input weights `W_in` | unconstrained sign, init N(0, 1/n_in) |
| Readout | **from E units only**, `y = sigmoid(h_E W_out + b_out)`, init W_out ~ 0.01 · N(0,1) |
| Bias | trained, init 0 |
| Recurrent noise | Gaussian on the pre-activation, σ = sqrt(2/α) · 0.05 ≈ 0.158, scaled by `noise_level` = **0.1** for Dale training (≈ 0.016); 0 at evaluation |
| Convention | `h @ W`, so `W[pre, post]` |

**Initialisation of W_rec:**

1. Magnitudes = 0.5 · |random orthogonal matrix| (Householder, as in Yang).
2. Zero diagonal.
3. Inhibitory rows multiplied by n_E / n_I = **3.92**, so total E and total I drive are balanced at init.
4. Scaled so the spectral radius ρ(W_rec) = **1.0** (near critical).
5. Stored as `w_raw = softplus⁻¹(magnitude)`.

ρ is only set at init and can drift during training.

**Baseline `LeakyRNN`** (Yang): same dynamics, fused unconstrained `[W_in; W_rec]`, readout from all
units, recurrent noise level 1.0. Used only as a reference, not in the cost experiments.

Seeding: `torch.manual_seed(seed)` + `numpy.RandomState(seed)` in the constructor, so a network
is fully determined by (cost settings, seed). This is exact on CPU only; GPU (CUDA) is not bit-reproducible.

## 2. Tasks (Yang et al. 2019 battery)

**Input (53 channels):**

- 1 fixation channel;
- 2 stimulus rings (two modalities) of 16 units each, with a Gaussian bump of tuning width π/8 and peak 0.8;
- 20 one-hot rule channels, all present even when only 5 tasks are trained.

Input noise is σ_x = 0.01 · sqrt(2/α) ≈ 0.032 during training.

**Output (17 channels):**

- 1 fixation output: 0.8 while it should fixate, 0.05 after release;
- a 16-unit response ring with a Gaussian target bump of 0.8 on a 0.05 baseline.

**core5** (used for every cost experiment):

| Task | Computation |
|---|---|
| fdgo | respond to the stimulus direction when fixation goes off |
| fdanti | respond in the **opposite** direction |
| dm1 | integrate two noisy stimuli in modality 1, choose the stronger |
| contextdm1 | same, but stimuli in both modalities; the rule says attend modality 1 |
| dmsgo | delayed match-to-sample: remember stimulus 1 over a delay (200-1600 ms), respond only if stimulus 2 matches |

All 20 tasks are implemented (`rules_dict["all"]`). `sanity3` = fdgo, contextdm1, dmsgo.

`easy_task` coherence for the decision tasks is {0.10, 0.15, 0.20}. That keeps them real
evidence-integration problems: per-step SNR is ~2-3x, so the network cannot read the answer off one frame.

## 3. Loss and training

**Task loss:** masked MSE between output and target.

- Mask weights: 0 for the first 100 ms; 1 during fixation and stimulus; 0 for the first 100 ms
  after the go cue (reaction time, ungraded); 5 for the response period after that.
  The fixation output has double weight.
- Normalised **per trial**, so long tasks such as dmsgo do not get more gradient.
- Batches are **mixed**: every batch contains equal numbers of trials from each active task.

**Metabolic cost (rate):** `mean(r²)` over neurons, time and trials (L2). It is population-weighted
so E and I count by their population fraction; with the default this equals the plain mean.

**Wiring cost:** `mean(|W_rec| / row_scale)` over all off-diagonal synapses (L1). `row_scale` is
3.92 for I rows and 1 for E rows, so E and I synapses feel equal *fractional* pruning pressure.
Without it an L1 would prune inhibition first and give runaway excitation.

Reported but never trained on: `conn_frac`, the fraction of synapses with normalised magnitude > 0.01.

**Two ways to apply the costs:**

1. **Weighted sum (lambda grid):** `loss = task + λ_rate · rate + λ_conn · wiring`.
2. **Budget, the ε-constraint (moo):** no lambdas.
   - Rate ≤ budget is enforced by an augmented Lagrangian. A learned multiplier λ(t) is updated
     in log space after every step, with an EMA of the violation, clipped, plus a quadratic
     penalty above budget.
   - Wiring ≤ budget is enforced **exactly** by projecting W_rec onto the L1 ball after every
     optimizer step (Duchi et al. 2008 soft-threshold). Pruned synapses go to ~1e-8.
   - A multiplier for wiring winds up and prunes the network to death (tested), hence the projection.

**Optimizer and length:**

| | Value |
|---|---|
| Optimizer | Adam, lr 1e-3, gradient-norm clip 1.0 |
| Batch | 32 trials |
| Steps | 4000 for core5 cost runs; the early single-network checkpoints went to 50k |

## 4. Evaluation

**Seeds:** 10 fixed eval seeds (10000-10009) × 64 trials per task. Noise-free, with the same task
definition as training.

**Trial correct (Yang criterion), scored on the graded response window:**

- go trials: the decoded direction is within **36° (π/5)** of the target **and** fixation is
  released (fixation output < 0.5);
- no-go trials: still fixating.

Direction is decoded as the circular population vector of the output ring (atan2), and the angular
error is wrapped.

**Objectives:**

- `min_task_acc`: accuracy of the **worst** task, not the mean. On core5 this is always dmsgo.
- Metabolic cost and wiring cost, the same functions as in training.

**Feasibility:** a network is feasible if `min_task_acc ≥ 0.6`. Infeasible networks are excluded
from fronts; otherwise chance-level networks sit on the front as the "cheapest" points.

## 5. Experiments built on this

| Script | What | Main run |
|---|---|---|
| `cmc.lambda_pareto` | grid of (λ_rate, λ_conn) × seeds, metrics only | core5 6x6 |
| `cmc.lambda_zoom` | many seeds at chosen λ points, networks saved | |
| `cmc.moo` | NSGA-III. Genome = log (rate budget, wiring budget); objectives (1 - min_task_acc, log metabolic, log wiring); constraint min_task_acc ≥ 0.6. Each evaluation is one full training run. Das-Dennis reference directions. | pop 24 × 10 generations |
| `cmc.moo_zoom` | 5 budget points × 10 seeds, networks saved | `runs/moo_zoom/dale_core5_5pt_10seed` |

**Zoom points:**

- control: no budget;
- rate_limited, middle and wiring_limited: three points along the ~0.65 iso-accuracy curve;
- knee: the best compromise.

## 6. Results so far (one line each)

**Front:**

- The budget front is ~50% larger than the 6x6 lambda grid by hypervolume (0.63 vs 0.41).
- It is non-convex only at the feasibility cliff.
- The feasibility edge follows rate × wiring ≈ const.
- dmsgo (working memory) is the bottleneck in every network; the other 4 tasks are at ceiling.

**Under any cost:**

- The networks form more, more task-specific clusters: k ≈ 2.3 in the control vs 4-6 under a cost.
- A dmsgo-specific cluster appears.
- **E/I balance becomes targeted:** corr(E input, I input) goes from -0.35 to +0.2 to +0.4.
- **Inhibition dominates:** the rate-weighted I/E ratio is 0.88 in the control vs 1.6-2.6 under a cost.
- Inhibitory synapses get stronger.

**Open:** shuffle nulls for these effects, and cluster-to-cluster connectivity.

## 7. Known limits / deliberate choices

- ρ(W_rec) is only set at init.
- L1 is a proxy for connection count, not wire length: there is no spatial embedding.
- Only E/I identity exists: no layers and no interneuron subtypes are built in.
- The rate cost is L2 (homeostatic-like); L1, closer to ATP per spike, is available but unused.
- Under projection, the density of the four E/I blocks is set by the method, not learned.
- Results are exactly reproducible on CPU only, not on GPU.
