# Model, costs and multi-objective training: theory and definitions

What every quantity in `cmc` means, and why it is defined that way. Functions are named
(`module.function`) instead of quoted by line, so this file does not go stale when code moves.
Default values are the ones used in every core5 run.

---

## 0. Symbols

| Symbol | In code | Meaning |
|---|---|---|
| **B** | `batch_size` | trials simulated in parallel (32 in training) |
| **T** | `tdim` | time steps in **one** trial (not the number of tasks) |
| **dt** | `config["dt"]` | ms per step, 20 |
| **τ** | `config["tau"]` | neuron time constant, 100 ms |
| **α** | `alpha = dt/τ` | leak per step, 0.2 |
| **N** | `n_rnn` | recurrent units, 256 |
| **n_E, n_I** | `model.n_e`, `model.n_i` | 204 excitatory, 52 inhibitory (`frac_e = 0.8`, E first) |
| **n_eachring** | ring width | units per stimulus / response ring, **16** |
| **n_rule** | `n_rule` | rule channels, 20 (all present even when fewer tasks are trained) |
| **W[pre, post]** | `h @ W` | weight convention: row = presynaptic, column = postsynaptic |

Real time ≈ T × dt.

---

## 1. Tasks (`cmc.task`)

### 1.1 Input: `n_input = 1 + 2·16 + 20 = 53`

```
index 0          fixation cue (1 while the animal must fixate)
indices 1..16    stimulus ring, modality 1
indices 17..32   stimulus ring, modality 2
indices 33..52   rule one-hot (which task)
```

A stimulus at angle θ is a Gaussian bump on its ring, peak 0.8, width π/8, using the circular distance
(`Trial.add_x_loc`, `get_dist`). In decision tasks the bump strength is scaled by the coherence. Input noise:
σ_x = 0.01·sqrt(2/α) during training.

### 1.2 Output: `n_output = 1 + 16 = 17`

```
index 0        fixation output: target 0.8 while fixating, 0.05 after release
indices 1..16  response ring: Gaussian bump (0.8 on a 0.05 baseline) at the target direction
```

`trial.y_loc` is the target direction, or −1 when the correct response is to keep fixating.

### 1.3 core5

| Task | Computation |
|---|---|
| fdgo | respond to the stimulus direction when fixation goes off |
| fdanti | respond in the opposite direction |
| dm1 | integrate two noisy stimuli in modality 1, choose the stronger |
| contextdm1 | stimuli in both modalities; the rule says attend modality 1 |
| dmsgo | delayed match-to-sample: hold stimulus 1 over a delay, respond only if stimulus 2 matches |

Decision-task coherences are `EASY_STIM_COH_RANGE = {0.10, 0.15, 0.20}`, so the answer cannot be read
off one frame and the network has to integrate evidence. All 20 Yang tasks exist (`rules_dict["all"]`).

**Mixed batches** (`generate_mixed_trials`): each batch has equal numbers of trials from every active
task. Shorter trials are padded to the longest one.

### 1.4 Shapes

`trial.x` (T, B, 53) and `trial.y`, `trial.c_mask` (T, B, 17) are converted by `train_cog.trial_to_tensors`
to the network layout: input (B, 53, T); targets and mask (B, T, 17).

---

## 2. Network (`cmc.network`)

### 2.1 Dynamics (both models)

$$h_t = (1-\alpha)\,h_{t-1} + \alpha\,\mathrm{ReLU}\big(x_t W_{in} + h_{t-1} W_{rec} + b + \sigma\,\xi_t\big)$$

- Euler discretisation of τ dh/dt = −h + f(input), α = dt/τ = 0.2.
- ReLU keeps rates ≥ 0, which Dale's law needs (tanh is refused).
- Recurrent noise: σ = sqrt(2/α)·`sigma_rec` (0.05) × `noise_level`. Dale training uses `noise_level`
  = 0.1 (`DALE_DEFAULT_NOISE_LEVEL`); evaluation is noise-free.
- Readout: y_t = sigmoid(h_t W_out + b_out).

`LeakyRNN` (Yang 2019) uses one fused, unconstrained `[W_in; W_rec]` and reads out from all units. It is
only the reference baseline.

### 2.2 `DaleRNN`: E/I constraints

| Constraint | Implementation | Why |
|---|---|---|
| Dale's law | `W_rec = softplus(w_raw) · sign_pre`, sign +1 for E rows, −1 for I rows, fixed | a neuron releases one transmitter type: all its outgoing synapses have the same sign. The network learns magnitudes only. |
| 80 / 20 | `frac_e = 0.8` | cortical E:I ratio |
| No autapses | diagonal masked to 0 at every forward pass | no self-connections |
| E-only readout | `y = sigmoid(h_E W_out)` | long-range output of cortex comes from pyramidal (E) cells |
| W_in | unconstrained sign | afferents are not part of the recurrent circuit we study |

### 2.3 Initialisation (`_init_dale_w_rec`)

1. Magnitudes = 0.5 · |random orthogonal matrix| (as in Yang). Zero diagonal.
2. Inhibitory rows × n_E/n_I (= 3.92), so total E and total I drive are equal at init.
3. Scale the signed matrix to spectral radius **g** (`target_rho`).
4. Store `w_raw = softplus⁻¹(magnitude)`.

**About g.** With g = 1 (near critical) the network can solve the tasks by changing a random matrix
only slightly: in the unconstrained control, corr(W, W₀) ≈ 0.98 even after 8k steps, so the final weights
mostly show the init. With **g = 0.3** the initial recurrence is weak and learning has to build the
recurrent structure. The latest runs use g = 0.3. ρ(W_rec) is only set at init and can drift.

W_in ~ N(0, 1/n_in), W_out ~ 0.01·N(0, 1), biases 0. Torch and numpy are seeded in the constructor, so a
network is determined by (settings, seed). This is exact on CPU only.

---

## 3. Task loss and accuracy (`cmc.train_cog`)

### 3.1 Loss mask (`Trial.add_c_mask`)

| Period | Weight | Why |
|---|---|---|
| first 100 ms | 0 | settling from the initial state |
| fixation and stimulus | 1 | hold fixation, don't respond early |
| first 100 ms after go | 0 | reaction time, deliberately ungraded |
| response period | 5 | the actual answer |
| fixation output channel | ×2 | fixation breaks are the most common error |

### 3.2 Per-trial normalised MSE (`masked_mse`)

$$\mathcal{L}_{task} = \frac{1}{B}\sum_b \frac{\sum_{t,o} c_{b,t,o}\,(\hat y_{b,t,o}-y_{b,t,o})^2}{\sum_{t,o} c_{b,t,o}}$$

Each trial is divided by its own mask total, so every trial counts equally. A plain mean over (B, T, o)
gives long tasks (dmsgo) more gradient only because they are longer.

The loss is already circular, because the target bump is built with a circular distance.

### 3.3 Trial accuracy (`batch_accuracy`)

Scored only in the graded response window (`response_mask` = the weight-5 region), as in Yang:

- **go trials** (y_loc ≥ 0): correct if the decoded direction is within **36° (π/5)** of the target
  **and** fixation is released (mean fixation output < 0.5 in the window);
- **no-go trials**: correct if still fixating.

Decoding: the population vector of the response ring, `atan2(Σ y_k sin θ_k, Σ y_k cos θ_k)`, then a
circular mean over the window, and the error is wrapped to [−π, π] (`wrap_to_pi`). So 181° vs 180° is 1°,
not 179°. Without the release requirement, a network that never lets go of fixation scored 100% on
dmsgo.

Evaluation: 10 fixed eval seeds (10000–10009) × 64 trials per task, no noise.

### 3.4 Task objective: `min_task_acc`

$$\text{min\_task\_acc} = \min_{\text{task}} \text{acc}_{\text{task}}$$

The **worst** task, not the mean. With 5 tasks, a network can drop one completely and still have a mean of
~0.8–0.9. A brain-like solution must do all of them. On core5 the worst task is always dmsgo, so
min_task_acc = dmsgo accuracy.

---

## 4. The two costs

Both are computed by the same functions in training and in evaluation (`DEFAULT_REG`), so the front
shows exactly the quantity the gradient saw.

### 4.1 Metabolic cost: `rate_reg`, mean(r²)

$$C_{met} = \frac{1}{B\,T\,N}\sum_{b,t,i} r_{b,t,i}^2$$

The average squared firing rate over neurons, time and trials.

**Why a rate cost at all.** Most of the brain's energy goes to action potentials and synaptic
transmission, both roughly proportional to firing rate (Attwell & Laughlin 2001). Activity is therefore the
first quantity a metabolic budget should limit.

**Why squared (L2) rather than mean rate (L1).**

- **What L2 does:** quadratic, so it barely penalises many weakly active neurons and heavily penalises a
  few strong firers. It pushes the network toward **distributed, moderate activity**, which is closer to
  a homeostatic constraint (keep every neuron in range) than to a strict energy count.
- **What L1 would do:** ATP per spike is roughly linear, so L1 (= mean rate, since rates ≥ 0) is the
  more literal metabolic cost. With ReLU, though, L1 has a constant gradient down to zero and silences
  units outright: it pushes towards a sparse code with many dead units, and I units are an easy target
  because they are not read out.
- **Smooth and standard:** L2 has a vanishing gradient near zero (no dead-unit pressure), is what Yang et
  al. used for their rate regulariser (`l2_h`), and keeps continuity with earlier runs.
- **Means, not sums:** the cost does not depend on N, T or B, so a budget means the same thing for any
  network size or task length.
- **E/I weighting:** `_population_weighted` weights E and I by their population fractions. With
  `inh_scale = 1` (default) this is exactly the plain mean. `inh_scale < 1` would make I activity cheaper.

L1 is implemented (`--rate-kind l1`) but unused.

### 4.2 Wiring cost: `connectivity_reg`, L1 of W_rec

$$C_{wire} = \frac{1}{|\mathcal{S}|}\sum_{(i,j)\in\mathcal{S}} \frac{|W_{rec}[i,j]|}{s_i},\qquad s_i = \begin{cases}1 & i \in E\\ n_E/n_I & i \in I\end{cases}$$

where 𝒮 = all off-diagonal synapses.

- **W_rec, not W_in:** the hypothesis is about the cost of the *recurrent circuit*. An earlier version
  penalised W_in with L2, which shrinks afferent projections and prunes nothing.
- **L1, not L2:** wiring cost is about *how many* connections exist (Chklovskii 2002). The number of
  nonzero weights (L0) is not differentiable. L1 is its closest convex surrogate and drives weights to
  exactly zero. L2 shrinks every weight a bit and keeps all of them.
- **E/I normalisation (s_i):** at init each I synapse is ~3.9× larger than an E synapse (§2.3). A raw L1
  would spend most of its budget on inhibitory rows and prune inhibition first, and with ReLU that means
  runaway excitation. Dividing by s_i gives E and I synapses the same *fractional* pressure.
- **Limits:** L1 counts weight, not wire length, so there is no spatial embedding. `conn_frac` (the
  fraction of synapses with normalised magnitude > 0.01) is reported as the connection density but never
  trained on.

---

## 5. Several objectives at once

### 5.1 Basics of multi-objective optimisation

We want three things at once: **high accuracy, low firing, low wiring**. They conflict: a network that
fires less or has fewer synapses eventually gets worse at the tasks. There is no single best network,
only trade-offs.

- **Dominance:** network A *dominates* B if A is at least as good on every objective and strictly better
  on at least one. Then there is no reason to prefer B.
- **Pareto set / front:** the networks that no other network dominates. Each is a different best
  trade-off. In objective space they form a surface: the **Pareto front** (`front.pareto_mask`).
- **The goal of the search:** find the whole front, not one point, and then ask how the circuit changes
  as you move along it.
- **Scalarisation:** collapse the objectives into one number and optimise that. The two standard ways are
  the weighted sum (§5.2) and the ε-constraint (§5.3).
- **Hypervolume:** the volume of objective space that the front dominates, up to a reference point. One
  number for "how good and how complete is this front". It is used to compare search methods.
- **Feasibility:** a network only counts if it actually does every task: **min_task_acc ≥ 0.6**
  (`FEASIBLE_MIN_TASK_ACC`). Without this rule, dead or chance-level networks are the cheapest of all and
  nothing can dominate them on cost (35 of 37 front points in the first core5 grid were such networks).
  0.6 is above always-fixate on dmsgo (~0.5) and well above a dead network on the others (~0.2–0.3).

The objectives as minimised by the search (`moo.objectives_from_metrics`):

$$F = \big(1-\text{min\_task\_acc},\;\log_{10} C_{met},\;\log_{10} C_{wire}\big),\qquad G = 0.6 - \text{min\_task\_acc} \le 0$$

The costs are on a log scale because they span orders of magnitude. The log is monotone, so the Pareto set
is unchanged.

### 5.2 Weighted sum: what λ is

$$\mathcal{L} = \mathcal{L}_{task} + \lambda_{rate}\,C_{met} + \lambda_{conn}\,C_{wire}$$

λ is an **exchange rate**: how much task loss the network may give up for one unit of cost reduction. For
a fixed (λ_rate, λ_conn), gradient descent finds one point on the front: the point where the front's slope
equals the ratio of the λs. Sweeping λ over a grid (`cmc.lambda_pareto`, 6×6) gives a set of points.

**Why this is not enough:**

1. **Convex part only.** A weighted sum can only reach points on the convex hull of the front. Any part
   that bends inward (non-convex) cannot be reached by any λ. Our front is non-convex at the
   feasibility cliff, exactly where the cheapest networks that still work are.
2. **λ has no meaning outside training.** Its units are "task loss per unit cost", and both scales change
   during training: the rate cost is ~26× smaller and the task loss ~30× larger at a trained solution
   than at init. Break-even λs measured at init overshoot by ~500× (rate) and ~20× (wiring). You cannot
   say "this network may spend X". You can only try a λ and see what cost comes out.
3. **Uneven coverage.** An even grid in λ gives a very uneven spread on the front. Many λs land on the same
   region, and whole regions are skipped. The 6×6 grid covers hypervolume 0.41; the budget search
   reaches 0.63 (~50% more).
4. **Grid cost explodes.** Every grid point is a full training run. Refining in 2D multiplies the runs,
   and most of them land in uninteresting regions.

### 5.3 Budgets: the ε-constraint

Turn costs into constraints:

$$\min\ \mathcal{L}_{task}\quad \text{s.t.}\quad C_{met} \le B_{rate},\quad C_{wire} \le B_{conn}$$

- Budgets are in the units of the costs themselves ("at most this much wiring"), so they are interpretable.
- Every point on the front, convex or not, is the solution for *some* pair of budgets.
- Two different mechanisms enforce them:

**Wiring budget: exact projection** (`project_w_rec_to_budget`). After every optimizer step, the
normalised magnitudes z = |W_rec|/s are projected onto the L1 ball {z ≥ 0, Σz ≤ B_conn·|𝒮|}. The
projection soft-thresholds every synapse by the same amount τ, chosen so the sum exactly hits the budget
(Duchi et al. 2008). Synapses below τ become ~0, i.e. are **pruned**. This is projected gradient descent:
the network is always inside the budget, and gradient descent decides *which* synapses are worth keeping.

**Rate budget: learned multiplier** (`BudgetConstraint`, an augmented Lagrangian). The rate cannot be
projected, because it is a property of the dynamics, not a parameter. Instead λ_rate becomes a learned
variable:

$$\text{term} = \lambda(t)\,C_{met} + \tfrac{\rho}{2}\,B\,\mathrm{relu}(v)^2,\qquad v = C_{met}/B - 1$$

$$\log\lambda \leftarrow \log\lambda + \eta\cdot\mathrm{clip}(\mathrm{EMA}(v), -1, 1)$$

Over budget, λ grows; under budget, it shrinks. So λ settles where the budget is just met. The quadratic
term only acts above budget and lets the method settle on non-convex parts of the front. Log space,
because useful λs span decades (the weighted-sum sweeps used 0.02–40).

**Why a learned multiplier fails for wiring** (tested, `--conn-budget-mode lagrangian`). The wiring cost
reacts to λ with a **lag** (weights shrink gradually under Adam) and a **threshold** (below some λ the
task gradient wins and nothing is pruned). While the cost is still catching up, the multiplier keeps
integrating the violation and winds up to 50–500× the needed value. By the time the cost responds, the
pressure is far too strong and the network is pruned to death. Proportional (PI) terms and a budget ramp
(Stooke et al. 2020) did not fix it. The projection avoids the problem, because there is no controller,
only a hard constraint. The rate does not have this problem: it responds within a few steps.

### 5.4 Searching budget space: NSGA-III (`cmc.moo`, pymoo)

The budgets are inputs; the front is in objective space. We need a search that decides **which budget
pairs to train**, so that the resulting networks spread evenly along the front.

- **Genetic algorithm:** a population of genomes x = (log₁₀ B_rate, log₁₀ B_conn). Each generation, new
  genomes are made by crossover and mutation from the best ones so far, and each is evaluated by **one
  full training run plus evaluation** (`runner.train_and_evaluate`).
- **Selection:** first by non-dominated sorting (front 1, front 2, ...), with feasibility handled by
  pymoo's constraint domination (feasible beats infeasible; among infeasible, the smaller violation
  wins).
- **Why NSGA-III and not NSGA-II:** NSGA-II keeps diversity with a crowding distance, which works poorly
  with three or more objectives. NSGA-III uses a fixed set of **reference directions** (Das–Dennis, evenly
  spread over the objective simplex) and keeps the solutions closest to each direction. The result is an
  evenly covered 3D front.
- **Why it beats the grid:** sampling is adaptive (training is spent near the front, not in dominated
  regions); budgets reach non-convex parts; feasibility is a proper constraint; coverage is even.
- **Why pymoo ask/tell:** we drive the loop ourselves (`algorithm.ask()` → train the networks in
  parallel on the GPU → `algorithm.tell()`). Given the seed, `ask()` is deterministic, so a stopped run
  can be resumed by replaying it from `runs.csv` without retraining.
- **Main run:** population 24 × 10 generations, core5. The weighted-sum genome (log λ) also exists, and
  was used only to validate the loop against the 6×6 grid.

### 5.5 From the front to circuits: zoom points (`cmc.moo_zoom`)

The front is built from one seed per budget. To analyse circuits, chosen budget points are retrained with
many seeds and every network is saved:

| Point | Isolates |
|---|---|
| control | no budget: what the tasks alone produce |
| rate_limited | cheap firing, dense wiring |
| middle | both moderately tight |
| wiring_limited | sparse wiring, high firing |
| knee | best compromise |

rate_limited, middle and wiring_limited lie on the same iso-accuracy curve. They are equally competent,
but a different cost binds, so differences between their circuits can be attributed to *which* cost is
limiting, not to how well the tasks are solved.

---

## 6. Task variance (functional identity of neurons)

Labels neurons by function: which task(s) a neuron works for. Whether such functional clusters
emerge, and how they change with the costs, is one of the main questions.

For neuron i and task A (`runner.activity_summary`, noise-free):

$$\mathrm{TV}_i(A) = \frac{1}{T_{post}}\sum_{t > \text{fix onset}} \mathrm{Var}_{\text{trials}}\big[r_{t,i}\big]$$

This is the variance across trials (stimulus conditions) at each time step, averaged over time. It is
high if the neuron's activity depends on *which* stimulus was shown in task A. It is not within-trial
noise.

- **Normalise** each neuron's vector by its maximum over tasks, so a neuron is described by *which*
  tasks drive it, not how strongly.
- **Active neurons:** summed TV above 1% of the network's maximum. The threshold is relative because a rate
  budget changes the absolute scale ~30×.
- **Cluster** the normalised vectors with k-means, choosing k by silhouette score (k = 2–10), as in Yang
  et al. 2019. A neuron's **label** is its cluster. Each cluster is named after its dominant task (the
  argmax of the cluster's mean normalised TV).

---

## 7. Connectivity analysis: E/I balance

All statistics use the trained recurrent matrix `W = model.effective_w_rec()` in the convention
**W[pre, post]**: a row holds a neuron's outgoing synapses, a column everything it receives. Blocks:
`EE`, `EI` (E pre → I post), `IE`, `II` (`motifs.blocks`), on magnitudes A = |W|.

### 7.1 E/I balance (per neuron)

For each neuron j, its total recurrent input from each population:

$$E_j = \sum_{i\in E} W[i,j], \qquad I_j = \sum_{i\in I} |W[i,j]|$$

These are column sums: the excitation and inhibition the neuron *could* receive if all inputs fired equally.
**Currents** weight each presynaptic row by that neuron's mean rate (averaged over trials, time and tasks):
$E^{r}_j = \sum_{i\in E} \bar r_i W[i,j]$, and the same for $I^{r}_j$.

Two numbers per network:

- **Ratio** `median_j(I_j / E_j)`, in weights and in currents. 1 = excitation and inhibition cancel;
  > 1 = inhibition-dominated. It is exactly balanced at init by construction (§2.3, I rows × n_E/n_I).
- **Detailed balance** `corr_j(E_j, I_j)`, the Pearson correlation across the 256 neurons. Positive means
  inhibition *tracks* excitation: neurons that get more excitation also get more inhibition. This is the
  cortical finding of a constant E/I ratio across neurons (Xue et al. 2014). Global balance (ratio ≈ 1)
  can hold with zero correlation; the correlation is the stronger, neuron-by-neuron claim.

Caveat: a positive correlation can also come from some neurons having more input of every kind. A
degree-preserving shuffle null (swap pairs (a→b, c→d) → (a→d, c→b) within a sign class) would rule this
out; it is still to do.

*Motif statistics, shuffle nulls, the learned change ΔW and the motif literature (formerly §7.2-§8) are
archived in `reports/archive/theory_motifs_2026-10-07.md`.*
