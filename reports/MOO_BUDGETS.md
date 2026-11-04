# MOO, budgets, and how they are enforced

## Summary

- **Two nested loops.** NSGA-III (pymoo) searches over just **two numbers**: a
  rate budget and a wiring budget. For each pair, **gradient descent (Adam)**
  trains a full 256-unit DaleRNN under those budgets. NSGA-III replaces the
  hand-set lambda grid. It does not replace gradient descent, and it can't do
  so practically (see §2).
- **Objectives** (all minimized): `1 - min_task_acc`, `log10 metabolic`,
  `log10 wiring`. **Constraint:** `min_task_acc >= 0.6`.
- **Metabolic cost** = mean squared hidden activity per unit per time step
  (L2 by default).
- **Wiring cost** = mean |W_rec| per possible synapse, with inhibitory rows
  divided by ~4 so E and I are treated alike.
- **Wiring budget is enforced exactly:** after every Adam step, W_rec is
  projected onto the L1 ball (every synapse is shrunk by the same amount, and
  weak ones reach zero).
- **Rate budget is enforced softly:** a penalty whose weight λ tunes itself
  (it rises when over budget, falls when under).
- **Main issues** (§6): with ReLU and an unpenalized input/readout, the rate
  budget can partly be met by rescaling; pruned synapses never regrow, and
  without a ramp they are chosen at random init; L1 limits total weight, not
  connection count or wire length; and an L2 rate cost is homeostatic rather
  than metabolic.

---

## 1. NSGA-III in one page

**Problem.** Genome **x** → objectives **F(x)** (minimize) and constraints
**G(x) ≤ 0**. A dominates B if A is no worse on every objective and strictly
better on at least one. The Pareto front is the set of non-dominated points.

**Outer loop** (every genetic algorithm):

```
P = N random genomes, evaluated
repeat n_gen:
    Q = N offspring of P      # tournament selection -> SBX crossover -> polynomial mutation
    evaluate Q                # here: N full RNN trainings
    P = survival(P ∪ Q, N)    # elitist: parents compete with children
```

Tournament selection prefers feasible solutions, then smaller constraint
violation. Duplicate offspring are regenerated (`dedup_eps` adds a looser
check).

**Survival (the NSGA-III part):**

1. **Non-dominated sorting** of the 2N candidates into fronts F1, F2, ….
   Whole fronts are added until the next one, F_l, doesn't fit.
2. **Normalization.** Subtract the ideal point (best value per objective).
   Find the extreme points with an achievement scalarizing function, put a
   hyperplane through them, and divide by its intercepts (the nadir
   estimate). pymoo falls back to the worst observed values when the plane is
   degenerate, which is common with small populations.
3. **Niching on reference directions.** Das-Dennis places evenly spaced points
   on the unit simplex: C(H+M-1, M-1) of them, which is 10 for M=3, H=3. Each
   solution is attached to its nearest ray. Remaining slots go first to the
   least-populated rays, so the front stays spread out.

Infeasible solutions are only used to fill slots left over after all feasible
ones, ordered by least violation. NSGA-III beats NSGA-II's crowding distance
when there are many objectives; at M=3 the two are about equivalent.

**ask/tell** ([cmc/moo.py](../cmc/moo.py)): `ask()` returns N genomes. Each is
trained outside pymoo (single, SLURM, or batched). The F and G results are
attached with `StaticProblem` + `Evaluator`, and `tell()` runs survival.
`ask()` is deterministic given the seed and the history of `tell()` calls,
which is what makes resuming possible.

**In practice here:** 100–250 evaluations is far below what NSGA-III was
designed for, so treat it as adaptive sampling that concentrates around the
front, not as a converged optimizer. Objectives are noisy across seeds, so a
lucky network can survive. The `moo_zoom` multi-seed points are the reliable
ones.

## 2. Why NSGA-III does not replace gradient descent

Evolving all ~90k weights directly is not realistic:

- One backward pass gives the exact descent direction in all dimensions. An
  evolutionary algorithm gets one scalar per evaluation, so it needs on the
  order of d evaluations to find a comparable direction.
- SBX crossover and polynomial mutation are built for genomes of up to ~100
  variables. Crossing over two trained RNNs gives garbage, because their
  hidden units aren't aligned (the permutation problem).
- Most of the difficulty is reaching feasibility (dmsgo needs working
  memory), which gradient descent through time does well and random search
  does very poorly.

The division of labour: **gradient descent finds the best network for a given
trade-off; NSGA-III chooses which trade-offs to ask for.** Gradient-based ways
to cover the front in fewer runs exist, e.g. preference-conditioned networks /
Pareto hypernetworks and MGDA.

## 3. How the costs are calculated

### Metabolic cost: `rate_reg` (`cmc/train_cog.py`)

On hidden activity `h` of shape `(batch, T, 256)`:

```
per_unit  = h**2                 # kind="l2" (default); "l1" -> |h|
rate_cost = 0.8 * mean_E(per_unit) + 0.2 * inh_scale * mean_I(per_unit)
```

With `inh_scale = 1` this is just the mean over units, time steps and trials.
It depends on the trials: training uses the current batch, and the reported
`metabolic_cost` is averaged over evaluation trials of all tasks.

### Wiring cost: `connectivity_reg` (`cmc/train_cog.py`)

Weights only, no activity:

```
W       = softplus(w_raw) * sign          # Dale
mag_ij  = |W_ij| / row_scale_i            # row_scale = 1 (E), n_e/n_i ≈ 4 (I)
wiring  = sum(mask * mag) / sum(mask)     # mask excludes autapses
```

Without `row_scale`, the init's E/I balance scaling makes inhibitory synapses
~4× larger, and L1 would prune inhibition first, causing runaway excitation.
`connection_fraction` (the share of synapses with normalized magnitude
> 0.01) is reported but never trained on.

## 4. From genome to training

Genome = (log rate budget, log wiring budget), within rate ∈ [0.002, 0.08] and
wiring ∈ [0.002, 0.025] (`cmc/moo.py` defaults). Each genome becomes
`train(..., rate_budget=b_r, conn_budget=b_w)`. There are no lambdas: this is
an epsilon-constraint method, so it can reach non-convex parts of the front
that no weighted sum can.

## 5. How the budgets are enforced

### Wiring: exact L1-ball projection (default `conn_budget_mode="projection"`)

The wiring cost is not in the loss. After every Adam step,
`project_w_rec_to_budget`:

1. Computes the normalized magnitudes `z = mask * |W| / row_scale`.
2. Does nothing if `sum(z) <= budget * n_synapses`.
3. Otherwise finds the one threshold τ with
   `sum(max(z - τ, 0)) = budget * n_synapses` (Duchi et al. 2008, sort-based,
   exact) and soft-thresholds every synapse by τ.
4. Writes back `w_raw = softplus⁻¹(z_new * row_scale)`. Zeroed synapses
   become ~1e-8, i.e. pruned.

The wiring cost is **≤ budget after every step**. Because τ is applied in
normalized units, E and I rows shrink in equal fractions.

### Rate: learned multiplier (`BudgetConstraint`)

Activity can't be projected, so the constraint goes into the loss:

```
v       = rate_cost / budget - 1
penalty = λ * rate_cost + (ρ/2) * budget * relu(v)**2
```

After each step (in log space, smoothed with an EMA, clipped):

```
v_ema   = 0.9 * v_ema + 0.1 * v
log λ  += 0.01 * clip(v_ema, -1, 1)     # integral controller; kp = 0 by default
λ ∈ [1e-6, 1e4], λ_init = 0.01
```

Over budget, λ rises; under budget, it falls, so the network isn't pushed
below its budget. The quadratic term only acts above budget; it is the
augmented-Lagrangian part. The result is approximately on budget, not
strictly below it.

### Why the two are enforced differently

| | Rate | Wiring |
|---|---|---|
| Depends on | activity (trials, noise) | weights only |
| Projectable? | no | yes (L1 ball) |
| Response to pressure | fast | slow, lagged, threshold-like |
| Method | learned λ | exact projection |

A multiplier on wiring (`--conn-budget-mode lagrangian`) was tested. λ keeps
rising while the weights catch up, winds up to 50–500× the needed value, and
prunes the network to death. Neither the `kp` nor the `ramp` option fixed
this.

## 6. Issues with the current approach

Overall the framework (epsilon-constraint, projection + adaptive multiplier)
is sound. The concerns are about what the costs measure. Ranked by how much
each could change conclusions:

### 6.1 The rate budget can be met by rescaling (most important)

ReLU is positively homogeneous. With h₀ = 0, scaling `W_in` and `bias` by c
scales every activity by c (W_rec unchanged), which lowers the L2 rate cost by
c². Scaling `w_out` by 1/c then leaves the output unchanged. None of `W_in`,
`bias`, `w_out` is penalized. The only thing that breaks the symmetry is the
fixed-size recurrent noise, so what the rate budget really costs the network
is **signal-to-noise**. That is a defensible biological story, but it should
be a stated claim, not an accident.

*Fix:* freeze `w_out` (`--freeze w_out --w-out-init fanin`), or give `W_in` /
`w_out` a norm budget. Report input/readout norms next to the rate cost.
Check whether runs with a trainable `w_out` show those norms scaling with the
rate budget.

### 6.2 Pruning is irreversible and starts at random init

A projected-out synapse has `w_raw ≈ -18`. Its gradient goes through
sigmoid(-18) ≈ 1e-8, below Adam's eps, so it effectively never regrows. With
`budget_ramp = 0` (the default), the first projection cuts the random-init
W_rec to the budget, so in tight-budget networks part of the structure is
chosen by init magnitudes rather than by the tasks.

*Fix (cheapest first):*
- Use `--budget-ramp`. The projection already follows the ramp schedule.
- Parameterize magnitudes as `clamp(w, 0)` instead of softplus, which makes
  this proper projected gradient descent and lets synapses regrow.
- DEEP R rewiring (Bellec et al. 2018): a fixed synapse count with random
  regrowth, close to biological synaptic turnover.

### 6.3 L1 limits total weight, not wiring

The budget can't tell many weak synapses from a few strong ones, which are
biologically different costs. `conn_frac` shows which one the network chose,
but the budget doesn't control it.

| Alternative | Constrains | Effort |
|---|---|---|
| Top-k projection | exact synapse count (L0) | trivial, drop-in |
| Hard-concrete gates (Louizos et al. 2018) | expected count | moderate |
| Distance-weighted L1, Σ\|W_ij\|·d_ij (seRNN, Achterberg et al. 2023) | wire length | moderate, needs a spatial layout |

The seRNN option is closest to the project's hypothesis: modularity and
small-world structure emerge from that cost. Any of these redefines the wiring
axis, so new runs won't be comparable to existing fronts.

### 6.4 L2 rate cost is homeostatic, not metabolic

L2 punishes a few strong firers far more than broad weak activity. With ReLU,
L1 is the mean firing rate, which is the cleanest metabolic (ATP) quantity.

*Fix:* use `--rate-kind l1` for the main analysis, or at least check that the
main results survive the switch.

### 6.5 Minor

- The rate budget is soft. This is harmless because the objectives use the
  measured cost, but the genome-to-objective mapping is noisier; log the final
  violation.
- Projecting after an Adam step is not strict projected gradient descent
  (different geometry). Rarely matters in practice.
- The multiplier is fine for rate: rates respond quickly, so the wiring-style
  wind-up does not occur.

### Suggested order

1. Diagnose: in existing checkpoints, compare `W_in` / `bias` / `w_out` norms
   against the rate budget (§6.1).
2. Cheap fixes: projection ramp, frozen fanin `w_out`, and an L1-rate check
   at the zoom points.
3. Decide (with the supervisor) whether to move wiring to a count- or
   length-based cost.
