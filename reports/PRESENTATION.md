# Multitask Dale's-law RNNs under metabolic and wiring budgets

Atharv Suryawanshi, Helmholtz Munich

> Format: one `---` block = one slide. `Notes:` = what to say; not for the slide.
> Numbers come from `runs/` and `notebooks/` as of 2026-10-07. Anything marked **[UPDATE]** depends on the
> long-training front, which has not been rerun yet.
> Scope: E and I only. Not the full canonical microcircuit (that needs layers and PV/SST/VIP subtypes).
> The motif slides (old 1, 24-27 and motif parts of 2, 9, 10, 18, 28-30) are archived in
> `reports/archive/PRESENTATION_motifs_2026-10-07.md`.

---

## Part I: The question

---

## 2. Why costs?

- Cortical circuit structure could be built in by development (genetic wiring rules).
- Or it could be what **any** circuit converges to when it must:
  1. solve **many** tasks with the same neurons,
  2. pay for **activity** (metabolic cost),
  3. pay for **connections** (wiring cost).

> Question: what do metabolic and wiring budgets do to a network trained on many tasks? (1) Do functional
> task-variance clusters emerge that the unconstrained network lacks? (2) How do the E→E, E→I, I→E and I→I
> weight distributions change?

Notes: We deliberately do not try to explain the canonical microcircuit (Douglas & Martin). That is a
simplified wiring diagram of laminar circuits with three interneuron classes; our network has neither layers
nor subtypes. "Unconstrained" = the same Dale network with no costs.

---

## 3. The brain is on a budget

- **Metabolic cost**: the brain is ~2% of body mass, ~20% of energy; most of it goes to spikes and
  synaptic transmission (Attwell & Laughlin 2001; Laughlin & Sejnowski 2003). → sparse, low firing.
- **Wiring cost**: axons and dendrites take volume, material and conduction time
  (Cherniak 1994; Chklovskii et al. 2002; Bullmore & Sporns 2012). → few connections.
- **Functional demand**: one cortex serves many behaviours, so it cannot specialise for one.

Hypothesis: **circuit structure = solution to (many tasks) under (rate budget) × (wiring budget).**

---

## 4. What we need for the test

- A **network** with E and I cells (Dale's law, 80/20), rate units.
- A **battery of tasks**, not one task, so the solution is not task-specific.
- **Explicit costs** we can dial: firing (metabolic) and recurrent connectivity (wiring).
- **All** trade-offs between them, not one hand-picked setting → **Pareto front**.

---

## Part II: Prior work

---

## 5. Yang et al. 2019, *Nature Neuroscience*
"Task representations in neural networks trained to perform many cognitive tasks"

- One rate RNN trained on **20 cognitive tasks** at once: go / anti, delayed response, decision making,
  context-dependent DM, match-to-sample (DMS / DNMS / DMC).
- Inputs: fixation, two stimulus "rings" (two modalities), a one-hot **rule** input.
  Output: fixation + response ring (saccade direction).
- **Task variance** per neuron: how much a unit's activity varies across trials of task A.
  Normalised task-variance profiles → clustering → **functional clusters**
  (units specialised for groups of tasks).
- Findings: clustered, compositional representations; related tasks share clusters;
  continual-learning experiments.

Notes: We use their task battery and their main analysis tool almost unchanged. The PyTorch port reproduces
their LeakyRNN results (fdgo, delaygo) before we change anything.

---

## 6. What Yang 2019 does *not* ask

- Unconstrained weights: any unit can excite and inhibit (no Dale's law in the main model).
- Costs are a small L1/L2 regulariser for training stability, not an object of study.
- Result is about **function** (what clusters represent), not **circuitry**
  (who connects to whom, E vs I).

→ Gap: does cost shape the *connectivity* and the functional clusters, and do E and I cells take different roles?

Notes: Song, Yang & Wang 2016 introduced the E/I (Dale) RNN framework; Yang 2019 mostly used the
unconstrained one. We combine the two.

---

## 7. Khona et al. 2022/23 (Fiete lab)
"Winning the lottery with neural connectivity constraints: faster learning across cognitive tasks with
spatially constrained sparse RNNs" (*Neural Computation* 2023; bioRxiv 2022)

- Same family of cognitive tasks (Yang battery).
- Neurons placed in **space**; connectivity is **sparse and local**, with a wiring cost that grows with
  distance.
- Main result: spatially constrained sparse networks **learn faster** and generalise across tasks,
  and develop modular, localised structure, compared with dense / random-sparse networks.
- Message: **connectivity constraints are not just a cost, they are a useful inductive bias.**

Notes: Check the exact citation details before the talk. Related: Achterberg, Akarca et al. 2023
(*Nat. Mach. Intell.*, spatially embedded RNNs: modularity, small-world, energy-efficient codes emerge from
a wiring + communication cost).

---

## 8. What Khona et al. does *not* ask

- Wiring only: **no metabolic (firing) cost**, so no rate–wiring trade-off.
- One cost setting at a time: no map of the full trade-off.
- The question is learning speed / modularity, not **E/I circuit structure**.

---

## 9. Where we sit

| | Tasks | Dale's law | Metabolic cost | Wiring cost | Full trade-off | Read out |
|---|---|---|---|---|---|---|
| Yang 2019 | 20 | no (main model) | weak reg. | weak reg. | no | functional clusters |
| Khona 2022/23 | battery | no | no | yes, spatial | no | learning speed, modularity |
| **This work** | core5 (→ 20) | **yes, 80/20** | **budget** | **budget (L1, W_rec)** | **yes, Pareto front** | **functional clusters + E/I weights** |

Also related: Song, Yang & Wang 2016 (E/I RNNs for cognitive tasks); Achterberg et al. 2023 (spatially
embedded RNNs).

---

## Part III: What we do differently

---

## 10. Three changes

1. **Biological units**: Dale's-law RNN, 80% E / 20% I, readout from E cells only, no autapses.
2. **Two costs, treated as objectives**: metabolic (firing rate) and wiring (recurrent connectivity),
   next to task performance.
3. **Map the whole Pareto front**, then zoom in on networks that are equally good but limited by
   *different* costs, and compare their functional clusters and E/I weights.

> Key contrast: same competence, different binding cost → what changes?

---

## 11. Why a front, not a single λ

- Usual approach: `loss = task + λ_rate · rate + λ_conn · wiring`, pick λ by hand.
- Problems we hit:
  - the answer depends on λ, and the "right" λ is arbitrary;
  - a weighted sum can only reach the **convex** part of the front;
  - λ is in units of loss, not of cost: hard to say "the network may spend X".
- Our approach: **budgets** (ε-constraint). "Be as accurate as possible with firing ≤ B_rate and
  wiring ≤ B_conn", and search over budgets with a multi-objective optimiser.

---

## Part IV: Implementation (the important parts)

---

## 12. Network: `DaleRNN`

- N = 256 rate units: **204 E, 52 I**. dt = 20 ms, τ = 100 ms (α = 0.2), ReLU.
- `h_t = (1−α) h_{t−1} + α · ReLU(x_t W_in + h_{t−1} W_rec + b + noise)`
- **Dale's law**: `W_rec = softplus(w_raw) · sign_pre`. The network learns magnitudes; each presynaptic
  neuron's sign is fixed.
- Zero diagonal. Readout `y = σ(h_E W_out)` from E cells only.
- Init: |orthogonal| magnitudes, I rows × n_E/n_I (balanced E and I drive), spectral radius g = 1
  (exploratory g = 0.3 runs exist but are out of scope: low-rank-RNN regime).

---

## 13. Tasks: core5 from the Yang battery

| Task | Computation |
|---|---|
| fdgo | saccade to the stimulus when fixation goes off |
| fdanti | saccade **opposite** to the stimulus |
| dm1 | integrate two noisy stimuli, choose the stronger |
| contextdm1 | two modalities, rule says which one to attend |
| dmsgo | delayed match-to-sample: hold stimulus 1 over a delay, respond if stimulus 2 matches |

- Mixed batches (all tasks in every batch), per-trial normalised loss.
- Coherences kept low ({0.10, 0.15, 0.20}) so decision tasks need real evidence integration.
- All 20 tasks are implemented; core5 = the cheapest set that covers perception, decision,
  context and working memory.

---

## 14. The three objectives

- **Task**: `min_task_acc`, accuracy on the **worst** task (Yang criterion: within 36°, fixation correct).
  Not the mean: a network could drop one task and still score 0.9 on average.
- **Metabolic**: `mean(r²)` over neurons, time, trials.
- **Wiring**: `mean(|W_rec|)` (L1) over recurrent synapses, **E/I-normalised** so inhibition is not pruned
  first (otherwise L1 kills I and the network runs away).
- **Feasibility**: `min_task_acc ≥ 0.6`. Otherwise chance-level networks sit on the front as the
  "cheapest" solutions (35 of 37 front points in an early grid).

Notes: L1 on W_rec is the convex stand-in for the number of connections. It is not wire length; we have no
spatial embedding yet (that is where Khona comes back in).

---

## 15. Enforcing budgets during training

- **Wiring budget: exact projection.** After every Adam step, W_rec is projected onto the L1 ball of
  radius B_conn (Duchi et al. 2008 soft-threshold). Pruned synapses go to ~0.
- **Rate budget: learned Lagrange multiplier.** λ_rate(t) updated in log space from the (EMA) violation,
  plus a quadratic penalty above budget.
- Lesson learned: a multiplier for wiring does **not** work. Wiring reacts to λ with a lag and a threshold,
  the multiplier winds up 50–500× and prunes the network to death. Hence the projection.

---

## 16. Searching the front: NSGA-III

- Genome = (log B_rate, log B_conn). Each evaluation = **one full training run** + evaluation.
- Objectives: (1 − min_task_acc, log metabolic, log wiring), constraint min_task_acc ≥ 0.6.
- pymoo NSGA-III, population 24 × 10 generations, on GPU (LRZ).
- Compared against the 6×6 λ grid (weighted sum).

---

## 17. Zoom: from front to circuits

`cmc.moo_zoom`: pick budget points on the front, retrain **many seeds**, save every network.

| Point | What it isolates |
|---|---|
| control | no budget: what the tasks alone produce |
| rate_limited | cheap firing, dense wiring |
| middle | both moderately tight |
| wiring_limited | sparse wiring (~5% of synapses), high firing |
| knee | best compromise |

rate_limited / middle / wiring_limited lie on the same **iso-accuracy curve** (~0.65 at 4k steps).

---

## 18. Analysis toolkit

1. **Task variance + clustering** (Yang 2019): functional identity of every neuron.
2. **E/I currents**: per neuron, excitatory vs inhibitory input; balance = corr(E in, I in); I/E ratio.
3. **Training trajectories**: the same network saved at 2k / 4k / 8k / 16k / 32k steps.

---

## Part V: What we see

> **[UPDATE]** The front (slides 19–20) was trained at **4000 steps**. Later runs show 4k is too short:
> accuracy still climbs to ~16k steps, and the wiring budget's cost was mostly a *learning-speed* cost.
> Re-run NSGA-III with longer training (g = 1) and replace these two slides.

---

## 19. The front (4k steps) **[UPDATE]**

- Budget front is ~**50% larger** than the 6×6 λ grid by hypervolume (0.63 vs 0.41).
- Non-convex (unreachable by any λ) only at the feasibility cliff.
- Feasibility edge follows **rate × wiring ≈ const**: the two costs trade off against each other.
- **dmsgo (working memory) is the bottleneck in every network**; the other four tasks are at ceiling.

Figure: `notebooks/moo_pareto_analysis.ipynb` (3D front + 2D projections).

---

## 20. Zoom points (10 seeds, 4k steps) **[UPDATE]**

| Point | min task acc | metabolic | wiring | synapses kept |
|---|---|---|---|---|
| control | 0.87 | 0.96 | 0.027 | 76% |
| rate_limited | 0.66 | 0.0025 | 0.023 | 64% |
| middle | 0.73 | 0.0068 | 0.0083 | 23% |
| wiring_limited | 0.68 | 0.035 | 0.0027 | 6% |
| knee | 0.75 | 0.022 | 0.0045 | 12% |

Costs are pinned by the budgets; accuracy is what varies between seeds.

---

## 21. Training longer changes the cost of wiring

Control vs wiring_limited, g = 0.3, 5 seeds, one run saved along the way:

| min task acc | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|
| control | 0.08 | 0.63 | 0.90 | 0.95 | **0.97** |
| wiring_limited (~3% synapses at 32k) | 0.13 | 0.59 | 0.87 | 0.95 | **0.96** |

- With enough training, a network with ~3% of its synapses is as good as the dense one,
  at ~1/20 of the firing cost.
- At 4k steps the budget looked expensive; it mostly **slows learning**.
- → the front must be recomputed at convergence (this is the [UPDATE]).

Notes: Ties back to Khona: they found sparse spatial constraints *speed up* learning. At g = 0.3 we see no
speed-up (the curves are close; wiring_limited is slightly behind from 4k to 16k). Our sparsity is L1, not
spatial. The end result agrees with theirs: a sparse circuit is enough.

---

## 22. Constraints make neurons specialise

- Control: ~2 functional clusters (mean k = 2.3), most neurons mixed-selective.
- Any constrained point: **4–6 clusters**, each dominated by one task.
- A **dmsgo-specific cluster** appears that the control lacks.
- Share of neurons whose top task is dmsgo: ~11% (control) → 14–27% (constrained).
  Resources move to the bottleneck task.
- Caveat: silhouette ~0.31–0.33 everywhere: soft clusters, not discrete cell types.

---

## 23. E/I balance becomes targeted, inhibition dominates

- corr(E input, I input) per neuron: **−0.35** (control) → **+0.2 to +0.4** (constrained):
  neurons that get more excitation get more inhibition (detailed balance).
- Rate-weighted I/E: **0.88** (control) → **1.6–2.6** (constrained). Under a budget, I cells fire more
  than E cells.
- Inhibitory synapses get stronger.
- In the long run (wiring_limited, 32k): I/E input currents rise to ~6×.

---

## 28. Summary

1. A Dale RNN can solve 5 cognitive tasks with ~3% of its recurrent synapses and ~1/20 of the firing,
   given enough training.
2. Mapping the full trade-off (budgets + NSGA-III) finds ~50% more of the front than a λ grid.
3. Under cost: more specialised neurons, a working-memory cluster, targeted E/I balance,
   inhibition-dominated activity.

**[UPDATE]** once the long-training front is in: restate (1)–(2) with the new numbers.

---

## 29. Limitations

- core5 only; dmsgo is always the bottleneck → 20 tasks next.
- L1 counts connection strength, not **wire length**: no spatial embedding (yet).
- Only E vs I, no layers: by design, not the canonical microcircuit (which needs PV / SST / VIP).
- Spectral radius only set at init; rate cost is L2 (homeostatic-like), not spike-count L1.
- Under projection, the density of the four E/I blocks is set by the method, not learned.
- Reproducible bit-for-bit only on CPU.

---

## 30. Next steps

1. **Recompute the front at convergence** (≥16k steps, g = 1) → replace slides 19–20.
2. Stricter nulls for E/I balance (degree-preserving swaps).
3. **Spatial wiring cost** (distance-weighted, à la Khona / Achterberg): does locality change the structure?
4. All 20 tasks.
5. Later: interneuron subtypes (deferred).

---

## Backup: key references

- Yang, Joglekar, Song, Newsome & Wang (2019). Task representations in neural networks trained to perform
  many cognitive tasks. *Nat. Neurosci.*
- Khona, Chandra, Ma & Fiete (2023). Winning the lottery with neural connectivity constraints. *Neural
  Comput.* (bioRxiv 2022).
- Achterberg, Akarca, Strouse, Duncan & Astle (2023). Spatially embedded recurrent neural networks.
  *Nat. Mach. Intell.*
- Song, Yang & Wang (2016). Training excitatory-inhibitory RNNs for cognitive tasks. *PLoS Comput. Biol.*
- Xue, Atallah & Scanziani (2014). Equalizing excitation–inhibition ratios. *Nature*.
- Attwell & Laughlin (2001); Laughlin & Sejnowski (2003). Energy budget of the brain.
- Chklovskii, Schikorski & Stevens (2002); Bullmore & Sporns (2012). Wiring economy.
- Duchi et al. (2008). Projection onto the L1 ball. Deb & Jain (2014). NSGA-III.

## Backup: code map

`cmc/network.py` (DaleRNN) · `cmc/task.py` (battery) · `cmc/train_cog.py` (loss, budgets, projection) ·
`cmc/moo.py` (NSGA-III) · `cmc/moo_zoom.py` (seeds at points) · `notebooks/moo_zoom_analysis.ipynb` (zoom analysis)
