# Multitask Dale's-law RNNs under metabolic and wiring budgets
### How excitation and inhibition organise into task modules

Atharv Suryawanshi, Helmholtz Munich

> Format: one `---` block = one slide. `Notes:` = what to say; not for the slide.
> **[to be removed]** = does not fit the current direction (E/I task modularity, 20 tasks, 150k steps).
> These slides are kept for now and not deleted.
> **[UPDATE]** = the content stays, but the numbers or the figure still have to be filled in.
> New results come from `notebooks/ei_modularity_150k.ipynb` and `reports/ei_modularity/` (Nov 2026),
> run `runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k`.
> Scope: E and I only. Not the full canonical microcircuit (that needs layers and PV/SST/VIP subtypes).
> The motif slides are archived in `reports/archive/PRESENTATION_motifs_2026-10-07.md`.

---

## Part I: The question

---

## 2. Why costs?

- Cortical circuit structure could be built in by development (genetic wiring rules).
- Or it could be what **any** circuit converges to when it must:
  1. solve **many** tasks with the same neurons,
  2. pay for **activity** (metabolic cost),
  3. pay for **connections** (wiring cost).

> Question: when a Dale network solves 20 tasks under metabolic and wiring budgets, how are its
> excitatory and inhibitory neurons organised into task modules? (1) Do functional task-variance clusters
> emerge that the unconstrained network lacks? (2) Is inhibition shared, part of each module, or between
> modules, and which budget decides?

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
- A way to read out **task modules for E and I separately**, and whether the wiring follows them.

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
- Clusters are found over all units together: no separate E and I modules.
- Result is about **function** (what clusters represent), not **circuitry**
  (who connects to whom, E vs I).

→ Gap: do costs shape the functional modules, how does inhibition relate to them, and does the wiring
follow the modules?

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
- No Dale's law: modularity, but not the role of **inhibition** in the modules.

---

## 9. Where we sit

| | Tasks | Dale's law | Metabolic cost | Wiring cost | Full trade-off | Read out |
|---|---|---|---|---|---|---|
| Yang 2019 | 20 | no (main model) | weak reg. | weak reg. | no | functional clusters |
| Khona 2022/23 | battery | no | no | yes, spatial | no | learning speed, modularity |
| **This work** | **20** | **yes, 80/20** | **budget** | **budget (L1, W_rec)** | **yes, Pareto front** | **E and I task modules + their wiring** |

Also related: Song, Yang & Wang 2016 (E/I RNNs for cognitive tasks); Achterberg et al. 2023 (spatially
embedded RNNs).

---

## 9b. Inhibition in cortical modules (literature)

- **Like-to-like E→E**: E cells with similar tuning connect more (Ko et al. 2011; MICrONS, Ding et al. 2025).
- **Tuned E↔I**: PV cells inhibit most strongly the E cells that drive them and share their tuning, which
  gives feedback inhibition within an ensemble and competition between ensembles (Znamenskiy et al. 2024).
- **E–I assemblies from plasticity**: plasticity on both E→I and I→E is needed to wire I cells into their
  own E assembly (Mackwood, Naumann & Sprekeler 2021).
- **Specific inhibitory targeting at connectome scale** (Schneider-Mizell et al. 2025).

→ The open question is not whether cortex has modules, but **what role inhibition plays in them**. We ask
which cost produces which role.

Notes: Details and references in `reports/theory.md` §8.

---

## Part III: What we do differently

---

## 10. Three changes

1. **Biological units**: Dale's-law RNN, 80% E / 20% I, readout from E cells only, no autapses.
2. **Two costs, treated as objectives**: metabolic (firing rate) and wiring (recurrent connectivity),
   next to task performance.
3. **Map the whole Pareto front**, then train many seeds at a few budget points (no budget, rate only,
   wiring only, both) and compare how E and I organise into task modules.

> Key contrast: which cost binds → what role inhibition plays.

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
- Zero diagonal. Readout `y = σ(h_E W_out)` from E cells only, W_out trained, "fanin" init (randn / √204).
- Init: |orthogonal| magnitudes, I rows × n_E/n_I (balanced E and I drive), spectral radius g = 1
  (exploratory g = 0.3 runs exist but are out of scope: low-rank-RNN regime).
- Training: **150k steps**, batch 32. The networks compared are trained together in one batched run
  (`cmc.batched`, shared trial stream).

---

## 13. Tasks: core5 from the Yang battery **[to be removed]**

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

## 13b. Tasks: all 20 Yang tasks

| Family | Tasks |
|---|---|
| Go | Go, RT Go, Dly Go |
| Anti | Anti, RT Anti, Dly Anti |
| Decision making | DM 1, DM 2, Ctx DM 1, Ctx DM 2, MultSen DM |
| Delayed decision making | Dly DM 1, Dly DM 2, Ctx Dly DM 1, Ctx Dly DM 2, MultSen Dly DM |
| Matching | DMS, DNMS, DMC, DNMC |

- Mixed batches (all 20 tasks in every batch), per-trial normalised loss.
- Coherences kept low ({0.10, 0.15, 0.20}) so decision tasks need real evidence integration.
- The hardest task is usually **DNMS**, so it sets `min_task_acc`.

Notes: Show the task GIFs from `reports/task_gifs/` here.

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

Notes: Details in `reports/MOO_BUDGETS.md`.

---

## 16. Searching the front: NSGA-III

- Genome = (log B_rate, log B_conn). Each evaluation = **one full training run** + evaluation.
- Objectives: (1 − min_task_acc, log metabolic, log wiring), constraint min_task_acc ≥ 0.6.
- pymoo NSGA-III on GPU (LRZ). 20-task run: population 48 × 3 generations, 50k steps each
  (144 networks, 118 feasible, 27 on the front).

**[UPDATE]** figure of the 20-task front (`runs/moo/dale_all_nsga3_budget_p48g3_50k`) with the four points.

---

## 17. Zoom: from front to circuits **[to be removed]**

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

## 17b. Four budget points, 5 seeds each

| Point | Rate budget | Wiring budget | Isolates |
|---|---|---|---|
| control | none | none | what the tasks alone produce |
| A3 | 0.0011 | none | rate only |
| B0 | none | 0.0054 | wiring only |
| C0 | 0.0115 | 0.0044 | both |

What the trained networks spend:
- A3 fires **~250× less** than the control.
- B0 and C0 use **~6× less wiring**.
- Synapses kept: control 70%, A3 61%, **B0 6%**, **C0 0.7%**.
- All solve the tasks: mean accuracy > 0.95 everywhere. The worst task (usually DNMS) is 0.96 in the
  control and 0.85–0.88 at the budget points.

Figure: `reports/ei_modularity/fig/fig1_points.pdf`.

---

## 18. Analysis toolkit **[to be removed]**

1. **Task variance + clustering** (Yang 2019): functional identity of every neuron.
2. **E/I currents**: per neuron, excitatory vs inhibitory input; balance = corr(E in, I in); I/E ratio.
3. **Training trajectories**: the same network saved at 2k / 4k / 8k / 16k / 32k steps.

---

## 18b. Finding E and I task modules

1. **Task variance** (Yang 2019) per unit and task → 20 × 256 table. Drop inactive units; normalise each
   unit by its maximum → **task profile**.
2. **Cluster E and I separately** (k-means, k by silhouette) → E groups and I groups, each with a mean
   task profile.
3. **Tasks per neuron** = number of tasks with profile > 0.5: how broad a neuron is.

Figure: the top panels of `fig_net_*.pdf` (units sorted by group, bright = involved in that task).

Notes: Yang clusters all units together. Clustering E and I separately is what lets us ask how inhibition
relates to the excitatory modules.

---

## 18c. Three ways inhibition could relate to E modules

| Type | Inhibition | Tuning | Wiring |
|---|---|---|---|
| 1. shared inhibition | a common pool, inhibits everyone | I broader than E | ignores tasks |
| 2. mixed E+I modules | each module has its own I cells (feedback) | I group matches an E group | I ↔ its own E group |
| 3. inhibition between modules | I cells suppress *other* modules (lateral) | I group matches an E group | I → other E groups |

Types 2 and 3 look the same in tuning. **Only the wiring separates them.**

Figure: `reports/ei_modularity/fig/fig2_types.pdf`.

---

## 18d. Measuring whether the wiring follows the tasks

For every (I group, E group) pair:
- **task similarity** r = correlation of the two task profiles;
- **wiring strength** = mean |w| from one group to the other (pruned = 0), as a **z-score against 500
  shuffles** of the group labels (group sizes kept).

Then **ρ = Spearman(similarity, wiring)** over all pairs, separately for I→E and E→I:
- ρ > 0 → type 2 (wired to the same-task group);
- ρ < 0 → type 3 (wired to other-task groups);
- ρ ≈ 0 → type 1.

The same for **E→E and I→I**: within-group ÷ between-group wiring (> 1 = the group wires to itself).

Plus **lesions**: silence one I group and check whether the accuracy drop lands on that group's own tasks.

Figure: `reports/ei_modularity/fig/fig3_method.pdf` (C0, seed 2).

---

## Part V: What we see

> **[to be removed]** The note below and slides 19–23 are the core5, 4k-step results.

> **[UPDATE]** The front (slides 19–20) was trained at **4000 steps**. Later runs show 4k is too short:
> accuracy still climbs to ~16k steps, and the wiring budget's cost was mostly a *learning-speed* cost.
> Re-run NSGA-III with longer training (g = 1) and replace these two slides.

---

## 19. The front (4k steps) **[to be removed]**

- Budget front is ~**50% larger** than the 6×6 λ grid by hypervolume (0.63 vs 0.41).
- Non-convex (unreachable by any λ) only at the feasibility cliff.
- Feasibility edge follows **rate × wiring ≈ const**: the two costs trade off against each other.
- **dmsgo (working memory) is the bottleneck in every network**; the other four tasks are at ceiling.

Figure: `notebooks/moo_pareto_analysis.ipynb` (3D front + 2D projections).

---

## 20. Zoom points (10 seeds, 4k steps) **[to be removed]**

| Point | min task acc | metabolic | wiring | synapses kept |
|---|---|---|---|---|
| control | 0.87 | 0.96 | 0.027 | 76% |
| rate_limited | 0.66 | 0.0025 | 0.023 | 64% |
| middle | 0.73 | 0.0068 | 0.0083 | 23% |
| wiring_limited | 0.68 | 0.035 | 0.0027 | 6% |
| knee | 0.75 | 0.022 | 0.0045 | 12% |

Costs are pinned by the budgets; accuracy is what varies between seeds.

---

## 21. Training longer changes the cost of wiring **[to be removed]**

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

## 22. Constraints make neurons specialise **[to be removed]**

- Control: ~2 functional clusters (mean k = 2.3), most neurons mixed-selective.
- Any constrained point: **4–6 clusters**, each dominated by one task.
- A **dmsgo-specific cluster** appears that the control lacks.
- Share of neurons whose top task is dmsgo: ~11% (control) → 14–27% (constrained).
  Resources move to the bottleneck task.
- Caveat: silhouette ~0.31–0.33 everywhere: soft clusters, not discrete cell types.

---

## 23. E/I balance becomes targeted, inhibition dominates **[to be removed]**

- corr(E input, I input) per neuron: **−0.35** (control) → **+0.2 to +0.4** (constrained):
  neurons that get more excitation get more inhibition (detailed balance).
- Rate-weighted I/E: **0.88** (control) → **1.6–2.6** (constrained). Under a budget, I cells fire more
  than E cells.
- Inhibitory synapses get stronger.
- In the long run (wiring_limited, 32k): I/E input currents rise to ~6×.

---

## 24. A rate budget sharpens tuning

| | control | A3 rate | B0 wiring | C0 both |
|---|---|---|---|---|
| E tasks per neuron | 4.7 | **2.9** | 5.2 | 3.5 |
| I tasks per neuron | 4.8 | 4.1 | 4.8 | 4.0 |
| E silhouette | 0.19 | **0.28** | 0.23 | 0.24 |
| I silhouette | 0.24 | **0.31** | 0.24 | 0.26 |

- **Rate budget (A3, and C0)**: the most specialised E neurons, many small E groups, several dedicated to
  one task (RT Go, RT Anti, DMS, DNMS, DMC).
- **Wiring budget alone (B0)**: the broadest neurons, few large groups.
- **Control**: broad groups = task *families* (all DM, all delayed DM, the matching tasks).
- Every I group has an E group with a similar profile (best match r ≈ 0.7–0.9 everywhere): I cells are
  always tuned. The points differ in how broad I is, and above all in the wiring.

Caveat: silhouettes 0.17–0.37: overlapping groups, not discrete cell types.

---

## 25. Control: inhibition between modules (type 3)

- I→E wiring vs task similarity: **ρ = −0.44 ± 0.16, negative in all 5 seeds** (−0.31 to −0.71).
- An I group inhibits the E groups of *other* tasks more than chance, and its own E partner less.
  Example, seed 1: I4 (17 units) shares the tasks of E3–E6 (r 0.6–0.9) but sends them weaker synapses than
  chance (z −2 to −3.6).
- E→I is unstructured (ρ ≈ 0).
- I neurons are not broader than E (4.8 vs 4.7 tasks), so it is not shared inhibition.
- 70% of synapses exist: this is about synapse **strength**, not presence.

Figure: `reports/ei_modularity/fig/fig_net_control.pdf`.

Notes: Lateral inhibition between task modules is what the unconstrained network learns.

---

## 26. A3, rate only: modules in tuning, not in wiring (type 1)

- Sharpest tuning of all points, but **wiring ignores the tasks**: ρ within ±0.06 of 0 in every seed, in
  both directions.
- I neurons are broader than E (4.1 vs 2.9 tasks) and form fewer groups, at the level of task families.
- Recurrent weights stay dense (61%) and close to random.
- Lesions do the least damage of all points.

Figure: `reports/ei_modularity/fig/fig_net_A3.pdf`.

---

## 27. B0, wiring only: weak E+I modules, and the I part matters

- Wiring leans towards same-task groups: ρ ≈ +0.2 in both directions, but **seed-dependent** (2 of 5 seeds
  clearly positive in both).
- E neurons are the broadest (5.2 tasks), few large groups.
- **Lesions**: silencing an I group hurts its *own* tasks most clearly of any point
  (ρ = 0.46 ± 0.07, every seed 0.39–0.55).
- 6% of synapses left: the measure now also reflects *which* synapses exist.

Figure: `reports/ei_modularity/fig/fig_net_B0.pdf`.

---

## 28. C0, both budgets: mixed E+I modules (type 2)

- **E→I wiring follows the tasks in every seed**: ρ = +0.46 ± 0.13 (+0.32 to +0.63). An E group connects
  mostly to the I groups of the same tasks.
- I→E also positive in every seed, but weaker (+0.10 to +0.33).
- Sharp E tuning (3.5 tasks), I only slightly broader (4.0).
- **< 1% of synapses left**: this is about which connections the network *kept*.
- The only point that matches type 2 in both tuning and wiring.
- But: lesioning one I group alone barely hurts its own tasks (ρ 0.11). The I part is not individually
  needed.

Figure: `reports/ei_modularity/fig/fig_net_C0.pdf`.

Notes: Type 2 is the E–I assembly structure found in V1 (Znamenskiy et al. 2024) and predicted from
plasticity (Mackwood et al. 2021).

---

## 29. E→E and I→I: do the modules wire to themselves?

| | control | A3 rate | B0 wiring | C0 both |
|---|---|---|---|---|
| E→E within ÷ between (shuffled ≈ 1.0) | 1.07 ± 0.04 | 1.11 ± 0.07 | 1.24 ± 0.25 | **5.2 ± 1.8** |
| I→I within ÷ between (shuffled ≈ 1.0–1.15) | 0.90 ± 0.04 | 0.93 ± 0.07 | **1.59 ± 0.27** | **3.4 ± 2.4** |

- **C0: E neurons of one group excite each other ~5× more** than they excite other groups, in every seed
  (4.0–8.2). The most self-wired groups are the delayed-DM and matching groups, i.e. the tasks that must
  hold information over a delay.
- **I groups inhibit themselves under a wiring budget** (B0 1.6×, every seed 1.4–2.0; C0 3.4×, noisy
  because I groups are small). Without one, slightly below 1.
- A rate budget alone gives no self-wired groups.
- Between different groups, wiring barely follows task similarity (ρ −0.2 to +0.2): the structure is almost
  all **within** groups.

Figure: section 6 of `notebooks/ei_modularity_150k.ipynb` (E→E | E→I / I→E | I→I panels).

---

## 30. Summary of the E/I modularity results

| | control | A3 rate | B0 wiring | C0 both |
|---|---|---|---|---|
| E tasks per neuron | 4.7 | 2.9 | 5.2 | 3.5 |
| ρ for I→E | **−0.44** | 0.01 | 0.19 | 0.18 |
| ρ for E→I | 0.04 | 0.01 | 0.20 | **0.46** |
| E→E within ÷ between | 1.07 | 1.11 | 1.24 | **5.2** |
| synapses kept | 70% | 61% | 6% | 0.7% |
| closest type | **3**, between modules | **1**, shared | weak 2 | **2**, E+I modules |

- **The role of inhibition changes with the budget.**
- **No budget**: competitive. I groups suppress the E groups of other tasks.
- **Rate budget**: modules in **tuning** (the sharpest neurons), but the wiring does not follow them.
- **Wiring budget**: modules in **wiring**: E groups excite themselves, I groups inhibit themselves, and
  I is wired into its own E group.
- **Both budgets (C0)**: the only point with modules in tuning *and* in wiring.

Figure: `reports/ei_modularity/fig/fig4_summary.pdf`.

---

## 31. Summary (core5, 4k steps) **[to be removed]**

1. A Dale RNN can solve 5 cognitive tasks with ~3% of its recurrent synapses and ~1/20 of the firing,
   given enough training.
2. Mapping the full trade-off (budgets + NSGA-III) finds ~50% more of the front than a λ grid.
3. Under cost: more specialised neurons, a working-memory cluster, targeted E/I balance,
   inhibition-dominated activity.

---

## 32. Summary

1. A Dale RNN solves all 20 Yang tasks (mean accuracy > 0.95) with < 1% of its recurrent synapses and
   ~1/27 of the firing (C0; mean metabolic cost 0.018 vs 0.49 in the control).
2. Budgets + NSGA-III map the full trade-off between accuracy, firing and wiring.
3. **Rate cost → sharp tuning. Wiring cost → modular wiring. Both → E+I task modules**, with inhibition
   wired into its own module.
4. Without costs, inhibition acts **between** modules (lateral, competitive).

---

## 33. Limitations

- core5 only; dmsgo is always the bottleneck → 20 tasks next. **[to be removed]**
- 5 seeds per point. Each network's ρ comes from its own groups (12–170 (I, E) pairs), and the number of
  groups changes from seed to seed. Claims are made only where the seeds barely overlap.
- Groups overlap (silhouette 0.17–0.37): soft modules, not discrete cell types.
- In the control and A3 almost every synapse exists, so their wiring numbers describe weight strength only.
- I groups are small (2–8 units), so I→I numbers are noisy.
- The four points come from one batched run (shared trial stream); not comparable to single-run networks.
- L1 counts connection strength, not **wire length**: no spatial embedding (yet).
- Only E vs I, no layers: by design, not the canonical microcircuit (which needs PV / SST / VIP).
- Spectral radius only set at init; rate cost is L2 (homeostatic-like), not spike-count L1.
- Pruned synapses never regrow (`reports/MOO_BUDGETS.md` §6).
- Reproducible bit-for-bit only on CPU.

---

## 34. Next steps

1. **Recompute the front at convergence** (≥16k steps, g = 1) → replace slides 19–20. **[to be removed]**
2. Stricter nulls for E/I balance (degree-preserving swaps). **[to be removed]**
3. **Spatial wiring cost** (distance-weighted, à la Khona / Achterberg): does locality change the structure?
4. All 20 tasks. **[to be removed]**
5. Later: interneuron subtypes (deferred).

**[UPDATE]** next steps for the E/I modularity direction.

---

## Backup: key references

- Yang, Joglekar, Song, Newsome & Wang (2019). Task representations in neural networks trained to perform
  many cognitive tasks. *Nat. Neurosci.*
- Khona, Chandra, Ma & Fiete (2023). Winning the lottery with neural connectivity constraints. *Neural
  Comput.* (bioRxiv 2022).
- Achterberg, Akarca, Strouse, Duncan & Astle (2023). Spatially embedded recurrent neural networks.
  *Nat. Mach. Intell.*
- Song, Yang & Wang (2016). Training excitatory-inhibitory RNNs for cognitive tasks. *PLoS Comput. Biol.*
- Znamenskiy et al. (2024). Functional specificity of recurrent inhibition in visual cortex. *Neuron*.
- Mackwood, Naumann & Sprekeler (2021). Learning excitatory-inhibitory neuronal assemblies in recurrent
  networks. *eLife*.
- Ko et al. (2011). Functional specificity of local synaptic connections. *Nature*. Ding et al. (2025).
  *Nature*. Schneider-Mizell et al. (2025). *Nature*.
- Xue, Atallah & Scanziani (2014). Equalizing excitation–inhibition ratios. *Nature*.
- Attwell & Laughlin (2001); Laughlin & Sejnowski (2003). Energy budget of the brain.
- Chklovskii, Schikorski & Stevens (2002); Bullmore & Sporns (2012). Wiring economy.
- Duchi et al. (2008). Projection onto the L1 ball. Deb & Jain (2014). NSGA-III.

## Backup: code map

`cmc/network.py` (DaleRNN) · `cmc/task.py` (battery) · `cmc/train_cog.py` (loss, budgets, projection) ·
`cmc/moo.py` (NSGA-III) · `cmc/moo_zoom.py` (seeds at points) · `cmc/batched.py` (batched training) ·
`notebooks/ei_modularity_150k.ipynb` (E/I modularity, lesions) · `notebooks/ei_groups_figures.ipynb` (figures) ·
`reports/ei_modularity/make_figures.py` (report figures) ·
`notebooks/moo_zoom_analysis.ipynb` (core5 zoom analysis) **[to be removed]**
