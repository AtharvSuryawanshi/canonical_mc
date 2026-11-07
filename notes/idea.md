# Idea: do cost budgets make neurons divide the tasks among themselves?

## Original (rough)
A Dale RNN, trained on multiple tasks. How does it learn the task. Are the tasks divided amongst the
neurons or they are mixed? How do constraints on firing/wiring affect this?

## My question
In Dale RNNs trained on 20 cognitive tasks, does a metabolic (firing-rate) budget make excitatory
neurons more task-specialized than in unconstrained networks, and does a wiring budget do the same?

## Why it's worth asking
In the cortex it costs to fire and to wire: both use energy and space, so cortical circuits are
shaped by function and by cost together (Attwell & Laughlin 2001; Chklovskii et al. 2002). Networks
trained on many tasks without costs already form task-variance clusters (Yang et al. 2019), and
connectivity constraints are known to change multitask RNNs (Khona et al. 2023; Achterberg et al.
2023). What is not separated is *which* cost pushes neurons from mixed to task-specialized.

Earlier exploratory analyses in the wider project motivated this question, so the expectation below
is informed by them; this slice tests it with a fixed metric, controls and replication.

## What I expect, and why
- **A3 and C0 < control:** a rate budget makes activity expensive on every task a neuron joins, so
  it should pay to let each neuron carry fewer tasks.
- **B0 ≈ control:** a wiring budget (L1 on W_rec) prunes *connections*; it limits who talks to whom,
  not how many tasks a neuron's activity varies with. Any effect on selectivity would be indirect.
- **Specialization exists without any budget:** unconstrained multitask networks already form
  task clusters (Yang et al. 2019). The claim is about the *shift* under a budget, not its emergence.

## What I'm unsure about
- **Mechanism.** The rate cost is L2 (`rate_kind = "l2"`). The textbook sparse-coding argument is
  for L1, and L2 on its own favours spreading activity thinly, not concentrating it. So "rate cost ->
  specialization" is not automatic; it may come from the rectified (non-negative) rates, or from
  something else. This analysis can show *whether*, not *why*.
- **Accuracy.** Budgeted networks can be less accurate than unconstrained ones. Fewer tasks per
  neuron could just mean a network that does less, not one that divides the work.
- **Who is counted.** Only active units enter the metric. If a rate budget silences units, the
  population changes between conditions.
- **Which budget matters more** when both are on (C0) cannot be
  separated with four points; that needs the front, not this slice.
- **n = 5 networks per condition.** The smallest possible two-sided Mann-Whitney p for 5 vs 5 is
  ~0.008 (complete separation). Report effect sizes and every seed, not only a p-value.

## Data (no new training)
- `runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k/`: 4 points x 5 seeds, 150k steps, 20 tasks,
  256 units (80% E), W_out trained, fan-in init. Budgets from `summary.json`:

  | point   | rate budget | wiring budget | meaning            |
  |---------|-------------|---------------|--------------------|
  | control | none        | none          | unconstrained      |
  | A3      | 0.00111     | 1.0 (loose)   | rate-limited       |
  | B0      | 10 (loose)  | 0.00536       | wiring-limited     |
  | C0      | 0.0115      | 0.00439       | both               |

- Replication: `runs/moo_zoom/dale_all_4pt_12seed_50k/` (control / rate_limited / wiring_limited /
  knee, 12 seeds, 50k steps). Different budget values and training length: a check of direction,
  not of exact numbers.
- Each checkpoint stores `task_variance` (20 tasks x 256 units) and `mean_rate`, computed in
  `cmc.runner.activity_summary`: noise-free eval trials, variance across trials per time step after
  fixation onset, averaged over time and eval seeds (Yang et al. 2019).

## Metric
- Active unit: total task variance > 1% of the network's max (as in `ei_modularity_150k.ipynb`).
- Normalize each unit's task variance by its own max over tasks (so a quieter unit is not
  automatically "more selective").
- **Breadth** = number of tasks with normalized variance > 0.5 (range 1-20). Low = specialized
  ("tasks divided"), high = mixed. Report per network: mean over active E units (I separately).
- Unit of analysis is the network (seed), not the neuron: n = 5 per condition.
- Implementation: `cmc/selectivity.py` (new), run by the Snakefile.

## Wrong if
Decided before the pipeline runs. Each line is checked against the 150k set unless it says otherwise.

1. **No rate effect:** A3's 5 seeds overlap control's 5 seeds in E breadth (any A3 seed >= the
   lowest control seed).
2. **Threshold artefact:** the ordering A3 < control does not hold at threshold 0.3 *and* 0.7.
3. **Accuracy, not budget:** across all 20 networks, breadth is explained by min task accuracy as
   well as by condition. Concretely: the most accurate A3 seeds sit inside the control range.
4. **Population artefact:** A3 loses so many active E units that the units left are not comparable
   (active E count below half of control's).
5. **Does not replicate:** in the 12-seed 50k set, rate_limited is not below control.
6. **Wiring does specialize** (the claim "B0 ≈ control" is wrong): B0's seeds are all below
   control's, by the same rule as 1.

If 1-5 all pass, the claim is: *a rate budget, not a wiring budget, makes E neurons in this model
carry fewer tasks*. If any fails, the report says which one, and the claim is narrowed accordingly.

## Checks
- Ground truth: synthetic task variance with known breadth -> the function recovers it.
- Threshold sweep 0.3 / 0.5 / 0.7: does the ordering hold? (Wrong-if 2)
- Accuracy confound: breadth vs min_task_acc across seeds. (Wrong-if 3)
- Active-unit count per condition. (Wrong-if 4)
- Replication on the 12-seed 50k set. (Wrong-if 5)
- Delete the derived outputs, rerun -> identical.

## References
- Yang, Joglekar, Song, Newsome & Wang (2019). Task representations in neural networks trained to
  perform many cognitive tasks. *Nat. Neurosci.*
- Khona, Chandra, Ma & Fiete (2023). Winning the lottery with neural connectivity constraints.
  *Neural Comput.*
- Achterberg, Akarca, Strouse, Duncan & Astle (2023). Spatially embedded recurrent neural networks.
  *Nat. Mach. Intell.*
- Attwell & Laughlin (2001). An energy budget for signaling in the grey matter of the brain.
  *J. Cereb. Blood Flow Metab.*
- Chklovskii, Schikorski & Stevens (2002). Wiring optimization in cortical circuits. *Neuron.*

## Plan
<!-- filled after the spec is agreed (plan mode), before building -->
