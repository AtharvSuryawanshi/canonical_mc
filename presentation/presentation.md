# Multitask Dale's-law RNNs under metabolic and wiring budgets

> Format: one `---` block = one slide. `Make:` = how the slide is built (Figma / Beamer / animate) and which
> file it uses. `Notes:` = what to say; not for the slide. Built in `presentation/main.tex`.
> Based on `reports/PRESENTATION_old.md`.
>
> **Names (2026-11-20):** on slides the four points are *control*, *firing constraint* (A3), *wiring constraint*
> (B0) and *both constraints* (C0); never "rate only" / "wiring only" / "unconstrained". Codes stay in these notes.
> **Two kinds of figures:** analysis figures in `results/front/` (detailed, for us; unchanged) and talk figures in
> `presentation/figs/talk/` (drawn at slide size, big labels, little text) from `presentation/figs/talk_figs.py`
> (`python presentation/figs/talk_figs.py [frames front convergence like2like silhouette tv]`). Slides 10–18 use the
> talk figures; detail left off them is in the notes below. Accuracy colour: RdYlGn, 0 red to 1 green, always.

---

## 0. Title

Multitask Dale's-law RNNs under metabolic and wiring budgets

Atharv Suryawanshi, Helmholtz Munich

Make: Beamer `\titlepage`.

---

## 1. One circuit, many tasks

[Fig1: many tasks → many neurons → **Shared?** (every neuron helps with every task) or **Separate?**
(each group of neurons handles its own task); bar profile of one neuron's involvement per task]

Make: Figma frame `Fig1_intro_question` (page "Talk figures") → `figs/fig1_intro_question.pdf`.
Earlier, more detailed version (20 tasks, 5 families): `Fig1_intro` → `figs/fig1_intro_detailed.pdf`, kept for
the setup slide.

Notes:
- The brain uses the same neurons for many different tasks.
- So how does one circuit divide the work between its neurons?
- Shared: every neuron takes part in every task; a task is a pattern over the whole population.
- Separate: groups of neurons specialise; each group handles its own task, like modules.
- Point at the bars: one neuron, how much it takes part in each task. Broad bars = shared, one tall bar = separate.
- Both are seen in the brain: mixed-selective neurons in prefrontal cortex, and task-specialised groups of
  neurons. So the question is open, and it may depend on the conditions the circuit works under.
- Recordings cannot show us every neuron and every synapse. A trained recurrent network can: we train one
  network on many tasks and look inside it.
- No details yet: which tasks, how many, and which conditions come on the next slides.

---

## 2. Yang et al. 2019: one network, 20 tasks

[Left: Yang Fig. 1b, inputs → network → outputs. Right: Yang Fig. 2a–c, one unit's activity and task variance,
task-variance heatmap with clusters.] One short caption under each, full reference at the bottom.

Make: Beamer, two columns. Figures are vector crops of `papers/yang2019.pdf`:
`figs/yang2019_crops.tex` → `figs/yang2019_crops.pdf` (page 1 = Fig. 1b, page 2 = Fig. 2a–c).

Notes:
- The closest prior work, and the one we build on most: Yang et al. trained one recurrent network on 20
  cognitive tasks at the same time.
- Every trial: a fixation input, two stimulus rings (two modalities), and a rule input that says which task
  to do. Output: keep fixating, then respond with a direction on the response ring.
- Their tool: task variance. For each neuron and each task, how much the neuron's activity varies across
  trials of that task, i.e. how much it takes part in the task (panels a, b).
- Normalise each neuron's task variances and cluster the neurons (panel c): the network forms clusters, each
  specialised for a group of related tasks.
- So for slide 1's question, an unconstrained network lands somewhere towards "separate".
- But their neurons can excite some targets and inhibit others, and nothing costs anything. Next: what
  happens with real E and I neurons and with costs.

---

## 3. Song, Yang & Wang 2016: excitatory and inhibitory neurons

[Left: Song Fig. 1, E/I network schematic. Right: model equations with Dale's law, two bullets.]

$$\tau\dot{\mathbf{x}} = -\mathbf{x} + W^{\text{rec}}\mathbf{r} + W^{\text{in}}\mathbf{u} + \text{noise},\quad
\mathbf{r} = [\mathbf{x}]_+,\quad \mathbf{z} = W^{\text{out}}\mathbf{r}_{\text{E}}$$
$$W^{\text{rec}} = W^{\text{rec},+} D,\quad W^{\text{rec},+} \geq 0,\quad D = \mathrm{diag}(+1,\dots,+1,\,-1,\dots,-1)$$

Make: Beamer, two columns. Figure: vector crop `figs/song2016_crops.tex` → `figs/song2016_crops.pdf`
(page 3 of `papers/song2016.pdf`). Equations are their Eqs 1–3 and 12 (rectified sum, `W^rec = W^{rec,+} D`).
E blue / I red on this slide follows their figure.

Notes:
- Yang 2019's main network has no cell types: one neuron can excite some targets and inhibit others.
  Real cortical neurons cannot.
- Song et al. 2016 is a framework paper: how to train rate RNNs with biological constraints built in, and
  show they still learn the tasks monkeys do.
- Dale's law: each neuron is either excitatory or inhibitory, about 4 E for every I. They build it into the
  weights: a non-negative matrix times a fixed sign per presynaptic neuron (D). Training changes the
  strengths, never the signs.
- Readout only from E cells, because long-range projections in cortex are excitatory. Self-connections are
  set to zero. They also allow masks to fix which connections exist.
- They trained one network per task (perceptual decision making, context-dependent integration, multisensory
  integration, parametric working memory, eye-movement sequences), and the activity resembles recordings.
- What we take: the E/I network, the E-only readout. Yang 2019 asked about many tasks without E/I; Song 2016
  had E/I but one task at a time. We need both, which is what lets us ask what inhibition does.

---

## 4. Real neurons pay for firing and for wiring

- **Firing costs energy.** Spikes and synaptic transmission dominate the brain's energy budget, so activity has
  to be sparse. *(Attwell & Laughlin 2001; Lennie 2003)*
- **Wiring costs space.** Axons and dendrites fill most of the grey matter, so connections have to be few.
  *(Chklovskii, Schikorski & Stevens 2002)*
- **… and the circuit must still solve its tasks.**

Make: Beamer, three bullets, small grey citations. No math here: the cost definitions go on the setup slide.

Notes:
- Yang: many tasks. Song: E and I neurons. Neither network pays anything for its activity or its connections.
  Real circuits do.
- Firing: action potentials and synaptic transmission are the largest items in the grey-matter energy budget.
  The cost per spike limits how many neurons can be active at once: sparse activity.
- Wiring: axons and dendrites take up most of the cortical volume; circuits keep wiring short and connections
  few, given the connectivity they need.
- But a circuit that is cheap and useless is no good: it still has to solve its tasks. Three demands at once
  (perform, fire little, wire little): how to trade them off is the next slide.

---

## 5. Setup

[Figma setup schematic: input panels (fixation, two stimulus rings, rule) → 53 input neurons → 256 recurrent
(E triangles, I circles) → 17 outputs → fixation output and response; cost bars.]
One line: 256 rate neurons (204 E, 52 I), all 20 Yang tasks trained together in one network, 150k steps.

Make: Figma frame `Fig_setup` (page "Talk figures"; a copy of the Page 1 schematic) → `figs/fig_setup.pdf`.

Notes:
- One trial: fixation input on, stimuli on one or both rings (16 directions each), a one-hot rule input says
  which of the 20 tasks. Inputs: 1 + 16 + 16 + 20 = 53.
- 256 recurrent neurons, 80% E, 20% I. Output: fixation unit plus a 16-unit response ring = 17.
- The network must hold fixation, then respond in the right direction when fixation goes off.
- All 20 tasks are trained together, mixed in every batch, as in Yang 2019.
- Bottom right: every network is scored on accuracy and on its two costs.

---

## 6. The network

$$\mathbf{r}_t = (1-\alpha)\mathbf{r}_{t-1} + \alpha\big[W^{\text{in}}\mathbf{u}_t + W^{\text{rec}}\mathbf{r}_{t-1} + \mathbf{b} + \sigma\boldsymbol{\xi}_t\big]_+,\quad \alpha = \Delta t/\tau = 20/100$$
$$W^{\text{rec}} = \big(\mathrm{softplus}(\Theta)\odot(\mathbf{1}-I)\big)D,\quad D = \mathrm{diag}(+1\times204\ \text{E},\ -1\times52\ \text{I})$$
$$\hat{\mathbf{y}}_t = \mathrm{sigmoid}(W^{\text{out}}\mathbf{r}^{\text{E}}_t + \mathbf{b}^{\text{out}})$$

Bullets: softplus(Θ) ≥ 0 learned strengths, D fixed presynaptic sign (Dale), 1 − I no autapses; readout from E
only, W_in / W_rec / W_out all trained; Adam lr 1e-3, batch 32, 150k steps, spectral radius 1 at init.

Make: Beamer, equations + three bullets. Checked against `DaleRNN.simulate` / `_effective_w_rec` and the run's
`summary.json` (`n_rnn` 256, `frac_e` 0.8, `g` 1.0, `sigma_rec` 0.05, `lr` 1e-3, `batch_size` 32, 150k
steps; `alpha = dt/tau` from `cmc/task.py`). Column convention: W_ij is the synapse from j to i, so D scales
columns (presynaptic), as in Song's W^{rec,+}D. In code it is `h @ W` with presynaptic rows.

Notes:
- Leaky rate units, ReLU, Euler step of 20 ms with a 100 ms time constant; noise on the recurrent input.
- Dale's law exactly as in Song 2016: the network learns non-negative strengths, the sign belongs to the
  presynaptic neuron and never changes. softplus instead of rectification, so the gradient never dies.
- No self-connections. Readout from E only (long-range projections are excitatory).
- Unlike Yang (W_out kept small at init), our readout starts at O(1) ("fanin" init) and everything is trained.

---

## 7. Three objectives

$$L_{\text{task}} = \Big\langle \frac{\sum_{t,k} m_{tk}(\hat y_{tk}-y_{tk})^2}{\sum_{t,k} m_{tk}} \Big\rangle_{\text{trials}}$$
$$C_{\text{fire}} = \langle r_i(t)^2 \rangle_{\text{trials},t,i}$$
$$C_{\text{wire}} = \frac{1}{N(N-1)}\sum_{i\neq j}\frac{|W^{\text{rec}}_{ij}|}{c_j},\qquad c_j = 1\ (\text{E}),\ n_E/n_I\ (\text{I})$$

Bullets: task = squared error where the target matters, each trial weighted equally; firing = mean squared
rate; wiring = L1 of recurrent weights (weak synapses go to zero), c_j divides out the ~4× stronger I synapses.

Make: Beamer, equations + three bullets. Code (defaults in every run):
- `L_task` = `masked_mse(..., per_trial=True)`: each trial divided by its own mask weight `c_mask`.
- `C_fire` = `cmc.train_cog.rate_reg`, `kind="l2"`, `inh_scale=1`: mean of r² over trials, time, neurons.
- `C_wire` = `cmc.train_cog.connectivity_reg`, `kind="l1"`, `target="w_rec"`: |W_rec| divided per presynaptic
  neuron by `DaleRNN.ei_row_scale()` (1 for E, 204/52 ≈ 3.9 for I), mean over off-diagonal synapses.
- C_fire and C_wire are the reported `metabolic_cost` and `wiring_cost`. In the front, the task axis is
  `min_task_acc` (worst task), not L_task: that comes on the multi-objective slide.

Notes:
- Three things to make small at once: error, firing, wiring.
- Why L2 for firing (if asked): L1 = mean rate is the literal ATP cost, but with ReLU it silences units outright,
  I first (they are not read out). L2 has no dead-unit pressure and is what Yang used. It is closer to a
  homeostatic constraint. L1 (`--rate-kind l1`) is implemented, never run.
- Why L1 for wiring: the cost is about how many connections exist; L1 is the convex surrogate for that count and
  drives weights to exactly zero; L2 only shrinks them. Weight strength, not wire length: no space in the model.
- c_j: inhibitory synapses start ~4× stronger (so total E and I drive balance). Without the division L1 would
  prune inhibition first, and with ReLU that means runaway excitation.

---

## 8. Holding the network to a budget

- **Wiring: hard limit.** Not in the loss. After every Adam step, all synapses shrink by the same amount until
  the cost fits; weak synapses reach zero (projection onto the L1 ball, Duchi et al. 2008).
  $|W_{ij}| \leftarrow \max(|W_{ij}| - \theta, 0)$, θ chosen so that $C_{\text{wire}} = B_{\text{wire}}$.
- **Firing: soft limit.** Activity cannot be clipped, so it is penalised with a weight λ that tunes itself.
  $L = L_{\text{task}} + \lambda C_{\text{fire}} + \tfrac{\rho}{2} B_{\text{fire}}[C_{\text{fire}}/B_{\text{fire}} - 1]_+^2$,
  $\log\lambda \leftarrow \log\lambda + \eta(C_{\text{fire}}/B_{\text{fire}} - 1)$.
- No hand-set trade-off weights: each network gets two ceilings and is trained to be as accurate as possible
  under them.

Make: Beamer, two columns. Code: `project_w_rec_to_budget` (L1 ball in the E/I-normalised magnitudes, so E and I
get the same fractional shrinkage; pruned synapses become softplus(w_raw) ≈ 1e-8) and `BudgetConstraint`
(defaults in the 150k runs: η = 0.01, kp = 0, no ramp, ρ = 1, λ_init = 0.01, EMA 0.9). The slide's λ update is the
simplified form; in code the violation is EMA-smoothed and clipped to ±1, λ ∈ [1e-6, 1e4].

Notes:
- Two ways to keep a cost under a ceiling. Wiring: exact, by projection. Firing: a penalty whose weight finds
  itself, rises while over budget, falls while under, so the network is not pushed below what the budget asks.
- Why not a self-tuning weight for wiring too: wiring reacts to λ with a lag and a threshold, so the weight
  winds up to 50–500× what is needed and prunes the network to death (tested; ramp and PI terms did not fix it).
- Honest: the firing limit is soft. At 150k steps networks end at ~1.7× their rate budget
  (`fig_grid_proposal_150k.py`); the front is always plotted at the costs actually reached.
- This is the ε-constraint method: ceilings instead of weights.

---

## 9. Mapping the front: NSGA-III (pymoo)

- **Minimise** (1 − worst-task accuracy, log C_fire, log C_wire), subject to worst-task accuracy ≥ 0.6.
- **Pareto front**: networks where no objective can improve without another getting worse.
- **Two nested loops**: NSGA-III chooses the budgets (B_fire, B_wire); gradient descent trains the best network
  under each.
- Loop: sample budgets → train & evaluate → rank (non-dominated) → keep a spread → breed ↺
- Worst task, not mean: a network could drop one task and still average ~0.9.
- Budgets, not a weighted sum: also reaches the non-convex parts of the front.

Make: Beamer, bullets + one-line loop. References: Deb & Jain 2014 (NSGA-III); Blank & Deb 2020 (pymoo).
Code: `python -m cmc.moo` without `--points` (with `--points` it only trains the listed budgets, no search).
cog20 settings used before (`runs/moo/dale_all_nsga3_budget_p48g3_50k`): pop 48 × 3 generations, 8 Das-Dennis
partitions, rate budget ∈ [0.002, 0.15], wiring budget ∈ [0.002, 0.03], batched 24 × 2 workers.

Notes:
- Why not let NSGA-III evolve the weights: ~90k weights; backprop gives the exact direction for all of them per
  step, evolution gets one number per network. NSGA-III picks the trade-offs, gradient descent achieves them.
- "Keep a spread": NSGA-III attaches every network to its nearest reference direction in objective space and
  fills the least-populated directions first, so the front stays spread out.
- All networks ever trained are pooled; infeasible ones (worst task < 0.6) are dropped; the non-dominated rest is
  the front. Without the feasibility rule, chance-level networks sit on the front as the cheapest points.
- With ~150 evaluations this is adaptive sampling that concentrates around the front, not a converged optimiser;
  the multi-seed zoom points are the reliable ones.

---

## 10. The front (placeholder: budget point scan)

[Rotating 3D front, `animate` frames from `make_frames.py`.] Caption: point scan, not yet NSGA-III: 29 budget
points × 3 seeds + 4 zoom points, 150k steps (111 networks, 89 feasible, 38 on the front). To be replaced by the
NSGA-III run.

Make: Beamer `\animategraphics` on `figs/talk/frames/front_000–059.png` (`talk_figs.py frames`; the seed-4
networks ringed, as on slide 11; the old frames from `make_frames.py` are kept in `frames/`). **To replace** with the cog20 fanin 150k NSGA-III
run from the cluster when it is done (re-render the frames from that run).

Notes:
- How the point scan was chosen: the earlier cog20 NSGA-III run (50k steps) located the front; a log-spaced grid
  of rate × wiring budgets over that region (points far past the feasibility cliff dropped), 3 seeds each,
  trained to 150k because 50k was not converged.
- Wiring budget 10^-2.75 is infeasible everywhere.

---

## 11. Four points on the front

[Left: top view of the 150k fanin front with the four points labelled. Right: table, 5-seed means.]

| | control | A3 | B0 | C0 |
|---|---|---|---|---|
| budget | none | rate | wiring | both |
| rate budget | – | 0.0011 | – | 0.0115 |
| wiring budget | – | – | 0.0054 | 0.0044 |
| worst-task accuracy | 0.96 | 0.85 | 0.87 | 0.88 |
| mean accuracy | 0.99 | 0.98 | 0.99 | 0.98 |
| firing cost | 0.49 | 0.0020 | 0.81 | 0.018 |
| wiring cost | 0.030 | 0.047 | 0.0054 | 0.0044 |
| synapses kept | 70% | 61% | 6% | 0.7% |

Plus a table row for the example network (seed 4): worst-task accuracy 0.95 / 0.78 / 0.89 / 0.89.

Make: Beamer. Figure `figs/talk/front_top.pdf` (`talk_figs.py front`; analysis version
`results/front/cog20_150k_fanin_front_top.png` from `fig_front_top_fanin150k.py`; the seed-4 networks are ringed,
`EXAMPLE` in the script). Background = filled contours of worst-task accuracy (RdYlGn, 0–1): thin-plate-spline fit
through the 33 per-budget seed means, only inside their convex hull; lines at 0.8 and 0.9, dashed at 0.6
(feasibility). Symbols: × infeasible network, dot feasible, white circle on the Pareto front. Table header uses
the point names; firing budget = rate budget in the code. Numbers: 5-seed means from `reports/ei_modularity/fig/summary.csv`
(run `dale_all_woutfree_fanin_4pt_5seed_150k`).

**Example network = seed 4 of that run, at every point.** In this run the seed fixes the initial weights and all
20 networks were trained together on one trial stream, so control seed 4 and C0 seed 4 start from the same network
and see the same trials: only the budget differs. Seed 4 was chosen as the clearest contrast of the 5 (control
I→E −0.69, the most negative; C0 E→I +0.59 and E→E within/between 7.77). Say so: "example: seed 4, the clearest
of 5", and give the 5-seed ranges with it. Per-seed numbers for all 5 seeds: see slides 16–19.

Notes:
- Four points to compare: no budget, rate budget only, wiring budget only, both. 5 seeds each.
- A3 fires ~250× less than control. B0 and C0 keep 6% and 0.7% of the recurrent synapses.
- All of them solve the tasks: mean accuracy > 0.95 everywhere; the worst task (usually DNMS) drops from 0.96
  to ~0.85–0.88 under a budget.
- Side facts if asked: A3's wiring cost is *higher* than control (0.047 vs 0.030): fewer, quieter neurons drive
  each other harder. Firing ends ~1.7× above its budget in A3 and C0 (soft limit).

---

## 12. Do they converge?

[2 × 2: accuracy of all 20 tasks over training, mean (black), worst task (red), per point.]

Make: `figs/talk/convergence.pdf` (`talk_figs.py convergence`: 1 × 4, each task grey, mean black, worst task
red with its 150k value; analysis version `results/front/acc_vs_steps_fanin_grid.pdf` from
`results/scripts/fig_acc_vs_steps_grid.py`, 2 × 2 with every task named; data: `runs/moo_zoom/dale_all_woutfree_fanin_4pt_1seed_150k_saveat/runs_by_step.csv`).

Notes:
- Convergence is checked on a **separate run** of the same four points (one network each, seed 0, accuracy saved
  along training). The 5-seed run logged no accuracy during training, so there are no curves for the example network.
- Most tasks are at ceiling within ~25k steps; the delayed DM and matching tasks are the slow ones.
- Control flat (worst task ~0.96) from 75k, C0 flat (~0.87) from 100k, B0 still rising slowly to 0.91.
- **A3 is not converged**: its worst task is still rising at 150k (0.50 at 100k → 0.73). Say so; it is one of the
  open questions (more save-at seeds / longer training).
- These are the save-at networks, a separate batched run from the 5-seed one; worst-task values differ from the
  table (1 seed vs 5-seed mean).

---

## 12a. Like-to-like wiring (moved up from the Experimental section; no longer tagged experimental)

### 12a.1 Like-to-like wiring (math)

s_ij = corr(x_i, x_j) of the period profiles; connected j → i = |W_ij|/c_j > 0.01 (c_j as in the wiring cost);
per type E→E, E→I, I→E, I→I: AUC = P(s_conn > s_unconn) = (R₁ − n₁(n₁+1)/2)/(n₁n₀); ROC from a sliding threshold,
AUC = ∫ TPR dFPR. Null: 500 shuffles of profiles within E and within I (wiring fixed), two-sided
p = (1 + #{|AUC_null − 0.5| ≥ |AUC − 0.5|})/501.

Code: `results/scripts/fig_wiring_like2like.py::network`.

Notes:
- AUC does not depend on connection density (control keeps 67% of pairs, C0 0.7–0.9%), so the points compare.
- Pairs are not independent (a neuron appears in many pairs), hence the shuffle null, not a rank test p-value.

### 12a.2 Like-to-like wiring: example network

Make: `figs/talk/like2like_s4.pdf` (`talk_figs.py like2like`, ~3 min: same 20-bin histograms, ROC beside each,
number = AUC, grey "n.s." when p ≥ 0.05; no counts or medians on the talk version). Analysis version:
`python results/scripts/fig_wiring_roc_all.py` → `results/front/wiring_roc_all_s4.png` (4 types × 4 points,
histograms of similarity for connected vs unconnected, 20 bins of width 0.1 (same for both; chosen over 40 and
√n bins on 2026-11-20), each scaled to area 1, ROC in a panel to the right).
No smoothing: a Gaussian KDE (used until 2026-11-20) widened C0's few connected pairs (22–146) 3–4× more than the
unconnected ones and cut mass off at +1, hiding the like-to-like spike near 0.9. AUC, ROC and medians were always
computed on the raw values.

### 12a.3 Like-to-like wiring: all 20 networks

Make: `figs/talk/like2like_5seed.pdf` (`talk_figs.py like2like`: top mean ROC per point, band = min–max over
seeds; bottom AUC per network, filled = p < 0.05, bar = mean, grey lines join the same seed). Analysis version:
`fig_wiring_roc_all.py` → `results/front/wiring_roc_all_5seed.png`. Mean AUC (sd) over 5 seeds:

| | control | A3 | B0 | C0 |
|---|---|---|---|---|
| E→E | 0.53 (0.01) | 0.52 (0.00) | 0.52 (0.01) | 0.68 (0.02) |
| E→I | 0.54 (0.00) | 0.54 (0.01) | 0.56 (0.01) | 0.73 (0.02) |
| I→E | 0.45 (0.01) | 0.49 (0.01) | 0.51 (0.01) | 0.72 (0.02) |
| I→I | 0.48 (0.01) | 0.49 (0.01) | 0.56 (0.02) | 0.71 (0.08) |

Caption: sparsity caveat (`fig_wiring_matched_sparsity.py`): B0 cut post hoc to C0's sparsity keeps E→E at 0.52.

Notes:
- Only C0 wires like-to-like, in every seed and every type.
- Inhibition flips: control I→E < 0.5 in all seeds (I targets E doing other computations), C0 0.72 (same
  computation).
- Post-hoc pruning is not training under the budget; the clean test (wiring-only networks at C0's sparsity) is open.


---

## 13. Task variance and the number of groups

[Left: definitions. Right: silhouette vs k, E and I, four points, chosen k dashed.]

$$\mathrm{TV}_{i,a} = \big\langle \mathrm{Var}_{\text{trials}}\, r_i(t) \big\rangle_{t > \text{fixation}},\qquad s = \frac{b-a}{\max(a,b)}$$

Make: Beamer, definitions in the left column, `figs/talk/silhouette.pdf` (`talk_figs.py silhouette`; dot =
chosen k; analysis version `results/front/tv_ei_fanin_5seed_s4_150k_sil.png` from
`python results/scripts/fig_tv_ei_fanin.py --run 5seed --seed 4`: the example network at each point).
Code: `cmc.runner.activity_summary` (variance across trials at each time step after fixation onset, averaged over
time and eval seeds, noise-free); `fig_tv_ei_fanin.cluster_pop` (active units: total TV > 1% of max; each unit
divided by its max; k-means, n_init 10; k = best local maximum of the silhouette with k > 2; E up to 20, I up to 10).

Notes:
- Task variance (Yang 2019): how much a neuron's activity varies across trials of a task, i.e. how much it takes
  part in that task. Divide by the neuron's maximum: a task profile, 20 numbers per neuron.
- Cluster E and I separately: that is what lets us ask later how inhibition relates to the E groups (Yang clusters
  all units together).
- Number of groups k: silhouette, how much closer a neuron is to its own group than to the next one. We take the
  best local maximum with k > 2: k = 2 is almost always a trivial split.
- Silhouettes are low (0.15–0.45): overlapping groups, not discrete cell types. Highest under a rate budget (A3),
  for E and especially I.
- Chosen k (example network, seed 4): E control 4, A3 16, B0 4, C0 13; I control 3, A3 9, B0 5, C0 5. k varies from
  seed to seed;
  only per-network numbers (tasks per neuron, silhouette, ρ) are averaged across seeds.

---

## 14. E and I groups: control vs. firing constraint

[Task-variance heatmaps, units sorted by group, E left, I right: control, then A3.]

Make: `figs/talk/tv_control_A3.pdf` (`talk_figs.py tv`: E and I heatmaps side by side per point, group numbers
under the columns, number of groups in the panel titles; units, silhouette and tasks/neuron left for these
notes). Analysis version: `results/front/tv_ei_fanin_5seed_s4_150k.png`; the old crops `figs/tv_ei_*.png`
(`crop_tv_rows.py`) are kept. Example network (seed 4).

Notes:
- Control: broad groups that follow task families (all DM, all delayed DM, the matching tasks); neurons take
  part in many tasks (E 4.7, I 4.8 tasks per neuron, 5-seed means).
- A3: sharpest tuning; many small E groups, several dedicated to one task (RT Go, RT Anti, DMS, DNMS, DMC).
  E 2.9 tasks per neuron. I broader than E (4.1) and grouped by task family.

---

## 15. E and I groups: wiring constraint vs. both constraints

[Same for B0 and C0.]

Make: `figs/talk/tv_B0_C0.pdf`, as slide 14.

Notes:
- B0: the broadest neurons (E 5.2 tasks per neuron), few large groups.
- C0: sharp E tuning again (3.5), I slightly broader (4.0). Only 145 E / 43 I units active in this network.
- Summary for the next part: a rate budget sharpens tuning (A3, C0); a wiring budget alone does not (B0).
  Every I group has an E group with a similar profile (best match r ≈ 0.7–0.9 at every point), so I is always
  tuned; the points differ in how broad I is, and in the wiring (next).

---

## 16–19. Task similarity and wiring: control, A3, B0, C0

One slide per point, example network (seed 4). Left: task similarity r between the groups' mean task profiles.
Right: wiring from row group to column group, z vs 500 shuffles of group labels (within E and within I; grey: group
< 4 neurons). Caption: this network's numbers and the 5-seed range.

| | ρ I→E | ρ E→I | E→E within/between | I→I within/between |
|---|---|---|---|---|
| control, seed 4 | −0.69 | −0.13 | 1.07 | 0.94 |
| control, 5 seeds | −0.69 to −0.31 | −0.13 to +0.05 | 1.05–1.15 | 0.90–1.15 |
| A3, seed 4 | +0.09 | 0.00 | 1.04 | 1.08 |
| A3, 5 seeds | −0.05 to +0.09 | −0.15 to +0.14 | 1.02–1.09 | 1.03–1.30 |
| B0, seed 4 | +0.17 | +0.33 | 1.22 | 2.55 |
| B0, 5 seeds | −0.05 to +0.45 | −0.03 to +0.40 | 1.15–5.89 | 1.27–2.55 |
| C0, seed 4 | +0.16 | +0.59 | 7.77 | 2.05 |
| C0, 5 seeds | +0.07 to +0.39 | +0.36 to +0.62 | 2.84–7.77 | 2.05–27.5 |

Make: crops of `results/front/tv_ei_fanin_5seed_s4_150k_wiring.png` → `figs/wiring_{control,A3,B0,C0}.png`
(same crop script). Numbers: `fig_tv_ei_fanin.wiring` on all 5 seeds of the 5-seed run.

**Method slide still to come** (math being written): definition of the block wiring, the shuffle null, ρ and
within/between. Open choices from the review of the code: strength vs presence (mean |w| counts pruned synapses
as 0, so in C0 it mostly measures how many connections a block has); ρ is on z-scores, within/between on raw block
means; small groups have wide nulls, so I→E / I→I are noisy.

Notes:
- Control: I groups inhibit the E groups of *other* tasks more than chance (I→E ρ −0.69, negative in all 5 seeds):
  inhibition between modules. E→I unstructured. No self-wiring.
- A3: tuning is sharp, but the wiring ignores the tasks (both ρ ≈ 0 in every seed).
- B0: weak, seed-dependent E+I modules (ρ from about 0 to +0.45 depending on seed).
- C0: E→I follows the tasks in every seed (+0.36 to +0.62): an E group drives the I groups of the same tasks.
  E groups excite themselves 2.8–7.8× more than other groups. I→E is weak in every seed (+0.07 to +0.39): the
  I→E contrast with control comes mostly from control being strongly negative.

---

## Experimental (draft; the user decides later what stays)

Divider slide, then three slides tagged "experimental" (E1–E3; the like-to-like slides E4–E6 moved up to 12a). Seed 4 = example network throughout.

### E1. Task variance per task period (math)

Left: TV per (task, period), periods stimulus 1 / memory 1 / stimulus 2 / memory 2 / response, fixation out → 59
rows; normalise each active neuron (total TV > 1% of max) by its max; Ward on neurons and on task periods
(merge cost Δ(A,B) = |A||B|/(|A|+|B|)·‖μ_A − μ_B‖²); k per axis = highest silhouette over 3..39; sparsity =
fraction of entries < 0.15. Right: vector crop of Driscoll et al. 2024 Fig. 3a (`figs/driscoll2024_crops.pdf`,
build with `pdflatex driscoll2024_crops.tex` in `figs/`, page 6 of `papers/driscoll2024.pdf`).

Code: `cmc/runner.py::period_variance` (same trials and noise-free simulation as the per-task TV; the length-weighted
mean over a task's periods reproduces the stored per-task TV to 3e-7).

Notes:
- Driscoll split by period because a task is several computations in sequence (encode, hold, respond); per task,
  a neuron that only holds memory looks like it "does Dly Go".
- Differences from their code: active rule (> 1% of max vs > 1e-3), random eval trials instead of a stimulus grid,
  k by silhouette instead of the fixed cut height in their figure notebook; E and I clustered together (strip
  under the dendrogram marks E/I).

### E2–E3. Task periods: control | A3, B0 | C0

Make: `python results/scripts/fig_tv_period_driscoll.py --points control A3 B0 C0 --seed 4` →
`results/front/tv_period_driscoll_<point>_s4.png` (uses the cached `tv_period_<point>_s4.npz`).

| seed 4 | active (E/I) | k units (sil) | k periods (sil) | sparsity E / I |
|---|---|---|---|---|
| control | 256 (204/52) | 3 (0.17) | 4 (0.41) | 0.43 / 0.39 |
| A3 | 247 (195/52) | 3 (0.25) | 5 (0.28) | 0.79 / 0.62 |
| B0 | 256 (204/52) | 3 (0.22) | 3 (0.35) | 0.48 / 0.48 |
| C0 | 188 (145/43) | 39 (0.19, upper bound) | 20 (0.22) | 0.69 / 0.63 |

Notes:
- Control and B0: neuron groups follow task families (Ctx/MultSen DM, matching tasks, the rest).
- A3 and C0: groups follow computations shared across tasks (stimulus per modality, memory, response); the
  memory-guided response is separate.
- Silhouette is low everywhere (≤ 0.25 on units) and C0 has no natural k (hits 39): clusters are soft; the
  sparsity numbers are the robust part. One seed only.

---

## Lesions on task-period groups (experimental, 2 slides, at the end)

Math slide + `results/front/lesion_period_s4_ward_contrast_grid.png` as it is (no talk version yet).
Code: `results/scripts/fig_lesion_period.py --method ward` per point (seed 4, v1 networks), then
`fig_lesion_contrast_grid.py --seed 4 --tag _ward`.
- Groups: per-period task variance, Ward, E and I separately, k = 14 E (5 families × stimulus/memory/response,
  DM has no memory) and 5 I (families).
- Lesion: rates of the group set to 0 at every step (Driscoll). Damage Δ = (acc_lesioned − acc_full)/acc_full per
  task, 10 eval seeds × 64 noise-free trials.
- Module contrast C = mean Δ over module tasks − mean Δ over reference tasks (Anti vs not Anti; delay tasks (Dly,
  Go, Anti) vs the rest; modality 1 vs 2; DMC vs Dly tasks), as Driscoll Fig. 5. C < 0: module hurt more.
- Mean change per lesion: control −0.16, firing −0.05, wiring −0.16, both −0.13; |C| ≥ 0.2 in 6, 5, 9, 9 of 19
  groups. Results depend on k (not on k-means vs Ward). No group-size control. v1 battery: redo on the retrained
  networks.

---

## Thank you

Make: Beamer, centred.
