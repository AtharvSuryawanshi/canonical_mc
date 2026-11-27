# Claims and ideas (started 2026-11-27)

Running list of what the results might let us claim and what to test next. Status: **supported** (numbers in
hand), **to test**, **idea** (needs new runs or theory). Citations are from memory; check before they go on slides.

Numbers: front run `runs/moo_zoom/dale_all_woutfree_fanin_front4_5seed_150k_yang`, 4 points x 5 seeds, per-task
task-variance profiles, like-to-like AUC = P(r_conn > r_unconn) with 500 shuffles (`like2like.py --basis task`).

## A. Candidate claims

### A1. The two costs leave dissociable marks (double dissociation) — supported, needs the formal test

| | like-to-like AUC E→E | tasks per E neuron |
|---|---|---|
| control | 0.53 | 5.0 |
| rate constraint | 0.52 | **3.1** |
| synaptic constraint | **0.57** | 4.7 |
| both constraints | **0.63** | **3.4** |

- Synaptic budget → wiring rule (like-to-like); the rate budget alone does not give it. Pruning a dense network to
  the same sparsity does not give it either (matched-sparsity check), so it comes from training under scarcity.
- Rate budget → selectivity (fewer tasks per neuron, more distinct groups); the synaptic budget alone barely moves it.
- Each maps onto a known cortical property: like-to-like wiring (Ko 2011, Cossell 2015, MICrONS 2025) and
  sparse / selective coding under an energy budget (Lennie 2003; Whittington et al. 2023 ICLR, Dorrell et al. 2025
  ICLR: non-negativity + energy cost → specialised "cell types").
- **To do:** 2 x 2 analysis (rate budget x synaptic budget, seeds as replicates) with sparsity as a covariate.
- **Caveats:** both > synaptic in AUC may be sparsity alone (1.9% vs 2.7% kept; AUC vs sparsity Spearman −0.9).
  Selectivity: part of the rate effect is weakly active neurons (strongly active only: control 5.1, rate 4.4,
  loose 4.1), and the loose point (rate cost 10x the rate point) is not less selective than the rate point.

### A2. The synaptic budget flips inhibition from anti-tuned to co-tuned — supported

- I→E AUC: control 0.46 (5/5 seeds significantly below 0.5), rate 0.47, synaptic 0.58, both 0.61. I→I the same
  (0.46 → 0.60 / 0.62).
- Without a synapse budget, I neurons inhibit E neurons that do *different* tasks (competition between task groups);
  with it, they inhibit neurons doing the *same* tasks (co-tuned feedback inhibition, as in mouse V1:
  Znamenskiy et al. 2024).
- Not a sparsity effect: pruning control to both's sparsity makes I→E *more* anti-tuned (0.41).
- Robust to the similarity measure: Pearson and Spearman AUCs agree (rank correlation 0.98 over 80 values).

### A3. The rate budget makes the network functionally modular — to test (lesions, B1)

"Fewer tasks per neuron" is a statement about activity. The causal version: removing a group of neurons damages
fewer tasks under a rate budget.

### A4. Each budget reproduces a different part of the canonical microcircuit — idea

Douglas & Martin: weak input amplified by recurrent excitation, kept in check by co-tuned inhibition. A2 gives the
co-tuned inhibition (synaptic budget). B5 (inhibition stabilisation) and B3 (connectivity statistics) would test
the rest.

## B. Things to hunt for

### Safe (existing networks, cheap)

- **B1 Lesion specificity** (→ A3). Literature (2026-11-27), simplest first:
  - Yang et al. 2019: lesion = outgoing weights of a unit set to 0 (= silencing); each task-variance cluster
    lesioned, performance change on all 20 tasks as a cluster x task matrix (Fig. 2e); FTV groups for Ctx DM 1 vs 2
    give a double dissociation (Fig. 5). No size control. Fig. 3f-g: L1 rate and L1 weight regularisation barely
    changed the number of clusters (weighted penalty, likely weaker than our budgets) — consistent with our "rate
    changes selectivity, not the number of groups".
  - Driscoll et al. 2024: clusters' rates set to 0 for the whole trial, fractional performance change, no size
    control.
  - Prince, Alvarez & Konkle 2024 (Sci Adv): zero selective groups; control = random groups of the same size
    (near-zero effect); deficit profile tracks the group's activity profile (r ~ -0.57) = our "alignment".
  - Morcos et al. 2018: unit selectivity barely predicts ablation importance (warning).
  - Saha, Bergen & Khosla 2026 (arXiv 2610.12842): double-dissociation index on task pairs, Yang battery; modularity
    can sit in subspaces when units look mixed.
  - Fakhar & Hilgetag 2022: single lesions can mislead under redundancy (Shapley multi-lesion; only if needed).
  - Zero vs mean ablation: interpretability prefers mean/resample; the neuroscience RNN papers all use zero.
  - **Plan.** Step 1 = Yang Fig. 2e on one network (control seed 0, then rate seed 0): silence each per-task TV
    cluster (E and I separately, chosen k), accuracy change on all 20 tasks, plus 20 random size-matched groups per
    cluster. Step 2 only after looking: tasks hurt by > 10% per lesion, Prince-style alignment, then other points.
    Deferred unless step 1 is ambiguous: mean-clamp, fixed-size top-m groups, PR, DDI, Shapley.
  - **Step 1 result (seed 0, `lesion_clusters.py`):** rate network: half the cluster lesions do nothing, the rest hit
    the group's own task (7/11); control and synaptic: broad damage, about as large as random groups. But group
    sizes differ (k per network) and the rate network is robust to everything (mean change -0.03 vs -0.21).
  - **Step 1b (seed 0, `lesion_taskgroups.py`):** per task, the 10 E neurons with the largest raw TV; silenced or
    mean-clamped (trial-average per time step); 20 random groups of 10 as control. Own-task / other-task damage:
    control 2.3 / 2.2 (zero / mean), rate **5.3 / 5.0**, synaptic 2.1 / 2.4, both 2.3 / 2.7. Mean-clamp cuts control's
    damage by 2/3 (much of its silencing damage was lost drive) but barely changes the rate network's. Damage spreads
    within task families everywhere (Dly DM family). Groups overlap least under the rate budget (Jaccard 0.10 vs
    0.13-0.16). One seed: needs seeds 1-4.
  - **Seeds 0-4 (`lesion_taskgroups.py --seeds 0-4`, `results/front/lesion_taskgroups_seeds.png`):** own/other
    damage, mean +- sd: silenced control 2.4+-0.3, rate **5.2+-1.0**, synaptic 1.9+-0.2, both 2.4+-0.2;
    mean-clamped control 2.1+-0.2, rate **4.7+-1.0**, synaptic 2.2+-0.2, both 2.6+-0.2. Rate vs control p = 0.008
    (Mann-Whitney, all 5 rate networks above all 5 control) under both lesion types. Synaptic below control only when
    silenced (drive effect; p = 0.42 mean-clamped). **Both constraints is not specific**, though its E neurons are
    nearly as selective as the rate point's (3.4 vs 3.1 tasks per neuron): selective activity without
    confined damage. Its rate cost is 8x the rate point's (0.017 vs 0.002), and damage there spreads across the
    whole DM / Dly DM family. Open: is it the looser rate budget, or do sparse synapses couple tasks?
  - **Family level** (Yang's 5 families: Go, Anti, DM, Dly DM, Matching; damage on the other tasks of the group's
    family / on other families, seeds 0-4): silenced control 1.7+-0.4, rate **4.4+-1.1**, synaptic 1.3+-0.2, both
    1.8+-0.1; mean-clamped 1.6 / **4.3** / 1.4 / 1.9. Family-wide damage = a shared computation was hit (Yang's
    "common substrate", Driscoll's motifs), i.e. functional specificity; under the rate budget damage stays inside
    the family, elsewhere it crosses families too. Families are a priori; a data-driven grouping would be better.
  - **Dose-response, seed 0, mean-clamped (`--dose`, `results/front/lesion_dose_s0.png`), m = 2/5/10/20/40:** family
    specificity rate 3.0/3.7/4.3/5.2/3.8, control 2.0/2.0/1.7/1.6/1.5, synaptic 1.0/1.0/1.4/1.2/1.3, both
    2.2/1.9/1.7/1.5/1.6. Rate above all others at every size; peaks at m = 20 and falls at 40 (family units ~20 E
    neurons?). Damage on the group's own family is the same in rate and control at every size; the whole
    difference is damage leaking to other families (m = 40: rate 0.07, control 0.17, synaptic 0.21, both 0.23).
  - **Dose-response, seeds 0-4 (`--dose --seeds 0-4`, `results/front/lesion_dose_seeds_mean.png`, CSV
    `results/cache/front/lesion_dose_seeds_mean.csv`):** family specificity mean +- sd, m = 2/5/10/20/40: control
    1.7/1.7/1.6/1.6/1.6, rate **3.7/3.8/4.3/4.2/3.3**, synaptic 1.4/1.4/1.4/1.4/1.3, both 2.0/2.0/1.9/1.7/1.6. Rate vs
    control p = 0.008 at every size (all 5 rate networks above all 5 control). The seed-0 peak at m = 20 does not
    hold (mean peaks at 10, per-seed peaks vary): **no unit-size claim**. Same-family damage rate = control at every
    size (m = 40: 0.29 vs 0.27); other-family damage rate 0.09 vs control 0.17, synaptic 0.20, both 0.21. Both
    constraints has the largest same-family damage (0.34) and is not more specific than control at any size.
  - **I-group lesions, seed 0, mean-clamped (`--pop I --m 1,2,5,10`, `results/front/lesion_dose_s0_I.png`):** top-m I
    neurons per task (m = 1/2/5/10 = 2-19% of 52 I). Family specificity control 1.5/1.6/1.4/1.2, rate 1.8/2.0/1.7/1.3,
    synaptic 1.4/1.4/1.4/1.4, both 1.4/1.5/1.4/1.1: all similar, all > 1. (a) **I neurons carry task information**:
    clamping the top I neurons of a task hurts more than random I groups (m = 2, other families 0.03-0.05 vs random
    0.02-0.03; own task 0.08-0.14), unlike silencing (step 1), where any I group did as much damage as random. Per
    fraction of the population, I lesions hurt about as much as E lesions. (b) The A2 prediction **fails**: control is
    not below 1 (anti-tuned I does not push damage onto other families), synaptic / both are not more specific. (c) The
    rate budget's specificity is an E-population property; its I neurons are only slightly more specific. One seed.
  - **I groups, seeds 0-4 (`--dose --seeds 0-4 --pop I`, `results/front/lesion_dose_seeds_mean_I.png`):** family
    specificity m = 1/2/5/10: control 1.8/1.8/1.4/1.2, rate 2.1/1.8/1.6/1.2, synaptic 1.1/1.2/1.2/1.2, both
    1.9/1.6/1.4/1.2. No point above control (rate p >= 0.1); synaptic *below* control at m = 1, 2 (p = 0.008), equal at
    10. Rate's higher own-task damage (seeds 0-2) does not hold: 0.074 vs 0.054, p = 0.22.
  - **To explore (user, 2026-11-27): does the synaptic budget make large I lesions specific?** Seeds 0-2, I family
    specificity m = 1/2/5/10: synaptic 1.1/1.2/1.2/1.3 (flat or rising), the others fall to ~1.2 (control 1.6 -> 1.2,
    rate 2.5 -> 1.25). Synaptic E specificity was flat with m too. Not group overlap (I-group Jaccard the same as
    control). Possible mechanism: sparse (~3%) co-tuned I->E synapses (A2) confine a big I lesion to the group's
    own-family E targets, while in dense networks a big lesion perturbs everyone. Test after seeds 3-4: m = 15/20/26
    (29-50% of I); check from W alone the share of each I group's output onto its family's top E neurons.
    5 seeds: synaptic stays flat (1.1-1.2) and the others fall to it (1.2 at m = 10); no crossing yet.
  - **Note (user, 2026-11-27):** both constraints may simply not be tight enough on rate (rate cost 0.017 vs 0.002 at the
    rate point). On a dense front, a point with both budgets *and* a rate cost near the rate point's might be specific
    too: then specificity is set by the rate budget alone, and both's failure is not caused by sparse synapses.
  - **Open issues:** (1) fixed m = 10 equalises neurons removed, not the fraction of a functional unit removed →
    dose-response over m = 2-40; (2) both vs rate → lesion the loose point (rate cost 0.023 ~ both's, synapses dense,
    52% kept): specific = sparse synapses couple tasks in both; not specific = both's rate budget is too loose.
- **B2 Sparseness without silencing.** Population sparseness (fraction active per time step), lifetime sparseness
  over active neurons only, at several activity thresholds. Separates "a rate budget silences neurons" from "it
  creates specialists".
- **B3 Connectivity statistics vs cortex** (Song et al. 2005; MICrONS 2025): reciprocity (cortex ~4x chance),
  log-normal weights, over-represented triplet motifs. Like-to-like should come with more reciprocal pairs.
  - Earlier (2026-10, core5, v1 battery, wiring_limited point, `cmc/motifs.py`, reports/archive/PRESENTATION_motifs):
    I<->I 5.3x chance, back inhibition E->I->same E 0.02x, E<->E only as corr of dW. Not comparable with now.
  - **Pilot on the 20-task front (2026-11-27, seeds 0-4, connected = |W|/c > 0.01, ratio = P(a<->b) / (P(a->b)
    P(b->a)), null = senders shuffled per target, ~1):** E<->E control 1.02, rate 1.09, synaptic **3.5** (2.8-4.1),
    both **5.0** (3.2-6.4); I<->I 1.02 / 1.04 / **5.1** / **5.1**; E<->I (back inhibition) 0.98 / 0.97 / **0.02** /
    **0.28**. Synaptic budget gives cortex-like E<->E reciprocity (Song 2005 ~4x) and I<->I reciprocity, and avoids
    reciprocal E<->I (cortex: common). Control and rate are 68% / 56% connected at this threshold, so binary ratios
    are ~1 by construction there. Small counts at 2-3% density (~8 E<->E pairs expected by chance). To check:
    reciprocity beyond like-to-like (condition on similarity), triad census vs Song's pair-preserving null.
  - **Triad census pilot (E->E, seeds 0-4, Song's null: pairs independent with the measured none / one-way /
    two-way rates; obs / expected):** synaptic: 030T 1.8 (154 vs 85), 030C 1.8 (50 vs 28), 120D/U/C 2.5-2.7 (~10 vs
    4), 210 5.4 (4 vs 0.8); both: 030T 2.9 (39 vs 13), 120x 5-9, 210 15 (counts 1-5). Open 2-edge triads (021x, 111x)
    ~1-1.25. Control and rate ~1 (dense; rate 300 1.16, 201 0.87). Pattern = more-connected triads and triangles beyond
    pair statistics (as in Song 2005 / Perin 2011: clustering). Caveats: tiny counts for 120/210, no degree
    heterogeneity in the null, and like-to-like clustering alone may produce triangles.
- **B4 Dimensionality:** participation ratio of activity per point; expected lower under the rate budget.

### Riskier / more interesting

- **B5 E/I balance and inhibition stabilisation (ISN).** Cortex is an ISN (Sanzeni et al. 2020); efficient coding
  says a rate cost forces tight E/I balance (Denève & Machens 2016). Test: paradoxical response (drive I, does its
  rate drop?), correlation of each neuron's E and I input over time. Claim if true: the rate budget puts the network
  in the cortical operating regime.
- **B6 Working-memory format.** Rate cost → activity-silent memory in Masse et al. 2019 (they had short-term
  plasticity; we don't); regularisation changed persistent vs sequential codes in Orhan & Ma 2019. Test:
  cross-temporal decoding in the delay tasks per point.
- **B7 Motif reuse and transfer (Driscoll et al. 2024).** Do the budgets change how much tasks share fixed-point
  structure? Train on 19 tasks, add the 20th: do constrained networks learn it faster (sparse networks learning
  faster: Khona et al. 2023)? Strong claim if yes; needs new training runs.
- **B8 Predictive coding from the energy cost** (Ali et al. 2022, Patterns). Onset transient vs sustained response
  to a stimulus per point; prediction: transient under the rate budget.
- **Later:** interneuron classes by connectivity (PV / SOM / VIP-like), out of scope for now (E/I motifs first).

## C. Reference: what the key papers claimed

- **Driscoll, Shenoy & Sussillo 2024 (Nat Neurosci):** tasks reuse a small set of dynamical motifs (fixed points,
  decision boundaries, rotations); each motif lives in a cluster of units; lesioning the cluster removes the motif in
  every task that uses it; motif pretraining speeds up new tasks. Headline: reusable modular computational parts.
- **Yang et al. 2019:** functional clusters; compositional rule representations.
- **Achterberg et al. 2023 (Nat Mach Intell):** a spatial wiring cost alone gives modularity, small-worldness, mixed
  selectivity, energy-efficient codes. Closest prior work; we differ by two separate costs, Dale's law, a Pareto
  front.
- **Whittington et al. 2023; Dorrell et al. 2025 (ICLR):** non-negativity + energy → modular single-factor neurons,
  with a theory of when (the range of the variables) — testable predictions for the rate budget.
