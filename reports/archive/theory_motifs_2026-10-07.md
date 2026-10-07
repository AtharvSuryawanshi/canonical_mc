> **Archived 2026-10-07.** Motif analysis is parked for now (see the project redirection: task-variance
> clusters and E/I weight distributions first). Kept verbatim so it can be picked up later.
> Code and figures are still live: `cmc/motifs.py`, `notebooks/moo_zoom_analysis.ipynb` §3-§5.

# Motif theory and literature (archived from reports/theory.md §7.2-§8)

Section numbers refer to reports/theory.md as it was; §6 (task variance) and §7.1 (E/I balance) stay in theory.md.

### 7.2 Motifs of 2–3 neurons (`cmc.motifs`)

**Intensity of a weighted motif.** Synapses have strengths, not just presence. The intensity of a path is
the **geometric mean** of its weights (Onnela et al. 2005). A two-synapse path a → b → c therefore has
intensity $\sqrt{A[a,b]\,A[b,c]}$. It is large only if *both* synapses are strong, and it scales like a
single weight. Passing a 0/1 matrix instead of magnitudes gives plain motif counts, as in paired-recording
studies.

| Motif | Path | Statistic | Cortical counterpart |
|---|---|---|---|
| back inhibition | E₁ → I → E₁ | mean over (E, I) of √(EI·IE) | reciprocal E↔I pairs (Yoshimura & Callaway 2005) |
| lateral inhibition | E₁ → I → E₂, E₁ ≠ E₂ | mean of √EI · √IE over E pairs, per I | surround / competitive inhibition |
| disinhibition | I₁ → I₂ → E, I₁ ≠ I₂ | mean of √II · √IE | e.g. VIP→SST→E; here any I→I→E |
| I↔I reciprocity | I₁ ↔ I₂ | mean over I pairs of √(II[a,b]·II[b,a]) | reciprocal interneuron pairs |
| E↔E reciprocity | E₁ ↔ E₂ | in ΔW, corr(D[a,b], D[b,a]) (§7.4) | over-represented ~4× (Song et al. 2005) |
| I→I concentration | — | share of I→I output held by the top 20% of I | do a few interneurons do most of it? |

Back inhibition is the same quantity as E↔I reciprocity: E₁ excites I, and I inhibits E₁ back.

**Same- vs different-task connections** (`group_stats`). With neurons labelled by task-variance cluster
(§6), every block and every motif is averaged per (presynaptic cluster, postsynaptic cluster). The
**same-task preference** is log₂(within-cluster / between-cluster), each relative to its null. Positive
means like-to-like wiring (cortex: Ko et al. 2011 for E→E; Znamenskiy et al. 2024 for E↔I). Lateral
inhibition is also split into same- and different-cluster pairs: does inhibition go to *competing* groups?

### 7.3 Shuffle nulls

Raw motif strengths cannot be compared across networks: a wiring budget shrinks every weight. Each
statistic is therefore compared with a **null that keeps the weights and destroys only the structure in
question**. It is reported as `ratio = observed / null mean` (1 = chance) and `z = (obs − mean) / std`
over 200 shuffles per network.

- **`target` null** (`_permute_columns`): for each postsynaptic neuron, permute *which* presynaptic
  neurons its input weights come from, within each E/I block (diagonal kept empty). It keeps Dale's law,
  the weight distribution and each neuron's in-strength, and destroys who-connects-to-whom. It tests
  **specificity**: back vs lateral inhibition, reciprocity, same vs different cluster, I→I concentration.
- **`hub` null**: permute the identity of the *middle* interneuron, i.e. pair one I neuron's inputs with
  another I neuron's outputs. Every weight and block stays the same. Only the pairing "strong input" ↔
  "strong output" through the same interneuron is broken. The overall chain strength is
  Σᵢ inᵢ · outᵢ, so over-representation of E→I→E (or I→I→E) as a whole means exactly that: **hub
  interneurons** that receive a lot and also send a lot.

Not yet done: a **degree-preserving** null (swap pairs (a→b, c→d) → (a→d, c→b) within a sign class), which
also fixes every neuron's out-strength. This is the stricter test for E/I balance (§7.1).

### 7.4 The learned change ΔW (`learned_change`, `delta_stats`)

**Why.** The final W = random init + what learning added. At g = 1 the control network stays
corr(W, W₀) ≈ 0.98, so the learned part is ~6% of W and motifs in W look like chance even if learning
built them. Under a wiring budget, the projection removes much of the random init and the learned part is
~30% of W. Comparing W across points therefore partly compares *how much init is left*, not what learning
did.

**Definition.** Work in normalised magnitudes M = |W|/s (the units of the wiring cost, §4.2). Bring the
init to the trained network's total wiring the same way the budget does:
- if the init has more total wiring, project M₀ onto the L1 ball with radius ΣM (soft threshold);
- otherwise, scale it up.

Then

$$D = M - \mathrm{matched}(M_0)$$

D > 0: learning strengthened the synapse beyond the uniform shrink. D < 0: weakened or pruned it. A plain
W − W₀ would mostly show the shrink itself.

**Statistics on D** (correlations, so they are scale-free):
- **reciprocity**: corr(D[a,b], D[b,a]) for E↔E, E↔I (= back inhibition) and I↔I. Do the two directions
  of a pair change together?
- **hubs**: corr over interneurons of (total change of inputs from E or I, total change of outputs to E).
  Do interneurons that gain input also gain output?
- **same-task preference**: (mean D within cluster − mean D between clusters) / std D, per block.

Each is tested against the `target` null applied to D.

**How to read it together with W.** ΔW tells you what learning *does* (the direction of the motifs). W
tells you what the circuit *is*. A motif that is present in ΔW at every point but only visible in W under
a wiring budget is set by the tasks and made dominant by the cost.

---

## 8. This is the literature: E/I motifs of 2 and 3 neurons

What experiments find and what theory says about small E/I motifs in cortex. These are the reference
values §7 is compared against. These findings come from specific areas and layers (mostly rodent V1, L2/3
and L5) and are not universal constants. Where a link is given, the source was checked online; the other
references are standard ones and should be checked against the PDF before they are cited in a submission.

### 8.1 Vocabulary

- **2-neuron motifs** (pairs): unidirectional A→B, or **reciprocal** A↔B. With E/I identity there are
  E↔E, E↔I and I↔I pairs.
- **3-neuron motifs** with two edges (Zhao et al. 2011; Hu et al. 2013):
  - **divergent**: A→B, A→C (shared input);
  - **convergent**: B→A, C→A (common target);
  - **chain**: A→B→C.

  With E/I labels, the chains of interest are:

  | Chain | Name |
  |---|---|
  | E₁→I→E₁ | feedback / back inhibition |
  | E₁→I→E₂ | lateral inhibition |
  | I₁→I₂→E | disinhibition |
  | E→E→E | recurrent excitation |

- **Triplet census**: all 16 connection patterns among 3 neurons, counted against a model that preserves
  the pair statistics (Song et al. 2005).
- **Second-order statistics** is the theory name for the frequencies of reciprocal, divergent, convergent
  and chain motifs beyond what the connection probability predicts.

### 8.2 E→E

- **Reciprocal E↔E pairs are over-represented**, ~4× more than in a random network with the same
  connection probability, in rat visual cortex L5. Reciprocal connections are also stronger on average
  (Song, Sjöström, Reigl, Nelson & Chklovskii 2005, *PLoS Biol* 3:e68).
- **Certain triplet patterns are over-represented**, especially densely connected ones (Song et al. 2005).
- **Clustering**: two neurons are more likely to be connected the more common neighbours they share
  (Perin, Berger & Markram 2011, *PNAS* 108:5419).
- **Like-to-like**: E cells with similar responses connect more often and more strongly, and bidirectional
  connections are concentrated among them (Ko et al. 2011, *Nature* 473:87; Cossell et al. 2015, *Nature*
  518:399).
- **Like-to-like holds at connectome scale** across layers and areas in the MICrONS EM volume. There is also
  a higher-order rule: the postsynaptic partners of one cell are more similar to each other than pairwise
  like-to-like predicts. RNNs trained on a classification task reproduce both rules
  ([Ding et al. 2025, *Nature* 640:459](https://ideas.repec.org/a/nat/nature/v640y2025i8058d10.1038_s41586-025-08840-3.html)).

### 8.3 E↔I (feedback vs lateral inhibition)

- **Dense**: interneurons, especially PV basket cells, contact a large fraction of nearby pyramidal cells
  (Holmgren et al. 2003, *J Physiol* 551:139; Packer & Yuste 2011, *J Neurosci* 31:13260; Fino & Yuste
  2011, *Neuron* 69:1188).
- **Reciprocal E↔I is common**: E cells that excite an FS interneuron are preferentially inhibited back by
  it (Yoshimura & Callaway 2005, *Nat Neurosci* 8:1552; Holmgren et al. 2003). This is the back-inhibition
  motif E₁→I→E₁.
- **Dense but not unstructured**: PV cells inhibit most strongly the pyramidal cells that excite them
  strongly and share their tuning. The result is feedback inhibition within feature-specific E ensembles,
  which also supports competition between ensembles
  ([Znamenskiy et al. 2024, *Neuron*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7618320/)).
- **Connectome-scale inhibitory specificity**: many interneurons target spatially intermingled
  subpopulations of E cells selectively. Inhibitory cells form "motif groups" that jointly target the same
  E cells ([Schneider-Mizell et al. 2025, *Nature* 640:448](https://pmc.ncbi.nlm.nih.gov/articles/PMC11981935/)).
- **Large paired-recording atlas** of connection probabilities and strengths between E and I subclasses, in
  mouse and human ([Campagnola, Seeman et al. 2022, *Science*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9970277)).

### 8.4 I→I and disinhibition

- **I→I connections are common, and PV↔PV pairs are often reciprocal**, with chemical *and* electrical
  synapses (Galarreta & Hestrin 2002, *PNAS* 99:12438).
- **Disinhibition (I₁→I₂→E) is cell-type specific**: VIP→SST→E, and SST inhibits most other interneuron
  types (Pfeffer et al. 2013, *Nat Neurosci* 16:1068). EM also finds a class of disinhibitory specialists
  that target basket cells (Schneider-Mizell et al. 2025).
- With E and I only, our model can show *whether* I→I→E chains are over-represented, but not the
  subtype-specific pathway.

### 8.5 E/I balance

- **Global**: in vivo, excitation is tracked by inhibition of comparable size (balanced state:
  van Vreeswijk & Sompolinsky 1996, *Science* 274:1724). Cortex operates as an **inhibition-stabilised
  network**: the E subnetwork alone would be unstable (Tsodyks et al. 1997, *J Neurosci* 17:4382;
  Ozeki et al. 2009, *Neuron* 62:578; review: Sadeh & Clopath 2021, *Nat Rev Neurosci* 22:21).
- **Detailed**: across pyramidal cells, the E/I ratio is held constant. Cells receiving more excitation
  receive proportionally more PV inhibition (Xue, Atallah & Scanziani 2014, *Nature* 511:596).

### 8.6 What theory says about *why*

**Optimal storage → E↔E reciprocity and sparsity.** A network of E cells that stores the maximum number
of attractor patterns robustly has (i) sparse connectivity with many zero weights, (ii) over-represented
bidirectional pairs, and (iii) stronger weights on bidirectional than unidirectional pairs. All three match
cortical data quantitatively
([Brunel 2016, *Nat Neurosci* 19:749](https://www.nature.com/articles/nn.4292.pdf)). This is an
optimality argument like ours, but for memory capacity rather than energy and wiring.

**Plasticity → assemblies and reciprocity.**
- Voltage-based STDP produces reciprocal E↔E connections when neurons code with rates, and
  unidirectional ones when they code with spike timing (Clopath et al. 2010, *Nat Neurosci* 13:344).
- Inhibitory plasticity produces detailed E/I balance (Vogels et al. 2011, *Science* 334:1569).
- Plasticity on both E→I and I→E synapses is needed to form **E–I assemblies**: like-to-like E→E plus
  tuned, reciprocal E↔I feedback, as found by Znamenskiy et al.
  ([Mackwood, Naumann & Sprekeler 2021, *eLife* 10:e59715](https://elifesciences.org/articles/59715)).

**Function of balance.** Balanced, inhibition-dominated networks allow high-capacity, noise-robust
selectivity (Rubin, Abbott & Sompolinsky 2017, *PNAS* 114:E9366). Clustered E connectivity with balanced
inhibition gives slow switching between assemblies (Litwin-Kumar & Doiron 2012, *Nat Neurosci*
15:1498).

**Motifs → dynamics.** Second-order motif frequencies alone predict much of the network's correlation
structure.
- Chains and divergent motifs increase correlations; convergent motifs don't
  ([Hu, Trousdale, Josić & Shea-Brown 2013, *J Stat Mech*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3403395/)).
- Chains increase synchrony and convergent motifs decrease it
  ([Zhao, Beverlin, Netoff & Nykamp 2011, *Front Comput Neurosci* 5:28](https://www.frontiersin.org/articles/10.3389/fncom.2011.00028/pdf)).

So motif statistics are not only anatomy: they set the dynamics.

**Feedback vs lateral inhibition.** Two classic roles:
- **Feedback** (E₁→I→E₁) controls a cell's or an assembly's own gain and stabilises it (the ISN role).
- **Lateral** (E₁→I→E₂) and **I↔I mutual inhibition** implement competition, i.e. winner-take-all between
  groups.

Znamenskiy et al. 2024 argue that cortex does both at once, with stabilisation within an ensemble and
competition between ensembles.

### 8.7 Where our results stand

| Motif | Literature | Our networks (§7, wiring_limited) |
|---|---|---|
| E↔E reciprocity | over-represented (Song 2005; Brunel 2016) | positive in ΔW at every point |
| like-to-like E→E | yes (Ko 2011; Ding 2025) | same-task preference under a wiring budget |
| tuned E→I / I→E | yes (Znamenskiy 2024; Mackwood 2021) | same-task preference under a wiring budget |
| detailed E/I balance | yes (Xue 2014) | appears under any cost |
| I↔I reciprocity | common for PV (Galarreta & Hestrin 2002) | strongly over-represented |
| back inhibition E₁→I→E₁ | **common** (Yoshimura & Callaway 2005; Znamenskiy 2024) | **avoided**: inhibition goes lateral |

The last row is the clear disagreement, but there is a nuance. At the level of **task groups**, E→I and I→E are
both within-task, so an assembly does inhibit itself. At the level of **single cells**, an interneuron avoids
inhibiting the particular E cells that excite it. The model therefore has feedback inhibition per assembly but
lateral inhibition per cell. Cortex (Yoshimura & Callaway 2005; Znamenskiy et al. 2024) has both
assembly-level and cell-level reciprocity.

Possible reasons for the cell-level difference:
- our tasks may need between-group competition more than within-group gain control;
- the L1 wiring cost may favour reusing each interneuron for other targets;
- the stabilising role may be taken over by I↔I and hub interneurons.

Which of these is true is an open question and a testable prediction.
