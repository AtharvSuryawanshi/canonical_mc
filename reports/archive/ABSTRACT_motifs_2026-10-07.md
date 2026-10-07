> **Archived 2026-10-07.** Motif analysis is parked for now (see the project redirection: task-variance
> clusters and E/I weight distributions first). Kept verbatim so it can be picked up later.
> Code and figures are still live: `cmc/motifs.py`, `notebooks/moo_zoom_analysis.ipynb` §3-§5.

# COSYNE 2027 abstract: planning notes

## Submission requirements (cosyne.org/abstracts-submission, checked 2026-10-07)

- **Deadline: 18 Oct 2026** per the website (we planned for 16 Oct; check the exact time zone on the
  submission portal). Decisions in mid-December 2026.
- **Format:** a 2-page PDF (A4 or US Letter), font ≥ 12 pt (10 pt allowed for captions and references),
  margins ≥ 0.5". Anything after page 2 is cut.
- **Title:** ≤ 100 characters including spaces, in sentence case.
- **300-word summary** (required). It becomes the program text if accepted, so it must stand alone.
- **Rest of the 2 pages:** methods, results and conclusions in more detail. Figures and equations
  are encouraged.
- **Double-blind:** no names, affiliations or acknowledgements in the text, figures or PDF metadata.
  Self-citations go in the third person. Check that figure files and paths don't contain "atharv" /
  "Helmholtz".
- **One presenting-author submission per person** (co-authoring others is fine).
- **Review criteria:** significance, originality, clarity, relevance to COSYNE. Pure ML without
  ties to the brain may be rejected → lead with the cortical motif data.

## Title options (≤ 100 characters)

- Metabolic and wiring costs shape cortical E/I connectivity motifs in multitask networks (87)
- Wiring cost reveals cortex-like E/I motifs in recurrent networks trained on many tasks (86)
- Which cortical connectivity motifs are explained by metabolic and wiring cost? (78)

## Crucial points for the 300-word summary

**Background / gap**
- Cortical connectivity has robust non-random 2–3 neuron E/I motifs: E↔E reciprocity, like-to-like E→E,
  function-specific E↔I, detailed E/I balance, I↔I reciprocity.
- It is unknown which of these follow from computing many tasks under the brain's energy and wiring
  constraints.
- Previous multitask RNNs (Yang et al. 2019) had no Dale's law and no costs. Wiring-constrained RNNs
  (Khona et al. 2023; Achterberg et al. 2023) had no metabolic cost and no E/I motif analysis.

**Approach**
- Dale's-law RNN (80% E / 20% I) trained on 5 cognitive tasks from the Yang battery: perception,
  anti-response, evidence integration, context-dependent decision, working memory.
- Firing (metabolic) and recurrent wiring as **budgets** rather than hand-tuned penalties. Wiring is
  enforced exactly by L1-ball projection, rate by a learned Lagrange multiplier.
- NSGA-III maps the 3-way Pareto front (worst-task accuracy, firing, wiring). It reaches ~50% more of
  the front than a weighted-sum grid **[UPDATE with the long-training front]**.
- Many seeds at chosen points, with motifs measured against Dale- and degree-preserving shuffle nulls.

**Results**
- With enough training, ~3% of recurrent synapses and ~1/20 of the firing give the same accuracy as the
  unconstrained network (0.96 vs 0.97). Earlier, the wiring budget mainly slowed learning.
- Under a wiring budget, cortex-like motifs appear: like-to-like E→E, function-specific E→I and I→E
  (task E–I assemblies), I↔I reciprocity at 5× chance, and detailed E/I balance (corr −0.35 → +0.2 to +0.4).
- One mismatch is also a prediction: back inhibition (E→I→same E) is avoided (0.02× chance) and inhibition
  becomes lateral, whereas cortex shows dense reciprocal E↔I.
- The same motif directions are already in the learned ΔW without any cost. The wiring budget removes the
  random initial structure, so the learned motifs become the circuit.

**Take-home**
- Task demands set the motifs; wiring cost makes them dominate. Firing cost mainly drives inhibition-
  dominated activity (I/E 0.88 → 1.6–2.6).
- Motifs that cost does **not** reproduce (back inhibition) point to computations or cell types the
  model lacks.

## Figure plan for the detail section (1–2 figures)

1. **A:** schematic of the network, tasks and costs. **B:** 3D Pareto front with the zoom points.
   **C:** accuracy against training steps, control vs wiring_limited.
2. **Motif scorecard:** observed/null per motif, control vs wiring_limited, next to the cortical
   reference values. **Inset:** sharpening of the motifs over training.

## Before submitting

- [ ] Re-run the front at ≥16k steps with g = 0.3, and replace every [UPDATE] number.
- [ ] Degree-preserving null for E/I balance (currently only out-degree preserving).
- [ ] At least 5 seeds for every claimed motif. Report the z-scores.
- [ ] Verify all citations (Khona, Znamenskiy year and journal).
