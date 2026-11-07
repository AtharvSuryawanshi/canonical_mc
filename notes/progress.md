# Progress

Goal: does a rate budget, but not a wiring budget, make E neurons more task-selective? See
notes/idea.md (the spec; self-contained, written for the report).

## Done
- Branch `agent-course` created from main.
- Locked analysis env: pixi.toml + pixi.lock (CPU torch, snakemake, quarto, graphviz) — evidence:
  `pixi run test` 3 passed; a 150k checkpoint loads with `task_variance` (20, 256).
- Spec: notes/idea.md, with "Wrong if" fixed before any pipeline run.

## Next — one at a time
- [ ] Plan (plan mode) -> fill "## Plan" in notes/idea.md — done when: the plan names rules, files
      and one check per step.
- [ ] `cmc/selectivity.py` + ground-truth test — done when: `pixi run test` passes, including a
      synthetic case with known breadth.
- [ ] First rule: one checkpoint -> per-unit table — done when: `pixi run dry` lists 20 jobs for the
      150k set.

## Internal cross-checks (project context, not for the report)
- An earlier notebook computed the same metric: `notebooks/ei_modularity_150k.ipynb` ->
  `notebooks/ei_modularity_150k_summary.csv` (column "E: tasks per neuron"). The pipeline should
  reproduce it to rounding; a mismatch means one of the two implementations is wrong — find out
  which before trusting either.
- PRESENTATION.md slide 22 ("constraints make neurons specialise", from 4k-step runs) depends on
  this result; update it once the verdict is in.

## Open questions
- Rate cost is L2: why would it produce specialization? (mechanism, out of scope for this slice)
