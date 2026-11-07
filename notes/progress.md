# Progress

Goal: does a rate budget, but not a wiring budget, make E neurons more task-selective? See
notes/idea.md (the spec; self-contained, written for the report).

## Done
- Branch `agent-course` created from main.
- Locked analysis env: pixi.toml + pixi.lock (CPU torch, snakemake, quarto, graphviz) — evidence:
  `pixi run test` 3 passed; a 150k checkpoint loads with `task_variance` (20, 256).
- Spec: notes/idea.md, with "Wrong if" fixed before any pipeline run.
- Plan: "## Plan" in notes/idea.md (2026-11-07), with two pre-run clarifications (rule 5 on feasible
  networks only; rule 2 by seed separation at 0.3 or 0.7).
- Found while planning: in the 50k replication set only 3/12 rate_limited networks are feasible
  (min_task_acc >= 0.6; control 12/12, wiring_limited 10/12, knee 4/12, from summary.json). So rule 5
  rests on 3 networks: weak either way. Also: that set has 96 .pt files, 48 of them `step_025000/`
  snapshots; the pipeline uses the 48 final ones only.
- `cmc/selectivity.py` + `tests/test_selectivity.py` — evidence: `pixi run test` 8 passed (5 new:
  known breadth 1/5/20/3/12 recovered, scale-invariant, strict threshold, inactive excluded, E/I by
  sign).
- Snakefile, first rule `unit_table` (threshold as wildcard) — evidence: `pixi run dry` lists 20 jobs
  per threshold for 150k, 48 for 50k (204 total); one checkpoint (150k control seed 0, thr 0.5) in
  4 s gives E breadth 4.691, I 4.635 = notebook CSV.
- Per-unit tables are no longer tracked (`results/selectivity/*/thr*/units/` in .gitignore); the
  committed outputs will be networks.csv, wrong_if.csv and the figure. "## Plan" steps 2-3 now refer
  to "an independent earlier implementation" (the notebook below) instead of naming it.
- `summary` rule -> `{set}/thr{thr}/networks.csv` — evidence: 150k thr 0.5 built in 33 s (20 unit
  tables, -c4); all 20 networks match `notebooks/ei_modularity_150k_summary.csv`: max |diff| 0.000
  for E breadth, I breadth and min_task_acc (rounded to 3 decimals, merged 1:1 on point/seed).
- `verdicts` rule -> `results/selectivity/wrong_if.csv` (`cmc.selectivity.wrong_if`; operators checked
  against "## Plan" step 4; 2 boundary tests: ties count against the claim, control range closed).
  Evidence: tests/test_selectivity.py 7 passed; full build 2 min (-c4). All rules
  PASS:
  1. A3 max 3.288 < control min 4.515.
  2. thr 0.3: A3 max 5.116 < control min 8.260; thr 0.7: 2.091 < 2.632 (narrowest margin).
  3. one A3 seed (3, acc 0.963) is above control's lowest acc 0.939; breadth 2.918, outside control
     [4.515, 4.798]. Rests on that one seed. Spearman(breadth, acc) over 20: rho 0.37, p 0.11.
  4. A3 mean active E 195.0 vs half of control 101.9.
  5. feasible: rate_limited 3.388 (n=3) < control 4.920 (n=12) — weak by the pre-run rule; all-12:
     3.359 vs 4.920 (confounded, reported only).
  6. B0 is not below control: B0 max 5.485, and every B0 seed (min 4.764) is >= control min 4.515.

## Next — one at a time
- [ ] `figure` rule -> `breadth.png`; then rerun-from-scratch sha256 check; then `pixi run dag`.

## Internal cross-checks (project context, not for the report)
- An earlier notebook computed the same metric: `notebooks/ei_modularity_150k.ipynb` ->
  `notebooks/ei_modularity_150k_summary.csv` (column "E: tasks per neuron"). The pipeline should
  reproduce it to rounding; a mismatch means one of the two implementations is wrong — find out
  which before trusting either.
- PRESENTATION.md slide 22 ("constraints make neurons specialise", from 4k-step runs) depends on
  this result; update it once the verdict is in.

## Open questions
- `pixi` is not on PATH in this shell; it is at `~/.pixi/bin/pixi`.
- Rate cost is L2: why would it produce specialization? (mechanism, out of scope for this slice)
