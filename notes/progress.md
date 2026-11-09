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
- `figure` rule -> `results/selectivity/breadth.png` (E breadth per seed: 150k at thr 0.3/0.5/0.7, 50k
  at 0.5; breadth vs min_task_acc for both sets; palette checked with the dataviz validator).
  Evidence: re-plotted from networks.csv and read back the dots, all 6 panels: one dot per network
  (20/20/20/48/20/48), max |diff| 0; hollow dots 19 = infeasible networks 19.
- Rerun from scratch — evidence: full build, sha256 of all 212 outputs, `rm -r results/selectivity`,
  `pixi run pipeline` (213 steps, 3 min 24 s), sha256 again: all 212 identical (211 CSVs + the PNG).
- `pixi run dag` -> `reports/dag.svg` — evidence: 213 job nodes; rule graph is unit_table -> summary
  -> {verdicts, figure} -> all. The job-level SVG is wide (320 kB, 204 unit_table jobs).
- Reviewer run (2026-11-09): one fresh-context subagent (read-only: Read/Grep/Glob/Bash) checked three
  summary claims against results/selectivity/, notes/idea.md and the runs' runs.csv (budget multipliers,
  cost tails, metabolic_cost). 5 findings, all **accepted**; the key numbers of 1-3 re-checked by hand:
  1. **Accepted** — 50k wiring_limited narrows with no active rate budget: final_lambda_rate at the
     1e-6 floor in 12/12 seeds, rate tail 0.0216-0.0225 < budget 0.0231; wiring binds (0.0020). E
     breadth fully separated from control at 0.3/0.5/0.7, but metabolic_cost 0.030-0.032 vs control
     0.36-0.49 (~13x lower).
  2. **Accepted** — C0's rate budget is only nominally active: multiplier clearly above floor in 1/5
     seeds (0.034; another 1.2e-4, three at 1e-6), rate tail 0.0111-0.0115 at the 0.01148 ceiling;
     metabolic_cost 0.017-0.019 (~25x below control).
  3. **Accepted** — B0 (wiring binds, rate free) is at or above control, not merely equal: at thr 0.3
     every B0 seed (9.09-9.78) > every control seed (8.26-8.76); mean 5.15 vs 4.70 at 0.5; B0 also fires
     more (metabolic_cost 0.76-0.88 vs 0.44-0.55). One moderate wiring level only (0.0054).
  4. **Accepted** — claim 1 holds: A3 max < control min at 0.3/0.5/0.7 (5.116 < 8.260, 3.288 < 4.515,
     2.091 < 2.632); A3 rate budget binds (multiplier 0.33-0.64). "Every" describes 5 vs 5, not a
     population (exact one-sided p for full separation 1/252).
  5. **Accepted** — claim 3 holds in substance, "as inaccurate as" is loose: min_task_acc A3 0.747-0.963
     (median 0.855), B0 0.756-0.933 (median 0.894); matched pairs (B0 s0 0.933/5.49 vs A3 s2 0.928/2.74)
     and A3 s3 (0.963, breadth 2.92) carry the point.
- **Revised claim wording** (registered verdicts unchanged: rules 1-6 all pass as before; only the
  explanation is revised):
  - Registered: *in the 150k set, a binding rate budget (A3) narrows E task breadth (all 5 A3 networks
    below all 5 control networks at thresholds 0.3, 0.5, 0.7), and a moderate wiring budget alone (B0)
    does not (B0 at or above control).* Replication (50k, rate_limited below control) is weak: 3/12
    feasible.
  - Accuracy: B0 has similar accuracy to A3 (overlapping ranges) yet stays broad, and the most accurate
    A3 network (inside the control accuracy range) is still narrow.
  - Withdrawn: "where narrowing appears with a wiring budget, a rate budget is also active".
  - **Exploratory (not pre-registered):** *narrowing tracks realised firing rate.* Narrow E tuning goes
    with low metabolic_cost whether a rate budget (A3, 50k rate_limited) or a tight wiring budget (50k
    wiring_limited, C0) produces it; the moderate wiring budget (B0) raises firing and stays broad.
- `reports/study.qmd` (question, approach, results, workflow + reproduce, agents, limitations, next
  steps). Edits after the first draft: expectation says it was informed by earlier exploratory work;
  the 50k set is called "second set (consistency check)" except rule 5's registered name; A3 metabolic
  cost vs rate budget explained (below); delegated / decided / checked table and planner / builder /
  reviewer roles; reproduce section (clone, `pixi install`, where the checkpoints live).
  New rule `tables` -> `results/selectivity/{wrong_if,conditions}.md`, included in the report (prose
  numbers stay static). `pixi run dag` now uses `--rulegraph` (6 nodes, 6.5 kB).
  Evidence: `pixi run report` exit 0, **no warnings**; study.html has 4 tables, both includes
  resolved, figure and DAG embedded; no notebook paths or earlier numbers in the HTML.
- Metabolic cost vs rate budget (checked in code): same quantity (`rate_reg`, population-weighted mean
  squared rate). The budget acts on training batches (one random task per step, input noise on,
  recurrent noise 0.1); `train_rate_cost_tail` = 0.0011 for A3. `metabolic_cost` = same formula after
  training on noise-free eval trials (20 tasks x 10 eval seeds x 64): 0.0018-0.0022 for A3.
  Ratio eval/train 1.3-2.0 in every budgeted condition incl. slack-rate B0 (1.34-1.43). Noise is not
  it: A3 seed 0 re-simulated, eval conditions 0.00183 vs training-like noise 0.00187.

## Next — one at a time
- [ ] Update PRESENTATION.md slide 22 with the verdict (see Internal cross-checks) — if still wanted.

## Internal cross-checks (project context, not for the report)
- An earlier notebook computed the same metric: `notebooks/ei_modularity_150k.ipynb` ->
  `notebooks/ei_modularity_150k_summary.csv` (column "E: tasks per neuron"). The pipeline should
  reproduce it to rounding; a mismatch means one of the two implementations is wrong — find out
  which before trusting either.
- PRESENTATION.md slide 22 ("constraints make neurons specialise", from 4k-step runs) depends on
  this result; update it once the verdict is in.

## Open questions
- Why is the evaluated rate cost 1.3-2.0x the training-time cost the budget acts on (same formula,
  same weights; not noise)? Candidates not yet checked: per-task weighting of the training stream,
  trial lengths, the end-of-training tail window.
- **50k set, wiring_limited is as narrow as rate_limited** (thr 0.5, feasible: wiring_limited 3.339,
  n=10, max 3.733; rate_limited 3.388, n=3; control 4.920, min 4.309). Rule 6 is decided on the 150k
  set only, so the verdict stands. Reviewer finding 1: its rate budget does not bind, so this is a
  wiring-driven narrowing at a tighter wiring level (0.0020 vs B0's 0.0054), with a ~13x lower firing
  rate. Is it the wiring level or the firing rate? Needs a wiring sweep with rate measured.
- 50k rate_limited's wiring budget (0.0237) also binds (conn tail = budget); looser than the 150k
  control's realised wiring cost, probably minor, unchecked against the 50k control.
- `pixi` is not on PATH in this shell; it is at `~/.pixi/bin/pixi`.
- Rate cost is L2: why would it produce specialization? (mechanism, out of scope for this slice)
