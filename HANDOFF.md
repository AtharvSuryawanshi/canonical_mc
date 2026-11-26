# Session handoff, updated 2026-11-26 (Windows) — read §0 first, then §12 (deck revision + HTML deck, newest), §11 (analyses on the new points), §10 (Yang front + new 5 points), then §9

Read this first in a new Claude Code session, together with `CLAUDE.md`. **Section 8 is the newest**: the
Driscoll-style analysis (task variance per period, fixed points) and the clustering-free like-to-like wiring test,
which is currently the strongest result. Both are now in the deck as a draft **Experimental** section (§7).
§8.5 records the open "how does it work locally?" question and what was ruled out. Section 7 is the talk in
`presentation/`. Sections 1–6 are the project state.

## 0. BIG FIX, 2026-11-19 (Windows): task generators and loss now match Yang. All earlier runs are "v1".

A line-by-line audit against `gyyang/multitask` (subagent + bit-exact comparison on the same seeds) found
that 6 of 20 tasks differed from Yang since the first commit, and that the loss mask entered linearly:
- **Go/Anti/RT/Dly Go/Dly Anti never used ring 2** (no `mods`; `Trial.add` defaulted to ring 1).
- **RT Go/Anti switched the fixation input off at stimulus onset** (Yang keeps it on: the stimulus is the go signal).
- **Go/Anti switched the stimulus off at go** (Yang keeps it on through the response).
- **Loss mask linear (1/2/5/10) instead of Yang's squared (1/4/25/100).**

Fixed in `cmc/task.py`, `cmc/train_cog.py`, `cmc/batched.py`; details, checks and remaining deliberate
differences (easy coherence band, 16-unit rings, mixed batches of 32) in `reports/FIXED_ISSUES.md` #15-#18.
`config["task_version"]`: `"yang"` for new runs, old checkpoints restore as `"v1"` (same accuracy as before,
checked). All six fixed tasks are bit-identical to Yang now. `tests/test_tasks.py` (21 tests incl.
`test_batched.py`, all pass; pytest installed in cmc_env).

**Consequence:** every run and result below (fronts, 4pt and grid runs, modularity, AUC, lesions, deck numbers)
is on the v1 battery. Comparisons between budget points stay internally consistent, but nothing is "the Yang
battery"; modality-related results are the most affected. The NSGA-III cluster run (§3) is also v1.

Smoke test (`results/scripts/smoke_task_fix.py`, 1k steps, the 6 changed tasks, 256 Dale units, fanin, no budget,
yang vs v1, same seed): both train. Go/Anti learn faster with the fix (stimulus visible during the response), RT
the same, Dly Go/Anti slower (0.47/0.45 vs 1.00/0.69 at 1k: memory for two rings). Figures
`results/front/smoke_task_fix_curves.png`, `smoke_task_fix_trials.png`.

**Status 2026-11-26:** the retraining is done (§10): the 4-point Yang run, the Yang NSGA-III front (4 searches,
573 networks) and 5 new points chosen on that front. The analyses are not yet rerun on the new points.

**Next (agreed 2026-11-19):** retrain on the fixed battery, bigger runs after the smoke test: first 4 points x 5 seeds, 150k,
fanin (+ one save-at seed), then the front. Then rerun the analyses (tuning breadth, like-to-like AUC, E/I
modularity, per-period TV, lesions) and compare with v1.

Lesion work from 2026-11-19 (v1 networks, seed 4, `results/scripts/fig_lesion_period.py`,
`fig_lesion_contrast_grid.py`): per-period clusters, Driscoll Fig. 5 module contrasts. Results depend on k,
not on k-means vs Ward; Driscoll's own k rule picks 3 for control/A3/B0 and 39 for C0 (E+I together), 3-5 for
everything when E and I are separate. At a shared k (Ward 14 E / 5 I) A3 lesions do the least damage and are the
most specific; control and C0 differ only modestly. Eval trials: Go/Anti matching-task rings drawn once per batch
(as in Yang); task-variance profiles are reliable (split-half r ≈ 0.97).

## 1. Working rules

- **Plan first, edit only after "execute" / "go".** Read-only exploration is fine.
- **Answer only what is asked.** No unrequested suggestions, next steps, offers or commands. If something
  looks urgent, state the fact in one line without proposing an action.
- **Python env:** everything runs in `cmc_env` (`conda run -n cmc_env ...`), never system Python.
- **g stays 1.0** (init gain). The g = 0.3 runs were exploratory (low-rank-RNN regime, out of scope).
- **Scope:** E and I only, no interneuron subtypes, no canonical microcircuit, no spatial wiring.
- **Baseline** = the Dale network with no budget (`control`). Non-Dale `LeakyRNN` is out of scope.
- **Task names in figures:** Yang 2019 display names (`cmc.task.rule_name`); internal names only as data keys.
- **The `agent-course` branch is a side project** (an agentic-workshop demo on this project). The main work
  is `main`, and right now the presentation.
- The user works on Linux and Windows on the same repo; **presents on Windows**. gh CLI is not on Windows.
- Supervisor focus (since 2026-10-07): task-variance clusters along the front (20 tasks), and E→E, E→I,
  I→E, I→I weights under the constraints.
- **How the user wants explanations:** slowly, step by step, little text, no walls of text. A GIF over *time*
  (one dot = one trial, colour = stimulus angle) worked very well; the GIF over the interpolation α did not.
  Side-by-side unconstrained vs constrained comparisons are wanted. The user likes the
  "two distributions + ROC/AUC" figure style (`wiring_roc_all_*`).
- Temporary GIFs go in the scratchpad; the user asked to keep the useful ones in `results/front/dynamics_gifs/`.
- **Figures and slides (2026-11-20):** accuracy is always RdYlGn on 0–1 (red 0, green 1) and called "worst-task
  accuracy"; on slides the points are control / firing constraint / wiring constraint / both constraints; talk
  figures are separate from analysis figures (§9), crisp, big normal-weight text; never delete the originals; no
  AI-sounding phrasing on slides.

## 2. Where the project is

Direction: **how E and I organise into task modules** under metabolic (rate) and wiring budgets. cog20 (all
20 Yang tasks), 150k steps, W_in and W_out trained, W_out `fanin` init, g = 1. **These are fixed decisions.**

Main result (5 seeds, `runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k`, write-up
`reports/ei_modularity/ei_modularity.pdf`, numbers `reports/ei_modularity/fig/summary.csv`):

| point | rate budget | wiring budget | closest type |
|---|---|---|---|
| control | none | none | 3: inhibition between modules (I→E ρ < 0 in all 5 seeds) |
| A3 | 0.0011 | none | 1: sharp E tuning, wiring ignores tasks |
| B0 | none | 0.0054 | weak, seed-dependent 2 |
| C0 | 0.0115 | 0.0044 | 2: E+I modules (E→I ρ +0.36 to +0.62, E→E within/between 2.8–7.8) |

Caveats found today:
- **C0's rate budget barely binds:** the rate multiplier is at its floor (1e-6) in 3/5 seeds. C0's narrow tuning
  likely comes from low firing caused by the wiring budget, not an active rate limit (from the `agent-course`
  reviewer pass). Affects how "both budgets" is phrased.
- The firing limit is soft: networks end ~1.7× above their rate budget (A3, C0).
- **Rate cost is L2** (`rate_kind="l2"`, inherited from Yang, no dead-unit pressure); L1 = mean rate is the literal
  ATP cost but was never run. Wiring L1 is principled (convex surrogate for connection count).
- The `agent-course` branch has the robust version of the breadth result: E tasks/neuron control 4.70,
  A3 2.90, B0 5.15, C0 3.52; all 5 A3 below all 5 control at thresholds 0.3/0.5/0.7; accuracy and active-unit
  count ruled out. `notes/idea.md`, `notes/progress.md`, `results/selectivity/` on that branch.

## 3. Runs that matter (all cog20, 150k, nothing frozen)

| run | init | points × seeds | per-step accuracy? | checkpoints in git? |
|---|---|---|---|---|
| `runs/moo/dale_all_grid29_fanin_150k_s{0,1,2}_{A,B}` | fanin | 29 × 3 (budget grid via `cmc.moo --points`, no search) | no | no |
| `runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k` | fanin | 4 × 5 (Oct 25) | no | yes (final) |
| `runs/moo_zoom/dale_all_woutfree_fanin_4pt_1seed_150k_saveat` | fanin | 4 × 1 (Nov 17) | yes (1k…150k) | yes (all steps) |

- Combined fanin front: 111 networks, 89 feasible, 38 Pareto. Wiring −2.75 (log10) infeasible everywhere.
- The 150k front is a **point scan**, not NSGA-III: a log-spaced grid over the region the earlier cog20 NSGA-III
  run (`runs/moo/dale_all_nsga3_budget_p48g3_50k`, pop 48 × 3 gen) found. (v1. The real NSGA-III front, on the
  Yang battery, is in §10.) Same command without `--points`:
  `python -m cmc.moo --task-battery all --steps 150000 --w-out-init fanin --g 1.0 --seed 0 --pop-size 48 --n-gen 3
  --n-partitions 8 --rate-budget-min 0.002 --rate-budget-max 0.15 --conn-budget-min 0.002 --conn-budget-max 0.03
  --batched --batch-pop 24 --workers 2`.
- **Same seed ≠ same network across these two zoom runs:** batched networks share one trial stream, and the
  runs batched 20 vs 4 networks. Within the 5-seed run, the same seed at different points = same initial weights
  and same trials, only the budget differs (a matched comparison).
- Save-at curves: control worst task ~0.96 from 75k; **A3 not converged** (0.50 at 100k → 0.73); B0 → 0.91; C0 ~0.87.

### Example network: seed 4 of the 5-seed run (chosen 2026-11-18)

Chosen as the **clearest constrained-vs-unconstrained contrast of the 5 seeds**, on all three measures; to be
labelled as such on the slides, with 5-seed ranges next to it. Numbers (`fig_tv_ei_fanin.wiring`):

| seed 4 | worst / mean acc | ρ I→E | ρ E→I | E→E w/b | I→I w/b | k E / I |
|---|---|---|---|---|---|---|
| control | 0.95 / 0.99 | −0.69 | −0.13 | 1.07 | 0.94 | 4 / 3 |
| A3 | 0.78 / 0.96 | +0.09 | 0.00 | 1.04 | 1.08 | 16 / 9 |
| B0 | 0.89 / 0.99 | +0.17 | +0.33 | 1.22 | 2.55 | 4 / 5 |
| C0 | 0.89 / 0.98 | +0.16 | +0.59 | 7.77 | 2.05 | 13 / 5 |

Other findings from today's checks: the report method (`make_figures.py`) and `fig_tv_ei_fanin.py` agree to
~0.05 on the same network. The report's displayed seeds (`best_seed.json`: control 1, A3 3, B0 0, C0 2) are
**not** typical (C0 seed 2 is the least typical). Most typical seeds: control 3, A3 1, B0 4, C0 3. Yesterday's
save-at network is weaker only in B0 I→E (≈ 0) and C0 E→E w/b (2.45). I→E in C0 is weak in every seed.

## 4. Scripts (run from repo root, in cmc_env)

| script | output (in `results/front/` unless stated) |
|---|---|
| `results/scripts/fig_front_top_fanin150k.py` | `cog20_150k_fanin_front_top.png` (filled contours of worst-task accuracy since 2026-11-20: thin-plate fit of per-budget seed means, inside their hull; rings the example network, `EXAMPLE = (run, seed)`), `cog20_150k_fanin_front.gif`, `front_networks_150k_fanin.csv` |
| `results/scripts/fig_tv_ei_fanin.py [step] [--run saveat\|5seed] [--seed N]` | `tv_ei_fanin_<tag>.png`, `_sil.png`, `_wiring.png`; default tag `150k` (save-at, seed 0); `--run 5seed --seed 4` → `tv_ei_fanin_5seed_s4_150k*` |
| `results/scripts/fig_acc_vs_steps_grid.py` (new) | `acc_vs_steps_fanin_grid.pdf`: 2 × 2 learning curves, one shared legend (save-at run) |
| `results/scripts/fig_acc_vs_steps.py <point> [rays3\|fanin]` | `acc_vs_steps_<point>[_fanin].png` |
| `results/scripts/fig_grid_proposal_150k.py` | `cog20_150k_grid_proposal.png`, `grid_proposal_150k.csv` |
| `presentation/make_frames.py` | `presentation/frames/front_000…059.png` (rotating fanin-only front for `animate`) |
| `presentation/figs/crop_tv_rows.py [tag]` | `presentation/figs/tv_ei_<point>.png`, `wiring_<point>.png` (default tag `5seed_s4_150k`) |
| `presentation/figs/{yang2019,song2016}_crops.tex` | vector crops of paper figures (`pdflatex` in `presentation/figs/`; PDFs in `papers/`) |
| `results/scripts/like2like.py [--run R] [--plot-only\|--recompute]` (§11) | cached like-to-like + matched sparsity: `results/cache/<R>/`, `presentation/figs/talk/like2like_{example_s<seed>,seeds,matched}_<R>.pdf` |
| `results/scripts/silhouette_seeds.py [--run R] [--basis task\|period] [--plot-only\|--recompute]` (§11) | cached silhouette vs k over seeds: `results/cache/<R>/silhouette_<basis>.csv`, `presentation/figs/talk/silhouette_seeds_<basis>_<R>.pdf`, `results/front/silhouette_k_<basis>_<R>.png` |

Group wiring math (`fig_tv_ei_fanin.wiring`): w̃_ij = |W_ij| / c_i (i presynaptic = row; c = 1 E, n_E/n_I I);
block mean B_gh over pairs without self-pairs; z vs 500 label shuffles within E and within I; groups < 4 greyed;
r_gh = Pearson of mean group task profiles; ρ = Spearman(r, z) over different-group pairs per quadrant;
within/between = mean diagonal B / mean off-diagonal B (raw, not z). **Open points:** strength vs presence
(pruned synapses count as 0; in C0 it mostly measures connection count), z vs raw inconsistency, small-group noise.
**Superseded by the clustering-free like-to-like AUC (§8.3)**, which the user prefers; slides 16–19 still use this.

## 5. Docs

- `reports/theory.md`, `reports/MOO_BUDGETS.md` (budget enforcement, open issues), `reports/FIXED_ISSUES.md`.
- `reports/PRESENTATION_old.md`: the old outline (renamed by the user); the new one is `presentation/presentation.md`.
- `reports/Pymoo_and_the_front.pdf`: one-page note on NSGA-III and budget enforcement.
- `papers/yang2019.pdf`, `papers/song2016.pdf`.

## 6. Figma

File **Dale-RNNs** (`https://www.figma.com/design/X0MquCTzak5plVGUsZgnDq/Dale-RNNs`), user's student team.
- Page 1: the user's original setup schematic (untouched by Claude).
- Page "Talk figures": `Fig1_intro` (detailed: 20 tasks / 5 families, kept for later), `Fig1_intro_question`
  (slide 1: Task A–D → many neurons → Shared? / Separate?), `Fig_setup` (copy of Page 1 in a frame, exported for
  the setup slide; the user has since edited small things in Figma, re-export if needed).
- Export: Figma MCP `download_assets` with `defaultFormat: pdf` → `presentation/figs/*.pdf`.
- Style: Inter, neuron circles #D9D9D9 with black outline, Okabe–Ito colours for tasks/families.

## 7. The talk (`presentation/`)

Built in Beamer, `presentation/main.tex`; outline with talking points, sources and code mapping per slide:
**`presentation/presentation.md`** (keep it in sync with main.tex). Build: `cd presentation && pdflatex main.tex`
twice (needs `animate`; Okular is installed on Linux; on Windows present with Acrobat Reader or Okular).

Current deck (30 pages), outline numbering. **Since 2026-11-20:** slides 10–18 use *talk figures*
(`presentation/figs/talk/`, made by `presentation/figs/talk_figs.py`, drawn at slide size with big labels and
little text); the detailed analysis figures in `results/front/` are unchanged and stay the reference. Points are
named *control / firing constraint / wiring constraint / both constraints* on slides (codes A3/B0/C0 only in notes
and analysis figures, which now read e.g. "Firing constraint (A3)"). Like-to-like slides are no longer tagged
experimental. Two experimental lesion slides (math + `lesion_period_s4_ward_contrast_grid.png`) sit before
"Thank you". Wording was de-AI'd ("X, not Y" phrasing removed).

| # | slide | made from |
|---|---|---|
| 0 | title | Beamer |
| 1 | One circuit, many tasks | Figma `fig1_intro_question.pdf` |
| 2 | Yang et al. 2019 | vector crops Fig 1b, 2a–c |
| 3 | Song, Yang & Wang 2016 + Dale math (W = W⁺D) | vector crop Fig 1 |
| 4 | Real neurons pay for firing and wiring | 3 bullets, small grey citations, no math |
| 5 | Setup | Figma `fig_setup.pdf` |
| 6 | The network (math from `DaleRNN`) | Beamer |
| 7 | Three objectives (L_task, C_fire, C_wire) | Beamer |
| 8 | Holding the network to a budget (L1 projection; self-tuning λ) | Beamer |
| 9 | Mapping the front: NSGA-III (pymoo) | Beamer |
| 10 | The front (**placeholder: point scan**, animated) | `frames/` |
| 11 | Four points on the front (5-seed means + seed-4 row, seed 4 ringed) | front top + table |
| 12 | Do they converge? (separate save-at run, labelled) | `acc_vs_steps_fanin_grid.pdf` |
| 12a | Like-to-like wiring (math: s_ij, connected, AUC, ROC, shuffle p), tagged experimental | Beamer |
| 12b | Like-to-like, seed 4 | `wiring_roc_all_s4.png` |
| 12c | Like-to-like, all 20 networks + sparsity caveat | `wiring_roc_all_5seed.png` |
| 13 | Task variance and the number of groups (TV + silhouette math) | seed-4 `_sil.png` |
| 14–15 | E and I groups (2 points per slide) | seed-4 crops |
| 16–19 | Task similarity and wiring, one per point | seed-4 crops, captions with 5-seed ranges |
| – | **Experimental** divider (draft section, user decides later what stays; like-to-like moved up to 12a–c on 2026-11-19) | Beamer |
| E1 | Task variance per task period (math) + Driscoll Fig 3a crop | `figs/driscoll2024_crops.pdf` |
| E2–E3 | Task periods, Driscoll style: control/A3, B0/C0 | `tv_period_driscoll_<p>_s4.png` |
| L1–L2 | Lesions on task-period groups (math; contrast grid as is), experimental | `lesion_period_s4_ward_contrast_grid.png` |
| – | Thank you | |

User's style preferences for the slides:
- Professional audience: **state findings as the papers do, no pop-science lines** (no "2% body mass, 20% energy").
- Little text, mostly visual; talking points go in `presentation.md`, not on slides. Math only where it is the
  setup (slides 6–8, 13); intro slides stay math-free and don't give away the setup.
- **No animations unless they add information.** Plan: replace the rotating front by a static 3D view + top view
  when the NSGA-III run arrives.
- Prior work builds up one constraint at a time: Yang (many tasks) → Song (E/I) → costs. Khona/Achterberg dropped
  from the main line (spatial; our wiring cost is not).

### Still to do

1. **Wiring method slide**: the user is writing the math; then check it against `fig_tv_ei_fanin.wiring` and
   settle the open points in §4.
2. Replace slide 10 with the NSGA-III front when the cluster run is done (re-render; static, not animated).
3. "Tasks per neuron" slide (rate budget sharpens tuning; `agent-course` robustness checks) — missing.
4. Summary, limitations (L2 rate cost, L1 ≠ wire length, 5 seeds, soft clusters, A3 not converged, C0 rate budget
   barely binds), open questions, thank you.
5. **One E/I colour convention**: Figma setup figure = red E / blue I; Beamer (Song colours) = blue E / red I.
6. Figma `Fig_setup` says "Connectivity cost (L1)"; slides say "wiring". The talk figures now say "wiring cost";
   the analysis figures (`results/front/`) and Figma still say "Connectivity cost". Unify.
7. Commits: HEAD `fee34bf` (task-battery fix + lesions, Windows). Everything in §9 is uncommitted until the
   user commits it. `presentation/figs/driscoll2024_crops.pdf` is untracked and 12.8 MB (the crop embeds the whole
   page's images); building it needs `papers/driscoll2024.pdf`, which is gitignored, so commit the PDF if the deck
   must build on Windows. LaTeX aux files and the frame PNGs could go in `.gitignore`.
8. Experimental section now = Driscoll per-period TV (E1–E3) + lesions (L1–L2). Like-to-like (§8.3) moved into
   the main line after "Do they converge?" and is no longer tagged experimental. Not in the deck: fixed points
   (§8.2), the E→E direction chain. The old group-wiring slides (task similarity and wiring, one per point) and
   the like-to-like slides overlap: pick one. Cosmetic: the Driscoll Fig 3a figures are small; the grey
   "experimental" tag sits mid-title.
9. Like-to-like results are not cached: `fig_wiring_like2like.network` (20 networks × 500 shuffles, ~3 min) is
   recomputed by `fig_wiring_like2like_seeds.py`, `fig_wiring_matched_sparsity.py`, `fig_wiring_roc_all.py` and
   `talk_figs.py like2like`. Offered to cache per network (`results/front/like2like_<point>_s<seed>.npz`, own
   shuffle seed per network: AUCs identical, p-values move slightly); the user said fine for now, not done.
10. Talk versions still missing for: the task-similarity/wiring slides, the Driscoll slides, the lesion grid
   (user: later).

## 8. Driscoll et al. 2024 and like-to-like wiring (2026-11-18, evening)

Paper: `papers/driscoll2024.pdf`, Driscoll, Shenoy & Sussillo, *Flexible multitask computation in recurrent
networks utilizes shared dynamical motifs*, Nat Neurosci 27:1349 (2024). Code: `github.com/lauradriscoll/flexible_multitask`
(their variance code: `stepnet/analysis/variance_lnd.py`, `clustering.py`, `tools_lnd.py::make_dendro`);
fixed points: `github.com/mattgolub/fixed-point-finder` (has a PyTorch version); tasks: `github.com/gyyang/multitask`.
Key ideas: task variance **per task period** (not per task); clusters = computations ("dynamical motifs") shared
across tasks; fixed points + input interpolation show which motifs are shared; cluster lesions are modular;
transfer learning by learning only a new rule input. Clusters align with single units only with non-negative rates
+ activity regularisation (their tanh nets have the same structure, rotated off the unit axes).
Their setup: 15 tasks, softplus, no Dale, L2 on activity and weights at only 1e-6.

### 8.1 Task variance per period (all networks, seed 4 in detail)

- `cmc/runner.py::period_variance` (new, next to the unchanged `activity_summary`): same quantity (variance across
  trials per time step, noise-free, same trials), averaged per non-fixation epoch (stim1, delay1, stim2, delay2,
  go1) → 59 (task, period) rows for cog20. **Exact check:** the length-weighted mean of a task's periods reproduces
  the stored per-task task variance to 3e-7.
- `results/scripts/fig_tv_period.py [--point P] [--seed S]`: per task vs per period heatmaps, k-means E and I
  (same `cluster_pop`), crosstab + ARI; caches `results/front/tv_period_<point>_s<seed>.npz` (all 20 exist now).
- `results/scripts/fig_tv_period_driscoll.py [--points ...] [--seed S]`: Driscoll Fig 3a replica (Ward on units
  and on task periods, k by global silhouette 3..39, one population + E/I strip) → `tv_period_driscoll_<p>_s<s>.png`
  (all four points, seed 4, in the deck as E2–E3; k units/periods: control 3/4, A3 3/5, B0 3/3, C0 39/20). C0 unit k hits the upper bound 39 (no natural k there).
- Seed 4 findings: control = groups follow **task families**; A3 and C0 = groups follow **computations** shared
  across tasks (stimulus per modality, memory, shared response, memory-guided response separate). B0 ≈ control.
  Driscoll sparsity (fraction of entries < 15% of unit max), E / I: control 0.43/0.39, A3 0.79/0.62, B0 0.48/0.48,
  C0 0.69/0.63. I groups stay task-family-like and broad everywhere. One seed only for these; k weakly determined.

### 8.2 Fixed points (C0 seed 4; control only in the GIFs)

- `cmc/fixed_points.py` (new): speed minimisation (Adam) + **exact ReLU solve** (fixed point of the active set,
  self-consistency check) + exact Jacobian `(1-a)I + a diag(on) W^T`; `relax` returns one fixed or slow point
  per start (slow = speed < 1e-3/step).
- `results/scripts/fig_fp_delaygo.py --point C0`: Dly Go, delay → response input interpolation (Driscoll Fig 1) →
  `fp_delaygo_C0_s4.png/.npz`. C0: memory sits on a **slow region**, not on exact fixed points (13–23 exact points,
  max|eig| 1.002–1.02, ~6 stored angles); state slows 60× by end of delay; correct after +0/+2/+5/+10 s extra
  delay 1.00/0.92/0.84/0.80; fixed points rotate into readout space (fixation output 0.8 → 0.05, bump amplitude
  0.2 → 0.9), one reorganisation at α ≈ 0.5.
- `results/scripts/fig_fp_task_pair.py --point C0`: Dly Go vs Dly Anti (matched inputs, only the rule swapped),
  per period, + control pair Dly Anti vs RT Go → `fp_pair_C0_s4.png/.npz` (~9 min). Subspace overlap (B's
  variance in A's top-2 PCs vs own): stimulus 0.07 vs 0.54 (different motifs), delay 0.33 vs 0.49 and response
  0.46 vs 0.58 (shared); control pair 0.06 vs 0.44 (different, stability 1.00 → 0.80). Delay/response relevant
  points are slow points (≤ 2% exact), so their tracking is noisy; the overlap is the robust measure.
- Go vs Anti over time (64 trials): Dly Anti flips the angle during the stimulus and then stores the flipped angle
  on Dly Go's memory ring (nearest Go trial at end of delay 174–180° away), in **both** control and C0.
- Control vs C0 at end of delay: active neurons/trial 124 vs 61; ring angle order 0.91 vs 0.65; correct after
  +10 s 0.91 vs 0.78. The unconstrained ring is better but spread over more neurons/dimensions; **don't judge
  rings by eye in 3D**.
- GIFs + scripts + README: `results/front/dynamics_gifs/` (time GIFs C0, Go vs Anti, control vs C0 side by side,
  α GIF, ROC threshold GIF).

### 8.3 Like-to-like wiring without clusters (the strongest result so far)

Replaces the old group wiring analysis (§4, which the user found sloppy and hard to follow).
- Per network: fingerprint = each active neuron's per-period task variance / its max; similarity s_ij = Pearson
  of fingerprints for every ordered pair pre→post; connected = |W_ij|/c_i > 0.01 (the `connection_fraction`
  threshold; softplus weights are never exactly 0). Per type E→E, E→I, I→E, I→I:
  **AUC = P(a connected pair is more similar than an unconnected pair)** = Mann–Whitney U / (n1 n0), via ranks;
  0.5 = no relation, > 0.5 like-to-like, < 0.5 like-to-unlike; independent of connection density (control 67%
  connected, C0 0.7–0.9%). p from 500 shuffles of fingerprints within E and within I (wiring fixed), because
  pairs are not independent. Direction: each neuron's main period type (stimulus/memory/response), E→E
  P(connected) type→type, observed / shuffled.
- Scripts: `fig_wiring_like2like.py` (one seed; network() has `keep_frac` for matched sparsity),
  `fig_wiring_like2like_seeds.py` (all 20 → `wiring_like2like_5seed.png/.csv`), `fig_wiring_matched_sparsity.py`,
  `fig_wiring_roc.py`, **`fig_wiring_roc_all.py` → `wiring_roc_all_s4.png` (4 types × 4 points, 20-bin histograms,
  no KDE since 2026-11-20: it blurred C0's few connected pairs; ROC panel beside each) and `wiring_roc_all_5seed.png` (mean ROC per point + AUC per network)**: the
  figures the user liked.
- Results, mean AUC (sd) over 5 seeds:

| | control | A3 | B0 | C0 |
|---|---|---|---|---|
| E→E | 0.53 (0.01) | 0.52 (0.00) | 0.52 (0.01) | **0.68 (0.02)** |
| E→I | 0.54 (0.00) | 0.54 (0.01) | 0.56 (0.01) | **0.73 (0.02)** |
| I→E | **0.45 (0.01)** | 0.49 (0.01) | 0.51 (0.01) | **0.72 (0.02)** |
| I→I | 0.48 (0.01) | 0.49 (0.01) | 0.56 (0.02) | 0.71 (0.08) |

  Only C0 has strong like-to-like wiring (all seeds). **Inhibition flips:** control I→E < 0.5 in all 5 seeds
  (I targets E doing other computations = competition/switching), C0 I→E 0.72 (I targets E doing the same
  computation = co-tuned/balance). Seed 4 is typical for C0 except I→I (lowest).
- C0 E→E direction chain (obs/shuffled, mean (min–max) over 5 seeds): stimulus→memory 2.05 (1.46–2.47),
  memory→memory 2.69 (2.00–3.49), memory→response 1.53 (0.81–2.23, one seed < 1), response→stimulus 0.32
  (0.26–0.39). Control flat (0.94–1.04). No p-values per cell yet; C0 has few synapses (≈150 E→E, ≈55 I→E, ≈20 I→I).
- **Sparsity caveat:** C0 is also ~9× sparser than B0 (0.7% vs 6% connected). Post-hoc test (keep each network's
  strongest synapses at the same-seed C0 sparsity): B0 E→E stays 0.52 (C0 0.68) → E→E like-to-like is not just
  sparsity; for I-involving types B0 rises to 0.57–0.65, still below C0. Control/A3 strongest synapses are
  like-to-*unlike* (control I→E 0.35, I→I 0.26). Post-hoc ≠ trained under the budget. **Clean test still open:**
  wiring-only grid networks at C0's sparsity exist in `dale_all_grid29_fanin_150k` (rate budget 10, conn budget
  0.0032, ~0.4–0.5% connected, worst task 0.81, 3 seeds) but **only `runs.csv` is in the repo, no weights**:
  pull them from LRZ if they were saved, or retrain those few (~6 h each).
- Theory to cite for "why": Whittington et al. 2023 (ICLR), *Disentanglement with biological constraints*:
  non-negative rates + energy cost → modular single-neuron codes for independent variables (task periods never
  overlap in time). Framing agreed with the user: constraints break the rotation symmetry of the solution.

### 8.4 Open next steps (none requested yet)

- Clean sparsity test (grid29 weights or retrain).
- Lesions on the per-period groups (Driscoll Fig 5), E vs I, across the four points.
- Transfer learning (freeze W_rec, learn one new rule vector) across the four points.
- Fixed-point/pair analysis and per-period clustering for the other points and seeds (only C0 seed 4 done for
  fixed points; GIFs for control and C0).
- p-values for the direction-table cells; the I→E question at the group level is answered by the AUC flip.

### 8.5 "How does it work locally?" (discussed 2026-11-18 night, nothing run)

The user's question: the AUC is a global statistic ("it's there"); they want the mechanism: how exactly the
wiring differs and how the constraints change the computation, i.e. the original compositionality vs clustering
question. What was discussed:
- **Reframe agreed as a hypothesis, not tested:** both control and C0 compose at the level of dynamics (Dly Anti
  stores its flipped angle on Dly Go's memory ring in both, §8.2). The open question is *where* the shared part
  lives: a shared activity subspace spread over many neurons (control?) vs shared neurons and synapses (C0?).
  The constraints may tie the same computation to single units (rotation symmetry broken; Whittington 2023).
- **Ruled out by the user: local wiring statistics in C0.** Plan was Dly Go vs Dly Anti, W_ij vs preferred-angle
  difference per wire class (memory→memory ring, Go-stim→memory at 0°, Anti-stim→memory at 180°,
  memory→response), plus wire-class lesions. The user tried it (`results/front/wiring_tuning_s4.png`): C0 keeps
  too few synapses (~0.8% of pairs; ≈150 E→E, ≈55 I→E, ≈20 I→I) for per-class, per-angle statistics.
- **Proposed, not run (activity / state space, Driscoll-like):** for every pair of the 59 task periods,
  (1) subspace overlap O(A→B) = tr(U_Aᵀ C_B U_A) / tr(U_Bᵀ C_B U_B) with U_A the top-k PCs of period A
  (invariant to rotating the neurons: measures shared computation) and (2) neuron overlap
  s(A,B) = corr(diag C_A, diag C_B) (basis-dependent: measures alignment to units). Same-computation period
  pairs vs different, distributions + ROC/AUC for O and s, control vs C0. Reading: control ≈ C0 on O but only C0
  high on s → same computation, constraints only align it to neurons; C0 > control on O → constraints change the
  computation (more reuse). Optional: varimax rotation of control activity to look for hidden modules.
- **The user said the analysis was "getting out of hand"** and switched to putting the existing results in the
  deck first. Don't push new analyses; wait for the user to pick one.

## 9. Deck polish, 2026-11-20 (Linux, after pulling `fee34bf`)

All in `presentation/` unless stated; deck builds (30 pages, no errors). Details per slide in
`presentation/presentation.md`.

- **Slide order:** the three like-to-like slides moved up to right after "Do they converge?" (before the task
  variance slides) and lost the "experimental" tag.
- **Like-to-like histogram (seed-4 figure):** the Gaussian KDE was dropped. It widened C0's few connected pairs
  (22–146) 3–4× more than the unconnected ones and lost up to 4% of mass past +1, hiding the like-to-like spike near
  0.9. Now plain histograms, **20 bins of width 0.1**, same bins for both, each scaled to area 1 (chosen over 40
  bins and √n bins after comparing all three); ROC moved from an inset (which covered the right end) to its own
  panel. `fig_wiring_roc_all.py --bins {20,40,sqrt}` (20 default). AUC/ROC/medians were always on raw values.
- **Front, top view:** filled contours of worst-task accuracy instead of coloured points: thin-plate-spline fit
  (scipy `RBFInterpolator`, smoothing 1e-3) through the 33 per-budget seed means, only inside their convex hull;
  lines 0.7/0.8/0.9, dashed 0.6. `fig_front_top_fanin150k.py` (`budget_means`, `accuracy_surface`).
- **Accuracy colour rule (user, permanent):** RdYlGn on 0–1, red 0, green 1, every accuracy figure (also the 3D
  front, `fig_front_gif_150k.py`, vmin 0.5 → 0).
- **"Worst-task accuracy" everywhere** in user-visible labels (never "min task accuracy"); internal keys unchanged.
- **Point names on slides:** control / firing constraint / wiring constraint / both constraints. Analysis figures
  of the 4-point run: `POINTS` in `fig_tv_ei_fanin.py`, `fig_acc_vs_steps_grid.py`, `fig_lesion_contrast_grid.py`,
  `HIGHLIGHT_NAMES` in `fig_front_gif_150k.py` and the annotation in `fig_front_top_fanin150k.py` now read
  "Control", "Firing constraint (A3)", "Wiring constraint (B0)", "Both constraints (C0)". Regenerated: front top
  + GIF, `tv_ei_fanin_5seed_s4_150k*` and its crops, `tv_period_driscoll_*_s4`, the lesion contrast grid.
- **Wording:** AI-sounding phrases rewritten in the user's voice (no "X, not Y", no slogan colons, no dramatic
  "…"; "we" and full sentences). 15 changes, slides 4–13, 15 and E1.
- **Talk figures (new):** `presentation/figs/talk_figs.py` → `presentation/figs/talk/` (`frames/front_000–059.png`,
  `front_top.pdf`, `convergence.pdf`, `like2like_s4.pdf`, `like2like_5seed.pdf`, `silhouette.pdf`,
  `tv_control_A3.pdf`, `tv_B0_C0.pdf`). Drawn at slide size (text width 5.6 in), 7–9 pt normal-weight fonts,
  minimal text; slides 10–18 use them. Originals untouched (old frames still in `presentation/frames/`, old crops
  in `presentation/figs/`). User's rule: two kinds of figures, analysis (detailed) and talk (crisp), never delete
  the analysis ones; detail that does not fit goes in `presentation.md`.
- **Lesion slides (experimental, before "Thank you"):** math slide (Ward groups 14 E / 5 I on per-period TV;
  rates of the group set to 0 all trial; Δ = (acc_lesioned − acc_full)/acc_full; Driscoll module contrast
  C = ⟨Δ⟩_module − ⟨Δ⟩_reference, C < 0 = module hurt more) + the existing contrast grid as it is. Mean change per
  lesion control −0.16, firing −0.05, wiring −0.16, both −0.13; |C| ≥ 0.2 in 6/5/9/9 of 19 groups. Seed 4, v1.
- **Checked on request:** A3's wiring cost is really above control's, in all 5 seeds (0.047 vs 0.030 mean,
  1.4–1.7× per seed, same init and trials), though A3 keeps fewer synapses (61% vs 70%): fewer but stronger.
  Guess, not tested: with ~250× less firing it needs larger weights to drive its targets.
- Memory updated: `accuracy-colormap.md` (new), `talk-slide-style.md` (names, wording, two figure types, no bold).

## 10. Yang-battery front and 5 new points on it (2026-11-20 to 11-26)

Everything here is on the fixed battery (`task_version = "yang"`), cog20, 150k, fanin, nothing frozen, g = 1,
batched, eval as before. Trained on the remote cluster by a Claude session there (`CLUSTER_HANDOFF_NSGA.md` is its
instruction sheet); results come back through git.

### 10.1 Runs

| run | what | networks | weights? |
|---|---|---|---|
| `runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k_yang` (`793d75c`) | old 4 points (control/A3/B0/C0, old budgets) x 5 seeds | 20 | yes |
| `runs/moo/..._p48g3_fanin_150k_yang_fullgpu_ms1`, `_ms2` (`26b20e5`) | NSGA-III, NSGA seeds 1, 2, training seed 0, rate budget 0.002-0.15, wiring 0.002-0.03 | 144 + 144 | no |
| `runs/moo/..._p48g3_fanin_150k_yang_fullgpu_ext_ms3`, `_ext_ms4` (`b427bed`) | same, **extended box**: rate 0.001-0.5, wiring 0.002-0.06, NSGA seeds 3, 4 | 142 + 143 | no |
| `runs/moo_zoom/dale_all_woutfree_fanin_front4_5seed_150k_yang` (`89a8ddd`) | **the new points**: control + loose / firing / wiring / both, 5 seeds each | 25 | yes |

`results/scripts/fig_front_yang_ms12.py [ext]` pools ms1+ms2 -> `..._ms12/` or ext_ms3+ext_ms4 -> `..._ext_ms34/`
(runs.csv, front.csv, `front_all.csv` = front with no feasibility cut, front_top.png). **No committed script pools
all four searches yet**; the numbers below are from scratch analyses (pareto_mask over the 4 runs.csv, feasible =
worst-task accuracy >= 0.6).

### 10.2 The front (all four searches pooled)

- 573 networks, 532 feasible, 116 on the front; each search contributes 27-32, so the searches agree. Later
  generations never beat generation 0 (best ~0.99 from the start): most of the box is at ceiling.
- **Wiring sets accuracy, firing barely does:** a plateau at 0.93-0.99 with a cliff at wiring ~0.003
  (log -2.5). Spearman over feasible (first box): accuracy vs log wiring 0.63, vs log firing 0.14.
- **At the lowest wiring, firing buys accuracy:** at wiring 0.0023-0.0034, networks firing 0.38-0.50 still reach
  0.62-0.92, where low-firing ones fail. The feasibility edge bends down at high firing (only visible with the
  extended box).
- dmsnogo is the worst task in about half the networks, then contextdelaydm1/2 and dmsgo.
- The front is "thick": accuracy differences along the plateau (~0.01-0.02) are seed-noise size, so single-network
  Pareto membership is noisy; the cliff is the robust feature. Reached firing is ~1.5x the rate budget (IQR
  1.36-1.65); wiring lands on its budget (projection).
- **Old points vs this front:** control and B0 are dominated in all seeds (B0 fires 0.75-0.89; a front network
  reaches 0.958 at firing 0.27, wiring 0.0051). A3 seeds 1, 2 and two C0 seeds are not dominated. That is why
  new points were chosen on the front (user's narrative: map the front, then pick points on it).

### 10.3 The 5 new points (picked 2026-11-25 on a smoothed surface of the pooled front)

Budgets were read off a thin-plate fit of worst-task accuracy over (log rate budget, log wiring budget), smoothing
1.0: firing / wiring / both near the ~0.93 contour (equal competence, different binding cost), loose = best part
of the front, control = no budgets. Result (5 seeds each, all feasible):

| point | rate, wiring budget | worst-task acc mean (min-max) | firing | wiring | synapses kept | gap to front |
|---|---|---|---|---|---|---|
| control | none | 0.946 (0.894-0.984) | 0.47 | 0.029 | 70% | dominated (all seeds) |
| loose | 0.015, 0.028 | 0.979 (0.969-0.992) | 0.023 | 0.028 | 50% | 0.008 |
| firing | 0.0013, 0.05 | 0.921 (0.886-0.955) | 0.0023 | 0.046 | 61% | 0.042 |
| wiring | 0.18, 0.0043 | 0.944 (0.923-0.953) | 0.23 | 0.0043 | 2.6% | 0.010 |
| both | 0.010, 0.0057 | 0.925 (0.897-0.944) | 0.017 | 0.0057 | 1.7% | 0.004 |

- Gap to front = best accuracy of any NSGA network with no higher firing and wiring than the point's 5-seed mean,
  minus that mean. Seed sd: loose 0.010, wiring 0.012, both 0.019, firing 0.029. So loose, wiring and both are on
  the front within seed noise (their means are dominated by 1-8 of 532 networks, and some single seeds by none).
  Firing is 1.4 sd below; its region is sparsely sampled and the best network there (ext_ms3, budget 0.0013/0.053,
  0.963) is one lucky draw. Selection bias of ~0.01-0.02 was expected.
- Control's mean (0.946) is lower than in the old Yang 4-point run (0.976) because of seed 4 (0.894); same
  firing/wiring. Different batches = different trial streams, so seeds are not the same networks across runs.
- Worst tasks: dmsnogo in most constrained networks; contextdelaydm2 in control.
- Point names in the run folder: `control`, `loose`, `firing`, `wiring`, `both` (not A3/B0/C0).

### 10.4 To do

- **Rerun the analyses on the new run** (like-to-like, silhouette, per-period TV, lesions, E/I modularity).
  **Like-to-like and silhouette done (§11)**; per-period TV heatmaps, lesions and E/I modularity (group wiring) not yet.
- A committed script (or an `all` set in `fig_front_yang_ms12.py`) that pools all four searches and draws the new
  points on the front; then the front slides (top view, 3D GIF/frames) move to the Yang front.
- Decide whether `loose` or `control` is the reference on slides (control is off the front by construction).
- Deck numbers and the E/I report are all still v1 (§0).

## 11. Like-to-like and silhouette on the new front points (2026-11-26, Windows)

Run `runs/moo_zoom/dale_all_woutfree_fanin_front4_5seed_150k_yang` (§10.3), key `"front"` in
`fig_tv_ei_fanin.RUNS`. Points analysed: control, firing, wiring, both (5 seeds each). **`loose` not analysed yet.**
The run has **final checkpoints only** (no `step_*` folders, `save_at: []`); the only save-at run is the old v1
1-seed one.

### 11.1 Cached analysis scripts (built 2026-11-20, extended today)

Pattern: compute -> cache per network -> plot from cache. `--run R` (a `fig_tv_ei_fanin.RUNS` key), `--plot-only`
(figures from cache, ~15 s, mostly torch import), `--recompute`; otherwise only missing networks are computed.

- `results/scripts/like2like.py`: like-to-like AUC (as `fig_wiring_like2like.network`, unchanged), two modes per
  network: `trained` (|W|/c > 0.01) and `matched` (strongest synapses, as many as the same-seed reference network:
  `C0` in old runs, `both` in the front run; `REF` is set from the points). rng seeded per network. Cache
  `results/cache/<R>/like2like_<point>_s<seed>.npz` + `like2like.csv`. Figures `presentation/figs/talk/
  like2like_example_s<seed>_<R>.pdf` (slide 15 style), `like2like_seeds_<R>.pdf` (slide 16), `like2like_matched_<R>.pdf`.
  **Checked:** `--run 5seed` reproduces the old v1 numbers exactly (all AUCs, connected counts, p < 0.05 calls).
- `results/scripts/silhouette_seeds.py`: silhouette vs k for every seed, E and I, `--basis task|period`, k-means via
  `fig_tv_ei_fanin.cluster_pop` (unchanged). Cache `results/cache/<R>/silhouette_<basis>.csv`. Figures: talk
  `silhouette_seeds_<basis>_<R>.pdf` (mean + seed band, dots = chosen k) and **`results/front/silhouette_k_<basis>_<R>.png`**
  (4 rows = points x 2 columns = E, I; light = seeds, dark = mean, dots = each seed's chosen k, dashed = k the same
  rule picks on the mean curve, with the reason and the other peaks in the title). **I now goes to k = 20**
  (`K_MAX`, was 10); only the front per-task cache is recomputed with it; other caches still have I <= 10 until
  `--recompute`. `fig_tv_ei_fanin.py` (slide-17 heatmaps) still uses I <= 10.
  **Checked:** v1 seed 4 gives the slide-17 k (E 4/16/4/13, I 3/9/5/5).
- `results/scripts/talk_style.py`: shared talk style (moved out of `talk_figs.py`); `LABEL` (old and new point
  names), `RUN_POINTS` (`front` -> control, firing, wiring, both), `run_points(run)`. `talk_figs.like2like` removed.
- `fig_tv_period.load(point, seed, run="5seed")`: 5seed caches stay in `results/front/`, other runs in
  `results/cache/<run>/tv_period_*.npz`.

### 11.2 Like-to-like on the new points (5 seeds, mean AUC)

| | control | firing | wiring | both |
|---|---|---|---|---|
| E→E | 0.53 | 0.52 | 0.55 | 0.62 |
| E→I | 0.54 | 0.54 | 0.61 | 0.65 |
| I→E | 0.45 | 0.49 | 0.57 | 0.60 |
| I→I | 0.46 | 0.48 | 0.61 | 0.61 |
| synapses kept | 70% | 61% | 2.7% | 1.8% |

- both: p < 0.05 in 5/5 seeds, every type, but weaker than old C0 (0.68-0.73). Wiring is now also clearly > 0.5
  (old B0: 0.51-0.56).
- Control I→E < 0.5 in 5/5 seeds (inhibition to other-computation neurons): holds.
- Matched sparsity (cut to both's 1.6-1.9%): control and firing stay <= 0.50; **wiring reaches 0.57-0.63, about
  equal to both** except E→E (0.57 vs 0.62).
- **Across the 20 wiring-budget networks (yang B0, C0 and front wiring, both) AUC follows sparsity:** Spearman(AUC,
  fraction kept) -0.89 to -0.97 per type; E→E: B0 (6.4%) 0.51, wiring (2.7%) 0.55, both (1.8%) 0.62, C0 (1.0%) 0.65.
- **Claim status:** "only both constraints wire like-to-like" is no longer supported: the old C0 vs B0 contrast
  confounded the firing budget with sparsity (0.8% vs 6.4%). What holds: like-to-like needs sparsity *reached in
  training under a wiring budget*; post-hoc pruning of dense networks does not produce it.
- Representative seed for the example figure: **seed 4** (largest both minus mean of the other three, 0.103; seed 1
  is marginally ahead against control + firing only).
- Slide-16 caption says "0.7% of synapses"; over active units the v1 C0 networks keep 0.84-1.01%.

### 11.3 Silhouette and k (decision: **per-task variance** is the profile to keep)

- Per task = 20 values per neuron (whole post-fixation trial per task, Yang 2019). Per period = 59 values (each
  non-fixation epoch of each task: RT tasks 1 row, Go/Anti/DM 2, Dly Go/Anti/DMS/DNMS/DMC/DNMC 3, delayed DM 5;
  Driscoll 2024). A task's per-task value is the length-weighted mean of its period rows.
- Per period gives **fewer** E groups (medians drop ~4-8) and lower, flatter silhouettes, because the extra rows are
  mostly copies (effective dimension 6.3 -> 7.7 for control, seed 4), "which task" explains 44-59% of a neuron's
  period profile and "which period type" only 14-21% (neurons are task, not period, specialists), and short-epoch
  variances are noisier. Per-task and per-period clusters agree poorly (ARI 0.16-0.49). Per period stays only for the
  Driscoll comparisons (heatmaps, lesions). Note: like-to-like similarity uses per-period profiles, slide 17 per task.
- k rule (unchanged): the highest local peak (higher than both neighbours) with k > 2; no peak -> overall maximum.
  The last k tested can't be a peak.
- Per task, front run, k on the mean curve (seeds' k): E control 9 (10,10,9,14,12), firing 11 (17,19,18,14,11),
  wiring 10 (9,11,11,9,10), both 18 (18,13,7,12,14); I (k <= 20) control 3 (3,3,6,5,3), firing 4 (4,7,4,5,6),
  wiring 14 (15,19,16,10,4), both 11 (14,10,11,13,11).
- **The curves are nearly flat past k ~ 6**; the winning peak usually beats the next by < 0.01, so k is a soft
  choice. Clear cases: E wiring, E control, I control, and I both (clear peak at 11, only visible with k up to 20).
  I wiring is flat (0.17-0.19), its k means little.
- **Robust result = the height of the curve:** firing highest (E 0.27, I 0.30 at chosen k), then both, control and
  wiring lowest, at almost every k. The old claim (highest under a firing budget, especially I; firing-constrained
  networks have more E groups) holds, with control now picking more E groups (median 10) than on the old points.

### 11.4 To do (from §10.4, still open)

- Per-period TV heatmaps, lesions and E/I modularity (group wiring) on the front points; `loose` not analysed.
- ~~Deck still shows v1 figures and numbers~~: done in §12 (slides 1–18 use the front run; the experimental
  slides after the divider are still v1).

## 12. Deck revision after supervisor feedback, and an HTML version (2026-11-26, Windows)

Commits `64ecc14`, `424492b` (+ the HTML deck commit). Beamer deck: `presentation/main.tex` (29 slides). HTML deck:
`presentation/html/` (same slides, different look). **Keep both in sync by hand** (no shared source).

### 12.1 Supervisor feedback, as applied

- **Nomenclature (no ML terms):** rate cost / synaptic cost; points **Control, Rate constraint, Synaptic
  constraint, Both constraints** (never short names); "like-to-like connectivity" (not "wiring"); title "Multitask
  Dale's-law RNNs under rate and synaptic budgets". `talk_style.LABEL` carries the names for all talk figures.
  "Firing-rate homeostasis" and quoted paper captions keep their words.
- **Every content slide ends with a take-away** (`\takeaway{...}` in main.tex, `.take` in the HTML).
- **No bold in figure labels.** Bigger fonts: a global pass is still to do.
- Slide-by-slide (PDF page numbers):
  - 3 "256 RNN units" (no "vanilla");
  - 5 Fakhar & Astle 2026 Fig. 1 (Pareto trade-off landscape; vector crop `figs/fakhar2026_crops.tex`,
    `papers/Fakhar2026.pdf`);
  - 6 Figma `Fig_setup` recoloured (E blue, I red; bars "Synaptic cost" / "Rate cost"), re-exported;
  - 9 synaptic limit = subtractive normalisation (Miller & MacKay 1994; one budget for the whole network); rate
    limit = integral control of firing-rate homeostasis (Turrigiano 2008; O'Leary et al. 2014); the ρ/2 term is the
    augmented-Lagrangian push-back in `BudgetConstraint` (ρ = 1, acts only above budget); networks end ~1.5× above
    their rate budget (median 1.51, IQR 1.36–1.65 over the 4 searches);
  - 10 NSGA-III animation from the real ms1 search (`results/scripts/anim_nsga3.py` → `figs/talk/frames_nsga/`,
    156 frames): initialise → train → check ≥ 0.6 → rank → breed, generations 0 and 1;
  - 11 rotating 3D front of the 4 pooled searches (`results/scripts/front_yang.py frames` → `figs/talk/frames_yang/`);
  - 12 top view with the 4 points + per-point summary (accuracy dots red→green, log cost bars, % synapses kept);
  - 13 convergence on the front points (`talk_figs.py convergence_front` → `convergence_front.pdf`), from the save-at
    run `runs/moo_zoom/dale_all_woutfree_fanin_front4_1seed_150k_yang_saveat` (seed 0, same 5 budgets as the 5-seed
    run, batched; checkpoints at 1k, 5k, 10k, 25k, 50k, 75k, 100k, 125k + final 150k). Seed 0 final accuracies
    match the 5-seed run's seed 0 within GPU noise (not bit-identical). All four reach every task ≥ 0.6 by 100k
    (control 25k, synaptic 50k, both 50k, rate 100k); with the rate constraint DNMS (dmsnogo) sits at 0.5 from 25k
    to 75k. Mean accuracy is flat by ~75k; worst task still rising at 150k for rate (0.90 → 0.96) and both
    (0.85 → 0.95). The loose point is in the run but not on the slide;
  - 14 like-to-like schematic from real neurons (`results/scripts/fig_like2like_schematic.py`: connected pair
    r = +0.76 (90th percentile, not the extreme), unconnected pair r = −0.10, histogram of all E→E pairs, ROC with
    AUC 0.63; both constraints, seed 0) + definitions;
  - 15–16 like-to-like on **per-task** profiles (`like2like.py --basis task`), example seed 0 (strongest contrast);
    axis labels, significance legend, point names on the AUC x axis;
  - 17 silhouette vs k from 3 to 20 (E and I), chosen k as dashed lines (highest peak of the 5-seed mean);
  - 18 one 2×2 figure of the 4 points' group heatmaps, seed 0 (most typical of the 5 by chosen k and silhouette;
    `talk_figs.py tv_front`); conclusion: rate-budget E neurons take part in 3.1 tasks each vs 5.0 (control),
    synaptic 4.7, both 3.4 (every seed);
  - the "Experimental" divider follows slide 18; the 4 group-synapse slides and the Driscoll/lesion slides (old
    points, v1) sit behind it.

### 12.2 New or changed scripts

| script | what |
|---|---|
| `results/scripts/front_yang.py [frames top summary]` | pooled Yang front (ms1, ms2, ext_ms3, ext_ms4; 573 / 532 / 116): 3D frames, `front_top_front.pdf`, `points_summary_front.pdf` |
| `results/scripts/anim_nsga3.py [--dpi]` | NSGA-III animation frames (slide 10) |
| `results/scripts/fig_like2like_schematic.py` | slide-14 schematic |
| `results/scripts/fig_nsga3_scheme.py` | static 4-panel NSGA-III draft (not in the deck) |
| `results/scripts/like2like.py --basis task\|period` | per-task similarity (caches `like2like_task_*`, `like2like_task.csv`); period caches keep their names |
| `results/scripts/silhouette_seeds.py` | I now k = 2–20; `plot_panels` → `results/front/silhouette_k_<basis>_<run>.png` (4 points × E/I) |
| `results/scripts/fig_tv_ei_fanin.py` | `analyse(ckpt, k_max_I=10)`; `RUNS["front"]` |
| `presentation/figs/talk_figs.py tv_front` | slide-18 figure |

### 12.3 HTML deck (`presentation/html/`)

- Quarto + reveal.js; one self-contained `slides.html` (~40 MB; MathJax, plotly, figures and GIFs inlined), works
  offline in any browser (F full screen, S notes, O overview). Light minimal style (`style.css`).
- **Faithful to main.pdf** (user's rule): same slide order, titles, text, equations, figures, captions, take-aways
  and nomenclature. Extras: a Summary slide before the Experimental divider (numbers from slides 11, 16, 18) and two
  optional appendix slides tagged "extra" (interactive plotly 3D front, like-to-like AUC table).
- Build (from the repo root, cmc_env; plotly was pip-installed into cmc_env, not in requirements.txt):
  `python presentation/html/build_assets.py` (PDF → SVG, paper crops → PNG because pdftocairo crashes on them,
  frames → GIFs), `python presentation/html/build_figs.py` (plotly extras, `assets/plotly.min.js`),
  `quarto render presentation/html/slides.qmd`.
- Gotchas found: `#` or `###` markdown headings inside a slide make reveal.js split slides (use HTML `<h1>`/`<h3>`,
  `slide-level: 2`); plotly in iframes fails in the one-file deck (Chrome refuses data-URL frames > 2 MB), so plots
  are inline divs resized on `slidechanged` (`_plotly.html`); `::: {}` makes no wrapper div (name it); the browser
  pane caches `slides.html` (add `?v=N`). Local preview: `.claude/launch.json` "html-deck" (port 8765).

### 12.4 Still open

- Global font/label-size pass on the talk figures (supervisor).
- ~~Slide 13: new intermediate checkpoints of the 4 points~~ done (one seed, see 12.1).
- Slide 10/12 figures were drafts for discussion (NSGA-III animation, points summary).
- Experimental slides: still old points, v1 battery.
- §11.4: per-period TV, lesions, E/I group wiring on the front points; `loose` not analysed.
