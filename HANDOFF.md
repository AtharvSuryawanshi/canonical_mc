# Session handoff, updated 2026-11-18 (night, Linux)

Read this first in a new Claude Code session, together with `CLAUDE.md`. **Section 8 is the newest**: the
Driscoll-style analysis (task variance per period, fixed points) and the clustering-free like-to-like wiring test,
which is currently the strongest result. Both are now in the deck as a draft **Experimental** section (§7).
§8.5 records the open "how does it work locally?" question and what was ruled out. Section 7 is the talk in
`presentation/`. Sections 1–6 are the project state.

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
  run (`runs/moo/dale_all_nsga3_budget_p48g3_50k`, pop 48 × 3 gen) found. **A real cog20 fanin 150k NSGA-III run
  is running on the cluster**; when it lands, it replaces the front slide. Same command without `--points`:
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
| `results/scripts/fig_front_top_fanin150k.py` | `cog20_150k_fanin_front_top.png` (rings the example network, `EXAMPLE = (run, seed)`), `cog20_150k_fanin_front.gif`, `front_networks_150k_fanin.csv` |
| `results/scripts/fig_tv_ei_fanin.py [step] [--run saveat\|5seed] [--seed N]` | `tv_ei_fanin_<tag>.png`, `_sil.png`, `_wiring.png`; default tag `150k` (save-at, seed 0); `--run 5seed --seed 4` → `tv_ei_fanin_5seed_s4_150k*` |
| `results/scripts/fig_acc_vs_steps_grid.py` (new) | `acc_vs_steps_fanin_grid.pdf`: 2 × 2 learning curves, one shared legend (save-at run) |
| `results/scripts/fig_acc_vs_steps.py <point> [rays3\|fanin]` | `acc_vs_steps_<point>[_fanin].png` |
| `results/scripts/fig_grid_proposal_150k.py` | `cog20_150k_grid_proposal.png`, `grid_proposal_150k.csv` |
| `presentation/make_frames.py` | `presentation/frames/front_000…059.png` (rotating fanin-only front for `animate`) |
| `presentation/figs/crop_tv_rows.py [tag]` | `presentation/figs/tv_ei_<point>.png`, `wiring_<point>.png` (default tag `5seed_s4_150k`) |
| `presentation/figs/{yang2019,song2016}_crops.tex` | vector crops of paper figures (`pdflatex` in `presentation/figs/`; PDFs in `papers/`) |

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

Current deck (28 pages), outline numbering:

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
| 13 | Task variance and the number of groups (TV + silhouette math) | seed-4 `_sil.png` |
| 14–15 | E and I groups (2 points per slide) | seed-4 crops |
| 16–19 | Task similarity and wiring, one per point | seed-4 crops, captions with 5-seed ranges |
| – | **Experimental** divider (draft section, user decides later what stays) | Beamer |
| E1 | Task variance per task period (math) + Driscoll Fig 3a crop | `figs/driscoll2024_crops.pdf` |
| E2–E3 | Task periods, Driscoll style: control/A3, B0/C0 | `tv_period_driscoll_<p>_s4.png` |
| E4 | Like-to-like wiring (math: s_ij, connected, AUC, ROC, shuffle p) | Beamer |
| E5 | Like-to-like, seed 4 | `wiring_roc_all_s4.png` |
| E6 | Like-to-like, all 20 networks + sparsity caveat | `wiring_roc_all_5seed.png` |
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
6. Figma `Fig_setup` says "Connectivity cost (L1)"; slides say "wiring". Unify.
7. Commits: HEAD `cf34f4a AUROC for correlation vs wiring` has the wiring scripts, their outputs, the
   `tv_period_*` caches and `results/front/dynamics_gifs/`. **Uncommitted** (2026-11-18 night): the Experimental
   slides (`presentation/main.tex`, `main.pdf`, `presentation.md`), the Driscoll crop
   (`presentation/figs/driscoll2024_crops.{tex,pdf}`), `results/front/tv_period_driscoll_{A3,B0}_s4.png`, this
   file, `papers/driscoll2024.pdf` (still untracked) and `results/front/wiring_tuning_s4.png` (the user's own try
   at wiring vs tuning, no script in the repo). LaTeX aux files and `frames/` PNGs could go in `.gitignore`.
8. §8.1 and §8.3 are in the deck as a draft **Experimental** section (E1–E6, 2026-11-18); the user decides
   later what stays. Not in yet: fixed points (§8.2), the E→E direction chain; the old group-wiring slides 16–19
   and the new AUC slides overlap (pick one). Known cosmetic issues: the two portrait Fig 3a figures per slide
   are small (crop the suptitles/legends if they stay); the grey "experimental" tag sits mid-title, not
   right-aligned.

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
  `fig_wiring_roc.py`, **`fig_wiring_roc_all.py` → `wiring_roc_all_s4.png` (4 types × 4 points, KDE
  distributions + ROC inset) and `wiring_roc_all_5seed.png` (mean ROC per point + AUC per network)**: the
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
