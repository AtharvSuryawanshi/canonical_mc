# Session handoff, updated 2026-11-18 (evening, Linux)

Read this first in a new Claude Code session, together with `CLAUDE.md`. Section 7 is the newest: the talk
being built in `presentation/`. Sections 1–6 are the project state; they were written on Windows in the morning
and corrected here where they changed.

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

Current deck (21 pages), outline numbering:

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
7. **Nothing from today is committed.** New/changed: `presentation/` (main.tex, presentation.md, make_frames.py,
   figs/, frames/), `results/scripts/fig_acc_vs_steps_grid.py`, `fig_tv_ei_fanin.py`, `fig_front_top_fanin150k.py`,
   new files in `results/front/`, `papers/`. LaTeX aux files and `frames/` PNGs could go in `.gitignore`.
