# Session handoff (Windows → Linux), 2026-11-18

Read this first in a new Claude Code session, together with `CLAUDE.md`. Everything referenced here is in
git; nothing lives only on the Windows machine except the auto-memory, which is copied in full below.

## 1. Working rules (from the Windows auto-memory, apply them here too)

- **Answer only what is asked.** No unrequested suggestions, next steps, offers or commands. If something
  looks urgent, state the fact in one line without proposing an action.
- **Python env:** everything runs in the `cmc_env` conda env (`conda run -n cmc_env ...` or activate it),
  never system Python. Setup: see README (`conda create -n cmc_env python=3.12`,
  `pip install -r requirements.txt`, `pip install -e .`).
- **g stays 1.0** (init gain). The g = 0.3 runs were exploratory (low-rank-RNN regime, out of scope).
- **Scope:** E and I only, no interneuron subtypes, no canonical microcircuit. Spatial wiring
  (Khona 2023) and later steps are deferred.
- **Baseline** = the Dale network with no budget (`control`). Non-Dale `LeakyRNN` is out of scope.
- **Task names in figures:** Yang 2019 display names (`cmc.task.rule_name`): Go, RT Go, Dly Go, Anti,
  RT Anti, Dly Anti, DM 1, DM 2, Ctx DM 1, Ctx DM 2, MultSen DM, Dly DM 1, Dly DM 2, Ctx Dly DM 1,
  Ctx Dly DM 2, MultSen Dly DM, DMS, DNMS, DMC, DNMC. Internal names only as data keys.
- **gh CLI is not available** on the Windows machine: pushing works, PRs are opened by hand.
- Supervisor focus (since 2026-10-07): task-variance clusters along the front (20 tasks), and E→E, E→I,
  I→E, I→I weight distributions under the constraints.

## 2. Where the project is

Current direction: **how E and I organise into task modules** under metabolic (rate) and wiring budgets,
cog20 (all 20 Yang tasks), 150k training steps, W_in and W_out trained, W_out `fanin` init, g = 1.

Main result (5 seeds, `runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k`, write-up in
`reports/ei_modularity/ei_modularity.pdf`, notebook `notebooks/ei_modularity_150k.ipynb`):

| point | rate budget | wiring budget | closest type |
|---|---|---|---|
| control | none | none | 3: inhibition between modules (I→E ρ ≈ −0.44) |
| A3 | 0.0011 | none | 1: sharp E tuning, wiring ignores tasks |
| B0 | none | 0.0054 | weak 2 |
| C0 | 0.0115 | 0.0044 | 2: E+I modules (E→I ρ ≈ +0.46, E→E within/between ≈ 5) |

ρ = Spearman(task similarity of two groups, z-scored wiring between them); E and I clustered separately.

## 3. Runs that matter now (all cog20, 150k, nothing frozen)

| run | init | points × seeds | per-step accuracy? | checkpoints in git? |
|---|---|---|---|---|
| `runs/moo/dale_all_grid29_fanin_150k_s{0,1,2}_{A,B}` | fanin | 29 × 3 (budget grid, no control) | no | no |
| `runs/moo_zoom/dale_all_woutfree_fanin_4pt_5seed_150k` | fanin | 4 × 5 | no | yes (final) |
| `runs/moo_zoom/dale_all_woutfree_fanin_4pt_1seed_150k_saveat` | fanin | 4 × 1 | yes (1k…150k) | yes (all steps) |
| `runs/moo_zoom/dale_all_rays3_5seed_150k` | small | 16 × 5 | yes | no |
| `runs/moo_zoom/dale_all_curves5_4seed_150k` | small | 5 × 4 | yes | no |

- The grid is the 30-point proposal (`results/front/grid_proposal_150k.csv`) minus control (`cmc.moo
  --points` cannot train unbudgeted); control comes from the 4pt fanin runs.
- Combined fanin front: 111 networks, 89 feasible, 38 Pareto. Wiring −2.75 (log10) is infeasible everywhere.
- Fanin vs small at the 4 shared points: costs identical, fanin worst-task accuracy +0.01 (control) to
  +0.05–0.08 (A3/B0/C0), seed spreads overlap. Batched runs with different trial streams.
- Save-at curves (1 seed): control worst task flat ~0.96 from 75k; **A3 still rising at 150k**
  (0.50 at 100k → 0.73); B0 rising to 0.91; C0 flat ~0.87 from 100k.

## 4. Scripts made in the last session (run from repo root, in cmc_env)

| script | output (in `results/front/`) |
|---|---|
| `results/scripts/fig_front_top_fanin150k.py` | `cog20_150k_fanin_front_top.png`, `cog20_150k_fanin_front.gif`, `front_networks_150k_fanin.csv` |
| `results/scripts/fig_front_gif_150k.py` | 150k front pooled over the small+fanin zoom runs (older) |
| `results/scripts/fig_grid_proposal_150k.py` | `cog20_150k_grid_proposal.png`, `grid_proposal_150k.csv` |
| `results/scripts/fig_acc_vs_steps.py <point> [rays3\|fanin]` | `acc_vs_steps_<point>[_fanin].png` |
| `results/scripts/fig_tv_ei_fanin.py [step]` | `tv_ei_fanin_150k.png` (TV heatmaps, E/I separately), `_sil.png` (silhouette vs k, chosen k dashed), `_wiring.png` (group×group similarity + z-scored wiring, ρ and within/between per quadrant) |

`fig_tv_ei_fanin.py` copies `cluster_pop` / `tv_panel` from `reports/ei_modularity/make_figures.py` and the
wiring computation from `notebooks/ei_groups_figures.ipynb` (Figure 2). It is **one seed** (save-at seed 0);
the 5-seed numbers are in the report. Clusters differ per seed, so only per-network scalars (ρ, within/between,
tasks per neuron, silhouette) can be averaged across seeds.

## 5. Docs

- `reports/theory.md`: model, costs, multi-objective theory, connectivity analysis, literature on E/I motifs.
- `reports/PRESENTATION.md`: talk outline, updated to the E/I modularity results; `[to be removed]` marks
  slides that no longer fit, `[UPDATE]` marks missing numbers/figures.
- `reports/ABSTRACT.md`: COSYNE 2027 notes (deadline has passed; kept for reference).
- `reports/MOO_BUDGETS.md`: NSGA-III, cost definitions, budget enforcement, open issues.

## 6. Beamer test deck (`presentation/`)

`main.tex`: title / rotating 150k front (`animate` package) / thank-you. Build:

```bash
python presentation/make_frames.py
cd presentation && pdflatex main.tex && pdflatex main.tex
```

Needs the LaTeX packages `animate`, `media9` (provides `pdfbase.sty`), `ocgx2`, `zref`: with TeX Live
`tlmgr install animate media9 ocgx2 zref`; on Debian/Ubuntu they are in `texlive-latex-extra`. The
animation plays in Okular or Acrobat only (first frame elsewhere).

## 7. Open questions at handoff

- Whether to run more fanin save-at seeds (A3 not converged at 150k with 1 seed).
- A seed-independent version of the group wiring matrix (bin neurons by preferred task or task family
  instead of per-network clusters) if a 5-seed averaged picture is wanted.
