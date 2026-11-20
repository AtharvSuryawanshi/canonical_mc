# Handoff: cog20 NSGA-III budget front, fixed task battery, 150k steps

For the Claude instance running this on a new SLURM cluster. Follow the steps in order, and report back at
the marked checkpoints (**REPORT**).

## 0. What this run is

- **Search:** an NSGA-III search (pymoo) over two cost budgets: a rate budget and a wiring budget.
  - Every candidate is one full training of a 256-unit Dale RNN.
  - Objectives: worst-task error, log10 metabolic cost, log10 wiring cost.
  - Constraint: worst-task accuracy >= 0.6.
- **Fixed for this run** (do not change):
  - **20 tasks** (`--task-battery all`);
  - **150,000 steps**;
  - **W_out fanin init, nothing frozen** (`--w-out-init fanin`, no `--freeze`);
  - **g = 1.0**;
  - **network seed 0**, one search;
  - **budget range**: rate 0.002–0.15, wiring 0.002–0.03;
  - **3 generations**.
- **Task battery:** the fixed generators, which match Yang et al. 2019 (`task_version = "yang"`, the default
  since 2026-11-19).
  - The code must be `main` at commit `793d75c` or later.
  - Step 3 checks that the run summary says `"task_version": "yang"`.
- **Population:** as large as the cluster allows (§2). Generations run one after another, but all networks within
  a generation train in parallel. A larger population therefore costs GPUs, not wall time.

## 1. Setup

```bash
git clone https://github.com/AtharvSuryawanshi/canonical_mc.git && cd canonical_mc
git log --oneline -1                      # must be 793d75c or later
conda create -n cmc_env python=3.12 -y && conda activate cmc_env
pip install torch --index-url https://download.pytorch.org/whl/<cuXXX>   # CUDA build matching the cluster's driver
pip install -r requirements.txt           # includes pymoo; keeps the torch installed above
pip install -e .
python -c "import cmc, torch, pymoo; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/test_tasks.py tests/test_batched.py -q     # all must pass
```

Find the hardware with `sinfo -o "%P %G %l %D"` and the cluster docs. Note:
- the GPU partition, the GPU type, the number of GPUs per node, the maximum job time, and any QoS or account flag;
- that `cmc.moo` runs as one process on **one node**, so the most GPUs a run can use is the number of GPUs on one node.

**REPORT:** the partition, the GPU type, GPUs per node, and the time limit.

## 2. Job script and benchmark

Copy `slurm/moo.sjob` to `slurm/moo_cluster.sjob`. Change only the header:
- `--partition`, and `--qos` / `--account` if the cluster needs them;
- `--gres=gpu:<G>`, where G is the number of GPUs used, all on one node;
- `--cpus-per-task`: 4 × the number of workers;
- `--mem`: about 8 GB × the number of workers;
- `--time`: the cluster's maximum;
- `CONDA_ROOT`, if conda is not in `~/miniconda3`.

Leave the body as it is: `conda activate`, `python -m cmc.moo --device cuda "$@"`.

**Benchmark** on one GPU: one generation, 2000 steps, one batch. Use a scratch output directory and do not commit
it.

```bash
sbatch --gres=gpu:1 slurm/moo_cluster.sjob --task-battery all --steps 2000 --w-out-init fanin --g 1.0 --seed 0 \
  --pop-size 24 --n-gen 1 --n-partitions 5 --batched --batch-pop 24 --workers 1 \
  --output-dir runs/moo/_bench_bp24
```

Repeat with `--pop-size 48 --batch-pop 48` (and `--n-partitions 8`) to `runs/moo/_bench_bp48`. From the log
(`peak mem`) and `runs.csv` (`train_time_s`), take:
- the hours per batch at 150k, which is `train_time_s × 75 / 3600`;
- the peak GPU memory.

Choose `--batch-pop`: the largest value that fits in memory and doesn't take much longer per batch.
- For reference, LRZ ran 15 networks at 150k in about 6 h on 3/7 of an A100.

Choose `--workers`: the number of batches that run at once.
- Use 1 per GPU. Use 2 per GPU only if a benchmark with 2 workers on 1 GPU shows the GPU was not saturated.
- Set `--devices cuda:0,cuda:1,...` to match.

Then: `pop_size = batch_pop × workers`, so each generation is exactly one round of batches. Set `--n-partitions`
from this table; it needs C(p+2, 2) ≤ pop_size:

| pop_size | --n-partitions | reference directions |
|---|---|---|
| 48 | 8 | 45 |
| 96 | 12 | 91 |
| 144 | 15 | 136 |
| 192 | 18 | 190 |
| 240 | 20 | 231 |

The previous cog20 search used a population of 48. Use more if the hardware allows.

Delete `runs/moo/_bench_*` after the benchmark.

**REPORT:** hours per batch at 150k, peak memory, and the chosen batch_pop / workers / pop_size, with the expected
total time (≈ 3 × hours per batch).

## 3. The run

Wait for the user's go after the report in §2. Then submit, filling in the numbers from §2:

```bash
RUN=runs/moo/dale_all_nsga3_budget_p<POP>g3_150k_fanin_yang
sbatch slurm/moo_cluster.sjob --task-battery all --steps 150000 --w-out-init fanin --g 1.0 --seed 0 \
  --pop-size <POP> --n-gen 3 --n-partitions <P> \
  --rate-budget-min 0.002 --rate-budget-max 0.15 --conn-budget-min 0.002 --conn-budget-max 0.03 \
  --batched --batch-pop <BP> --workers <W> --devices <cuda:0,...> \
  --output-dir $RUN
```

**Check right away:**
- `$RUN/summary.json` has `"task_version": "yang"`.
- The `args` in it have `w_out_init: fanin`, `freeze: ""`, `steps: 150000` and `g: 1.0`.

**If the job hits its time limit:** resubmit the **identical** command with the same `--output-dir`.
- Finished networks are read back from `runs.csv`, and the search replays them deterministically.
- A batch that was still training when the job was killed starts again.
- Never change `--batch-pop`, `--workers`, `--seed`, `--batched` or any other flag between submissions.

Monitor with `squeue --me` and `tail -f cmc_moo.*.out`. One line is printed per network; `FAILS A TASK` means
infeasible, which is expected for some networks.

**After each generation, REPORT:**
- networks done;
- how many are feasible;
- the best worst-task accuracy;
- the range of the metabolic and wiring costs among the feasible networks.

## 4. Return the results

`cmc.moo` saves metrics only (`runs.csv`, `summary.json`, plots), no weights.

```bash
git add $RUN slurm/moo_cluster.sjob
git commit -m "cog20 NSGA-III budget front, p<POP>g3, 150k, fanin, fixed (yang) tasks"
git pull --rebase && git push
```

Do not commit the `cmc_moo.*.out/.err` logs, the benchmark directories, or anything outside these paths.

## 5. Rules

- Don't edit anything in `cmc/`. If something fails, stop and **REPORT** the error.
- Don't change the fixed settings in §0.
- Use cmc_env only.
- **Extending to a 4th generation later** happens only if the user asks for it. It would be the same command
  with `--n-gen 4` and the same `--output-dir`: generations 0–2 are replayed from `runs.csv`. This is not yet
  tested; check that no network of generations 0–2 is retrained.
