# Of Canonical Microcircuits

### Atharv Suryawanshi

Why are canonical circuits canonical? Why are they so widely present in the cortex?

The working hypothesis here is that cortical circuit motifs are what you get when
a network has to solve many tasks under **firing** and **wiring** budgets. So we
train sign-constrained (Dale's law) RNNs on the Yang et al. (2019) cognitive task
battery while charging them for higher firing rates and for recurrent connectivity, and
look at the Pareto front that trades task performance against those two costs.

## Layout

```
cmc/                    installable package -- all the code that runs
  task.py               Yang et al. (2019) task battery, trial generation, c_mask
  network.py            LeakyRNN and DaleRNN (sign-constrained, E-only readout)
  train_cog.py          objectives, regularizers, budgets, training loop, CLI
  front.py              Pareto dominance and the feasibility rule (shared)
  runner.py             train + evaluate one network, common CLI flags (shared)
  lambda_pareto.py      lambda-grid sweeps (weighted sum) and their fronts
  lambda_zoom.py        many seeds at a few chosen lambdas, every network saved
  moo.py                NSGA-III (pymoo) search over cost budgets
  moo_zoom.py           many seeds at a few chosen budget points, every network saved
  batched.py            many networks trained together in one process (--batched)
  paths.py              repo-anchored runs/ locations
  pareto.py, zoom_lambda.py   deprecated aliases of lambda_pareto / lambda_zoom
notebooks/              lambda_pareto_analysis, moo_pareto_analysis, network analysis
slurm/                  LRZ batch scripts (train, lambda_pareto, lambda_zoom, moo, moo_zoom)
runs/                   all experiment output, committed (git is the cluster transfer)
  checkpoints/          trained weights (written by cmc.train_cog)
  lambda_pareto/<run>/  one directory per lambda sweep
  lambda_zoom/<run>/    saved networks at chosen lambdas, weights included
  moo/<run>/            one directory per NSGA-III search
  moo_zoom/<run>/       saved networks at chosen budget points, weights included
archives/               legacy code, kept for reference, not imported
reports/                docs and write-ups:
  theory.md             the neuroscience the objectives are meant to encode
  FIXED_ISSUES.md       audited mismatches between that theory and the code
  IMPLEMENTATION.md     implementation summary with exact values
  PRESENTATION.md       talk outline; ABSTRACT.md: COSYNE 2027 plan
```

## Install

```bash
conda create -n cmc_env python=3.12 -y
conda activate cmc_env
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs the package in *editable* mode: `import cmc` resolves to
this working tree, so edits take effect without reinstalling, and the notebooks and
batch scripts can import it from any directory.

Check it worked:

```bash
python -c "import cmc, torch; print(cmc.__version__, torch.__version__, torch.cuda.is_available())"
```

`requirements.txt` installs the default torch wheel for your platform (CPU-only on
Windows and macOS). For a specific CUDA build, install torch first from the
[PyTorch index](https://pytorch.org/get-started/locally/) and then run
`pip install -r requirements.txt` -- the existing install already satisfies the pin
and will not be replaced.

`requirements.txt` includes pymoo (for `cmc.moo`). Without requirements.txt,
the same extra is available as:

```bash
pip install -e ".[optim]"
```

## Usage

Train a single model (the CLI is `cmc.train_cog`; `cmc-train` is installed as an
equivalent console script):

```bash
python -m cmc.train_cog --model dale --task-battery sanity3 --n-neurons 256 --steps 5000
```

Weights land in `runs/checkpoints/{model}_{n_tasks}_{n_steps}_{timestamp}.pt`, regardless
of which directory you launched from -- `cmc/paths.py` anchors the output directories
to the repo root. Set `CMC_ROOT` to redirect them (e.g. to cluster scratch).

Before choosing lambda ranges, print how large each loss term actually is and the
lambda at which it breaks even against the task loss:

```bash
python -m cmc.train_cog --model dale --task-battery sanity3 --report-scales
```

This reports at *initialization*. Between init and a trained solution the rate
cost grows ~26x and the task loss falls ~30x, so the break-even lambdas it prints
overshoot the useful range by ~500x (rate) and ~20x (wiring) -- treat it as a
check that the terms are finite rather than as the centre of the grid. The
`--lambda-*-min/max` defaults in `cmc/lambda_pareto.py` are already calibrated against
trained lambda=0 solutions, and the comment above them records the measurements.

Sweep the front:

```bash
python -m cmc.lambda_pareto --model dale --task-battery core5 --n-lambda 8 --n-seeds 3
```

Each sweep writes `runs/lambda_pareto/<run>/` containing `runs.csv` (one row per seed),
`summary.csv` (seed means, which the notebooks read) and `summary.json` (the full
configuration, including the regularizer settings the gradient actually saw).

To inspect networks rather than locate the front, train many seeds at a few
chosen points and keep every network (default: a control / rate / wiring / both
2x2 design taken from the core5 front, 10 seeds each, ~5 h on a GPU):

```bash
python -m cmc.lambda_zoom
sbatch slurm/lambda_zoom.sjob
```

Each network lands in `runs/lambda_zoom/<run>/<point>/seed_XX.pt`, loadable with
`cmc.train_cog.load_checkpoint`, together with its metrics and per-neuron task
variance; `runs.csv` lists which seeds still do every task. Weights are committed
(about 1 MB per network).

Instead of a lambda grid, NSGA-III (pymoo) can choose which networks to train.
Objectives are worst-task error and log metabolic / wiring cost, with
`min_task_acc >= 0.6` as a constraint. The default genome is a pair of **cost
budgets**, not lambdas. Each network is trained to do the tasks as well as it
can while staying under its ceilings:
- **Wiring:** W_rec is projected onto the budget after every step.
- **Rate:** a learned multiplier.

The default run is population 12 x 6 generations = 72 networks.
`--genome lambda` (weighted sum, population 8 x 4) was used to validate the loop
against the 6x6 grid.

```bash
python -m cmc.moo --steps 200 --pop-size 10 --n-gen 2 --device cpu        # smoke test
python -m cmc.moo --points "0.0087,0.0045; 0.0144,0.0056"                 # fixed budgets, no search
sbatch slurm/moo.sjob --workers 4 --reference-run runs/lambda_pareto/dale_core5_6x6_2026_09_23_05_45_12_5802320
```

`moo_vs_reference.png` and `summary.json` report how many of the found networks
are dominated by a seed-0 network of the reference sweep. A job that hits its
time limit resumes when resubmitted with the same `--output-dir` and arguments.

To inspect networks on the budget front, `cmc.moo_zoom` trains 10 seeds at each of 5
budget points and saves every network: an unconstrained control, three points along
the ~0.65 iso-accuracy curve (rate-limited, middle, wiring-limited: equally competent,
different cost binding) and the knee. That's 50 networks, ~6-7 h one at a time.

```bash
python -m cmc.moo_zoom --steps 40 --n-seeds 2 --device cpu --eval-seeds 10000   # smoke test
sbatch slurm/moo_zoom.sjob --workers 4 --output-dir runs/moo_zoom/dale_core5_5pt_10seed
```

Seed 0 at a point is the network `cmc.moo` trained there: bit-identical on CPU, and on
the GPU equal up to CUDA nondeterminism (well within seed noise). Each
`runs/moo_zoom/<run>/<point>/seed_XX.pt` holds the weights, budgets, the learned
rate multiplier per step, metrics and per-neuron task variance.

**Batched training (`--batched`, opt-in).** A single small RNN leaves the GPU mostly
idle, so `cmc.moo` and `cmc.moo_zoom` can train up to `--batch-pop` networks (default
24) together in one process, with their weights stacked into the same tensors
(`cmc/batched.py`). P networks then cost roughly what one costs. Each network keeps
its own init, budgets, Adam state and gradient clipping, but all networks in a batch
see the same training trials. They are therefore not bit-identical to the default
one-at-a-time networks (they are equally valid samples), so use one mode per
comparison. With `--workers N` and `--batched`, N batches run at once, spread over
`--devices`. Without `--batched` nothing changes.

```bash
python -m cmc.batched --bench --task-battery all --pop 1,4,24 --steps 20        # time per step vs P
python -m cmc.moo --batched --batch-pop 24 --task-battery all --steps 40000     # whole generation per batch
python -m cmc.moo_zoom --batched --batch-pop 50 --workers 2 --devices cuda:0,cuda:1
python tests/test_batched.py      # batched == single-network training, to float precision
```

Analysis lives in `notebooks/`: `lambda_pareto_analysis.ipynb` for the fronts,
`analysis_of_network.ipynb` for task variance and clustering,
`task_exploration.ipynb` for the task battery itself.

## Train on GPU (LRZ)

`slurm/train.sjob` follows the [LRZ batch-script template](https://doku.lrz.de/running-large-memory-jobs-on-the-linux-cluster-1311572409.html#RunninglargememoryjobsontheLinuxCluster-Step1:Prepareabatchjobscript)
and requests one A100 MIG slice (`gpu:3g.20gb`, QoS `mig`) on
`lrz-dgx-a100-40x8-mig` ([AI Systems single-GPU jobs](https://doku.lrz.de/5-2-slurm-batch-jobs-single-gpu-1898974516.html)).
Submit from the repo root on an AI Systems login node; the scripts activate
`cmc_env` (override with `CONDA_ENV`) and run the package via `python -m`:

```bash
sbatch slurm/train.sjob
```

Pass extra flags straight through:

```bash
sbatch slurm/train.sjob --steps 5000 --model dale --n-neurons 256
sbatch slurm/lambda_pareto.sjob --n-lambda 5 --model dale
```

Check the queue and follow logs (`%x.%j.%N` is job name, job ID, and node; they are
written to the directory you submitted from):

```bash
squeue --me
tail -f cmc_train.*.out
```

Cancel with `scancel <job_id>`. To use another GPU partition (see `sinfo`), edit
`--partition` in the job script or override at submit time:

```bash
sbatch --partition=lrz-hgx-h100-94x4 slurm/train.sjob
```

Both CLIs pick CUDA when a GPU is visible; the job scripts force `--device cuda`.
