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
cmc/                 installable package -- all the code that runs
  task.py            Yang et al. (2019) task battery, trial generation, c_mask
  network.py         LeakyRNN and DaleRNN (sign-constrained, E-only readout)
  train_cog.py       objectives, regularizers, accuracy scoring, training loop, CLI
  pareto.py          (lambda_rate, lambda_connectivity) sweeps and Pareto fronts
  paths.py           repo-anchored checkpoints/ and pareto_runs/ locations
notebooks/           analysis and exploration, imports the package
slurm/               LRZ batch scripts (train.sjob, pareto.sjob)
archives/            legacy code, kept for reference, not imported
checkpoints/         trained weights (written by cmc.train_cog)
pareto_runs/         one directory per sweep (written by cmc.pareto)
theory.md            the neuroscience the objectives are meant to encode
FIXED_ISSUES.md      audited mismatches between that theory and the code
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

Planned extras, not installed by default:

```bash
pip install -e ".[optim]"      # pymoo, for optimising over the front
```

## Usage

Train a single model (the CLI is `cmc.train_cog`; `cmc-train` is installed as an
equivalent console script):

```bash
python -m cmc.train_cog --model dale --task-battery sanity3 --n-neurons 256 --steps 5000
```

Weights land in `checkpoints/{model}_{n_tasks}_{n_steps}_{timestamp}.pt`, regardless
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
`--lambda-*-min/max` defaults in `cmc/pareto.py` are already calibrated against
trained lambda=0 solutions, and the comment above them records the measurements.

Sweep the front:

```bash
python -m cmc.pareto --model dale --task-battery core5 --n-lambda 8 --n-seeds 3
```

Each sweep writes `pareto_runs/<run>/` containing `runs.csv` (one row per seed),
`summary.csv` (seed means, which the notebooks read) and `summary.json` (the full
configuration, including the regularizer settings the gradient actually saw).

Analysis lives in `notebooks/`: `pareto_analysis.ipynb` for the fronts,
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
sbatch slurm/pareto.sjob --n-lambda 5 --model dale
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
