# Of Canonical Microcircuits

### Atharv Suryawanshi

Why are canonical circuits canonical? Why are they so widely present in the cortex?

## Train on GPU (LRZ)

`train.sjob` follows the [LRZ batch-script template](https://doku.lrz.de/running-large-memory-jobs-on-the-linux-cluster-1311572409.html#RunninglargememoryjobsontheLinuxCluster-Step1:Prepareabatchjobscript) and requests one A100 MIG slice (`gpu:3g.20gb`, QoS `mig`) on `lrz-dgx-a100-40x8-mig` ([AI Systems single-GPU jobs](https://doku.lrz.de/5-2-slurm-batch-jobs-single-gpu-1898974516.html)). Submit from this directory on an AI Systems login node:

```bash
sbatch train.sjob
```

Pass extra flags through to `train_cog.py`:

```bash
sbatch train.sjob --steps 5000 --model dale --n-neurons 256
```

Check the queue and follow logs (`%x.%j.%N` is job name, job ID, and node):

```bash
squeue --me
tail -f cmc_train.*.out
```

Cancel a job with `scancel <job_id>`. To use another GPU partition (see `sinfo`), edit `--partition` in `train.sjob` or override at submit time:

```bash
sbatch --partition=lrz-hgx-h100-94x4 train.sjob
```

`train_cog.py` already picks CUDA when a GPU is visible; the job script forces `--device cuda`.

