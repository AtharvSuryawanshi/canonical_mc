# results/

Figures and tables for the presentation. Each script in `scripts/` makes one
figure family and writes it to `results/<figure>/<run>/`; rerunning a script
overwrites its output. Run from the repo root in `cmc_env`.

| Script | Output | Data |
|---|---|---|
| `scripts/fig_learning_curves.py` | `learning_curves/<run>/<point>.png`, `best_seeds.csv` | `runs/moo_zoom/<run>/runs_by_step.csv` (a `--save-at` run) |
| `scripts/fig_unit_task_variance.py` | `unit_task_variance/<run>/<point>_unit<k>_<task>.png` (Yang Fig. 3a,b) | checkpoints of the best seed per point in `runs/moo_zoom/<run>/runs.csv` |
| `scripts/fig_task_io.py` | `task_io/<run>/<point>_<task>_{inputs,outputs}.png` (Yang Fig. 1-style input/output panels) | best-seed checkpoint of `--point` in `runs/moo_zoom/<run>/` |
| `scripts/fig_cost_bars.py` | `cost_bars/<run>/<point>.png` (accuracy + L1 wiring + L2 firing cost, costs scaled to max over points) | `runs/moo_zoom/<run>/runs.csv` |
| `scripts/fig_task_variance_clusters.py` | `unit_task_variance/<run>/<point>_tv_clusters.png` (Yang Fig. 3c), `<point>_silhouette.png`, `clusters_summary.csv` | best-seed checkpoints in `runs/moo_zoom/<run>/` |
| `scripts/fig_front_gif.py` | `front/cog20_50k_front.gif`, `_pic.png` still, `_top.png` 2D top view, `front_networks.csv` | all cog20 50k batched runs in `runs/moo/` and `runs/moo_zoom/` (408 networks) |
| `scripts/fig_ei_weight_hist.py` | `ei_weights/<run>/ei_weight_hist.png` (1x4 E vs I |W_rec| histograms) | best-seed checkpoints in `runs/moo_zoom/<run>/` |
| `scripts/fig_ei_weights.py` | `ei_weights/<run>/ei_weight_hist.png` (1x4: E→E, E→I, I→E, I→I, one curve per point) | best-seed checkpoints in `runs/moo_zoom/<run>/` |
| `scripts/fig_conn_modules.py` | `conn_modules/<run>/conn_modules.png` (1x4 W_rec ordered by E/I + Louvain module, Q vs block-shuffle null), `modules_summary.csv` | best-seed checkpoints in `runs/moo_zoom/<run>/` |
