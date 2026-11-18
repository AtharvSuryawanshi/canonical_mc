# Dynamics GIFs (Dly Go / Dly Anti, 5-seed run, fanin, 150k, seed 4)

One dot = one trial (the whole network state, 256 rates, projected on PCs); colour = stimulus angle.

- `gif1_time.gif`: C0, a Dly Go trial over time (stimulus, delay, go); axes PC1, PC2 of delay states + readout axis.
- `gif2_alpha.gif`: C0, input slid from delay (alpha 0) to response (alpha 1); where trials settle + fixed points.
- `gif3_go_vs_anti.gif`: C0, Dly Go (circles) vs Dly Anti (triangles), same stimuli, over time.
- `gif4_dlygo_compare.gif`: unconstrained (left) vs C0 (right), Dly Go over time.
- `gif5_goanti_compare.gif`: unconstrained (left) vs C0 (right), Dly Go vs Dly Anti over time.
- `roc_threshold.gif`: the like-to-like AUC explained: a similarity threshold slides down, each threshold gives one ROC point (E→E, unconstrained top, C0 bottom). Made by `roc_gif.py <out_dir>` (imports fig_wiring_like2like / fig_wiring_roc).

Made by the scripts here (`fp_gifs.py` -> gif1, gif2; `fp_gif3.py` -> gif3; `fp_gif_compare.py` -> gif4, gif5), run with
the cmc_env from `results/scripts/` (they import fig_fp_delaygo / fig_fp_task_pair; gif2 needs the
`results/front/fp_delaygo_C0_s4.npz` cache). Their output path `OUT` points to a scratch folder: change it before
re-running.

Numbers (end of delay, 64 trials): unconstrained 124 active neurons/trial, ring angle order 0.91, correct after +10 s
extra delay 0.91; C0 61 neurons, 0.65, 0.78. Dly Anti stores the flipped angle on the Dly Go ring in both
(nearest Go trial at end of delay: 180 deg / 179 deg away).
