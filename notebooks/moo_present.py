import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent
    FRONT_DIR = ROOT / "runs/moo/dale_core5_nsga3_budget_p24g10"
    ZOOM_DIR = ROOT / "runs/moo_zoom/dale_core5_5pt_10seed"
    FEAS = 0.6
    return FEAS, FRONT_DIR, ZOOM_DIR, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # Cost-constrained multitask RNNs: the budget front

    NSGA-III over (rate budget, wiring budget); every evaluation is one full training.
    Objectives: worst-task accuracy, metabolic cost, wiring cost. Constraint: min task acc ≥ 0.6.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. NSGA-III front
    """)
    return


@app.cell
def _(FEAS, FRONT_DIR, np, pd, plt):
    def feasibility_map(f, ax):
        x, y = np.log10(f.rate_budget), np.log10(f.conn_budget)
        tc = ax.tricontourf(x, y, f.min_task_acc, levels=np.linspace(0.1, 0.9, 17), cmap="RdYlGn")
        ax.tricontour(x, y, f.min_task_acc, levels=[FEAS], colors="k", linewidths=2.5)
        ok = f.min_task_acc >= FEAS
        ax.scatter(x[ok], y[ok], s=8, c="k", alpha=.4)
        ax.scatter(x[~ok], y[~ok], s=40, marker="x", c="k", label="fails a task")
        ax.set_xlabel("log10 rate_budget"); ax.set_ylabel("log10 conn_budget")
        return tc

    front_df = pd.read_csv(FRONT_DIR / "runs.csv")
    fig_front, ax_front = plt.subplots(figsize=(7.5, 5.8))
    _tc = feasibility_map(front_df, ax_front)
    fig_front.colorbar(_tc, ax=ax_front, label="min_task_acc")
    ax_front.legend(loc="upper right")
    ax_front.set_title("does it do every task? (bold: min_task_acc = 0.6)")
    fig_front.tight_layout()
    return (fig_front,)


@app.cell
def _(FRONT_DIR, fig_front, mo):
    mo.hstack(
        [mo.image(FRONT_DIR / "front_rotate.gif", width=620), fig_front],
        widths="equal", align="center",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Zoom points (10 seeds each)

    control (no budget) · rate_limited · middle · wiring_limited · knee
    """)
    return


@app.cell
def _(ZOOM_DIR, mo):
    mo.hstack(
        [mo.image(ZOOM_DIR / "zoom_rotate.gif", width=620),
         mo.image(ZOOM_DIR / "zoom_topview.png", width=620)],
        widths="equal", align="center",
    )
    return


@app.cell
def _(ZOOM_DIR, pd):
    # Per-seed numbers, computed from the checkpoints by notebooks/present_data.py.
    POINTS = ["control", "rate_limited", "middle", "wiring_limited", "knee"]
    COLORS = dict(zip(POINTS, ["#444444", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]))
    _d = ZOOM_DIR / "present"
    ei_df = pd.read_csv(_d / "ei_ratio.csv")
    wi_df = pd.read_csv(_d / "w_vs_init.csv")
    dm_df = pd.read_csv(_d / "delta_motifs.csv")

    def strip(ax, df, col, null=None):
        # one column of dots per point (seeds), black bar = mean ± std, grey band = null ± 2 sd
        import numpy as _np
        for x, p in enumerate(POINTS):
            v = df[df.point == p]
            jit = _np.random.default_rng(0).uniform(-0.15, 0.15, len(v))
            ax.scatter(x + jit, v[col], s=16, color=COLORS[p], alpha=0.7)
            ax.errorbar(x, v[col].mean(), yerr=v[col].std(), fmt="_", color="k", ms=18, capsize=4)
            if null is not None:
                lo = (v.null_mean - 2 * v.null_std).mean()
                hi = (v.null_mean + 2 * v.null_std).mean()
                ax.fill_between([x - 0.3, x + 0.3], lo, hi, color="grey", alpha=0.25, lw=0)
        ax.set_xticks(range(len(POINTS)))
        ax.set_xticklabels(POINTS, rotation=30, ha="right", fontsize=8)

    return dm_df, ei_df, strip, wi_df


@app.cell
def _(ei_df, mo, plt, strip):
    _fig, _ax = plt.subplots(figsize=(7, 3.6))
    strip(_ax, ei_df, "I/E")
    _ax.axhline(1, color="k", lw=0.8, ls="--")
    _ax.set_ylabel("I / E input ratio")
    _ax.set_title("inhibition / excitation, feasible seeds (g = 1)")
    _fig.tight_layout()
    mo.vstack([
        mo.md("""
        ## 3. E/I balance across the zoom points (g = 1)

        Per neuron: total inhibitory input weight / total excitatory input weight (column sums of
        the trained W_rec), median over the 256 neurons. One dot = one feasible seed.
        Dashed line = balanced (I = E).
        """),
        _fig,
    ])
    return


@app.cell
def _(mo, plt, strip, wi_df):
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 3.8))
    for _ax, _col in zip(_axes, ["corr(W, W0)", "|ΔW| / |W|"]):
        strip(_ax, wi_df, _col)
        _ax.set_title(_col)
    _fig.suptitle("g = 1: how much does training move W from its random init?")
    _fig.tight_layout()
    mo.vstack([
        mo.md("""
        ## 4. Motif analysis (g = 1): training barely moves W

        Per feasible seed: correlation of the trained W with its own random init W₀, and the size of
        the learned change. Control: corr(W, W₀) = 0.99, the learned change is 6% of W.
        Budgets move W further (corr 0.91 → 0.43), but the learned change stays
        ≤ 31% of W. So motifs in W mostly measure the random init → look at ΔW = W − W₀ instead.
        """),
        _fig,
    ])
    return


@app.cell
def _(dm_df, mo, plt, strip):
    _stats = list(dict.fromkeys(dm_df.stat))
    _fig, _axes = plt.subplots(1, len(_stats), figsize=(4 * len(_stats), 3.8))
    for _ax, _s in zip(_axes, _stats):
        _v = dm_df[dm_df.stat == _s]
        strip(_ax, _v, "obs", null=True)
        _ax.axhline(0, color="grey", lw=0.6)
        _ax.set_title(_s, fontsize=9)
    _axes[0].set_ylabel("correlation in ΔW")
    _fig.suptitle("motifs in the learned change ΔW (grey band = null mean ± 2 sd)")
    _fig.tight_layout()
    mo.vstack([
        mo.md("""
        ## 5. ΔW = W − W₀: what learning added

        Correlation-type motifs of the learned change, against a shuffle null (each neuron keeps its
        input changes, which presynaptic neuron they come from is shuffled within each E/I block).
        """),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Motifs at g = 0.3

    *g* scales the random recurrent weights at init. At g = 0.3 the init is weak, so the learned
    weights grow to dominate W.

    **TODO:** g = 0.3 motif figures, computed from the g = 0.3 checkpoints.
    """)
    return


if __name__ == "__main__":
    app.run()
