import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from pathlib import Path

    from cmc.front import FEASIBLE_MIN_TASK_ACC, compute_front
    from cmc.task import rule_name

    ROOT = Path(__file__).resolve().parent.parent
    FRONT_DIR = ROOT / "runs/moo/dale_all_nsga3_budget_p48g3_50k"          # NSGA-III cog20 search, 1 seed/budget
    DENSE_DIR = ROOT / "runs/moo_zoom/dale_all_front27_2seed_50k"          # its 27 front budgets x 2 new seeds
    ZOOM4_DIR = ROOT / "runs/moo_zoom/dale_all_4pt_12seed_50k"             # 3 of them + control x 12 new seeds
    GAP_DIR = ROOT / "runs/moo_zoom/dale_all_gap12_2seed_50k"              # 12 budgets in the gaps x 2 seeds
    LINES_DIR = ROOT / "runs/moo_zoom/dale_all_lines5_4seed_50k"           # 5 iso-cost lines x 6 budgets x 4 seeds
    OBJ = ("min_task_acc", "metabolic_cost", "wiring_cost")
    SIL_REF = 0.2    # reference silhouette, dashed line on every silhouette-vs-k plot
    return (
        DENSE_DIR,
        FEASIBLE_MIN_TASK_ACC,
        FRONT_DIR,
        GAP_DIR,
        LINES_DIR,
        OBJ,
        SIL_REF,
        ZOOM4_DIR,
        compute_front,
        go,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md("""
    # cog20 budget front explorer

    All runs: 20 tasks, 50k steps, g = 1, batched training, same 10 eval seeds.

    * **search**: the NSGA-III run (`dale_all_nsga3_budget_p48g3_50k`), 144 budgets × **1 seed**. Its front
      keeps the luckiest seed at each budget, so it is optimistic.
    * **retrained**: the search's 27 front budgets trained again with new seeds: 2 each from the dense zoom
      (`dale_all_front27_2seed_50k`), plus 12 more for f00 / f03 / f23 from the 4-point zoom
      (`dale_all_4pt_12seed_50k`, there called wiring_limited / knee / rate_limited).
    * **gap**: 12 budgets the search never put on its front, placed in the largest holes of the budget plane
      (farthest from every budget with saved networks), 2 seeds each (`dale_all_gap12_2seed_50k`, g00..g11).
    * **lines**: 5 iso-cost lines across the feasibility cliff, log10 rate + log10 wiring = −4.7, −4.5, −4.3,
      −4.1, −3.9 (L0..L4), 6 budgets each from the rate-limited end (p0) to the wiring-limited end (p5),
      4 seeds each (`dale_all_lines5_4seed_50k`).
    * **honest front**: the non-dominated budgets among all 69 (27 front + 12 gap + 30 lines), using the **mean over
      retrained seeds only** (the search's own seed is left out: it was selected for being good). The gap
      and line budgets have no search seed, so no inflation.

    Surface options: the **attainment surface** is what a Pareto front is: at each (metabolic, wiring) cost,
    the best min_task_acc reached by any network that costs no more. It is a monotone staircase and invents
    nothing between points. The triangulated mesh (the old view) stretches facets across gaps.
    """)
    return


@app.cell
def _(
    DENSE_DIR,
    FEASIBLE_MIN_TASK_ACC,
    FRONT_DIR,
    GAP_DIR,
    LINES_DIR,
    OBJ,
    ZOOM4_DIR,
    compute_front,
    np,
    pd,
):
    moo = pd.read_csv(FRONT_DIR / "runs.csv").drop_duplicates(["x0", "x1"]).reset_index(drop=True)
    moo["is_feasible"] = moo.min_task_acc >= FEASIBLE_MIN_TASK_ACC
    moo["is_pareto"], _ = compute_front(moo.to_dict("records"), OBJ)
    dense = pd.read_csv(DENSE_DIR / "runs.csv")
    zoom4 = pd.read_csv(ZOOM4_DIR / "runs.csv")
    gap = pd.read_csv(GAP_DIR / "runs.csv")
    lines = pd.read_csv(LINES_DIR / "runs.csv")
    for _d, _dir, _run in ((dense, DENSE_DIR, "dense"), (zoom4, ZOOM4_DIR, "4pt"), (gap, GAP_DIR, "gap"),
                           (lines, LINES_DIR, "lines")):
        _d["is_feasible"] = _d.min_task_acc >= FEASIBLE_MIN_TASK_ACC
        _d["ckpt"] = [str(_dir / c) for c in _d.checkpoint]
        _d["run"] = _run

    # Every retrained seed, keyed by the dense zoom's point name (f00..f26). The 4-point zoom's budgets are
    # exact copies of three of them; match on the budget values.
    _names = dense[dense.point != "control"].groupby("point")[["rate_budget", "conn_budget"]].first()
    def _name_of(rb, cb):
        hit = _names[np.isclose(_names.rate_budget, rb, rtol=1e-9) & np.isclose(_names.conn_budget, cb, rtol=1e-9)]
        return hit.index[0] if len(hit) else None
    _z = zoom4[zoom4.point != "control"].copy()
    _z["alias"] = _z.point
    _z["point"] = [_name_of(r.rate_budget, r.conn_budget) for r in _z.itertuples()]
    _d = dense[dense.point != "control"].copy()
    _d["alias"] = ""
    _g = gap[gap.point != "control"].copy()
    _g["alias"] = ""
    _l = lines.copy()
    _l["alias"] = ""
    retrained = pd.concat([_d, _z.dropna(subset=["point"]), _g, _l], ignore_index=True)
    controls = pd.concat([dense[dense.point == "control"], zoom4[zoom4.point == "control"]], ignore_index=True)

    _cols = ["min_task_acc", "mean_acc", "metabolic_cost", "wiring_cost", "conn_frac"]
    honest = retrained.groupby("point").agg(
        rate_budget=("rate_budget", "first"), conn_budget=("conn_budget", "first"),
        alias=("alias", lambda s: ",".join(sorted({a for a in s if a}))),
        n_seeds=("seed", "size"), n_feasible=("is_feasible", "sum"),
        min_task_acc_sd=("min_task_acc", "std"), **{c: (c, "mean") for c in _cols},
    ).reset_index()
    _search = []
    for _r in honest.itertuples():
        _hit = moo[np.isclose(moo.rate_budget, _r.rate_budget, rtol=1e-9) & np.isclose(moo.conn_budget, _r.conn_budget, rtol=1e-9)]
        _search.append(_hit.min_task_acc.iloc[0] if len(_hit) else np.nan)
    honest["search_min_task_acc"] = _search
    honest["inflation"] = honest.search_min_task_acc - honest.min_task_acc
    honest["is_feasible"] = honest.min_task_acc >= FEASIBLE_MIN_TASK_ACC
    honest["is_pareto"], _ = compute_front(honest.to_dict("records"), OBJ)
    return controls, honest, lines, moo, retrained


@app.cell
def _(mo):
    surface_kind = mo.ui.dropdown(
        options=["attainment: honest front", "attainment: search front", "triangulated: search front", "none"],
        value="attainment: honest front", label="surface")
    surface_opacity = mo.ui.slider(0.1, 1.0, step=0.05, value=0.5, label="surface opacity")
    show_seeds = mo.ui.checkbox(value=True, label="retrained single seeds")
    show_lines = mo.ui.checkbox(value=True, label="search → retrained lines")
    show_background = mo.ui.checkbox(value=True, label="search: dominated / failing")
    mo.hstack([surface_kind, surface_opacity, show_seeds, show_lines, show_background], justify="start", gap=2)
    return (
        show_background,
        show_lines,
        show_seeds,
        surface_kind,
        surface_opacity,
    )


@app.cell
def _(
    FEASIBLE_MIN_TASK_ACC,
    controls,
    go,
    honest,
    moo,
    np,
    pd,
    retrained,
    show_background,
    show_lines,
    show_seeds,
    surface_kind,
    surface_opacity,
):
    lx = lambda s: np.log10(s.astype(float))

    def xyz(sub):
        return dict(x=lx(sub.metabolic_cost), y=lx(sub.wiring_cost), z=sub.min_task_acc.astype(float))

    def hover(sub, label):
        out = []
        for r in sub.itertuples():
            budget = (f"budgets: rate {r.rate_budget:.4g}, wiring {r.conn_budget:.4g}"
                      if pd.notna(getattr(r, "rate_budget", np.nan)) else "no budget")
            extra = ""
            if hasattr(r, "n_seeds"):
                extra = (f"<br>mean of {r.n_seeds} retrained seeds ({r.n_feasible} feasible), sd {r.min_task_acc_sd:.3f}"
                         + (f"<br>search seed: {r.search_min_task_acc:.3f} (inflation {r.inflation:+.3f})"
                            if pd.notna(r.search_min_task_acc) else "<br>gap budget (not in the search)"))
            name = f"{label} {getattr(r, 'point', '')}" + (f" seed {r.seed}" if hasattr(r, "seed") else "")
            if getattr(r, "alias", ""):
                name += f" ({r.alias})"
            out.append(f"<b>{name}</b><br>{budget}<br>min_task_acc={r.min_task_acc:.3f}  mean_acc={r.mean_acc:.3f}"
                       f"<br>metabolic={r.metabolic_cost:.4g}  wiring={r.wiring_cost:.4g}  conn_frac={r.conn_frac:.3f}{extra}")
        return out

    allx = lx(pd.concat([moo.metabolic_cost, retrained.metabolic_cost, controls.metabolic_cost]))
    ally = lx(pd.concat([moo.wiring_cost, retrained.wiring_cost, controls.wiring_cost]))
    pad = lambda v: (v.min() - 0.05 * (v.max() - v.min()), v.max() + 0.05 * (v.max() - v.min()))
    (x0, x1), (y0, y1) = pad(allx), pad(ally)

    def attainment(df, n=90):
        """z(x, y) = best min_task_acc among networks with log costs <= (x, y); NaN where none."""
        px, py, pz = lx(df.metabolic_cost).values, lx(df.wiring_cost).values, df.min_task_acc.values
        xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
        Z = np.full((n, n), np.nan)
        for j, yv in enumerate(ys):
            for i, xv in enumerate(xs):
                m = (px <= xv) & (py <= yv)
                if m.any():
                    Z[j, i] = pz[m].max()
        return xs, ys, Z

    traces = []
    sfr = moo[moo.is_pareto]
    hfr = honest[honest.is_pareto]
    cbar = dict(title="min_task_acc", x=1.02, len=0.5, y=0.3)
    kind = surface_kind.value
    if kind.startswith("attainment"):
        src = honest if "honest" in kind else moo
        xs, ys, Z = attainment(src)
        traces.append(go.Surface(x=xs, y=ys, z=Z, surfacecolor=Z, colorscale="Viridis", cmin=0, cmax=1,
                                 opacity=surface_opacity.value, name=kind, showlegend=True, hoverinfo="skip",
                                 colorbar=cbar))
    elif kind.startswith("triangulated") and len(sfr) >= 3:
        traces.append(go.Mesh3d(**xyz(sfr), delaunayaxis="z", intensity=sfr.min_task_acc.astype(float),
                                colorscale="Viridis", cmin=0, cmax=1, opacity=surface_opacity.value,
                                name=kind, showlegend=True, hoverinfo="skip", colorbar=cbar))

    if show_background.value:
        bad, dom = moo[~moo.is_feasible], moo[moo.is_feasible & ~moo.is_pareto]
        traces += [
            go.Scatter3d(**xyz(bad), mode="markers", name=f"search: fails a task ({len(bad)})",
                         marker=dict(size=2.5, color="lightgrey", symbol="x"), hovertext=hover(bad, "search"),
                         hoverinfo="text"),
            go.Scatter3d(**xyz(dom), mode="markers", name=f"search: dominated ({len(dom)})",
                         marker=dict(size=2.5, color="silver", opacity=0.6), hovertext=hover(dom, "search"),
                         hoverinfo="text"),
        ]
    traces.append(go.Scatter3d(**xyz(sfr), mode="markers", name=f"search front, 1 seed ({len(sfr)})",
                               marker=dict(size=4, color="#2b6cb0", line=dict(color="white", width=0.5)),
                               hovertext=hover(sfr, "search front"), hoverinfo="text"))

    if show_lines.value:
        lxs, lys, lzs = [], [], []
        for _r in honest.itertuples():
            _hit = moo[np.isclose(moo.rate_budget, _r.rate_budget, rtol=1e-9) & np.isclose(moo.conn_budget, _r.conn_budget, rtol=1e-9)]
            if len(_hit):
                _h = _hit.iloc[0]
                lxs += [np.log10(_h.metabolic_cost), np.log10(_r.metabolic_cost), None]
                lys += [np.log10(_h.wiring_cost), np.log10(_r.wiring_cost), None]
                lzs += [_h.min_task_acc, _r.min_task_acc, None]
        traces.append(go.Scatter3d(x=lxs, y=lys, z=lzs, mode="lines", name="search seed → retrained mean",
                                   line=dict(color="#a0aec0", width=3), hoverinfo="skip"))

    if show_seeds.value:
        for ok, sym in ((True, "circle"), (False, "x")):
            s = retrained[retrained.is_feasible == ok]
            traces.append(go.Scatter3d(**xyz(s), mode="markers", legendgroup="seeds",
                                       name="retrained seeds" + ("" if ok else " (fail a task)"),
                                       marker=dict(size=2.8, color="#ed8936", symbol=sym, opacity=0.7),
                                       hovertext=hover(s, "retrained"), hoverinfo="text"))
    dom_h = honest[~honest.is_pareto]
    traces.append(go.Scatter3d(**xyz(dom_h), mode="markers", name=f"retrained mean, dominated ({len(dom_h)})",
                               marker=dict(size=6, color="#ed8936", symbol="diamond", line=dict(color="black", width=1)),
                               hovertext=hover(dom_h, "retrained mean"), hoverinfo="text"))
    traces.append(go.Scatter3d(**xyz(hfr), mode="markers", name=f"HONEST FRONT ({len(hfr)})",
                               marker=dict(size=9, color="#d69e2e", symbol="diamond", line=dict(color="black", width=2)),
                               hovertext=hover(hfr, "honest front"), hoverinfo="text"))

    cm = controls[["metabolic_cost", "wiring_cost", "min_task_acc", "mean_acc", "conn_frac"]].mean()
    crow = pd.DataFrame([{**cm.to_dict(), "rate_budget": np.nan}])
    if show_seeds.value:
        traces.append(go.Scatter3d(**xyz(controls), mode="markers", legendgroup="control", showlegend=False,
                                   marker=dict(size=3, color="black", opacity=0.6),
                                   hovertext=hover(controls, "control"), hoverinfo="text"))
    traces.append(go.Scatter3d(**xyz(crow), mode="markers", legendgroup="control",
                               name=f"control, no cost ({len(controls)} seeds)",
                               marker=dict(size=10, color="black", symbol="diamond"),
                               hovertext=[f"<b>control</b>, mean of {len(controls)} seeds<br>"
                                          f"min_task_acc={cm.min_task_acc:.3f} ± {controls.min_task_acc.std():.3f}"],
                               hoverinfo="text"))
    traces.append(go.Surface(x=[x0, x1], y=[y0, y1], z=[[FEASIBLE_MIN_TASK_ACC] * 2] * 2,
                             colorscale=[[0, "orange"], [1, "orange"]], opacity=0.12, showscale=False,
                             name=f"min_task_acc = {FEASIBLE_MIN_TASK_ACC}", showlegend=True, hoverinfo="skip"))

    fig = go.Figure(traces)
    fig.update_layout(
        title="cog20 budget front: search (1 seed) vs retrained means",
        scene=dict(xaxis_title="log10(metabolic cost)", yaxis_title="log10(wiring cost)",
                   zaxis_title="min_task_acc", zaxis=dict(range=[0, 1]), aspectmode="cube"),
        height=850, legend=dict(x=0.0, y=1.0, font=dict(size=10)), margin=dict(l=0, r=0, t=40, b=0),
    )
    fig
    return


@app.cell
def _(honest, mo):
    _t = honest.sort_values("conn_budget")[[
        "point", "alias", "rate_budget", "conn_budget", "n_seeds", "n_feasible", "search_min_task_acc",
        "min_task_acc", "min_task_acc_sd", "inflation", "metabolic_cost", "wiring_cost", "conn_frac", "is_pareto",
    ]].round(4)
    mo.vstack([
        mo.md(f"""
        ## The {len(honest)} budgets: search seed vs retrained mean

        `search_min_task_acc` is the single seed the search kept; `min_task_acc` is the mean over the retrained
        seeds. Inflation = search − retrained (mean **{honest.inflation.mean():+.3f}**). `is_pareto` = on the
        honest front ({int(honest.is_pareto.sum())} of {len(honest)}).
        """),
        mo.ui.table(_t, selection=None, page_size=30),
    ])
    return


@app.cell
def _(FEASIBLE_MIN_TASK_ACC, lines, mo, np, rule_name):
    import matplotlib.pyplot as _plt

    _d = lines.assign(line=lines.point.str[:2], pos=lines.point.str[3].astype(int))
    _acc = [c for c in _d.columns if c.startswith("acc_")]
    _worst = _d[_acc].idxmin(axis=1).str[4:].value_counts()
    _cost = {"L0": -4.7, "L1": -4.5, "L2": -4.3, "L3": -4.1, "L4": -3.9}
    _cols = _plt.cm.viridis(np.linspace(0.1, 0.85, 5))
    _f, (_a1, _a2) = _plt.subplots(1, 2, figsize=(15, 4.6))
    for _c, (_ln, _g) in zip(_cols, _d.groupby("line")):
        _m = _g.groupby("pos").min_task_acc.mean()
        _a1.plot(_m.index, _m.values, "-", color=_c, lw=2, label=f"{_ln} (log r + log w = {_cost[_ln]})")
        _a1.scatter(_g.pos + 0.04 * (_g.seed - 1.5), _g.min_task_acc, color=_c, s=14, alpha=0.7)
        _fr = _g.groupby("pos").is_feasible.mean()
        _a2.plot(_fr.index, _fr.values, "o-", color=_c, lw=2, label=_ln)
    _a1.axhline(FEASIBLE_MIN_TASK_ACC, color="k", lw=1, ls="--")
    for _a in (_a1, _a2):
        _a.set_xticks(range(6))
        _a.set_xticklabels(["p0\nrate-limited", "p1", "p2", "p3", "p4", "p5\nwiring-limited"])
    _a1.set_ylabel("min_task_acc (dots: seeds, line: mean)")
    _a1.legend(fontsize=8)
    _a1.set_title("accuracy along each iso-cost line")
    _a2.set_ylabel(f"fraction of seeds with min_task_acc ≥ {FEASIBLE_MIN_TASK_ACC}")
    _a2.set_ylim(-0.05, 1.05)
    _a2.legend(fontsize=8)
    _a2.set_title("fraction of seeds that solve every task")
    _f.tight_layout()

    mo.vstack([
        mo.md(f"""
        ## The 5 iso-cost lines

        Each line holds log10 rate budget + log10 wiring budget fixed and trades one for the other: p0 is
        the rate-limited end (tight rate, loose wiring), p5 the wiring-limited end. 4 seeds per budget.
        The worst task is **{rule_name[_worst.index[0]]}** in {_worst.iloc[0]} of {len(_d)} networks, and
        its accuracy is bimodal (chance ~0.5, or solved ~0.75), so a budget's mean min_task_acc mixes
        solved and unsolved seeds: the right panel (fraction of seeds that clear 0.6) is the cleaner read.
        """),
        _f,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Task-variance clusters of one network (Yang 2019 Fig 2b)

    Every network with saved weights: the 27 budgets × 2 dense-zoom seeds, the 4-point zoom's 12 seeds at
    f00 / f03 / f23, the 12 gap budgets × 2 seeds, the 30 line budgets × 4 seeds, and the controls (the 144 search networks, small black dots, have no saved weights).
    **Click a network in the budget plane** (or use the dropdown; the plane wins while it has a
    selection, the reset button in its toolbar clears it). The plane is the one from
    `moo_pareto_analysis`: rate budget × wiring budget, background = min_task_acc interpolated
    (red-yellow-green) from the best estimate at every budget: the retrained mean where there is one (the
    69 budgets), the single search seed elsewhere; bold line = 0.6. The seeds of one budget sit on a small ring around
    it; the controls have no budget and sit in a ring to the right.

    Each network picks its own k, as in Yang: active units (total task variance > 1% of the network's
    max), each unit's variance normalised by its max over tasks, k-means for k = 2-20, k = argmax
    silhouette. Next to the heatmap, the silhouette curve with a **shuffled null** (each unit's
    20-vector permuted across tasks, 3 shuffles per k: keeps how selective a unit is, destroys which
    tasks go together). Read k with the curve: a sharp peak well above the null is a real k; a plateau
    means k is not determined and the argmax is close to arbitrary.
    """)
    return


@app.cell
def _(controls, np, pd, retrained):
    nets = pd.concat([controls, retrained], ignore_index=True)
    nets["alias"] = nets.alias.fillna("")
    nets["_later"] = nets.point != "control"
    nets = nets.sort_values(["_later", "point", "run", "seed"]).drop(columns="_later").reset_index(drop=True)
    nets["label"] = [
        f"{r.point}{f' ({r.alias})' if r.alias else ''}  seed {r.seed}  [{r.run}]  min_acc {r.min_task_acc:.2f}"
        for r in nets.itertuples()]
    nets["lx"], nets["ly"] = np.log10(nets.metabolic_cost.astype(float)), np.log10(nets.wiring_cost.astype(float))
    return (nets,)


@app.cell
def _(FEASIBLE_MIN_TASK_ACC, go, honest, lines, mo, moo, nets, np):
    from scipy.interpolate import griddata

    # Budget plane as in moo_pareto_analysis: background = min_task_acc interpolated (linear, like
    # tricontourf) from the retrained mean at the 69 retrained budgets and the search seed at the others.
    _sx, _sy = np.log10(moo.rate_budget.astype(float)), np.log10(moo.conn_budget.astype(float))
    _retr = np.zeros(len(moo), bool)
    for _r in honest.itertuples():
        _retr |= np.isclose(moo.rate_budget, _r.rate_budget, rtol=1e-9) & np.isclose(moo.conn_budget, _r.conn_budget, rtol=1e-9)
    _bx = np.r_[_sx[~_retr], np.log10(honest.rate_budget.astype(float))]
    _by = np.r_[_sy[~_retr], np.log10(honest.conn_budget.astype(float))]
    _bz = np.r_[moo.min_task_acc[~_retr], honest.min_task_acc]
    _xr, _yr = (_bx.min(), _bx.max()), (_by.min(), _by.max())
    _gx, _gy = np.linspace(*_xr, 200), np.linspace(*_yr, 200)
    _Z = griddata((_bx, _by), _bz, tuple(np.meshgrid(_gx, _gy)), method="linear")

    # Saved networks: seeds of one budget on a small ring around it, so each one can be clicked.
    # The controls have no budget: a ring right of the plane.
    _dx, _dy = 0.025 * (_xr[1] - _xr[0]), 0.025 * (_yr[1] - _yr[0])
    _ctrl_xy = (_xr[1] + 0.12 * (_xr[1] - _xr[0]), _yr[1] - 0.1 * (_yr[1] - _yr[0]))
    _px, _py = np.full(len(nets), np.nan), np.full(len(nets), np.nan)
    for _p, _g in nets.groupby("point"):
        if _p == "control":
            _cx, _cy = _ctrl_xy
        else:
            _cx, _cy = np.log10(_g.rate_budget.iloc[0]), np.log10(_g.conn_budget.iloc[0])
        _a = 2 * np.pi * np.arange(len(_g)) / len(_g)
        _rad = 1.0 if len(_g) <= 4 else 1.8
        _px[_g.index] = _cx + _rad * _dx * np.cos(_a) * (len(_g) > 1)
        _py[_g.index] = _cy + _rad * _dy * np.sin(_a) * (len(_g) > 1)

    # the 5 iso-cost lines, through their 6 budgets (no hover, so they never take a click)
    _iso = lines.groupby("point")[["rate_budget", "conn_budget"]].first()
    _iso_tr = [go.Scatter(x=np.log10(_b.rate_budget), y=np.log10(_b.conn_budget), mode="lines", hoverinfo="skip",
                          line=dict(color="#1f4e79", width=1, dash="dot"), legendgroup="iso",
                          showlegend=(_ln == "L0"), name="iso-cost lines L0..L4")
               for _ln, _b in _iso.groupby(_iso.index.str[:2])]
    _hf = honest[honest.is_pareto]
    _fig = go.Figure([
        go.Contour(x=_gx, y=_gy, z=_Z, colorscale="RdYlGn", zmin=0, zmax=1,
                   contours=dict(start=0, end=1, size=0.05), line=dict(width=0), hoverinfo="skip",
                   colorbar=dict(title="min_task_acc", len=0.9)),
        go.Contour(x=_gx, y=_gy, z=_Z, showscale=False, hoverinfo="skip", showlegend=True,
                   name=f"min_task_acc = {FEASIBLE_MIN_TASK_ACC}",
                   contours=dict(coloring="none", start=FEASIBLE_MIN_TASK_ACC, end=FEASIBLE_MIN_TASK_ACC, size=1),
                   line=dict(color="black", width=3)),
        *_iso_tr,
        go.Scatter(x=_sx, y=_sy, mode="markers", name="search network (1 seed, no weights)",
                   marker=dict(size=4, color="black", opacity=0.35), hoverinfo="skip"),
        go.Scatter(x=np.log10(_hf.rate_budget), y=np.log10(_hf.conn_budget), mode="markers",
                   name="honest front budget", hoverinfo="skip",
                   marker=dict(size=34, color="rgba(0,0,0,0)", line=dict(color="#1f4e79", width=2))),
        # mode includes "lines" (width 0): marimo only forwards plotly clicks for such scatter traces
        go.Scatter(x=_px, y=_py, mode="markers+lines", line=dict(width=0), name="saved network (click)",
                   marker=dict(size=9, color=nets.min_task_acc, colorscale="RdYlGn", cmin=0, cmax=1,
                               symbol=["circle" if ok else "x" for ok in nets.is_feasible],
                               line=dict(color="black", width=1)),
                   hovertext=nets.label, hoverinfo="text"),
    ])
    _fig.add_annotation(x=_ctrl_xy[0], y=_ctrl_xy[1] + 3 * _dy, text="control<br>(no budget)", showarrow=False,
                        font=dict(size=11))
    _fig.update_layout(width=860, height=760, margin=dict(l=60, r=20, t=40, b=60), dragmode="select",
                       title="click a network (x = fails a task; blue ring = honest front; bold = 0.6)",
                       xaxis=dict(title="log10 rate_budget", range=[_xr[0] - 3 * _dx, _ctrl_xy[0] + 3 * _dx]),
                       yaxis=dict(title="log10 conn_budget", range=[_yr[0] - 3 * _dy, _yr[1] + 3 * _dy]),
                       legend=dict(orientation="h", y=-0.12), plot_bgcolor="white")
    tv_plane = mo.ui.plotly(_fig)
    tv_pick = mo.ui.dropdown(options=dict(zip(nets.label, nets.index)), value=nets.label.iloc[0],
                             label="network", searchable=True)
    tv_xy = np.c_[_px, _py]
    mo.vstack([tv_pick, tv_plane])
    return tv_pick, tv_plane, tv_xy


@app.cell
def _(np):
    import functools
    import matplotlib.pyplot as plt
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    TV_K_RANGE = range(2, 21)

    @functools.lru_cache(maxsize=256)
    def tv_cluster(ckpt, n_null=3):
        """Yang-style clustering of one saved network; k = argmax silhouette, plus a shuffled null per k."""
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        tv = np.asarray(ck["task_variance"], dtype=float)                 # (tasks, units)
        n_e = int(round(ck["model_kwargs"]["frac_e"] * tv.shape[1]))
        total = tv.sum(axis=0)
        active = total > 1e-2 * total.max()
        norm = tv[:, active] / tv[:, active].max(axis=0, keepdims=True)
        X = norm.T
        rng = np.random.default_rng(0)
        nulls = [np.array([rng.permutation(row) for row in X]) for _ in range(n_null)]
        scores, null, labs = {}, {}, {}
        for k in TV_K_RANGE:
            if k >= len(X):
                break
            labs[k] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
            scores[k] = silhouette_score(X, labs[k])
            null[k] = [silhouette_score(Xs, KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(Xs))
                       for Xs in nulls]
        k = max(scores, key=scores.get)
        lab = labs[k]
        # order clusters by preferred task so plots are comparable across networks
        pref = {c: int(np.argmax(norm[:, lab == c].mean(axis=1))) for c in range(k)}
        order = sorted(range(k), key=lambda c: (pref[c], -norm[pref[c], lab == c].mean()))
        remap = {c: i for i, c in enumerate(order)}
        labels = np.full(tv.shape[1], -1)
        labels[active] = [remap[c] for c in lab]
        return dict(tasks=list(ck["active_tasks"]), n_e=n_e, active=active, labels=labels, k=k,
                    norm_full=tv / np.maximum(tv.max(axis=0, keepdims=True), 1e-30),
                    scores=scores, null=null)

    W_BLOCKS = {"E→E": ("E", "E"), "E→I": ("E", "I"), "I→E": ("I", "E"), "I→I": ("I", "I")}
    W_THRESH = 1e-2     # present synapse: E/I-normalised |w| above this (the conn_frac threshold)

    @functools.lru_cache(maxsize=256)
    def w_blocks(ckpt):
        """|w| of present synapses per block (W[pre, post]), trained and at init (rebuilt from the seed)."""
        import contextlib, io
        from cmc.task import default_config
        from cmc.train_cog import load_checkpoint, make_dale_model
        with contextlib.redirect_stdout(io.StringIO()):
            model, _cfg, ck = load_checkpoint(ckpt)
            init = make_dale_model(default_config(n_eachring=ck["config"].get("n_eachring", 16), seed=0),
                                   **ck["model_kwargs"])
        n_e = model.n_e
        sl = {"E": slice(0, n_e), "I": slice(n_e, None)}
        scale = model.ei_row_scale().detach().numpy()[:, None]
        mask = model.no_autapse.numpy() > 0
        out = {}
        for tag, m in (("trained", model), ("init", init)):
            W = m.effective_w_rec().detach().numpy()
            present = (np.abs(W) / scale > W_THRESH) & mask
            for b, (pre, post) in W_BLOCKS.items():
                p = present[sl[pre], sl[post]]
                out[tag, b] = dict(vals=np.abs(W[sl[pre], sl[post]])[p], density=p.sum() / mask[sl[pre], sl[post]].sum())
            out[tag, "W"], out[tag, "present"] = W, present
        return out

    def cluster_conn(W, present, labels, n_e, n_shuffle=500, seed=0):
        """W[pre, post] reordered by (E/I, task-variance cluster), plus block statistics vs a label shuffle.

        Unit order: E units, then I units; inside each, active units by cluster (as in the task-variance
        heatmap), inactive units (label -1) last. Block statistic = mean |w| over present synapses and
        pruned zeros, from group A (pre) to group B (post), self-connections excluded. Null: cluster labels
        permuted among active E units and, separately, among active I units (sizes and E/I kept).
        """
        N = len(labels)
        isI = np.arange(N) >= n_e
        order = np.lexsort((np.arange(N), labels < 0, labels, isI))
        A = np.where(present, np.abs(W), 0.0)
        act = np.flatnonzero(labels >= 0)
        Aa, lab_a, isI_a = A[np.ix_(act, act)], labels[act], isI[act]
        groups = sorted({(int(t), int(c)) for t, c in zip(isI_a, lab_a)})
        gid = {g: i for i, g in enumerate(groups)}

        def block_mean(lab):
            G = np.zeros((len(act), len(groups)))
            G[np.arange(len(act)), [gid[(int(t), int(c))] for t, c in zip(isI_a, lab)]] = 1
            n = G.sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                return (G.T @ Aa @ G) / (np.outer(n, n) - np.diag(n))

        def ee_ratio(lab):
            e = ~isI_a
            same = lab[e][:, None] == lab[e][None, :]
            off = ~np.eye(e.sum(), dtype=bool)
            Ae = Aa[np.ix_(e, e)]
            return Ae[same & off].mean() / Ae[~same].mean()

        obs, R = block_mean(lab_a), ee_ratio(lab_a)
        rng = np.random.default_rng(seed)
        nulls, R_null = [], []
        for _ in range(n_shuffle):
            lab = lab_a.copy()
            for m in (~isI_a, isI_a):
                lab[m] = rng.permutation(lab[m])
            nulls.append(block_mean(lab))
            R_null.append(ee_ratio(lab))
        nulls = np.array(nulls)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (obs - nulls.mean(0)) / nulls.std(0)
        return dict(order=order, A=A, groups=groups, obs=obs, z=z, R=R, R_null=np.array(R_null))

    @functools.lru_cache(maxsize=256)
    def conn_communities(ckpt, n_null=10, seed=0):
        """Communities from the connectivity alone (Louvain, max modularity Q) on |W| made symmetric.

        Null: the weights shuffled inside each Dale block (E→E, E→I, I→E, I→I), which keeps density,
        weight distribution and E/I strengths and destroys any grouping; Q of each shuffle, same method.
        ARI = agreement of the communities with the task-variance clusters (active units only).
        """
        import networkx as nx
        from networkx.algorithms.community import louvain_communities, modularity
        from sklearn.metrics import adjusted_rand_score
        wb, cl = w_blocks(ckpt), tv_cluster(ckpt)
        A = np.where(wb["trained", "present"], np.abs(wb["trained", "W"]), 0.0)
        N, n_e = len(A), cl["n_e"]

        def q_of(M):
            G = nx.from_numpy_array((M + M.T) / 2)
            comms = louvain_communities(G, weight="weight", seed=seed)
            return modularity(G, comms, weight="weight"), comms

        Q, comms = q_of(A)
        comms = sorted(comms, key=len, reverse=True)
        comm = np.empty(N, int)
        for i, c in enumerate(comms):
            comm[list(c)] = i
        rng = np.random.default_rng(seed)
        off = ~np.eye(N, dtype=bool)
        blocks = [(slice(0, n_e), slice(0, n_e)), (slice(0, n_e), slice(n_e, N)),
                  (slice(n_e, N), slice(0, n_e)), (slice(n_e, N), slice(n_e, N))]
        Q_null = []
        for _ in range(n_null):
            M = A.copy()
            for r, c in blocks:
                sub, m = M[r, c], off[r, c]
                sub[m] = rng.permutation(sub[m])
            Q_null.append(q_of(M)[0])
        act = cl["labels"] >= 0
        return dict(comm=comm, n_comm=len(comms), Q=Q, Q_null=np.array(Q_null),
                    ari=adjusted_rand_score(cl["labels"][act], comm[act]))

    return W_BLOCKS, cluster_conn, conn_communities, plt, tv_cluster, w_blocks


@app.cell
def _(
    SIL_REF,
    W_BLOCKS,
    cluster_conn,
    conn_communities,
    mo,
    nets,
    np,
    plt,
    tv_cluster,
    tv_pick,
    tv_plane,
    tv_xy,
    w_blocks,
):
    # clicked / boxed saved networks, matched by position (the plane also holds the search dots)
    _sel = [int(np.argmin(np.hypot(*(tv_xy - [_p["x"], _p["y"]]).T))) for _p in tv_plane.value
            if "x" in _p and np.hypot(*(tv_xy - [_p["x"], _p["y"]]).T).min() < 1e-9]
    _i = _sel[0] if _sel else int(tv_pick.value)
    _net = nets.loc[_i]
    with mo.status.spinner(title=f"clustering {_net.label} ..."):
        _cl = tv_cluster(_net.ckpt)
        _wb = w_blocks(_net.ckpt)

    _ks = np.array(sorted(_cl["scores"]))
    _s = np.array([_cl["scores"][k] for k in _ks])
    _nm = np.array([np.mean(_cl["null"][k]) for k in _ks])
    _nlo = np.array([np.min(_cl["null"][k]) for k in _ks])
    _nhi = np.array([np.max(_cl["null"][k]) for k in _ks])
    _k = _cl["k"]
    _runner_up = _ks[np.argsort(_s)[-2]]
    _gap = _cl["scores"][_k] - _cl["scores"][_runner_up]
    _excess = _cl["scores"][_k] - np.mean(_cl["null"][_k])

    _fig, (_ax, _axs) = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw=dict(width_ratios=[3.2, 1]))
    _idx = np.flatnonzero(_cl["active"])
    _idx = _idx[np.lexsort((_idx >= _cl["n_e"], _cl["labels"][_idx]))]       # by cluster, E before I inside
    _im = _ax.imshow(_cl["norm_full"][:, _idx], cmap="hot", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    _nt = len(_cl["tasks"])
    _ax.set_yticks(range(_nt))
    _ax.set_yticklabels([rule_name[_t] for _t in _cl["tasks"]], fontsize=7)
    _ax.set_xticks([])
    _ax.set_xlabel("Clusters", fontsize=8, labelpad=14)
    _ax.set_title(f"{_net.label}\nUnits: {len(_idx)} active, k={_k}, silhouette {_cl['scores'][_k]:.3f}", fontsize=9)
    _labs = _cl["labels"][_idx]
    _colors = plt.cm.tab20(np.linspace(0, 1, 20))
    _y = _nt - 0.5 + 0.9
    for _c in range(_k):
        _pos = np.flatnonzero(_labs == _c)
        if len(_pos):
            _ax.plot([_pos[0] - 0.5, _pos[-1] + 0.5], [_y, _y], lw=4, solid_capstyle="butt",
                     color=_colors[_c % 20], clip_on=False)
            _ax.text(_pos.mean(), _y + 1.0, str(_c + 1), ha="center", va="top", fontsize=6)
    _ax.set_ylim(_nt - 0.5, -0.5)
    _cb = _fig.colorbar(_im, ax=_ax, ticks=[0, 1], fraction=0.015, pad=0.01)
    _cb.set_label("Normalized Task Variance", fontsize=7, labelpad=0)

    _axs.fill_between(_ks, _nlo, _nhi, color="grey", alpha=0.3, label="shuffled null (min-max)")
    _axs.plot(_ks, _nm, color="grey", lw=1)
    _axs.plot(_ks, _s, "o-", color="#2b6cb0", ms=4, label="network")
    _axs.axvline(_k, color="#d69e2e", ls="--", lw=1, label=f"chosen k = {_k}")
    _axs.axhline(SIL_REF, color="black", ls="--", lw=1, label=f"silhouette = {SIL_REF}")
    _axs.set_xlabel("k")
    _axs.set_ylabel("silhouette")
    _axs.set_xticks(_ks[::2])
    _axs.legend(fontsize=7)
    _axs.set_title(f"excess over null {_excess:+.3f}\nrunner-up k={_runner_up}, {_gap:.3f} lower", fontsize=9)
    _fig.tight_layout()

    # E/I connectivity: |w| of present synapses per block, trained vs this network's own init
    _wfig, _waxes = plt.subplots(1, 4, figsize=(18, 3.8))
    for _wax, _b in zip(_waxes, W_BLOCKS):
        _tr, _in = _wb["trained", _b], _wb["init", _b]
        _all = np.concatenate([_tr["vals"], _in["vals"]])
        if not len(_all):
            _wax.set_title(f"{_b}: no synapses")
            continue
        _bins = np.logspace(np.log10(np.percentile(_all, 0.1)), np.log10(_all.max()), 60)
        _wax.hist(_in["vals"], bins=_bins, density=True, histtype="step", lw=2.5, ls="--", color="#bbbbbb",
                  label=f"init ({_in['density']:.0%})")
        if len(_tr["vals"]):
            _wax.hist(_tr["vals"], bins=_bins, density=True, histtype="step", lw=1.8, color="#c05621",
                      label=f"trained ({_tr['density']:.1%})")
        _wax.set_xscale("log")
        _wax.set_title(_b)
        _wax.set_xlabel("|w|")
        _wax.legend(fontsize=8, title="present synapses", title_fontsize=7)
    _waxes[0].set_ylabel("density")
    _wfig.suptitle(f"{_net.label}: recurrent weights, W[pre, post], present = E/I-normalised |w| > 1e-2", fontsize=9)
    _wfig.tight_layout()

    # W_rec reordered by (E/I, task-variance cluster): left the matrix, right block mean |w| as z vs label shuffle
    from matplotlib.colors import SymLogNorm as _SymLog
    _cc = cluster_conn(_wb["trained", "W"], _wb["trained", "present"], _cl["labels"], _cl["n_e"])
    _o = _cc["order"]
    _Wo = np.where(_wb["trained", "present"], _wb["trained", "W"], 0.0)[np.ix_(_o, _o)]
    _nz = np.abs(_Wo[_Wo != 0])
    _mfig, (_max, _zax) = plt.subplots(1, 2, figsize=(18, 8.2), gridspec_kw=dict(width_ratios=[1, 1]))
    if len(_nz):
        _vmax = np.percentile(_nz, 99.5)
        _mim = _max.imshow(_Wo, cmap="RdBu_r", interpolation="nearest",
                           norm=_SymLog(linthresh=np.percentile(_nz, 5), vmin=-_vmax, vmax=_vmax))
        _mfig.colorbar(_mim, ax=_max, fraction=0.04, pad=0.01, label="w (red E, blue I; symlog)")
    # group boundaries and labels along both axes
    _lo = _cl["labels"][_o]
    _isI = _o >= _cl["n_e"]
    _key = [(int(_t), int(_c)) for _t, _c in zip(_isI, _lo)]
    _starts = [0] + [_j for _j in range(1, len(_key)) if _key[_j] != _key[_j - 1]]
    _ticks, _names = [], []
    for _a, _b2 in zip(_starts, _starts[1:] + [len(_key)]):
        _t, _c = _key[_a]
        _ticks.append((_a + _b2 - 1) / 2)
        _names.append(("I" if _t else "E") + ("–" if _c < 0 else str(_c + 1)))
        if _a:
            _ei = _key[_a][0] != _key[_a - 1][0]
            for _f in (_max.axhline, _max.axvline):
                _f(_a - 0.5, color="black", lw=1.6 if _ei else 0.8, alpha=1.0 if _ei else 0.6)
    for _setT, _setL in ((_max.set_xticks, _max.set_xticklabels), (_max.set_yticks, _max.set_yticklabels)):
        _setT(_ticks)
        _setL(_names, fontsize=6)
    _max.tick_params(axis="x", rotation=90)
    _max.set_xlabel("postsynaptic (to)")
    _max.set_ylabel("presynaptic (from)")
    _max.set_title("W_rec[pre, post] ordered by E/I, then task-variance cluster (E3 = E units of cluster 3,\n"
                   "E– = inactive E units); present synapses only", fontsize=9)

    _gn = [("I" if _t else "E") + str(_c + 1) for _t, _c in _cc["groups"]]
    _zim = _zax.imshow(_cc["z"], cmap="RdBu_r", vmin=-4, vmax=4, interpolation="nearest")
    _zax.set_xticks(range(len(_gn)))
    _zax.set_xticklabels(_gn, rotation=90, fontsize=7)
    _zax.set_yticks(range(len(_gn)))
    _zax.set_yticklabels(_gn, fontsize=7)
    if len(_gn) <= 24:
        for (_r, _c), _v in np.ndenumerate(_cc["z"]):
            if np.isfinite(_v) and abs(_v) >= 2:
                _zax.text(_c, _r, f"{_v:.0f}", ha="center", va="center", fontsize=6)
    _nE = sum(1 for _t, _ in _cc["groups"] if not _t)
    for _f in (_zax.axhline, _zax.axvline):
        _f(_nE - 0.5, color="black", lw=1.4)
    _mfig.colorbar(_zim, ax=_zax, fraction=0.04, pad=0.01, label="z vs label shuffle")
    _p = (np.sum(_cc["R_null"] >= _cc["R"]) + 1) / (len(_cc["R_null"]) + 1)
    _zax.set_xlabel("postsynaptic group")
    _zax.set_ylabel("presynaptic group")
    _zax.set_title(f"block mean |w| (pruned = 0), z-scored against {len(_cc['R_null'])} label shuffles\n"
                   f"E→E within / between cluster = {_cc['R']:.2f} (shuffle {np.mean(_cc['R_null']):.2f}"
                   f" ± {np.std(_cc['R_null']):.2f}, p = {_p:.3f})", fontsize=9)
    _mfig.tight_layout()

    # communities from the connectivity alone (Louvain on |W|), matrix shown signed, ordered E/I then community
    with mo.status.spinner(title="finding connectivity communities ..."):
        _cm = conn_communities(_net.ckpt)
    _N = len(_cm["comm"])
    _o2 = np.lexsort((np.arange(_N), _cm["comm"], np.arange(_N) >= _cl["n_e"]))
    _W2 = np.where(_wb["trained", "present"], _wb["trained", "W"], 0.0)[np.ix_(_o2, _o2)]
    _cfig, (_cax, _tax) = plt.subplots(1, 2, figsize=(18, 8.2), gridspec_kw=dict(width_ratios=[1, 0.8]))
    if len(_nz):
        _cim = _cax.imshow(_W2, cmap="RdBu_r", interpolation="nearest",
                           norm=_SymLog(linthresh=np.percentile(_nz, 5), vmin=-_vmax, vmax=_vmax))
        _cfig.colorbar(_cim, ax=_cax, fraction=0.04, pad=0.01, label="w (red E, blue I; symlog)")
    _k2 = [(int(_t), int(_c)) for _t, _c in zip(_o2 >= _cl["n_e"], _cm["comm"][_o2])]
    _st2 = [0] + [_j for _j in range(1, _N) if _k2[_j] != _k2[_j - 1]]
    _tk2, _nm2 = [], []
    for _a, _b2 in zip(_st2, _st2[1:] + [_N]):
        _tk2.append((_a + _b2 - 1) / 2)
        _nm2.append(("I" if _k2[_a][0] else "E") + f"m{_k2[_a][1] + 1}")
        if _a:
            _ei = _k2[_a][0] != _k2[_a - 1][0]
            for _f in (_cax.axhline, _cax.axvline):
                _f(_a - 0.5, color="black", lw=1.6 if _ei else 0.8, alpha=1.0 if _ei else 0.6)
    if len(_tk2) <= 60:
        for _setT, _setL in ((_cax.set_xticks, _cax.set_xticklabels), (_cax.set_yticks, _cax.set_yticklabels)):
            _setT(_tk2)
            _setL(_nm2, fontsize=6)
        _cax.tick_params(axis="x", rotation=90)
    _cax.set_xlabel("postsynaptic (to)")
    _cax.set_ylabel("presynaptic (from)")
    _cax.set_title(f"W_rec ordered by E/I, then connectivity community (Em2 = E units of module 2)\n"
                   f"Louvain on |W|: {_cm['n_comm']} modules, Q = {_cm['Q']:.3f} "
                   f"(shuffled weights {_cm['Q_null'].mean():.3f} ± {_cm['Q_null'].std():.3f})", fontsize=9)
    # do the modules match the task-variance clusters? units per (TV cluster, module), active units
    _act = _cl["labels"] >= 0
    _ct = np.zeros((_cl["k"], _cm["n_comm"]), int)
    np.add.at(_ct, (_cl["labels"][_act], _cm["comm"][_act]), 1)
    _ct = _ct[:, _ct.sum(0) > 0]
    _tax.imshow(_ct / np.maximum(_ct.sum(1, keepdims=True), 1), cmap="Greys", vmin=0, vmax=1, aspect="auto")
    if _ct.size <= 400:
        for (_r, _c), _v in np.ndenumerate(_ct):
            if _v:
                _tax.text(_c, _r, str(_v), ha="center", va="center", fontsize=6, color="tab:red")
    _tax.set_yticks(range(_cl["k"]))
    _tax.set_yticklabels([f"TV cluster {_c + 1}" for _c in range(_cl["k"])], fontsize=7)
    _tax.set_xlabel("connectivity module (with active units)")
    _tax.set_title(f"active units per (task-variance cluster, connectivity module); shade = row fraction\n"
                   f"agreement ARI = {_cm['ari']:.2f} (0 = unrelated, 1 = identical)", fontsize=9)
    _cfig.tight_layout()

    # per-task accuracy of this network, as in cog20_zoom_analysis section 0 (same colormap and range)
    _acc = np.array([[_net[f"acc_{_t}"] for _t in _cl["tasks"]]], dtype=float)
    _afig, _aax = plt.subplots(figsize=(18, 1.9))
    _aim = _aax.imshow(_acc, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    _aax.set_xticks(range(_nt))
    _aax.set_xticklabels([rule_name[_t] for _t in _cl["tasks"]], rotation=60, ha="right", fontsize=8)
    _aax.set_yticks([])
    for _j, _v in enumerate(_acc[0]):
        _aax.text(_j, 0, f"{_v:.2f}", ha="center", va="center", fontsize=8,
                  weight="bold" if _v == _acc.min() else "normal")
    _afig.colorbar(_aim, ax=_aax, label="accuracy", fraction=0.02, pad=0.01)
    _aax.set_title(f"{_net.label}: per-task accuracy (min {_acc.min():.2f} bold, mean {_acc.mean():.2f})", fontsize=9)
    _afig.tight_layout()

    mo.vstack([
        mo.md(f"*{len(_sel)} networks in the box, showing the first; use the dropdown for the others.*"
              if len(_sel) > 1 else ""),
        mo.md(f"**{_net.label}** · budgets rate {_net.rate_budget:.4g}, wiring {_net.conn_budget:.4g} · "
              f"metabolic {_net.metabolic_cost:.4g}, wiring {_net.wiring_cost:.4g}, conn_frac {_net.conn_frac:.3f}"
              if _net.point != "control" else f"**{_net.label}** · no budget · metabolic "
              f"{_net.metabolic_cost:.4g}, wiring {_net.wiring_cost:.4g}, conn_frac {_net.conn_frac:.3f}"),
        _fig,
        _wfig,
        _mfig,
        _cfig,
        _afig,
    ])
    return


if __name__ == "__main__":
    app.run()
