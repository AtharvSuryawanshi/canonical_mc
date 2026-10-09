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

    ROOT = Path(__file__).resolve().parent.parent
    FRONT_DIR = ROOT / "runs/moo/dale_all_nsga3_budget_p48g3_50k"          # NSGA-III cog20 search, 1 seed/budget
    DENSE_DIR = ROOT / "runs/moo_zoom/dale_all_front27_2seed_50k"          # its 27 front budgets x 2 new seeds
    ZOOM4_DIR = ROOT / "runs/moo_zoom/dale_all_4pt_12seed_50k"             # 3 of them + control x 12 new seeds
    OBJ = ("min_task_acc", "metabolic_cost", "wiring_cost")
    return (DENSE_DIR, FEASIBLE_MIN_TASK_ACC, FRONT_DIR, OBJ, ZOOM4_DIR, compute_front, go, mo, np, pd)


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
    * **honest front**: the non-dominated budgets among the 27, using the **mean over retrained seeds only**
      (the search's own seed is left out: it was selected for being good).

    Surface options: the **attainment surface** is what a Pareto front is: at each (metabolic, wiring) cost,
    the best min_task_acc reached by any network that costs no more. It is a monotone staircase and invents
    nothing between points. The triangulated mesh (the old view) stretches facets across gaps.
    """)
    return


@app.cell
def _(DENSE_DIR, FEASIBLE_MIN_TASK_ACC, FRONT_DIR, OBJ, ZOOM4_DIR, compute_front, np, pd):
    moo = pd.read_csv(FRONT_DIR / "runs.csv").drop_duplicates(["x0", "x1"]).reset_index(drop=True)
    moo["is_feasible"] = moo.min_task_acc >= FEASIBLE_MIN_TASK_ACC
    moo["is_pareto"], _ = compute_front(moo.to_dict("records"), OBJ)
    dense = pd.read_csv(DENSE_DIR / "runs.csv")
    zoom4 = pd.read_csv(ZOOM4_DIR / "runs.csv")
    for _d in (dense, zoom4):
        _d["is_feasible"] = _d.min_task_acc >= FEASIBLE_MIN_TASK_ACC

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
    retrained = pd.concat([_d, _z.dropna(subset=["point"])], ignore_index=True)
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
    return controls, dense, honest, moo, retrained, zoom4


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
    return show_background, show_lines, show_seeds, surface_kind, surface_opacity


@app.cell
def _(FEASIBLE_MIN_TASK_ACC, controls, go, honest, moo, np, pd, retrained, show_background, show_lines,
      show_seeds, surface_kind, surface_opacity):
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
                         f"<br>search seed: {r.search_min_task_acc:.3f} (inflation {r.inflation:+.3f})")
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
        ## The 27 budgets: search seed vs retrained mean

        `search_min_task_acc` is the single seed the search kept; `min_task_acc` is the mean over the retrained
        seeds. Inflation = search − retrained (mean **{honest.inflation.mean():+.3f}**). `is_pareto` = on the
        honest front ({int(honest.is_pareto.sum())} of {len(honest)}).
        """),
        mo.ui.table(_t, selection=None, page_size=30),
    ])
    return


if __name__ == "__main__":
    app.run()
