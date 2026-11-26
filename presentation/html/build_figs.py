"""Optional interactive extras for the deck appendix (slides.qmd), from the same data as the PDF figures.

  _front3d.qmd            rotatable 3D front (4 NSGA-III searches pooled), hover = one network; the 4 points as stars
  _auc.qmd                like-to-like AUC table, 4 connection types x 4 points (mean over 5 seeds), colour-coded
  assets/plotly.min.js    plotly.js, loaded once by the deck (inlined by Quarto: works offline)
  assets/nsga_data.js     data for the NSGA-III animation (nsga_anim.js): generations 0 and 1 of the ms1 search
  assets/front_data.js    data for the rotating front (front_anim.js): 4 searches pooled, the 4 points, the surface
  assets/l2l_data.js      data for the like-to-like animation (l2l_anim.js): the slide-15 example network

    python presentation/html/build_figs.py [nsga] [front] [l2l]   (cmc_env, repo root; plotly; names = only those)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import colormaps, colors

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "results/scripts"))
from front_yang import POINTS, load_front, load_points  # noqa: E402
from talk_style import LABEL  # noqa: E402

OUT = HERE / "assets"
COL = {"control": "#4a4a4a", "firing": "#d62728", "wiring": "#1f77b4", "both": "#2ca02c"}
RDYLGN = colormaps["RdYlGn"]
FONT = dict(family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif", size=13, color="#1d1d1f")
CSCALE = [[i / 10, colors.to_hex(RDYLGN(i / 10))] for i in range(11)]


def hover(d):
    return [f"{r.run} · budgets rate {r.rate_budget:.3g}, synaptic {r.conn_budget:.3g}<br>"
            f"worst-task accuracy <b>{r.z:.2f}</b><br>rate cost {r.metabolic_cost:.3g} · synaptic cost "
            f"{r.wiring_cost:.3g} · {100 * r.conn_frac:.1f}% synapses kept" for r in d.itertuples()]


def write(fig, name, height):
    """The plot as an HTML div in _<name>.qmd, included straight into a slide. plotly.js itself is loaded once for
    the whole deck (assets/plotly.min.js, see slides.qmd); iframes don't work in the one-file deck (Chrome refuses
    data-URL frames over 2 MB)."""
    div = fig.to_html(full_html=False, include_plotlyjs=False, div_id=name, default_height=f"{height}px",
                      default_width="100%",
                      config=dict(displaylogo=False, modeBarButtonsToRemove=["toImage"], responsive=True))
    (HERE / f"_{name}.qmd").write_text("```{=html}\n" + div + "\n```\n", encoding="utf-8")
    print("->", HERE / f"_{name}.qmd")


def front3d(d, z):
    feas, par, inf = d[d.feasible & ~d.pareto], d[d.pareto], d[~d.feasible]
    fig = go.Figure()
    fig.add_mesh3d(x=par.x, y=par.y, z=par.z, intensity=par.z, colorscale=CSCALE, cmin=0, cmax=1, opacity=0.22,
                   delaunayaxis="z", showscale=False, hoverinfo="skip")
    fig.add_scatter3d(x=feas.x, y=feas.y, z=feas.z, mode="markers", name="network",
                      marker=dict(size=2.5, color="#9a9a9a", opacity=0.6), text=hover(feas), hoverinfo="text")
    fig.add_scatter3d(x=par.x, y=par.y, z=par.z, mode="markers", name="on the front",
                      marker=dict(size=4, color=par.z, colorscale=CSCALE, cmin=0, cmax=1,
                                  line=dict(color="#222", width=1)), text=hover(par), hoverinfo="text")
    fig.add_scatter3d(x=inf.x, y=inf.y, z=np.full(len(inf), 0.6), mode="markers", name="fails a task",
                      marker=dict(size=3, symbol="x", color="#E0393E"), text=hover(inf), hoverinfo="text")
    for p in POINTS:
        g = z[z.point == p]
        fig.add_scatter3d(x=[g.x.mean()], y=[g.y.mean()], z=[g.z.mean()], mode="markers", name=LABEL[p],
                          marker=dict(size=9, symbol="diamond", color=COL[p], line=dict(color="white", width=2)),
                          hovertemplate=f"<b>{LABEL[p]}</b><br>5 seeds, mean accuracy {g.z.mean():.2f}<extra></extra>")
    axis = dict(backgroundcolor="white", gridcolor="#e5e5e5", zerolinecolor="#e5e5e5", showspikes=False)
    fig.update_layout(font=FONT, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(x=0.0, y=0.98, bgcolor="rgba(255,255,255,0.8)", font=dict(size=12)),
                      scene=dict(xaxis=dict(title="rate cost (log₁₀)", **axis),
                                 yaxis=dict(title="synaptic cost (log₁₀)", **axis),
                                 zaxis=dict(title="worst-task accuracy", range=[0.6, 1.0], **axis),
                                 camera=dict(eye=dict(x=-1.6, y=-1.5, z=0.8)), aspectmode="manual",
                                 aspectratio=dict(x=1.25, y=1, z=0.75)))
    write(fig, "front3d", 510)


def auc_table():
    d = pd.read_csv(REPO / "results/cache/front/like2like_task.csv")
    t = d[d["mode"] == "trained"]
    m = t.groupby(["type", "point"]).auc.mean()
    sig = t.assign(s=t.p < 0.05).groupby(["type", "point"]).s.sum()
    div = colormaps["PuOr"]                                       # purple = like-to-like (> 0.5), orange = unlike
    rows = ""
    for q in ["E→E", "E→I", "I→E", "I→I"]:
        pre, post = q.split("→")
        cells = ""
        for p in POINTS:
            a = m[(q, p)]
            bg = colors.to_hex(div(0.5 + (a - 0.5) * 2.2))
            fg = "white" if abs(a - 0.5) > 0.1 else "#1d1d1f"
            cells += f'<td style="background:{bg};color:{fg}" title="{int(sig[(q, p)])}/5 seeds p &lt; 0.05">{a:.2f}</td>'
        rows += (f'<tr><th><span class="{pre}">{pre}</span> → <span class="{post}">{post}</span></th>{cells}</tr>')
    head = "".join(f'<th style="color:{COL[p]}">{LABEL[p]}</th>' for p in POINTS)
    html = f'<table class="auc"><tr><th></th>{head}</tr>{rows}</table>'
    (HERE / "_auc.qmd").write_text("```{=html}\n" + html + "\n```\n", encoding="utf-8")
    print("->", HERE / "_auc.qmd")


def nsga_data():
    """Generations 0 and 1 of the real ms1 search, as anim_nsga3.py uses them: budgets (before training), reached
    costs (after), worst-task accuracy and its RdYlGn colour, feasibility, front of gen 0 and of gens 0 + 1."""
    import json
    from anim_nsga3 import front1, load
    d = load()
    g0 = d[d.generation == 0].reset_index(drop=True)
    both = d[d.generation <= 1].reset_index(drop=True)
    f0, fb = front1(g0), front1(both)
    rows = []
    for i, r in both.iterrows():
        rows.append(dict(g=int(r.generation), bx=round(r.bx, 4), by=round(r.by, 4), rx=round(r.rx, 4),
                         ry=round(r.ry, 4), acc=round(r.acc, 4), col=colors.to_hex(RDYLGN(r.acc)),
                         ok=bool(r.acc >= 0.6), f0=bool(i < len(g0) and f0[i]), fb=bool(fb[i])))
    (OUT / "nsga_data.js").write_text("window.NSGA_DATA = " + json.dumps(rows, separators=(",", ":")) + ";\n",
                                      encoding="utf-8")
    print("->", OUT / "nsga_data.js", f"({len(rows)} networks, front gen 0: {int(f0.sum())}, gens 0+1: {int(fb.sum())})")


def front_data(n=44):
    """The pooled Yang front for front_anim.js: every network (log costs, worst-task accuracy, kind 0 = fails a task,
    1 = feasible, 2 = on the front), the 4 points (5-seed mean and seeds), the smoothed accuracy surface that the top
    view (front_yang.top) contours, on an n x n grid (null outside the hull), and the RdYlGn lookup table."""
    import json
    from front_yang import surface
    d, z = load_front(), load_points()
    kind = np.where(d.pareto, 2, np.where(d.feasible, 1, 0))
    nets = [[round(a, 4), round(b, 4), round(c, 4), int(k)] for a, b, c, k in zip(d.x, d.y, d.z, kind)]
    pts = []
    for p in POINTS:
        g = z[z.point == p]
        pts.append(dict(name=LABEL[p], col=COL[p], x=round(g.x.mean(), 4), y=round(g.y.mean(), 4),
                        z=round(g.z.mean(), 4), seeds=[[round(a, 4), round(b, 4), round(c, 4)]
                                                       for a, b, c in zip(g.x, g.y, g.z)]))
    xx, yy, zz = surface(d, n=n)
    grid = [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in zz]
    out = dict(nets=nets, points=pts, xs=[round(v, 4) for v in xx[0]], ys=[round(v, 4) for v in yy[:, 0]],
               zz=grid, lut=[colors.to_hex(RDYLGN(i / 100)) for i in range(101)],
               counts=dict(n=len(d), feasible=int(d.feasible.sum()), front=int(d.pareto.sum())))
    (OUT / "front_data.js").write_text("window.FRONT_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n",
                                       encoding="utf-8")
    print("->", OUT / "front_data.js", out["counts"], "x", d.x.min().round(2), d.x.max().round(2),
          "y", d.y.min().round(2), d.y.max().round(2), "z", d.z.min().round(2))


def l2l_data(run="front", point="both", seed=0):
    """Like-to-like on one network, as fig_like2like_schematic.py (E->E pairs, per-task profiles): the connected and
    the unconnected example pair (profiles, r), histograms of r for connected and unconnected pairs, and the ROC as a
    threshold on r sweeps from +1 to -1."""
    import json
    from fig_like2like_schematic import network, pick_pairs
    from fig_wiring_like2like import auc
    tasks, X, S, C, E = network(run, point, seed)
    pc, pu = pick_pairs(S, C, X)
    off = ~np.eye(len(S), dtype=bool)
    r, c = S[off], C[off]
    bins = np.linspace(-1, 1, 41)
    hc = np.histogram(r[c], bins)[0] / c.sum()
    hu = np.histogram(r[~c], bins)[0] / (~c).sum()
    th = np.linspace(1, -1, 201)
    out = dict(
        pairs=[dict(a=[round(v, 3) for v in X[:, i]], b=[round(v, 3) for v in X[:, j]], r=round(float(S[i, j]), 2),
                    conn=bool(k == 0)) for k, (i, j) in enumerate((pc, pu))],
        bins=[round(v, 3) for v in bins], hc=[round(v, 5) for v in hc], hu=[round(v, 5) for v in hu],
        th=[round(v, 3) for v in th], tpr=[round(float((r[c] >= t).mean()), 4) for t in th],
        fpr=[round(float((r[~c] >= t).mean()), 4) for t in th], auc=round(float(auc(r, c)), 2),
        n_conn=int(c.sum()), n_unconn=int((~c).sum()), n_e=len(E), label=f"{LABEL[point]}, seed {seed}")
    (OUT / "l2l_data.js").write_text("window.L2L_DATA = " + json.dumps(out, separators=(",", ":")) + ";\n",
                                     encoding="utf-8")
    print("->", OUT / "l2l_data.js", f"AUC {out['auc']}, {out['n_conn']} connected / {out['n_unconn']} unconnected "
          f"E->E pairs, examples r = {out['pairs'][0]['r']}, {out['pairs'][1]['r']}")


def main():
    only = sys.argv[1:]
    if only:
        for name in only:
            {"nsga": nsga_data, "front": front_data, "l2l": l2l_data}[name]()
        return
    from plotly.offline import get_plotlyjs
    OUT.mkdir(exist_ok=True)
    (OUT / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    for old in ("front3d.html", "front_top.html"):
        (OUT / old).unlink(missing_ok=True)
    d, z = load_front(), load_points()
    front3d(d, z)
    auc_table()
    nsga_data()
    front_data()
    l2l_data()


if __name__ == "__main__":
    main()
