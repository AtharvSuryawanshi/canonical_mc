"""Optional interactive extras for the deck appendix (slides.qmd), from the same data as the PDF figures.

  _front3d.qmd            rotatable 3D front (4 NSGA-III searches pooled), hover = one network; the 4 points as stars
  _auc.qmd                like-to-like AUC table, 4 connection types x 4 points (mean over 5 seeds), colour-coded
  assets/plotly.min.js    plotly.js, loaded once by the deck (inlined by Quarto: works offline)

    python presentation/html/build_figs.py        (cmc_env, repo root; plotly)
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


def main():
    from plotly.offline import get_plotlyjs
    OUT.mkdir(exist_ok=True)
    (OUT / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    for old in ("front3d.html", "front_top.html"):
        (OUT / old).unlink(missing_ok=True)
    d, z = load_front(), load_points()
    front3d(d, z)
    auc_table()


if __name__ == "__main__":
    main()
