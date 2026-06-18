"""Modern README figures (Plotly -> static PNG).

Two figures:
  1. pareto.png   — ratio vs speed, two panels (compress | decompress).
                    Top-right is best; shows quoin Pareto-dominating.
  2. ratio_breadth.png — compression ratio across real f64 columns.
"""
import sys
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/quoin_bench.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/okhsunrog/tmp_zfs/quoin/docs/images"
df = pd.read_csv(CSV)

# Brand palette. quoin = green family (the hero); baselines = muted.
COL = {
    "quoin-balanced": "#10b981",
    "quoin-high":     "#059669",
    "quoin-max":      "#065f46",
    "lz4":            "#94a3b8",
    "zlib-6":         "#f472b6",
    "zstd-3":         "#fbbf24",
    "zstd-19":        "#fb923c",
}
NAME = {
    "quoin-balanced": "quoin · Balanced",
    "quoin-high":     "quoin · High",
    "quoin-max":      "quoin · Max",
    "lz4": "lz4", "zlib-6": "zlib −6", "zstd-3": "zstd −3", "zstd-19": "zstd −19",
}
ORDER = ["lz4", "zlib-6", "zstd-3", "zstd-19", "quoin-balanced", "quoin-high", "quoin-max"]
FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=15, color="#1e293b")

# Plain-number tick labels on a log axis (no 10^3).
def log_ticks(lo, hi):
    vals, base = [], 1
    while base <= hi * 10:
        for m in (1, 2, 5):
            v = base * m
            if lo / 1.5 <= v <= hi * 1.5:
                vals.append(v)
        base *= 10
    def fmt(v):
        return f"{v/1000:g}k" if v >= 1000 else f"{v:g}"
    return vals, [fmt(v) for v in vals]


# ----------------------------------------------------------------------------
# Figure 1 — Pareto: ratio (x) vs throughput (y), one point per codec.
# Uses the full 9.9M-value real column (no cache effects).
# ----------------------------------------------------------------------------
vol = df[(df.section == "volume") & (df.n == df[df.section == "volume"].n.max())]

fig = make_subplots(
    rows=1, cols=2, horizontal_spacing=0.13,
    subplot_titles=("<b>Compression speed</b> vs ratio",
                    "<b>Decompression speed</b> vs ratio"),
)

PARETO = [c for c in ORDER if c != "quoin-high"]  # High ~ Max overlaps; keep table-only
for col_i, metric in enumerate(["enc_mbps", "dec_mbps"], start=1):
    for codec in PARETO:
        r = vol[vol.codec == codec]
        if r.empty:
            continue
        ratio = float(r.ratio.iloc[0])
        speed = float(r[metric].iloc[0])
        is_q = codec.startswith("quoin")
        fig.add_trace(go.Scatter(
            x=[ratio], y=[speed], mode="markers+text",
            text=[f" {NAME[codec]}"], textposition="middle right",
            textfont=dict(size=12.5, color=COL[codec],
                          family="Inter, Segoe UI, sans-serif"),
            marker=dict(size=20 if is_q else 14, color=COL[codec],
                        line=dict(width=1.5, color="white"),
                        symbol="circle" if is_q else "diamond"),
            showlegend=False, cliponaxis=False,
            hovertemplate=f"{NAME[codec]}<br>ratio %{{x:.2f}}×<br>%{{y:.0f}} MB/s<extra></extra>",
        ), row=1, col=col_i)

    speeds = vol[metric]
    tv, tt = log_ticks(speeds.min(), speeds.max())
    fig.update_yaxes(type="log", tickvals=tv, ticktext=tt, row=1, col=col_i,
                     title_text="throughput  (MB/s, higher ↑ better)" if col_i == 1 else "",
                     gridcolor="#eef2f7", zeroline=False)
    fig.update_xaxes(row=1, col=col_i, title_text="compression ratio  (higher → better)",
                     range=[1.1, 3.05], gridcolor="#eef2f7", zeroline=False)

fig.update_layout(
    template="plotly_white", font=FONT,
    title=dict(text="<b>quoin sits in the top-right: better ratio AND faster</b>"
                    "<br><sup>real 9.9 M-value f64 column · Intel Core Ultra 5 125H</sup>",
               x=0.5, xanchor="center", font=dict(size=20)),
    width=1280, height=560, margin=dict(t=110, b=70, l=80, r=40),
    plot_bgcolor="white", paper_bgcolor="white",
)
fig.write_image(f"{OUT}/pareto.png", scale=2)
print("wrote pareto.png")


# ----------------------------------------------------------------------------
# Figure 2 — Ratio across real columns (grouped bars), sorted by quoin-Max.
# ----------------------------------------------------------------------------
breadth = df[~df.section.isin(["volume"])].copy()
datasets = list(dict.fromkeys(breadth.section))
# sort datasets by quoin-max ratio ascending for a clean staircase
def qmax(d):
    s = breadth[(breadth.section == d) & (breadth.codec == "quoin-max")]
    return float(s.ratio.iloc[0]) if not s.empty else 0
datasets.sort(key=qmax)
labels = [d.replace("_f", "").replace("_", " ") for d in datasets]
bars = ["lz4", "zstd-3", "zstd-19", "quoin-balanced", "quoin-max"]

fig2 = go.Figure()
for codec in bars:
    vals = []
    for d in datasets:
        s = breadth[(breadth.section == d) & (breadth.codec == codec)]
        vals.append(float(s.ratio.iloc[0]) if not s.empty else None)
    fig2.add_trace(go.Bar(
        name=NAME[codec], x=labels, y=vals, marker_color=COL[codec],
        marker_line_width=0,
        hovertemplate=f"{NAME[codec]}<br>%{{x}}<br>%{{y:.2f}}×<extra></extra>",
    ))

fig2.update_layout(
    template="plotly_white", font=FONT, barmode="group", bargap=0.28, bargroupgap=0.08,
    title=dict(text="<b>Compression ratio across real f64 columns</b>"
                    "<br><sup>ALP benchmark corpus · higher is better</sup>",
               x=0.5, xanchor="center", font=dict(size=20)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1280, height=600, margin=dict(t=120, b=90, l=70, r=30),
    yaxis=dict(title="compression ratio  ×", gridcolor="#eef2f7", zeroline=False),
    xaxis=dict(tickangle=-20),
    plot_bgcolor="white", paper_bgcolor="white",
)
fig2.write_image(f"{OUT}/ratio_breadth.png", scale=2)
print("wrote ratio_breadth.png")


# ----------------------------------------------------------------------------
# Figure 3 — Integer & Decimal lanes (typed CSV from bench_typed.rs).
# Two panels: real ClickBench integer columns | real-derived Decimal128 columns.
# Grouped bars, linear ratio. Excludes the incompressible / outlier columns
# (WatchID ~1×, CounterID ~19000×) — those live in the table instead.
# ----------------------------------------------------------------------------
TYPED = sys.argv[3] if len(sys.argv) > 3 else "/tmp/quoin_typed.csv"
try:
    td = pd.read_csv(TYPED)
except FileNotFoundError:
    td = None

if td is not None:
    int_cols = ["EventTime", "UserID", "RegionID", "IPNetworkID"]
    dec_cols = ["food_prices", "city_temperature", "bitcoin_tx"]
    bars = ["lz4", "zstd-3", "zstd-19", "quoin-balanced", "quoin-max"]
    fig3 = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.09,
        subplot_titles=("<b>Integer columns</b>  ·  real ClickBench i64 / i32",
                        "<b>Decimal128 columns</b>  ·  real values as fixed-point"),
    )
    def add_panel(cols, col_i, showleg):
        labels = [c.replace("_", " ") for c in cols]
        for codec in bars:
            vals = []
            for d in cols:
                s = td[(td.section == d) & (td.codec == codec)]
                vals.append(float(s.ratio.iloc[0]) if not s.empty else None)
            fig3.add_trace(go.Bar(
                name=NAME[codec], x=labels, y=vals, marker_color=COL[codec],
                marker_line_width=0, legendgroup=codec, showlegend=showleg,
                hovertemplate=f"{NAME[codec]}<br>%{{x}}<br>%{{y:.1f}}×<extra></extra>",
            ), row=1, col=col_i)
    add_panel(int_cols, 1, True)
    add_panel(dec_cols, 2, False)
    fig3.update_layout(
        template="plotly_white", font=FONT, barmode="group", bargap=0.3, bargroupgap=0.06,
        title=dict(text="<b>Type-aware lanes: ratio on integers & decimals</b>"
                        "<br><sup>where quoin's specialization pays off most · higher is better</sup>",
                   x=0.5, xanchor="center", font=dict(size=20)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1280, height=560, margin=dict(t=120, b=70, l=60, r=30),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig3.update_yaxes(title_text="compression ratio  ×", gridcolor="#eef2f7", zeroline=False, row=1, col=1)
    fig3.update_yaxes(gridcolor="#eef2f7", zeroline=False, row=1, col=2)
    fig3.write_image(f"{OUT}/typed_ratio.png", scale=2)
    print("wrote typed_ratio.png")
