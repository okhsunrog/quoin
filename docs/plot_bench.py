"""Modern README figures (Plotly -> static PNG).

Per-type ratio⇄speed Pareto charts (top-right = best), one per data type, plus
"ratio across many columns" bar charts.

  pareto_int.png      — real ClickBench i64 EventTime
  pareto_decimal.png  — city_temperature as Decimal128
  pareto_float.png    — real arade4 f64
  ratio_breadth.png   — f64 columns (ALP corpus)
  typed_ratio.png     — integer & decimal columns
"""
import sys
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BENCH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/quoin_bench.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/okhsunrog/tmp_zfs/quoin/docs/images"
TYPED = sys.argv[3] if len(sys.argv) > 3 else "/tmp/quoin_typed.csv"
df = pd.read_csv(BENCH)
td = pd.read_csv(TYPED)

COL = {
    "quoin-fastest": "#6ee7b7", "quoin-fast": "#34d399", "quoin-balanced": "#10b981",
    "quoin-high": "#047857", "quoin-max": "#064e3b",
    "lz4": "#94a3b8", "zlib-6": "#f472b6", "zstd-3": "#fbbf24", "zstd-19": "#fb923c",
}
NAME = {
    "quoin-fastest": "quoin · Fastest", "quoin-fast": "quoin · Fast",
    "quoin-balanced": "quoin · Balanced", "quoin-high": "quoin · High", "quoin-max": "quoin · Max",
    "lz4": "lz4", "zlib-6": "zlib −6", "zstd-3": "zstd −3", "zstd-19": "zstd −19",
}
PARETO = ["lz4", "zlib-6", "zstd-3", "zstd-19",
          "quoin-fastest", "quoin-fast", "quoin-balanced", "quoin-high", "quoin-max"]
FONT = dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=15, color="#1e293b")


def log_ticks(lo, hi):
    vals, base = [], 1
    while base <= hi * 10:
        for m in (1, 2, 5):
            v = base * m
            if lo / 1.5 <= v <= hi * 1.5:
                vals.append(v)
        base *= 10
    fmt = lambda v: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"
    return vals, [fmt(v) for v in vals]


def make_pareto(sub, headline, subtitle, fname):
    """sub: DataFrame with columns codec, ratio, enc_mbps, dec_mbps."""
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.13,
        subplot_titles=("<b>Compression speed</b> vs ratio",
                        "<b>Decompression speed</b> vs ratio"),
    )
    rmin = sub.ratio.min(); rmax = sub.ratio.max()
    for col_i, metric in enumerate(["enc_mbps", "dec_mbps"], start=1):
        for codec in PARETO:
            r = sub[sub.codec == codec]
            if r.empty:
                continue
            is_q = codec.startswith("quoin")
            fig.add_trace(go.Scatter(
                x=[float(r.ratio.iloc[0])], y=[float(r[metric].iloc[0])],
                mode="markers", name=NAME[codec], legendgroup=codec, showlegend=(col_i == 1),
                marker=dict(size=20 if is_q else 14, color=COL[codec],
                            line=dict(width=1.5, color="white"),
                            symbol="circle" if is_q else "diamond"),
                cliponaxis=False,
                hovertemplate=f"{NAME[codec]}<br>ratio %{{x:.2f}}×<br>%{{y:.0f}} MB/s<extra></extra>",
            ), row=1, col=col_i)
        tv, tt = log_ticks(sub[metric].min(), sub[metric].max())
        fig.update_yaxes(type="log", tickvals=tv, ticktext=tt, row=1, col=col_i,
                         title_text="throughput  (MB/s, higher ↑ better)" if col_i == 1 else "",
                         gridcolor="#eef2f7", zeroline=False)
        fig.update_xaxes(row=1, col=col_i, title_text="compression ratio  (higher → better)",
                         range=[rmin * 0.88, rmax * 1.08], gridcolor="#eef2f7", zeroline=False)
    fig.update_layout(
        template="plotly_white", font=FONT,
        title=dict(text=f"<b>{headline}</b><br><sup>{subtitle}</sup>",
                   x=0.5, xanchor="center", font=dict(size=20)),
        legend=dict(orientation="h", yanchor="top", y=-0.17, xanchor="center", x=0.5,
                    font=dict(size=12)),
        width=1280, height=560, margin=dict(t=110, b=110, l=80, r=40),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.write_image(f"{OUT}/{fname}", scale=2)
    print(f"wrote {fname}")


# Top → bottom by how dramatic quoin's advantage is (absolute ratio gap).
vol = df[(df.section == "volume") & (df.n == df[df.section == "volume"].n.max())]
make_pareto(
    td[td.section == "EventTime"],
    "Integers — quoin's widest ratio lead, and the fastest high-ratio decode",
    "real ClickBench EventTime (i64 timestamp) · 1 M values · single-threaded, Intel Core Ultra 5 125H",
    "pareto_int.png")
make_pareto(
    td[td.section == "city_temperature"],
    "Decimals — quoin-Max takes the best ratio; quoin leads on decode",
    "city_temperature stored as Decimal128 · 2 M values · single-threaded",
    "pareto_decimal.png")
make_pareto(
    vol[["codec", "ratio", "enc_mbps", "dec_mbps"]],
    "Floats — the narrowest gap (quoin's hardest case); still best ratio + fast decode",
    "real arade4 f64 column · 9.9 M values · single-threaded",
    "pareto_float.png")


# ---- ratio breadth across f64 columns ----
breadth = df[~df.section.isin(["volume"])].copy()
datasets = list(dict.fromkeys(breadth.section))
qmax = lambda d: float(breadth[(breadth.section == d) & (breadth.codec == "quoin-max")].ratio.iloc[0])
datasets.sort(key=qmax)
labels = [d.replace("_f", "").replace("_", " ") for d in datasets]
bars = ["lz4", "zstd-3", "zstd-19", "quoin-balanced", "quoin-max"]
fig2 = go.Figure()
for codec in bars:
    vals = [float(breadth[(breadth.section == d) & (breadth.codec == codec)].ratio.iloc[0]) for d in datasets]
    fig2.add_trace(go.Bar(name=NAME[codec], x=labels, y=vals, marker_color=COL[codec], marker_line_width=0,
                          hovertemplate=f"{NAME[codec]}<br>%{{x}}<br>%{{y:.2f}}×<extra></extra>"))
fig2.update_layout(
    template="plotly_white", font=FONT, barmode="group", bargap=0.28, bargroupgap=0.08,
    title=dict(text="<b>Compression ratio across real f64 columns</b><br><sup>ALP benchmark corpus · higher is better</sup>",
               x=0.5, xanchor="center", font=dict(size=20)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1280, height=600, margin=dict(t=120, b=90, l=70, r=30),
    yaxis=dict(title="compression ratio  ×", gridcolor="#eef2f7", zeroline=False),
    xaxis=dict(tickangle=-20), plot_bgcolor="white", paper_bgcolor="white")
fig2.write_image(f"{OUT}/ratio_breadth.png", scale=2)
print("wrote ratio_breadth.png")


# ---- ratio breadth across integer & decimal columns ----
int_cols = ["EventTime", "UserID", "RegionID", "IPNetworkID"]
dec_cols = ["food_prices", "city_temperature", "bitcoin_tx"]
fig3 = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                     subplot_titles=("<b>Integer columns</b>  ·  real ClickBench i64 / i32",
                                     "<b>Decimal128 columns</b>  ·  real values as fixed-point"))
def add_panel(cols, col_i, showleg):
    lbls = [c.replace("_", " ") for c in cols]
    for codec in bars:
        vals = [float(td[(td.section == d) & (td.codec == codec)].ratio.iloc[0]) for d in cols]
        fig3.add_trace(go.Bar(name=NAME[codec], x=lbls, y=vals, marker_color=COL[codec], marker_line_width=0,
                              legendgroup=codec, showlegend=showleg,
                              hovertemplate=f"{NAME[codec]}<br>%{{x}}<br>%{{y:.1f}}×<extra></extra>"), row=1, col=col_i)
add_panel(int_cols, 1, True)
add_panel(dec_cols, 2, False)
fig3.update_layout(
    template="plotly_white", font=FONT, barmode="group", bargap=0.3, bargroupgap=0.06,
    title=dict(text="<b>Type-aware lanes: ratio on integers & decimals</b><br><sup>where quoin's specialization pays off most · higher is better</sup>",
               x=0.5, xanchor="center", font=dict(size=20)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    width=1280, height=560, margin=dict(t=120, b=70, l=60, r=30),
    plot_bgcolor="white", paper_bgcolor="white")
fig3.update_yaxes(title_text="compression ratio  ×", gridcolor="#eef2f7", zeroline=False, row=1, col=1)
fig3.update_yaxes(gridcolor="#eef2f7", zeroline=False, row=1, col=2)
fig3.write_image(f"{OUT}/typed_ratio.png", scale=2)
print("wrote typed_ratio.png")
