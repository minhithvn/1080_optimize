"""Biểu đồ Plotly – gộp giá/khối lượng/chỉ báo trên cùng trục thời gian, ẩn ngày nghỉ."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

UP, DOWN = "#16a34a", "#dc2626"
TEMPLATE = "plotly_dark"


def _rangebreaks(df):
    all_days = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = all_days.difference(df.index)
    return [dict(values=missing)]


def main_chart(df: pd.DataFrame, symbol: str, cone: dict | None = None, plan: dict | None = None,
               show: tuple = ("MA20", "MA50", "MA200", "BB", "Supertrend")) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.55, 0.15, 0.15, 0.15],
                        subplot_titles=(None, "Khối lượng", "RSI (14)", "MACD"))
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                 name="Giá", increasing_line_color=UP, decreasing_line_color=DOWN), 1, 1)
    colors = {"MA20": "#f59e0b", "MA50": "#3b82f6", "MA200": "#a855f7"}
    for ma in ("MA20", "MA50", "MA200"):
        if ma in show:
            fig.add_trace(go.Scatter(x=df.index, y=df[ma], name=ma, line=dict(color=colors[ma], width=1.4)), 1, 1)
    if "BB" in show:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_upper"], name="BB trên", line=dict(color="rgba(148,163,184,.5)",
                                 dash="dot", width=1)), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_lower"], name="BB dưới", line=dict(color="rgba(148,163,184,.5)",
                                 dash="dot", width=1), fill="tonexty", fillcolor="rgba(148,163,184,.07)"), 1, 1)
    if "Supertrend" in show:
        st_up = df["Supertrend"].where(df["Supertrend_direction"] == 1)
        st_dn = df["Supertrend"].where(df["Supertrend_direction"] == -1)
        fig.add_trace(go.Scatter(x=df.index, y=st_up, name="Supertrend ↑", line=dict(color=UP, width=1.2)), 1, 1)
        fig.add_trace(go.Scatter(x=df.index, y=st_dn, name="Supertrend ↓", line=dict(color=DOWN, width=1.2)), 1, 1)

    if cone:
        d, q = cone["dates"], cone["q"]
        fig.add_trace(go.Scatter(x=d, y=q["p97"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=d, y=q["p2"], fill="tonexty", fillcolor="rgba(250,204,21,.10)",
                                 line=dict(width=0), name="Vùng 95%"), 1, 1)
        fig.add_trace(go.Scatter(x=d, y=q["p90"], line=dict(width=0), showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=d, y=q["p10"], fill="tonexty", fillcolor="rgba(250,204,21,.22)",
                                 line=dict(width=0), name="Vùng 80%"), 1, 1)
        fig.add_trace(go.Scatter(x=d, y=q["p50"], name="Trung vị dự phóng",
                                 line=dict(color="#facc15", width=2, dash="dot")), 1, 1)

    if plan:
        for y, name, col in ((plan["stop"], "Cắt lỗ", DOWN), (plan["t1"], "Mục tiêu 1", UP),
                             (plan["t2"], "Mục tiêu 2", "#22d3ee")):
            fig.add_hline(y=y, line=dict(color=col, dash="dash", width=1), row=1, col=1,
                          annotation_text=f"{name}: {y:,.0f}", annotation_position="right")

    vcol = np.where(df["close"] >= df["open"], UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=vcol, name="KL", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Volume_MA"], line=dict(color="#f59e0b", width=1), name="KL TB20",
                             showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#38bdf8", width=1.3), name="RSI",
                             showlegend=False), 3, 1)
    for y, col in ((70, DOWN), (30, UP)):
        fig.add_hline(y=y, line=dict(color=col, dash="dot", width=1), row=3, col=1)
    hcol = np.where(df["MACD_hist"] >= 0, UP, DOWN)
    fig.add_trace(go.Bar(x=df.index, y=df["MACD_hist"], marker_color=hcol, showlegend=False, name="Hist"), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#38bdf8", width=1), showlegend=False,
                             name="MACD"), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_signal"], line=dict(color="#f59e0b", width=1),
                             showlegend=False, name="Signal"), 4, 1)

    fig.update_layout(template=TEMPLATE, height=820, margin=dict(l=8, r=8, t=40, b=8),
                      title=f"{symbol} – Phân tích kỹ thuật", xaxis_rangeslider_visible=False,
                      hovermode="x unified", legend=dict(orientation="h", y=1.04, x=0, font=dict(size=10)),
                      dragmode="pan")
    if not cone:
        fig.update_xaxes(rangebreaks=_rangebreaks(df))
    return fig


def indicator_panel(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("ADX / DI", "MFI & Stochastic", "Chaikin Money Flow", "OBV"))
    fig.add_trace(go.Scatter(x=df.index, y=df["ADX"], name="ADX", line=dict(color="#a855f7", width=2)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DI_plus"], name="+DI", line=dict(color=UP)), 1, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["DI_minus"], name="-DI", line=dict(color=DOWN)), 1, 1)
    fig.add_hline(y=25, line=dict(dash="dot", color="gray"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MFI"], name="MFI", line=dict(color="#ec4899")), 2, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["Stoch_K"], name="%K", line=dict(color="#38bdf8", width=1)), 2, 1)
    fig.add_trace(go.Bar(x=df.index, y=df["CMF"], name="CMF",
                         marker_color=np.where(df["CMF"] >= 0, UP, DOWN)), 3, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV"], name="OBV", line=dict(color="#22d3ee")), 4, 1)
    fig.add_trace(go.Scatter(x=df.index, y=df["OBV_MA20"], name="OBV MA20", line=dict(color="#f59e0b", dash="dot")),
                  4, 1)
    fig.update_layout(template=TEMPLATE, height=760, margin=dict(l=8, r=8, t=40, b=8), hovermode="x unified",
                      legend=dict(orientation="h", y=1.05, font=dict(size=10)))
    fig.update_xaxes(rangebreaks=_rangebreaks(df))
    return fig


def score_gauge(score: float, label: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score if np.isfinite(score) else 0,
        title={"text": label, "font": {"size": 18}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#e5e7eb"},
               "steps": [{"range": [0, 35], "color": "#7f1d1d"}, {"range": [35, 45], "color": "#b45309"},
                         {"range": [45, 62], "color": "#525252"}, {"range": [62, 75], "color": "#166534"},
                         {"range": [75, 100], "color": "#15803d"}]}))
    fig.update_layout(template=TEMPLATE, height=230, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def components_bar(comp: dict) -> go.Figure:
    k = list(comp.keys())
    v = [comp[x] for x in k]
    fig = go.Figure(go.Bar(x=v, y=k, orientation="h", marker_color=[UP if x >= 0 else DOWN for x in v],
                           text=[f"{x:+.0f}" for x in v], textposition="outside"))
    fig.update_layout(template=TEMPLATE, height=320, margin=dict(l=8, r=30, t=30, b=8), title="Đóng góp điểm",
                      xaxis=dict(zeroline=True))
    return fig


def compare_chart(closes: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for sym, s in closes.items():
        s = s.dropna()
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s / s.iloc[0] * 100, name=sym, mode="lines"))
    fig.update_layout(template=TEMPLATE, height=480, title="Hiệu suất chuẩn hóa (=100)", hovermode="x unified",
                      margin=dict(l=8, r=8, t=40, b=8), legend=dict(orientation="h", y=1.08))
    return fig


def corr_heatmap(closes: dict[str, pd.Series]) -> go.Figure:
    rets = pd.DataFrame(closes).pct_change().dropna(how="all")
    corr = rets.corr()  # tương quan LỢI NHUẬN (bản cũ dùng tương quan giá -> sai, luôn rất cao)
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdYlGn", zmid=0,
                               zmin=-1, zmax=1, text=corr.round(2).values, texttemplate="%{text}"))
    fig.update_layout(template=TEMPLATE, height=460, title="Tương quan lợi nhuận ngày",
                      margin=dict(l=8, r=8, t=40, b=8))
    return fig


def sector_treemap(snap: pd.DataFrame, col: str = "1D %") -> go.Figure:
    d = snap[snap["GTGD TB20 (tỷ)"] > 0.5].copy()
    secs = sorted(d["Ngành"].unique())
    labels = list(d["Mã"]) + secs
    parents = list(d["Ngành"]) + [""] * len(secs)
    values = list(d["GTGD TB20 (tỷ)"].clip(lower=0.5)) + [0] * len(secs)
    chg = list(d[col].clip(-7, 7)) + [0] * len(secs)
    custom = np.vstack([np.stack([d[col].values, d["Giá"].values], axis=1), np.zeros((len(secs), 2))])
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values, customdata=custom,
        marker=dict(colors=chg, colorscale=[[0, DOWN], [0.5, "#3f3f46"], [1, UP]], cmid=0, cmin=-7, cmax=7),
        hovertemplate="<b>%{label}</b><br>%{customdata[0]:+.2f}%<br>Giá %{customdata[1]:,.0f}<extra></extra>",
        texttemplate="%{label}<br>%{customdata[0]:+.1f}%"))
    fig.update_layout(template=TEMPLATE, height=620, margin=dict(l=4, r=4, t=30, b=4),
                      title=f"Bản đồ nhiệt thị trường ({col}, diện tích = GTGD TB20)")
    return fig


def sector_bar(snap: pd.DataFrame, col: str = "1W %") -> go.Figure:
    g = snap.groupby("Ngành").apply(
        lambda x: np.average(x[col], weights=x["GTGD TB20 (tỷ)"].clip(lower=0.1))).sort_values()
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h",
                           marker_color=[UP if v >= 0 else DOWN for v in g.values],
                           text=[f"{v:+.2f}%" for v in g.values], textposition="outside"))
    fig.update_layout(template=TEMPLATE, height=max(380, 24 * len(g)), title=f"Hiệu suất ngành ({col}, gia quyền GTGD)",
                      margin=dict(l=8, r=40, t=40, b=8))
    return fig


def index_chart(dfs: dict[str, pd.DataFrame]) -> go.Figure:
    fig = make_subplots(rows=1, cols=1)
    for name, d in dfs.items():
        if d is not None and len(d):
            d = d.tail(250)
            fig.add_trace(go.Scatter(x=d.index, y=d["close"] / d["close"].iloc[0] * 100, name=name))
    fig.update_layout(template=TEMPLATE, height=380, title="Các chỉ số (1 năm, chuẩn hóa = 100)",
                      hovermode="x unified", margin=dict(l=8, r=8, t=40, b=8), legend=dict(orientation="h", y=1.1))
    return fig


def equity_chart(res: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res["equity"].index, y=res["equity"], name="Chiến lược", line=dict(color=UP)))
    fig.add_trace(go.Scatter(x=res["buy_hold"].index, y=res["buy_hold"], name="Mua & giữ",
                             line=dict(color="#94a3b8", dash="dot")))
    fig.update_layout(template=TEMPLATE, height=420, title="Đường vốn", hovermode="x unified",
                      margin=dict(l=8, r=8, t=40, b=8), legend=dict(orientation="h", y=1.1))
    return fig
