"""
PHÂN TÍCH CỔ PHIẾU VIỆT NAM – bản tối ưu
Chạy:  streamlit run 1080.py
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Phân Tích Cổ Phiếu VN", page_icon="📊", layout="wide",
                   initial_sidebar_state="auto")

from vnapp import charts, data, news, tracking  # noqa: E402
from vnapp.backtest import run_backtest  # noqa: E402
from vnapp.config import PRICE_LIMIT  # noqa: E402
from vnapp.forecast import ml_trend, price_cone  # noqa: E402
from vnapp.indicators import add_indicators, candlestick_today  # noqa: E402
from vnapp.signals import generate_signal, label_for, market_regime, score_frame, trade_plan  # noqa: E402

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.sig {padding: 14px 16px; border-radius: 12px; font-weight: 700; font-size: 1.15rem; text-align:center; color:#fff;}
.sig-buy {background:#15803d;} .sig-sell {background:#b91c1c;} .sig-hold {background:#a16207;}
.card {border:1px solid rgba(148,163,184,.25); border-radius:12px; padding:12px 14px; margin-bottom:8px;}
.small {font-size:.85rem; opacity:.8;}
@media (max-width: 640px) {
  h1 {font-size: 1.45rem !important;} h2 {font-size: 1.2rem !important;}
  [data-testid="stMetricValue"] {font-size: 1.15rem;}
}
</style>""", unsafe_allow_html=True)

tracking.init_session(st)

DISCLAIMER = ("⚠️ Công cụ hỗ trợ phân tích, **không phải khuyến nghị đầu tư**. Không mô hình nào dự đoán chính xác "
              "giá cổ phiếu; các xác suất/vùng giá dưới đây là ước lượng thống kê từ dữ liệu quá khứ. "
              "Luôn đặt cắt lỗ và tự chịu trách nhiệm với quyết định của mình.")


# ============================================================================
# Helpers có cache
# ============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def load_bench(count: int = 800):
    df, _ = data.get_history("VNINDEX", count)
    return df


@st.cache_data(ttl=900, show_spinner=False)
def analyze(symbol: str, count: int):
    df, src = data.get_history(symbol, count)
    if df is None:
        return None, None, src
    bench = load_bench(max(count, 800))
    return add_indicators(df, bench), market_regime(bench), src


@st.cache_data(ttl=1800, show_spinner=False)
def cached_ml(symbol: str, count: int, horizon: int):
    df, regime, _ = analyze(symbol, count)
    return ml_trend(df, horizon, regime)


def fmt(x, d=0, suffix=""):
    return "N/A" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{d}f}{suffix}"


def sig_box(signal: str, score: float):
    cls = "sig-buy" if signal.startswith("MUA") else "sig-sell" if signal in ("BÁN", "GIẢM TỶ TRỌNG") else "sig-hold"
    st.markdown(f'<div class="sig {cls}">{signal} · {fmt(score)}/100</div>', unsafe_allow_html=True)


listing = data.get_listing()
ALL_SYMBOLS = listing["symbol"].tolist()
SYMBOL_SET = set(ALL_SYMBOLS)
META = listing.set_index("symbol")

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.header("📊 Cổ phiếu VN")
    mode = st.radio("Chế độ", ["🏠 Tổng quan thị trường", "🔍 Phân tích cổ phiếu", "🚀 Bộ lọc cổ phiếu",
                               "📰 Tin tức & Vĩ mô", "📊 So sánh", "📈 Backtest"])
    st.caption(f"{len(ALL_SYMBOLS):,} mã · nguồn danh sách: "
               f"{'trực tiếp (Vietcap/VNDirect)' if listing.attrs.get('source') == 'live' else 'dự phòng nội bộ'}")
    ex_counts = listing["exchange"].value_counts()
    st.caption(" · ".join(f"{k}: {v}" for k, v in ex_counts.items()))
    if st.button("🔄 Làm mới dữ liệu"):
        st.cache_data.clear()
        st.rerun()

st.title("📊 Phân tích cổ phiếu Việt Nam")

# ============================================================================
# 1) TỔNG QUAN THỊ TRƯỜNG
# ============================================================================
if mode == "🏠 Tổng quan thị trường":
    idx = {}
    cols = st.columns(4)
    for i, name in enumerate(["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]):
        d, _ = data.get_history(name, 300)
        idx[name] = d
        if d is not None and len(d) > 1:
            chg = (d["close"].iloc[-1] / d["close"].iloc[-2] - 1) * 100
            cols[i].metric(name, f"{d['close'].iloc[-1]:,.2f}", f"{chg:+.2f}%")
        else:
            cols[i].metric(name, "N/A")

    bench = idx.get("VNINDEX")
    if bench is not None and len(bench) > 60:
        bi = add_indicators(bench)
        reg = market_regime(bench).iloc[-1]
        txt = {1: "🟢 VNINDEX trong **xu hướng tăng** (giá > MA50 > MA200) – ưu tiên nắm giữ/mua cổ phiếu dẫn dắt.",
               0: "🟡 VNINDEX **trung tính / tích lũy** – giải ngân từng phần, chọn lọc cổ phiếu khỏe hơn thị trường.",
               -1: "🔴 VNINDEX trong **xu hướng giảm** – hạ tỷ trọng, giữ tiền mặt, chỉ trading ngắn với cắt lỗ chặt."}[reg]
        st.info(txt + f"  RSI: **{bi['RSI'].iloc[-1]:.0f}** · ADX: **{bi['ADX'].iloc[-1]:.0f}** · "
                f"Biến động 20 phiên: **{bi['HV20'].iloc[-1]:.0f}%/năm**")
    st.plotly_chart(charts.index_chart({k: v for k, v in idx.items() if v is not None}), width="stretch")

    c1, c2 = st.columns([1, 1])
    universe_ex = c1.multiselect("Sàn cho bản đồ thị trường", ["HOSE", "HNX", "UPCOM"], default=["HOSE"])
    period_col = c2.selectbox("Khung", ["1D %", "1W %", "1M %", "3M %"], index=0)
    uni = tuple(listing[listing["exchange"].isin(universe_ex)]["symbol"]) if listing.attrs.get("source") == "live" \
        else tuple(data.get_group("VN30"))
    if uni:
        pb = st.progress(0.0, text=f"Đang tải {len(uni)} mã (lần đầu ~20–40 giây, sau đó dùng cache 15 phút)...")
        hist = data.get_history_many(uni, 80, progress=lambda d, t: pb.progress(d / t, text=f"Đang tải {d}/{t} mã"))
        pb.empty()
        snap = data.snapshot_from_histories(hist, listing)
        if snap.empty:
            st.warning("Không tải được dữ liệu thị trường – nguồn dữ liệu có thể đang bận, thử lại sau ít phút.")
        else:
            liquid = snap[snap["GTGD TB20 (tỷ)"] >= 1]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Mã tăng / giảm", f"{(snap['1D %'] > 0).sum()} / {(snap['1D %'] < 0).sum()}")
            b2.metric("% mã trên MA50", f"{snap['Trên MA50'].mean() * 100:.0f}%",
                      help=">60%: thị trường khỏe; <30%: yếu/quá bán diện rộng")
            b3.metric("GTGD hôm nay", f"{snap['GTGD hôm nay (tỷ)'].sum():,.0f} tỷ")
            b4.metric("Mã KL đột biến (>2x)", int((liquid["Đột biến KL"] > 2).sum()))

            t1, t2, t3 = st.tabs(["🗺️ Bản đồ nhiệt", "🏭 Ngành", "🔥 Dòng tiền & Top"])
            with t1:
                st.plotly_chart(charts.sector_treemap(snap, period_col), width="stretch")
            with t2:
                st.plotly_chart(charts.sector_bar(liquid if len(liquid) > 20 else snap, period_col),
                                width="stretch")
            with t3:
                cols_show = ["Mã", "Ngành", "Giá", "1D %", "1W %", "GTGD hôm nay (tỷ)", "Đột biến KL"]
                fm = {"Giá": "{:,.0f}", "1D %": "{:+.2f}", "1W %": "{:+.2f}", "GTGD hôm nay (tỷ)": "{:,.1f}",
                      "Đột biến KL": "{:.1f}x"}
                x1, x2 = st.columns(2)
                x1.markdown("**💰 Hút tiền mạnh nhất**")
                x1.dataframe(snap.nlargest(15, "GTGD hôm nay (tỷ)")[cols_show].style.format(fm), hide_index=True,
                             width="stretch")
                x2.markdown("**⚡ Khối lượng đột biến (thanh khoản ≥1 tỷ)**")
                x2.dataframe(liquid.nlargest(15, "Đột biến KL")[cols_show].style.format(fm), hide_index=True,
                             width="stretch")
                x1.markdown("**📈 Tăng mạnh nhất**")
                x1.dataframe(liquid.nlargest(15, period_col)[cols_show].style.format(fm), hide_index=True,
                             width="stretch")
                x2.markdown("**📉 Giảm mạnh nhất**")
                x2.dataframe(liquid.nsmallest(15, period_col)[cols_show].style.format(fm), hide_index=True,
                             width="stretch")

    st.subheader("🗞️ Đang được nhắc đến trên báo")
    with st.spinner("Đang đọc tin..."):
        mk_news = news.market_news()
    n1, n2 = st.columns(2)
    tr = news.trending_tickers(mk_news, SYMBOL_SET)
    n1.markdown("**Mã nóng trên tin tức (≈ 1–2 ngày gần nhất)**")
    n1.dataframe(tr.style.format({"Cảm xúc TB": "{:+.2f}"}), hide_index=True, width="stretch")
    n2.markdown("**Chủ đề nóng**")
    n2.dataframe(news.hot_keywords(mk_news).style.format({"Cảm xúc TB": "{:+.2f}"}), hide_index=True,
                 width="stretch")

# ============================================================================
# 2) PHÂN TÍCH CỔ PHIẾU
# ============================================================================
elif mode == "🔍 Phân tích cổ phiếu":
    with st.sidebar:
        labels = [f"{s} – {META['name'].get(s, '')}"[:48] for s in ALL_SYMBOLS]
        default = ALL_SYMBOLS.index("FPT") if "FPT" in ALL_SYMBOLS else 0
        pick = st.selectbox("Mã cổ phiếu (gõ để tìm)", labels, index=default)
        symbol = pick.split(" – ")[0]
        period = st.selectbox("Hiển thị", ["3mo", "6mo", "1y", "2y", "3y"], index=2)
        horizon = st.slider("Tầm dự báo (phiên)", 5, 40, 10)
        run_ml = st.checkbox("🤖 Mô hình ML (kiểm định walk-forward)", value=True)

    tracking.track_search(symbol) if st.session_state.get("_last_sym") != symbol else None
    st.session_state["_last_sym"] = symbol
    exch = META["exchange"].get(symbol, "HOSE")
    name = META["name"].get(symbol, "")
    st.header(f"{symbol} · {name}")
    st.caption(f"Sàn {exch} · Ngành: {META['industry'].get(symbol, 'Khác')} · Biên độ ±{PRICE_LIMIT.get(exch, .07) * 100:.0f}%")

    count = data.bars_for_period("5y" if run_ml else period)
    with st.spinner("Đang tải dữ liệu..."):
        df, regime, src = analyze(symbol, count)
    if df is None or len(df) < 60:
        st.error(f"Không tải được đủ dữ liệu cho {symbol} (nguồn: {src}). Mã có thể mới niêm yết, bị hủy hoặc "
                 "các nguồn đang bận – thử lại sau.")
        st.stop()

    with st.spinner("Đang đọc tin tức..."):
        sn = news.stock_news(symbol, name if len(name) > 3 else "")
    nscore = news.news_score(sn)
    sig = generate_signal(df, regime, nscore)
    plan = trade_plan(df, sig, exch)
    cone = price_cone(df, horizon, exch, targets=(plan["t1"], plan["stop"]))
    last, prev = df.iloc[-1], df.iloc[-2]
    view = df.tail(data.PERIOD_TO_BARS.get(period, 252))

    m = st.columns(4)
    m[0].metric("Giá", fmt(last["close"]), f"{(last['close'] / prev['close'] - 1) * 100:+.2f}%")
    m[1].metric("GTGD TB20", f"{last['Value_MA20_bn']:,.1f} tỷ")
    m[2].metric("So với đỉnh 52T", f"{last['Dist_High52']:+.1f}%")
    m[3].metric("RS vs VNINDEX 3T", fmt(last.get("RS_3M"), 1, "%"))
    st.caption(f"Dữ liệu: {src} · phiên cuối {df.index[-1]:%d/%m/%Y}")

    g1, g2 = st.columns([1, 1])
    with g1:
        st.plotly_chart(charts.score_gauge(sig["score"], "Điểm kỹ thuật tổng hợp"), width="stretch")
        sig_box(sig["signal"], sig["score"])
        chg5 = sig.get("score_change_5d", 0)
        st.caption(f"Điểm thay đổi 5 phiên: {chg5:+.0f} · Tin tức 7 ngày: {nscore:+.2f} ({len(sn)} tin)")
    with g2:
        st.plotly_chart(charts.components_bar(sig["components"]), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**🟢 Điểm cộng**")
        for r in sig["pos"]:
            st.markdown(f"- {r}")
    with b:
        st.markdown("**🔴 Điểm trừ / rủi ro**")
        for r in sig["neg"]:
            st.markdown(f"- {r}")
    pats = candlestick_today(df)
    if pats:
        st.caption("Mẫu nến phiên cuối: " + ", ".join(pats))

    st.subheader("🎯 Kế hoạch giao dịch tham khảo")
    p = st.columns(4)
    p[0].metric("Vùng mua", f"{plan['entry'][0]:,.0f} – {plan['entry'][1]:,.0f}")
    p[1].metric("Cắt lỗ", fmt(plan["stop"]), f"{(plan['stop'] / plan['price'] - 1) * 100:.1f}%")
    p[2].metric("Mục tiêu 1", fmt(plan["t1"]), f"{(plan['t1'] / plan['price'] - 1) * 100:+.1f}%")
    p[3].metric("Mục tiêu 2", fmt(plan["t2"]), f"{(plan['t2'] / plan['price'] - 1) * 100:+.1f}%")
    advice = []
    if sig["signal"].startswith("MUA"):
        advice.append(f"Có thể giải ngân **từng phần (30–50%)** trong vùng mua, thêm khi giá vượt kháng cự "
                      f"{plan['sr']['resistance'][0][0]:,.0f} kèm thanh khoản.")
    elif sig["signal"] == "THEO DÕI / NẮM GIỮ":
        advice.append("Đang nắm giữ: tiếp tục giữ, dời cắt lỗ theo Supertrend. Chưa có: chờ điểm > 62 hoặc vượt kháng cự.")
    else:
        advice.append("**Không mở vị thế mới.** Đang nắm giữ: cân nhắc hạ tỷ trọng khi hồi lên kháng cự, "
                      "tuân thủ cắt lỗ.")
    if plan["rr"] < 1.5:
        advice.append(f"Tỷ lệ lợi nhuận/rủi ro tới MT1 chỉ **{plan['rr']:.1f}** – kém hấp dẫn.")
    if plan["trailing"]:
        advice.append(f"Trailing stop (Supertrend): **{plan['trailing']:,.0f}**.")
    advice.append(f"Khung thời gian phù hợp: **{plan['horizon']}**. Mỗi lệnh không nên rủi ro quá 1–2% tổng vốn.")
    st.markdown("\n".join(f"- {x}" for x in advice))

    st.subheader(f"🔮 Dự phóng {horizon} phiên")
    f = st.columns(4)
    if cone:
        q = cone["q"]
        f[0].metric("Trung vị", fmt(q["p50"][-1]), f"{(q['p50'][-1] / cone['p0'] - 1) * 100:+.1f}%")
        f[1].metric("Vùng 80%", f"{q['p10'][-1]:,.0f} – {q['p90'][-1]:,.0f}")
        f[2].metric("XS chạm MT1 trước cắt lỗ", f"{cone.get('p_target_first', np.nan) * 100:.0f}%")
        f[3].metric("XS chạm cắt lỗ trước", f"{cone.get('p_stop_first', np.nan) * 100:.0f}%")
    ml = None
    if run_ml:
        with st.spinner("Đang huấn luyện & kiểm định mô hình..."):
            ml = cached_ml(symbol, count, horizon)
    if ml:
        stats = (f"Độ chính xác ngoài mẫu {ml['oos_accuracy'] * 100:.1f}% vs baseline {ml['baseline'] * 100:.1f}% · "
                 f"AUC {ml['auc']:.2f} · {ml['n_samples']} mẫu")
        if ml["reliable"]:
            col = "🟢" if ml["direction"] == "TĂNG" else "🔴"
            st.success(f"{col} ML: xác suất **tăng sau {horizon} phiên ≈ {ml['prob_up'] * 100:.0f}%**  \n{stats}")
        else:
            st.warning(f"🤖 ML **không có lợi thế** dự báo cho mã này ({stats}) → bỏ qua tín hiệu ML, "
                       f"dựa vào điểm kỹ thuật & quản trị rủi ro. (Tham khảo: mô hình nghiêng "
                       f"{ml['direction'].lower()}, nhưng không đáng tin.)")
    elif run_ml:
        st.caption("Không đủ dữ liệu lịch sử (≥ 250 phiên) để chạy ML.")

    tabs = st.tabs(["📈 Biểu đồ", "📊 Chỉ báo", "📰 Tin tức", "💼 Cơ bản", "🧱 Hỗ trợ/Kháng cự"])
    with tabs[0]:
        st.plotly_chart(charts.main_chart(view, symbol, cone, plan), width="stretch",
                        config={"scrollZoom": True, "displaylogo": False})
    with tabs[1]:
        st.plotly_chart(charts.indicator_panel(view), width="stretch")
        keys = ["RSI", "MACD", "MACD_signal", "ADX", "DI_plus", "DI_minus", "MFI", "CMF", "Stoch_K", "CCI",
                "ATR", "ATR_pct", "HV20", "BB_pctB", "Volume_ratio", "MA20", "MA50", "MA200"]
        st.dataframe(pd.DataFrame({"Chỉ báo": keys, "Giá trị": [last.get(k) for k in keys]}).style.format(
            {"Giá trị": "{:,.2f}"}), hide_index=True, width="stretch")
    with tabs[2]:
        if sn.empty:
            st.info("Chưa tìm thấy tin trong 14 ngày gần đây.")
        for _, r in sn.head(25).iterrows():
            icon = "🟢" if r["Cảm xúc"] > 0.2 else "🔴" if r["Cảm xúc"] < -0.2 else "⚪"
            t = r["Thời gian"].strftime("%d/%m %H:%M") if pd.notna(r["Thời gian"]) else ""
            st.markdown(f"{icon} [{r['Tiêu đề']}]({r['Link']})  \n<span class='small'>{t} · {r['Nguồn']}</span>",
                        unsafe_allow_html=True)
    with tabs[3]:
        fu = data.get_fundamentals(symbol)
        if fu.get("source"):
            def pct(x):
                return None if x is None else (x * 100 if abs(x) < 5 else x)
            k = st.columns(4)
            k[0].metric("P/E", fmt(fu.get("pe"), 1))
            k[1].metric("P/B", fmt(fu.get("pb"), 2))
            k[2].metric("ROE", fmt(pct(fu.get("roe")), 1, "%"))
            eps = df["close"].iloc[-1] / fu["pe"] if fu.get("pe") and fu["pe"] > 0 else None
            k[3].metric("EPS (ước tính)", fmt(eps))
            k = st.columns(4)
            k[0].metric("ROA", fmt(pct(fu.get("roa")), 1, "%"))
            k[1].metric("Nợ/VCSH", fmt(fu.get("de"), 2))
            k[2].metric("Biên LN ròng", fmt(pct(fu.get("net_margin")), 1, "%"))
            k[3].metric("Tỷ suất cổ tức", fmt(pct(fu.get("dividend_yield")), 1, "%"))
            if fu.get("market_cap_bn"):
                st.caption(f"Vốn hóa: {fu['market_cap_bn']:,.0f} tỷ VND"
                           + (f" · NPL: {fu['npl'] * 100:.2f}% · NIM: {fu['nim'] * 100:.2f}%"
                              if fu.get("npl") and fu.get("nim") else ""))
            notes = []
            pe, roe = fu.get("pe"), pct(fu.get("roe"))
            if pe and 0 < pe < 10 and roe and roe > 15:
                notes.append("✅ P/E thấp + ROE cao: định giá hấp dẫn nếu lợi nhuận bền vững.")
            if pe and pe > 25:
                notes.append("⚠️ P/E > 25: thị trường đang kỳ vọng tăng trưởng cao, rủi ro nếu KQKD hụt.")
            if roe is not None and fu.get("roe_prev_year"):
                d_roe = roe - pct(fu["roe_prev_year"])
                notes.append(f"{'✅' if d_roe > 0 else '⚠️'} ROE {'cải thiện' if d_roe > 0 else 'suy giảm'} "
                             f"{d_roe:+.1f} điểm % so với cùng kỳ năm trước.")
            if roe is not None and roe < 8:
                notes.append("⚠️ ROE < 8%: hiệu quả sử dụng vốn thấp.")
            if fu.get("de") and fu["de"] > 2 and META["industry"].get(symbol) not in ("Ngân hàng", "Dịch vụ tài chính"):
                notes.append("⚠️ Đòn bẩy cao (Nợ/VCSH > 2).")
            for n in notes:
                st.markdown(n)
            st.caption(f"Nguồn: {fu['source']} {fu.get('period', '')}")
            if isinstance(fu.get("history"), pd.DataFrame):
                with st.expander("Lịch sử chỉ số theo quý"):
                    st.dataframe(fu["history"], width="stretch")
        else:
            st.warning("Không lấy được dữ liệu cơ bản.")
        st.markdown(f"[CafeF](https://cafef.vn/tim-kiem.chn?keywords={symbol}) · "
                    f"[Vietstock](https://finance.vietstock.vn/{symbol}/tai-chinh.htm) · "
                    f"[Simplize](https://simplize.vn/co-phieu/{symbol})")
    with tabs[4]:
        sr = plan["sr"]
        rows = ([{"Mức": f"Kháng cự {i + 1}", "Giá": lv, "Số lần chạm": n, "Cách giá %": (lv / sr['current'] - 1) * 100}
                 for i, (lv, n) in enumerate(sr["resistance"])][::-1]
                + [{"Mức": "Giá hiện tại", "Giá": sr["current"], "Số lần chạm": None, "Cách giá %": 0.0}]
                + [{"Mức": f"Hỗ trợ {i + 1}", "Giá": lv, "Số lần chạm": n, "Cách giá %": (lv / sr['current'] - 1) * 100}
                   for i, (lv, n) in enumerate(sr["support"])])
        st.dataframe(pd.DataFrame(rows).style.format({"Giá": "{:,.0f}", "Cách giá %": "{:+.1f}"}),
                     hide_index=True, width="stretch")
        st.caption(f"Pivot: {sr['pivot']:,.0f} · R1: {sr['R1']:,.0f} · S1: {sr['S1']:,.0f}. "
                   "Mức được xác định từ các đỉnh/đáy swing 120 phiên, gom cụm ±1.5%.")

# ============================================================================
# 3) BỘ LỌC
# ============================================================================
elif mode == "🚀 Bộ lọc cổ phiếu":
    with st.sidebar:
        scope = st.selectbox("Phạm vi", ["VN30", "HOSE", "HNX", "UPCOM", "Theo ngành", "Toàn thị trường"])
        if scope == "Theo ngành":
            ind = st.selectbox("Ngành", sorted(listing["industry"].unique()))
            pool = listing[listing["industry"] == ind]["symbol"].tolist()
        elif scope == "VN30":
            pool = data.get_group("VN30")
        elif scope == "Toàn thị trường":
            pool = ALL_SYMBOLS
        else:
            pool = listing[listing["exchange"] == scope]["symbol"].tolist()
        min_liq = st.number_input("GTGD TB20 tối thiểu (tỷ VND)", 0.0, 500.0, 3.0, 1.0)
        min_score = st.slider("Điểm tối thiểu", 0, 100, 62)
        only_up = st.checkbox("Chỉ mã trong xu hướng tăng (giá > MA50 > MA200)", value=False)
    st.header("🚀 Bộ lọc cổ phiếu")
    st.caption(f"Phạm vi: {len(pool)} mã. Tải song song 10 luồng; quét toàn thị trường lần đầu mất khoảng 2–4 phút, "
               "các lần sau trong 15 phút dùng cache.")
    if st.button("🔍 Quét", type="primary", width="stretch"):
        bench = load_bench()
        regime = market_regime(bench)
        pb = st.progress(0.0, text="Đang tải dữ liệu...")
        hist = data.get_history_many(tuple(pool), 320,
                                     progress=lambda d, t: pb.progress(d / t, text=f"Đang tải {d}/{t} mã"))
        pb.empty()
        prog = st.progress(0.0)
        rows = []
        items = list(hist.items())
        for i, (sym, d) in enumerate(items):
            prog.progress((i + 1) / max(len(items), 1))
            try:
                if len(d) < 80 or (d["close"] * d["volume"]).tail(20).mean() / 1e9 < min_liq:
                    continue
                d = add_indicators(d, bench)
                s = score_frame(d, regime)
                sc = s["score"].iloc[-1]
                lt = d.iloc[-1]
                if not np.isfinite(sc) or sc < min_score:
                    continue
                if only_up and not (lt["close"] > lt["MA50"] > lt["MA200"]):
                    continue
                rows.append({"Mã": sym, "Ngành": META["industry"].get(sym, ""), "Sàn": META["exchange"].get(sym, ""),
                             "Điểm": sc, "Tín hiệu": label_for(sc), "Δ điểm 5P": s["score"].diff().tail(5).sum(),
                             "Giá": lt["close"], "1D %": (lt["close"] / d["close"].iloc[-2] - 1) * 100,
                             "1M %": lt["ROC20"], "RSI": lt["RSI"], "ADX": lt["ADX"],
                             "RS 3T": lt.get("RS_3M", np.nan), "KL/TB20": lt["Volume_ratio"],
                             "GTGD TB20 (tỷ)": lt["Value_MA20_bn"]})
            except Exception:
                continue
        prog.empty()
        if rows:
            res = pd.DataFrame(rows).sort_values(["Điểm", "RS 3T"], ascending=False)
            st.success(f"{len(res)} mã đạt tiêu chí (đã tải {len(hist)}/{len(pool)} mã).")
            st.dataframe(res.style.format({"Điểm": "{:.0f}", "Δ điểm 5P": "{:+.0f}", "Giá": "{:,.0f}",
                                           "1D %": "{:+.2f}", "1M %": "{:+.1f}", "RSI": "{:.0f}", "ADX": "{:.0f}",
                                           "RS 3T": "{:+.1f}", "KL/TB20": "{:.1f}", "GTGD TB20 (tỷ)": "{:,.1f}"})
                         .background_gradient(subset=["Điểm"], cmap="Greens"),
                         hide_index=True, width="stretch", height=600)
            st.download_button("📥 Tải CSV", res.to_csv(index=False, encoding="utf-8-sig"),
                               f"loc_{datetime.now():%Y%m%d_%H%M}.csv", "text/csv")
            st.caption("Gợi ý: ưu tiên mã có Điểm cao **và** RS 3T dương (khỏe hơn thị trường) **và** Δ điểm 5P dương.")
        else:
            st.warning(f"Không có mã đạt tiêu chí (đã tải {len(hist)}/{len(pool)} mã).")

# ============================================================================
# 4) TIN TỨC & VĨ MÔ
# ============================================================================
elif mode == "📰 Tin tức & Vĩ mô":
    st.header("📰 Tin tức – Chính trị – Kinh tế")
    with st.spinner("Đang tải tin & dữ liệu vĩ mô..."):
        mk = news.market_news()
        macro = data.get_macro()
    if not macro.empty:
        st.subheader("🌍 Vĩ mô quốc tế")
        st.dataframe(macro.style.format({"Giá trị": "{:,.2f}", "1 ngày %": "{:+.2f}", "1 tháng %": "{:+.2f}"}),
                     hide_index=True, width="stretch")
        st.caption("Tác động thường gặp: DXY/USD-VND tăng → áp lực tỷ giá, khối ngoại bán; lợi suất Mỹ tăng → "
                   "định giá cổ phiếu giảm; dầu tăng → tốt cho GAS/PVS/PVD/BSR, xấu cho hàng không/vận tải; "
                   "S&P 500 giảm mạnh → tâm lý chung tiêu cực.")
    score_all = news.news_score(mk, 2)
    st.metric("Cảm xúc tin thị trường (2 ngày)", f"{score_all:+.2f}", help="-1 rất tiêu cực … +1 rất tích cực")
    t1, t2 = st.tabs(["🗞️ Dòng tin", "🏛️ Theo chủ đề vĩ mô"])
    with t1:
        c1, c2 = st.columns(2)
        srcs = c1.multiselect("Nguồn", sorted(mk["Nguồn"].unique()) if not mk.empty else [])
        kw = c2.text_input("Lọc từ khóa (vd: lãi suất, VHM, thuế)")
        d = mk
        if srcs:
            d = d[d["Nguồn"].isin(srcs)]
        if kw:
            d = d[d["Tiêu đề"].str.contains(kw, case=False) | d["Tóm tắt"].str.contains(kw, case=False)]
        for _, r in d.head(60).iterrows():
            icon = "🟢" if r["Cảm xúc"] > 0.2 else "🔴" if r["Cảm xúc"] < -0.2 else "⚪"
            t = r["Thời gian"].strftime("%d/%m %H:%M") if pd.notna(r["Thời gian"]) else ""
            st.markdown(f"{icon} **[{r['Tiêu đề']}]({r['Link']})**  \n<span class='small'>{t} · {r['Nguồn']}</span>",
                        unsafe_allow_html=True)
        if d.empty:
            st.info("Không có tin (nguồn RSS có thể tạm thời không truy cập được).")
    with t2:
        topics = news.macro_news()
        for name, d in topics.items():
            sc = news.news_score(d, 14)
            with st.expander(f"{name} · {len(d)} tin · cảm xúc {sc:+.2f}"):
                for _, r in d.head(8).iterrows():
                    t = r["Thời gian"].strftime("%d/%m") if pd.notna(r["Thời gian"]) else ""
                    st.markdown(f"- [{r['Tiêu đề']}]({r['Link']}) <span class='small'>{t}</span>",
                                unsafe_allow_html=True)

# ============================================================================
# 5) SO SÁNH
# ============================================================================
elif mode == "📊 So sánh":
    with st.sidebar:
        picks = st.multiselect("Chọn 2–8 mã", ALL_SYMBOLS, default=[s for s in ["FPT", "VCB", "HPG"] if s in SYMBOL_SET],
                               max_selections=8)
        period = st.selectbox("Thời gian", ["3mo", "6mo", "1y", "2y", "3y"], index=2)
    st.header("📊 So sánh cổ phiếu")
    if len(picks) >= 2:
        n = data.PERIOD_TO_BARS[period]
        bm = load_bench()
        closes, rows = ({"VNINDEX": bm["close"].tail(n)} if bm is not None else {}), []
        for s in picks:
            d, rg, _ = analyze(s, data.bars_for_period(period))
            if d is None:
                continue
            sig = generate_signal(d, rg)
            closes[s] = d["close"].tail(n)
            lt = d.iloc[-1]
            r = np.log(d["close"].tail(n)).diff()
            rows.append({"Mã": s, "Giá": lt["close"], "Lợi nhuận %": (d["close"].iloc[-1] / d["close"].tail(n).iloc[0] - 1) * 100,
                         "Biến động năm %": r.std() * np.sqrt(252) * 100,
                         "Max DD %": ((d["close"].tail(n) / d["close"].tail(n).cummax()) - 1).min() * 100,
                         "Điểm": sig["score"], "Tín hiệu": sig["signal"], "RSI": lt["RSI"],
                         "RS 3T": lt.get("RS_3M", np.nan)})
        st.dataframe(pd.DataFrame(rows).style.format({"Giá": "{:,.0f}", "Lợi nhuận %": "{:+.1f}",
                                                      "Biến động năm %": "{:.0f}", "Max DD %": "{:.1f}",
                                                      "Điểm": "{:.0f}", "RSI": "{:.0f}", "RS 3T": "{:+.1f}"}),
                     hide_index=True, width="stretch")
        st.plotly_chart(charts.compare_chart(closes), width="stretch")
        st.plotly_chart(charts.corr_heatmap({k: v for k, v in closes.items()}), width="stretch")
        st.caption("Tương quan thấp giữa các mã giúp danh mục đa dạng hóa tốt hơn.")
    else:
        st.info("Chọn ít nhất 2 mã ở thanh bên.")

# ============================================================================
# 6) BACKTEST
# ============================================================================
else:
    with st.sidebar:
        symbol = st.selectbox("Mã", ALL_SYMBOLS, index=ALL_SYMBOLS.index("FPT") if "FPT" in SYMBOL_SET else 0)
        years = st.select_slider("Số năm", [2, 3, 5], value=3)
        capital = st.number_input("Vốn (VND)", 10_000_000, 10_000_000_000, 100_000_000, 10_000_000)
        buy_th = st.slider("Mua khi điểm ≥", 55, 90, 70)
        sell_th = st.slider("Bán khi điểm <", 20, 60, 45)
        atr_stop = st.slider("Trailing stop (× ATR)", 1.0, 5.0, 2.5, 0.5)
    st.header(f"📈 Backtest chiến lược điểm kỹ thuật · {symbol}")
    st.caption("Khớp giá mở cửa phiên sau tín hiệu · phí mua 0.15%, bán 0.25% (gồm thuế) · T+2 · lô 100 cp.")
    d, rg, _ = analyze(symbol, 252 * years + 220)
    res = run_backtest(d, rg, capital, buy_th, sell_th, atr_stop) if d is not None else None
    if res:
        s, b = res["strategy"], res["benchmark"]
        k = st.columns(4)
        k[0].metric("Lợi nhuận chiến lược", f"{s['Tổng LN %']:+.1f}%", f"{s['Tổng LN %'] - b['Tổng LN %']:+.1f}% vs mua&giữ")
        k[1].metric("Max Drawdown", f"{s['Max Drawdown %']:.1f}%", f"mua&giữ {b['Max Drawdown %']:.1f}%",
                    delta_color="off")
        k[2].metric("Tỷ lệ thắng", fmt(res["win_rate"], 0, "%"), f"{res['n_trades']} lệnh", delta_color="off")
        k[3].metric("Sharpe", fmt(s["Sharpe"], 2), f"mua&giữ {fmt(b['Sharpe'], 2)}", delta_color="off")
        st.caption(f"Thời gian có vị thế: {res['exposure']:.0f}% · LN TB/lệnh: {fmt(res['avg_trade'], 2, '%')}")
        st.plotly_chart(charts.equity_chart(res), width="stretch")
        st.dataframe(pd.DataFrame({"Chiến lược": s, "Mua & giữ": b}).T.style.format("{:.2f}"),
                     width="stretch")
        if not res["trades"].empty:
            with st.expander("Lịch sử lệnh"):
                t = res["trades"].copy()
                t["Ngày"] = pd.to_datetime(t["Ngày"]).dt.strftime("%d/%m/%Y")
                st.dataframe(t, hide_index=True, width="stretch")
        st.info("Kết quả quá khứ không đảm bảo tương lai. Nếu chiến lược thua mua&giữ nhưng drawdown thấp hơn nhiều, "
                "nó vẫn có giá trị như công cụ **quản trị rủi ro**.")
    else:
        st.error("Không đủ dữ liệu để backtest.")

# ============================================================================
# Footer
# ============================================================================
st.divider()
st.caption(DISCLAIMER)
sm = tracking.summary()
st.caption(f"👥 Online {sm['online']} · Lượt truy cập {sm['total_visits']:,} · Hôm nay {sm['today']} · "
           f"Tìm kiếm {sm['searches']:,} · Cập nhật {datetime.now():%H:%M %d/%m/%Y} · "
           "Nguồn: Vietcap, VNDirect, Yahoo Finance, CafeF, VnExpress, Vietstock, Google News")
