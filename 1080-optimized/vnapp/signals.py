"""
Chấm điểm kỹ thuật 0–100, vector hóa cho mọi phiên (dùng chung cho phân tích,
quét thị trường và backtest -> kết quả backtest phản ánh đúng logic đang khuyến nghị).

Nhóm điểm (cộng quanh mốc 50):
  Xu hướng ±20 | Động lượng ±15 | Sức mạnh xu hướng ±8 | Dòng tiền ±12
  Sức mạnh tương đối vs VNINDEX ±8 | Bối cảnh thị trường ±8 | Tin tức ±5 | Phạt rủi ro
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BANDS = [(75, "MUA MẠNH"), (62, "MUA"), (45, "THEO DÕI / NẮM GIỮ"), (35, "GIẢM TỶ TRỌNG"), (0, "BÁN")]


def label_for(score: float) -> str:
    for th, name in BANDS:
        if score >= th:
            return name
    return "BÁN"


def _pm(cond_pos, cond_neg, pts):
    return np.where(cond_pos, pts, np.where(cond_neg, -pts, 0))


def market_regime(bench: pd.DataFrame | None) -> pd.Series | None:
    """+1 thị trường tăng, 0 trung tính, -1 thị trường giảm (VNINDEX vs MA50/MA200)."""
    if bench is None or bench.empty or len(bench) < 60:
        return None
    c = bench["close"]
    ma50, ma200 = c.rolling(50).mean(), c.rolling(200, min_periods=120).mean()
    up = (c > ma50) & (ma50 > ma200)
    dn = (c < ma50) & (ma50 < ma200)
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=bench.index)


def score_frame(df: pd.DataFrame, regime: pd.Series | None = None) -> pd.DataFrame:
    c = df["close"]
    comp = pd.DataFrame(index=df.index)
    comp["Xu hướng"] = (
        _pm(c > df["MA20"], c < df["MA20"], 4)
        + _pm(df["MA20"] > df["MA50"], df["MA20"] < df["MA50"], 4)
        + _pm(df["MA50"] > df["MA200"], df["MA50"] < df["MA200"], 4)
        + _pm(c > df["MA200"], c < df["MA200"], 3)
        + _pm(df["Supertrend_direction"] == 1, df["Supertrend_direction"] == -1, 5)
    )
    r = df["RSI"]
    rsi_pts = np.select(
        [r > 80, r > 70, r >= 50, r >= 40, r >= 30, r < 30],
        [-6, 2, 6, 1, -3, -6], 0)
    comp["Động lượng"] = (
        rsi_pts
        + _pm(df["MACD"] > df["MACD_signal"], df["MACD"] < df["MACD_signal"], 4)
        + _pm(df["MACD_hist"] > df["MACD_hist"].shift(), df["MACD_hist"] < df["MACD_hist"].shift(), 2)
        + _pm(df["ROC20"] > 0, df["ROC20"] < 0, 3)
    )
    strong = df["ADX"] > 25
    comp["Sức mạnh XH"] = _pm(strong & (df["DI_plus"] > df["DI_minus"]), strong & (df["DI_minus"] > df["DI_plus"]), 8)
    up_day = c > c.shift()
    spike = df["Volume_ratio"] > 1.5
    comp["Dòng tiền"] = (
        _pm(df["CMF"] > 0.05, df["CMF"] < -0.05, 4)
        + _pm(df["OBV"] > df["OBV_MA20"], df["OBV"] < df["OBV_MA20"], 3)
        + _pm(spike & up_day, spike & ~up_day, 5)
    )
    if "RS_3M" in df:
        rs = df["RS_3M"]
        comp["Sức mạnh tương đối"] = np.select([rs > 10, rs > 0, rs < -10, rs < 0], [8, 4, -8, -4], 0)
    else:
        comp["Sức mạnh tương đối"] = 0
    if regime is not None:
        comp["Thị trường chung"] = regime.reindex(df.index).ffill().fillna(0).values * 8
    else:
        comp["Thị trường chung"] = 0
    ext = (c / df["MA20"] - 1) * 100
    illiquid = df["Value_MA20_bn"] < 1.0
    comp["Rủi ro"] = np.where(ext > 15, -5, 0) + np.where(illiquid, -5, 0) + np.where(df["ATR_pct"] > 6, -3, 0)
    comp = comp.astype(float)
    comp["score"] = (50 + comp.sum(axis=1)).clip(0, 100)
    warm = df["MA50"].isna() | df["ADX"].isna()
    comp.loc[warm, "score"] = np.nan
    return comp


def explain(df: pd.DataFrame, comp: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Diễn giải bằng lời cho phiên cuối."""
    last, pos, neg = df.iloc[-1], [], []
    c = last["close"]

    def add(cond, good, bad=None):
        if cond is True:
            pos.append(good)
        elif cond is False and bad:
            neg.append(bad)

    add(bool(last["MA20"] > last["MA50"] > last["MA200"]) or None, "MA20 > MA50 > MA200: xu hướng tăng chuẩn")
    if last["MA20"] < last["MA50"] < last["MA200"]:
        neg.append("MA20 < MA50 < MA200: xu hướng giảm dài hạn")
    add(bool(c > last["MA200"]), "Giá trên MA200 (xu hướng dài hạn tăng)", "Giá dưới MA200 (xu hướng dài hạn yếu)")
    add(bool(last["Supertrend_direction"] == 1), "Supertrend đang báo TĂNG", "Supertrend đang báo GIẢM")
    rsi = last["RSI"]
    if rsi > 80:
        neg.append(f"RSI {rsi:.0f}: quá mua mạnh, dễ điều chỉnh")
    elif rsi > 70:
        pos.append(f"RSI {rsi:.0f}: động lượng mạnh nhưng đã cao – không mua đuổi")
    elif rsi >= 50:
        pos.append(f"RSI {rsi:.0f}: vùng động lượng tích cực")
    elif rsi < 30:
        neg.append(f"RSI {rsi:.0f}: quá bán – chờ tín hiệu đảo chiều, chưa bắt đáy")
    add(bool(last["MACD"] > last["MACD_signal"]), "MACD trên đường tín hiệu", "MACD dưới đường tín hiệu")
    if last["ADX"] > 25:
        (pos if last["DI_plus"] > last["DI_minus"] else neg).append(
            f"ADX {last['ADX']:.0f}: xu hướng {'tăng' if last['DI_plus'] > last['DI_minus'] else 'giảm'} có lực")
    else:
        neg.append(f"ADX {last['ADX']:.0f}: thị trường đi ngang, tín hiệu xu hướng kém tin cậy")
    add(bool(last["CMF"] > 0.05) if abs(last["CMF"]) > 0.05 else None,
        f"CMF {last['CMF']:.2f}: dòng tiền vào", f"CMF {last['CMF']:.2f}: dòng tiền ra")
    if last["Volume_ratio"] > 1.5:
        (pos if c > df["close"].iloc[-2] else neg).append(
            f"Khối lượng đột biến x{last['Volume_ratio']:.1f} trong phiên {'tăng' if c > df['close'].iloc[-2] else 'giảm'}")
    if "RS_3M" in df and pd.notna(last.get("RS_3M")):
        add(bool(last["RS_3M"] > 0), f"Mạnh hơn VNINDEX {last['RS_3M']:+.1f}% (3 tháng)",
            f"Yếu hơn VNINDEX {last['RS_3M']:+.1f}% (3 tháng)")
    if comp["Thị trường chung"].iloc[-1] > 0:
        pos.append("VNINDEX đang trong xu hướng tăng (hỗ trợ cổ phiếu)")
    elif comp["Thị trường chung"].iloc[-1] < 0:
        neg.append("VNINDEX đang trong xu hướng giảm – ưu tiên quản trị rủi ro")
    if last["Value_MA20_bn"] < 1:
        neg.append(f"Thanh khoản thấp ({last['Value_MA20_bn']:.2f} tỷ/phiên) – khó mua/bán khối lượng lớn")
    if (c / last["MA20"] - 1) > 0.15:
        neg.append("Giá cách MA20 > 15%: đã chạy nóng")
    return pos, neg


def generate_signal(df: pd.DataFrame, regime: pd.Series | None = None, news_score: float = 0.0) -> dict:
    if df is None or len(df) < 60:
        return {"signal": "N/A", "score": np.nan, "pos": [], "neg": ["Không đủ dữ liệu (cần ≥ 60 phiên)"],
                "components": {}}
    comp = score_frame(df, regime)
    s = comp["score"].iloc[-1]
    news_pts = float(np.clip(news_score * 5, -5, 5))
    if pd.notna(s):
        s = float(np.clip(s + news_pts, 0, 100))
    pos, neg = explain(df, comp)
    if news_pts >= 2:
        pos.append("Tin tức gần đây nghiêng tích cực")
    elif news_pts <= -2:
        neg.append("Tin tức gần đây nghiêng tiêu cực – kiểm tra kỹ nội dung")
    components = comp.drop(columns="score").iloc[-1].to_dict()
    components["Tin tức"] = news_pts
    # độ ổn định: điểm 5 phiên gần nhất
    trend5 = comp["score"].tail(5).diff().sum()
    return {"signal": label_for(s) if pd.notna(s) else "N/A", "score": s, "pos": pos, "neg": neg,
            "components": components, "score_series": comp["score"], "score_change_5d": trend5}


def trade_plan(df: pd.DataFrame, sig: dict, exchange: str = "HOSE") -> dict:
    """Kế hoạch giao dịch dựa trên ATR & hỗ trợ/kháng cự – điểm mua, cắt lỗ, mục tiêu, R:R."""
    from .indicators import support_resistance

    last = df.iloc[-1]
    price, atr = float(last["close"]), float(last["ATR"])
    sr = support_resistance(df)
    sup1 = sr["support"][0][0]
    res1 = sr["resistance"][0][0]
    entry_lo = max(sup1, price - 0.5 * atr)
    entry_hi = price
    stop = min(entry_lo - 1.0 * atr, price - 2 * atr)
    stop = max(stop, price * 0.88)  # không để cắt lỗ quá xa >12%
    risk = price - stop
    t1 = max(res1, price + 1.5 * risk)
    t2 = price + 3 * risk
    rr = (t1 - price) / risk if risk > 0 else np.nan
    trend_up = last["MA20"] > last["MA50"] and last["Supertrend_direction"] == 1
    horizon = "Trung hạn 1–3 tháng" if (trend_up and last["ADX"] > 25) else (
        "Ngắn hạn 1–3 tuần" if trend_up else "Đứng ngoài / chờ xác nhận")
    return {"price": price, "entry": (entry_lo, entry_hi), "stop": stop, "t1": t1, "t2": t2, "rr": rr,
            "horizon": horizon, "trailing": float(last["Supertrend"]) if last["Supertrend_direction"] == 1 else None,
            "sr": sr}
