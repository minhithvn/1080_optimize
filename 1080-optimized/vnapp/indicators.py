"""
Chỉ báo kỹ thuật – đã sửa công thức theo chuẩn (Wilder RSI/ATR/ADX, Supertrend
có final band, VWAP theo cửa sổ thay vì cộng dồn cả lịch sử...).
Toàn bộ vector hóa; Supertrend dùng vòng lặp numpy (nhanh hơn ~50 lần bản cũ dùng .iloc).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _safe_div(a, b):
    return a / b.replace(0, np.nan) if isinstance(b, pd.Series) else a / (b if b else np.nan)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = _wilder(d.clip(lower=0), n), _wilder(-d.clip(upper=0), n)
    rs = _safe_div(up, dn)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def supertrend(df: pd.DataFrame, atr: pd.Series, mult: float = 3.0):
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    a = atr.values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    n = len(df)
    fub, flb = ub.copy(), lb.copy()
    st, dirn = np.full(n, np.nan), np.zeros(n, dtype=int)
    for i in range(1, n):
        if np.isnan(a[i]):
            continue
        if np.isnan(fub[i - 1]):
            fub[i - 1], flb[i - 1] = ub[i], lb[i]
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        prev = dirn[i - 1] if dirn[i - 1] != 0 else 1
        if prev == 1:
            dirn[i] = -1 if c[i] < flb[i] else 1
        else:
            dirn[i] = 1 if c[i] > fub[i] else -1
        st[i] = flb[i] if dirn[i] == 1 else fub[i]
    return pd.Series(st, index=df.index), pd.Series(dirn, index=df.index).replace(0, np.nan)


def add_indicators(df: pd.DataFrame, bench: pd.DataFrame | None = None) -> pd.DataFrame:
    """Thêm ~45 chỉ báo cốt lõi. `bench` = VNINDEX để tính sức mạnh tương đối."""
    if df is None or df.empty:
        return df
    df = df.copy()
    c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]

    # --- Trung bình động
    for n in (5, 10, 20, 50, 100, 200):
        df[f"MA{n}"] = c.rolling(n).mean()
    for n in (12, 26, 50):
        df[f"EMA{n}"] = c.ewm(span=n, adjust=False).mean()
    df["VWMA20"] = (c * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # --- Động lượng
    df["RSI"] = rsi(c)
    rmin, rmax = df["RSI"].rolling(14).min(), df["RSI"].rolling(14).max()
    df["StochRSI"] = 100 * _safe_div(df["RSI"] - rmin, rmax - rmin)
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]
    df["ROC10"] = c.pct_change(10) * 100
    df["ROC20"] = c.pct_change(20) * 100
    l14, h14 = l.rolling(14).min(), h.rolling(14).max()
    df["Stoch_K"] = 100 * _safe_div(c - l14, h14 - l14)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    df["Williams_R"] = -100 * _safe_div(h14 - c, h14 - l14)
    tp = (h + l + c) / 3
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["CCI"] = _safe_div(tp - tp.rolling(20).mean(), 0.015 * md)

    # --- Biến động
    tr = true_range(df)
    df["ATR"] = _wilder(tr, 14)
    df["ATR_pct"] = df["ATR"] / c * 100
    df["BB_middle"] = df["MA20"]
    sd = c.rolling(20).std()
    df["BB_upper"], df["BB_lower"] = df["MA20"] + 2 * sd, df["MA20"] - 2 * sd
    df["BB_width"] = _safe_div(df["BB_upper"] - df["BB_lower"], df["MA20"])
    df["BB_pctB"] = _safe_div(c - df["BB_lower"], df["BB_upper"] - df["BB_lower"])
    df["Keltner_upper"] = df["EMA12"].ewm(span=20).mean() + 2 * df["ATR"]
    df["Donchian_upper"], df["Donchian_lower"] = h.rolling(20).max(), l.rolling(20).min()
    df["HV20"] = np.log(c).diff().rolling(20).std() * np.sqrt(252) * 100  # biến động năm hóa %

    # --- ADX (Wilder, DM chuẩn)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    atr_w = df["ATR"]
    df["DI_plus"] = 100 * _safe_div(_wilder(pdm, 14), atr_w)
    df["DI_minus"] = 100 * _safe_div(_wilder(mdm, 14), atr_w)
    dx = 100 * _safe_div((df["DI_plus"] - df["DI_minus"]).abs(), df["DI_plus"] + df["DI_minus"])
    df["ADX"] = _wilder(dx, 14)

    # --- Supertrend
    df["Supertrend"], df["Supertrend_direction"] = supertrend(df, df["ATR"], 3.0)

    # --- Khối lượng & dòng tiền
    df["Volume_MA"] = v.rolling(20).mean()
    df["Volume_ratio"] = _safe_div(v, df["Volume_MA"].shift(1))
    df["Value_bn"] = c * v / 1e9
    df["Value_MA20_bn"] = df["Value_bn"].rolling(20).mean()
    df["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20).mean()
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(), 0).rolling(14).sum()
    neg = rmf.where(tp < tp.shift(), 0).rolling(14).sum()
    df["MFI"] = 100 - 100 / (1 + _safe_div(pos, neg))
    mfm = _safe_div((c - l) - (h - c), h - l).fillna(0)
    df["CMF"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    df["VWAP20"] = (tp * v).rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)

    # --- Ichimoku
    df["Ichimoku_conversion"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    df["Ichimoku_base"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    df["Ichimoku_span_a"] = ((df["Ichimoku_conversion"] + df["Ichimoku_base"]) / 2).shift(26)
    df["Ichimoku_span_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)

    # --- Pivot (theo phiên trước)
    p = (h.shift() + l.shift() + c.shift()) / 3
    df["Pivot"], df["R1"], df["S1"] = p, 2 * p - l.shift(), 2 * p - h.shift()
    df["R2"], df["S2"] = p + (h.shift() - l.shift()), p - (h.shift() - l.shift())

    # --- Nến
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    df["Doji"] = (body / rng < 0.1).astype(int)
    df["Hammer"] = ((lower > 2 * body) & (upper < body) & (c < df["MA20"])).astype(int)  # chỉ có nghĩa sau nhịp giảm
    df["Shooting_Star"] = ((upper > 2 * body) & (lower < body) & (c > df["MA20"])).astype(int)
    po, pc = o.shift(), c.shift()
    df["Bullish_Engulfing"] = ((c > o) & (pc < po) & (c >= po) & (o <= pc) & (body > (pc - po).abs())).astype(int)
    df["Bearish_Engulfing"] = ((c < o) & (pc > po) & (c <= po) & (o >= pc) & (body > (pc - po).abs())).astype(int)

    # --- Vị thế 52 tuần
    df["High52"], df["Low52"] = h.rolling(252, min_periods=60).max(), l.rolling(252, min_periods=60).min()
    df["Dist_High52"] = (c / df["High52"] - 1) * 100

    # --- Sức mạnh tương đối so với VNINDEX
    if bench is not None and not bench.empty:
        b = bench["close"].reindex(df.index).ffill()
        ratio = c / b
        df["RS_line"] = ratio / ratio.dropna().iloc[0] * 100 if ratio.notna().any() else np.nan
        # RS 3 tháng: chênh lệch lợi nhuận so với chỉ số (điểm %)
        df["RS_3M"] = (c.pct_change(63) - b.pct_change(63)) * 100
        df["RS_1M"] = (c.pct_change(21) - b.pct_change(21)) * 100
    return df


def candlestick_today(df: pd.DataFrame) -> list[str]:
    if df is None or len(df) < 2:
        return []
    last = df.iloc[-1]
    names = {"Doji": "Doji", "Hammer": "Búa (Hammer)", "Shooting_Star": "Sao băng",
             "Bullish_Engulfing": "Nhấn chìm tăng", "Bearish_Engulfing": "Nhấn chìm giảm"}
    return [v for k, v in names.items() if last.get(k, 0) == 1]


def support_resistance(df: pd.DataFrame, lookback: int = 120, tol: float = 0.015) -> dict:
    """Hỗ trợ/kháng cự từ các đỉnh/đáy swing thực tế (gom cụm), thay vì lấy BB/quantile."""
    d = df.tail(lookback)
    price = d["close"].iloc[-1]
    hi, lo = d["high"].values, d["low"].values
    pivots = []
    w = 3
    for i in range(w, len(d) - w):
        if hi[i] == hi[i - w:i + w + 1].max():
            pivots.append(hi[i])
        if lo[i] == lo[i - w:i + w + 1].min():
            pivots.append(lo[i])
    pivots = sorted(pivots)
    clusters: list[list[float]] = []
    for p in pivots:
        if clusters and abs(p / np.mean(clusters[-1]) - 1) < tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    levels = [(float(np.mean(cl)), len(cl)) for cl in clusters]
    res = sorted([lv for lv in levels if lv[0] > price * 1.003], key=lambda x: x[0])[:3]
    sup = sorted([lv for lv in levels if lv[0] < price * 0.997], key=lambda x: -x[0])[:3]
    last = df.iloc[-1]
    if not res:
        res = [(float(last.get("High52", d["high"].max())), 1)]
    if not sup:
        sup = [(float(d["low"].min()), 1)]
    return {"current": float(price), "resistance": res, "support": sup,
            "pivot": float(last.get("Pivot", np.nan)), "R1": float(last.get("R1", np.nan)),
            "S1": float(last.get("S1", np.nan))}
