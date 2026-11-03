"""
Backtest thực tế hơn bản cũ:
- Tín hiệu tính ở phiên T, khớp lệnh giá MỞ CỬA phiên T+1 (không nhìn trước tương lai)
- Phí mua 0.15%, phí bán 0.15% + thuế 0.1%
- T+2: không được bán trước khi cổ phiếu về
- Lô chẵn 100 cổ phiếu, cắt lỗ theo ATR, thoát khi điểm < ngưỡng
- Tính indicator 1 lần (bản cũ tính lại toàn bộ ở mỗi phiên -> O(n²), rất chậm)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEE_BUY, FEE_SELL, SETTLEMENT_DAYS
from .signals import score_frame


def run_backtest(df: pd.DataFrame, regime=None, capital: float = 100_000_000, buy_th: float = 70,
                 sell_th: float = 45, atr_stop: float = 2.5, start: int = 200) -> dict | None:
    if df is None or len(df) < start + 30:
        return None
    score = score_frame(df, regime)["score"].values
    o, c, atr = df["open"].values, df["close"].values, df["ATR"].values
    cash, shares, entry_i, stop = capital, 0, -1, 0.0
    equity, trades, in_pos = [], [], []
    pending = None
    for i in range(start, len(df)):
        # 1) khớp lệnh chờ ở giá mở cửa hôm nay
        if pending == "BUY" and shares == 0:
            px = o[i]
            qty = int(cash / (px * (1 + FEE_BUY)) // 100 * 100)
            if qty > 0:
                cash -= qty * px * (1 + FEE_BUY)
                shares, entry_i, stop = qty, i, px - atr_stop * atr[i - 1]
                trades.append({"Ngày": df.index[i], "Lệnh": "MUA", "Giá": px, "SL": qty})
        elif pending == "SELL" and shares > 0:
            px = o[i]
            cash += shares * px * (1 - FEE_SELL)
            buy_px = trades[-1]["Giá"]
            trades.append({"Ngày": df.index[i], "Lệnh": "BÁN", "Giá": px, "SL": shares,
                           "Lãi/lỗ %": (px * (1 - FEE_SELL) / (buy_px * (1 + FEE_BUY)) - 1) * 100})
            shares = 0
        pending = None
        # 2) trailing stop
        if shares > 0 and np.isfinite(atr[i]):
            stop = max(stop, c[i] - atr_stop * atr[i])
        # 3) sinh lệnh cho phiên sau
        s = score[i]
        if shares == 0 and np.isfinite(s) and s >= buy_th:
            pending = "BUY"
        elif shares > 0 and (i - entry_i) >= SETTLEMENT_DAYS and ((np.isfinite(s) and s < sell_th) or c[i] < stop):
            pending = "SELL"
        equity.append(cash + shares * c[i])
        in_pos.append(shares > 0)

    eq = pd.Series(equity, index=df.index[start:])
    bh = c[start:] / c[start] * capital
    bh = pd.Series(bh, index=eq.index)

    def stats(series):
        r = series.pct_change().dropna()
        years = max(len(series) / 252, 1e-9)
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1
        mdd = (series / series.cummax() - 1).min()
        sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
        return {"Tổng LN %": (series.iloc[-1] / series.iloc[0] - 1) * 100, "CAGR %": cagr * 100,
                "Max Drawdown %": mdd * 100, "Sharpe": sharpe}

    closed = [t for t in trades if t["Lệnh"] == "BÁN"]
    wins = [t for t in closed if t["Lãi/lỗ %"] > 0]
    exposure = float(np.mean(in_pos)) * 100 if in_pos else 0.0
    return {"equity": eq, "buy_hold": bh, "trades": pd.DataFrame(trades),
            "strategy": stats(eq), "benchmark": stats(bh),
            "n_trades": len(closed), "win_rate": len(wins) / len(closed) * 100 if closed else np.nan,
            "avg_trade": float(np.mean([t["Lãi/lỗ %"] for t in closed])) if closed else np.nan,
            "exposure": exposure}
