"""
Dự báo – làm lại để TRUNG THỰC và kiểm chứng được:

1) ML xu hướng: target = lợi nhuận N phiên tới > 0. Kiểm định walk-forward
   (TimeSeriesSplit có 'gap' = N để không rò rỉ tương lai). Hiển thị độ chính xác
   ngoài mẫu so với baseline. Bản cũ: gán nhãn sai cho N phiên cuối, không kiểm định,
   lấy predict_proba làm "độ tin cậy" -> con số đẹp nhưng không có ý nghĩa.

2) Dự báo giá: thay đa thức bậc 3 + nhiễu ngẫu nhiên (mỗi lần F5 ra kết quả khác)
   bằng 'nón xác suất' từ phân phối lợi nhuận thực tế (bootstrap theo khối),
   có giới hạn biên độ sàn. Cho ra vùng giá 50%/80%/95% và xác suất chạm mục tiêu/cắt lỗ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PRICE_LIMIT

FEATURES = [
    "RSI", "StochRSI", "ROC10", "ROC20", "ATR_pct", "BB_pctB", "BB_width", "ADX", "CMF", "MFI",
    "Volume_ratio", "HV20", "Dist_High52", "RS_1M", "RS_3M",
]


def _feature_frame(df: pd.DataFrame, regime: pd.Series | None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for f in FEATURES:
        if f in df:
            X[f] = df[f]
    c = df["close"]
    X["MACDh_pct"] = df["MACD_hist"] / c * 100
    X["DI_diff"] = df["DI_plus"] - df["DI_minus"]
    for n in (20, 50, 200):
        X[f"dist_MA{n}"] = (c / df[f"MA{n}"] - 1) * 100
    X["ret1"] = c.pct_change() * 100
    X["ret5"] = c.pct_change(5) * 100
    X["st_dir"] = df["Supertrend_direction"]
    if regime is not None:
        X["regime"] = regime.reindex(df.index).ffill()
    return X.replace([np.inf, -np.inf], np.nan)


def ml_trend(df: pd.DataFrame, horizon: int = 10, regime: pd.Series | None = None) -> dict | None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    if df is None or len(df) < 250:
        return None
    X = _feature_frame(df, regime)
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    y = (fwd > 0).astype(float).where(fwd.notna())  # N phiên cuối = NaN (chưa biết) -> loại khỏi huấn luyện
    data = X.join(y.rename("y")).dropna(subset=["y"])
    data = data.dropna(thresh=int(len(X.columns) * 0.8))
    if len(data) < 200:
        return None
    Xd, yd = data.drop(columns="y"), data["y"].astype(int)

    def model():
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                              l2_regularization=1.0, min_samples_leaf=20, random_state=42)

    tss = TimeSeriesSplit(n_splits=5, gap=horizon)
    accs, bases, aucs = [], [], []
    for tr, te in tss.split(Xd):
        if len(np.unique(yd.iloc[tr])) < 2:
            continue
        m = model().fit(Xd.iloc[tr], yd.iloc[tr])
        p = m.predict_proba(Xd.iloc[te])[:, 1]
        yt = yd.iloc[te]
        accs.append(((p > 0.5).astype(int) == yt).mean())
        bases.append(max(yt.mean(), 1 - yt.mean()))
        if len(np.unique(yt)) == 2:
            aucs.append(roc_auc_score(yt, p))
    if not accs:
        return None
    final = model().fit(Xd, yd)
    x_last = X.iloc[[-1]]
    prob_up = float(final.predict_proba(x_last)[0, 1])
    acc, base = float(np.mean(accs)), float(np.mean(bases))
    auc = float(np.mean(aucs)) if aucs else np.nan
    edge = acc - base
    reliable = edge > 0.03 and (np.isnan(auc) or auc > 0.55)
    return {
        "horizon": horizon, "prob_up": prob_up,
        "direction": "TĂNG" if prob_up >= 0.5 else "GIẢM",
        "oos_accuracy": acc, "baseline": base, "auc": auc, "edge": edge, "reliable": reliable,
        "n_samples": len(Xd),
    }


def price_cone(df: pd.DataFrame, horizon: int = 20, exchange: str = "HOSE", n_paths: int = 3000,
               targets: tuple[float, float] | None = None, seed: int = 7) -> dict | None:
    """Bootstrap theo khối 5 phiên từ 2 năm lợi nhuận gần nhất, khử drift về mức thận trọng."""
    if df is None or len(df) < 120:
        return None
    rets = np.log(df["close"]).diff().dropna().tail(500).values
    # Khử drift lịch sử (quá khứ tăng mạnh không có nghĩa tương lai sẽ vậy), giữ 30% drift
    rets = rets - rets.mean() * 0.7
    # Điều chỉnh độ biến động về mức hiện tại (EWMA) để phản ánh rủi ro hiện thời
    ew_vol = pd.Series(rets).ewm(alpha=0.06).std().iloc[-1]
    hist_vol = rets.std()
    if hist_vol > 0 and np.isfinite(ew_vol):
        rets = rets * np.clip(ew_vol / hist_vol, 0.6, 1.8)
    lim = PRICE_LIMIT.get(exchange, 0.07)
    rng = np.random.default_rng(seed)
    block = 5
    n_blocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, len(rets) - block, size=(n_paths, n_blocks))
    paths = np.concatenate([rets[s:s + block][None, :] for s in starts.ravel()], axis=0)
    paths = paths.reshape(n_paths, n_blocks * block)[:, :horizon]
    paths = np.clip(paths, np.log(1 - lim), np.log(1 + lim))
    p0 = float(df["close"].iloc[-1])
    prices = p0 * np.exp(np.cumsum(paths, axis=1))
    q = np.percentile(prices, [2.5, 10, 25, 50, 75, 90, 97.5], axis=0)
    dates = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=horizon)
    out = {"dates": dates, "p0": p0,
           "q": {k: q[i] for i, k in enumerate(["p2", "p10", "p25", "p50", "p75", "p90", "p97"])},
           "prob_up": float((prices[:, -1] > p0).mean())}
    if targets:
        tgt, stop = targets
        hit_t = (prices >= tgt).any(axis=1)
        hit_s = (prices <= stop).any(axis=1)
        first_t = np.where(hit_t, (prices >= tgt).argmax(axis=1), 10 ** 6)
        first_s = np.where(hit_s, (prices <= stop).argmax(axis=1), 10 ** 6)
        out["p_target_first"] = float((first_t < first_s).mean())
        out["p_stop_first"] = float((first_s < first_t).mean())
    return out
