"""Kiểm thử offline (không cần mạng): python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vnapp import data, news  # noqa: E402
from vnapp.backtest import run_backtest  # noqa: E402
from vnapp.forecast import ml_trend, price_cone  # noqa: E402
from vnapp.indicators import add_indicators, rsi, support_resistance  # noqa: E402
from vnapp.signals import generate_signal, market_regime, trade_plan  # noqa: E402


def make_ohlc(n=900, drift=0.0004, vol=0.02, seed=1, p0=50000):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = p0 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, vol / 3, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.integers(200_000, 2_000_000, n).astype(float)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


def test_rsi_bounds_and_trend():
    up = pd.Series(np.linspace(10, 20, 100))
    assert rsi(up).iloc[-1] == 100
    s = make_ohlc()["close"]
    r = rsi(s).dropna()
    assert r.between(0, 100).all()


def test_indicators_and_signal():
    df = make_ohlc()
    bench = make_ohlc(seed=2, p0=1200)
    di = add_indicators(df, bench)
    for col in ["RSI", "ADX", "Supertrend", "Supertrend_direction", "MFI", "CMF", "RS_3M", "ATR"]:
        assert di[col].notna().sum() > 500, col
    assert set(di["Supertrend_direction"].dropna().unique()) <= {1, -1}
    # supertrend: khi hướng tăng, đường ST nằm dưới giá đóng cửa
    upm = di["Supertrend_direction"] == 1
    assert (di.loc[upm, "Supertrend"] <= di.loc[upm, "close"] * 1.0001).mean() > 0.99
    sig = generate_signal(di, market_regime(bench), 0.3)
    assert 0 <= sig["score"] <= 100 and sig["signal"] != "N/A"
    plan = trade_plan(di, sig)
    assert plan["stop"] < plan["price"] < plan["t1"] <= plan["t2"] or plan["t1"] >= plan["price"]
    sr = support_resistance(di)
    assert all(lv > sr["current"] for lv, _ in sr["resistance"])


def test_ml_no_edge_on_random_walk():
    df = add_indicators(make_ohlc(n=1200, drift=0.0, seed=5))
    res = ml_trend(df, 10)
    assert res is not None
    assert 0 <= res["prob_up"] <= 1
    # Bước ngẫu nhiên: không mô hình trung thực nào được vượt trội đáng kể
    assert res["oos_accuracy"] < res["baseline"] + 0.08


def test_cone_deterministic_and_limits():
    df = make_ohlc()
    a = price_cone(df, 20, "HOSE", targets=(df["close"].iloc[-1] * 1.1, df["close"].iloc[-1] * 0.93))
    b = price_cone(df, 20, "HOSE")
    assert np.allclose(a["q"]["p50"], b["q"]["p50"])  # không còn nhiễu ngẫu nhiên mỗi lần chạy
    p0 = df["close"].iloc[-1]
    assert a["q"]["p97"][0] <= p0 * 1.07 + 1 and a["q"]["p2"][0] >= p0 * 0.93 - 1
    assert 0 <= a["p_target_first"] <= 1


def test_backtest_runs():
    df = make_ohlc(drift=0.001)
    bench = make_ohlc(seed=3, p0=1200, drift=0.0005)
    res = run_backtest(add_indicators(df, bench), market_regime(bench))
    assert res is not None and len(res["equity"]) > 100
    assert np.isfinite(res["strategy"]["Tổng LN %"])


def test_price_unit_normalize():
    df = make_ohlc(n=30, p0=60)  # giá kiểu nghìn đồng
    out = data._finalize(df.copy())
    assert out["close"].median() > 10000


class FakeResp:
    def __init__(self, payload=None, content=b""):
        self._p, self.content, self.encoding = payload, content, "utf-8"

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def test_vci_parser(monkeypatch):
    t = [int(pd.Timestamp("2025-01-02").timestamp()) + i * 86400 for i in range(5)]
    payload = [{"symbol": "VCB", "o": [90] * 5, "h": [91] * 5, "l": [89] * 5, "c": [90.5] * 5, "v": [1e6] * 5, "t": t}]
    monkeypatch.setattr(data._session, "post", lambda *a, **k: FakeResp(payload))
    out = data._vci_ohlc(["VCB"], 5)
    df = data._finalize(out["VCB"])
    assert len(df) == 5 and df["close"].iloc[-1] == 90500


def test_rss_and_sentiment():
    xml = """<?xml version='1.0' encoding='utf-8'?><rss><channel>
    <item><title>HPG lai ky luc, VCB mua rong</title><link>http://x</link><description>FPT tang truong</description>
    <pubDate>Mon, 03 Nov 2026 08:00:00 +0700</pubDate></item>
    <item><title>Doanh nghiep ABC bị khởi tố, thua lỗ nặng</title><link>http://y</link>
    <pubDate>Mon, 03 Nov 2026 09:00:00 +0700</pubDate></item></channel></rss>""".encode("utf-8")
    items = news._parse_rss(xml, "test")
    df = news._to_df(items)
    assert len(df) == 2
    assert news.sentiment("Lãi kỷ lục, khối ngoại mua ròng") > 0
    assert news.sentiment("Bị khởi tố, thua lỗ") < 0
    tr = news.trending_tickers(df, {"HPG", "VCB", "FPT"})
    assert set(tr["Mã"]) == {"HPG", "VCB", "FPT"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
