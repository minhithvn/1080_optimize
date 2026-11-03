"""Kiểm thử offline (không cần mạng): python -m pytest -q tests"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

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


# Payload THẬT ghi lại từ API ngày 03/11/2026
VCI_REAL = [{"symbol": "VCB", "o": [57100, 56100, 54200], "h": [57200, 56200, 54300], "l": [56300, 53300, 53100],
             "c": [56300, 53600, 53200], "v": [3621700, 14009100, 1337700],
             "t": ["1793318400", "1793577600", "1793664000"]}]
VND_REAL = {"t": [1793318400, 1793577600, 1793664000], "c": [56.3, 53.6, 53.2], "o": [57.1, 56.1, 54.2],
            "h": [57.2, 56.2, 54.3], "l": [56.3, 53.3, 53.1], "v": [3621700, 14045203, 1392600], "s": "ok"}
VND_INDEX_REAL = {"t": [1793318400, 1793577600, 1793664000], "c": [1741.62, 1710.72, 1701.91],
                  "o": [1747.09, 1731.9, 1710.04], "h": [1748.68, 1744.35, 1712.19], "l": [1727.0, 1693.75, 1696.23],
                  "v": [573601499, 724512489, 93839280], "s": "ok"}


def _expect_vcb(df):
    assert list(df["close"]) == [56300, 53600, 53200]
    assert [d.strftime("%Y-%m-%d") for d in df.index] == ["2026-10-30", "2026-11-02", "2026-11-03"]


def test_vci_parser_real(monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(json)
        return FakeResp(VCI_REAL)
    monkeypatch.setattr(data._session, "post", fake_post)
    _expect_vcb(data._finalize(data._vci_ohlc("VCB", 3)))
    assert sent["symbols"] == ["VCB"] and sent["to"] <= __import__("time").time() + 1  # không gửi 'to' tương lai
    data._vci_ohlc("HNXINDEX", 3)
    assert sent["symbols"] == ["HNXIndex"]


def test_vnd_parser_real(monkeypatch):
    monkeypatch.setattr(data._session, "get", lambda *a, **k: FakeResp(VND_REAL))
    _expect_vcb(data._finalize(data._vnd_ohlc("VCB", 3)))
    monkeypatch.setattr(data._session, "get", lambda *a, **k: FakeResp(VND_INDEX_REAL))
    idx = data._finalize(data._vnd_ohlc("VNINDEX", 3), is_index=True)
    assert idx["close"].iloc[-1] == 1701.91  # chỉ số KHÔNG bị nhân 1000


def test_fallback_when_vci_empty(monkeypatch):
    data._health.clear()
    monkeypatch.setattr(data._session, "post", lambda *a, **k: FakeResp([]))
    monkeypatch.setattr(data._session, "get", lambda *a, **k: FakeResp(VND_REAL))
    df, src = data._fetch_one("VCB", 3, allow_yahoo=False)
    assert src == "VNDirect"
    _expect_vcb(df)


def test_circuit_breaker(monkeypatch):
    data._health.clear()
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise requests.ConnectionError("blocked")
    monkeypatch.setattr(data._session, "post", boom)
    monkeypatch.setattr(data._session, "get", lambda *a, **k: FakeResp(VND_REAL))
    for _ in range(8):
        data._fetch_one("VCB", 3, allow_yahoo=False)
    assert calls["n"] == 5  # sau 5 lỗi liên tiếp, Vietcap bị tạm bỏ qua
    data._health.clear()


def test_listing_excludes_delisted(monkeypatch):
    import itertools
    names = ["A" + a + b for a, b in itertools.product("ABCDEFGHIJKLMNOPQRSTUVWXYZ", repeat=2)][:350]
    rows = [{"symbol": n, "type": "STOCK", "board": "HSX", "organShortName": "x", "icbCode2": "8300"} for n in names]
    rows += [{"symbol": "ZZZ", "type": "STOCK", "board": "DELISTED", "organShortName": "dead", "icbCode2": "8600"},
             {"symbol": "CW1", "type": "CW", "board": "HSX", "organShortName": "cw", "icbCode2": None},
             {"symbol": "B01", "type": "STOCK", "board": "UPCOM", "organShortName": "b", "icbCode2": 4500}]
    monkeypatch.setattr(data._session, "get", lambda *a, **k: FakeResp(rows))
    data._listing_live.clear()
    df = data.get_listing()
    data._listing_live.clear()
    assert "ZZZ" not in set(df["symbol"]) and "CW1" not in set(df["symbol"])
    assert df.set_index("symbol").loc["AAA", "industry"] == "Ngân hàng"
    assert df.set_index("symbol").loc["B01", "industry"] == "Y tế"
    assert df.attrs["source"] == "live"


def test_fundamentals_iq(monkeypatch):
    rows = [{"year": "2025", "quarter": 3, "pe": 8.0, "pb": 1.0, "roe": 0.15, "roa": 0.08, "debtToEquity": 0.9},
            {"year": "2026", "quarter": 3, "pe": 6.27, "pb": 1.16, "roe": 0.197, "roa": 0.10,
             "debtToEquity": 0.895, "afterTaxProfitMargin": 0.1277, "marketCap": 1.72e14, "dividendYield": 0},
            {"year": "2026", "quarter": 2, "pe": 7.0}]
    monkeypatch.setattr(data._session, "get", lambda *a, **k: FakeResp({"data": rows}))
    data.get_fundamentals.clear()
    f = data.get_fundamentals("HPG")
    data.get_fundamentals.clear()
    assert f["source"] == "Vietcap" and f["pe"] == 6.27 and f["period"].startswith("Q3/2026")
    assert f["roe_prev_year"] == 0.15 and round(f["market_cap_bn"]) == 172000


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


CAFEF_REAL = """
<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>Thị trường chứng khoán | cafef</title>
<item><title><![CDATA[VN-Index chưa thủng đáy, vì sao nhiều cổ phiếu đã giảm sâu?]]></title>
<link>https://cafef.vn/a.chn</link>
<description><![CDATA[<a href="https://cafef.vn/a.chn"><img src="x.png"></a> Thanh khoản suy yếu và khối ngoại bán ròng mạnh.]]></description>
<pubDate>Tue, 03 Nov 26 09:00:00 +0700</pubDate></item></channel></rss>"""

GNEWS_REAL = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><rss version="2.0"><channel><title>t</title>
<item><title>Cổ phiếu HPG bất ngờ dẫn đầu thanh khoản - Báo Nghệ An</title><link>https://news.google.com/x</link>
<pubDate>Thu, 29 Oct 2026 05:04:01 GMT</pubDate><description>&lt;a href="x"&gt;Cổ phiếu HPG bất ngờ dẫn đầu thanh khoản&lt;/a&gt;&amp;nbsp;&lt;font color="#6f6f6f"&gt;Báo Nghệ An&lt;/font&gt;</description>
<source url="https://baonghean.vn">Báo Nghệ An</source></item></channel></rss>"""


def test_real_rss_formats():
    c = news._to_df(news._parse_rss(CAFEF_REAL.encode("utf-8"), "CafeF"))
    assert len(c) == 1 and c["Thời gian"].iloc[0].year == 2026 and c["Thời gian"].iloc[0].hour == 9
    assert "<img" not in c["Tóm tắt"].iloc[0] and c["Cảm xúc"].iloc[0] < 0  # "bán ròng"
    g = news._to_df(news._parse_rss(GNEWS_REAL.encode("utf-8"), "Google News"))
    assert g["Tiêu đề"].iloc[0] == "Cổ phiếu HPG bất ngờ dẫn đầu thanh khoản"
    assert g["Nguồn"].iloc[0] == "Báo Nghệ An"
    assert g["Thời gian"].iloc[0].hour == 12  # 05:04 GMT = 12:04 giờ VN
