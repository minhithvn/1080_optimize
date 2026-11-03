"""Chạy toàn bộ giao diện với dữ liệu giả lập (không mạng): python -m pytest -q tests"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_offline import make_ohlc  # noqa: E402
from vnapp import data, news  # noqa: E402

SYMS = ["FPT", "VCB", "HPG", "SSI", "VNM", "MWG", "ACB", "DGC"]


def fake_listing():
    df = pd.DataFrame({"symbol": SYMS, "exchange": ["HOSE"] * 6 + ["HNX"] * 2,
                       "name": [f"Cty {s}" for s in SYMS],
                       "industry": ["CNTT", "Ngân hàng", "Thép", "CK", "TP", "Bán lẻ", "Ngân hàng", "Hóa chất"]})
    df.attrs["source"] = "live"
    return df


def fake_hist(symbol, count=520):
    seed = sum(map(ord, symbol))
    p0 = 1200 if "INDEX" in symbol or symbol == "VN30" else 40000
    return make_ohlc(n=min(count, 1400), seed=seed, p0=p0, drift=0.0005).tail(count), "Fake"


def fake_news(*a, **k):
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    return news._to_df([{"Thời gian": now, "Tiêu đề": "FPT lãi kỷ lục, khối ngoại mua ròng HPG",
                         "Tóm tắt": "lãi suất giảm", "Nguồn": "Test", "Link": "http://x"},
                        {"Thời gian": now, "Tiêu đề": "VCB cảnh báo nợ xấu", "Tóm tắt": "", "Nguồn": "Test2",
                         "Link": "http://y"}])


@pytest.fixture(autouse=True)
def patch(monkeypatch):
    monkeypatch.setattr(data, "get_listing", fake_listing)
    monkeypatch.setattr(data, "get_history", fake_hist)
    monkeypatch.setattr(data, "get_history_many", lambda syms, count=300: {s: fake_hist(s, count)[0] for s in syms})
    monkeypatch.setattr(data, "get_group", lambda g="VN30": SYMS[:5])
    monkeypatch.setattr(data, "get_fundamentals", lambda s: {"source": "Fake", "pe": 12.0, "pb": 2.1, "roe": 0.21,
                                                              "roa": 0.08, "eps": 4500, "de": 0.6,
                                                              "net_margin": 0.12, "eps_growth_yoy": 0.15})
    monkeypatch.setattr(data, "get_macro", lambda: pd.DataFrame(
        [{"Chỉ số": "DXY", "Giá trị": 104.2, "1 ngày %": 0.1, "1 tháng %": -1.0}]))
    monkeypatch.setattr(news, "market_news", fake_news)
    monkeypatch.setattr(news, "stock_news", fake_news)
    monkeypatch.setattr(news, "macro_news", lambda: {"Lãi suất": fake_news()})


MODES = ["🏠 Tổng quan thị trường", "🔍 Phân tích cổ phiếu", "🚀 Bộ lọc cổ phiếu", "📰 Tin tức & Vĩ mô",
         "📊 So sánh", "📈 Backtest"]


@pytest.mark.parametrize("mode", MODES)
def test_mode(mode, tmp_path):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "1080.py"), default_timeout=240)
    at.run()
    assert not at.exception, at.exception
    at.sidebar.radio[0].set_value(mode).run()
    if mode == "🚀 Bộ lọc cổ phiếu":
        at.sidebar.slider[0].set_value(0).run()
        at.button[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    if mode == "🚀 Bộ lọc cổ phiếu":
        assert len(at.dataframe) >= 1
