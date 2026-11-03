"""
Kiểm thử TÍCH HỢP: chỉ giả lập tầng HTTP (requests) – toàn bộ code thật của app chạy qua,
gồm cả st.cache_data, tải song song, cầu dao, parser, chỉ báo, ML, UI.
(Bài học từ lỗi UnhashableParamError: test UI cũ giả lập ở tầng hàm nên bỏ sót lỗi cache.)
"""
import itertools
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_offline import CAFEF_REAL, GNEWS_REAL, make_ohlc  # noqa: E402
from vnapp import data  # noqa: E402

REAL = ["FPT", "VCB", "HPG", "SSI", "VNM", "MWG", "ACB", "DGC", "SHS", "PVS"]
FILLER = ["Q" + a + b for a, b in itertools.product("ABCDEFGHIJKLMNOPQRSTUVWXYZ", repeat=2)][:340]
INDEX_VCI = {"VNINDEX", "VN30", "HNXIndex", "HNXUpcomIndex", "HNX30"}


class Resp:
    def __init__(self, payload=None, status=200, content=None):
        self._p, self.status_code = payload, status
        self.content = content if content is not None else json.dumps(payload).encode()
        self.encoding = None

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}", response=self)

    def json(self):
        return self._p


def series(sym, n):
    seed = sum(map(ord, sym))
    p0 = 1300 if sym in INDEX_VCI or sym in ("HNX", "UPCOM") else 20000 + seed * 10
    return make_ohlc(n=n, seed=seed, p0=p0, drift=0.0003)


class FakeNet:
    def __init__(self, vci_down=False):
        self.vci_down = vci_down
        self.calls = {"vci": 0, "vnd": 0, "rss": 0}

    def post(self, url, json=None, headers=None, timeout=None):
        if "gap-chart" in url:
            self.calls["vci"] += 1
            if self.vci_down:
                return Resp({"err": 1}, 403)
            sym, n = json["symbols"][0], json["countBack"]
            if len(json["symbols"]) != 1:
                return Resp([])  # hành vi thật: nhiều mã -> rỗng
            d = series(sym, min(n, 1600))
            t = [str(int(x.timestamp())) for x in d.index]
            return Resp([{"symbol": sym, "o": list(d.open.round(0)), "h": list(d.high.round(0)),
                          "l": list(d.low.round(0)), "c": list(d.close.round(0)), "v": list(d.volume), "t": t}])
        return Resp({}, 404)

    def get(self, url, params=None, headers=None, timeout=None, **kw):
        u = urlparse(url)
        if "symbols/getAll" in url:
            rows = [{"symbol": s, "type": "STOCK", "board": "HSX" if i < 200 else ("HNX" if i < 300 else "UPCOM"),
                     "organShortName": f"Cty {s}", "icbCode2": ["8300", "9500", "1700", "8700"][i % 4]}
                    for i, s in enumerate(REAL + FILLER)]
            rows.append({"symbol": "OLD", "type": "STOCK", "board": "DELISTED", "organShortName": "x"})
            return Resp(rows)
        if "getByGroup" in url:
            return Resp([{"symbol": s} for s in REAL[:8]])
        if "dchart" in url:
            self.calls["vnd"] += 1
            sym = params["symbol"]
            d = series(sym, 900)
            k = 1 if sym in ("VNINDEX", "VN30", "HNX", "UPCOM") else 1000
            return Resp({"t": [int(x.timestamp()) for x in d.index], "o": list(d.open / k), "h": list(d.high / k),
                         "l": list(d.low / k), "c": list(d.close / k), "v": list(d.volume), "s": "ok"})
        if "statistics-financial" in url:
            return Resp({"data": [{"year": "2025", "quarter": 3, "pe": 9.0, "roe": 0.15},
                                  {"year": "2026", "quarter": 3, "pe": 11.2, "pb": 2.0, "roe": 0.18, "roa": 0.07,
                                   "debtToEquity": 0.8, "afterTaxProfitMargin": 0.1, "marketCap": 5e13,
                                   "dividendYield": 0.02}]})
        if "news.google.com" in u.netloc:
            self.calls["rss"] += 1
            return Resp(content=GNEWS_REAL.encode("utf-8"))
        if url.endswith(".rss"):
            self.calls["rss"] += 1
            return Resp(content=CAFEF_REAL.encode("utf-8"))
        return Resp({}, 404)


@pytest.fixture
def net(monkeypatch):
    import streamlit as st
    import yfinance

    st.cache_data.clear()
    data.clear_cache()
    fake = FakeNet()
    monkeypatch.setattr(data._session, "get", fake.get)
    monkeypatch.setattr(data._session, "post", fake.post)

    class NoYahoo:
        def __init__(self, *a, **k):
            raise RuntimeError("yahoo offline in test")
    monkeypatch.setattr(yfinance, "Ticker", NoYahoo)
    yield fake
    st.cache_data.clear()
    data.clear_cache()


MODES = ["🏠 Tổng quan thị trường", "🔍 Phân tích cổ phiếu", "🚀 Bộ lọc cổ phiếu", "📰 Tin tức & Vĩ mô",
         "📊 So sánh", "📈 Backtest"]


def run_mode(mode):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "1080.py"), default_timeout=300)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    at.sidebar.radio[0].set_value(mode).run()
    if mode == "🚀 Bộ lọc cổ phiếu":
        at.sidebar.slider[0].set_value(0).run()
        scan = [b for b in at.button if "Quét" in b.label][0]
        scan.click().run()
        assert not at.exception, [e.message + "\n".join(e.stack_trace) for e in at.exception]
        assert len(at.dataframe) >= 1, "bộ lọc phải ra kết quả"
        # bấm quét lần 2 -> dùng cache, không lỗi
        [b for b in at.button if "Quét" in b.label][0].click().run()
    assert not at.exception, [e.message + "\n".join(e.stack_trace) for e in at.exception]
    return at


@pytest.mark.parametrize("mode", MODES)
def test_all_modes_real_code_path(net, mode):
    at = run_mode(mode)
    if mode == "🔍 Phân tích cổ phiếu":
        assert at.header[0].value.startswith("FPT")
        assert any("Vietcap" in c.value for c in at.caption)  # nguồn giá thật được dùng
        assert any("ML" in w.value for w in list(at.warning) + list(at.success))  # ML chạy được với 5 năm dữ liệu
    if mode == "🏠 Tổng quan thị trường":
        labels = [m.label for m in at.metric]
        assert "VNINDEX" in labels and "Mã tăng / giảm" in labels


def test_vci_blocked_falls_back_to_vnd(net):
    net.vci_down = True
    at = run_mode("🔍 Phân tích cổ phiếu")
    assert any("VNDirect" in c.value for c in at.caption)
    assert net.calls["vci"] <= 6  # cầu dao ngắt Vietcap sau vài lỗi 403


def test_refresh_button(net):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "1080.py"), default_timeout=300)
    at.run()
    [b for b in at.sidebar.button if "Làm mới" in b.label][0].click().run()
    assert not at.exception


def test_history_many_progress_and_cache(net):
    seen = []
    syms = REAL + FILLER[:40]
    h1 = data.get_history_many(syms, 120, progress=lambda d, t: seen.append((d, t)))
    assert len(h1) == len(syms) and seen[-1] == (len(syms), len(syms))
    before = net.calls["vci"]
    h2 = data.get_history_many(syms, 120, progress=lambda d, t: None)
    assert net.calls["vci"] == before and len(h2) == len(h1)  # lần 2 lấy từ cache
    assert np.isclose(h1["FPT"]["close"].iloc[-1], h2["FPT"]["close"].iloc[-1])
