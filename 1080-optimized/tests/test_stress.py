"""Kiểm thử biên/stress: dữ liệu bất thường giống thị trường thật (mã mới niêm yết, tạm ngừng GD,
không thanh khoản, giá trần liên tiếp, VNINDEX ngắn hơn/thiếu, giá nghìn đồng...)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_offline import make_ohlc  # noqa: E402
from vnapp import charts, data, news  # noqa: E402
from vnapp.backtest import run_backtest  # noqa: E402
from vnapp.forecast import ml_trend, price_cone  # noqa: E402
from vnapp.indicators import add_indicators, candlestick_today, support_resistance  # noqa: E402
from vnapp.signals import generate_signal, market_regime, score_frame, trade_plan  # noqa: E402


def flat(n=400, p=10000.0, vol=0.0):
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": vol}, index=idx)


def scenario(name):
    if name == "long_short_bench":
        return make_ohlc(1520), make_ohlc(1520, seed=2, p0=1200).tail(800)
    if name == "no_bench":
        return make_ohlc(900), None
    if name == "bench_misaligned":
        return make_ohlc(600), make_ohlc(600, seed=2, p0=1200).iloc[::2]
    if name.startswith("short_"):
        n = int(name.split("_")[1])
        return make_ohlc(1520).tail(n), make_ohlc(1520, seed=2, p0=1200)
    if name == "suspended_flat":
        return flat(400), make_ohlc(400, seed=2, p0=1200)
    if name == "flat_then_moves":
        d = make_ohlc(600)
        d.iloc[200:320, :4] = d["close"].iloc[199]
        d.iloc[200:320, 4] = 0
        return d, make_ohlc(600, seed=2, p0=1200)
    if name == "zero_volume":
        d = make_ohlc(500)
        d["volume"] = 0.0
        return d, make_ohlc(500, seed=2, p0=1200)
    if name == "limit_up_streak":
        d = make_ohlc(500)
        for i in range(470, 500):
            p = d["close"].iloc[i - 1] * 1.07
            d.iloc[i, :4] = p
        return d, make_ohlc(500, seed=2, p0=1200)
    if name == "crash":
        d = make_ohlc(500, drift=-0.01)
        return d, make_ohlc(500, seed=2, p0=1200)
    if name == "penny":
        return make_ohlc(500, p0=900, vol=0.05), make_ohlc(500, seed=2, p0=1200)
    if name == "with_nan_rows":
        d = make_ohlc(500)
        d.iloc[[50, 51, 300], :] = np.nan
        return data._finalize(d.dropna(how="all")), make_ohlc(500, seed=2, p0=1200)
    raise ValueError(name)


SCENARIOS = ["long_short_bench", "no_bench", "bench_misaligned", "short_30", "short_61", "short_130",
             "short_249", "short_260", "suspended_flat", "flat_then_moves", "zero_volume", "limit_up_streak",
             "crash", "penny", "with_nan_rows"]


@pytest.mark.parametrize("name", SCENARIOS)
def test_full_pipeline(name):
    df, bench = scenario(name)
    di = add_indicators(df, bench)
    regime = market_regime(bench)
    assert np.isfinite(di[["open", "high", "low", "close"]].values).all()
    sig = generate_signal(di, regime, 0.5)
    assert sig["signal"] in ("N/A", "MUA MẠNH", "MUA", "THEO DÕI / NẮM GIỮ", "GIẢM TỶ TRỌNG", "BÁN")
    candlestick_today(di)
    score_frame(di, regime)
    if len(di) >= 60:
        plan = trade_plan(di, sig)
        for k in ("stop", "t1", "t2", "price"):
            assert np.isfinite(plan[k]), (k, plan)
        support_resistance(di)
        cone = price_cone(di, 20, "UPCOM", targets=(plan["t1"], plan["stop"]))
        if cone:
            assert np.isfinite(cone["q"]["p50"]).all()
        charts.main_chart(di.tail(252), "TEST", cone, plan)
        charts.indicator_panel(di.tail(252))
        charts.components_bar(sig["components"])
    charts.score_gauge(sig["score"] if np.isfinite(sig.get("score", np.nan)) else np.nan, "x")
    res = ml_trend(di, 10, regime)
    if res:
        assert 0 <= res["prob_up"] <= 1
    run_backtest(di, regime)


def test_snapshot_and_market_charts_edge():
    listing = pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"], "exchange": "HOSE", "name": "", "industry": ["X", "X", "Y"]})
    hist = {"AAA": make_ohlc(80), "BBB": flat(80, vol=0), "CCC": make_ohlc(10)}
    snap = data.snapshot_from_histories(hist, listing)
    assert set(snap["Mã"]) == {"AAA", "BBB"}
    charts.sector_treemap(snap)
    charts.sector_bar(snap)
    empty = data.snapshot_from_histories({}, listing)
    assert empty.empty


def test_news_empty_and_bad_xml():
    assert news._parse_rss("<html>not xml", "x") == []
    df = news._to_df([])
    assert news.news_score(df) == 0.0
    assert news.trending_tickers(df, {"AAA"}).empty
    assert news.hot_keywords(df).empty
    d = news._to_df([{"Thời gian": None, "Tiêu đề": "abc", "Tóm tắt": "", "Nguồn": "s", "Link": ""}])
    assert news.news_score(d) == 0.0


def test_compare_charts_mixed_lengths():
    closes = {"A": make_ohlc(300)["close"], "B": make_ohlc(100)["close"], "C": flat(50)["close"]}
    charts.compare_chart(closes)
    charts.corr_heatmap(closes)
