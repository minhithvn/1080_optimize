"""
Lớp dữ liệu đa nguồn, miễn phí, cho thị trường chứng khoán Việt Nam.

Thứ tự ưu tiên (tự động chuyển nguồn khi một nguồn lỗi):
  Giá lịch sử : Vietcap (VCI)  ->  VNDirect dchart  ->  TCBS  ->  Yahoo Finance
  Danh sách mã: Vietcap (VCI)  ->  VNDirect finfo   ->  danh sách dự phòng trong code
  Ngành ICB   : Vietcap GraphQL ->  bảng ngành dự phòng
  Cơ bản      : TCBS  ->  Yahoo Finance

Mọi giá trả về đều quy về đơn vị VND (không phải nghìn đồng).
Đây là các API công khai mà website/app của các CTCK dùng; chúng không có SLA,
nên code luôn có fallback và không bao giờ làm sập app khi một nguồn lỗi.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from .config import HTTP_TIMEOUT, PERIOD_TO_BARS, USER_AGENT, WARMUP_BARS
from .sectors_fallback import FALLBACK_SECTORS
from .symbols_fallback import FALLBACK_SYMBOLS

log = logging.getLogger(__name__)

try:  # cache của Streamlit nếu có, nếu không thì cache trong bộ nhớ đơn giản
    import streamlit as st

    def cached(ttl: int):
        return st.cache_data(ttl=ttl, show_spinner=False)
except Exception:  # pragma: no cover
    from functools import lru_cache

    def cached(ttl: int):
        return lru_cache(maxsize=512)

_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*"})

VCI_HEADERS = {
    "Referer": "https://trading.vietcap.com.vn/",
    "Origin": "https://trading.vietcap.com.vn",
    "Content-Type": "application/json",
}
VCI_BASE = "https://trading.vietcap.com.vn/api"
VCI_GRAPHQL = "https://trading.vietcap.com.vn/data-mt/graphql"
VND_DCHART = "https://dchart-api.vndirect.com.vn/dchart/history"
TCBS_BASE = "https://apipubaws.tcbs.com.vn"

_BOARD_MAP = {"HSX": "HOSE", "HOSE": "HOSE", "HNX": "HNX", "UPCOM": "UPCOM", "UPC": "UPCOM"}


# ----------------------------------------------------------------------------
# Tiện ích
# ----------------------------------------------------------------------------
def _get(url, **kw):
    kw.setdefault("timeout", HTTP_TIMEOUT)
    r = _session.get(url, **kw)
    r.raise_for_status()
    return r.json()


def _post(url, payload, headers=None):
    r = _session.post(url, json=payload, headers=headers, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _normalize_price_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Một số nguồn trả giá theo nghìn đồng (60.5) – quy hết về VND (60500)."""
    if df.empty:
        return df
    med = float(df["close"].median())
    if 0 < med < 500:  # cổ phiếu VN không có thị giá trung vị < 500đ theo đơn vị VND
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * 1000.0
    return df


def _finalize(df: pd.DataFrame, is_index: bool = False) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["close"] > 0)]
    if not is_index:
        df = _normalize_price_unit(df)
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df if len(df) else None


def bars_for_period(period: str) -> int:
    """Số nến cần tải = số phiên hiển thị + phần 'khởi động' cho chỉ báo dài."""
    return PERIOD_TO_BARS.get(period, 252) + WARMUP_BARS


# ----------------------------------------------------------------------------
# Nguồn giá lịch sử
# ----------------------------------------------------------------------------
def _vci_ohlc(symbols: list[str], count: int) -> dict[str, pd.DataFrame]:
    to_ts = int((datetime.now() + timedelta(days=1)).timestamp())
    payload = {"timeFrame": "ONE_DAY", "symbols": symbols, "to": to_ts, "countBack": count}
    data = None
    for ep in ("chart/OHLCChart/gap-chart", "chart/OHLCChart/gap"):
        try:
            data = _post(f"{VCI_BASE}/{ep}", payload, headers=VCI_HEADERS)
            if data:
                break
        except Exception as e:  # thử endpoint kế tiếp
            log.debug("VCI %s lỗi: %s", ep, e)
    out = {}
    if not isinstance(data, list):
        return out
    for i, item in enumerate(data):
        try:
            sym = item.get("symbol") or symbols[i]
            t = pd.to_datetime(pd.Series(item["t"]).astype("int64"), unit="s")
            df = pd.DataFrame(
                {"open": item["o"], "high": item["h"], "low": item["l"], "close": item["c"], "volume": item["v"]},
                index=t + pd.Timedelta(hours=7),
            )
            out[sym.upper()] = df
        except Exception:
            continue
    return out


def _vnd_ohlc(symbol: str, count: int) -> pd.DataFrame | None:
    to_ts = int(time.time()) + 86400
    from_ts = to_ts - int(count * 1.5 * 86400) - 30 * 86400
    j = _get(VND_DCHART, params={"resolution": "D", "symbol": symbol, "from": from_ts, "to": to_ts})
    if not j or j.get("s") != "ok" or not j.get("t"):
        return None
    idx = pd.to_datetime(j["t"], unit="s") + pd.Timedelta(hours=7)
    return pd.DataFrame({"open": j["o"], "high": j["h"], "low": j["l"], "close": j["c"], "volume": j["v"]}, index=idx)


def _tcbs_ohlc(symbol: str, count: int, is_index: bool) -> pd.DataFrame | None:
    url = f"{TCBS_BASE}/stock-insight/v2/stock/bars-long-term"
    params = {"ticker": symbol, "type": "index" if is_index else "stock", "resolution": "D",
              "to": int(time.time()), "countBack": count}
    j = _get(url, params=params)
    rows = (j or {}).get("data") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df["tradingDate"].str[:10])
    return df[["open", "high", "low", "close", "volume"]]


def _yahoo_ohlc(symbol: str, count: int) -> pd.DataFrame | None:
    import yfinance as yf

    period = "max" if count > 1300 else ("5y" if count > 760 else "3y" if count > 510 else "2y")
    for suffix in (".VN", ".HN"):
        df = yf.Ticker(symbol + suffix).history(period=period, auto_adjust=False)
        if df is not None and not df.empty:
            df = df.rename(columns=str.lower)
            return df[["open", "high", "low", "close", "volume"]].tail(count)
    return None


@cached(ttl=900)
def get_history(symbol: str, count: int = 520) -> tuple[pd.DataFrame | None, str]:
    """Lấy giá ngày. Trả về (DataFrame, tên nguồn)."""
    symbol = symbol.upper().strip()
    is_index = symbol in ("VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VN100")
    attempts = [
        ("Vietcap", lambda: _vci_ohlc([symbol], count).get(symbol)),
        ("VNDirect", lambda: _vnd_ohlc({"HNXINDEX": "HNX", "UPCOMINDEX": "UPCOM"}.get(symbol, symbol), count)),
        ("TCBS", lambda: _tcbs_ohlc(symbol, count, is_index)),
    ]
    if not is_index:
        attempts.append(("Yahoo", lambda: _yahoo_ohlc(symbol, count)))
    for name, fn in attempts:
        try:
            df = _finalize(fn(), is_index=is_index)
            if df is not None and len(df) >= 5:
                return df.tail(count), name
        except Exception as e:
            log.info("Nguồn %s lỗi cho %s: %s", name, symbol, e)
    return None, "none"


@cached(ttl=900)
def get_history_many(symbols: tuple[str, ...], count: int = 300) -> dict[str, pd.DataFrame]:
    """Tải nhiều mã một lúc (dùng cho quét thị trường).
    Vietcap cho phép nhiều mã/1 request -> nhanh hơn ~40 lần so với từng mã."""
    symbols = [s.upper() for s in symbols]
    result: dict[str, pd.DataFrame] = {}
    chunk = 40
    chunks = [symbols[i:i + chunk] for i in range(0, len(symbols), chunk)]

    def _job(ch):
        try:
            return _vci_ohlc(ch, count)
        except Exception:
            return {}

    with ThreadPoolExecutor(max_workers=4) as ex:
        for fut in as_completed([ex.submit(_job, c) for c in chunks]):
            for sym, df in fut.result().items():
                df = _finalize(df)
                if df is not None:
                    result[sym] = df

    missing = [s for s in symbols if s not in result]
    if missing:  # fallback từng mã, song song có giới hạn để không bị chặn
        def _one(sym):
            try:
                return sym, _finalize(_vnd_ohlc(sym, count))
            except Exception:
                return sym, None

        with ThreadPoolExecutor(max_workers=8) as ex:
            for sym, df in ex.map(_one, missing):
                if df is not None:
                    result[sym] = df.tail(count)
    return result


# ----------------------------------------------------------------------------
# Danh sách mã & ngành
# ----------------------------------------------------------------------------
@cached(ttl=24 * 3600)
def get_listing() -> pd.DataFrame:
    """DataFrame: symbol, exchange (HOSE/HNX/UPCOM), name, industry. Toàn bộ cổ phiếu niêm yết + UPCoM."""
    df = None
    try:
        j = _get(f"{VCI_BASE}/price/symbols/getAll", headers=VCI_HEADERS)
        df = pd.DataFrame(j)
        df = df[df["type"].astype(str).str.upper() == "STOCK"]
        df = pd.DataFrame({
            "symbol": df["symbol"].str.upper(),
            "exchange": df["board"].astype(str).str.upper().map(_BOARD_MAP).fillna("OTHER"),
            "name": df.get("organShortName", df.get("organName", "")),
        })
    except Exception as e:
        log.info("Listing VCI lỗi: %s", e)

    if df is None or df.empty:
        for host in ("https://api-finfo.vndirect.com.vn", "https://finfo-api.vndirect.com.vn"):
            try:
                j = _get(f"{host}/v4/stocks", params={"q": "type:STOCK~status:listed", "size": 3000})
                d = pd.DataFrame(j["data"])
                df = pd.DataFrame({
                    "symbol": d["code"].str.upper(),
                    "exchange": d["floor"].astype(str).str.upper().map(_BOARD_MAP).fillna("OTHER"),
                    "name": d.get("shortName", d.get("companyName", "")),
                })
                break
            except Exception as e:
                log.info("Listing VNDirect lỗi: %s", e)

    if df is None or df.empty:
        df = pd.DataFrame({"symbol": FALLBACK_SYMBOLS, "exchange": "N/A", "name": ""})
        df.attrs["source"] = "fallback"
    else:
        df.attrs["source"] = "live"

    df = df[df["symbol"].str.fullmatch(r"[A-Z0-9]{3}")].drop_duplicates("symbol")
    ind = get_industries()
    df["industry"] = df["symbol"].map(ind).fillna("Khác")
    return df.sort_values("symbol").reset_index(drop=True)


@cached(ttl=24 * 3600)
def get_industries() -> dict[str, str]:
    """Map mã -> ngành ICB cấp 2 (tiếng Việt)."""
    query = "{ CompaniesListingInfo { ticker organName icbName2 icbName3 } }"
    try:
        j = _post(VCI_GRAPHQL, {"query": query, "variables": {}}, headers=VCI_HEADERS)
        rows = j["data"]["CompaniesListingInfo"]
        m = {r["ticker"].upper(): (r.get("icbName2") or r.get("icbName3") or "Khác") for r in rows if r.get("ticker")}
        if len(m) > 500:
            return m
    except Exception as e:
        log.info("ICB VCI lỗi: %s", e)
    m = {}
    for sector, syms in FALLBACK_SECTORS.items():
        for s in syms:
            m.setdefault(s, sector)
    return m


@cached(ttl=24 * 3600)
def get_group(group: str = "VN30") -> list[str]:
    try:
        j = _get(f"{VCI_BASE}/price/symbols/getByGroup", params={"group": group}, headers=VCI_HEADERS)
        syms = [x["symbol"].upper() for x in j if x.get("symbol")]
        if syms:
            return syms
    except Exception:
        pass
    return ["ACB", "BCM", "BID", "CTG", "DGC", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB", "MBB", "MSN", "MWG",
            "PLX", "SAB", "SHB", "SSB", "SSI", "STB", "TCB", "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM",
            "VPB", "VRE"]


# ----------------------------------------------------------------------------
# Dữ liệu cơ bản
# ----------------------------------------------------------------------------
@cached(ttl=6 * 3600)
def get_fundamentals(symbol: str) -> dict:
    """Chỉ số cơ bản quý gần nhất. Khóa trả về đã chuẩn hóa; thiếu thì None."""
    symbol = symbol.upper()
    out: dict = {"source": None}
    try:
        ov = _get(f"{TCBS_BASE}/tcanalysis/v1/ticker/{symbol}/overview")
        out.update({
            "industry": ov.get("industry"), "exchange": ov.get("exchange"),
            "shares_out_mil": ov.get("outstandingShare"), "foreign_pct": ov.get("foreignPercent"),
        })
        rows = _get(f"{TCBS_BASE}/tcanalysis/v1/finance/{symbol}/financialratio",
                    params={"yearly": 0, "isAll": "false"})
        if rows:
            r = rows[0]
            out.update({
                "period": f"Q{r.get('quarter')}/{r.get('year')}",
                "pe": r.get("priceToEarning"), "pb": r.get("priceToBook"),
                "roe": r.get("roe"), "roa": r.get("roa"),
                "eps": r.get("earningPerShare"), "bvps": r.get("bookValuePerShare"),
                "de": r.get("debtOnEquity"), "gross_margin": r.get("grossProfitMargin"),
                "net_margin": r.get("postTaxMargin"),
                "eps_growth_yoy": r.get("epsChange"),
            })
            out["history"] = pd.DataFrame(rows[:8])
        out["source"] = "TCBS"
        return out
    except Exception as e:
        log.info("Fundamental TCBS lỗi %s: %s", symbol, e)
    try:
        import yfinance as yf

        info = yf.Ticker(symbol + ".VN").info or {}
        if len(info) > 5:
            out.update({
                "pe": info.get("trailingPE"), "pb": info.get("priceToBook"),
                "roe": info.get("returnOnEquity"), "roa": info.get("returnOnAssets"),
                "eps": info.get("trailingEps"), "de": info.get("debtToEquity"),
                "net_margin": info.get("profitMargins"), "industry": info.get("industry"),
                "source": "Yahoo",
            })
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------
# Vĩ mô quốc tế (Yahoo Finance – miễn phí)
# ----------------------------------------------------------------------------
MACRO_TICKERS = {
    "USD/VND": "VND=X",
    "DXY (USD Index)": "DX-Y.NYB",
    "Vàng (USD/oz)": "GC=F",
    "Dầu Brent": "BZ=F",
    "S&P 500": "^GSPC",
    "Lợi suất TP Mỹ 10Y": "^TNX",
    "Nikkei 225": "^N225",
    "Shanghai Comp.": "000001.SS",
}


@cached(ttl=1800)
def get_macro() -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for name, tk in MACRO_TICKERS.items():
        try:
            h = yf.Ticker(tk).history(period="3mo")["Close"].dropna()
            if len(h) < 2:
                continue
            last = h.iloc[-1]
            rows.append({
                "Chỉ số": name, "Giá trị": last,
                "1 ngày %": (last / h.iloc[-2] - 1) * 100,
                "1 tháng %": (last / h.iloc[-min(22, len(h))] - 1) * 100,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def snapshot_from_histories(hist: dict[str, pd.DataFrame], listing: pd.DataFrame) -> pd.DataFrame:
    """Bảng tổng hợp thị trường từ dữ liệu nhiều mã: giá, %thay đổi, GTGD, ngành."""
    meta = listing.set_index("symbol")
    rows = []
    for sym, df in hist.items():
        if df is None or len(df) < 25:
            continue
        c, v = df["close"], df["volume"]
        value20 = (c * v).tail(20).mean()
        rows.append({
            "Mã": sym,
            "Sàn": meta["exchange"].get(sym, ""),
            "Ngành": meta["industry"].get(sym, "Khác"),
            "Giá": c.iloc[-1],
            "1D %": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
            "1W %": (c.iloc[-1] / c.iloc[-6] - 1) * 100,
            "1M %": (c.iloc[-1] / c.iloc[-22] - 1) * 100,
            "3M %": (c.iloc[-1] / c.iloc[-min(66, len(c))] - 1) * 100,
            "GTGD hôm nay (tỷ)": c.iloc[-1] * v.iloc[-1] / 1e9,
            "GTGD TB20 (tỷ)": value20 / 1e9,
            "Đột biến KL": v.iloc[-1] / max(v.tail(21).iloc[:-1].mean(), 1),
            "Trên MA50": bool(c.iloc[-1] > c.tail(50).mean()),
        })
    return pd.DataFrame(rows)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default
