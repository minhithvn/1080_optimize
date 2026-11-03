"""
Lớp dữ liệu đa nguồn, miễn phí, cho thị trường chứng khoán Việt Nam.

Thứ tự ưu tiên (tự động chuyển nguồn khi một nguồn lỗi):
  Giá lịch sử : Vietcap (VCI)  ->  VNDirect dchart  ->  Yahoo Finance
  Danh sách mã: Vietcap (VCI)  ->  VNDirect finfo   ->  danh sách dự phòng trong code
  Ngành ICB   : Vietcap (icbCode2) ->  bảng ngành dự phòng
  Cơ bản      : Vietcap IQ  ->  Yahoo Finance

Mọi giá trả về đều quy về đơn vị VND (không phải nghìn đồng).
Đây là các API công khai mà website/app của các CTCK dùng; chúng không có SLA,
nên code luôn có fallback và không bao giờ làm sập app khi một nguồn lỗi.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
VCI_IQ = "https://iq.vietcap.com.vn/api/iq-insight-service/v1"
VND_DCHART = "https://dchart-api.vndirect.com.vn/dchart/history"

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
    """Làm sạch dữ liệu thô từ mọi nguồn: kiểu số, ngày trùng, giá thiếu, high/low lệch, KL rỗng."""
    if df is None or len(df) == 0:
        return None
    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    df = df[df["close"].notna() & (df["close"] > 0)]
    if df.empty:
        return None
    for c in ("open", "high", "low"):
        df[c] = df[c].where(df[c] > 0, df["close"])
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if not is_index:
        df = _normalize_price_unit(df)
    return df if len(df) else None


def bars_for_period(period: str) -> int:
    """Số nến cần tải = số phiên hiển thị + phần 'khởi động' cho chỉ báo dài."""
    return PERIOD_TO_BARS.get(period, 252) + WARMUP_BARS


# ----------------------------------------------------------------------------
# Nguồn giá lịch sử
# ----------------------------------------------------------------------------
# --- Cầu dao tự ngắt: nguồn lỗi liên tiếp sẽ bị bỏ qua 10 phút (tránh chờ timeout hàng loạt khi quét)
_health: dict[str, list] = {}
_lock = threading.Lock()


def _source_ok(name: str) -> bool:
    fails, until = _health.get(name, [0, 0.0])
    return time.time() >= until


def _mark(name: str, ok: bool):
    with _lock:
        fails, until = _health.get(name, [0, 0.0])
        if ok:
            _health[name] = [0, 0.0]
        else:
            fails += 1
            _health[name] = [fails, time.time() + 600 if fails >= 5 else 0.0]


# Tên chỉ số theo từng nguồn (đã kiểm tra thực tế 11/2026)
_VCI_INDEX = {"HNXINDEX": "HNXIndex", "UPCOMINDEX": "HNXUpcomIndex", "HNX30": "HNX30", "VNINDEX": "VNINDEX",
              "VN30": "VN30"}
_VND_INDEX = {"HNXINDEX": "HNX", "UPCOMINDEX": "UPCOM"}
INDEXES = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VN100"}


def _vci_ohlc(symbol: str, count: int) -> pd.DataFrame | None:
    """Vietcap: 1 mã/request (gửi nhiều mã -> trả rỗng); 'to' không được ở tương lai; giá đơn vị VND."""
    payload = {"timeFrame": "ONE_DAY", "symbols": [_VCI_INDEX.get(symbol, symbol)], "to": int(time.time()),
               "countBack": int(count)}
    j = _post(f"{VCI_BASE}/chart/OHLCChart/gap-chart", payload, headers=VCI_HEADERS)
    if not isinstance(j, list) or not j:
        return None
    item = j[0]
    t = pd.to_datetime(pd.Series(item["t"]).astype("int64"), unit="s")
    return pd.DataFrame({"open": item["o"], "high": item["h"], "low": item["l"], "close": item["c"],
                         "volume": item["v"]}, index=(t + pd.Timedelta(hours=7)).values)


def _vnd_ohlc(symbol: str, count: int) -> pd.DataFrame | None:
    """VNDirect dchart: giá đơn vị nghìn đồng (được quy đổi ở _finalize)."""
    to_ts = int(time.time()) + 3600
    from_ts = to_ts - int(count * 1.5 * 86400) - 30 * 86400
    j = _get(VND_DCHART, params={"resolution": "D", "symbol": _VND_INDEX.get(symbol, symbol),
                                 "from": from_ts, "to": to_ts})
    if not j or j.get("s") != "ok" or not j.get("t"):
        return None
    idx = pd.to_datetime(j["t"], unit="s") + pd.Timedelta(hours=7)
    return pd.DataFrame({"open": j["o"], "high": j["h"], "low": j["l"], "close": j["c"], "volume": j["v"]}, index=idx)


def _yahoo_ohlc(symbol: str, count: int) -> pd.DataFrame | None:
    import yfinance as yf

    period = "max" if count > 1300 else ("5y" if count > 760 else "3y" if count > 510 else "2y")
    for suffix in (".VN", ".HN"):
        df = yf.Ticker(symbol + suffix).history(period=period, auto_adjust=False)
        if df is not None and not df.empty:
            df = df.rename(columns=str.lower)
            return df[["open", "high", "low", "close", "volume"]].tail(count)
    return None


class DataUnavailable(RuntimeError):
    pass


def _fetch_one(symbol: str, count: int, allow_yahoo: bool = True) -> tuple[pd.DataFrame | None, str]:
    is_index = symbol in INDEXES
    attempts = [("Vietcap", _vci_ohlc), ("VNDirect", _vnd_ohlc)]
    if allow_yahoo and not is_index:
        attempts.append(("Yahoo", _yahoo_ohlc))
    for name, fn in attempts:
        if not _source_ok(name):
            continue
        try:
            df = _finalize(fn(symbol, count), is_index=is_index)
            _mark(name, True)
            if df is not None and len(df) >= 2:
                return df.tail(count), name
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", 0) or 0
            if code in (401, 403, 429) or code >= 500:  # bị chặn / quá tải -> tính là nguồn hỏng
                _mark(name, False)
            log.info("Nguồn %s HTTP %s cho %s", name, code, symbol)
        except Exception as e:
            _mark(name, False)
            log.info("Nguồn %s lỗi cho %s: %s", name, symbol, e)
    return None, "none"


def get_history(symbol: str, count: int = 520) -> tuple[pd.DataFrame | None, str]:
    """Lấy giá ngày. Trả về (DataFrame, tên nguồn). Lỗi không bị cache -> lần sau sẽ thử lại."""
    try:
        return _get_history(symbol.upper().strip(), int(count))
    except DataUnavailable:
        return None, "none"


@cached(ttl=900)
def _get_history(symbol: str, count: int) -> tuple[pd.DataFrame, str]:
    df, src = _fetch_one(symbol, count)
    if df is None:
        raise DataUnavailable(symbol)
    return df, src


# Cache riêng cho tải hàng loạt: dict trong bộ nhớ (không qua st.cache_data -> không phải hash/pickle
# callback tiến trình, và mỗi mã được cache độc lập nên quét lại phạm vi khác vẫn tận dụng được).
_MANY_TTL = 900
_many_cache: dict[tuple[str, int], tuple[float, pd.DataFrame]] = {}


def get_history_many(symbols, count: int = 300, progress=None) -> dict[str, pd.DataFrame]:
    """Tải nhiều mã song song (dùng cho quét thị trường). `progress(done, total)` tùy chọn, gọi ở luồng chính."""
    count = int(count)
    syms = sorted({str(s).upper().strip() for s in symbols if s})
    now = time.time()
    result: dict[str, pd.DataFrame] = {}
    todo = []
    for s in syms:
        hit = _many_cache.get((s, count))
        if hit and now - hit[0] < _MANY_TTL:
            result[s] = hit[1]
        else:
            todo.append(s)
    total, done = len(syms), len(result)

    def _report():
        if progress is not None:
            try:
                progress(done, max(total, 1))
            except Exception:
                pass

    _report()
    if todo:
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_fetch_one, s, count, False): s for s in todo}
            for fut in as_completed(futs):
                sym = futs[fut]
                try:
                    df, _ = fut.result()
                except Exception:
                    df = None
                if df is not None:
                    result[sym] = df
                    with _lock:
                        _many_cache[(sym, count)] = (time.time(), df)
                done += 1
                if done % 10 == 0 or done == total:
                    _report()
    # dọn cache cũ để không phình bộ nhớ trên Streamlit Cloud
    if len(_many_cache) > 6000:
        with _lock:
            for k in [k for k, (t, _) in _many_cache.items() if time.time() - t > _MANY_TTL]:
                _many_cache.pop(k, None)
    return result


def clear_cache():
    _many_cache.clear()
    _health.clear()


# ----------------------------------------------------------------------------
# Danh sách mã & ngành
# ----------------------------------------------------------------------------
# Mã ngành ICB cấp 2 (supersector) -> tên tiếng Việt (Vietcap trả về icbCode2)
ICB2_NAMES = {
    "0500": "Dầu khí", "1300": "Hóa chất", "1700": "Tài nguyên cơ bản", "2300": "Xây dựng và Vật liệu",
    "2700": "Hàng & Dịch vụ Công nghiệp", "3300": "Ô tô và phụ tùng", "3500": "Thực phẩm và đồ uống",
    "3700": "Hàng cá nhân & Gia dụng", "4500": "Y tế", "5300": "Bán lẻ", "5500": "Truyền thông",
    "5700": "Du lịch và Giải trí", "6500": "Viễn thông", "7500": "Điện, nước & xăng dầu khí đốt",
    "8300": "Ngân hàng", "8500": "Bảo hiểm", "8600": "Bất động sản", "8700": "Dịch vụ tài chính",
    "9500": "Công nghệ Thông tin",
}


def get_listing() -> pd.DataFrame:
    """DataFrame: symbol, exchange (HOSE/HNX/UPCOM), name, industry. Toàn bộ cổ phiếu đang giao dịch."""
    try:
        df = _listing_live().copy()
        df.attrs["source"] = "live"
    except DataUnavailable:
        df = pd.DataFrame({"symbol": FALLBACK_SYMBOLS, "exchange": "N/A", "name": "", "industry": None})
        df.attrs["source"] = "fallback"
    fb = get_industries()
    df["industry"] = df["industry"].fillna(df["symbol"].map(fb)).fillna("Khác")
    return df


@cached(ttl=24 * 3600)
def _listing_live() -> pd.DataFrame:
    df = None
    try:
        j = _get(f"{VCI_BASE}/price/symbols/getAll", headers=VCI_HEADERS)
        d = pd.DataFrame(j)
        d = d[d["type"].astype(str).str.upper() == "STOCK"] if "type" in d else d
        name = d["organShortName"] if "organShortName" in d else d.get("organName", pd.Series("", index=d.index))
        icb = d["icbCode2"] if "icbCode2" in d else pd.Series(None, index=d.index)
        df = pd.DataFrame({
            "symbol": d["symbol"].astype(str).str.upper(),
            "exchange": d["board"].astype(str).str.upper().map(_BOARD_MAP),  # DELISTED -> NaN -> loại
            "name": name.fillna("").astype(str),
            "industry": icb.astype(str).str.zfill(4).map(ICB2_NAMES),
        })
    except Exception as e:
        log.info("Listing VCI lỗi: %s", e)

    if df is None or df["exchange"].notna().sum() < 300:
        for host in ("https://api-finfo.vndirect.com.vn", "https://finfo-api.vndirect.com.vn"):
            try:
                j = _get(f"{host}/v4/stocks", params={"q": "type:STOCK~status:listed", "size": 3000})
                d = pd.DataFrame(j["data"])
                name = d["shortName"] if "shortName" in d else d.get("companyName", pd.Series("", index=d.index))
                df = pd.DataFrame({
                    "symbol": d["code"].astype(str).str.upper(),
                    "exchange": d["floor"].astype(str).str.upper().map(_BOARD_MAP),
                    "name": name.fillna("").astype(str),
                    "industry": None,
                })
                break
            except Exception as e:
                log.info("Listing VNDirect lỗi: %s", e)

    if df is None:
        raise DataUnavailable("listing")
    df = df[df["exchange"].notna() & df["symbol"].str.fullmatch(r"[A-Z0-9]{3}")].drop_duplicates("symbol")
    if len(df) < 300:
        raise DataUnavailable("listing")
    return df.sort_values("symbol").reset_index(drop=True)


def get_industries() -> dict[str, str]:
    """Bảng ngành dự phòng (dùng khi nguồn listing không có mã ngành)."""
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
def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


@cached(ttl=6 * 3600)
def get_fundamentals(symbol: str) -> dict:
    """Chỉ số cơ bản TTM quý gần nhất (Vietcap IQ) -> fallback Yahoo. Khóa thiếu = None."""
    symbol = symbol.upper()
    out: dict = {"source": None}
    try:
        j = _get(f"{VCI_IQ}/company/{symbol}/statistics-financial", headers=VCI_HEADERS)
        rows = [r for r in (j or {}).get("data") or [] if r.get("year")]
        if rows:
            rows.sort(key=lambda r: (int(r.get("year") or 0), int(r.get("quarter") or 0)))
            r = rows[-1]
            prev = next((x for x in reversed(rows) if int(x["year"]) == int(r["year"]) - 1
                         and x.get("quarter") == r.get("quarter")), None)
            out.update({
                "period": f"Q{r.get('quarter')}/{r.get('year')} (TTM)",
                "pe": _num(r.get("pe")), "pb": _num(r.get("pb")), "ps": _num(r.get("ps")),
                "roe": _num(r.get("roe")), "roa": _num(r.get("roa")),
                "de": _num(r.get("debtToEquity")), "gross_margin": _num(r.get("grossMargin")),
                "net_margin": _num(r.get("afterTaxProfitMargin")),
                "market_cap_bn": (_num(r.get("marketCap")) or 0) / 1e9 or None,
                "dividend_yield": _num(r.get("dividendYield")), "ev_ebitda": _num(r.get("evToEbitda")),
                "npl": _num(r.get("npl")), "nim": _num(r.get("netInterestMargin")),
                "roe_prev_year": _num(prev.get("roe")) if prev else None,
                "source": "Vietcap",
            })
            hist = pd.DataFrame(rows[-8:])
            keep = [c for c in ["year", "quarter", "pe", "pb", "roe", "roa", "debtToEquity", "grossMargin",
                                "afterTaxProfitMargin", "dividendYield"] if c in hist]
            out["history"] = hist[keep].iloc[::-1].reset_index(drop=True)
            return out
    except Exception as e:
        log.info("Fundamental Vietcap lỗi %s: %s", symbol, e)
    try:
        import yfinance as yf

        info = yf.Ticker(symbol + ".VN").info or {}
        if len(info) > 5:
            out.update({
                "pe": _num(info.get("trailingPE")), "pb": _num(info.get("priceToBook")),
                "roe": _num(info.get("returnOnEquity")), "roa": _num(info.get("returnOnAssets")),
                "de": _num(info.get("debtToEquity")), "net_margin": _num(info.get("profitMargins")),
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
