"""
Tin tức thị trường – chính trị – kinh tế (RSS miễn phí, chính thống):
CafeF, VnExpress, Vietstock, VietnamNet, Tuổi Trẻ, Thanh Niên + Google News (tìm theo mã / chủ đề).

Chấm cảm xúc tin bằng từ điển tiếng Việt chuyên ngành (nhanh, minh bạch, không cần API key).
Đây là chỉ báo hỗ trợ, không thay thế việc đọc tin gốc.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import pandas as pd

from .data import _session, cached

RSS_FEEDS = {
    "CafeF – Chứng khoán": "https://cafef.vn/thi-truong-chung-khoan.rss",
    "CafeF – Vĩ mô": "https://cafef.vn/vi-mo-dau-tu.rss",
    "CafeF – Doanh nghiệp": "https://cafef.vn/doanh-nghiep.rss",
    "CafeF – Tài chính quốc tế": "https://cafef.vn/tai-chinh-quoc-te.rss",
    "VnExpress – Kinh doanh": "https://vnexpress.net/rss/kinh-doanh.rss",
    "VnExpress – Thế giới": "https://vnexpress.net/rss/the-gioi.rss",
    "VnExpress – Thời sự": "https://vnexpress.net/rss/thoi-su.rss",
    "Vietstock – Cổ phiếu": "https://vietstock.vn/830/chung-khoan/co-phieu.rss",
    "Vietstock – Vĩ mô": "https://vietstock.vn/761/kinh-te/vi-mo.rss",
    "VietnamNet – Kinh doanh": "https://vietnamnet.vn/rss/kinh-doanh.rss",
    "Tuổi Trẻ – Kinh doanh": "https://tuoitre.vn/rss/kinh-doanh.rss",
    "Thanh Niên – Kinh tế": "https://thanhnien.vn/rss/kinh-te.rss",
}

MACRO_TOPICS = {
    "Chính sách tiền tệ / lãi suất": "Ngân hàng Nhà nước lãi suất điều hành",
    "Tỷ giá USD/VND": "tỷ giá USD VND",
    "Fed / kinh tế Mỹ": "Fed lãi suất",
    "Thuế quan & thương mại Mỹ–Việt": "thuế quan Mỹ Việt Nam",
    "Nâng hạng thị trường (FTSE/MSCI)": "nâng hạng thị trường chứng khoán FTSE MSCI",
    "Khối ngoại": "khối ngoại mua ròng bán ròng",
    "GDP / CPI / FDI": "GDP CPI FDI Việt Nam",
    "Đầu tư công / Chính phủ": "đầu tư công giải ngân Chính phủ",
    "Giá dầu & hàng hóa": "giá dầu thế giới",
    "Địa chính trị": "căng thẳng địa chính trị thị trường",
}

POS_WORDS = [
    "tăng trưởng", "lãi kỷ lục", "kỷ lục", "lợi nhuận tăng", "vượt kế hoạch", "vượt kỳ vọng", "bứt phá",
    "mua ròng", "nâng hạng", "khởi sắc", "hồi phục", "phục hồi", "tích cực", "trúng thầu", "ký kết",
    "chia cổ tức", "cổ tức tiền mặt", "mở rộng", "tăng vốn thành công", "nâng mục tiêu", "khuyến nghị mua",
    "tăng trần", "dòng tiền mạnh", "hạ lãi suất", "giảm lãi suất", "nới lỏng", "hỗ trợ", "thông qua",
    "lãi lớn", "doanh thu tăng", "tăng mạnh", "đột phá", "thỏa thuận thương mại", "giải ngân", "hút vốn",
]
NEG_WORDS = [
    "thua lỗ", "lỗ ròng", "lỗ lũy kế", "lợi nhuận giảm", "sụt giảm", "lao dốc", "bán tháo", "bán ròng",
    "giảm sàn", "khởi tố", "bắt tạm giam", "bị bắt", "điều tra", "vi phạm", "xử phạt", "bị phạt",
    "cảnh báo", "kiểm soát", "đình chỉ", "hủy niêm yết", "hạn chế giao dịch", "nợ xấu", "chậm trả",
    "vỡ nợ", "phá sản", "giải chấp", "call margin", "thao túng", "tiêu cực", "căng thẳng", "chiến tranh",
    "thuế đối ứng", "áp thuế", "tăng lãi suất", "lạm phát tăng", "suy thoái", "giảm mạnh", "rủi ro",
    "không đạt kế hoạch", "ý kiến ngoại trừ", "từ chối đưa ra ý kiến", "chậm công bố", "trái phiếu quá hạn",
]

_TAG = re.compile(r"<[^>]+>")


def sentiment(text: str) -> float:
    t = text.lower()
    p = sum(t.count(w) for w in POS_WORDS)
    n = sum(t.count(w) for w in NEG_WORDS)
    if p + n == 0:
        return 0.0
    return (p - n) / (p + n)


def _parse_rss(xml_text: str, source: str) -> list[dict]:
    items = []
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError:
        return items
    for it in root.iter("item"):
        title = html.unescape((it.findtext("title") or "").strip())
        link = (it.findtext("link") or "").strip()
        desc = html.unescape(_TAG.sub(" ", it.findtext("description") or "")).strip()
        pub = it.findtext("pubDate")
        try:
            dt = parsedate_to_datetime(pub) if pub else None
            if dt is not None and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except Exception:
            dt = None
        src = it.findtext("source") or source
        if title:
            items.append({"Thời gian": dt, "Tiêu đề": title, "Tóm tắt": desc[:300], "Nguồn": src, "Link": link})
    return items


def _fetch(url: str, source: str) -> list[dict]:
    try:
        r = _session.get(url, timeout=10, headers={"Accept": "application/rss+xml, application/xml, text/xml"})
        r.raise_for_status()
        return _parse_rss(r.content.decode(r.encoding or "utf-8", errors="ignore"), source)
    except Exception:
        return []


def _to_df(items: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=["Thời gian", "Tiêu đề", "Tóm tắt", "Nguồn", "Link", "Cảm xúc"])
    df = df.drop_duplicates("Tiêu đề")
    df["Thời gian"] = pd.to_datetime(df["Thời gian"], utc=True, errors="coerce").dt.tz_convert("Asia/Ho_Chi_Minh")
    df["Cảm xúc"] = (df["Tiêu đề"] + ". " + df["Tóm tắt"]).map(sentiment)
    return df.sort_values("Thời gian", ascending=False, na_position="last").reset_index(drop=True)


def google_news_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}+when:14d&hl=vi&gl=VN&ceid=VN:vi"


@cached(ttl=900)
def market_news() -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = ex.map(lambda kv: _fetch(kv[1], kv[0]), RSS_FEEDS.items())
    return _to_df([x for lst in res for x in lst])


@cached(ttl=900)
def stock_news(symbol: str, company: str = "") -> pd.DataFrame:
    q = f'"{symbol}" cổ phiếu' + (f' OR "{company}"' if company else "")
    df = _to_df(_fetch(google_news_url(q), "Google News"))
    if not df.empty:  # chỉ giữ tin thực sự nhắc tới mã/tên công ty
        pat = re.compile(rf"\b{re.escape(symbol)}\b" + (rf"|{re.escape(company)}" if company else ""), re.I)
        df = df[df["Tiêu đề"].str.contains(pat) | df["Tóm tắt"].str.contains(pat)]
    return df.reset_index(drop=True)


@cached(ttl=1800)
def macro_news() -> dict[str, pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(ex.map(lambda kv: (kv[0], _to_df(_fetch(google_news_url(kv[1]), "Google News"))),
                          MACRO_TOPICS.items()))
    return dict(res)


def news_score(df: pd.DataFrame, days: int = 7) -> float:
    """Điểm tin tức -1..1, tin mới có trọng số cao hơn."""
    if df is None or df.empty:
        return 0.0
    now = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
    d = df.dropna(subset=["Thời gian"])
    d = d[d["Thời gian"] > now - pd.Timedelta(days=days)]
    if d.empty:
        return 0.0
    age = (now - d["Thời gian"]).dt.total_seconds() / 86400
    w = 0.5 ** (age / 3)  # bán rã 3 ngày
    return float((d["Cảm xúc"] * w).sum() / w.sum())


def trending_tickers(news_df: pd.DataFrame, symbols: set[str], top: int = 20) -> pd.DataFrame:
    """Mã được nhắc nhiều nhất trong tin thị trường + cảm xúc trung bình."""
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["Mã", "Số tin", "Cảm xúc TB"])
    stop = {"CEO", "GDP", "CPI", "FDI", "USD", "VND", "ETF", "IPO", "EPS", "ROE", "VAT", "FED", "MSCI", "FTSE",
            "HNX", "TPHCM", "BOT", "TOP", "API", "NAV", "ESG", "PMI", "OPEC", "ASEAN", "IMF", "WTO", "AI"}
    rows: dict[str, list[float]] = {}
    for _, r in news_df.iterrows():
        txt = f"{r['Tiêu đề']} {r['Tóm tắt']}"
        found = set(re.findall(r"\b[A-Z][A-Z0-9]{2}\b", txt)) & symbols - stop
        for s in found:
            rows.setdefault(s, []).append(r["Cảm xúc"])
    out = pd.DataFrame([{"Mã": k, "Số tin": len(v), "Cảm xúc TB": sum(v) / len(v)} for k, v in rows.items()])
    if out.empty:
        return pd.DataFrame(columns=["Mã", "Số tin", "Cảm xúc TB"])
    return out.sort_values("Số tin", ascending=False).head(top).reset_index(drop=True)


def hot_keywords(news_df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Chủ đề nóng: đếm cụm từ khóa theo danh sách chủ đề chính."""
    themes = {
        "Lãi suất": ["lãi suất"], "Tỷ giá": ["tỷ giá", "usd"], "Thuế quan": ["thuế quan", "thuế đối ứng", "áp thuế"],
        "Nâng hạng": ["nâng hạng", "ftse", "msci"], "Khối ngoại": ["khối ngoại", "nước ngoài"],
        "Bất động sản": ["bất động sản", "nhà ở"], "Ngân hàng": ["ngân hàng", "tín dụng"],
        "Đầu tư công": ["đầu tư công", "cao tốc", "giải ngân"], "Fed / Mỹ": ["fed", "mỹ"],
        "Trung Quốc": ["trung quốc"], "Giá dầu": ["giá dầu", "opec"], "Vàng": ["giá vàng", "vàng miếng"],
        "Trái phiếu": ["trái phiếu"], "Lợi nhuận/KQKD": ["lợi nhuận", "kết quả kinh doanh", "báo cáo tài chính"],
        "Chính trị – nhân sự": ["bổ nhiệm", "quốc hội", "nghị quyết", "bộ chính trị"],
        "Công nghệ / AI": ["chip", "bán dẫn", "trí tuệ nhân tạo", " ai "],
    }
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=["Chủ đề", "Số tin", "Cảm xúc TB"])
    txt = (" " + news_df["Tiêu đề"] + " " + news_df["Tóm tắt"] + " ").str.lower()
    rows = []
    for name, kws in themes.items():
        mask = txt.apply(lambda t: any(k in t for k in kws))
        if mask.sum():
            rows.append({"Chủ đề": name, "Số tin": int(mask.sum()), "Cảm xúc TB": news_df.loc[mask, "Cảm xúc"].mean()})
    return pd.DataFrame(rows).sort_values("Số tin", ascending=False).head(top).reset_index(drop=True)
