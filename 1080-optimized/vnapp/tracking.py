"""Thống kê truy cập (giữ tính năng cũ nhưng chỉ ghi file khi cần, ghi nguyên tử, không ghi mỗi lần rerun)."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta

from .config import DATA_DIR

VISITOR_FILE = DATA_DIR / "visitors.json"
STATS_FILE = DATA_DIR / "stats.json"
ONLINE_FILE = DATA_DIR / "online_users.json"


def _load(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save(path, data):
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:
        pass


def _stats():
    s = _load(STATS_FILE, {})
    s.setdefault("daily_visits", {})
    s.setdefault("peak_online", 0)
    s.setdefault("total_searches", 0)
    s.setdefault("popular_stocks", {})
    if "total_visits" not in s:  # kế thừa số liệu từ bản cũ
        s["total_visits"] = _load(VISITOR_FILE, {}).get("total_visits", 0)
    return s


def init_session(st):
    now = time.time()
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
        s = _stats()
        s["total_visits"] = s.get("total_visits", 0) + 1
        today = datetime.now().strftime("%Y-%m-%d")
        s["daily_visits"][today] = s["daily_visits"].get(today, 0) + 1
        cutoff = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        s["daily_visits"] = {k: v for k, v in s["daily_visits"].items() if k >= cutoff}
        _save(STATS_FILE, s)
        st.session_state.last_ping = 0
    if now - st.session_state.get("last_ping", 0) > 60:  # tối đa 1 lần/phút
        online = {k: v for k, v in _load(ONLINE_FILE, {}).items() if now - v < 300}
        online[st.session_state.session_id] = now
        _save(ONLINE_FILE, online)
        st.session_state.last_ping = now
        s = _stats()
        if len(online) > s["peak_online"]:
            s["peak_online"] = len(online)
            s["peak_online_time"] = datetime.now().isoformat()
            _save(STATS_FILE, s)


def track_search(symbol: str):
    s = _stats()
    s["popular_stocks"][symbol] = s["popular_stocks"].get(symbol, 0) + 1
    s["total_searches"] += 1
    _save(STATS_FILE, s)


def summary() -> dict:
    s = _stats()
    now = time.time()
    online = sum(1 for v in _load(ONLINE_FILE, {}).values() if now - v < 300)
    v = _load(VISITOR_FILE, {})
    total = max(s.get("total_visits", 0), v.get("total_visits", 0))
    top = sorted(s["popular_stocks"].items(), key=lambda x: -x[1])[:5]
    return {"online": online, "total_visits": total, "searches": s["total_searches"],
            "today": s["daily_visits"].get(datetime.now().strftime("%Y-%m-%d"), 0), "top": top,
            "peak": s["peak_online"]}
