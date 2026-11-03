"""Cấu hình chung."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

HTTP_TIMEOUT = 12  # giây
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Biên độ dao động giá theo sàn (quy định hiện hành của UBCKNN)
PRICE_LIMIT = {"HOSE": 0.07, "HNX": 0.10, "UPCOM": 0.15}

# Phí giao dịch ước tính (phí CTCK ~0.15% mỗi chiều + thuế bán 0.1%)
FEE_BUY = 0.0015
FEE_SELL = 0.0015 + 0.001
SETTLEMENT_DAYS = 2  # T+2: cổ phiếu về tài khoản chiều T+2

INDEX_SYMBOLS = ["VNINDEX", "VN30", "HNXINDEX", "UPCOMINDEX"]

# Khoảng thời gian -> số phiên giao dịch xấp xỉ
PERIOD_TO_BARS = {"1mo": 22, "3mo": 66, "6mo": 130, "1y": 252, "2y": 504, "3y": 756, "5y": 1260}

# Số phiên tối thiểu để tính đủ chỉ báo (MA200 + đệm)
WARMUP_BARS = 260
