# Phân tích cổ phiếu Việt Nam – bản tối ưu

```bash
pip install -r requirements.txt
streamlit run 1080.py
python -m pytest -q tests      # kiểm thử offline (không cần mạng)
```

## Nguồn dữ liệu (miễn phí, tự động chuyển nguồn khi lỗi)
| Dữ liệu | Thứ tự nguồn |
|---|---|
| Giá ngày (OHLCV) | Vietcap (VCI) → VNDirect → TCBS → Yahoo Finance |
| Danh sách **toàn bộ** mã HOSE/HNX/UPCoM | Vietcap → VNDirect → danh sách dự phòng ~1.500 mã |
| Ngành ICB | Vietcap GraphQL → bảng ngành dự phòng |
| Chỉ số cơ bản (P/E, P/B, ROE…) | TCBS → Yahoo |
| Vĩ mô quốc tế (DXY, USD/VND, vàng, dầu, S&P, lợi suất Mỹ) | Yahoo Finance |
| Tin tức | RSS CafeF, VnExpress, Vietstock, VietnamNet, Tuổi Trẻ, Thanh Niên + Google News |

## Cấu trúc
```
1080.py              # giao diện Streamlit (6 chế độ)
vnapp/data.py        # lấy dữ liệu đa nguồn, quét theo lô 40 mã/request
vnapp/indicators.py  # chỉ báo chuẩn (Wilder RSI/ATR/ADX, Supertrend, CMF, RS vs VNINDEX…)
vnapp/signals.py     # chấm điểm 0–100 + diễn giải + kế hoạch giao dịch
vnapp/forecast.py    # ML walk-forward + nón xác suất giá
vnapp/backtest.py    # backtest có phí, thuế, T+2, khớp giá mở cửa phiên sau
vnapp/news.py        # tin tức, cảm xúc, mã/chủ đề nóng
vnapp/charts.py      # biểu đồ
vnapp/tracking.py    # thống kê truy cập (giữ từ bản cũ)
```

## Lỗi chính đã sửa so với bản cũ
1. Dự đoán giá cộng **nhiễu ngẫu nhiên** → mỗi lần tải lại ra số khác. Nay dùng bootstrap có seed cố định.
2. ML gán nhãn sai cho N phiên cuối (so sánh với NaN thành "giảm"), **không kiểm định**, lấy `predict_proba` làm "độ tin cậy". Nay kiểm định walk-forward, có gap, so với baseline.
3. Supertrend thiếu final band → sai hướng; ADX không lọc +DM/−DM; RSI dùng SMA thay vì Wilder; VWAP cộng dồn cả lịch sử.
4. Hàm `calculate_advanced_indicators` lồng bên trong chính nó (code chết, chứa `(...)`), `detect_candlestick_patterns` khai báo 2 lần.
5. Backtest tính lại toàn bộ chỉ báo ở **mỗi phiên** (O(n²)), không phí/thuế/T+2, khớp ở giá đóng cửa cùng phiên tín hiệu (nhìn trước).
6. Màu ML luôn đỏ (`ml_trend == "TĂNG"` không bao giờ đúng vì chuỗi là "TĂNG (7 ngày)").
7. Ma trận tương quan dùng **giá** thay vì **lợi nhuận** (luôn cao giả tạo).
8. Yahoo `.VN` thiếu hầu hết mã HNX/UPCoM; danh sách mã cứng có mã đã hủy niêm yết/chỉ số (HNX, VNI, FLC, ROS…); nhiều mã xếp sai ngành.
9. Chỉ báo bị tính trên đúng khoảng thời gian hiển thị → MA200 trống khi xem 3–6 tháng. Nay luôn tải thêm 260 phiên khởi động.
10. Ghi file thống kê ở mỗi lần rerun; khung thống kê `position: fixed` che nội dung trên điện thoại.

> Công cụ hỗ trợ phân tích, không phải khuyến nghị đầu tư.
