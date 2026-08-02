# Hệ thống cảnh báo nguy cơ ngập lụt cục bộ tại khu vực hạ lưu sông Hương - sông Bồ

Niên luận (TIN3142) - Nguyễn Lê Xuân Tùng, MSSV 21T1020169, lớp K45G
Trường Đại học Khoa học, Đại học Huế. GVHD: Lê Quang Chiến.

Xây dựng mô hình phân loại nhị phân (Logistic Regression, Random Forest) dự báo nguy cơ ngập lụt tại khu vực hạ lưu sông Hương - sông Bồ, tỉnh Thừa Thiên Huế, dựa trên dữ liệu lượng mưa công khai từ Open-Meteo và danh mục sự kiện ngập lụt lịch sử.

## Cấu trúc thư mục

```
.
├── fetch_openmeteo.py       # Bước 1: tải dữ liệu mưa thô từ Open-Meteo API
├── preprocess.py            # Bước 2: tiền xử lý, tính đặc trưng, gán nhãn, chia train/test
├── flood_events.xlsx         # Danh mục 10 đợt ngập lụt lịch sử (2016-2025), tự tổng hợp
├── 03_train_evaluate.py     # Bước 3: dò siêu tham số (walk-forward CV) + huấn luyện + đánh giá
├── 04_plots.py               # Bước 4: vẽ các biểu đồ trong Chương 4 (hội tụ, ma trận nhầm lẫn, ROC/PR)
├── 05_threshold_tuning.py   # Bước 5: khảo sát ngưỡng quyết định thay thế cho 0,5
├── requirements.txt
├── data_raw/                 # (tạo tự động) dữ liệu mưa thô theo trạm
├── (tạo tự động) train.xlsx
├── (tạo tự động) test.xlsx
├── results/                  # (tạo tự động) metrics_test_results.json, test_predictions.npz
└── figures/                   # (tạo tự động) 4 biểu đồ .png dùng trong báo cáo
```

## 1. Cài đặt môi trường

Yêu cầu Python 3.10+ (đề tài phát triển và kiểm thử trên Python 3.14, scikit-learn 1.8).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`:
```
requests
pandas
numpy
scikit-learn
matplotlib
```

## 2. Chuẩn bị dữ liệu

Đề tài **không sử dụng pretrained model** - toàn bộ mô hình được huấn luyện từ đầu trên dữ liệu mưa công khai, không có bước tải trọng số có sẵn.

### Bước 2.1 - Tải dữ liệu mưa thô
```bash
python fetch_openmeteo.py
```
Gọi Open-Meteo Historical Weather API cho 2 trạm (Kim Long, Phú Ốc), giai đoạn 2016-01-01 đến 2025-12-31. Kết quả lưu vào `data_raw/kim_long_raw.xlsx` và `data_raw/phu_oc_raw.xlsx` (~3.653 dòng/trạm). Cần kết nối Internet; không cần API key.

### Bước 2.2 - Chuẩn bị danh mục sự kiện ngập lụt
File `flood_events.xlsx` đã có sẵn trong repo (10 đợt lũ 2016-2025, tổng hợp thủ công từ Bao Tuoi Tre, VietnamPlus,...). Không cần tải lại - chỉ cần đảm bảo file này nằm cùng thư mục gốc khi chạy bước tiếp theo.

### Bước 2.3 - Tiền xử lý và tạo tập train/test
```bash
python preprocess.py
```
Đọc dữ liệu từ `data_raw/`, nội suy giá trị khuyết, tính 3 đặc trưng mưa tích lũy (3/5/7 ngày), gán nhãn theo `flood_events.xlsx`, chia theo mốc thời gian (train:
2016-2023, test: 2024-2025). Kết quả: `train.xlsx` (5.844 dòng, 66 dương), `test.xlsx` (1.462 dòng, 46 dương).

## 3. Huấn luyện và đánh giá (Train + Evaluation)

```bash
python 03_train_evaluate.py
```
Thực hiện tuần tự:
1. Dò siêu tham số bằng walk-forward CV (k=3, mốc thời gian cố định - xem `FOLD_CUTS` trong script) cho cả Logistic Regression (`C`) và Random Forest (`n_estimators`, `max_depth`, `min_samples_leaf`).
2. Huấn luyện mô hình cuối cùng trên toàn bộ tập Train với siêu tham số tốt nhất.
3. Đánh giá một lần duy nhất trên tập Test (Accuracy, Precision, Recall, F1, ROC-AUC).
4. Lưu kết quả vào `results/metrics_test_results.json` và xác suất dự đoán vào `results/test_predictions.npz` (dùng cho bước 4, 5).

**Lưu ý về khả năng tái lập:** Logistic Regression cho kết quả xác định tuyệt đối (deterministic) trên mọi môi trường. Với Random Forest, dù đã cố định `random_state=42`, bước dò siêu tham số có thể chọn ra cấu hình khác nhau giữa các môi trường phần cứng/phiên bản thư viện khác nhau, do sai số dấu phẩy động ảnh hưởng đến việc phá vỡ thế cân bằng khi các điểm F1 giữa các cấu hình rất sát nhau. Kết quả chính thức báo cáo
trong luận văn được đo trên môi trường: Windows 10 Pro 64-bit, Python 3.14, scikit-learn 1.8, pandas 3.0, numpy 2.4 (chi tiết ở mục 4.1.1 của báo cáo). Nếu chạy lại trên môi trường khác, Random Forest có thể chọn `n_estimators` khác 200 và cho số liệu lệch nhẹ so với Bảng 4.2 - đây là giới hạn đã ghi nhận ở mục 5.2 của báo cáo, không phải lỗi thực thi.

Kết quả gốc dùng để viết Bảng 4.2:
| Mô hình | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (luôn đoán "không ngập") | 0,969 | 0,000 | 0,000 | 0,000 | 0,500 |
| Logistic Regression | 0,906 | 0,212 | 0,739 | 0,330 | 0,886 |
| Random Forest | 0,964 | 0,440 | 0,478 | 0,458 | 0,871 |

## 4. Vẽ biểu đồ

```bash
python 04_plots.py
```
Cần chạy sau bước 3 (đọc `results/metrics_test_results.json` và `results/test_predictions.npz`). Xuất 4 file vào `figures/`: `lr_convergence.png`, `rf_convergence.png`, `confusion_matrices.png`, `roc_pr_curves.png` - tương ứng
Hình 4.1-4.4 trong báo cáo.

## 5. Khảo sát ngưỡng quyết định

```bash
python 05_threshold_tuning.py
```
Cần chạy sau bước 3. Đọc `results/test_predictions.npz`, khảo sát đường cong Precision-Recall trên tập Test để tìm ngưỡng thay thế cho 0,5 (ngưỡng tối ưu F1, ngưỡng ưu tiên Recall ≥ 0,90) - tương ứng Bảng 4.3 trong báo cáo.

## Chạy toàn bộ pipeline từ đầu

```bash
python fetch_openmeteo.py
python preprocess.py
python 03_train_evaluate.py
python 04_plots.py
python 05_threshold_tuning.py
```

## Ghi chú

- Dữ liệu mưa lấy từ [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), miễn phí, không cần đăng ký.
- Danh mục sự kiện ngập lụt (`flood_events.xlsx`) tổng hợp thủ công từ báo cáo thiên tai chính thống; nguồn cụ thể từng dòng ghi ở cột `source`.
- Ứng dụng minh họa Web Demo (Streamlit) hiện thuộc phạm vi thiết kế (Chương 3), chưa
  triển khai trong repo này - xem hướng phát triển ở mục 5.3 của báo cáo.
