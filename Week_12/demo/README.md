# 🌍 FluScope — Global Influenza Surveillance Dashboard

## Setup

### 1. Chuẩn bị file
Tạo thư mục `data/` cùng cấp với `app.py`, copy các file output vào:

```
demo/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── final_results.parquet        ← từ NB4 (BẮT BUỘC)
    ├── model_results.parquet        ← từ NB4
    ├── graph_metrics.parquet        ← từ NB3
    ├── climate_similarity_pairs.parquet  ← từ NB2
    ├── lsh_validation.parquet       ← từ NB2
    ├── risk_scores_timeseries.parquet   ← từ NB3
    └── flu_model.pkl                ← từ NB4
```

Chỉ `final_results.parquet` là bắt buộc. Các file khác tùy chọn — tab nào thiếu data sẽ hiện warning.

### 2. Cài đặt
```bash
pip install -r requirements.txt
```

### 3. Chạy
```bash
streamlit run app.py
```

Mở browser tại `http://localhost:8501`

## Các tab trong Dashboard

| Tab | Hiển thị | Output tương ứng |
|-----|----------|-------------------|
| 📊 Overview | Trend toàn cầu + pipeline | final_results.parquet |
| 🌡️ Climate → Flu Lag | Cross-correlation heatmap | final_results.parquet |
| ✈️ Hub Ranking | PageRank + TSPR bar chart | graph_metrics.parquet |
| 🗺️ Risk Score Map | Bản đồ thế giới + slider tuần | risk_scores_timeseries.parquet |
| 🔗 Communities | Community map + LSH validation | graph_metrics.parquet + lsh_validation.parquet |
| 🤖 Model Performance | Baseline vs Enhanced + per-country | model_results.parquet + final_results.parquet |
