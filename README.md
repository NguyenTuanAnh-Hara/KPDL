# Bài tập lớn Khai phá dữ liệu – Thuật toán K-Nearest Neighbors (KNN)

Tìm hiểu thuật toán KNN và thực nghiệm chẩn đoán khối u vú lành tính / ác tính trên bộ dữ liệu
**Breast Cancer Wisconsin (Diagnostic)** (569 mẫu, 30 đặc trưng).

## Cấu trúc thư mục

| Thư mục | Nội dung |
| --- | --- |
| `0_TaiLieuGoc/` | Đề bài, mẫu báo cáo, mẫu slide |
| `1_BaoCao/` | Báo cáo Word `BTL_KPDL_KNN.docx` (theo mẫu Phenikaa) |
| `2_MaNguon/` | Mã nguồn thực nghiệm, kết quả `results.json`, hình trong `figures/` |
| `3_Slide/` | Slide LaTeX Beamer `main.tex` + bản PDF `slides_KNN.pdf` |

## Chạy thực nghiệm

Yêu cầu Python 3.8 trở lên. Dữ liệu có sẵn trong scikit-learn, không cần tải thêm.

```bash
pip install -r requirements.txt
python 2_MaNguon/knn_experiment.py   # 4 thí nghiệm -> 2_MaNguon/results.json + figures/ (~10 giây)
python 2_MaNguon/hinh_minh_hoa.py    # hình minh họa từng bước của thuật toán
```

Kết quả trùng khớp báo cáo (cố định `random_state = 42`).

## Biên dịch slide

Overleaf: nén thư mục `3_Slide` thành zip → New Project → Upload Project → Menu → Compiler: **XeLaTeX**.
Máy cá nhân (TeX Live hoặc MiKTeX): `cd 3_Slide && latexmk -xelatex main.tex`. Không dùng pdfLaTeX.

## Kết quả chính

| Thí nghiệm | Kết quả |
| --- | --- |
| 1. Chọn K (kiểm định chéo 10-fold, K lẻ) | K = 11, độ chính xác 96,93% |
| 2. Chuẩn hóa dữ liệu | 93,86% → 97,37% |
| 3. Mô hình cuối | Đúng 111/114 ca, bỏ sót 3 ca ác tính, không báo nhầm ca nào |
| 4. So sánh 5 thuật toán | KNN xếp thứ 3, sau Hồi quy Logistic và SVM (98,25%) |
