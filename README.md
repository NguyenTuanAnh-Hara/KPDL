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
python 2_MaNguon/knn_experiment.py   # 8 thí nghiệm -> 2_MaNguon/results.json + figures/ (~30 giây)
python 2_MaNguon/figs_theory.py      # hình minh họa lý thuyết (chạy sau lệnh trên)
```

Kết quả độ chính xác trùng khớp báo cáo (cố định `random_state = 42`); riêng thời gian chạy ở thí nghiệm 8 phụ thuộc tốc độ máy.

## Biên dịch slide

Overleaf: nén thư mục `3_Slide` thành zip → New Project → Upload Project → Menu → Compiler: **XeLaTeX**.
Máy cá nhân (TeX Live hoặc MiKTeX): `cd 3_Slide && latexmk -xelatex main.tex`. Không dùng pdfLaTeX.

## Kết quả chính

| Nội dung | Kết quả |
| --- | --- |
| K tốt nhất (10-fold CV, K lẻ) | K = 11, CV accuracy 96,93% |
| Mô hình cuối (Z-score + KNN, Euclid) | Test accuracy 97,37%, Precision ác tính 100%, Recall ác tính 92,86% |
| Ảnh hưởng chuẩn hóa | Test accuracy 93,86% → 97,37% |
| So sánh 5 thuật toán | KNN xếp thứ 3, sau Logistic Regression và SVM (98,25%) |
