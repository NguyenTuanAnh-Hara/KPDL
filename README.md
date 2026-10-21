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

```bash
cd 2_MaNguon
pip install numpy pandas matplotlib scikit-learn
python knn_experiment.py   # 8 thí nghiệm -> results.json + figures/
python figs_theory.py      # hình minh họa lý thuyết
```

## Biên dịch slide

Overleaf: Upload Project (nén thư mục `3_Slide`) → Menu → Compiler: **XeLaTeX**.
Máy cá nhân: `cd 3_Slide && latexmk -xelatex main.tex`.

## Kết quả chính

| Nội dung | Kết quả |
| --- | --- |
| K tốt nhất (10-fold CV, K lẻ) | K = 11, CV accuracy 96,93% |
| Mô hình cuối (Z-score + KNN, Euclid) | Test accuracy 97,37%, Precision ác tính 100%, Recall ác tính 92,86% |
| Ảnh hưởng chuẩn hóa | Test accuracy 93,86% → 97,37% |
| So sánh 5 thuật toán | KNN xếp thứ 3, sau Logistic Regression và SVM (98,25%) |
