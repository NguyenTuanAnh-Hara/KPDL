# Bài tập lớn Khai phá dữ liệu – Thuật toán K-Nearest Neighbors (KNN)

Tìm hiểu thuật toán KNN và thực nghiệm phân loại hoa **Iris** (150 bông hoa, 3 loài)
theo chiều dài và chiều rộng cánh hoa.

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
python 2_MaNguon/knn_experiment.py   # thực nghiệm Iris -> 2_MaNguon/results.json + figures/ (vài giây)
python 2_MaNguon/hinh_minh_hoa.py    # hình minh họa từng bước của thuật toán
```

Kết quả trùng khớp báo cáo (cố định `random_state = 42`).

## Biên dịch slide

Overleaf: nén thư mục `3_Slide` thành zip → New Project → Upload Project → Menu → Compiler: **XeLaTeX**.
Máy cá nhân (TeX Live hoặc MiKTeX): `cd 3_Slide && latexmk -xelatex main.tex`. Không dùng pdfLaTeX.

## Kết quả chính

| Nội dung | Kết quả |
| --- | --- |
| Dữ liệu | Iris: 120 mẫu huấn luyện, 30 mẫu kiểm tra, 2 đặc trưng cánh hoa |
| Lựa chọn K (kiểm định chéo 10-fold, K lẻ) | K = 19, độ chính xác 96,67% |
| Độ chính xác trên tập kiểm tra | 96,67% (29/30 mẫu) |
| Mẫu phân lớp sai | 1 bông Versicolor bị nhầm thành Virginica (vùng chồng lấn) |
