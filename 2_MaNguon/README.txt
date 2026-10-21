MA NGUON THUC NGHIEM KNN
========================
Yeu cau: Python 3.8 tro len.

1) Cai thu vien (chi can 1 lan):
     pip install -r requirements.txt

2) Chay thuc nghiem (8 thi nghiem, khoang 30 giay):
     python knn_experiment.py
   -> ghi results.json va 4 hinh vao thu muc figures/

3) Ve hinh minh hoa (chay SAU buoc 2):
     python figs_theory.py
   -> ghi 5 hinh con lai vao figures/

Ghi chu:
- Co the chay tu thu muc bat ky (vd: python 2_MaNguon/knn_experiment.py tu thu muc goc):
  file ket qua luon duoc ghi canh file script.
- Du lieu Breast Cancer Wisconsin co san trong scikit-learn, khong can tai hay can mang.
- Ket qua do chinh xac giong het bao cao (da co dinh random_state = 42).
  Rieng thoi gian chay o thi nghiem 8 se khac nhau tuy toc do may.
- Google Colab: tai 2 file .py len, chay "!python knn_experiment.py" roi "!python figs_theory.py".
