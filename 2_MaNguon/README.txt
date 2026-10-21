MA NGUON THUC NGHIEM KNN
========================
Yeu cau: Python 3.8 tro len.

1) Cai thu vien (chi can 1 lan):
     pip install -r requirements.txt

2) Chay 4 thi nghiem (khoang 10 giay):
     python knn_experiment.py
   -> results.json + 4 hinh tn1..tn4 trong figures/
      TN1 chon K | TN2 chuan hoa | TN3 ket qua cuoi | TN4 so sanh thuat toan

3) Ve hinh minh hoa cac buoc thuat toan:
     python hinh_minh_hoa.py
   -> 9 hinh h0..h8 trong figures/

Ghi chu:
- Chay duoc tu thu muc bat ky; ket qua luon ghi canh file script.
- Du lieu Breast Cancer Wisconsin co san trong scikit-learn, khong can tai.
- Ket qua giong het bao cao (co dinh random_state = 42).
