# -*- coding: utf-8 -*-
"""
Bài tập lớn Khai phá dữ liệu - Thuật toán K-Nearest Neighbors (KNN)
Thực nghiệm: chẩn đoán khối u vú lành tính / ác tính (Breast Cancer Wisconsin).

4 thí nghiệm:
  TN1. Chọn K tốt nhất bằng kiểm định chéo 10-fold
  TN2. So sánh có và không chuẩn hóa dữ liệu
  TN3. Đánh giá mô hình cuối cùng (ma trận nhầm lẫn)
  TN4. So sánh KNN với 4 thuật toán phân lớp khác

Chạy:  python knn_experiment.py        (cài thư viện: pip install -r requirements.txt)
Kết quả: results.json và các hình trong thư mục figures/ (nằm cạnh file này).
"""
import json
import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
try:
    sys.stdout.reconfigure(encoding="utf-8")   # in tiếng Việt không lỗi trên Windows
except AttributeError:
    pass
plt.rcParams.update({"font.size": 12})
BLUE, RED, GREY = "#1f77b4", "#d62728", "#9e9e9e"


# =============================================================================
# KNN TỰ CÀI ĐẶT theo 4 bước: khởi tạo - tính khoảng cách - xác định láng giềng - biểu quyết
# =============================================================================
class KNNScratch:
    def __init__(self, k=5):
        self.k = k                                        # Bước 1: khởi tạo tham số K

    def fit(self, X, y):                                  # Huấn luyện: lưu trữ dữ liệu
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def predict_one(self, x):
        d = np.sqrt(((self.X - x) ** 2).sum(axis=1))      # Bước 2: tính khoảng cách Euclid
        idx = np.argsort(d)[: self.k]                     # Bước 3: xác định K láng giềng
        return Counter(self.y[idx]).most_common(1)[0][0]  # Bước 4: biểu quyết đa số

    def predict(self, X):
        return np.array([self.predict_one(x) for x in np.asarray(X, dtype=float)])


# =============================================================================
# DỮ LIỆU
# =============================================================================
data = load_breast_cancer()                 # có sẵn trong scikit-learn, không cần tải
X, y = data.data, data.target               # y: 0 = ác tính, 1 = lành tính
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

R = {"data": {"n": int(len(y)), "d": int(X.shape[1]), "ac_tinh": int((y == 0).sum()),
              "lanh_tinh": int((y == 1).sum()), "train": int(len(y_tr)), "test": int(len(y_te))}}

# Kiểm tra KNN tự cài đặt cho kết quả giống thư viện scikit-learn
sc = StandardScaler().fit(X_tr)
p1 = KNNScratch(k=5).fit(sc.transform(X_tr), y_tr).predict(sc.transform(X_te))
p2 = KNeighborsClassifier(n_neighbors=5).fit(sc.transform(X_tr), y_tr).predict(sc.transform(X_te))
R["tu_cai_dat_trung_khop"] = int((p1 == p2).sum())


def knn(k, scale=True):
    """Mô hình KNN; nếu scale=True thì chuẩn hóa Z-score trước (fit trên tập train)."""
    m = KNeighborsClassifier(n_neighbors=k)
    return make_pipeline(StandardScaler(), m) if scale else m


# =============================================================================
# TN1. CHỌN K: thử K = 1..30, đo độ chính xác bằng kiểm định chéo 10-fold
# =============================================================================
Ks = list(range(1, 31))
acc_train, acc_cv = [], []
for k in Ks:
    acc_cv.append(cross_val_score(knn(k), X_tr, y_tr, cv=cv).mean())
    acc_train.append(knn(k).fit(X_tr, y_tr).score(X_tr, y_tr))
best_k = max([k for k in Ks if k % 2 == 1], key=lambda k: acc_cv[k - 1])   # chỉ xét K lẻ
R["TN1"] = {"K": Ks, "train": acc_train, "cv": acc_cv, "best_k": best_k}

plt.figure(figsize=(8, 4.3))
plt.plot(Ks, acc_train, "o-", ms=4, color=GREY, label="Tập huấn luyện")
plt.plot(Ks, acc_cv, "s-", ms=5, color=BLUE, label="Kiểm định chéo 10-fold")
plt.axvline(best_k, ls="--", color=RED)
plt.annotate(f"K = {best_k}\n{acc_cv[best_k - 1]:.1%}".replace(".", ","), (best_k, acc_cv[best_k - 1]),
             xytext=(best_k + 2.5, 0.99), color=RED, arrowprops=dict(arrowstyle="->", color=RED))
plt.text(1, 0.926, "← K nhỏ: quá khớp", color=GREY)
plt.text(30, 0.926, "K lớn: chưa khớp →", color=GREY, ha="right")
plt.xlabel("Số láng giềng K"); plt.ylabel("Độ chính xác")
plt.title("TN1. Lựa chọn tham số K")
plt.xticks(range(1, 31, 2)); plt.ylim(0.92, 1.005); plt.grid(alpha=.3); plt.legend(loc="upper right", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "tn1_chon_k.png"), dpi=200); plt.close()

# =============================================================================
# TN2. CÓ và KHÔNG chuẩn hóa (cùng K tốt nhất), đo trên tập test
# =============================================================================
acc_raw = knn(best_k, scale=False).fit(X_tr, y_tr).score(X_te, y_te)
acc_std = knn(best_k, scale=True).fit(X_tr, y_tr).score(X_te, y_te)
n_te = len(y_te)
R["TN2"] = {"khong_chuan_hoa": acc_raw, "co_chuan_hoa": acc_std,
            "sai_khong": int(round((1 - acc_raw) * n_te)), "sai_co": int(round((1 - acc_std) * n_te))}

plt.figure(figsize=(6.5, 4.3))
vals = [acc_raw, acc_std]
bars = plt.bar(["Không chuẩn hóa", "Có chuẩn hóa"], vals, color=[GREY, BLUE], width=0.55)
for b, v, s in zip(bars, vals, [R["TN2"]["sai_khong"], R["TN2"]["sai_co"]]):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2%}\n({s}/{n_te} mẫu sai)".replace(".", ","),
             ha="center", va="bottom", fontsize=12)
plt.ylim(0.85, 1.02); plt.ylabel("Độ chính xác trên tập test")
plt.title(f"TN2. Ảnh hưởng của chuẩn hóa (K = {best_k})")
plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "tn2_chuan_hoa.png"), dpi=200); plt.close()

# =============================================================================
# TN3. MÔ HÌNH CUỐI: chuẩn hóa + KNN (K tốt nhất), ma trận nhầm lẫn trên tập test
# =============================================================================
y_pred = knn(best_k).fit(X_tr, y_tr).predict(X_te)
cm = confusion_matrix(y_te, y_pred, labels=[0, 1])        # hàng: thực tế, cột: dự đoán
R["TN3"] = {"accuracy": accuracy_score(y_te, y_pred), "cm": cm.tolist(),
            "precision_ac": precision_score(y_te, y_pred, pos_label=0),
            "recall_ac": recall_score(y_te, y_pred, pos_label=0),
            "f1_ac": f1_score(y_te, y_pred, pos_label=0)}

fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.imshow([[1, 0], [0, 1]], cmap="Blues", vmin=-0.6, vmax=1.6)
txt = [["TP\n", "FN (âm tính giả)\n"], ["FP (dương tính giả)\n", "TN\n"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{txt[i][j]}{cm[i, j]}", ha="center", va="center", fontsize=14,
                color="white" if i == j else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Ác tính", "Lành tính"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Ác tính", "Lành tính"])
ax.set_xlabel("Nhãn dự đoán"); ax.set_ylabel("Nhãn thực tế")
ax.set_title(f"TN3. Ma trận nhầm lẫn ({n_te} mẫu kiểm tra, K = {best_k})")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "tn3_ma_tran_nham_lan.png"), dpi=200); plt.close()

# =============================================================================
# TN4. SO SÁNH KNN với 4 thuật toán khác (cùng chuẩn hóa, cùng cách chia dữ liệu)
# =============================================================================
models = {f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k),
          "Naive Bayes": GaussianNB(),
          "Cây quyết định": DecisionTreeClassifier(random_state=42),
          "Hồi quy Logistic": LogisticRegression(max_iter=5000),
          "SVM": SVC()}
R["TN4"] = {}
for name, clf in models.items():
    R["TN4"][name] = make_pipeline(StandardScaler(), clf).fit(X_tr, y_tr).score(X_te, y_te)

items = sorted(R["TN4"].items(), key=lambda t: t[1])
plt.figure(figsize=(8, 4.3))
bars = plt.barh([n for n, _ in items], [v for _, v in items],
                color=[BLUE if n.startswith("KNN") else GREY for n, _ in items], height=0.6)
for b, (_, v) in zip(bars, items):
    plt.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.2%}".replace(".", ","), va="center")
plt.xlim(0.85, 1.0); plt.xlabel("Độ chính xác trên tập test")
plt.title("TN4. So sánh với các thuật toán phân lớp"); plt.grid(axis="x", alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "tn4_so_sanh.png"), dpi=200); plt.close()

# =============================================================================
# LƯU KẾT QUẢ VÀ IN TÓM TẮT
# =============================================================================
with open(os.path.join(BASE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)

print("=" * 60)
print(f"Dữ liệu: {R['data']['n']} mẫu, {R['data']['d']} đặc trưng "
      f"({R['data']['ac_tinh']} ác tính, {R['data']['lanh_tinh']} lành tính)")
print(f"Chia: {R['data']['train']} mẫu train, {R['data']['test']} mẫu test")
print(f"KNN tự cài đặt trùng khớp scikit-learn: {R['tu_cai_dat_trung_khop']}/{n_te} dự đoán")
print(f"TN1  K tốt nhất = {best_k}  (kiểm định chéo {acc_cv[best_k - 1]:.2%})")
print(f"TN2  Không chuẩn hóa {acc_raw:.2%}  ->  Có chuẩn hóa {acc_std:.2%}")
print(f"TN3  Phân lớp đúng {cm[0, 0] + cm[1, 1]}/{n_te} mẫu | FN (âm tính giả) {cm[0, 1]} | FP (dương tính giả) {cm[1, 0]}")
for name, v in sorted(R["TN4"].items(), key=lambda t: -t[1]):
    print(f"TN4  {name:<17} {v:.2%}")
print("Đã lưu results.json và các hình trong", FIG)
