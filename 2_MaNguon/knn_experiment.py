# -*- coding: utf-8 -*-
"""
Bài tập lớn Khai phá dữ liệu - Thuật toán K-Nearest Neighbors (KNN)
Thực nghiệm: phân loại hoa Iris theo chiều dài và chiều rộng cánh hoa.

Các bước:
  1. Đọc dữ liệu, chia tập huấn luyện (80%) / tập kiểm tra (20%)
  2. Cài đặt KNN theo 4 bước của thuật toán
  3. Lựa chọn K bằng kiểm định chéo 10-fold
  4. Đánh giá trên tập kiểm tra: độ chính xác, ma trận nhầm lẫn, vùng phân lớp

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
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
try:
    sys.stdout.reconfigure(encoding="utf-8")   # in tiếng Việt không lỗi trên Windows
except AttributeError:
    pass
plt.rcParams.update({"font.size": 12})
TEN_LOP = ["Setosa", "Versicolor", "Virginica"]
MAU = ["#2ca02c", "#1f77b4", "#d62728"]
DAU = ["o", "s", "^"]


# =============================================================================
# 1. DỮ LIỆU: 150 bông hoa Iris, dùng 2 đặc trưng về cánh hoa (đơn vị cm)
# =============================================================================
iris = load_iris()                                  # có sẵn trong scikit-learn
X = iris.data[:, [2, 3]]                            # chiều dài, chiều rộng cánh hoa
y = iris.target                                     # 0 Setosa, 1 Versicolor, 2 Virginica
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# =============================================================================
# 2. KNN TỰ CÀI ĐẶT theo 4 bước của thuật toán
# =============================================================================
def knn_predict(X_train, y_train, x, k):
    d = np.sqrt(((X_train - x) ** 2).sum(axis=1))      # Bước 2: tính khoảng cách Euclid
    idx = np.argsort(d)[:k]                             # Bước 3: xác định K láng giềng gần nhất
    return Counter(y_train[idx]).most_common(1)[0][0]   # Bước 4: biểu quyết đa số


# =============================================================================
# 3. LỰA CHỌN K bằng kiểm định chéo 10-fold trên tập huấn luyện (Bước 1)
# =============================================================================
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
Ks = list(range(1, 31))
acc_cv = [cross_val_score(KNeighborsClassifier(n_neighbors=k), X_tr, y_tr, cv=cv).mean() for k in Ks]
acc_train = [KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr).score(X_tr, y_tr) for k in Ks]
k_le = [k for k in Ks if k % 2 == 1]
best_k = max(k_le, key=lambda k: (round(acc_cv[k - 1], 6), -k))   # K lẻ có độ chính xác cao nhất (nhỏ nhất nếu bằng nhau)

# =============================================================================
# 4. ĐÁNH GIÁ trên tập kiểm tra
# =============================================================================
y_pred = np.array([knn_predict(X_tr, y_tr, x, best_k) for x in X_te])
y_skl = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr, y_tr).predict(X_te)
acc = accuracy_score(y_te, y_pred)
cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2])
sai = np.where(y_pred != y_te)[0]
chi_tiet_sai = []
for i in sai:
    d = np.sqrt(((X_tr - X_te[i]) ** 2).sum(axis=1))
    phieu = np.bincount(y_tr[np.argsort(d)[:best_k]], minlength=3)
    chi_tiet_sai.append({"canh_hoa_cm": X_te[i].tolist(), "that": TEN_LOP[y_te[i]],
                         "du_doan": TEN_LOP[y_pred[i]], "phieu": dict(zip(TEN_LOP, phieu.tolist()))})

R = {"so_mau": int(len(y)), "train": int(len(y_tr)), "test": int(len(y_te)),
     "K": Ks, "acc_cv": acc_cv, "acc_train": acc_train, "best_k": best_k,
     "acc_test": acc, "dung": int((y_pred == y_te).sum()), "cm": cm.tolist(),
     "trung_khop_sklearn": int((y_pred == y_skl).sum()), "mau_sai": chi_tiet_sai}
with open(os.path.join(BASE, "results.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=float)

# =============================================================================
# 5. HÌNH
# =============================================================================
def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=200); plt.close()


# (a) Dữ liệu
plt.figure(figsize=(7, 5))
for c in range(3):
    plt.scatter(X_tr[y_tr == c, 0], X_tr[y_tr == c, 1], color=MAU[c], marker=DAU[c], edgecolor="k",
                s=45, alpha=.8, label=f"{TEN_LOP[c]} (huấn luyện)")
plt.scatter(X_te[:, 0], X_te[:, 1], facecolor="none", edgecolor="k", s=140, lw=1.3, label="Mẫu kiểm tra")
plt.xlabel("Chiều dài cánh hoa (cm)"); plt.ylabel("Chiều rộng cánh hoa (cm)")
plt.title("Dữ liệu Iris: 150 bông hoa, 3 loài"); plt.grid(alpha=.3); plt.legend(fontsize=9)
save("tn_du_lieu.png")

# (b) Lựa chọn K
plt.figure(figsize=(8, 4.3))
plt.plot(Ks, acc_train, "o-", ms=4, color="#9e9e9e", label="Tập huấn luyện")
plt.plot(Ks, acc_cv, "s-", ms=5, color="#1f77b4", label="Kiểm định chéo 10-fold")
plt.axvline(best_k, ls="--", color="#d62728")
plt.annotate(f"K = {best_k}\n{acc_cv[best_k - 1]:.2%}".replace(".", ","), (best_k, acc_cv[best_k - 1]),
             xytext=(best_k + 2.5, 0.925), color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
plt.xlabel("Số láng giềng K"); plt.ylabel("Độ chính xác")
plt.title("Lựa chọn tham số K"); plt.xticks(range(1, 31, 2)); plt.ylim(0.9, 1.005)
plt.grid(alpha=.3); plt.legend(loc="lower left", fontsize=10)
save("tn_chon_k.png")

# (c) Ma trận nhầm lẫn
fig, ax = plt.subplots(figsize=(5.4, 4.6))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(3)); ax.set_xticklabels(TEN_LOP); ax.set_yticks(range(3)); ax.set_yticklabels(TEN_LOP)
ax.set_xlabel("Nhãn dự đoán"); ax.set_ylabel("Nhãn thực tế")
ax.set_title(f"Ma trận nhầm lẫn ({len(y_te)} mẫu kiểm tra, K = {best_k})")
save("tn_ma_tran.png")

# (d) Vùng phân lớp
xx, yy = np.meshgrid(np.linspace(0.5, 7.5, 500), np.linspace(-0.2, 2.9, 500))
Z = KNeighborsClassifier(n_neighbors=best_k).fit(X_tr, y_tr).predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.figure(figsize=(7.5, 5))
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], cmap=ListedColormap(["#d9f2d9", "#d6e6f5", "#f8d7d7"]))
for c in range(3):
    m = y_te == c
    plt.scatter(X_te[m, 0], X_te[m, 1], color=MAU[c], marker=DAU[c], edgecolor="k", s=70, label=TEN_LOP[c])
for i in sai:
    plt.scatter(*X_te[i], s=420, facecolor="none", edgecolor="k", lw=2)
    plt.annotate(f"Phân lớp sai: {TEN_LOP[y_te[i]]} → {TEN_LOP[y_pred[i]]}", X_te[i],
                 xytext=(X_te[i][0] - 4.3, X_te[i][1] + 0.55), fontsize=10,
                 arrowprops=dict(arrowstyle="->"), bbox=dict(fc="white", ec="gray"))
plt.xlabel("Chiều dài cánh hoa (cm)"); plt.ylabel("Chiều rộng cánh hoa (cm)")
plt.title(f"Vùng phân lớp của KNN (K = {best_k}) và {len(y_te)} mẫu kiểm tra")
plt.legend(loc="lower right", fontsize=10)
save("tn_vung_phan_lop.png")

# =============================================================================
# 6. TÓM TẮT
# =============================================================================
print("=" * 60)
print(f"Dữ liệu: {len(y)} bông hoa, 2 đặc trưng | {len(y_tr)} mẫu huấn luyện, {len(y_te)} mẫu kiểm tra")
print(f"K tốt nhất (kiểm định chéo 10-fold): K = {best_k} ({acc_cv[best_k - 1]:.2%})")
print(f"Độ chính xác trên tập kiểm tra: {acc:.2%} ({R['dung']}/{len(y_te)} mẫu đúng)")
print("Ma trận nhầm lẫn (hàng: thực tế, cột: dự đoán):")
print(cm)
for s in chi_tiet_sai:
    print(f"Mẫu sai: cánh hoa {s['canh_hoa_cm']} cm, thực tế {s['that']}, dự đoán {s['du_doan']}, phiếu {s['phieu']}")
print(f"KNN tự cài đặt trùng khớp scikit-learn: {R['trung_khop_sklearn']}/{len(y_te)} dự đoán")
print("Đã lưu results.json và các hình trong", FIG)
