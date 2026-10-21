# -*- coding: utf-8 -*-
"""
Bài tập lớn Khai phá dữ liệu - Đề tài: Thuật toán K-Nearest Neighbors (KNN)
Thực nghiệm trên bộ dữ liệu Breast Cancer Wisconsin (Diagnostic).

Chạy (từ thư mục bất kỳ):  python knn_experiment.py
Cài thư viện:              pip install -r requirements.txt
Kết quả: các hình .png trong thư mục figures/ và file results.json (cạnh file này).
Dữ liệu được nạp sẵn từ scikit-learn, không cần tải thêm hay kết nối mạng.
"""
import json
import os
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
# Đường dẫn tính theo vị trí file script -> chạy được từ thư mục bất kỳ, trên mọi hệ điều hành
BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
RESULTS = os.path.join(BASE, "results.json")
os.makedirs(FIG, exist_ok=True)
# Tránh lỗi in tiếng Việt trên console Windows (cp1252/cp437)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
plt.rcParams.update({"font.size": 11, "figure.dpi": 100})
RES = {}


# ---------------------------------------------------------------------------
# 0. Cài đặt KNN từ đầu (from scratch) để minh họa đúng các bước thuật toán
# ---------------------------------------------------------------------------
class KNNScratch:
    """KNN tự cài đặt: lưu dữ liệu, tính khoảng cách Euclid, bỏ phiếu đa số."""

    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):          # Bước "huấn luyện": chỉ lưu dữ liệu (lazy learning)
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y)
        return self

    def _predict_one(self, x):
        d = np.sqrt(((self.X - x) ** 2).sum(axis=1))   # Bước 2: khoảng cách Euclid
        idx = np.argsort(d)[: self.k]                    # Bước 3: lấy K điểm gần nhất
        return Counter(self.y[idx]).most_common(1)[0][0]  # Bước 4: bỏ phiếu đa số

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X, dtype=float)])


# ---------------------------------------------------------------------------
# 1. Dữ liệu
# ---------------------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target          # target: 0 = ác tính (malignant), 1 = lành tính (benign)
df = pd.DataFrame(X, columns=data.feature_names)
RES["data"] = {
    "n_samples": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "n_malignant": int((y == 0).sum()),
    "n_benign": int((y == 1).sum()),
    "range_example": {
        "mean area": [float(df["mean area"].min()), float(df["mean area"].max())],
        "mean smoothness": [float(df["mean smoothness"].min()), float(df["mean smoothness"].max())],
    },
}

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
RES["split"] = {"train": int(len(y_tr)), "test": int(len(y_te))}
scaler = StandardScaler().fit(X_tr)     # fit chỉ trên tập train để tránh rò rỉ dữ liệu
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

# ---------------------------------------------------------------------------
# 2. Thí nghiệm 1: KNN tự cài đặt so với scikit-learn
# ---------------------------------------------------------------------------
t = time.perf_counter(); p_scr = KNNScratch(k=5).fit(X_tr_s, y_tr).predict(X_te_s); t_scr = time.perf_counter() - t
t = time.perf_counter(); p_skl = KNeighborsClassifier(n_neighbors=5).fit(X_tr_s, y_tr).predict(X_te_s); t_skl = time.perf_counter() - t
SV_SAME = int((p_scr == p_skl).sum())
RES["scratch_vs_sklearn"] = {
    "acc_scratch": accuracy_score(y_te, p_scr), "acc_sklearn": accuracy_score(y_te, p_skl),
    "same_predictions": int((p_scr == p_skl).sum()), "n_test": int(len(y_te)),
    "time_scratch_ms": t_scr * 1000, "time_sklearn_ms": t_skl * 1000,
}

# ---------------------------------------------------------------------------
# 3. Thí nghiệm 2: ảnh hưởng của K (có chuẩn hóa) + chọn K bằng 10-fold CV
# ---------------------------------------------------------------------------
Ks = list(range(1, 31))
tr_acc, cv_acc, te_acc = [], [], []
for k in Ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv_acc.append(cross_val_score(m, X_tr, y_tr, cv=cv, scoring="accuracy").mean())
    m.fit(X_tr, y_tr)
    tr_acc.append(accuracy_score(y_tr, m.predict(X_tr)))
    te_acc.append(accuracy_score(y_te, m.predict(X_te)))
# Chỉ xét K lẻ để tránh hòa phiếu ở bài toán 2 lớp
odd = [k for k in Ks if k % 2 == 1]
best_k = max(odd, key=lambda k: cv_acc[k - 1])
RES["k_sweep"] = {"K": Ks, "train": tr_acc, "cv": cv_acc, "test": te_acc, "best_k": best_k,
                  "best_cv": cv_acc[best_k - 1]}

plt.figure(figsize=(8, 4.5))
plt.plot(Ks, tr_acc, "o-", ms=4, label="Train accuracy", color="#9e9e9e")
plt.plot(Ks, cv_acc, "s-", ms=4, label="10-fold CV accuracy (trên tập train)", color="#1f77b4")
plt.plot(Ks, te_acc, "^-", ms=4, label="Test accuracy", color="#ff7f0e")
plt.axvline(best_k, ls="--", color="gray")
plt.text(best_k + 0.4, 0.997, f"K được chọn (CV, K lẻ) = {best_k}", color="gray")
plt.xlabel("Số láng giềng K"); plt.ylabel("Accuracy")
plt.title("Ảnh hưởng của K đến độ chính xác (dữ liệu đã chuẩn hóa)")
plt.xticks(range(1, 31, 2)); plt.grid(alpha=.3); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_k_sweep.png"), dpi=200); plt.close()

# ---------------------------------------------------------------------------
# 4. Thí nghiệm 3: có và không chuẩn hóa dữ liệu
# ---------------------------------------------------------------------------
cv_raw = []
for k in Ks:
    cv_raw.append(cross_val_score(KNeighborsClassifier(n_neighbors=k), X_tr, y_tr, cv=cv).mean())
RES["scaling"] = {"cv_raw": cv_raw, "cv_scaled": cv_acc,
                  "test_raw_bestk": accuracy_score(y_te, KNeighborsClassifier(n_neighbors=best_k).fit(X_tr, y_tr).predict(X_te)),
                  "test_scaled_bestk": te_acc[best_k - 1],
                  "best_raw": max(cv_raw), "best_raw_k": Ks[int(np.argmax(cv_raw))]}
plt.figure(figsize=(8, 4.5))
plt.plot(Ks, cv_raw, "o-", ms=4, label="Không chuẩn hóa", color="#d62728")
plt.plot(Ks, cv_acc, "s-", ms=4, label="Chuẩn hóa Z-score (StandardScaler)", color="#1f77b4")
plt.xlabel("Số láng giềng K"); plt.ylabel("10-fold CV accuracy")
plt.title("Ảnh hưởng của chuẩn hóa dữ liệu đến KNN")
plt.xticks(range(1, 31, 2)); plt.grid(alpha=.3); plt.legend(loc="center right")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_scaling.png"), dpi=200); plt.close()

# ---------------------------------------------------------------------------
# 5. Thí nghiệm 4: độ đo khoảng cách x cách bỏ phiếu
# ---------------------------------------------------------------------------
metrics = ["euclidean", "manhattan", "chebyshev", "cosine"]
grid = []
for met in metrics:
    for w in ["uniform", "distance"]:
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k, metric=met, weights=w))
        s = cross_val_score(m, X_tr, y_tr, cv=cv).mean()
        m.fit(X_tr, y_tr)
        grid.append({"metric": met, "weights": w, "cv": s, "test": accuracy_score(y_te, m.predict(X_te))})
RES["metric_grid"] = grid

# ---------------------------------------------------------------------------
# 6. Thí nghiệm 5: mô hình cuối cùng - ma trận nhầm lẫn và các chỉ số
# ---------------------------------------------------------------------------
final = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X_tr, y_tr)
y_pred = final.predict(X_te)
cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
RES["final"] = {
    "k": best_k, "accuracy": accuracy_score(y_te, y_pred),
    # coi "ác tính" (0) là lớp dương tính vì đây là lớp cần phát hiện
    "precision_malignant": precision_score(y_te, y_pred, pos_label=0),
    "recall_malignant": recall_score(y_te, y_pred, pos_label=0),
    "f1_malignant": f1_score(y_te, y_pred, pos_label=0),
    "precision_benign": precision_score(y_te, y_pred, pos_label=1),
    "recall_benign": recall_score(y_te, y_pred, pos_label=1),
    "f1_benign": f1_score(y_te, y_pred, pos_label=1),
    "cm": cm.tolist(),
}
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.imshow(cm, cmap="Blues")
lab = ["Ác tính", "Lành tính"]
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(lab); ax.set_yticks([0, 1]); ax.set_yticklabels(lab)
ax.set_xlabel("Nhãn dự đoán"); ax.set_ylabel("Nhãn thực tế")
ax.set_title(f"Ma trận nhầm lẫn (KNN, K = {best_k})")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_confusion.png"), dpi=200); plt.close()

# ---------------------------------------------------------------------------
# 7. Thí nghiệm 6: so sánh với các thuật toán phân lớp khác
# ---------------------------------------------------------------------------
models = {
    f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "SVM (RBF)": SVC(),
}
cmp_rows = []
for name, clf in models.items():
    pipe = make_pipeline(StandardScaler(), clf)
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv)
    t0 = time.perf_counter(); pipe.fit(X_tr, y_tr); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        p = pipe.predict(X_te)
    t_pred = (time.perf_counter() - t0) / 20
    cmp_rows.append({"model": name, "cv_mean": s.mean(), "cv_std": s.std(),
                     "test_acc": accuracy_score(y_te, p),
                     "test_f1_malignant": f1_score(y_te, p, pos_label=0),
                     "fit_ms": t_fit * 1000, "predict_ms": t_pred * 1000})
RES["compare"] = cmp_rows

# ---------------------------------------------------------------------------
# 8. Thí nghiệm 7: thời gian dự đoán theo kích thước tập huấn luyện
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
sizes = [1000, 5000, 10000, 20000, 40000]
tim = []
for n in sizes:
    Xs = rng.normal(size=(n, 30)); ys = rng.integers(0, 2, n); Xq = rng.normal(size=(200, 30))
    row = {"n": n}
    for algo in ["brute", "kd_tree"]:
        m = KNeighborsClassifier(n_neighbors=5, algorithm=algo).fit(Xs, ys)
        m.predict(Xq)  # chạy khởi động
        ts = []
        for _ in range(5):
            t0 = time.perf_counter(); m.predict(Xq); ts.append((time.perf_counter() - t0) * 1000)
        row[algo] = float(np.median(ts))
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); KNNScratch(5).fit(Xs, ys).predict(Xq); ts.append((time.perf_counter() - t0) * 1000)
    row["scratch"] = float(np.median(ts))
    tim.append(row)
RES["timing"] = tim

# ---------------------------------------------------------------------------
# 9. Hình minh họa ranh giới quyết định (2 thành phần PCA) với K = 1, 15, 50
# ---------------------------------------------------------------------------
pca = PCA(n_components=2).fit(X_tr_s)
Z_tr, Z_te = pca.transform(X_tr_s), pca.transform(X_te_s)
RES["pca_var"] = float(pca.explained_variance_ratio_.sum())
xx, yy = np.meshgrid(np.linspace(Z_tr[:, 0].min() - 1, Z_tr[:, 0].max() + 1, 400),
                     np.linspace(Z_tr[:, 1].min() - 1, Z_tr[:, 1].max() + 1, 400))
cm_bg = ListedColormap(["#f6c6c6", "#c6dbf6"]); cm_pt = ListedColormap(["#d62728", "#1f77b4"])
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
pca_res = []
for ax, k in zip(axes, [1, 15, 50]):
    m = KNeighborsClassifier(n_neighbors=k).fit(Z_tr, y_tr)
    ax.contourf(xx, yy, m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape), cmap=cm_bg, alpha=.9)
    ax.scatter(Z_tr[:, 0], Z_tr[:, 1], c=y_tr, cmap=cm_pt, s=10, edgecolor="k", linewidth=.2)
    a_tr, a_te = m.score(Z_tr, y_tr), m.score(Z_te, y_te)
    pca_res.append({"k": k, "train": a_tr, "test": a_te})
    ax.set_title(f"K = {k}  (train {a_tr:.3f}, test {a_te:.3f})")
    ax.set_xlabel("PC1")
axes[0].set_ylabel("PC2")
fig.suptitle("Ranh giới quyết định của KNN trên 2 thành phần chính (đỏ: ác tính, xanh: lành tính)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_boundary.png"), dpi=200); plt.close()
RES["pca_boundary"] = pca_res

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(RES, f, ensure_ascii=False, indent=2, default=float)

# ---------------------------------------------------------------------------
# Tóm tắt kết quả ra màn hình
# ---------------------------------------------------------------------------
print("=" * 64)
print(f"Dữ liệu: {RES['data']['n_samples']} mẫu, {RES['data']['n_features']} đặc trưng | "
      f"train {RES['split']['train']} / test {RES['split']['test']}")
print(f"TN1  Tự cài đặt vs scikit-learn: trùng {SV_SAME}/{len(y_te)} dự đoán")
print(f"TN2  K tốt nhất (CV, K lẻ) = {best_k}, CV acc = {cv_acc[best_k - 1]:.4f}")
print(f"TN3  Không chuẩn hóa: CV tốt nhất {max(cv_raw):.4f} | có chuẩn hóa: {cv_acc[best_k - 1]:.4f}")
for g in grid:
    print(f"TN4  {g['metric']:<10} {g['weights']:<9} CV {g['cv']:.4f}  test {g['test']:.4f}")
print(f"TN5  Test accuracy {RES['final']['accuracy']:.4f} | Recall ác tính {RES['final']['recall_malignant']:.4f} "
      f"| ma trận nhầm lẫn {RES['final']['cm']}")
for r in cmp_rows:
    print(f"TN7  {r['model']:<20} CV {r['cv_mean']:.4f}  test {r['test_acc']:.4f}")
for t in tim:
    print(f"TN8  n={t['n']:>6}: tự cài {t['scratch']:8.1f} ms | KD-Tree {t['kd_tree']:7.1f} ms | brute {t['brute']:6.1f} ms")
print(f"Đã lưu: {RESULTS}")
print(f"Hình:   {FIG}")
