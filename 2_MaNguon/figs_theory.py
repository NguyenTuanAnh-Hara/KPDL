# -*- coding: utf-8 -*-
"""Hình minh họa phần lý thuyết + hình tổng hợp kết quả thực nghiệm."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

FIG = "figures"
RED, BLUE, GREY = "#d62728", "#1f77b4", "#7f7f7f"
plt.rcParams.update({"font.size": 11})

# 1. Ví dụ tính tay: 6 điểm, điểm cần phân lớp Q(3,3), vòng tròn K = 1, 3, 5
pts = {"A": (1, 1, "Đỏ"), "B": (2, 1, "Đỏ"), "C": (3, 1.5, "Đỏ"),
       "D": (4, 3, "Xanh"), "E": (5, 5, "Xanh"), "F": (6, 4, "Xanh")}
Q = np.array([3, 3])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, (x, y, c) in pts.items():
    ax.scatter(x, y, s=140, color=RED if c == "Đỏ" else BLUE, marker="o" if c == "Đỏ" else "s",
               edgecolor="k", zorder=3)
    d = np.hypot(x - Q[0], y - Q[1])
    off = {"A": (-30, -22), "B": (6, -22), "E": (10, 4)}.get(name, (8, 6))
    ax.annotate(f"{name} (d={d:.2f})", (x, y), textcoords="offset points", xytext=off, fontsize=10)
ax.scatter(*Q, s=260, marker="*", color="gold", edgecolor="k", zorder=4)
ax.annotate("Q(3, 3) ?", Q, textcoords="offset points", xytext=(-30, 12), fontsize=11, weight="bold")
for r, k, ls in [(1.0, 1, ":"), (np.sqrt(5), 3, "--"), (np.sqrt(8), 5, "-")]:
    ax.add_patch(Circle(Q, r + 0.03, fill=False, ls=ls, lw=1.4, color=GREY))
    ax.text(Q[0] - (r + 0.05) * 0.71 - 0.55, Q[1] + (r + 0.05) * 0.71, f"K={k}", color=GREY, fontsize=10)
ax.scatter([], [], color=RED, marker="o", edgecolor="k", label="Lớp Đỏ")
ax.scatter([], [], color=BLUE, marker="s", edgecolor="k", label="Lớp Xanh")
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.5, 6.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
ax.legend(loc="lower right"); ax.set_title("Ví dụ: K = 1 → Xanh;  K = 3 → Đỏ;  K = 5 → Đỏ")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_example.png", dpi=200); plt.close()

# 2. Hình ý tưởng KNN: hai cụm điểm, một điểm mới, vòng K = 3 và K = 7
rng = np.random.default_rng(3)
a = rng.normal([2, 2], 0.9, (25, 2)); b = rng.normal([4.6, 4.2], 0.9, (25, 2))
q = np.array([3.3, 3.1])
allp = np.vstack([a, b]); lab = np.array([0] * 25 + [1] * 25)
d = np.hypot(*(allp - q).T); order = np.argsort(d)
fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.scatter(a[:, 0], a[:, 1], color=RED, edgecolor="k", s=50, label="Lớp A")
ax.scatter(b[:, 0], b[:, 1], color=BLUE, marker="s", edgecolor="k", s=50, label="Lớp B")
for k, ls in [(3, "--"), (7, "-")]:
    r = d[order[k - 1]] + 0.05
    ax.add_patch(Circle(q, r, fill=False, ls=ls, lw=1.5, color=GREY))
    na = int((lab[order[:k]] == 0).sum())
    ax.text(q[0] - 0.9, q[1] + r + 0.1, f"K={k}: {na} A / {k - na} B", color="k", fontsize=10,
            bbox=dict(fc="white", ec="none", alpha=.85))
for i in order[:7]:
    ax.plot([q[0], allp[i, 0]], [q[1], allp[i, 1]], color=GREY, lw=.7, zorder=1)
ax.scatter(*q, s=280, marker="*", color="gold", edgecolor="k", zorder=4, label="Điểm cần dự đoán")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="lower right", framealpha=.95)
ax.set_title("Ý tưởng KNN: láng giềng gần nhất bỏ phiếu")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_idea.png", dpi=200); plt.close()
print("idea votes", [(k, int((lab[order[:k]] == 0).sum())) for k in (3, 7)])

# 3. Sơ đồ khối các bước của thuật toán
steps = ["Dữ liệu huấn luyện D = {(xᵢ, yᵢ)}\nvà điểm cần dự đoán x",
         "Bước 1: Chọn K và hàm khoảng cách d(·,·)\n(chuẩn hóa dữ liệu trước)",
         "Bước 2: Tính d(x, xᵢ) với mọi i = 1..n",
         "Bước 3: Sắp xếp tăng dần, lấy K điểm\ngần nhất → N_K(x)",
         "Bước 4: Phân lớp: bỏ phiếu đa số\nHồi quy: lấy trung bình yᵢ",
         "Kết quả: nhãn (hoặc giá trị) dự đoán ŷ"]
fig, ax = plt.subplots(figsize=(6, 7.6)); ax.axis("off")
H, gap = 0.105, 0.058
for i, s in enumerate(steps):
    y = 0.95 - i * (H + gap)
    col = "#e8f0fb" if 0 < i < 5 else "#f2f2f2"
    ax.add_patch(FancyBboxPatch((0.08, y - H), 0.84, H, boxstyle="round,pad=0.01,rounding_size=0.02",
                                fc=col, ec="#333", lw=1.1))
    ax.text(0.5, y - H / 2, s, ha="center", va="center", fontsize=10.5)
    if i < len(steps) - 1:
        ax.annotate("", xy=(0.5, y - H - gap + 0.006), xytext=(0.5, y - H - 0.004),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_flow.png", dpi=200); plt.close()

# 4. Hình kết quả: so sánh thuật toán + thời gian dự đoán
R = json.load(open("results.json", encoding="utf-8"))
cmp = sorted(R["compare"], key=lambda r: r["test_acc"])
fig, ax = plt.subplots(figsize=(8, 4.6))
names = [r["model"] for r in cmp]; y = np.arange(len(cmp))
ax.barh(y - 0.18, [r["cv_mean"] for r in cmp], 0.36, xerr=[r["cv_std"] for r in cmp], color="#9ecae1",
        label="10-fold CV accuracy (± độ lệch chuẩn)", capsize=3)
ax.barh(y + 0.18, [r["test_acc"] for r in cmp], 0.36,
        color=[BLUE if n.startswith("KNN") else "#bdbdbd" for n in names], label="Test accuracy")
for i, r in enumerate(cmp):
    ax.text(r["test_acc"] + 0.002, i + 0.18, f"{r['test_acc']:.3f}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0.85, 1.01)
ax.set_xlabel("Accuracy"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9); ax.grid(axis="x", alpha=.3)
ax.set_title("So sánh KNN với các thuật toán phân lớp khác")
plt.tight_layout(); plt.savefig(f"{FIG}/fig_compare.png", dpi=200); plt.close()

tm = R["timing"]
fig, ax = plt.subplots(figsize=(7.5, 4))
n = [t["n"] for t in tm]
ax.plot(n, [t["scratch"] for t in tm], "o-", color=RED, label="KNN tự cài đặt (Python thuần + NumPy)")
ax.plot(n, [t["kd_tree"] for t in tm], "s-", color="#ff7f0e", label="scikit-learn, KD-Tree")
ax.plot(n, [t["brute"] for t in tm], "^-", color=BLUE, label="scikit-learn, brute force (vector hóa)")
ax.set_xlabel("Số mẫu huấn luyện n (d = 30 chiều)"); ax.set_ylabel("Thời gian dự đoán 200 điểm (ms)")
ax.set_title("Thời gian dự đoán tăng tuyến tính theo n"); ax.grid(alpha=.3); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(f"{FIG}/fig_timing.png", dpi=200); plt.close()
print("ok")
