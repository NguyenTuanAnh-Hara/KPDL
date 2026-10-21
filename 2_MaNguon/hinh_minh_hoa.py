# -*- coding: utf-8 -*-
"""
Vẽ các hình minh họa cho phần lý thuyết: ý tưởng KNN và từng bước của thuật toán
trên ví dụ 6 điểm A..F, điểm mới Q(3; 3), K = 3.

Chạy:  python hinh_minh_hoa.py      -> các file h*.png trong thư mục figures/
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass
plt.rcParams.update({"font.size": 12})
RED, BLUE, GREY, GOLD = "#d62728", "#1f77b4", "#7f7f7f", "#f2b705"

# ----------------------------------------------------------------- ví dụ chạy xuyên suốt
PTS = {"A": (1, 1, "Đỏ"), "B": (2, 1, "Đỏ"), "C": (3, 1.5, "Đỏ"),
       "D": (4, 3, "Xanh"), "E": (5, 5, "Xanh"), "F": (6, 4, "Xanh")}
Q = np.array([3.0, 3.0])
K = 3
DIST = {n: float(np.hypot(x - Q[0], y - Q[1])) for n, (x, y, _) in PTS.items()}
ORDER = sorted(PTS, key=lambda n: (DIST[n], n))
COL = lambda c: RED if c == "Đỏ" else BLUE
MRK = lambda c: "o" if c == "Đỏ" else "s"
vn = lambda v, nd=2: f"{v:.{nd}f}".replace(".", ",")


def base_axes(ax, title):
    ax.set_xlim(-0.3, 7.3); ax.set_ylim(-0.3, 6.3); ax.set_aspect("equal")
    ax.grid(alpha=.3); ax.set_title(title, fontsize=13)
    ax.set_xlabel("Đặc trưng 1"); ax.set_ylabel("Đặc trưng 2")


def draw_points(ax, labels=True, faded=()):
    for n, (x, y, c) in PTS.items():
        a = 0.25 if n in faded else 1
        ax.scatter(x, y, s=170, color=COL(c), marker=MRK(c), edgecolor="k", zorder=3, alpha=a)
        if labels:
            ax.annotate(n, (x, y), textcoords="offset points", xytext=(-4, 10), fontsize=12,
                        weight="bold", alpha=a)


def draw_q(ax, color=GOLD, text="Q ?"):
    ax.scatter(*Q, s=380, marker="*", color=color, edgecolor="k", zorder=4)
    ax.annotate(text, Q, textcoords="offset points", xytext=(-38, 8), fontsize=12, weight="bold")


def legend(ax):
    ax.scatter([], [], color=RED, marker="o", edgecolor="k", label="Lớp Đỏ")
    ax.scatter([], [], color=BLUE, marker="s", edgecolor="k", label="Lớp Xanh")
    ax.scatter([], [], color=GOLD, marker="*", edgecolor="k", s=150, label="Điểm truy vấn Q")
    ax.legend(loc="upper left", fontsize=10)


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=200); plt.close()


# ----------------------------------------------------------------- H0. Ý tưởng KNN
fig, ax = plt.subplots(figsize=(6.4, 5))
red = [(1.0, 1.2), (1.6, 2.4), (2.2, 1.0), (0.8, 2.9), (2.6, 2.2), (3.6, 3.1), (1.9, 3.4)]
blue = [(4.3, 4.0), (5.0, 3.0), (4.6, 2.3), (5.6, 4.4), (3.9, 2.0), (5.3, 1.9), (4.8, 5.0)]
q = np.array([3.4, 2.7])
for p in red:
    ax.scatter(*p, s=110, color=RED, edgecolor="k", zorder=3)
for p in blue:
    ax.scatter(*p, s=110, color=BLUE, marker="s", edgecolor="k", zorder=3)
allp = np.array(red + blue); lab = np.array([0] * len(red) + [1] * len(blue))
d = np.hypot(*(allp - q).T); o = np.argsort(d)
for k, ls in [(1, ":"), (5, "-")]:
    r = d[o[k - 1]] + 0.08
    ax.add_patch(Circle(q, r, fill=False, ls=ls, lw=1.8, color=GREY))
    nr = int((lab[o[:k]] == 0).sum())
    kq = "Đỏ" if nr > k - nr else "Xanh"
    ax.text(q[0] + 0.15, q[1] + r + 0.12, f"K={k}: {nr} Đỏ, {k - nr} Xanh → {kq}", fontsize=10,
            ha="center", bbox=dict(fc="white", ec="none", alpha=.9))
ax.scatter(*q, s=380, marker="*", color=GOLD, edgecolor="k", zorder=4)
ax.set_xlim(0, 6.4); ax.set_ylim(0.3, 5.9); ax.set_aspect("equal"); ax.grid(alpha=.3)
ax.set_title("Ý tưởng KNN: biểu quyết đa số của K láng giềng", fontsize=13)
legend(ax); ax.legend(loc="lower right", fontsize=10); save("h0_y_tuong.png")
print("H0:", [(k, int((lab[o[:k]] == 0).sum())) for k in (1, 5)])

# ----------------------------------------------------------------- H1. Đầu vào: dữ liệu + điểm mới
fig, ax = plt.subplots(figsize=(6, 5))
base_axes(ax, "Dữ liệu đầu vào: tập huấn luyện và điểm truy vấn Q")
draw_points(ax); draw_q(ax); legend(ax); save("h1_dau_vao.png")

# ----------------------------------------------------------------- H2. Bước 1: chuẩn hóa (dữ liệu thật Wine)
data = load_wine()
names = list(data.feature_names)
ip, ia = names.index("proline"), names.index("alcohol")
Xs = data.data[:, [ip, ia]]; ys = data.target
cols = [("#2ca02c", "o", "Giống 1"), (BLUE, "s", "Giống 2"), (RED, "^", "Giống 3")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
for c, (col, m, lb) in enumerate(cols):
    ax.scatter(Xs[ys == c, 0], Xs[ys == c, 1], s=22, color=col, marker=m, edgecolor="k", linewidth=.3)
ax.set_xlim(0, 1800); ax.set_ylim(-900, 900)            # hai trục cùng tỉ lệ 1:1
ax.set_aspect("equal")
ax.set_xlabel("Proline (278 – 1680)"); ax.set_ylabel("Nồng độ cồn (11 – 14,8)")
ax.set_title("TRƯỚC chuẩn hóa (hai trục cùng tỉ lệ):\nđặc trưng nồng độ cồn gần như bị triệt tiêu", fontsize=11)
ax.grid(alpha=.3)
Z = StandardScaler().fit_transform(Xs)
ax = axes[1]
for c, (col, m, lb) in enumerate(cols):
    ax.scatter(Z[ys == c, 0], Z[ys == c, 1], s=22, color=col, marker=m, edgecolor="k", linewidth=.3, label=lb)
ax.set_aspect("equal"); ax.set_xlim(-2.5, 3.5); ax.set_ylim(-3, 3)
ax.set_xlabel("Proline (đã chuẩn hóa)"); ax.set_ylabel("Nồng độ cồn (đã chuẩn hóa)")
ax.set_title("SAU chuẩn hóa Z-score:\nhai đặc trưng có cùng thang đo", fontsize=11)
ax.grid(alpha=.3); ax.legend(fontsize=9, loc="lower right")
save("h2_chuan_hoa.png")

# ----------------------------------------------------------------- H3. Bước 2: đo khoảng cách
fig, ax = plt.subplots(figsize=(6, 5))
base_axes(ax, "Bước 2: khoảng cách từ Q đến mọi điểm huấn luyện")
off = {"A": (-0.55, 0.05), "B": (-0.45, -0.2), "C": (0.12, 0.0), "D": (-0.25, -0.42),
       "E": (-0.55, 0.05), "F": (0.05, 0.25)}
for n, (x, y, c) in PTS.items():
    ax.plot([Q[0], x], [Q[1], y], color=COL(c), lw=1.3, ls="--", zorder=1)
    mx, my = (Q[0] + x) / 2 + off[n][0], (Q[1] + y) / 2 + off[n][1]
    ax.text(mx, my, vn(DIST[n]), fontsize=11, color=COL(c), weight="bold",
            bbox=dict(fc="white", ec="none", alpha=.85, pad=1))
draw_points(ax); draw_q(ax); save("h3_do_khoang_cach.png")

# ----------------------------------------------------------------- H4. Bước 3: xếp hạng, lấy K gần nhất
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.05]})
ax = axes[0]
base_axes(ax, f"Bước 3: tập {K} láng giềng gần nhất N{chr(0x2083)}(Q)")
r = DIST[ORDER[K - 1]] + 0.08
ax.add_patch(Circle(Q, r, fill=True, alpha=.08, color=GOLD))
ax.add_patch(Circle(Q, r, fill=False, lw=2, color=GOLD))
draw_points(ax, faded=ORDER[K:]); draw_q(ax)
ax = axes[1]
ys_ = np.arange(len(ORDER))[::-1]
for yy, n in zip(ys_, ORDER):
    c = PTS[n][2]; inside = ORDER.index(n) < K
    ax.barh(yy, DIST[n], color=COL(c), alpha=1 if inside else 0.25, edgecolor="k", height=0.6)
    ax.text(DIST[n] + 0.05, yy, f"{vn(DIST[n])}  ({c})", va="center", fontsize=11)
ax.set_yticks(ys_); ax.set_yticklabels([f"Hạng {i + 1}: {n}" for i, n in enumerate(ORDER)])
cut = (ys_[K - 1] + ys_[K]) / 2
ax.axhline(cut, color=GOLD, lw=2.5, ls="--")
ax.text(4.75, cut, f"K = {K}", color="#b38600", fontsize=11, ha="right", va="center", bbox=dict(fc="white", ec="none"))
ax.set_xlim(0, 4.8); ax.set_xlabel("Khoảng cách đến Q")
ax.set_title("Sắp xếp khoảng cách tăng dần", fontsize=13); ax.grid(axis="x", alpha=.3)
save("h4_xep_hang.png")

# ----------------------------------------------------------------- H5. Bước 4: bỏ phiếu
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 0.8]})
ax = axes[0]
base_axes(ax, "Bước 4: biểu quyết đa số → ŷ(Q) = Đỏ")
ax.add_patch(Circle(Q, r, fill=False, lw=2, color=GOLD))
for n in ORDER[:K]:
    x, y, c = PTS[n]
    ax.annotate("", xy=Q, xytext=(x, y), arrowprops=dict(arrowstyle="->", color=COL(c), lw=2))
draw_points(ax, faded=ORDER[K:]); draw_q(ax, color=RED, text="")
ax.text(0.0, 5.5, "ŷ(Q) = Đỏ", color=RED, fontsize=12, weight="bold")
ax = axes[1]
votes = {"Đỏ": sum(PTS[n][2] == "Đỏ" for n in ORDER[:K]), "Xanh": sum(PTS[n][2] == "Xanh" for n in ORDER[:K])}
b = ax.bar(list(votes), list(votes.values()), color=[RED, BLUE], edgecolor="k", width=0.55)
for bb, (lbl, v) in zip(b, votes.items()):
    who = ", ".join(n for n in ORDER[:K] if PTS[n][2] == lbl)
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.05, f"{v} phiếu\n({who})", ha="center", fontsize=12)
ax.set_ylim(0, 3); ax.set_yticks([0, 1, 2, 3]); ax.set_ylabel("Số phiếu")
ax.set_title(f"Kết quả biểu quyết của {K} láng giềng", fontsize=13); ax.grid(axis="y", alpha=.3)
save("h5_bo_phieu.png")

# ----------------------------------------------------------------- H6. Ảnh hưởng của K (K = 1, 3, 5)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
base_axes(ax, "Ảnh hưởng của K đến kết quả phân lớp")
res = {}
for k, ls in [(1, ":"), (3, "--"), (5, "-")]:
    rr = DIST[ORDER[k - 1]] + 0.06
    ax.add_patch(Circle(Q, rr, fill=False, ls=ls, lw=1.6, color=GREY))
    nr = sum(PTS[n][2] == "Đỏ" for n in ORDER[:k])
    res[k] = "Đỏ" if nr > k - nr else "Xanh"
    ax.text(Q[0] - rr * 0.72 - 0.15, Q[1] + rr * 0.72, f"K={k}", color=GREY, fontsize=11, ha="right")
draw_points(ax); draw_q(ax)
ax.text(3.6, 0.0, "K=1 → Xanh   K=3 → Đỏ   K=5 → Đỏ", fontsize=11, ha="center",
        bbox=dict(fc="white", ec=GREY))
save("h6_anh_huong_k.png")
print("H6:", res)

# ----------------------------------------------------------------- H7. Sơ đồ khối 4 bước (ngang)
fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off")
steps = [("Đầu vào", "Tập huấn luyện D\n+ điểm truy vấn x"),
         ("Bước 1", "Khởi tạo K, độ đo\nkhoảng cách; chuẩn hóa"),
         ("Bước 2", "Tính khoảng cách\nđến mọi điểm huấn luyện"),
         ("Bước 3", "Xác định K\nláng giềng gần nhất"),
         ("Bước 4", "Biểu quyết đa số\n→ nhãn dự đoán ŷ")]
W, G = 0.17, 0.035
for i, (t, s) in enumerate(steps):
    x0 = 0.01 + i * (W + G)
    fc = "#eef3fb" if 0 < i < 4 else ("#f2f2f2" if i == 0 else "#fdeaea")
    ax.add_patch(FancyBboxPatch((x0, 0.12), W, 0.76, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=fc, ec="#333", lw=1.2))
    ax.text(x0 + W / 2, 0.68, t, ha="center", va="center", fontsize=12.5, weight="bold")
    ax.text(x0 + W / 2, 0.36, s, ha="center", va="center", fontsize=10.5)
    if i < len(steps) - 1:
        ax.annotate("", xy=(x0 + W + G - 0.004, 0.5), xytext=(x0 + W + 0.004, 0.5),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); save("h7_so_do_khoi.png")

# ----------------------------------------------------------------- H8. Hình dạng "vùng lân cận" của 3 khoảng cách
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7))
t = np.linspace(0, 2 * np.pi, 400)
shapes = [("Manhattan (L1)\nhình thoi", np.sign(np.cos(t)) * np.cos(t) ** 2, np.sign(np.sin(t)) * np.sin(t) ** 2),
          ("Euclid (L2)\nhình tròn", np.cos(t), np.sin(t)),
          ("Chebyshev (L∞)\nhình vuông", np.array([1, -1, -1, 1, 1]), np.array([1, 1, -1, -1, 1]))]
for ax, (ttl, xx, yy) in zip(axes, shapes):
    ax.fill(xx, yy, alpha=.15, color=BLUE); ax.plot(xx, yy, color=BLUE, lw=2)
    ax.scatter(0, 0, color="k", zorder=3); ax.text(0.08, 0.08, "O")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_title(ttl, fontsize=12)
fig.suptitle("Các điểm cách O đúng 1 đơn vị theo từng cách đo", fontsize=12)
save("h8_hinh_khoang_cach.png")

print("Đã vẽ các hình h0..h8 vào", FIG)
