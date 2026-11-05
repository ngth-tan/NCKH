"""
Scaled Dot-Product Attention + Multi-Head Attention diagram (Vietnamese labels).

Run:    python attention_diagram.py
Output: attention_diagram.png

Coordinates are in "pixels" of a 1536 x 1024 canvas, y axis pointing DOWN,
so every box is easy to move: (x1, y1, x2, y2) = left, top, right, bottom.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path

# In the reference image the V arrow goes into the "QK^T" box.
# Set True to route V into the "Nhân với V" (weighted sum) step instead,
# which is what the attention formula actually does.
V_TO_WEIGHTED_SUM = False

# ---- palette ----------------------------------------------------------------
NAVY = "#1F3A6E"
GREEN = "#D5F0CE"
PURPLE = "#D9CCF5"
PURPLE_BACK = "#E9E2FA"
RED = "#FFBDBD"
YELLOW = "#FFE58F"
BLUE = "#CFE2FF"
EDGE = "#111111"

plt.rcParams["font.family"] = "DejaVu Sans"   # has full Vietnamese glyph coverage
plt.rcParams["mathtext.fontset"] = "dejavusans"

W, H = 1536, 1024
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # y goes down, like image coordinates
ax.set_aspect("equal")
ax.axis("off")


def box(x1, y1, x2, y2, text, color, fs=19, rounding=16, lw=2.6,
        bold=False, tcolor="black", z=2, text_dx=0):
    p = FancyBboxPatch((x1, y1), x2 - x1, y2 - y1,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=color, ec=EDGE, lw=lw, zorder=z)
    ax.add_patch(p)
    if text:
        ax.text((x1 + x2) / 2 + text_dx, (y1 + y2) / 2, text, ha="center",
                va="center", fontsize=fs, color=tcolor, linespacing=1.25,
                fontweight="bold" if bold else "normal", zorder=z + 1)


def up_arrow(x, y_from, y_to, lw=2.8):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from), zorder=5,
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=EDGE,
                                mutation_scale=24, shrinkA=0, shrinkB=0))


def right_brace(x, y1, y2, w=30, lw=2.8):
    ym = (y1 + y2) / 2
    h = w / 2
    verts = [(x, y1), (x + h, y1), (x + h, y1 + h),
             (x + h, ym - h),
             (x + h, ym), (x + w, ym),
             (x + h, ym), (x + h, ym + h),
             (x + h, y2 - h),
             (x + h, y2), (x, y2)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), fc="none", ec=EDGE, lw=lw, zorder=3))


# ============================ LEFT: Scaled Dot-Product =======================
box(26, 12, 728, 90, "Attention theo điểm (Scaled Dot-Product Attention)",
    NAVY, fs=16, rounding=22, lw=0, tcolor="white", bold=True)

box(215, 125, 561, 215, "Đầu ra\n(Attention Output)", GREEN)
box(197, 264, 581, 353, "Nhân với V\n(Weighted Sum)", PURPLE)
box(194, 396, 587, 489, "Softmax\n(Chuyển thành xác suất)", GREEN)
box(209, 521, 573, 610, "Áp dụng Mask\n(nếu cần)", RED)
box(215, 648, 573, 731, "Chia cho $\\sqrt{d_k}$\n(Scale)", YELLOW)
box(205, 763, 582, 851, "Tính điểm tương đồng\n(QK$^{T}$)", PURPLE)
box(137, 911, 296, 998, "Q\n(Query)", BLUE)
box(321, 911, 470, 998, "K\n(Key)", BLUE)
box(491, 911, 648, 998, "V\n(Value)", BLUE)

# vertical flow arrows (from top edge of lower box to bottom edge of upper box)
cx = 387
for y_from, y_to in [(263, 216), (395, 354), (520, 490), (647, 611), (762, 732)]:
    up_arrow(cx, y_from + 1, y_to)
# inputs
up_arrow(222, 910, 852)
up_arrow(390, 910, 852)
if V_TO_WEIGHTED_SUM:
    ax.plot([620, 620, 582], [910, 308, 308], color=EDGE, lw=2.8, zorder=1)
    up_arrow(620, 910, 880)   # visual hint; the elbow line carries the flow
    ax.annotate("", xy=(582, 308), xytext=(620, 308), zorder=5,
                arrowprops=dict(arrowstyle="-|>", lw=2.8, color=EDGE,
                                mutation_scale=24, shrinkA=0, shrinkB=0))
else:
    up_arrow(557, 910, 852)

# ============================ RIGHT: Multi-Head ==============================
box(832, 12, 1514, 112, "", NAVY, rounding=22, lw=0)
ax.text(1173, 44, "Multi-Head Attention", ha="center", va="center",
        fontsize=25, color="white", fontweight="bold", zorder=4)
ax.text(1173, 86, "(Attention nhiều đầu)", ha="center", va="center",
        fontsize=19, color="white", fontweight="bold", zorder=4)

box(1001, 155, 1334, 247, "Kết quả\n(Đặc trưng đầu ra)", GREEN)
box(1039, 293, 1302, 377, "Concat\n(Nối lại)", YELLOW)

# stacked "h heads" (back to front)
box(959, 409, 1434, 560, "", PURPLE_BACK, z=1)
box(925, 432, 1409, 577, "", PURPLE_BACK, z=1.5)
box(895, 455, 1385, 594, "Scaled Dot-Product\nAttention\n(cho từng đầu)", PURPLE,
    fs=20, z=2)
right_brace(1450, 445, 592)
ax.text(1505, 520, "h", fontsize=32, style="italic", family="serif",
        ha="center", va="center")

up_arrow(1168, 454, 378)       # heads -> concat
up_arrow(1168, 292, 248)       # concat -> result

# Linear layers
lin_txt = "Linear\n(Chiều tuyến tính)"
box(829, 669, 1033, 759, lin_txt, BLUE, fs=13.5, rounding=14)
box(1052, 669, 1240, 759, lin_txt, BLUE, fs=13.5, rounding=14)
box(1258, 669, 1462, 759, lin_txt, BLUE, fs=13.5, rounding=14)
for x in (949, 1139, 1340):
    up_arrow(x, 668, 595)

# inputs (V, K, Q – same order as the reference image)
box(881, 883, 985, 942, "V", BLUE, fs=22)
box(1088, 883, 1199, 942, "K", BLUE, fs=22)
box(1301, 883, 1415, 942, "Q", BLUE, fs=22)
for x in (932, 1143, 1357):
    up_arrow(x, 882, 760)

plt.savefig("attention_diagram.png", dpi=100, facecolor="white")
print("saved attention_diagram.png")
