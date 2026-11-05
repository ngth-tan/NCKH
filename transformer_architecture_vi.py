import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# Transformer Encoder–Decoder diagram (Vietnamese)
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor("#fbfcff")
ax.set_facecolor("#fbfcff")
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis("off")

def box(x, y, w, h, label, color, fs=10, edge="#222", bold=False):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.14",
        facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(p)
    if label:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, weight="bold" if bold else "normal")
    return p

def arrow(x1, y1, x2, y2, color="#222", rad=0, lw=1.5):
    ax.add_patch(FancyArrowPatch((x1,y1), (x2,y2),
        arrowstyle="-|>", mutation_scale=12, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}"))

BLUE, ORANGE = "#cce8fa", "#ffd49b"
YELLOW, PURPLE, PINK, GREEN = "#fff0a8", "#e5dcfa", "#f8caca", "#c9efbd"

# Title on the left, kept clear of the output head
box(0.55, 10.75, 5.0, 0.75, "KIẾN TRÚC TRANSFORMER",
    "#193b68", 19, edge="#193b68", bold=True)
ax.text(3.05, 10.53, "Mô hình Encoder – Decoder",
        ha="center", fontsize=11, color="#44546a")

# Encoder container
box(1.0, 3.15, 4.5, 6.45, "", "#edf6ff", edge="#5595c8")
ax.text(3.25, 9.25, "ENCODER (MÃ HÓA)", ha="center",
        fontsize=14, weight="bold", color="#194b78")

enc = [
    (4.0, "Multi-Head\\nAttention", ORANGE),
    (5.15, "Add & Norm\\n(Cộng và chuẩn hóa)", YELLOW),
    (6.30, "Feed Forward\\n(Mạng truyền thẳng)", BLUE),
    (7.45, "Add & Norm\\n(Cộng và chuẩn hóa)", YELLOW),
]
for y, label, color in enc:
    box(1.7, y, 3.1, 0.72, label, color, 10)
for y in [4.72, 5.87, 7.02]:
    arrow(3.25, y, 3.25, y+0.40)
# Residual paths (schematic)
arrow(1.9, 4.25, 1.9, 7.75, "#315b8a", rad=-0.18)
arrow(4.6, 5.35, 4.6, 8.55, "#315b8a", rad=0.18)
ax.text(0.62, 6.2, "N ×", fontsize=13, weight="bold")

# Decoder container
box(7.0, 3.15, 4.7, 6.45, "", "#f5f0ff", edge="#8b70bd")
ax.text(9.35, 9.25, "DECODER (GIẢI MÃ)", ha="center",
        fontsize=14, weight="bold", color="#61468e")

dec = [
    (3.55, "Masked Multi-Head\\nAttention", ORANGE),
    (4.42, "Add & Norm\\n(Cộng và chuẩn hóa)", YELLOW),
    (5.29, "Multi-Head Attention\\n(Cross Attention)", ORANGE),
    (6.16, "Add & Norm\\n(Cộng và chuẩn hóa)", YELLOW),
    (7.03, "Feed Forward\\n(Mạng truyền thẳng)", BLUE),
    (7.90, "Add & Norm\\n(Cộng và chuẩn hóa)", YELLOW),
]
for y, label, color in dec:
    box(7.75, y, 3.2, 0.62, label, color, 9.5)
for y in [4.17, 5.04, 5.91, 6.78, 7.65]:
    arrow(9.35, y, 9.35, y+0.22)
arrow(7.95, 3.8, 7.95, 8.65, "#7659a8", rad=-0.18)
arrow(10.75, 4.65, 10.75, 9.0, "#7659a8", rad=0.18)
ax.text(11.85, 6.2, "N ×", fontsize=13, weight="bold")

# Encoder output feeds decoder cross-attention
arrow(5.52, 7.8, 7.7, 5.65, "#2877a8", rad=-0.12, lw=2)
ax.text(6.35, 7.05, "K, V", fontsize=9, color="#2877a8")

# Output projection head
box(12.45, 7.95, 2.8, 0.62, "Linear\\n(Tuyến tính)", PURPLE, 10)
box(12.45, 8.85, 2.8, 0.62, "Softmax", GREEN, 11, bold=True)
arrow(11.72, 8.25, 12.42, 8.25)
arrow(13.85, 8.6, 13.85, 8.82)
ax.text(13.85, 9.7, "Xác suất từ tiếp theo",
        ha="center", fontsize=10)

# Embeddings and positional encoding
box(1.75, 2.0, 3.0, 0.65, "Input Embedding\\n(Nhúng đầu vào)", PINK, 10)
box(7.85, 2.0, 3.0, 0.65, "Output Embedding\\n(Nhúng đầu ra)", PINK, 10)
for cx in [2.5, 8.6]:
    ax.add_patch(Circle((cx, 1.45), 0.23, fill=False,
                        edgecolor="#555", linewidth=1.5))
    ax.text(cx, 1.45, "PE", ha="center", va="center", fontsize=8)
for cx in [3.25, 9.35]:
    ax.add_patch(Circle((cx, 1.45), 0.17, facecolor="white",
                        edgecolor="#222", linewidth=1.3))
    ax.text(cx, 1.45, "+", ha="center", va="center", fontsize=12)
    arrow(cx, 1.64, cx, 1.96)
arrow(3.25, 2.68, 3.25, 3.95)
arrow(9.35, 2.68, 9.35, 3.5)
ax.text(1.55, 1.42, "Mã hóa vị trí", ha="right", fontsize=9)
ax.text(7.65, 1.42, "Mã hóa vị trí", ha="right", fontsize=9)
ax.text(3.25, 0.65, 'ĐẦU VÀO\\nVí dụ: "The cat"',
        ha="center", fontsize=10)
ax.text(9.35, 0.65, 'ĐẦU RA (dịch sang phải)\\nVí dụ: "The cat sat"',
        ha="center", fontsize=10)
arrow(3.25, 0.95, 3.25, 1.20)
arrow(9.35, 0.95, 9.35, 1.20)

# Legend
legend = [(BLUE, "Feed Forward"), (ORANGE, "Attention"),
          (YELLOW, "Add & Norm"), (PINK, "Embedding")]
for i, (color, label) in enumerate(legend):
    x = 4.0 + i*2.15
    box(x, 0.08, 0.34, 0.22, "", color)
    ax.text(x+0.43, 0.19, label, va="center", fontsize=8)

plt.tight_layout()
plt.savefig("transformer_architecture_vi.png", dpi=250,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
