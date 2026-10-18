import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import numpy as np

C = dict(embed="#F6C9D0", attn="#FBD9A6", ffn="#BFE3F5", norm="#F7F3B0",
         lin="#D5D8F0", soft="#C9E8C9", gray="#F0F0F0", dot="#C9C3E8")

def box(ax, cx, cy, w, h, text, color, fs=9, z=3, lw=1.2):
    p = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                       fc=color, ec="black", lw=lw, zorder=z)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=z+1)

def arrow(ax, p0, p1, z=2):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", lw=1.1, color="black", shrinkA=0, shrinkB=0), zorder=z)

def line(ax, pts, z=2):
    xs, ys = zip(*pts); ax.plot(xs, ys, color="black", lw=1.1, zorder=z, solid_capstyle="butt")

def plus(ax, cx, cy):
    ax.add_patch(Circle((cx, cy), 0.17, fc="white", ec="black", lw=1.1, zorder=3))
    ax.plot([cx-0.17, cx+0.17], [cy, cy], c="black", lw=1, zorder=4)
    ax.plot([cx, cx], [cy-0.17, cy+0.17], c="black", lw=1, zorder=4)

def sine(ax, cx, cy):
    ax.add_patch(Circle((cx, cy), 0.27, fc="white", ec="black", lw=1.1, zorder=3))
    t = np.linspace(-1, 1, 50)
    ax.plot(cx + 0.2*t, cy + 0.12*np.sin(np.pi*t*1.0), c="black", lw=1, zorder=4)

def residual(ax, cx, y_from, y_to, right_edge=1.0, off=0.28):
    line(ax, [(cx, y_from), (cx, y_from)])
    xr = cx + right_edge + off
    line(ax, [(cx, y_from), (xr, y_from), (xr, y_to)])
    arrow(ax, (xr, y_to), (cx + right_edge, y_to))

# ---------------- Figure 1 ----------------
fig, ax = plt.subplots(figsize=(8, 11))
ax.set_xlim(0, 10.5); ax.set_ylim(-0.6, 12.0); ax.axis("off")
ex, dx = 3.0, 7.4
W, H = 2.0, 0.62

# stacks (grey rounded rectangles)
ax.add_patch(FancyBboxPatch((ex-1.45, 2.55), 2.9, 3.85, boxstyle="round,pad=0,rounding_size=0.15", fc=C["gray"], ec="black", lw=1.2, zorder=1))
ax.add_patch(FancyBboxPatch((dx-1.45, 2.55), 2.9, 6.2, boxstyle="round,pad=0,rounding_size=0.15", fc=C["gray"], ec="black", lw=1.2, zorder=1))
ax.text(ex-1.75, 4.5, "N×", fontsize=11, ha="right", va="center")
ax.text(dx+1.75, 5.6, "N×", fontsize=11, ha="left", va="center")

# --- encoder
ax.text(ex, -0.3, "Inputs", ha="center", fontsize=9)
arrow(ax, (ex, -0.05), (ex, 0.7))
box(ax, ex, 1.0, W, H, "Input\nEmbedding", C["embed"])
arrow(ax, (ex, 1.31), (ex, 1.83)); plus(ax, ex, 2.0)
sine(ax, ex-0.95, 2.0); arrow(ax, (ex-0.68, 2.0), (ex-0.17, 2.0))
ax.text(ex-1.65, 2.0, "Positional\nEncoding", ha="right", va="center", fontsize=8.5)
arrow(ax, (ex, 2.17), (ex, 3.0))
enc = [("Multi-Head\nAttention", C["attn"], 3.45), ("Add & Norm", C["norm"], 4.3),
       ("Feed\nForward", C["ffn"], 5.0), ("Add & Norm", C["norm"], 5.75)]
for t, c, y in enc:
    box(ax, ex, y, W, H if "Add" in t else 0.7, t, c, fs=8.5)
for y0, y1 in [(3.8, 4.0), (4.6, 4.65), (5.35, 5.45)]:
    arrow(ax, (ex, y0), (ex, y1))
for xo in (-0.55, 0, 0.55):
    arrow(ax, (ex+xo, 2.72), (ex+xo, 3.1))
line(ax, [(ex, 2.72), (ex, 2.72)])
line(ax, [(ex-0.55, 2.72), (ex+0.55, 2.72)])
residual(ax, ex, 2.75, 4.3); residual(ax, ex, 4.65, 5.75)

# --- decoder
ax.text(dx, -0.3, "Outputs\n(shifted right)", ha="center", va="center", fontsize=9)
arrow(ax, (dx, 0.05), (dx, 0.7))
box(ax, dx, 1.0, W, H, "Output\nEmbedding", C["embed"])
arrow(ax, (dx, 1.31), (dx, 1.83)); plus(ax, dx, 2.0)
sine(ax, dx+0.95, 2.0); arrow(ax, (dx+0.68, 2.0), (dx+0.17, 2.0))
ax.text(dx+1.65, 2.0, "Positional\nEncoding", ha="left", va="center", fontsize=8.5)
arrow(ax, (dx, 2.17), (dx, 3.0))
dec = [("Masked\nMulti-Head\nAttention", C["attn"], 3.55, 0.85), ("Add & Norm", C["norm"], 4.5, H),
       ("Multi-Head\nAttention", C["attn"], 5.8, 0.7), ("Add & Norm", C["norm"], 6.75, H),
       ("Feed\nForward", C["ffn"], 7.5, 0.7), ("Add & Norm", C["norm"], 8.25, H)]
for t, c, y, h in dec:
    box(ax, dx, y, W, h, t, c, fs=8.5)
for y0, y1 in [(3.98, 4.19), (6.15, 6.44), (6.94, 7.15), (7.85, 7.94)]:
    arrow(ax, (dx, y0), (dx, y1))
# masked MHA inputs
for xo in (-0.55, 0, 0.55):
    arrow(ax, (dx+xo, 2.72), (dx+xo, 3.0))
line(ax, [(dx-0.55, 2.72), (dx+0.55, 2.72)])
residual(ax, dx, 2.75, 4.5)
# stem after Add&Norm 1, Q branch to cross attention, residual 2
line(ax, [(dx, 4.81), (dx, 4.95)])
line(ax, [(dx, 4.95), (dx+1.28, 4.95), (dx+1.28, 6.75)])
arrow(ax, (dx+1.28, 6.75), (dx+1.0, 6.75))
line(ax, [(dx, 4.95), (dx+0.55, 4.95)])
arrow(ax, (dx+0.55, 4.95), (dx+0.55, 5.45))
# K, V from encoder output
line(ax, [(ex, 6.4), (ex, 6.9)])
line(ax, [(ex, 6.9), (dx-2.0, 6.9), (dx-2.0, 5.15), (dx, 5.15)])
arrow(ax, (dx-0.55, 5.15), (dx-0.55, 5.45)); arrow(ax, (dx, 5.15), (dx, 5.45))
# residual 3
residual(ax, dx, 7.05, 8.25)
# top of decoder
arrow(ax, (dx, 8.56), (dx, 9.1))
box(ax, dx, 9.4, W, 0.5, "Linear", C["lin"])
arrow(ax, (dx, 9.65), (dx, 10.0))
box(ax, dx, 10.25, W, 0.5, "Softmax", C["soft"])
arrow(ax, (dx, 10.5), (dx, 10.95))
ax.text(dx, 11.3, "Output\nProbabilities", ha="center", va="center", fontsize=9)
ax.set_title("Figure 1 (redrawn): The Transformer – model architecture", fontsize=10, pad=2)
fig.savefig("fig1_transformer_architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- Figure 2 ----------------
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 12); ax.set_ylim(-0.2, 7.4); ax.axis("off")
# left: scaled dot-product
lx = 2.2
ax.text(lx+0.6, 7.0, "Scaled Dot-Product Attention", ha="center", fontsize=11, family="serif")
items = [("MatMul", C["dot"], 1.4), ("Scale", C["norm"], 2.3), ("Mask (opt.)", "#F5B5C8", 3.2),
         ("SoftMax", C["soft"], 4.1), ("MatMul", C["dot"], 5.0)]
for t, c, y in items:
    box(ax, lx, y, 1.5, 0.5, t, c, fs=9)
for y0, y1 in [(1.65, 2.05), (2.55, 2.95), (3.45, 3.85), (4.35, 4.75)]:
    arrow(ax, (lx, y0), (lx, y1))
arrow(ax, (lx, 5.25), (lx, 6.2))
ax.text(lx-0.5, 0.45, "Q", ha="center", fontsize=10); ax.text(lx+0.5, 0.45, "K", ha="center", fontsize=10)
ax.text(lx+1.45, 0.45, "V", ha="center", fontsize=10)
arrow(ax, (lx-0.5, 0.7), (lx-0.5, 1.15)); arrow(ax, (lx+0.5, 0.7), (lx+0.5, 1.15))
line(ax, [(lx+1.45, 0.7), (lx+1.45, 5.0)]); arrow(ax, (lx+1.45, 5.0), (lx+0.75, 5.0))

# right: multi-head
rx = 8.2
ax.text(rx, 7.0, "Multi-Head Attention", ha="center", fontsize=11, family="serif")
def stack_box(cx, cy, w, h, text, color, n=3, d=0.1, fs=9):
    for i in range(n, 0, -1):
        ax.add_patch(FancyBboxPatch((cx-w/2+i*d, cy-h/2+i*d), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=color, ec="#888888", lw=0.8, zorder=2))
    box(ax, cx, cy, w, h, text, color, fs=fs, z=3)
box(ax, rx, 6.2, 1.0, 0.45, "Linear", C["lin"])
box(ax, rx, 5.3, 1.1, 0.45, "Concat", C["norm"])
arrow(ax, (rx, 5.55), (rx, 5.98)); 
arrow(ax, (rx, 5.98), (rx, 6.43)) if False else None
arrow(ax, (rx, 6.42), (rx, 6.75)) if False else None
stack_box(rx, 3.8, 3.4, 0.9, "Scaled Dot-Product\nAttention", C["dot"])
ax.text(rx+2.15, 4.0, "h", fontsize=10)
arrow(ax, (rx, 4.45), (rx, 5.07))
for xo, nm in zip((-1.1, 0, 1.1), ("V", "K", "Q")):
    stack_box(rx+xo, 2.3, 0.95, 0.42, "Linear", C["lin"], n=2, d=0.08, fs=8)
    arrow(ax, (rx+xo, 2.55), (rx+xo, 3.3))
    arrow(ax, (rx+xo, 0.7), (rx+xo, 2.06))
    ax.text(rx+xo, 0.45, nm, ha="center", fontsize=10)
arrow(ax, (rx, 6.43), (rx, 6.75))
fig.suptitle("Figure 2 (redrawn): (left) Scaled Dot-Product Attention; (right) Multi-Head Attention with h parallel heads",
             fontsize=10, y=0.04)
fig.savefig("fig2_attention_modules.png", dpi=220, bbox_inches="tight", facecolor="white")
