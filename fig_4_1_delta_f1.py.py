import matplotlib.pyplot as plt

# Du lieu uoc luong tu bieu do mau
models = [
    "Transformer\n(base)",
    "ConvS2S",
    "MoE",
    "Transformer\n(big)",
    "GNMT + RL",
    "ConvS2S\nEnsemble",
    "GNMT + RL\nEnsemble"
]

training_cost = [5e18, 1e19, 1.5e19, 1.6e19, 1.7e19, 8e19, 1.5e20]
bleu_scores = [27.3, 25.15, 26.03, 28.4, 24.6, 26.35, 26.3]

colors = [
    "saddlebrown",
    "darkorange",
    "green",
    "deeppink",
    "royalblue",
    "mediumorchid",
    "red"
]

# Tao bieu do
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(training_cost, bleu_scores, c=colors, s=65)

# Them ten cac mo hinh
for name, x, y in zip(models, training_cost, bleu_scores):
    ax.annotate(
        name,
        (x, y),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )

# Cau hinh bieu do
ax.set_xscale("log")
ax.set_title(
    "Translation quality vs. training cost: selected models from Table 2",
    fontsize=10
)
ax.set_xlabel("Training cost (FLOPs, logarithmic scale)")
ax.set_ylabel("BLEU score on WMT 2014 English → German")
ax.set_ylim(23.8, 29.2)
ax.grid(True, linestyle="--", alpha=0.3)

# Ghi chu nguon du lieu
ax.text(
    0.02,
    0.02,
    "Source: Attention Is All You Need, Table 2.\n"
    "Note: Values are approximate estimates read from the supplied image.",
    transform=ax.transAxes,
    fontsize=7
)

fig.tight_layout()

# Luu bieu do thanh anh PNG
plt.savefig("bieu_do_F1.png", dpi=300, bbox_inches="tight")

# Hien thi bieu do
plt.show()
