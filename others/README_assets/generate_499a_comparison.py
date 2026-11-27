"""Regenerate the CSE 499A historical comparison figure used in the README."""
import matplotlib.pyplot as plt

labels = ["XLM-RoBERTa\n(Text)", "ResNet50\n(Image)", "Late Fusion\n(Text + Image)"]
values = [84.83, 78.70, 78.91]  # saved CSE499A_Final.ipynb test accuracy (%)

fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=180)
fig.patch.set_facecolor("#f7f9fc")
ax.set_facecolor("#f7f9fc")
bars = ax.bar(labels, values, width=0.58,
              color=["#29466f", "#9aa8b8", "#17a398"],
              edgecolor="#203a5c", linewidth=0.8)
ax.set_ylim(0, 100)
ax.set_ylabel("Saved notebook test accuracy (%)")
ax.set_title("CSE 499A · Historical model comparison", loc="left",
             fontsize=16, fontweight="bold", color="#203a5c", pad=18)
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, value + 1.4,
            f"{value:.2f}%", ha="center", va="bottom", fontweight="bold")
ax.annotate("Late fusion was 5.92 percentage points\nbelow the text-only model",
            xy=(2, 78.91), xytext=(1.1, 52),
            arrowprops=dict(arrowstyle="->", color="#d97732", lw=1.5),
            fontsize=10, color="#9a4d19",
            bbox=dict(boxstyle="round,pad=0.45", fc="#fff4e8", ec="#f0c49b"))
plt.tight_layout()
plt.savefig("499a_historical_comparison.png", bbox_inches="tight",
            facecolor=fig.get_facecolor())
