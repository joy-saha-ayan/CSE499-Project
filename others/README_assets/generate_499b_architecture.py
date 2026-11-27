"""Regenerate the schematic CSE 499B pipeline diagram for the README."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 6.2), dpi=180)
fig.patch.set_facecolor("#f6f8fb")
ax.set_facecolor("#f6f8fb")
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.2)
ax.axis("off")

def box(x, y, w, h, title, subtitle="", face="#ffffff", edge="#29466f"):
    patch = FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        linewidth=1.6, edgecolor=edge, facecolor=face)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*.65, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#203a5c")
    if subtitle:
        ax.text(x+w/2, y+h*.30, subtitle, ha="center", va="center",
                fontsize=9.2, color="#536477")

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=15, linewidth=1.7, color="#718096"))

ax.text(.45, 5.75, "CSE 499B · Bangla multimodal early-fusion pipeline",
        fontsize=17, fontweight="bold", color="#203a5c")
box(.45, 3.8, 2.1, 1.15, "BTMD dataset", "post text + associated image", "#eaf1fb")
box(3.15, 4.25, 2.25, 1.05, "Bangla text", "preprocessing → BanglaBERT", "#eef5ff")
box(3.15, 2.75, 2.25, 1.05, "Post image", "image preprocessing → CLIP", "#e7fbf8", "#159b91")
box(6.15, 3.45, 2.25, 1.25, "Feature preparation", "training-stat standardization\n+ concatenation")
box(9.15, 3.45, 1.7, 1.25, "CCEF head", "MLP classifier", "#e7fbf8", "#159b91")
box(11.35, 3.45, 1.25, 1.25, "Prediction", "3 classes", "#fff4e8", "#d97732")
arrow(2.55, 4.38, 3.15, 4.78)
arrow(2.55, 4.18, 3.15, 3.28)
arrow(5.4, 4.78, 6.15, 4.18)
arrow(5.4, 3.28, 6.15, 3.85)
arrow(8.4, 4.08, 9.15, 4.08)
arrow(10.85, 4.08, 11.35, 4.08)
box(3.15, .75, 2.25, 1.05, "Training protocol", "AdamW · weighted CE\n3 seeds · early stopping")
box(6.15, .75, 2.25, 1.05, "Validation", "model selection\nmacro-F1")
box(9.15, .75, 3.45, 1.05, "Test + controls", "held-out evaluation · ablations\nshortcut / data-quality checks")
arrow(5.4, 1.28, 6.15, 1.28)
arrow(8.4, 1.28, 9.15, 1.28)
ax.text(.5, .25,
        "Encoders frozen in the reported experiment. Research classifier only—not an automated fact-checker.",
        fontsize=9.5, color="#65758a")
plt.tight_layout()
plt.savefig("499b_architecture.png", bbox_inches="tight", facecolor=fig.get_facecolor())
