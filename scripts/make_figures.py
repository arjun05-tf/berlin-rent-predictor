"""Render README figures into docs/. Numbers come from scripts/train.py and scripts/analyze_errors.py output."""

from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).parent.parent / "docs"
BG, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": BG, "figure.facecolor": BG})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def models():
    names = ["Mean baseline", "Median baseline", "Linear regression", "Ridge", "Random forest", "LightGBM"]
    rand = [535, 509, 186, 186, 182, 162]
    grouped = [550, 507, 291, 202, 208, 194]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = range(len(names))
    ax.barh([i + 0.2 for i in y], rand, 0.38, color=BLUE, label="Random split")
    ax.barh([i - 0.2 for i in y], grouped, 0.38, color=ORANGE, label="Postal-code grouped split")
    for i, (r, g) in enumerate(zip(rand, grouped)):
        ax.text(r + 6, i + 0.2, f"€{r}", va="center", fontsize=8, color=INK)
        ax.text(g + 6, i - 0.2, f"€{g}", va="center", fontsize=8, color=INK)
    ax.set_yticks(list(y), names, color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 640)
    ax.set_xlabel("Mean absolute error (€ / month, lower is better)", color=MUTED, fontsize=9)
    ax.set_title("Unseen postal codes cost accuracy; LightGBM stays best", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=160)


def districts():
    d = {"Mitte": 241, "Prenzlauer Berg": 241, "Wilmersdorf": 240, "Charlottenburg": 220, "Tiergarten": 189,
         "Friedrichshain": 162, "Kreuzberg": 162, "Neukölln": 141, "Wedding": 94, "Spandau": 85}
    names, vals = list(d), list(d.values())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(names, vals, 0.62, color=BLUE)
    for i, v in enumerate(vals):
        ax.text(v + 4, i, f"€{v}", va="center", fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 285)
    ax.set_xlabel("Mean absolute error (€ / month) on the random test set", color=MUTED, fontsize=9)
    ax.set_title("Pricey central districts are hardest to predict", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=INK, labelsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "error_by_district.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    models()
    districts()
