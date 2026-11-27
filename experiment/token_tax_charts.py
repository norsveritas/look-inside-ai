"""Charts for experiment 6 (the token tax)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "token_tax", "token_tax.json")
CH = os.path.join(HERE, "..", "charts", "token_tax")
os.makedirs(CH, exist_ok=True)

KURDISH = "#257de0"
OTHER = "#b8b8b8"
TEXT, TEXT2 = "#191919", "#666666"
MODERN = ["GPT-4o (2024)", "Llama 3 (2024)", "Gemma 3 (2025)", "Qwen3 (2025)", "DeepSeek-V3 (2024)",
          "Mistral Nemo (2024)"]
ORDER = ["eng_Latn", "tur_Latn", "arb_Arab", "pes_Arab", "kmr_Latn", "ckb_Arab", "kmr_Arab"]
SHORT = {"eng_Latn": "English", "tur_Latn": "Turkish", "pes_Arab": "Persian", "arb_Arab": "Arabic",
         "ckb_Arab": "Sorani", "kmr_Latn": "Kurmanji\n(Latin letters)", "kmr_Arab": "Kurmanji/Badini\n(Arabic letters)"}


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=TEXT2, length=0)


def main_chart(d):
    vals = [np.mean([d["tokenizers"][t]["langs"][c]["vs_english"] for t in MODERN]) for c in ORDER]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(ORDER))[::-1]
    colors = [KURDISH if c in ("kmr_Latn", "ckb_Arab", "kmr_Arab") else OTHER for c in ORDER]
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="white", linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + 0.04, yi, f"{v:.1f}×", va="center", color=TEXT, fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[c] for c in ORDER], color=TEXT, fontsize=10)
    ax.axvline(1, color="#cccccc", lw=1, zorder=0)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel("tokens needed for the same sentences, compared with English (English = 1×)", color=TEXT2)
    ax.set_title("The token tax: Kurdish text costs about 2–3 times as much as English,\n"
                 "and much more than Turkish, Persian or Arabic",
                 loc="left", color=TEXT, fontsize=12)
    ax.text(0, -0.2, "Average of 6 modern AI tokenizers (GPT-4o, Llama 3, Gemma 3, Qwen3, DeepSeek-V3, Mistral Nemo). "
            "1,004 professionally translated sentences.", transform=ax.transAxes, fontsize=8, color=TEXT2)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "token_tax.png"), dpi=150)
    plt.close(fig)


def table_chart(d):
    names = list(d["tokenizers"])
    M = np.array([[d["tokenizers"][t]["langs"][c]["vs_english"] for c in ORDER] for t in names])
    cmap = LinearSegmentedColormap.from_list("blues", ["#f2f7fd", "#9cc3f0", "#257de0", "#0d3f7a"])
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.imshow(np.minimum(M, 4.5), cmap=cmap, vmin=1, vmax=4.5, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=10,
                    color="white" if M[i, j] > 2.9 else TEXT)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([SHORT[c] for c in ORDER], fontsize=9, color=TEXT)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace(", built for 200 languages", ",\nbuilt for 200 languages") for n in names],
                       fontsize=9, color=TEXT)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Tokens needed compared with English, for each AI tokenizer (darker = more expensive)",
                 color=TEXT, fontsize=11, pad=40, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "token_tax_by_model.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    with open(RES) as f:
        d = json.load(f)
    main_chart(d)
    table_chart(d)
    print("charts in", CH)
