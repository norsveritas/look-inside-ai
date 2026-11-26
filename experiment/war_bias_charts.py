"""Charts for experiment 2 (war bias). Reads results/war_bias/war_bias_results.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import BLUE, CHARTS, GREY, ORANGE, RESULTS
import step3_make_charts  # noqa: F401  (applies the shared chart style)

OUT = os.path.join(CHARTS, "war_bias")
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(RESULTS, "war_bias", "war_bias_results.json")))
GROUPS = D["war_direction"]["groups"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("saved", name)


def paired_bars(before, after, xlabel, title, name, after_label):
    order = sorted(GROUPS, key=lambda g: before[g])
    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(y + h / 2 + 0.01, [before[g] * 100 for g in order], h, color=BLUE, label="Normal model")
    ax.barh(y - h / 2 - 0.01, [after[g] * 100 for g in order], h, color=ORANGE, label=after_label)
    for yi, g in zip(y, order):
        ax.text(before[g] * 100 + 1, yi + h / 2 + 0.01, f"{before[g] * 100:.0f}%", va="center", fontsize=8.5, color="#444")
    ax.set_yticks(y, order)
    ax.get_yticklabels()[order.index("Kurdish")].set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    save(fig, name)


def chart_next_word():
    before = {g: D["next_word"][g]["war_share"] for g in GROUPS}
    after = {g: D["fix2_next_word"][g]["war_share"] for g in GROUPS}
    paired_bars(before, after,
                "Of the model's war + culture guesses, how much goes to WAR words (%)",
                "'The ___ people are known for their...': war or culture?",
                "1_next_word_war_share.png", "After the fix (war direction held at 'peaceful' level)")


def chart_generation():
    before = {g: D["generation"][g]["war_rate"] for g in GROUPS}
    after = {g: D["fix2_generation"][g]["war_rate"] for g in GROUPS}
    paired_bars(before, after, "Texts the model wrote that contain a war word (%)",
                "What GPT-2 writes about each group (48 texts each)",
                "2_generated_text_war_rate.png", "After the fix (war direction held at 'peaceful' level)")


def chart_heatmap():
    rel = np.array(D["war_direction"]["group_relative_war_level"])
    order = np.argsort(-rel[:, :4].mean(1))
    rel = rel[order]
    names = [GROUPS[i] for i in order]
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f2f2f2", ORANGE])
    lim = np.abs(rel).max()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    im = ax.imshow(rel, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            ax.text(j, i, f"{rel[i, j]:+.1f}", ha="center", va="center", fontsize=8, color="#222")
    ax.set_xticks(range(12), [str(i) for i in range(12)])
    ax.set_yticks(range(len(names)), names)
    ax.get_yticklabels()[names.index("Kurdish")].set_fontweight("bold")
    ax.set_xlabel("Layer of GPT-2")
    ax.set_title("Inside the model: how 'war-like' is each group's name?")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("← more peaceful than average      more war-like than average →", fontsize=9)
    save(fig, "3_inside_war_direction_heatmap.png")


def chart_fix1():
    rows = D["fix1_clean_steering"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, L in zip(axes, (2, 6)):
        for kind, col in (("original", ORANGE), ("war part removed", BLUE)):
            r = [x for x in rows if x["layer"] == L and x["kind"] == kind]
            s = [x["strength"] for x in r]
            ax.plot(s, [x["war_rate"] * 100 for x in r], color=col, lw=2, marker="o", ms=6,
                    label=f"{kind}: texts with war words")
            ax.plot(s, [x["kurdish_word_rate"] * 100 for x in r], color=col, lw=2, ls="--", marker="s", ms=6,
                    label=f"{kind}: texts with Kurdish words")
        ax.set_title(f"Pushing 'Kurdish' in at layer {L}", fontsize=12)
        ax.set_xlabel("Steering strength")
        ax.set_xticks([4, 8, 12])
        ax.set_ylim(0, 60)
    axes[0].set_ylabel("% of 48 texts")
    axes[1].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Fix 1: steering with the Kurdish direction, before and after cleaning out its war part",
                 fontweight="bold", fontsize=12.5)
    save(fig, "4_fix1_clean_steering.png")


if __name__ == "__main__":
    chart_next_word()
    chart_generation()
    chart_heatmap()
    chart_fix1()
