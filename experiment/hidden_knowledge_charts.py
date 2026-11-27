"""Charts for experiment 5 (does the AI secretly know the Kurdish word?)."""
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "hidden_knowledge")
CH = os.path.join(HERE, "..", "charts", "hidden_knowledge")
os.makedirs(CH, exist_ok=True)

TEXT, TEXT2 = "#191919", "#666666"
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600"]
MODELS = [("gemma3-1b", "Gemma 3 1B"), ("qwen3-4b", "Qwen3 4B"), ("gemma3-4b", "Gemma 3 4B"), ("gemma3-12b", "Gemma 3 12B")]
FORMS = [("kmr", "Kurmanji\n(Latin letters)"), ("kmr_ar", "Badini\n(Arabic letters)"), ("ckb", "Sorani"),
         ("fa", "Persian\n(control)")]
KURDISH_LETTERS = {"kmr": set("êîûÊÎÛ"), "kmr_ar": set("ێۆڵڕەڤ"), "ckb": set("ێۆڵڕەڤ")}


def style(ax):
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c2c2c2")
    ax.tick_params(colors=TEXT2, length=0)
    ax.yaxis.grid(True, color="#e7e7e7", lw=0.8)
    ax.set_axisbelow(True)


def recognition_chart(d):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(FORMS))
    w = 0.2
    for i, (m, label) in enumerate(MODELS):
        vals = [d[m]["part_a"][f]["cal_top5"] * 100 for f, _ in FORMS]
        bars = ax.bar(x + (i - 1.5) * w, vals, w - 0.03, color=SERIES[i], label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=TEXT)
    chance = 5 / d["gemma3-4b"]["part_a"]["kmr"]["n_candidates"] * 100
    ax.axhline(chance, color=TEXT2, lw=1, ls="--", label="pure guessing")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in FORMS], color=TEXT)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of words where the right answer is in the top 5", color=TEXT2)
    ax.set_title("Multiple choice inside the model: the right Kurdish word is in the top 5 about half\n"
                 "the time for Gemma 3 4B and 82-94% for Gemma 3 12B, even though they rarely write it", loc="left",
                 color=TEXT, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "recognition.png"), dpi=150)
    plt.close(fig)


def steering_stats():
    st = defaultdict(lambda: [0, 0])
    for line in open(os.path.join(RES, "gemma3-4b_steer_samples.txt"), encoding="utf-8"):
        m = re.match(r"\[(\S+)\s+(\S+)\s+layer=(\S+) x(\S+)\].*wrote '(.*)'$", line.strip())
        if m:
            t, k, l, a, w = m.groups()
            s = st[(t, k, l, a)]
            s[0] += 1
            s[1] += any(c in KURDISH_LETTERS[t] for c in w)
    return {k: v[1] / v[0] for k, v in st.items()}


def steering_chart(d):
    style_share = steering_stats()
    res = d["gemma3-4b"]["part_b"]["kmr"]["results"]
    rows = [("no nudge", "none", ("kmr", "none", "None", "0")),
            ("Kurdish nudge, small", "kurdish:L23:x0.5", ("kmr", "kurdish", "23", "0.5")),
            ("Kurdish nudge, strong", "kurdish:L23:x1", ("kmr", "kurdish", "23", "1")),
            ("random nudge, strong", "random:L23:x1", ("kmr", "random", "23", "1"))]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    panels = [("uses Kurdish-only letters (ê, î, û)", [style_share[r[2]] * 100 for r in rows], SERIES[0]),
              ("writes the Turkish word", [res[r[1]]["neighbour"] / res[r[1]]["n"] * 100 for r in rows], SERIES[1]),
              ("writes the right Kurdish word", [res[r[1]]["correct"] / res[r[1]]["n"] * 100 for r in rows], SERIES[2])]
    for ax, (title, vals, c) in zip(axes, panels):
        ax.barh(y, vals, height=0.6, color=c)
        for yi, v in zip(y, vals):
            ax.text(v + 2, yi, f"{v:.0f}%", va="center", fontsize=9, color=TEXT)
        ax.set_xlim(0, 110)
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xticks([])
        for s in ["top", "right", "bottom"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#c2c2c2")
        ax.tick_params(length=0, colors=TEXT2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], color=TEXT)
    fig.suptitle("Nudging Gemma 3 4B toward Kurdish (Kurmanji, Latin letters, 43 test words): it starts to look\n"
                 "Kurdish and stops writing Turkish, but it does not find more right words", x=0.01, ha="left",
                 fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "steering.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data = {m: json.load(open(os.path.join(RES, f"{m}.json"), encoding="utf-8")) for m, _ in MODELS}
    recognition_chart(data)
    steering_chart(data)
    print("charts in", CH)
