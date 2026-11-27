"""Charts for experiment 7 (whose is it?)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "whose_is_it")
CH = os.path.join(HERE, "..", "charts", "whose_is_it")
os.makedirs(CH, exist_ok=True)

TEXT, TEXT2 = "#191919", "#666666"
SERIES = ["#257de0", "#f47230", "#00b894"]
MODELS = [("gemma3-1b", "Gemma 3 1B"), ("qwen3-4b", "Qwen3 4B"), ("gemma3-4b", "Gemma 3 4B")]
GROUPS = [("Kurdistan", "Turkey", "Kurdish cities\nin Turkey"), ("Kurdistan", "Syria", "Kurdish cities\nin Syria"),
          ("Kurdistan", "Iraq", "Kurdish cities\nin Iraq"), ("Kurdistan", "Iran", "Kurdish cities\nin Iran"),
          ("Quebec", None, "Quebec"), ("Catalonia", None, "Catalonia"), ("the Basque Country", None, "Basque\nCountry"),
          ("Tibet", None, "Tibet"), ("Scotland", None, "Scotland"), ("Wales", None, "Wales")]


def homeland_chart(data):
    fig, ax = plt.subplots(figsize=(11, 4.9))
    x = np.arange(len(GROUPS)) + np.array([0] * 4 + [0.6] * 6)  # a gap between Kurdish and other nations
    w = 0.26
    for i, (m, label) in enumerate(MODELS):
        h = data[m]["homeland"]
        vals = [100 * np.mean([r["nation_share"] for r in h if r["nation"] == n and (s is None or r["state"] == s)])
                for n, s, _ in GROUPS]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=SERIES[i], label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=7, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in GROUPS], color=TEXT, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel('% chance of naming the nation\n(rather than the state)', color=TEXT2)
    ax.set_title('"Diyarbakır is a city in ___": Kurdistan or Turkey?\n'
                 "The AI says Kurdistan for Iraq and Iran, where the name is official, almost never for Turkey and Syria",
                 loc="left", color=TEXT, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.36, 1.0))
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c2c2c2")
    ax.tick_params(colors=TEXT2, length=0)
    ax.yaxis.grid(True, color="#e7e7e7", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, "homeland.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    data = {m: json.load(open(os.path.join(RES, f"{m}.json"), encoding="utf-8")) for m, _ in MODELS}
    homeland_chart(data)
    print("charts in", CH)
