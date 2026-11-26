"""Charts for experiment 3 (old vs modern models). Reads results/modern_models/*.json."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AQUA, BLUE, CHARTS, GREY, ORANGE, RESULTS
import step3_make_charts  # noqa: F401  (shared chart style)
from modern_models import MODELS

YELLOW, MAGENTA = "#f2a600", "#ee7fac"
SRC = os.path.join(RESULTS, "modern_models")
OUT = os.path.join(CHARTS, "modern_models")
os.makedirs(OUT, exist_ok=True)
CALM = ["Basque", "Scottish", "Welsh", "Japanese", "Dutch"]

DATA = [json.load(open(os.path.join(SRC, f"{k}.json"))) for k in MODELS if os.path.exists(os.path.join(SRC, f"{k}.json"))]
LABELS = [d["label"].replace(" (", "\n(") for d in DATA]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("saved", name)


def rank_of_kurdish(scores):
    order = sorted(scores, key=lambda g: -scores[g])
    return order.index("Kurdish") + 1


def three_bars(get, title, xlabel, name):
    y = np.arange(len(DATA))[::-1]
    h = 0.26
    fig, ax = plt.subplots(figsize=(10, 1.25 * len(DATA) + 1.6))
    ku = [get(d)["Kurdish"] * 100 for d in DATA]
    hi = [np.mean([get(d)[g] for g in ("Palestinian", "Chechen")]) * 100 for d in DATA]
    calm = [np.mean([get(d)[g] for g in CALM]) * 100 for d in DATA]
    ax.barh(y + h + 0.02, ku, h, color=ORANGE, label="Kurdish")
    ax.barh(y, hi, h, color=YELLOW, label="Palestinian & Chechen (average)")
    ax.barh(y - h - 0.02, calm, h, color=BLUE, label="Basque, Scottish, Welsh, Japanese, Dutch (average)")
    for yi, d, k in zip(y, DATA, ku):
        ax.text(k + 1, yi + h + 0.02, f"{k:.0f}%  (Kurdish rank #{rank_of_kurdish(get(d))} of 10)",
                va="center", fontsize=8.5, color="#333")
    ax.set_yticks(y, LABELS, fontsize=9.5)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, name)


def chart_inside():
    cols = [GREY, BLUE, ORANGE, YELLOW, AQUA, MAGENTA]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for d, c in zip(DATA, cols):
        rel = np.array(d["inside_relative_war_level"])
        ki = ["Kurdish" in g for g in ["Kurdish", "Palestinian", "Chechen", "Armenian", "Tibetan", "Basque",
                                          "Scottish", "Welsh", "Japanese", "Dutch"]].index(True)
        x = np.linspace(0, 1, rel.shape[1])
        ax.plot(x, rel[ki], color=c, lw=2, marker="o", ms=3.5, label=d["label"])
    ax.axhline(0, color="#999", lw=1)
    ax.set_xlabel("Position inside the model (0 = first layer, 1 = last layer)")
    ax.set_ylabel("How 'war-like' the word Kurdish is,\ncompared with the average group")
    ax.set_title("Inside each model: does 'Kurdish' lean towards war?")
    ax.legend(fontsize=8.5)
    save(fig, "3_inside_kurdish_war_level.png")


def chart_fix():
    y = np.arange(len(DATA))[::-1]
    h = 0.26
    fig, ax = plt.subplots(figsize=(10, 1.25 * len(DATA) + 1.6))
    vals = [(d["next_word"]["Kurdish"]["war_share"], d["next_word_control"]["Kurdish"]["war_share"],
             d["next_word_fixed"]["Kurdish"]["war_share"]) for d in DATA]
    ax.barh(y + h + 0.02, [v[0] * 100 for v in vals], h, color=ORANGE, label="Normal")
    ax.barh(y, [v[1] * 100 for v in vals], h, color=GREY, label="Control (random direction)")
    ax.barh(y - h - 0.02, [v[2] * 100 for v in vals], h, color=BLUE, label="Fixed (war direction held at peaceful level)")
    ax.set_yticks(y, LABELS, fontsize=9.5)
    ax.set_xlabel("War share of the model's guesses about 'Kurdish' (%)")
    ax.set_title("Does the fix from experiment 2 work on every model?")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, "4_fix_on_every_model.png")


if __name__ == "__main__":
    three_bars(lambda d: {g: v["war_share"] for g, v in d["next_word"].items()},
               "Word guesses: how much goes to WAR words?",
               "Of the model's war + culture guesses, the share that is war words (%)", "1_word_guesses_by_model.png")
    three_bars(lambda d: {g: v["war_rate"] for g, v in d["generation"].items()},
               "Writing: texts that contain war words",
               "Texts containing a war word (% of 48 per group)", "2_writing_by_model.png")
    chart_inside()
    chart_fix()
