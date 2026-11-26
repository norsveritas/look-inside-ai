"""Step 3: turn the saved results into simple PNG charts in charts/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AQUA, BLUE, CHARTS, GREY, INK, INK2, ORANGE, RESULTS

YELLOW = "#f2a600"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": "#cccccc", "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eeeeee", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.titleweight": "bold", "axes.titlesize": 13,
    "legend.frameon": False,
})


def load(name):
    return json.load(open(os.path.join(RESULTS, name)))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS, name), dpi=150)
    plt.close(fig)
    print("saved", name)


def chart_layers():
    d = load("layer_scores.json")
    x = [r["layer"] for r in d]
    labels = [str(i) for i in range(12)] + ["out"]
    series = [
        ("english_heldout_d", "English: Kurdish topic vs other countries", BLUE),
        ("newsentences_d", "Same direction on brand-new sentences", AQUA),
        ("kurmanji_heldout_d", "Kurmanji text vs other foreign languages", ORANGE),
        ("cross_english_dir_on_kurmanji_d", "English-topic direction tested on Kurmanji text", YELLOW),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for key, lab, col in series:
        ax.plot(x, [r[key] for r in d], color=col, lw=2, marker="o", ms=5, label=lab)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Layer of GPT-2 (0 = first, 'out' = after the last layer)")
    ax.set_ylabel("Separation score (bigger = clearer)")
    ax.set_title("Where is 'Kurdish' clearest inside GPT-2?")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", fontsize=9.5)
    save(fig, "1_layers_where_kurdish_is_clearest.png")


def chart_neurons_per_layer():
    d = load("top_neurons.json")
    en, ku = d["per_layer_top200_english"], d["per_layer_top200_kurmanji"]
    x = np.arange(12)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.38
    ax.bar(x - w / 2 - 0.01, en, w, color=BLUE, label="Kurdish topic (English sentences)")
    ax.bar(x + w / 2 + 0.01, ku, w, color=ORANGE, label="Kurmanji language")
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("Layer")
    ax.set_ylabel("How many of the 200 most\nKurdish-reacting neurons")
    ax.set_title("Which layers hold the most Kurdish-reacting neurons?")
    ax.legend()
    save(fig, "2_neurons_per_layer.png")


def chart_top_neurons():
    d = load("top_neurons.json")["english_topic"][:10]
    names = [f"{n['neuron']}  ({', '.join(t['token'].strip() for t in n['top_tokens'][:3])})" for n in d][::-1]
    k = [n["mean_on_kurdish"] for n in d][::-1]
    o = [n["mean_on_other"] for n in d][::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    h = 0.38
    ax.barh(y + h / 2 + 0.01, k, h, color=BLUE, label="Sentences about Kurdistan")
    ax.barh(y - h / 2 - 0.01, o, h, color=GREY, label="Same sentences about other countries")
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlabel("Average strongest firing of the neuron")
    ax.set_title("Top 10 'Kurdish' neurons (and words that wake them up)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "3_top_kurdish_neurons.png")


def chart_scatter():
    pts = load("scatter_points.json")["6"]
    style = [
        ("english_kurdish", "English about Kurdistan", BLUE, "o"),
        ("english_other", "English about other countries", GREY, "o"),
        ("kurmanji", "Kurmanji sentences", ORANGE, "s"),
        ("other_foreign", "Other foreign languages", AQUA, "s"),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for key, lab, col, mk in style:
        ax.scatter(pts[key]["x"], pts[key]["y"], s=55, color=col, marker=mk, label=lab,
                   edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Kurdish-TOPIC direction  →")
    ax.set_ylabel("Kurmanji-LANGUAGE direction  →")
    ax.set_title("Every sentence as one dot (inside layer 6)")
    ax.legend(fontsize=9, loc="best")
    save(fig, "4_sentence_map_layer6.png")


def chart_steering():
    d = load("steering_scores.json")["english_direction"]
    layers = sorted({r["layer"] for r in d})
    cols = {2: BLUE, 6: ORANGE, 10: AQUA}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    panels = [("kurdish_word_rate", "Texts with a real Kurdish word (%)", 100),
              ("made_up_pesh_word_rate", "Texts with made-up 'Pesh...' words (%)", 100),
              ("war_word_rate", "Texts with a war/news word (%)", 100),
              ("nonsense", "Nonsense score\n(higher = less normal English)", 1)]
    for ax, (key, title, mult) in zip(axes, panels):
        for L in layers:
            rows = [r for r in d if r["layer"] == L]
            ax.plot([r["strength"] for r in rows], [r[key] * mult for r in rows], color=cols[L], lw=2,
                    marker="o", ms=6, label=f"pushed in at layer {L}")
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("Steering strength (0 = normal model)")
        ax.set_xticks([0, 4, 8, 12, 16])
    axes[0].set_ylim(0, 105)
    axes[1].set_ylim(0, 105)
    axes[2].set_ylim(0, 105)
    axes[0].legend(fontsize=9, loc="upper left")
    fig.suptitle("Before vs after steering: adding the 'Kurdish direction' to neutral prompts",
                 fontweight="bold", fontsize=13)
    save(fig, "5_steering_before_after.png")


def chart_removal():
    d = load("removal_results.json")["next_word"]
    conds = [("normal model", "Normal model", BLUE),
             ("control: random direction set to 'other countries' level", "Control: a random direction changed", GREY),
             ("naive: Kurdish direction set to zero", "Mistake: Kurdish direction set to zero", YELLOW),
             ("fixed: Kurdish direction set to 'other countries' level", "Kurdish direction set to 'other countries' level", ORANGE)]
    y = np.arange(len(d))[::-1]
    h = 0.2
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, (key, lab, col) in enumerate(conds):
        ax.barh(y + (1.5 - i) * (h + 0.02), [r[key]["p_kurdish_words"] * 100 for r in d], h, color=col, label=lab)
    ax.set_yticks(y, ['"' + r["prompt"] + ' ..."' for r in d], fontsize=9)
    ax.set_xlabel("Chance the next word is Kurdish / Kurds / Kurd / Kurdistan (%)")
    ax.set_title("Removing the Kurdish direction from Kurdish prompts")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "6_removal_before_after.png")


if __name__ == "__main__":
    chart_layers()
    chart_neurons_per_layer()
    chart_top_neurons()
    chart_scatter()
    if os.path.exists(os.path.join(RESULTS, "steering_scores.json")):
        chart_steering()
        chart_removal()
