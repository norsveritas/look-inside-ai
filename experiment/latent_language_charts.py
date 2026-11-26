"""Charts for experiment 4 (which language does the model think in?)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from latent_words import LANG_LABELS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "latent_language")
CH = os.path.join(HERE, "..", "charts", "latent_language")
os.makedirs(CH, exist_ok=True)

COLORS = {"en": "#2a6fdb", "fr": "#9a9a9a", "zh": "#d62728", "tr": "#e67e22", "fa": "#2ca02c", "ar": "#8c564b",
          "target": "#111111", "kurdish_other": "#b39ddb"}
TARGET_TITLES = {"kmr": "Kurmanji (Latin letters)", "kmr_ar": "Badini (Arabic letters)", "ckb": "Sorani",
                 "tr": "Turkish (comparison)", "fa": "Persian (comparison)", "ar": "Arabic (comparison)"}


def load(key):
    with open(os.path.join(RES, f"{key}.json")) as f:
        return json.load(f)


def curves_chart(d, task, fname):
    targets = list(TARGET_TITLES)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    for ax, t in zip(axes.flat, targets):
        r = d["tasks"][f"{task}:{t}"]
        x = np.linspace(0, 1, d["n_layers"])
        for l, v in r["mean"].items():
            name = "target" if l == t else l
            label = f"{LANG_LABELS[t]} (the answer)" if l == t else (
                "other Kurdish forms" if l == "kurdish_other" else LANG_LABELS[l])
            ax.plot(x, v, color=COLORS[name], lw=3 if l == t else 1.8,
                    ls="--" if l == "kurdish_other" else "-", label=label)
        ax.set_title(f"{TARGET_TITLES[t]}\n({r['n_words']} words, model correct on {r['correct']})", fontsize=11)
        ax.set_xlabel("layer (0 = first, 1 = last)")
        ax.grid(alpha=.3)
        ax.legend(fontsize=7.5, loc="upper left")
    for ax in axes[:, 0]:
        ax.set_ylabel("probability of the word\nin each language")
    what = "French word → target language" if task == "translate" else "repeat a word in the target language"
    fig.suptitle(f"{d['label']}: which language is the hidden guess in?  Task: {what}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(CH, fname), dpi=130)
    plt.close(fig)


def middle_summary(models, task, fname):
    """Bar chart: average probability in the middle of the model (layers 40-80%), per target and language."""
    langs = ["en", "zh", "tr", "fa", "ar", "fr"]
    targets = list(TARGET_TITLES)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True, squeeze=False)
    for ax, key in zip(axes[0], models):
        d = load(key)
        n = d["n_layers"]
        lo, hi = int(.4 * n), int(.8 * n)
        w = 0.12
        for j, l in enumerate(langs + ["target"]):
            vals = []
            for t in targets:
                m = d["tasks"][f"{task}:{t}"]["mean"]
                k = t if l == "target" else l
                vals.append(np.mean(m[k][lo:hi]) if k in m and (l == "target" or l != t) else 0)
            ax.bar(np.arange(len(targets)) + (j - 3) * w, vals, w, color=COLORS[l],
                   label="the answer language itself" if l == "target" else LANG_LABELS[l])
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels([TARGET_TITLES[t].replace(" (", "\n(") for t in targets], fontsize=8)
        ax.set_title(d["label"])
        ax.grid(axis="y", alpha=.3)
    axes[0][0].set_ylabel("average probability, middle layers (40–80%)")
    axes[0][-1].legend(fontsize=8)
    fig.suptitle(f"What language is the model's hidden guess in, in the middle of the model? ({task})")
    fig.tight_layout()
    fig.savefig(os.path.join(CH, fname), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    models = sys.argv[1:] or sorted(f[:-5] for f in os.listdir(RES) if f.endswith(".json"))
    for k in models:
        d = load(k)
        for task in ["translate", "repeat"]:
            curves_chart(d, task, f"{k}_{task}.png")
    for task in ["translate", "repeat"]:
        middle_summary(models, task, f"summary_middle_layers_{task}.png")
    print("charts in", CH)
