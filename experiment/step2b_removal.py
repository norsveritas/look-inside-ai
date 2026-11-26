"""Step 2b: removal - take the Kurdish direction AWAY while GPT-2 reads Kurdish prompts.

Run:  python experiment/step2b_removal.py   (after step1)
Writes: results/removal_results.json, results/removal_samples.txt

Lesson learned while building this: simply setting the Kurdish direction to ZERO does not mean
"not Kurdish". Sentences about other countries sit BELOW zero on this direction in the later layers,
so zeroing it actually nudges the model TOWARDS Kurdish. The fair way is to set it to the level that
sentences about other countries normally have. We keep both versions so you can see the difference.
"""
import json
import os

import numpy as np
import torch

import sentences as S
from common import N_LAYERS, RESULTS, kurdish_token_ids, load_model, record
from step2_steering import EN, KURDISH_WORDS, SEEDS, generate, has_any

PROMPTS = [
    "Zakho and Duhok are cities in Iraqi",
    "The Peshmerga are the soldiers of the",
    "Erbil is the capital of the",
    "Kurmanji and Sorani are two dialects of the",
    "Newroz is the new year festival of the",
    "The people of Duhok mostly speak",
    "Masoud Barzani is a famous",
    "Ez ji Zaxoyê me. This sentence is written in",
]
GEN_PROMPTS = [
    "My trip to Kurdistan",
    "In Duhok, people like to",
    "The Kurdish language is",
    "Zakho is famous for",
]
GEN_SEEDS = SEEDS + [5, 6]


def set_level_hook(u, level):
    """Move every word's position along direction u to a fixed level (all other directions untouched)."""
    def hook(act, hook):
        act[:, 1:] += (level - act[:, 1:] @ u)[..., None] * u
        return act
    return hook


def subtract_hook(vec):
    def hook(act, hook):
        act[:, 1:] -= vec
        return act
    return hook


def main():
    model = load_model()
    ids, words = kurdish_token_ids(model)

    unit_en = [EN[L] / EN[L].norm() for L in range(N_LAYERS)]
    g = torch.Generator().manual_seed(123)
    unit_rand = []
    for L in range(N_LAYERS):
        r = torch.randn(768, generator=g)
        unit_rand.append(r / r.norm())

    # what level do normal "other country" sentences have on each direction?
    R_other, _, _ = record(model, [s for _, s in S.english_other()])
    lvl_en = [float((R_other[:, L] @ unit_en[L].numpy()).mean()) for L in range(N_LAYERS)]
    lvl_rand = [float((R_other[:, L] @ unit_rand[L].numpy()).mean()) for L in range(N_LAYERS)]

    def all_layers(units, levels):
        return [(f"blocks.{L}.hook_resid_pre", set_level_hook(units[L], levels[L])) for L in range(N_LAYERS)]

    conditions = {
        "normal model": [],
        "control: random direction set to 'other countries' level": all_layers(unit_rand, lvl_rand),
        "naive: Kurdish direction set to zero": all_layers(unit_en, [0.0] * N_LAYERS),
        "fixed: Kurdish direction set to 'other countries' level": all_layers(unit_en, lvl_en),
        "push away: subtract 4x Kurdish direction at layer 10": [("blocks.10.hook_resid_pre", subtract_hook(4 * EN[10]))],
    }

    results = []
    for p in PROMPTS:
        row = dict(prompt=p)
        for name, hooks in conditions.items():
            probs = model.run_with_hooks(model.to_tokens(p), fwd_hooks=hooks)[0, -1].softmax(-1)
            top = torch.topk(probs, 5)
            row[name] = dict(
                p_kurdish_words=float(probs[ids].sum()),
                top5=[(model.to_string(i.view(1)), round(float(v), 3)) for v, i in zip(top.values, top.indices)],
            )
        results.append(row)
        print(p)
        for name in conditions:
            print(f"   {name:60s} P(Kurdish)={row[name]['p_kurdish_words']:.3f}  top: {row[name]['top5'][:3]}")

    lines = ["REMOVAL SAMPLES  (K = continuation contains a real Kurdish-related word)\n"]
    gen_rate = {}
    for name, hooks in conditions.items():
        hits = n = 0
        lines.append(f"=== {name} ===")
        for p in GEN_PROMPTS:
            for s in GEN_SEEDS:
                text, _ = generate(model, p, hooks, s)
                k = has_any(text[len(p):], KURDISH_WORDS)
                hits += k
                n += 1
                lines.append(f"[seed {s}] {'K' if k else ' '} " + text.replace("\n", " "))
        gen_rate[name] = hits / n
        lines.append("")
        print(f"{name}: {hits}/{n} continuations mention Kurdish things")

    json.dump(dict(levels_other_countries=lvl_en, next_word=results, generation_kurdish_word_rate=gen_rate),
              open(os.path.join(RESULTS, "removal_results.json"), "w"), indent=2, ensure_ascii=False)
    open(os.path.join(RESULTS, "removal_samples.txt"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
