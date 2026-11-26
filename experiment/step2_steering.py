"""Step 2: steering - add the Kurdish direction while GPT-2 writes from neutral prompts.

Run:  python experiment/step2_steering.py   (after step1)
Writes: results/steering_samples.txt, results/steering_scores.json
"""
import json
import os
import re

import numpy as np
import torch

from common import N_LAYERS, RESULTS, kurdish_token_ids, load_model

DIRS = np.load(os.path.join(RESULTS, "directions.npz"))
EN = torch.tensor(DIRS["english"], dtype=torch.float32)   # (13, 768) Kurdish-topic direction per layer
KU = torch.tensor(DIRS["kurmanji"], dtype=torch.float32)  # (13, 768) Kurmanji-language direction

NEUTRAL_PROMPTS = [
    "My favourite place is",
    "Today I want to eat",
    "I went to the market and saw",
    "The best music in the world is",
    "My best friend is",
    "Yesterday the weather was",
]
STRENGTHS = [0, 4, 8, 12, 16]
STEER_LAYERS = [2, 6, 10]
SEEDS = [1, 2, 3, 4]
N_NEW = 30

# Real Kurdish-related words (places, people, groups). Made-up words starting with "Pesh"
# (like "Peshistan") are counted separately, because they are not real words.
KURDISH_WORDS = ["kurd", "erbil", "irbil", "arbil", "duhok", "dohuk", "zakho", "peshmerga", "sulayman",
                 "sulaim", "kirkuk", "kurmanji", "sorani", "newroz", "nowruz", "barzani", "mosul", "iraq",
                 "pkk", "ypg", "rojava", "kobane", "kobani", "yazidi", "yezidi"]
PESH = ["pesh"]
CONFLICT_WORDS = ["isis", "islamic state", "militant", "forces", "killed", "army", "troops", "attack",
                  "rebel", "fighters", "military", "war ", "wars", "liberation", "protection unit", "bomb", "jihad",
                  "soldier", "terror", "insurgent"]


def has_any(text, words):
    """True if any word appears at the START of a word (so 'war' does not match 'toward')."""
    t = text.lower()
    return any(re.search(r"\b" + re.escape(w), t) for w in words)


def make_add_hook(vec):
    def hook(act, hook):
        act[:, 1:] += vec  # every word except the start-of-text token
        return act
    return hook


def generate(model, prompt, hooks, seed, n_new=N_NEW, temperature=0.8, top_k=40):
    toks = model.to_tokens(prompt)
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_new):
        logits = model.run_with_hooks(toks, fwd_hooks=hooks)[0, -1] / temperature
        top = torch.topk(logits, top_k)
        nxt = top.indices[torch.multinomial(top.values.softmax(-1), 1, generator=g)]
        toks = torch.cat([toks, nxt.view(1, 1)], dim=1)
    return model.to_string(toks[0, 1:]), toks


def nonsense_score(model, toks, prompt_len):
    """How surprised the NORMAL (unsteered) model is by the text. Higher = more nonsense."""
    logits = model(toks)
    logp = logits[0, :-1].log_softmax(-1)
    tgt = toks[0, 1:]
    nll = -logp[torch.arange(len(tgt)), tgt][prompt_len - 1:]
    return float(nll.mean())


def steering(model):
    lines, scores = [], []
    for L in STEER_LAYERS:
        for c in STRENGTHS:
            hooks = [(f"blocks.{L}.hook_resid_pre", make_add_hook(EN[L] * c))] if c else []
            hits_k = hits_c = hits_p = 0
            nons = []
            for p in NEUTRAL_PROMPTS:
                plen = model.to_tokens(p).shape[1]
                for s in SEEDS:
                    text, toks = generate(model, p, hooks, s)
                    cont = text[len(p):]
                    k, w = has_any(cont, KURDISH_WORDS), has_any(cont, CONFLICT_WORDS)
                    hits_k += k
                    hits_c += w
                    hits_p += has_any(cont, PESH) and not has_any(cont, ["peshmerga"])
                    nons.append(nonsense_score(model, toks, plen))
                    lines.append(f"[layer {L:2d} | strength {c:2d} | seed {s}] {'K' if k else ' '}{'W' if w else ' '} "
                                 + text.replace("\n", " "))
            n = len(NEUTRAL_PROMPTS) * len(SEEDS)
            scores.append(dict(layer=L, strength=c, kurdish_word_rate=hits_k / n, war_word_rate=hits_c / n,
                               made_up_pesh_word_rate=hits_p / n,
                               nonsense=float(np.mean(nons))))
            print(scores[-1])
    header = ("STEERING SAMPLES\nK = continuation contains a real Kurdish-related word, W = contains a war/news word.\n"
              "Strength = how many times the Kurdish direction was added. Strength 0 = normal model.\n\n")
    open(os.path.join(RESULTS, "steering_samples.txt"), "w").write(header + "\n".join(lines))
    return scores


def kurmanji_steering(model):
    """Bonus: add the Kurmanji LANGUAGE direction instead. Does GPT-2 start writing Kurdish-looking text?"""
    lines, out = [], []
    L = 6
    for c in [0, 4, 8, 12, 16]:
        hooks = [(f"blocks.{L}.hook_resid_pre", make_add_hook(KU[L] * c))] if c else []
        letters = 0
        total = 0
        for p in NEUTRAL_PROMPTS[:3]:
            for s in SEEDS[:3]:
                text, _ = generate(model, p, hooks, s)
                cont = text[len(p):]
                letters += sum(ch in "êîûçşÊÎÛÇŞ" for ch in cont)
                total += len(cont)
                lines.append(f"[kurmanji direction | layer {L} | strength {c:2d} | seed {s}] " + text.replace("\n", " "))
        out.append(dict(strength=c, kurmanji_letters_per_100_chars=100 * letters / max(total, 1)))
        print(out[-1])
    return lines, out


def main():
    model = load_model()
    scores = steering(model)
    k_lines, k_scores = kurmanji_steering(model)
    with open(os.path.join(RESULTS, "steering_samples.txt"), "a") as f:
        f.write("\n\nBONUS: KURMANJI-LANGUAGE DIRECTION\n" + "\n".join(k_lines))
    json.dump(dict(english_direction=scores, kurmanji_direction=k_scores),
              open(os.path.join(RESULTS, "steering_scores.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
