"""Experiment 5: when the AI fails to write a Kurdish word, does it secretly know it?

Uses the same words and prompts as experiment 4 (WRITE task: French word -> Kurdish word).

Part A  "multiple choice inside the model"
    For every test word we do not let the model write. Instead we measure how likely the model finds
    each candidate answer: all the Kurdish words of the list, and all the neighbour-language words
    (Turkish for Kurmanji in Latin letters, Persian for Badini and Sorani in Arabic letters).
    Short and common words always look likely, so each score is compared with the same word's
    average score after the *other* French words ("calibration"). If the right Kurdish word comes out
    on top, the model recognises it, even if it cannot produce it.

Part B  "nudge toward Kurdish"   (Gemma 3 4B only)
    We record the model's hidden state when it is asked for a Kurdish word and when it is asked for
    the neighbour-language word of the same French word. The average difference is a "Kurdish, not
    Persian" direction. We add it to the hidden state while the model writes, and count how often the
    right Kurdish word comes out. The direction is learned on half of the words and tested on the
    other half, so the model can never be "told" the answer.

Run:   python hidden_knowledge.py gemma3-4b qwen3-4b gemma3-1b      (Part B runs for gemma3-4b)
Writes: results/hidden_knowledge/<model>.json and <model>_steer_samples.txt
"""
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

from latent_language import (EXAMPLES, MODELS, N_EXAMPLES, ROWS, Lens, first, make_prompt, matches)
from latent_words import COLUMNS

NEIGHBOUR = {"kmr": "tr", "kmr_ar": "fa", "ckb": "fa", "fa": "ar"}  # "fa" -> "ar" is a control
HERE = os.path.dirname(os.path.abspath(__file__))
PREV = os.path.join(HERE, "..", "results", "latent_language")
OUT = os.path.join(HERE, "..", "results", "hidden_knowledge")
os.makedirs(OUT, exist_ok=True)
BY_EN = {r["en"]: r for r in ROWS}
STEER_LAYERS = [0.5, 0.65, 0.8]  # as a fraction of the model's depth
STEER_SCALES = [1, 2, 4]


def score_candidates(lens, prefix, cands):
    """log-probability of each candidate word (followed by the closing quote) after the prompt."""
    tok, m = lens.tok, lens.m
    pre = tok(prefix, return_tensors="pt").input_ids
    o = m(pre, use_cache=True)
    last = torch.log_softmax(o.logits[0, -1].float(), -1)
    enc = [tok.encode(c + '"', add_special_tokens=False) for c in cands]
    n, B = max(map(len, enc)), len(enc)
    ids = torch.full((B, n), tok.pad_token_id)
    mask = torch.zeros(B, n, dtype=torch.long)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e)
        mask[i, :len(e)] = 1
    cache = copy.deepcopy(o.past_key_values)
    cache.batch_repeat_interleave(B)
    am = torch.cat([torch.ones(B, pre.shape[1], dtype=torch.long), mask], 1)
    lg = torch.log_softmax(m(ids, past_key_values=cache, attention_mask=am).logits.float(), -1)
    return [last[e[0]].item() + sum(lg[i, j - 1, e[j]].item() for j in range(1, len(e))) for i, e in enumerate(enc)]


def part_a(lens, prev, es):
    out = {}
    for t, nb in NEIGHBOUR.items():
        words = prev["tasks"][f"write:{t}"]["words"]
        wrote = {w: prev["tasks"][f"write:{t}"]["per_word"][w]["ok"] for w in words}
        n = len(words)
        cands = [first(BY_EN[w][t]) for w in words] + [first(BY_EN[w][nb]) for w in words]
        S = np.array([score_candidates(lens, make_prompt("fr", t, BY_EN[w], es), cands) for w in words])
        others_mean = (S.sum(0, keepdims=True) - S) / (n - 1)  # each candidate's mean score after the OTHER words
        C = S - others_mean
        per = {}
        for i, w in enumerate(words):
            per[w] = dict(
                wrote_it=bool(wrote[w]),
                raw_rank_kurdish=int((S[i, :n] > S[i, i]).sum()) + 1,
                cal_rank_all=int((C[i] > C[i, i]).sum()) + 1,
                beats_neighbour_raw=bool(S[i, i] > S[i, n + i]),
                beats_neighbour_cal=bool(C[i, i] > C[i, n + i]))
        top1 = np.mean([p["cal_rank_all"] == 1 for p in per.values()])
        top5 = np.mean([p["cal_rank_all"] <= 5 for p in per.values()])
        failed = [p for p in per.values() if not p["wrote_it"]]
        out[t] = dict(neighbour=nb, n_words=n, n_candidates=2 * n, chance_top1=1 / (2 * n),
                      cal_top1=top1, cal_top5=top5,
                      cal_top1_when_failed=np.mean([p["cal_rank_all"] == 1 for p in failed]) if failed else None,
                      n_failed=len(failed),
                      raw_top1_kurdish=np.mean([p["raw_rank_kurdish"] == 1 for p in per.values()]),
                      beats_neighbour_raw=np.mean([p["beats_neighbour_raw"] for p in per.values()]),
                      beats_neighbour_cal=np.mean([p["beats_neighbour_cal"] for p in per.values()]),
                      per_word=per, candidates=cands, words=words)
        print(f"  A {t:<6} vs {nb}: n={n} top1={top1:.0%} (chance {1/(2*n):.1%}) top5={top5:.0%} "
              f"top1|failed={out[t]['cal_top1_when_failed']} beats-neighbour raw/cal="
              f"{out[t]['beats_neighbour_raw']:.0%}/{out[t]['beats_neighbour_cal']:.0%}", flush=True)
    return out


class Steer:
    def __init__(self, layer):
        self.vec, self.h = None, layer.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        if self.vec is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.vec.to(out[0].dtype),) + out[1:]
        return out + self.vec.to(out.dtype)


def part_b(lens, prev, es, samples):
    layers = lens.m.get_decoder().layers
    L_ids = sorted({int(f * lens.n_layers) for f in STEER_LAYERS})
    hooks = {l: Steer(layers[l]) for l in L_ids}
    out = {}
    for t in ["kmr", "kmr_ar", "ckb"]:
        nb = NEIGHBOUR[t]
        words = prev["tasks"][f"write:{t}"]["words"]
        # hidden state at the last position, after each steering layer, for "-> Kurdish" and "-> neighbour" prompts
        diff = {}
        for w in words:
            hk = lens.m(lens.tok(make_prompt("fr", t, BY_EN[w], es), return_tensors="pt").input_ids,
                        output_hidden_states=True).hidden_states
            hn = lens.m(lens.tok(make_prompt("fr", nb, BY_EN[w], es), return_tensors="pt").input_ids,
                        output_hidden_states=True).hidden_states
            diff[w] = {l: (hk[l + 1][0, -1] - hn[l + 1][0, -1]).float() for l in L_ids}
        rng = random.Random(1)
        shuffled = words[:]
        rng.shuffle(shuffled)
        folds = [shuffled[::2], shuffled[1::2]]
        configs = [("none", None, 0)] + [("kurdish", l, a) for l in L_ids for a in STEER_SCALES] + \
                  [("random", l, 2) for l in L_ids]
        res = {}
        for kind, l, a in configs:
            stats = dict(correct=0, neighbour=0, french=0, n=0)
            for k in range(2):
                train, test = folds[1 - k], folds[k]
                if kind != "none":
                    v = torch.stack([diff[w][l] for w in train]).mean(0)
                    if kind == "random":
                        g = torch.Generator().manual_seed(l)
                        r = torch.randn(v.shape, generator=g)
                        v = r / r.norm() * v.norm()
                    hooks[l].vec = a * v
                for w in test:
                    r = BY_EN[w]
                    said = lens.generate(make_prompt("fr", t, r, es)).strip()
                    stats["n"] += 1
                    stats["correct"] += matches(said, r[t])
                    stats["neighbour"] += matches(said, r[nb])
                    stats["french"] += matches(said, r["fr"])
                    samples.append(f"[{t:<6} {kind:<7} layer={l} x{a}] {w:<9} wanted {r[t]!r:<14} wrote {said!r}")
                for h in hooks.values():
                    h.vec = None
            name = "none" if kind == "none" else f"{kind}:L{l}:x{a}"
            res[name] = stats
            print(f"  B {t:<6} {name:<16} correct={stats['correct']:2d}/{stats['n']} "
                  f"neighbour={stats['neighbour']:2d} french={stats['french']:2d}", flush=True)
        out[t] = dict(neighbour=nb, layers=L_ids, results=res)
    for h in hooks.values():
        h.h.remove()
    return out


def run(key):
    cfg = MODELS[key]
    t0 = time.time()
    print(f"\n=== {cfg['label']} ===", flush=True)
    lens = Lens(cfg["hf"], cfg.get("dtype", torch.float32))
    with open(os.path.join(PREV, f"{key}.json")) as f:
        prev = json.load(f)
    es = random.Random(0).sample(EXAMPLES, N_EXAMPLES)  # the same first example set as experiment 4
    result = dict(model=key, label=cfg["label"], n_layers=lens.n_layers, part_a=part_a(lens, prev, es))
    samples = []
    if key == "gemma3-4b":
        result["part_b"] = part_b(lens, prev, es, samples)
        with open(os.path.join(OUT, f"{key}_steer_samples.txt"), "w") as f:
            f.write("\n".join(samples))
    with open(os.path.join(OUT, f"{key}.json"), "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False, default=float)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        run(k)
