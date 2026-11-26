"""Experiment 4: which language does the model "think in" when it writes Kurdish?

Method ("logit lens", as in Wendler et al. 2024, "Do Llamas Work in English?"):
  The model is shown a few translations, e.g.  Français: "eau" - Kurmancî: "av"
  and must write the next answer. At every layer we take the model's half-finished
  notebook at the last position, push it through the model's own output step, and ask:
  how much probability goes to the right word in English, Turkish, Persian, Arabic,
  Chinese, French (the input), and in the target language itself?

Tasks (X = one of the three Kurdish forms, or Turkish / Persian / Arabic as "normal" comparison languages):
  write : French word -> X        e.g.  Français: "eau" - Kurmancî: "
  read  : X word -> French        e.g.  Kurmancî: "av" - Français: "

Run:   python experiment/latent_language.py gemma3-4b      (or: all)
Writes: results/latent_language/<model>.json and <model>_samples.txt
"""
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_words import COLUMNS, LANG_NAMES, N_EXAMPLES, WORDS

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 4)

MODELS = {  # Gemma 3 is loaded from an exact, ungated copy of Google's weights (no sign-up needed)
    "gemma3-4b": dict(hf="unsloth/gemma-3-4b-pt", label="Gemma 3 4B (Google, 2025)", dtype=torch.bfloat16),
    "qwen3-4b": dict(hf="Qwen/Qwen3-4B-Base", label="Qwen3 4B (Alibaba, 2025)", dtype=torch.bfloat16),
    "gemma3-1b": dict(hf="unsloth/gemma-3-1b-pt", label="Gemma 3 1B (Google, 2025)"),
}
TARGETS = ["kmr", "kmr_ar", "ckb", "tr", "fa", "ar"]
KURDISH = ["kmr", "kmr_ar", "ckb"]
OTHERS = ["en", "fr", "zh", "tr", "fa", "ar"]  # languages the model might "think in"
TASKS = ["write", "read"]
N_PROMPT_SETS = 3  # each test word is asked with 3 different sets of examples
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "latent_language")
os.makedirs(OUT, exist_ok=True)

ROWS = [dict(zip(COLUMNS, r)) for r in WORDS]
EXAMPLES, TESTS = ROWS[:N_EXAMPLES * 2], ROWS[N_EXAMPLES * 2:]


class Lens:
    def __init__(self, hf, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.m = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype).eval()
        cfg = getattr(self.m.config, "text_config", self.m.config)  # Gemma 3 keeps text settings apart
        self.n_layers = cfg.num_hidden_layers
        self.softcap = getattr(cfg, "final_logit_softcapping", None)
        self.norm = self.m.get_decoder().norm
        self.head = self.m.get_output_embeddings()
        self.by_text = {}
        for i in range(len(self.tok)):
            s = self.tok.decode([i])
            if s and "�" not in s:
                self.by_text.setdefault(s, set()).add(i)

    def token_set(self, word):
        """Token ids that could be the start of `word` (right after an opening quote):
        its real first token, plus every vocabulary piece that spells the word's first 2+ letters."""
        forms = {word, word.capitalize()}
        ids = {self.tok.encode(f, add_special_tokens=False)[0] for f in forms}
        for f in forms:
            for n in range(min(2, len(f)), len(f) + 1):
                ids |= self.by_text.get(f[:n], set())
        return ids

    def layer_probs(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        hs = self.m(ids, output_hidden_states=True).hidden_states  # embeddings + one per layer
        # the last entry already has the final norm applied; the others need it
        h = torch.stack([self.norm(x[0, -1]) for x in hs[1:-1]] + [hs[-1][0, -1]])
        logits = self.head(h).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return torch.softmax(logits, -1)  # (layers, vocab)

    def generate(self, prompt, n=6):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        out = self.m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:]).split('"')[0]


def make_prompt(src, dst, test, examples):
    lines = [f'{LANG_NAMES[src]}: "{e[src]}" - {LANG_NAMES[dst]}: "{e[dst]}"' for e in examples]
    lines.append(f'{LANG_NAMES[src]}: "{test[src]}" - {LANG_NAMES[dst]}: "')
    return "\n".join(lines)


def run(key):
    cfg = MODELS[key]
    t0 = time.time()
    print(f"\n=== {cfg['label']} ===", flush=True)
    lens = Lens(cfg["hf"], cfg.get("dtype", torch.float32))
    sets = {(r["en"], c): lens.token_set(r[c]) for r in ROWS for c in COLUMNS}
    rng = random.Random(0)
    example_sets = [rng.sample(EXAMPLES, N_EXAMPLES) for _ in range(N_PROMPT_SETS)]

    result = dict(model=key, label=cfg["label"], n_layers=lens.n_layers, tasks={})
    samples = []
    for task in TASKS:
        for target in TARGETS:
            src, dst = ("fr", target) if task == "write" else (target, "fr")
            measured = [l for l in OTHERS if l != target] + [target]  # same 7 languages in both tasks
            kept, skipped = [], []
            curves = {l: [] for l in measured + ["kurdish_other"]}
            correct, out_lang, per_word = 0, {}, {}
            for r in TESTS:
                S = {l: sets[(r["en"], l)] for l in measured}
                # a word only counts if no two languages share a starting piece (e.g. Persian and Sorani "baran")
                clash = [(a, b) for i, a in enumerate(measured) for b in measured[i + 1:] if S[a] & S[b]]
                if clash:
                    skipped.append(r["en"])
                    continue
                other_k = set().union(*[sets[(r["en"], k)] for k in KURDISH if k != target]) - set().union(*S.values())
                kept.append(r["en"])
                word_curves = {l: [] for l in measured}
                for es in example_sets:
                    p = lens.layer_probs(make_prompt(src, dst, r, es))
                    for l in measured:
                        curves[l].append(p[:, sorted(S[l])].sum(-1).numpy())
                        word_curves[l].append(curves[l][-1])
                    curves["kurdish_other"].append(p[:, sorted(other_k)].sum(-1).numpy() if other_k
                                                   else np.zeros(lens.n_layers))
                # what does the model actually write? (first example set only)
                said = lens.generate(make_prompt(src, dst, r, example_sets[0])).strip()
                correct += said.lower() == r[dst].lower()
                match = [c for c in COLUMNS if said and said.lower() == r[c].lower()]
                lang = "+".join(match) if match else "other"
                out_lang[lang] = out_lang.get(lang, 0) + 1
                per_word[r["en"]] = dict(said=said, lang=lang, ok=said.lower() == r[dst].lower(),
                                         curves={l: np.round(np.mean(v, 0), 4).tolist() for l, v in word_curves.items()})
                samples.append(f"[{task:<5} {src:>6} -> {dst:<6}] {r['en']:<9} wanted {r[dst]!r:<14} "
                               f"wrote {said!r:<16} ({lang})")
            m = {l: np.mean(v, 0) for l, v in curves.items() if v}
            result["tasks"][f"{task}:{target}"] = dict(
                src=src, dst=dst, n_words=len(kept), words=kept, skipped=skipped,
                mean={l: v.tolist() for l, v in m.items()}, correct=correct, output_language=out_lang,
                per_word=per_word)
            lo, hi = int(.4 * lens.n_layers), int(.8 * lens.n_layers)
            print(f"{task:>5} {src:>6}->{dst:<6} words={len(kept):2d} correct={correct:2d}  mid/last: "
                  + "  ".join(f"{l}:{m[l][lo:hi].mean():.2f}/{m[l][-1]:.2f}" for l in m), flush=True)

    with open(os.path.join(OUT, f"{key}.json"), "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, f"{key}_samples.txt"), "w") as f:
        f.write("\n".join(samples))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    keys = list(MODELS) if sys.argv[1] == "all" else sys.argv[1:]
    for k in keys:
        run(k)
