"""Experiment 4: which language does the model "think in" when it writes Kurdish?

Method ("logit lens", as in Wendler et al. 2024, "Do Llamas Work in English?"):
  The model is shown a few translations, e.g.  Français: "eau" - Kurmancî: "av"
  and must write the next answer. At every layer we take the model's half-finished
  notebook at the last position, push it through the model's own output step, and ask:
  how much probability goes to the right word in English, Turkish, Persian, Arabic,
  Chinese, French (the input), and in the target language itself?

Tasks:
  translate : French word -> target language
  repeat    : target-language word -> the same word again (no translation needed)
Targets: the three Kurdish forms, plus Turkish, Persian and Arabic as "normal" comparison languages.

Run:   python experiment/latent_language.py qwen2.5-0.5b      (or: all)
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

MODELS = {
    "qwen2.5-0.5b": dict(hf="Qwen/Qwen2.5-0.5B", label="Qwen2.5-0.5B (2024)"),
    "qwen2.5-1.5b": dict(hf="Qwen/Qwen2.5-1.5B", label="Qwen2.5-1.5B (2024)"),
    "qwen3-1.7b": dict(hf="Qwen/Qwen3-1.7B-Base", label="Qwen3-1.7B (2025)"),
    "qwen3-4b": dict(hf="Qwen/Qwen3-4B-Base", label="Qwen3-4B (2025)", dtype=torch.bfloat16),
}
TARGETS = ["kmr", "kmr_ar", "ckb", "tr", "fa", "ar"]
KURDISH = ["kmr", "kmr_ar", "ckb"]
OTHERS = ["en", "fr", "zh", "tr", "fa", "ar"]  # languages the model might "think in"
TASKS = ["translate", "repeat"]
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
        self.n_layers = self.m.config.num_hidden_layers
        self.norm = self.m.model.norm
        self.vocab = [self.tok.decode([i]) for i in range(len(self.tok))]

    def token_set(self, word):
        """Token ids that could be the start of `word` (right after an opening quote)."""
        forms = {word, word.capitalize()}
        ids = {self.tok.encode(f, add_special_tokens=False)[0] for f in forms}
        for i, s in enumerate(self.vocab):
            if s and "�" not in s and any(f.startswith(s) and len(s) >= min(2, len(f)) for f in forms):
                ids.add(i)
        return ids

    def layer_probs(self, prompt):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        hs = self.m(ids, output_hidden_states=True).hidden_states  # embeddings + one per layer
        # the last entry already has the final norm applied; the others need it
        h = torch.stack([self.norm(x[0, -1]) for x in hs[1:-1]] + [hs[-1][0, -1]])
        return torch.softmax(self.m.lm_head(h).float(), -1)  # (layers, vocab)

    def generate(self, prompt, n=6):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        out = self.m.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:]).split('"')[0]


def make_prompt(task, target, test, examples):
    src = "fr" if task == "translate" else target
    lines = [f'{LANG_NAMES[src]}: "{e[src]}" - {LANG_NAMES[target]}: "{e[target]}"' for e in examples]
    lines.append(f'{LANG_NAMES[src]}: "{test[src]}" - {LANG_NAMES[target]}: "')
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
            measured = [l for l in OTHERS if l != target] + [target]
            kept, skipped = [], []
            curves = {l: [] for l in measured + ["kurdish_other"]}
            correct, out_lang = 0, {}
            for r in TESTS:
                S = {l: sets[(r["en"], l)] for l in measured}
                clash = [(a, b) for i, a in enumerate(measured) for b in measured[i + 1:] if S[a] & S[b]]
                if clash:
                    skipped.append((r["en"], clash))
                    continue
                other_k = set().union(*[sets[(r["en"], k)] for k in KURDISH if k != target]) - set().union(*S.values())
                kept.append(r["en"])
                for es in example_sets:
                    p = lens.layer_probs(make_prompt(task, target, r, es))
                    for l in measured:
                        curves[l].append(p[:, sorted(S[l])].sum(-1).numpy())
                    curves["kurdish_other"].append(p[:, sorted(other_k)].sum(-1).numpy() if other_k
                                                   else np.zeros(lens.n_layers))
                # what does the model actually write? (first example set only)
                prompt = make_prompt(task, target, r, example_sets[0])
                said = lens.generate(prompt).strip()
                ok = said.lower() == r[target].lower()
                correct += ok
                match = [c for c in COLUMNS if said and said.lower() == r[c].lower()]
                lang = "+".join(match) if match else "other"
                out_lang[lang] = out_lang.get(lang, 0) + 1
                samples.append(f"[{task} -> {target}] {r['en']:<9} wanted {r[target]!r:<14} wrote {said!r:<16} ({lang})")
            result["tasks"][f"{task}:{target}"] = dict(
                n_words=len(kept), words=kept, skipped=[s[0] for s in skipped],
                mean={l: np.mean(v, 0).tolist() for l, v in curves.items() if v},
                correct=correct, output_language=out_lang)
            m = {l: np.mean(v, 0) for l, v in curves.items() if v}
            mid = lens.n_layers // 2
            print(f"{task:>9} -> {target:<6} words={len(kept):2d} correct={correct:2d}  "
                  + "  ".join(f"{l}:{m[l][mid - 2:mid + 3].mean():.2f}/{m[l][-1]:.2f}" for l in m), flush=True)

    with open(os.path.join(OUT, f"{key}.json"), "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    with open(os.path.join(OUT, f"{key}_samples.txt"), "w") as f:
        f.write("\n".join(samples))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    keys = list(MODELS) if sys.argv[1] == "all" else sys.argv[1:]
    for k in keys:
        run(k)
