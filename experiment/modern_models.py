"""Experiment 3: is the "Kurdish = war" bias still there in modern small models?

Repeats the experiment-2 war-bias tests on several open models using plain Hugging Face `transformers`
(TransformerLens doesn't support all of these models), so every model, GPT-2 included, is measured the same way.

Run one model:   python experiment/modern_models.py gpt2
Run all:         python experiment/modern_models.py all
Writes: results/modern_models/<model>.json and <model>_samples.txt
"""
import json
import os
import re
import sys
import time
import warnings

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from war_bias import (CULTURE_TOKENS, CULTURE_WORDS, GROUPS, PEACE_SENTENCES, PROBE_TEMPLATES, PROMPTS,
                      WAR_SENTENCES, WAR_TOKENS, WAR_WORDS, has_any)
from step2_steering import KURDISH_WORDS
from common import RESULTS

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 4)

MODELS = {
    "gpt2": dict(hf="openai-community/gpt2", year=2019, maker="OpenAI", label="GPT-2 small (2019)"),
    "smollm2-360m": dict(hf="HuggingFaceTB/SmolLM2-360M", year=2024, maker="Hugging Face", label="SmolLM2-360M (2024)"),
    "qwen2.5-0.5b": dict(hf="Qwen/Qwen2.5-0.5B", year=2024, maker="Alibaba", label="Qwen2.5-0.5B (2024)"),
    "qwen2.5-0.5b-instruct": dict(hf="Qwen/Qwen2.5-0.5B-Instruct", year=2024, maker="Alibaba",
                                  label="Qwen2.5-0.5B-Instruct (2024, assistant-trained)"),
    "olmo2-1b": dict(hf="allenai/OLMo-2-0425-1B", year=2025, maker="AI2", label="OLMo-2-1B (2025)"),
}
OUT = os.path.join(RESULTS, "modern_models")
os.makedirs(OUT, exist_ok=True)
SEEDS = 8
N_NEW = 30


class Model:
    def __init__(self, hf):
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.m = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.layers = self.m.transformer.h if hasattr(self.m, "transformer") else self.m.model.layers
        self.n_layers = len(self.layers)
        self.hooks = []

    def ids(self, text):
        return self.tok(text, return_tensors="pt").input_ids

    # ---- reading the notebook (hidden states entering each layer) ----
    def hidden(self, text):
        out = self.m(self.ids(text), output_hidden_states=True)
        return torch.stack(out.hidden_states[: self.n_layers])[:, 0]  # (layers, positions, d)

    def mean_hidden(self, sentences):
        return torch.stack([self.hidden(s)[:, 1:].mean(1) for s in sentences])  # (n, layers, d)

    def at_group_word(self, template, g):
        pos = self.ids(template.split("{g}")[0] + g).shape[1] - 1
        return self.hidden(template.format(g=g))[:, pos]

    # ---- changing the notebook ----
    def set_level(self, units, levels):
        """Hold the notebook at a fixed level along one direction, in every layer."""
        self.clear()
        for L, layer in enumerate(self.layers):
            u, lvl = units[L], levels[L]

            def pre_hook(module, args, kwargs, u=u, lvl=lvl):
                h = args[0] if args else kwargs["hidden_states"]
                start = 1 if h.shape[1] > 1 else 0  # leave the very first prompt token alone
                h[:, start:] += (lvl - h[:, start:] @ u)[..., None] * u
                return None
            self.hooks.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    # ---- tests ----
    def word_probs(self, prompt, words):
        """Full probability of each word (all its pieces) coming next."""
        p_ids = self.ids(prompt)[0]
        seqs = [torch.cat([p_ids, self.tok(" " + w, add_special_tokens=False, return_tensors="pt").input_ids[0]])
                for w in words]
        n = max(len(s) for s in seqs)
        batch = torch.full((len(seqs), n), self.tok.pad_token_id)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for i, s in enumerate(seqs):
            batch[i, : len(s)] = s
            mask[i, : len(s)] = 1
        logp = self.m(batch, attention_mask=mask).logits.log_softmax(-1)
        out = []
        for i, s in enumerate(seqs):
            lp = sum(float(logp[i, t - 1, s[t]]) for t in range(len(p_ids), len(s)))
            out.append(np.exp(lp))
        return np.array(out)

    def generate(self, prompt, seed):
        torch.manual_seed(seed)
        ids = self.ids(prompt)
        out = self.m.generate(ids, attention_mask=torch.ones_like(ids), do_sample=True, temperature=0.8, top_k=40,
                              max_new_tokens=N_NEW, num_return_sequences=SEEDS, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]


def next_word(M):
    res = {}
    for g in GROUPS:
        shares = []
        for p in PROMPTS:
            pw = M.word_probs(p.format(g=g), WAR_TOKENS).sum()
            pc = M.word_probs(p.format(g=g), CULTURE_TOKENS).sum()
            shares.append(float(pw / (pw + pc)))
        res[g] = dict(war_share=float(np.mean(shares)), per_prompt=shares)
    return res


def generation(M, label, lines):
    res = {}
    for gi, g in enumerate(GROUPS):
        w = c = k = n = 0
        for pi, p in enumerate(PROMPTS):
            prompt = p.format(g=g)
            for s, cont in enumerate(M.generate(prompt, 1000 * gi + pi)):
                iw, ic = has_any(cont, WAR_WORDS), has_any(cont, CULTURE_WORDS)
                w, c, n = w + iw, c + ic, n + 1
                k += has_any(cont, KURDISH_WORDS)
                lines.append(f"[{label} | {g} | {s}] {'W' if iw else ' '}{'C' if ic else ' '} {prompt}"
                             + cont.replace("\n", " "))
        res[g] = dict(war_rate=w / n, culture_rate=c / n, n=n)
        if g == "Kurdish":
            res[g]["kurdish_word_rate"] = k / n
        print(f"    {label:8s} {g:12s} war {w:2d}/{n} culture {c:2d}/{n}", flush=True)
    return res


def run(key):
    info = MODELS[key]
    t0 = time.time()
    print(f"=== {info['label']} ===", flush=True)
    M = Model(info["hf"])

    # inside: war direction
    Rw, Rp = M.mean_hidden(WAR_SENTENCES), M.mean_hidden(PEACE_SENTENCES)
    war_dir = Rw.mean(0) - Rp.mean(0)
    unit = war_dir / war_dir.norm(dim=-1, keepdim=True)
    peace_level = (Rp * unit).sum(-1).mean(0)
    gap = ((Rw * unit).sum(-1).mean(0) - peace_level)
    proj = torch.stack([torch.stack([M.at_group_word(t, g) for t in PROBE_TEMPLATES]).mul(unit).sum(-1).mean(0)
                        for g in GROUPS])  # (groups, layers)
    rel = ((proj - proj.mean(0, keepdim=True)) / gap).numpy()
    print("  inside (avg over layers):", {g: round(float(rel[i].mean()), 2) for i, g in enumerate(GROUPS)}, flush=True)

    # Test 1
    nw = next_word(M)
    print("  word guesses war share:", {g: round(v["war_share"], 2) for g, v in nw.items()}, flush=True)

    # Fix 2 + control on word guesses
    M.set_level(unit, peace_level)
    nw_fix = next_word(M)
    g_ = torch.Generator().manual_seed(7)
    r = torch.randn(unit.shape, generator=g_)
    r_unit = r / r.norm(dim=-1, keepdim=True)
    M.set_level(r_unit, (Rp * r_unit).sum(-1).mean(0))
    nw_ctrl = next_word(M)
    M.clear()
    print("  after fix:", {g: round(v["war_share"], 2) for g, v in nw_fix.items()}, flush=True)

    # Test 2 (+ fix)
    lines = []
    gen = generation(M, "normal", lines)
    M.set_level(unit, peace_level)
    gen_fix = generation(M, "fixed", lines)
    M.clear()

    result = dict(model=key, **info, n_layers=M.n_layers, next_word=nw, next_word_fixed=nw_fix,
                  next_word_control=nw_ctrl, generation=gen, generation_fixed=gen_fix,
                  inside_relative_war_level=rel.tolist(), minutes=round((time.time() - t0) / 60, 1))
    json.dump(result, open(os.path.join(OUT, f"{key}.json"), "w"), indent=2)
    open(os.path.join(OUT, f"{key}_samples.txt"), "w").write(
        f"{info['label']}\nW = contains a war word, C = contains a culture/everyday word\n"
        "normal = normal model, fixed = war direction held at peaceful level\n\n" + "\n".join(lines))
    print(f"  done in {result['minutes']} min", flush=True)


if __name__ == "__main__":
    keys = list(MODELS) if sys.argv[1] == "all" else sys.argv[1:]
    for k in keys:
        if os.path.exists(os.path.join(OUT, f"{k}.json")) and "--redo" not in sys.argv:
            print("skip (already done):", k)
            continue
        run(k)
