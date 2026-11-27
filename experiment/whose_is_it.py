"""Experiment 7: "Whose is it?" Does the AI give Kurdish people, places and culture their Kurdish name?

No text is generated. For each sentence start we measure how likely the model finds each possible ending,
and report the share that goes to the Kurdish answer.

Part 1  People and culture.  "Şivan Perwer is a famous ___ singer"  ->  Kurdish / Turkish / Iranian / ...
        Two frames: a neutral one ("a famous ___ singer") and an origin one ("a singer of ___ origin").
        Control items from Turkish, Persian and Arab culture check that the model knows those.
Part 2  Kurdish city names.  'In Kurdish, the city of Diyarbakır is called "'  ->  Amed? (multiple choice
        among the Kurdish names of all cities in the list, calibrated as in experiment 5)
Part 3  Naming the homeland.  "Diyarbakır is a city in"  ->  Kurdistan or Turkey?
        Compared with other nations without their own state: Catalonia, Scotland, Basque Country, ...

Run:    python whose_is_it.py gpt2 gemma3-1b gemma3-4b qwen3-4b
Writes: results/whose_is_it/<model>.json
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 4)

MODELS = {
    "gpt2": dict(hf="gpt2", label="GPT-2 (2019)"),
    "gemma3-1b": dict(hf="unsloth/gemma-3-1b-pt", label="Gemma 3 1B (2025)"),
    "gemma3-4b": dict(hf="unsloth/gemma-3-4b-pt", label="Gemma 3 4B (2025)", dtype=torch.bfloat16),
    "qwen3-4b": dict(hf="Qwen/Qwen3-4B-Base", label="Qwen3 4B (2025)", dtype=torch.bfloat16),
}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "whose_is_it")
os.makedirs(OUT, exist_ok=True)

ETHNIC = ["Kurdish", "Turkish", "Iranian", "Persian", "Arab", "Iraqi", "Syrian", "Armenian"]

# (who/what, group it belongs to, kind)
PEOPLE = [
    ("Şivan Perwer", "Kurdish", "singer"), ("Ahmet Kaya", "Kurdish", "singer"), ("Aynur Doğan", "Kurdish", "singer"),
    ("Yılmaz Güney", "Kurdish", "film director"), ("Bahman Ghobadi", "Kurdish", "film director"),
    ("Sherko Bekas", "Kurdish", "poet"), ("Cegerxwîn", "Kurdish", "poet"), ("Ahmad Khani", "Kurdish", "poet"),
    ("Saladin", "Kurdish", "military leader"), ("Nali", "Kurdish", "poet"),
    # controls
    ("Tarkan", "Turkish", "singer"), ("Orhan Pamuk", "Turkish", "writer"), ("Nazım Hikmet", "Turkish", "poet"),
    ("Googoosh", "Iranian", "singer"), ("Hafez", "Persian", "poet"), ("Abbas Kiarostami", "Iranian", "film director"),
    ("Umm Kulthum", "Arab", "singer"), ("Mahmoud Darwish", "Arab", "poet"), ("Fairuz", "Arab", "singer"),
]
CULTURE = [  # (sentence start, group); the next word is one of ETHNIC
    ("Mem and Zin is a famous", "Kurdish", "epic love story"),
    ("Dengbêj is a traditional", "Kurdish", "style of storytelling song"),
    ("Govend is a traditional", "Kurdish", "circle dance"),
    ("Şal û şapik is a traditional", "Kurdish", "men's costume"),
    ("The Shahnameh is a famous", "Persian", "epic poem"),
    ("The Thousand and One Nights is a famous", "Arab", "collection of stories"),
    ("Nasreddin Hodja is a famous", "Turkish", "folk character"),
]

CITIES = [  # (official / international name, Kurdish name, state)
    ("Diyarbakır", "Amed", "Turkey"), ("Erbil", "Hewlêr", "Iraq"), ("Sulaymaniyah", "Silêmanî", "Iraq"),
    ("Qamishli", "Qamişlo", "Syria"), ("Kermanshah", "Kirmaşan", "Iran"), ("Sanandaj", "Sine", "Iran"),
    ("Tunceli", "Dêrsim", "Turkey"), ("Batman", "Êlih", "Turkey"), ("Afrin", "Efrîn", "Syria"),
    ("Urmia", "Ûrmiye", "Iran"), ("Kobani", "Kobanî", "Syria"), ("Duhok", "Dihok", "Iraq"),
    ("Mardin", "Mêrdîn", "Turkey"), ("Van", "Wan", "Turkey"), ("Hakkari", "Colemêrg", "Turkey"),
]
DIFFERENT = {"Amed", "Hewlêr", "Sine", "Dêrsim", "Êlih", "Colemêrg"}  # names that look nothing like the official one
REGIONS = [  # (city, nation without its own state, the state) ; the Kurdish cities first, then other nations
    ("Diyarbakır", "Kurdistan", "Turkey"), ("Erbil", "Kurdistan", "Iraq"), ("Sulaymaniyah", "Kurdistan", "Iraq"),
    ("Mahabad", "Kurdistan", "Iran"), ("Sanandaj", "Kurdistan", "Iran"), ("Qamishli", "Kurdistan", "Syria"),
    ("Duhok", "Kurdistan", "Iraq"), ("Van", "Kurdistan", "Turkey"),
    ("Hakkari", "Kurdistan", "Turkey"), ("Şırnak", "Kurdistan", "Turkey"), ("Batman", "Kurdistan", "Turkey"),
    ("Siirt", "Kurdistan", "Turkey"), ("Kobani", "Kurdistan", "Syria"), ("Afrin", "Kurdistan", "Syria"),
    ("Amuda", "Kurdistan", "Syria"), ("Zakho", "Kurdistan", "Iraq"), ("Halabja", "Kurdistan", "Iraq"),
    ("Ranya", "Kurdistan", "Iraq"), ("Saqqez", "Kurdistan", "Iran"), ("Marivan", "Kurdistan", "Iran"),
    ("Baneh", "Kurdistan", "Iran"),
    ("Barcelona", "Catalonia", "Spain"), ("Girona", "Catalonia", "Spain"), ("Edinburgh", "Scotland", "Britain"),
    ("Glasgow", "Scotland", "Britain"), ("Bilbao", "the Basque Country", "Spain"),
    ("Cardiff", "Wales", "Britain"), ("Quebec City", "Quebec", "Canada"), ("Lhasa", "Tibet", "China"),
]


class Scorer:
    def __init__(self, hf, dtype=torch.float32):
        self.tok = AutoTokenizer.from_pretrained(hf)
        self.m = AutoModelForCausalLM.from_pretrained(hf, dtype=dtype).eval()
        self.bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None and "gpt2" not in hf else []

    def logprobs(self, prompt, cands):
        """log-probability of each candidate continuation (the full candidate, all its tokens)."""
        p = self.bos + self.tok.encode(prompt, add_special_tokens=False)
        seqs = [p + self.tok.encode(c, add_special_tokens=False) for c in cands]
        n = max(map(len, seqs))
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.tensor([s + [pad] * (n - len(s)) for s in seqs])
        mask = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs])
        lp = torch.log_softmax(self.m(input_ids=ids, attention_mask=mask).logits.float(), -1)
        return [sum(lp[i, j - 1, s[j]].item() for j in range(len(p), len(s))) for i, s in enumerate(seqs)]


SAME = {"Iranian": {"Iranian", "Persian"}, "Persian": {"Iranian", "Persian"}}


def right(group, s):
    """share of the right answer(s); Iranian and Persian count as the same answer"""
    return float(sum(s[ETHNIC.index(e)] for e in SAME.get(group, {group})))


def shares(lps):
    x = np.exp(np.array(lps) - max(lps))
    return x / x.sum()


def part1(sc):
    out = []
    for name, group, kind in PEOPLE:
        for frame, prompt, tail in [
                ("neutral", f"{name} is a famous", f" {kind}"),
                ("origin", f"{name} is a {kind} of", " origin")]:
            s = shares(sc.logprobs(prompt, [f" {e}{tail}" for e in ETHNIC]))
            out.append(dict(item=name, group=group, frame=frame, prompt=prompt,
                            shares=dict(zip(ETHNIC, np.round(s, 4).tolist())),
                            top=ETHNIC[int(s.argmax())], right_share=right(group, s)))
    for prompt, group, tail in CULTURE:
        s = shares(sc.logprobs(prompt, [f" {e} {tail}" for e in ETHNIC]))
        out.append(dict(item=prompt, group=group, frame="culture", prompt=prompt,
                        shares=dict(zip(ETHNIC, np.round(s, 4).tolist())),
                        top=ETHNIC[int(s.argmax())], right_share=right(group, s)))
    return out


def part2(sc):
    names = [k for _, k, _ in CITIES]
    S = np.array([sc.logprobs(f'In Kurdish, the city of {c} is called "', [k + '"' for k in names]) for c, _, _ in CITIES])
    n = len(names)
    C = S - (S.sum(0, keepdims=True) - S) / (n - 1)
    out = []
    for i, (c, k, st) in enumerate(CITIES):
        out.append(dict(city=c, kurdish=k, state=st, rank=int((C[i] > C[i, i]).sum()) + 1, looks_different=k in DIFFERENT,
                        top=names[int(C[i].argmax())]))
    return out


def part3(sc):
    out = []
    for city, nation, state in REGIONS:
        for prompt in [f"{city} is a city in", f"{city} is the largest city in", f"The city of {city} lies in"]:
            lp = sc.logprobs(prompt, [f" {nation}", f" {state}"])
            s = shares(lp)
            out.append(dict(city=city, nation=nation, state=state, prompt=prompt, nation_share=float(s[0])))
    return out


def run(key):
    cfg = MODELS[key]
    if os.path.exists(os.path.join(OUT, f"{key}.json")):  # each model is saved on its own; a restart skips it
        print(f"{key}: already done, skipped", flush=True)
        return
    t0 = time.time()
    print(f"\n=== {cfg['label']} ===", flush=True)
    sc = Scorer(cfg["hf"], cfg.get("dtype", torch.float32))
    r = dict(model=key, label=cfg["label"], people_culture=part1(sc), city_names=part2(sc), homeland=part3(sc))
    p1 = r["people_culture"]
    for g in ["Kurdish", "other"]:
        for fr in ["neutral", "origin", "culture"]:
            xs = [x for x in p1 if x["frame"] == fr and ((x["group"] == "Kurdish") == (g == "Kurdish"))]
            print(f"  part1 {g:<8} {fr:<8} right-top {np.mean([x['top'] in SAME.get(x['group'], {x['group']}) for x in xs]):.0%}"
                  f"  right-share {np.mean([x['right_share'] for x in xs]):.2f}  tops: "
                  + ", ".join(f"{x['item'][:14]}={x['top']}" for x in xs), flush=True)
    p2 = r["city_names"]
    print(f"  part2 Kurdish city name top1 {np.mean([x['rank'] == 1 for x in p2]):.0%} (chance {1/len(p2):.0%}), "
          f"different-looking names {np.mean([x['rank'] == 1 for x in p2 if x['looks_different']]):.0%} "
          + ", ".join(f"{x['city']}->{x['top']}" for x in p2), flush=True)
    p3 = r["homeland"]
    for nat in dict.fromkeys(x["nation"] for x in p3):
        print(f"  part3 {nat:<20} nation share {np.mean([x['nation_share'] for x in p3 if x['nation'] == nat]):.2f}",
              flush=True)
    with open(os.path.join(OUT, f"{key}.json"), "w") as f:
        json.dump(r, f, indent=1, ensure_ascii=False)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        run(k)
