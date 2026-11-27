"""Experiment 6: the "token tax". How many pieces (tokens) does the same sentence cost in Kurdish?

AI models do not read letters or words; they read "tokens", pieces of text from a fixed list.
A language that the list was not built for is cut into many small pieces. More pieces means the
text costs more money, fills the model's memory sooner, is slower, and is usually understood worse.

Data: 1,004 sentences from SIB-200 (Adelani et al. 2024; the sentences come from FLORES-200), each one
translated by professionals into English, Turkish, Persian, Arabic, Sorani and Kurmanji (Latin letters).
The Kurmanji sentences are also converted to Arabic letters (kurdish_script.py) to see the cost of the
*same* Badini/Kurmanji text in the alphabet used in Duhok.

Run:    python token_tax.py
Writes: results/token_tax/token_tax.json
"""
import csv
import json
import os
import urllib.request

from transformers import AutoTokenizer

from kurdish_script import to_arabic

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "token_tax")
CACHE = os.path.join(OUT, "sib200")
os.makedirs(CACHE, exist_ok=True)

LANGS = {"eng_Latn": "English", "tur_Latn": "Turkish", "pes_Arab": "Persian", "arb_Arab": "Arabic",
         "ckb_Arab": "Sorani", "kmr_Latn": "Kurmanji (Latin letters)", "kmr_Arab": "Kurmanji/Badini (Arabic letters)"}

TOKENIZERS = {  # name shown in the report: Hugging Face repo (only the tokenizer files are downloaded)
    "GPT-2 (2019)": "gpt2",
    "GPT-4o (2024)": "Xenova/gpt-4o",
    "Llama 3 (2024)": "unsloth/Llama-3.2-1B",
    "Gemma 3 (2025)": "unsloth/gemma-3-4b-pt",
    "Qwen3 (2025)": "Qwen/Qwen3-4B-Base",
    "DeepSeek-V3 (2024)": "deepseek-ai/DeepSeek-V3",
    "Mistral Nemo (2024)": "unsloth/Mistral-Nemo-Base-2407",
    "SmolLM2 (2024)": "HuggingFaceTB/SmolLM2-360M",
    "NLLB-200 (2022, built for 200 languages)": "facebook/nllb-200-distilled-600M",
}


def load_sentences():
    data = {}
    for code in [c for c in LANGS if c != "kmr_Arab"]:
        rows = {}
        for split in ["train", "dev", "test"]:
            path = os.path.join(CACHE, f"{code}_{split}.tsv")
            if not os.path.exists(path):
                url = f"https://huggingface.co/datasets/Davlan/sib200/resolve/main/data/{code}/{split}.tsv"
                urllib.request.urlretrieve(url, path)
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    rows[r["index_id"]] = r["text"].strip()
        data[code] = rows
    ids = sorted(set.intersection(*[set(v) for v in data.values()]), key=int)
    data = {c: [data[c][i] for i in ids] for c in data}
    data["kmr_Arab"] = [to_arabic(s) for s in data["kmr_Latn"]]
    return data


def main():
    data = load_sentences()
    n = len(data["eng_Latn"])
    print(f"{n} parallel sentences")
    res = dict(n_sentences=n, languages=LANGS, example={c: v[0] for c, v in data.items()}, tokenizers={})
    for name, repo in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(repo)
        r = {}
        for c, sents in data.items():
            counts = [len(tok.encode(s, add_special_tokens=False)) for s in sents]
            words = [w for s in sents for w in s.split()]
            pieces = [len(tok.encode(" " + w, add_special_tokens=False)) for w in words]
            r[c] = dict(tokens=sum(counts), chars=sum(map(len, sents)), words=len(words),
                        tokens_per_word=sum(pieces) / len(words),
                        share_words_4plus=sum(p >= 4 for p in pieces) / len(words))
        eng = r["eng_Latn"]["tokens"]
        for c in r:
            r[c]["vs_english"] = r[c]["tokens"] / eng
        res["tokenizers"][name] = dict(repo=repo, vocab_size=len(tok), langs=r)
        print(f"{name:<42}" + "  ".join(f"{c[:3]}{'A' if c == 'kmr_Arab' else ''}:{r[c]['vs_english']:.2f}"
                                        for c in r), flush=True)
    with open(os.path.join(OUT, "token_tax.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
