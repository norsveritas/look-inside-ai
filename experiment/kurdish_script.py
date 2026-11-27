"""Rule-based conversion of Kurmanji from Latin letters (Hawar alphabet) to Arabic letters
(the Kurdish Arabic alphabet used for Badini in Duhok). Used by experiment 6.

Main rules: one letter maps to one letter; "i" is not written (except at the start of a word,
where it becomes ئ); a word that starts with a vowel gets ئ in front; "û" is written وو;
"r" at the start of a word is written ڕ.
"""
import re

MAP = {
    "a": "ا", "b": "ب", "c": "ج", "ç": "چ", "d": "د", "e": "ە", "ê": "ێ", "f": "ف", "g": "گ",
    "h": "ه", "ḧ": "ح", "i": "", "î": "ی", "j": "ژ", "k": "ک", "l": "ل", "m": "م", "n": "ن",
    "o": "ۆ", "p": "پ", "q": "ق", "r": "ر", "s": "س", "ş": "ش", "t": "ت", "u": "و", "û": "وو",
    "v": "ڤ", "w": "و", "x": "خ", "ẍ": "غ", "y": "ی", "z": "ز",
}
VOWELS = set("aeêiîouû")
PUNCT = {",": "،", "?": "؟", ";": "؛"}


def word_to_arabic(w):
    low = w.lower()
    out = []
    for i, ch in enumerate(low):
        if i == 0 and ch in VOWELS:
            out.append("ئ")
        if i == 0 and ch == "r":
            out.append("ڕ")
            continue
        out.append(MAP.get(ch, ch))
    return "".join(out)


def to_arabic(text):
    parts = re.split(r"([A-Za-zÇçÊêÎîŞşÛûḦḧẌẍ]+)", text)
    out = []
    for p in parts:
        if re.fullmatch(r"[A-Za-zÇçÊêÎîŞşÛûḦḧẌẍ]+", p or "-"):
            out.append(word_to_arabic(p))
        else:
            out.append("".join(PUNCT.get(c, c) for c in p))
    return "".join(out)


if __name__ == "__main__":
    from latent_words import COLUMNS, WORDS
    rows = [dict(zip(COLUMNS, r)) for r in WORDS]
    ok = 0
    for r in rows:
        lat, ar = r["kmr"].split("/")[0], r["kmr_ar"].split("/")[0]
        got = to_arabic(lat)
        ok += got == ar
        if got != ar:
            print(f"{r['en']:<10} {lat:<12} rule: {got:<10} list: {ar}")
    print(f"{ok}/{len(rows)} words match the checked word list exactly")
