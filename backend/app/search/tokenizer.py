"""Text preprocessing pipeline.

The pipeline is deliberately broken into separate, named steps rather than one
opaque call, so each decision is visible and testable independently:

    "Machine Learning is changing the world!"
            |
            v  lowercase
    "machine learning is changing the world!"
            |
            v  tokenize (word extraction via regex, punctuation dropped)
    ["machine", "learning", "is", "changing", "the", "world"]
            |
            v  stopword removal
    ["machine", "learning", "changing", "world"]
            |
            v  stemming (optional, default on)
    ["machin", "learn", "chang", "world"]

Design decisions (see README):
  - We use a small built-in stopword list instead of an NLTK download, so the
    project has no runtime dependency on downloading external corpora.
  - Stemming uses a compact Porter-stemmer implementation (no external library)
    so index and query terms are conflated consistently (e.g. "running"/"run").
    It's optional — pass stem=False to disable and index/search on surface forms.
"""
from __future__ import annotations

import re
from typing import List

# A standard general-English stopword list (comparable in coverage to NLTK's).
# Kept as a plain set so it's easy to read, audit, and extend.
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}

# Word extraction: sequences of ASCII letters/digits, with internal
# apostrophes/hyphens allowed so "don't" and "state-of-the-art" survive as
# single tokens before stopword filtering strips the former.
#
# This is deliberately ASCII-only (ships as an explicit [A-Za-z0-9] character
# class, not \w), matching the tokenizer/stopword-list/stemmer's overall
# English-only scope (documented in README Limitations). An earlier version of
# this comment incorrectly called the pattern "unicode-aware" and passed
# re.UNICODE -- that flag only changes the behavior of \w, \d, \s and similar
# escape classes, and has no effect on an explicit [A-Za-z0-9] character
# class, so it was doing nothing. Removed to avoid the misleading impression
# that non-Latin scripts are handled; they are not.
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")


def lowercase(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> List[str]:
    """Extract word tokens from text, dropping punctuation and whitespace."""
    return _WORD_RE.findall(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def normalize(tokens: List[str]) -> List[str]:
    """Basic normalization: strip stray leading/trailing punctuation left over
    from tokenization (e.g. a trailing apostrophe) and drop empty/pure-numeric
    noise tokens shorter than 1 char."""
    out = []
    for t in tokens:
        t = t.strip("'-")
        if t:
            out.append(t)
    return out


# --- Minimal Porter stemmer -------------------------------------------------
# A compact, dependency-free implementation of the classic Porter (1980)
# stemming algorithm, covering the common suffix-stripping steps (plurals,
# past tense, -ing forms, and common derivational suffixes). It is not a
# byte-for-byte reproduction of every edge case in the original paper, but it
# handles the vast majority of English words correctly and is transparent
# enough to read and explain in an interview.

_VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Count VC (vowel-consonant) sequences — the Porter algorithm's 'm'."""
    form = "".join("C" if _is_consonant(stem, i) else "V" for i in range(len(stem)))
    return form.count("VC")


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    if len(stem) < 3:
        return False
    i = len(stem) - 1
    return (
        _is_consonant(stem, i) and not _is_consonant(stem, i - 1) and _is_consonant(stem, i - 2)
        and stem[-1] not in "wxy"
    )


def porter_stem(word: str) -> str:
    if len(word) <= 2:
        return word
    w = word

    # Step 1a: plurals
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("ss"):
        pass
    elif w.endswith("s"):
        w = w[:-1]

    # Step 1b: -eed, -ed, -ing
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        stripped = None
        if w.endswith("ed") and _contains_vowel(w[:-2]):
            stripped = w[:-2]
        elif w.endswith("ing") and _contains_vowel(w[:-3]):
            stripped = w[:-3]
        if stripped is not None:
            w = stripped
            if w.endswith(("at", "bl", "iz")):
                w = w + "e"
            elif _ends_double_consonant(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif _measure(w) == 1 and _ends_cvc(w):
                w = w + "e"

    # Step 1c: y -> i
    if w.endswith("y") and _contains_vowel(w[:-1]):
        w = w[:-1] + "i"

    # Step 2: common derivational suffixes (subset covering the frequent cases)
    step2_suffixes = [
        ("ational", "ate"), ("tional", "tion"), ("izer", "ize"), ("iser", "ise"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ("logi", "log"), ("entli", "ent"),
    ]
    for suf, repl in step2_suffixes:
        if w.endswith(suf) and _measure(w[: -len(suf)]) > 0:
            w = w[: -len(suf)] + repl
            break

    # Step 3: further derivational suffixes
    step3_suffixes = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
        ("ical", "ic"), ("ful", ""), ("ness", ""),
    ]
    for suf, repl in step3_suffixes:
        if w.endswith(suf) and _measure(w[: -len(suf)]) > 0:
            w = w[: -len(suf)] + repl
            break

    # Step 4: remove trailing suffixes when measure > 1
    step4_suffixes = [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
        "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ]
    for suf in step4_suffixes:
        if w.endswith(suf):
            base = w[: -len(suf)]
            if suf == "ion":
                continue
            if _measure(base) > 1:
                w = base
            break

    # Step 5a: remove trailing 'e'
    if w.endswith("e"):
        base = w[:-1]
        m = _measure(base)
        if m > 1 or (m == 1 and not _ends_cvc(base)):
            w = base

    # Step 5b: remove one of double 'l' at end when measure > 1
    if w.endswith("ll") and _measure(w[:-1]) > 1:
        w = w[:-1]

    return w


def stem_tokens(tokens: List[str]) -> List[str]:
    return [porter_stem(t) for t in tokens]


def preprocess(text: str, *, stem: bool = True) -> List[str]:
    """Run the full pipeline: lowercase -> tokenize -> stopword removal -> normalize -> (stem)."""
    text = lowercase(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = normalize(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return tokens
