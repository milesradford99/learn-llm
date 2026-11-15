"""Reference solutions for chapter 25. Peek only after trying!"""

import zlib

from learn_llm.rawdata import seeded_hash


def normalise(text):
    return " ".join(text.lower().split())


def fingerprint(text):
    return zlib.crc32(normalise(text).encode("utf-8"))


def drop_copies(texts):
    seen = set()
    kept = []
    for text in texts:
        mark = fingerprint(text)
        if mark not in seen:
            seen.add(mark)
            kept.append(text)
    return kept


def shingles(words, n):
    return set([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])


def jaccard(a, b):
    return len(a & b) / len(a | b)


def minhash(shingle_set, seeds):
    return [min([seeded_hash(seed, s) for s in shingle_set]) for seed in seeds]


def agreement(sig_a, sig_b):
    return sum([a == b for a, b in zip(sig_a, sig_b)]) / len(sig_a)


def known_share(words, known):
    return sum([w in known for w in words]) / len(words)
