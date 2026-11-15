"""A raw pile of text for chapter 25: real stories with planted copies, near-copies and junk.

Every item carries a hidden label saying what it is, so the lesson can count what a learner's
cleaning removed and what it kept. The pile is built from the bundled stories with a fixed seed,
so it is the same on every machine.
"""

import zlib

import numpy as np

from learn_llm.data import stories
from learn_llm.tokenizer import to_words

STORY, COPY, NEAR, GIBBERISH, MENU, SPAM = "story", "copy", "near-copy", "gibberish", "menu", "spam"
_MENU = (
    "home | about us | contact | login | register | subscribe to our newsletter | privacy policy | "
    "terms of service | copyright {year} all rights reserved | click here to accept cookies | page {page} of {pages}"
)
_SPAM = ["buy", "cheap", "best", "price", "free", "now", "click", "win", "deal", "offer"]
_LETTERS = np.array(list("abcdefghijklmnopqrstuvwxyz"))


def seeded_hash(seed: int, item) -> int:
    """One of many hash functions, picked by `seed`: a number from 0 to about 4 billion."""
    return zlib.crc32(f"{seed}:{item}".encode("utf-8"))


def _recase(text: str, rng) -> str:
    """The same text as another site would show it: other spacing, sometimes other capitals."""
    words = text.split()
    gaps = rng.choice([" ", "  ", "\n"], size=len(words), p=[0.9, 0.07, 0.03])
    out = "".join(w + g for w, g in zip(words, gaps)).strip()
    return out.upper() if rng.random() < 0.2 else out


def _near(text: str, rng, pool: list[str]) -> str:
    """The same story lightly edited: about one word in twenty swapped for another."""
    words = text.split()
    for i in rng.choice(len(words), size=max(1, len(words) // 20), replace=False):
        words[i] = pool[rng.integers(len(pool))]
    return " ".join(words)


def _junk(kind: str, rng) -> str:
    if kind == GIBBERISH:
        lengths = rng.integers(2, 9, size=rng.integers(40, 160))
        return " ".join("".join(rng.choice(_LETTERS, size=n)) for n in lengths)
    if kind == MENU:
        return _MENU.format(year=rng.integers(2005, 2024), page=rng.integers(1, 9), pages=rng.integers(9, 40))
    return " ".join(rng.choice(_SPAM, size=rng.integers(40, 160)))


def raw_pile(seed: int = 0) -> tuple[list[str], list[str]]:
    """(texts, kinds): 1,000 stories plus 250 copies, 250 near-copies and 500 pieces of junk, shuffled."""
    rng = np.random.default_rng(seed)
    originals = stories()[0][2000:3000]
    pool = sorted({w for story in originals[:50] for w in to_words(story)})
    items = [(story, STORY) for story in originals]
    items += [(_recase(originals[i], rng), COPY) for i in rng.choice(1000, size=250, replace=False)]
    items += [(_near(originals[i], rng, pool), NEAR) for i in rng.choice(1000, size=250, replace=False)]
    items += [(_junk(kind, rng), kind) for kind in [GIBBERISH] * 200 + [MENU] * 150 + [SPAM] * 150]
    order = rng.permutation(len(items))
    # An original always comes before its copies, as the page a crawler happened to find first.
    order = sorted(order, key=lambda i: items[i][1] in (COPY, NEAR))
    return [items[i][0] for i in order], [items[i][1] for i in order]


def common_words(count: int = 2000) -> set[str]:
    """The `count` most frequent words of the training stories: the word models' vocabulary."""
    counts: dict[str, int] = {}
    for story in stories()[0]:
        for word in to_words(story):
            counts[word] = counts.get(word, 0) + 1
    return set(sorted(counts, key=counts.get, reverse=True)[:count])


def pile_tokens(texts: list[str]):
    """The texts as one tensor of token ids, in the form `learn_llm.scaling.small_run` trains on."""
    import torch

    from learn_llm.tokenizer import Tokenizer, story_words

    return torch.tensor(Tokenizer.load().encode(story_words(texts, common_words())))
