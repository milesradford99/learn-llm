"""A fast byte-level BPE tokenizer for the stories, the same method as chapter 19.

Chapter 19's tokenizer is written for clarity: it recounts every pair over the whole text at
each step. This one does the same merges two standard ways faster: it trains on a table of
word counts instead of the text, and it encodes each distinct word once and remembers it.
Merges never cross the start of a word, so every token belongs to one word.

    python -m learn_llm.tokenizer   # rebuild data/tinystories-bpe.json from the stories
"""

import json
import re
from pathlib import Path

MERGES_FILE = Path(__file__).resolve().parent / "data" / "tinystories-bpe.json"


def to_words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def story_words(story_list: list[str], known: set[str] | None = None) -> list[list[str]]:
    """Each story as chapter 12's lower-case words. With `known`, any other word becomes "<unk>",
    so a token model reads exactly the text the word models of chapters 13-18 read."""
    if known is None:
        return [to_words(story) for story in story_list]
    return [[w if w in known else "<unk>" for w in to_words(story)] for story in story_list]


def _merge(ids: tuple, pair: tuple, new_id: int) -> tuple:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return tuple(out)


def train(word_counts: dict[str, int], count: int) -> list[tuple[int, int]]:
    """The `count` merges chapter 19's train_bpe would find, counting each word once per use."""
    words = {tuple((" " + w).encode("utf-8")): n for w, n in word_counts.items()}
    merges = []
    for step in range(count):
        pairs: dict[tuple, int] = {}
        for ids, n in words.items():
            for pair in zip(ids, ids[1:]):
                pairs[pair] = pairs.get(pair, 0) + n
        if not pairs:
            break
        pair = max(pairs, key=pairs.get)
        merges.append(pair)
        words = {_merge(ids, pair, 256 + step): n for ids, n in words.items()}
    return merges


class Tokenizer:
    """Encode stories of words into token ids; `end` marks the end of a story and `unk` stands
    for "<unk>", a word outside the word models' vocabulary."""

    def __init__(self, merges: list[tuple[int, int]]):
        self.merges = [tuple(m) for m in merges]
        self.end = 256 + len(self.merges)
        self.unk = self.end + 1
        self.size = self.unk + 1
        self._cache: dict[str, tuple] = {"<unk>": (self.unk,)}
        table = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            table.append(table[a] + table[b])
        self.table = table + [b" <end>", b" <unk>"]

    @classmethod
    def load(cls) -> "Tokenizer":
        return cls(json.loads(MERGES_FILE.read_text(encoding="utf-8")))

    def word(self, w: str) -> tuple:
        if w not in self._cache:
            ids = tuple((" " + w).encode("utf-8"))
            for new_id, pair in enumerate(self.merges, start=256):
                if len(ids) == 1:
                    break
                ids = _merge(ids, pair, new_id)
            self._cache[w] = ids
        return self._cache[w]

    def encode(self, stories: list[list[str]]) -> list[int]:
        out = []
        for words in stories:
            for w in words:
                out.extend(self.word(w))
            out.append(self.end)
        return out

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.table[i] for i in ids).decode("utf-8", errors="replace").strip()


if __name__ == "__main__":
    from learn_llm.data import stories

    train_stories, _ = stories()
    counts: dict[str, int] = {}
    for words in story_words(train_stories):
        for w in words:
            counts[w] = counts.get(w, 0) + 1
    MERGES_FILE.write_text(json.dumps(train(counts, 2000)), encoding="utf-8")
    print(f"wrote {MERGES_FILE}")
