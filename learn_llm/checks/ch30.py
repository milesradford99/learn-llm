import torch

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain


def _trim(f):
    caches = [(torch.arange(5.0).reshape(1, 1, 5, 1), torch.arange(5.0).reshape(1, 1, 5, 1) + 10) for layer in range(2)]
    out = f(caches, 3)
    if not isinstance(out, list) or len(out) != 2:
        return _plain(out)
    return [[t.reshape(-1).tolist() for t in pair] for pair in out]


def _counting(f, guess_right, limit, stop):
    """A pretend model whose favourite token after a text is the text's length plus 1.

    Its cache holds the tokens fed, as a (1, 1, tokens, 1) tensor, so a cache that wasn't
    trimmed makes the text look longer and the choices come out wrong.
    """
    calls = []

    def step(new_ids, caches):
        calls.append(list(new_ids))
        K, V = caches[0]
        seen = K.shape[2]
        fed = torch.tensor(new_ids, dtype=torch.float32).reshape(1, 1, -1, 1)
        scores = torch.zeros(len(new_ids), 30)
        for i in range(len(new_ids)):
            scores[i, (seen + i + 1 + 1) % 30] = 1.0
        return scores, [(torch.cat([K, fed], dim=2), torch.cat([V, fed], dim=2))]

    def draft(ids, k):
        right = [ids[-1] + 1 + i for i in range(guess_right)]
        return (right + [0, 0, 0, 0, 0])[:k]

    start = [1, 2, 3]
    cache = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    out = f(step, [(cache, cache)], start, limit, stop, draft, 4)
    return out, len(calls), start


EXERCISES = {
    "accept": [
        Case("ch30.accept.hand", lambda f: f([5, 9, 2, 7], [5, 9, 4, 1, 8]), [5, 9, 4], hint="ch30.accept.hint"),
        Case("ch30.accept.all", lambda f: f([5, 9, 2, 7], [5, 9, 2, 7, 8]), [5, 9, 2, 7, 8], hint="ch30.accept.hint"),
        Case("ch30.accept.none", lambda f: (f([], [6]), f([3, 4], [1, 4, 5])), ([6], [1]), hint="ch30.accept.hint"),
    ],
    "trim": [
        Case("ch30.trim.three", _trim, [[[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]]] * 2, hint="ch30.trim.hint"),
    ],
    "lookup_draft": [
        Case("ch30.lookup_draft.hand", lambda f: f([4, 7, 1, 9, 4, 7, 2, 4, 7], 2, 3), [2, 4, 7], hint="ch30.lookup_draft.hint"),
        Case("ch30.lookup_draft.short", lambda f: (f([4, 7, 1, 9, 4, 7, 2, 4, 7], 2, 5), f([4, 7, 1, 9, 4, 7, 2, 4, 7], 2, 1)), ([2, 4, 7], [2]), hint="ch30.lookup_draft.hint"),
        Case("ch30.lookup_draft.none", lambda f: (f([1, 2, 3, 4], 2, 3), f([5, 5], 2, 3), f([5, 5, 5], 2, 3)), ([], [], [5]), hint="ch30.lookup_draft.hint"),
    ],
    "speculate": [
        Case("ch30.speculate.two", lambda f: _counting(f, 2, 20, 9), ([1, 2, 3, 4, 5, 6, 7, 8], 2, [1, 2, 3]), hint="ch30.speculate.hint"),
        Case("ch30.speculate.wrong", lambda f: _counting(f, 0, 20, 9), ([1, 2, 3, 4, 5, 6, 7, 8], 6, [1, 2, 3]), hint="ch30.speculate.hint"),
        Case("ch30.speculate.limit", lambda f: _counting(f, 4, 3, 9), ([1, 2, 3, 4, 5, 6], 1, [1, 2, 3]), hint="ch30.speculate.hint"),
    ],
    "expected_tokens": [
        Case("ch30.expected_tokens.example", lambda f: f(0.5, 2), 1.75, tolerance=1e-9, hint="ch30.expected_tokens.hint"),
        Case("ch30.expected_tokens.none", lambda f: (f(0.3, 0), f(1.0, 4)), (1, 5), tolerance=1e-9, hint="ch30.expected_tokens.hint"),
    ],
}
