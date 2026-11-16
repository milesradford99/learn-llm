"""Reference solutions for chapter 30. Peek only after trying!"""

import torch


def accept(draft, best):
    m = 0
    while m < len(draft) and draft[m] == best[m]:
        m = m + 1
    return draft[:m] + [best[m]]


def trim(caches, length):
    return [(K[:, :, :length], V[:, :, :length]) for K, V in caches]


def lookup_draft(ids, n, k):
    tail = ids[-n:]
    for start in range(len(ids) - n - 1, -1, -1):
        if ids[start:start + n] == tail:
            return ids[start + n:start + n + k]
    return []


def speculate(step, caches, ids, limit, stop, draft, k):
    prompt = len(ids)
    while len(ids) - prompt < limit:
        guess = draft(ids, k)
        scores, caches = step([ids[-1]] + guess, caches)
        new = accept(guess, torch.argmax(scores, dim=-1).tolist())
        caches = trim(caches, len(ids) + len(new) - 1)
        for token in new:
            if token == stop:
                return ids
            ids = ids + [token]
    return ids[:prompt + limit]


def expected_tokens(a, k):
    total = 0
    for i in range(k + 1):
        total = total + a ** i
    return total
