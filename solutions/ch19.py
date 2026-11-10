"""Reference solutions for chapter 19. Peek only after trying!"""


def to_bytes(text):
    return list(text.encode("utf-8"))


def count_pairs(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i = i + 2
        else:
            out.append(ids[i])
            i = i + 1
    return out


def train_bpe(ids, count):
    merges = {}
    for step in range(count):
        counts = count_pairs(ids)
        pair = max(counts, key=counts.get)
        new_id = 256 + step
        merges[pair] = new_id
        ids = merge(ids, pair, new_id)
    return merges


def encode(text, merges):
    ids = to_bytes(text)
    for pair, new_id in merges.items():
        ids = merge(ids, pair, new_id)
    return ids


def decode(ids, merges):
    table = [bytes([i]) for i in range(256)]
    for pair, new_id in merges.items():
        table.append(table[pair[0]] + table[pair[1]])
    joined = b"".join([table[i] for i in ids])
    return joined.decode("utf-8", errors="replace")
