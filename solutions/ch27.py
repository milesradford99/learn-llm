"""Reference solutions for chapter 27. Peek only after trying!"""

import json

import torch

from learn_llm.qwen import TYPES


def header_length(first_bytes):
    return int.from_bytes(first_bytes, "little")


def read_header(path):
    with open(path, "rb") as f:
        n = header_length(f.read(8))
        return json.loads(f.read(n)), 8 + n


def read_tensor(path, entry, data_start):
    start, stop = entry["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        data = f.read(stop - start)
    return torch.frombuffer(bytearray(data), dtype=TYPES[entry["dtype"]]).reshape(entry["shape"]).float()


def share_heads(x, times):
    return torch.repeat_interleave(x, times, dim=1)


def top_k_chances(scores, temperature, k):
    chances = torch.softmax(scores / temperature, dim=-1)
    top = torch.topk(chances, k)
    kept = torch.zeros_like(chances)
    kept[top.indices] = top.values
    return kept / torch.sum(kept)


def generate(next_scores, ids, limit, stop):
    for step in range(limit):
        token = torch.argmax(next_scores(ids)).item()
        if token == stop:
            return ids
        ids = ids + [token]
    return ids
