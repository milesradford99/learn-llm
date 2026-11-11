"""Reference solutions for chapter 12. Peek only after trying!"""

import numpy as np
import torch
import torch.nn.functional as F


def count_words(words):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


def similarity(a, b):
    length_a = np.sqrt(a @ a)
    length_b = np.sqrt(b @ b)
    return (a @ b) / (length_a * length_b)


def skipgram_pairs(ids, window):
    centres = []
    contexts = []
    for i in range(len(ids)):
        for j in range(max(0, i - window), min(len(ids), i + window + 1)):
            if j != i:
                centres.append(ids[i])
                contexts.append(ids[j])
    return centres, contexts


def pair_loss(u, v, n):
    real = torch.sum(u * v, dim=1)
    fake = torch.sum(n * u.unsqueeze(1), dim=2)
    surprise = -F.logsigmoid(real) - torch.sum(F.logsigmoid(-fake), dim=1)
    return torch.mean(surprise)
