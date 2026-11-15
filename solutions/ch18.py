"""Reference solutions for chapter 18. Peek only after trying!"""

import numpy as np
import torch


def blank_out(ids, hidden, blank):
    return [blank if i in hidden else i for i in ids]


def classify(features, head):
    return head(torch.mean(features, dim=1))


def majority_share(labels):
    return np.max(np.bincount(labels)) / len(labels)


def precision_recall(guesses, labels, c):
    tp = np.sum((guesses == c) & (labels == c))
    fp = np.sum((guesses == c) & (labels != c))
    fn = np.sum((guesses != c) & (labels == c))
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def margin(p, n):
    return 2 * np.sqrt(p * (1 - p) / n)


def clean_split(texts, n_test):
    cut = len(texts) - n_test
    test = texts[cut:]
    train = [t for t in texts[:cut] if t not in test]
    return train, test


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
