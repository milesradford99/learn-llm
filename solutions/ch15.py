"""Reference solutions for chapter 15. Peek only after trying!"""

import torch


def shift_right(Y):
    return torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)


def weighted_average(a, V):
    return a @ V


def attention_shares(q, K):
    return torch.softmax(K @ q, dim=0)


def attend(Q, K, V):
    scores = Q @ K.transpose(1, 2)
    A = torch.softmax(scores, dim=2)
    return A @ V, A
