"""Reference solutions for chapter 26. Peek only after trying!"""

import torch
import torch.nn.functional as F


def reward(prompt, answer):
    return 1.0 if int(prompt[0]) + int(prompt[2]) == int(answer) else 0.0


def answer_logprob(scores, answers):
    first = F.cross_entropy(scores[:, 0], answers[:, 0], reduction="none")
    second = F.cross_entropy(scores[:, 1], answers[:, 1], reduction="none")
    return -(first + second)


def dpo_loss(better, worse, ref_better, ref_worse, beta):
    margin = (better - ref_better) - (worse - ref_worse)
    return -torch.mean(F.logsigmoid(beta * margin))


def advantages(rewards):
    centred = rewards - torch.mean(rewards, dim=1, keepdim=True)
    spread = torch.sqrt(torch.mean(centred * centred, dim=1, keepdim=True))
    return centred / (spread + 1e-6)


def policy_loss(logprobs, advantage):
    return -torch.mean(advantage * logprobs)
