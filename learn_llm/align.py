"""A tiny world for chapter 26: sums like `3+4=07`, a small model that writes them, and the
plumbing between text and tensors, so the lesson can spend its page on the three losses.

A line is 6 tokens: digit, +, digit, =, digit, digit. The first 4 are the prompt, the last 2
the answer. The "pretrained" model has read thousands of such lines whose answers were random
numbers from 00 to 18, so it knows what an answer looks like and nothing about which is right.
"""

import torch
import torch.nn.functional as F

from learn_llm.scaling import SmallGPT

PLUS, EQUALS, VOCAB = 10, 11, 12
_SYMBOLS = "0123456789+="


def all_prompts() -> torch.Tensor:
    """The 100 prompts `a+b=`, as a (100, 4) tensor of token ids, in the order 0+0, 0+1, …, 9+9."""
    a = torch.arange(10).repeat_interleave(10)
    b = torch.arange(10).repeat(10)
    return torch.stack([a, torch.full_like(a, PLUS), b, torch.full_like(a, EQUALS)], dim=1)


def to_text(ids: torch.Tensor) -> list[str]:
    """Rows of token ids as strings: [[3, 10, 4, 11]] gives ["3+4="]."""
    return ["".join(_SYMBOLS[i] for i in row) for row in ids.tolist()]


def right_answers(prompts: torch.Tensor) -> torch.Tensor:
    """The correct two-digit answer of each prompt, as a (n, 2) tensor."""
    total = prompts[:, 0] + prompts[:, 2]
    return torch.stack([total // 10, total % 10], dim=1)


def pretrained(seed: int = 0) -> SmallGPT:
    """A small model trained on lines with random answers from 00 to 18. A few seconds."""
    torch.manual_seed(seed)
    model = SmallGPT(VOCAB, 32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    for _ in range(300):
        prompts = all_prompts()[torch.randint(100, (64,))]
        total = torch.randint(19, (64,))
        lines = torch.cat([prompts, torch.stack([total // 10, total % 10], dim=1)], dim=1)
        optimizer.zero_grad()
        F.cross_entropy(model(lines[:, :-1]).reshape(-1, VOCAB), lines[:, 1:].reshape(-1)).backward()
        optimizer.step()
    return model


def answer_scores(model, prompts: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    """The model's scores for the two answer tokens, shape (n, 2, 12), when it reads each prompt
    followed by its answer: row 0 scores the first digit, row 1 the second, given the first."""
    return model(torch.cat([prompts, answers[:, :1]], dim=1))[:, 3:]


def sample_answers(model, prompts: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`count` answers per prompt, drawn from the model's own chances. Gives back each prompt
    repeated `count` times, shape (n * count, 4), and the answers beside them, shape (n * count, 2)."""
    with torch.no_grad():
        many = prompts.repeat_interleave(count, dim=0)
        first = torch.multinomial(torch.softmax(model(many)[:, -1], dim=-1), 1)
        second = torch.multinomial(torch.softmax(model(torch.cat([many, first], dim=1))[:, -1], dim=-1), 1)
    return many, torch.cat([first, second], dim=1)


def score_samples(reward, prompts: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    """The learner's `reward(prompt, answer)`, called on each row as text: one number per row.
    An answer that isn't two digits earns 0 without the checker being asked."""
    return torch.tensor([float(reward(p, a)) if a.isdigit() else 0.0 for p, a in zip(to_text(prompts), to_text(answers))])


def judged_pairs(model, prompts: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pairs of the model's own answers, compared by a judge who prefers the one nearer the true
    sum and has no opinion when both are equally far. Gives back (prompts, better, worse)."""
    many, one = sample_answers(model, prompts, count)
    _, other = sample_answers(model, prompts, count)
    truth = many[:, 0] + many[:, 2]
    miss_one = torch.abs(one[:, 0] * 10 + one[:, 1] - truth)
    miss_other = torch.abs(other[:, 0] * 10 + other[:, 1] - truth)
    decided = miss_one != miss_other
    first_wins = (miss_one < miss_other)[:, None]
    return many[decided], torch.where(first_wins, one, other)[decided], torch.where(first_wins, other, one)[decided]


def best_answers(model, prompts: torch.Tensor) -> torch.Tensor:
    """The model's most likely answer to each prompt, digit by digit: shape (n, 2)."""
    with torch.no_grad():
        first = torch.argmax(model(prompts)[:, -1], dim=-1, keepdim=True)
        second = torch.argmax(model(torch.cat([prompts, first], dim=1))[:, -1], dim=-1, keepdim=True)
    return torch.cat([first, second], dim=1)


def share_right(model) -> float:
    """The share of the 100 prompts whose most likely answer is correct."""
    prompts = all_prompts()
    return round(torch.mean(torch.all(best_answers(model, prompts) == right_answers(prompts), dim=1).float()).item(), 2)
