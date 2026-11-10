"""Reference solutions for chapter 17. Peek only after trying!"""

import torch


def sample_word(scores, temperature):  # given in the lesson, from chapter 14
    p = torch.softmax(scores / temperature, dim=0)
    return torch.multinomial(p, 1).item()


def embed(ids, words, positions):
    return words(ids) + positions(torch.arange(ids.shape[1]))


def layer_norm(x):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(variance + 1e-5)


def block_step(x, norm1, attention, norm2, think):
    x = x + attention(norm1(x))
    x = x + think(norm2(x))
    return x


def gpt_forward(ids, words, positions, blocks, norm, out):
    x = embed(ids, words, positions)
    return out(norm(blocks(x)))


def write_ids(model, first, count, temperature, longest):
    words = [first]
    with torch.no_grad():
        for step in range(count):
            S = model(torch.tensor([words[-longest:]]))
            words.append(sample_word(S[0, -1], temperature))
    return words
