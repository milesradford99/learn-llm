"""Reference solutions for chapter 14. Peek only after trying!"""

import torch


def rnn_step(x, h, Wx, Wh, b):
    return torch.tanh(x @ Wx + h @ Wh + b)


def run_rnn(X, Wx, Wh, b):
    h = torch.zeros(X.shape[0], Wh.shape[0])
    states = []
    for t in range(X.shape[1]):
        h = rnn_step(X[:, t], h, Wx, Wh, b)
        states.append(h)
    return torch.stack(states, dim=1)


def windows(ids, length):
    n = (len(ids) - 1) // length
    X = ids[:n * length].reshape(n, length)
    Y = ids[1:n * length + 1].reshape(n, length)
    return X, Y


def lstm_step(x, h, c, L):
    both = torch.cat([x, h], dim=1)
    f = torch.sigmoid(L[0](both))
    i = torch.sigmoid(L[1](both))
    o = torch.sigmoid(L[2](both))
    n = torch.tanh(L[3](both))
    c = f * c + i * n
    h = o * torch.tanh(c)
    return h, c


def reader_forward(ids, memory, table, lstm, out):
    states, memory = lstm(table(ids), memory)
    return out(states), memory


def sample_word(scores, temperature):
    p = torch.softmax(scores / temperature, dim=0)
    return torch.multinomial(p, 1).item()
