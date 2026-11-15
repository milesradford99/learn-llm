import numpy as np
import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain as _plain_np
from learn_llm.checks.ch10 import _plain

X = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])
H = torch.tensor([[0.5, -0.5], [0.0, 1.0]])
WX = torch.tensor([[0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
WH = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
B = torch.tensor([0.0, 0.1])


def _reference_step():
    return torch.tanh(X @ WX + H @ WH + B).tolist()


def _run(f):
    return _plain(f(torch.stack([X, X * 2, X * 0], dim=1), WX, WH, B))


def _reference_run():
    h, out = torch.zeros(2, 2), []
    for x in (X, X * 2, X * 0):
        h = torch.tanh(x @ WX + h @ WH + B)
        out.append(h.tolist())
    return [[out[t][row] for t in range(3)] for row in range(2)]


def _run_shape(f):
    out = f(torch.zeros(3, 5, 3), WX, WH, B)
    return list(out.shape) if isinstance(out, torch.Tensor) else _plain(out)


def _pieces(f):
    X_out, Y_out = f(np.array([5, 2, 9, 7, 1]), 2)
    return _plain_np(X_out), _plain_np(Y_out)


def _layers(size_in=3, m=4):
    torch.manual_seed(4)
    return [nn.Linear(size_in + m, m) for _ in range(4)]


def _cell_inputs():
    torch.manual_seed(5)
    return torch.rand(2, 3), torch.rand(2, 4), torch.rand(2, 4)


def _step(f):
    with torch.no_grad():
        h, c = f(*_cell_inputs(), _layers())
    return round(torch.sum(h).item(), 3), round(torch.sum(c).item(), 3)


def _reference_lstm():
    x, h, c = _cell_inputs()
    L = _layers()
    both = torch.cat([x, h], dim=1)
    with torch.no_grad():
        c = torch.sigmoid(L[0](both)) * c + torch.sigmoid(L[1](both)) * torch.tanh(L[3](both))
        h = torch.sigmoid(L[2](both)) * torch.tanh(c)
    return round(torch.sum(h).item(), 3), round(torch.sum(c).item(), 3)


def _matches_pytorch(f):
    """The learner's step, with the knobs of a real nn.LSTMCell copied in."""
    torch.manual_seed(6)
    cell = nn.LSTMCell(3, 4)
    x, h, c = _cell_inputs()
    L = _layers()
    with torch.no_grad():
        # PyTorch keeps its four gates stacked in the order input, forget, new, output.
        for gate, layer in enumerate((L[1], L[0], L[3], L[2])):
            layer.weight[:] = torch.cat([cell.weight_ih[gate * 4:gate * 4 + 4], cell.weight_hh[gate * 4:gate * 4 + 4]], dim=1)
            layer.bias[:] = (cell.bias_ih + cell.bias_hh)[gate * 4:gate * 4 + 4]
        mine = f(x, h, c, L)
        theirs = cell(x, (h, c))
    return bool(torch.max(torch.abs(mine[0] - theirs[0])) < 1e-5 and torch.max(torch.abs(mine[1] - theirs[1])) < 1e-5)


ONE = (torch.tensor([[2.0]]), torch.tensor([[1.0]]), torch.tensor([-1.0]))


def _reader():
    torch.manual_seed(8)
    return nn.Embedding(7, 3), nn.LSTM(3, 4, batch_first=True), nn.Linear(4, 7)


IDS = torch.tensor([[1, 5, 2], [0, 0, 6]])


def _reader_shapes(f):
    with torch.no_grad():
        out = f(IDS, None, *_reader())
    if not (isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], torch.Tensor)):
        return _plain(out)
    return list(out[0].shape), [list(part.shape) for part in out[1]]


def _reader_values(f):
    table, lstm, out = _reader()
    with torch.no_grad():
        mine = f(IDS, None, table, lstm, out)[0]
        theirs = out(lstm(table(IDS))[0])
    return bool(torch.max(torch.abs(mine - theirs)) < 1e-5)


def _reader_carries(f):
    """Reading 3 words at once must equal reading 2, then 1 more from the memory handed back."""
    table, lstm, out = _reader()
    with torch.no_grad():
        whole = f(IDS, None, table, lstm, out)[0][:, -1]
        _scores, memory = f(IDS[:, :2], None, table, lstm, out)
        last = f(IDS[:, 2:], memory, table, lstm, out)[0][:, -1]
    return bool(torch.max(torch.abs(whole - last)) < 1e-5)


def _temperature(f, temperature):
    """How many of 1000 draws took id 0, with scores [1, 0]."""
    torch.manual_seed(7)
    scores = torch.tensor([1.0, 0.0])
    return sum(f(scores, temperature) == 0 for _ in range(1000))


EXERCISES = {
    "rnn_step": [
        Case("ch14.rnn_step.hand", lambda f: _plain(f(torch.tensor([[1.0]]), torch.tensor([[0.0]]), *ONE)), [[float(np.tanh(1.0))]], tolerance=1e-5, hint="ch14.rnn_step.hint"),
        Case("ch14.rnn_step.tiny", lambda f: _plain(f(X, H, WX, WH, B)), _reference_step(), tolerance=1e-5, hint="ch14.rnn_step.hint"),
    ],
    "run_rnn": [
        Case("ch14.run_rnn.hand", lambda f: _plain(f(torch.tensor([[[1.0], [0.0]]]), *ONE)), [[[float(np.tanh(1.0))], [float(np.tanh(np.tanh(1.0) - 1.0))]]], tolerance=1e-5, hint="ch14.run_rnn.hint"),
        Case("ch14.run_rnn.shape", _run_shape, [3, 5, 2], hint="ch14.run_rnn.hint"),
        Case("ch14.run_rnn.values", _run, _reference_run(), tolerance=1e-5, hint="ch14.run_rnn.hint"),
    ],
    "windows": [
        Case("ch14.windows.page", _pieces, ([[5, 2], [9, 7]], [[2, 9], [7, 1]]), hint="ch14.windows.hint"),
        Case("ch14.windows.exact", lambda f: _plain_np(f(np.arange(6), 3)[0]), [[0, 1, 2]], hint="ch14.windows.hint"),
        Case("ch14.windows.rest", lambda f: _plain_np(f(np.arange(8), 3)[0]), [[0, 1, 2], [3, 4, 5]], hint="ch14.windows.hint"),
    ],
    "lstm_step": [
        Case("ch14.lstm_step.values", _step, _reference_lstm(), tolerance=1e-2, hint="ch14.lstm_step.hint"),
        Case("ch14.lstm_step.pytorch", _matches_pytorch, True, hint="ch14.lstm_step.hint"),
    ],
    "reader_forward": [
        Case("ch14.reader_forward.shapes", _reader_shapes, ([2, 3, 7], [[1, 2, 4], [1, 2, 4]]), hint="ch14.reader_forward.hint"),
        Case("ch14.reader_forward.values", _reader_values, True, hint="ch14.reader_forward.hint"),
        Case("ch14.reader_forward.carries", _reader_carries, True, hint="ch14.reader_forward.memory_hint"),
    ],
    "sample_word": [
        Case("ch14.sample_word.gives_id", lambda f: type(f(torch.tensor([1.0, 0.0]), 1.0)).__name__, "int", hint="ch14.sample_word.hint"),
        Case("ch14.sample_word.cold", lambda f: _temperature(f, 0.1) > 990, True, hint="ch14.sample_word.hint"),
        Case("ch14.sample_word.hot", lambda f: 400 < _temperature(f, 100.0) < 600, True, hint="ch14.sample_word.hint"),
    ],
}
