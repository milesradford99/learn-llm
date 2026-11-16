import numpy as np
import torch

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain


def _attention(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


def _attend(f, new, total):
    """Compare with chapter 16's attention over all `total` tokens, keeping the last `new` rows."""
    torch.manual_seed(28)
    Q, K, V = torch.randn(1, 2, total, 4), torch.randn(1, 2, total, 4), torch.randn(1, 2, total, 4)
    out = f(Q[:, :, total - new:], K, V)
    if not isinstance(out, torch.Tensor):
        return _plain(out)
    if list(out.shape) != [1, 2, new, 4]:
        return f"shape {list(out.shape)}"
    future = torch.triu(torch.ones(total, total, dtype=torch.bool), diagonal=1)
    right = _attention(Q, K, V, future)[:, :, total - new:]
    if torch.max(torch.abs(out - right)) < 1e-5:
        return "right"
    nothing = _attention(Q[:, :, total - new:], K, V, torch.zeros(new, total, dtype=torch.bool))
    if torch.max(torch.abs(out - nothing)) < 1e-5:
        return "nothing blocked"
    return "wrong cells blocked"


def _remember(f):
    old = torch.arange(4.0).reshape(1, 1, 2, 2), torch.arange(4.0).reshape(1, 1, 2, 2) + 10
    out = f(old, torch.tensor([[[[4.0, 5.0]]]]), torch.tensor([[[[14.0, 15.0]]]]))
    if not isinstance(out, (tuple, list)) or len(out) != 2 or not all(isinstance(t, torch.Tensor) for t in out):
        return _plain(out)
    return [list(t.shape) for t in out], out[0][0, 0, :, 0].tolist(), out[1][0, 0, :, 0].tolist()


def _remember_empty(f):
    out = f((torch.zeros(1, 8, 0, 4), torch.zeros(1, 8, 0, 4)), torch.ones(1, 8, 3, 4), torch.ones(1, 8, 3, 4))
    if not isinstance(out, (tuple, list)) or len(out) != 2 or not all(isinstance(t, torch.Tensor) for t in out):
        return _plain(out)
    return [list(t.shape) for t in out]


def _generate(f, limit):
    """A pretend model whose cache is the list of ids it was fed: it favours their number plus 1."""
    fed = []

    def step(new_ids, caches):
        fed.append(list(new_ids))
        seen = caches + list(new_ids)
        scores = torch.zeros(20)
        scores[(len(seen) + 1) % 20] = 1.0
        return scores, seen

    start = [1, 2, 3]
    return f(step, [], start, limit, 7), fed, start


def _quantize(f, rows, bits, dtype):
    out = f(torch.tensor(rows, dtype=dtype), bits)
    if not isinstance(out, (tuple, list)) or len(out) != 2 or not all(isinstance(t, torch.Tensor) for t in out):
        return _plain(out)
    q, scale = out
    return q.tolist(), str(q.dtype), [round(s, 6) for s in scale.reshape(-1).tolist()]


def _matmul(f, x):
    q = torch.tensor([[3, -1, 2], [-127, 0, 50]], dtype=torch.int8)
    out = f(torch.tensor(x), q, torch.tensor([0.5, 0.01]))
    return [[round(v, 5) for v in row] for row in out.reshape(-1, out.shape[-1]).tolist()] if isinstance(out, torch.Tensor) else _plain(out)


EXERCISES = {
    "attend_new": [
        Case("ch28.attend_new.hand", lambda f: _attend(f, 2, 5), "right", hint="ch28.attend_new.hint"),
        Case("ch28.attend_new.one", lambda f: _attend(f, 1, 4), "right", hint="ch28.attend_new.hint"),
        Case("ch28.attend_new.all", lambda f: _attend(f, 3, 3), "right", hint="ch28.attend_new.hint"),
    ],
    "remember": [
        Case("ch28.remember.order", _remember, ([[1, 1, 3, 2], [1, 1, 3, 2]], [0.0, 2.0, 4.0], [10.0, 12.0, 14.0]), hint="ch28.remember.hint"),
        Case("ch28.remember.empty", _remember_empty, [[1, 8, 3, 4], [1, 8, 3, 4]], hint="ch28.remember.hint"),
    ],
    "generate_cached": [
        Case("ch28.generate_cached.stop", lambda f: _generate(f, 10), ([1, 2, 3, 4, 5, 6], [[1, 2, 3], [4], [5], [6]], [1, 2, 3]), hint="ch28.generate_cached.hint"),
        Case("ch28.generate_cached.limit", lambda f: _generate(f, 2), ([1, 2, 3, 4, 5], [[1, 2, 3], [4]], [1, 2, 3]), hint="ch28.generate_cached.hint"),
    ],
    "quantize": [
        Case("ch28.quantize.hand", lambda f: _quantize(f, [[1.4, -0.66, 0.25], [-0.07, 0.03, 0.012]], 4, torch.float64), ([[7, -3, 1], [-7, 3, 1]], "torch.int8", [0.2, 0.01]), hint="ch28.quantize.hint"),
        Case("ch28.quantize.eight", lambda f: _quantize(f, [[1.27, -0.5, 0.004]], 8, torch.float32), ([[127, -50, 0]], "torch.int8", [0.01]), hint="ch28.quantize.hint"),
    ],
    "quantized_matmul": [
        Case("ch28.quantized_matmul.hand", lambda f: _matmul(f, [1.0, 2.0, 4.0]), [[4.5, 0.73]], hint="ch28.quantized_matmul.hint"),
        Case("ch28.quantized_matmul.many", lambda f: _matmul(f, [[[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]]]), [[1.5, -1.27], [1.5, 1.0]], hint="ch28.quantized_matmul.hint"),
    ],
}
