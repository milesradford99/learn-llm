import numpy as np
import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain as _plain_np
from learn_llm.checks.ch10 import _plain


def _rms(f):
    x = torch.tensor([[3.0, 4.0], [1.0, -1.0]])
    return _plain(f(x, torch.tensor([1.0, 2.0])))


def _rms_reference():
    x = torch.tensor([[3.0, 4.0], [1.0, -1.0]])
    return (torch.tensor([1.0, 2.0]) * x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)).tolist()


def _rms_deep(f):
    out = f(torch.ones(2, 3, 4), torch.ones(4))
    return list(out.shape) if isinstance(out, torch.Tensor) else _plain(out)


def _swiglu(f):
    torch.manual_seed(4)
    W1, W2, W3 = nn.Linear(3, 4, bias=False), nn.Linear(4, 3, bias=False), nn.Linear(3, 4, bias=False)
    x = torch.rand(2, 3)
    with torch.no_grad():
        mine = f(x, W1, W2, W3)
        reference = W2(nn.functional.silu(W1(x)) * W3(x))
    return bool(torch.max(torch.abs(mine - reference)) < 1e-6)


def _rounded(out):
    return [round(float(v), 6) + 0.0 for v in out] if isinstance(out, np.ndarray) else _plain_np(out)


def _rope_relative(f):
    """The dot product of a turned query and key depends only on how far apart they are."""
    rng = np.random.default_rng(3)
    q, k, speeds = rng.random(6), rng.random(6), np.array([1.0, 0.1, 0.01])
    near, far = f(q, 4, speeds) @ f(k, 1, speeds), f(q, 14, speeds) @ f(k, 11, speeds)
    return bool(abs(near - far) < 1e-9 and abs(near - q @ k) > 1e-3)


def _block(f):
    """look doubles its input; the SwiGLU layers have known knobs."""
    W1, W2, W3 = nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)
    x = torch.tensor([[[1.0, 7.0], [3.0, -4.0]]])
    gain1, gain2 = torch.tensor([1.0, 2.0]), torch.tensor([0.5, 1.0])
    with torch.no_grad():
        for layer, knobs in ((W1, [[1.0, 0.0], [0.0, 1.0]]), (W2, [[1.0, 1.0], [0.0, 1.0]]), (W3, [[0.5, 0.0], [0.0, 2.0]])):
            layer.weight[:] = torch.tensor(knobs)

        def norm(v, gain):
            return gain * v / torch.sqrt(torch.mean(v * v, dim=-1, keepdim=True) + 1e-6)

        def think(v):
            return W2(nn.functional.silu(W1(v)) * W3(v))

        mine = f(x, gain1, lambda y: 2 * y, gain2, W1, W2, W3)
        middle = x + 2 * norm(x, gain1)
        right = middle + think(norm(middle, gain2))
        stale = middle + think(norm(x, gain2))
    if not isinstance(mine, torch.Tensor):
        return _plain(mine)
    if torch.max(torch.abs(mine - right)) < 1e-5:
        return "right"
    return "second half used the old x" if torch.max(torch.abs(mine - stale)) < 1e-5 else "different"


EXERCISES = {
    "rms_norm": [
        Case("ch20.rms_norm.hand", lambda f: _plain(f(torch.tensor([[1.0, 7.0]]), torch.tensor([1.0, 1.0]))), [[0.2, 1.4]], tolerance=1e-5, hint="ch20.rms_norm.hint"),
        Case("ch20.rms_norm.values", _rms, _rms_reference(), tolerance=1e-5, hint="ch20.rms_norm.hint"),
        Case("ch20.rms_norm.shape", _rms_deep, [2, 3, 4], hint="ch20.rms_norm.hint"),
    ],
    "rotate": [
        Case("ch20.rotate.quarter", lambda f: _plain(f(1.0, 0.0, np.pi / 2)), (0.0, 1.0), tolerance=1e-9, hint="ch20.rotate.hint"),
        Case("ch20.rotate.half", lambda f: _plain(f(2.0, 3.0, np.pi)), (-2.0, -3.0), tolerance=1e-9, hint="ch20.rotate.hint"),
        Case("ch20.rotate.none", lambda f: _plain(f(0.6, -0.8, 0.0)), (0.6, -0.8), tolerance=1e-9, hint="ch20.rotate.hint"),
    ],
    "rope_one": [
        Case("ch20.rope_one.page", lambda f: _rounded(f(np.array([1.0, 0.0, 0.0, 1.0]), 1, np.array([np.pi / 2, np.pi]))), [0.0, 0.0, 1.0, -1.0], tolerance=1e-6, hint="ch20.rope_one.hint"),
        Case("ch20.rope_one.start", lambda f: _rounded(f(np.array([0.5, -2.0, 3.0, 1.0]), 0, np.array([1.0, 0.1]))), [0.5, -2.0, 3.0, 1.0], tolerance=1e-6, hint="ch20.rope_one.hint"),
        Case("ch20.rope_one.position", lambda f: _rounded(f(np.array([1.0, 0.0, 0.0, 1.0]), 2, np.array([np.pi / 2, np.pi / 4]))), [-1.0, -1.0, 0.0, 0.0], tolerance=1e-6, hint="ch20.rope_one.hint"),
        Case("ch20.rope_one.relative", _rope_relative, True, hint="ch20.rope_one.hint"),
    ],
    "swiglu": [
        Case("ch20.swiglu.layers", _swiglu, True, hint="ch20.swiglu.hint"),
    ],
    "modern_block_step": [
        Case("ch20.modern_block_step.values", _block, "right", hint="ch20.modern_block_step.hint"),
    ],
}
