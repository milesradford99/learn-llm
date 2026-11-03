import numpy as np
import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain
from learn_llm.checks.ch09 import _plain as _plain_t

EDGE = np.array([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]])
STEP = np.array([[0.0, 0.0, 1.0, 1.0]] * 4)
POOL = np.array([[1.0, 3.0, 0.0, 2.0], [5.0, 2.0, 1.0, 0.0], [0.0, 0.0, 4.0, 1.0], [1.0, 0.0, 6.0, 2.0]])


def _odd_pool(f):
    return _plain(f(np.arange(1.0, 26.0).reshape(5, 5)))


def _kinds(f):
    model = f()
    return [type(layer).__name__ for layer in model] if isinstance(model, nn.Sequential) else _plain_t(model)


def _knobs(f):
    return sum(p.numel() for p in f().parameters())


def _paddings(f):
    return [layer.padding[0] for layer in f() if isinstance(layer, nn.Conv2d)]


def _out_shape(f, size):
    with torch.no_grad():
        return list(f()(torch.zeros(2, 1, size, size)).shape)


def _convs():
    torch.manual_seed(3)
    conv1 = nn.Conv2d(2, 2, 3, padding=1)
    conv2 = nn.Conv2d(2, 2, 3, padding=1)
    return torch.rand(1, 2, 4, 4), conv1, conv2


def _block(f):
    with torch.no_grad():
        return round(torch.sum(f(*_convs())).item(), 4)


def _block_nonnegative(f):
    with torch.no_grad():
        return bool(torch.min(f(*_convs())) >= 0)


def _reference_block():
    x, conv1, conv2 = _convs()
    with torch.no_grad():
        return round(torch.sum(torch.relu(x + conv2(torch.relu(conv1(x))))).item(), 4)


EXERCISES = {
    "convolve": [
        Case("ch10.convolve.step", lambda f: _plain(f(STEP, EDGE)), [[3.0, 3.0], [3.0, 3.0]], tolerance=1e-9, hint="ch10.convolve.hint"),
        Case("ch10.convolve.flat", lambda f: _plain(f(np.ones((3, 4)), EDGE)), [[0.0, 0.0]], tolerance=1e-9, hint="ch10.convolve.hint"),
        Case("ch10.convolve.one", lambda f: _plain(f(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[2.0]]))), [[2.0, 4.0], [6.0, 8.0]], tolerance=1e-9, hint="ch10.convolve.hint"),
    ],
    "max_pool": [
        Case("ch10.max_pool.page", lambda f: _plain(f(POOL)), [[5.0, 2.0], [1.0, 6.0]], tolerance=1e-9, hint="ch10.max_pool.hint"),
        Case("ch10.max_pool.odd", _odd_pool, [[7.0, 9.0], [17.0, 19.0]], tolerance=1e-9, hint="ch10.max_pool.hint"),
    ],
    "make_cnn": [
        Case("ch10.make_cnn.layers", _kinds, ["Conv2d", "ReLU", "MaxPool2d", "Conv2d", "ReLU", "MaxPool2d", "Conv2d", "ReLU", "AdaptiveMaxPool2d", "Flatten", "Linear"], hint="ch10.make_cnn.hint"),
        Case("ch10.make_cnn.knobs", _knobs, 6218, hint="ch10.make_cnn.knobs_hint"),
        Case("ch10.make_cnn.padding", _paddings, [1, 1, 1], hint="ch10.make_cnn.knobs_hint"),
        Case("ch10.make_cnn.shape", lambda f: _out_shape(f, 40), [2, 10], hint="ch10.make_cnn.hint"),
        Case("ch10.make_cnn.any_size", lambda f: _out_shape(f, 28), [2, 10], hint="ch10.make_cnn.hint"),
    ],
    "residual_block": [
        Case("ch10.residual_block.value", _block, _reference_block(), tolerance=1e-3, hint="ch10.residual_block.hint"),
        Case("ch10.residual_block.bend", _block_nonnegative, True, hint="ch10.residual_block.hint"),
    ],
}
