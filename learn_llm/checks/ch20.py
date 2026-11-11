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


def _swiglu(f):
    torch.manual_seed(4)
    W1, W2, W3 = nn.Linear(3, 4, bias=False), nn.Linear(4, 3, bias=False), nn.Linear(3, 4, bias=False)
    x = torch.rand(2, 3)
    with torch.no_grad():
        mine = f(x, W1, W2, W3)
        reference = W2(nn.functional.silu(W1(x)) * W3(x))
    return bool(torch.max(torch.abs(mine - reference)) < 1e-6)


def _adamw(f, steps):
    w, m, v = np.array([1.0, -2.0]), np.zeros(2), np.zeros(2)
    for t in range(1, steps + 1):
        out = f(w, m, v, np.array([0.5, -3.0]) * t, t, 0.1, 0.5)
        if not isinstance(out, tuple) or len(out) != 3:
            return _plain_np(out)
        w, m, v = out
    return _plain_np(w)


def _adamw_pytorch(steps):
    w = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.AdamW([w], lr=0.1, weight_decay=0.5)
    for t in range(1, steps + 1):
        w.grad = torch.tensor([0.5, -3.0], dtype=torch.float64) * t
        optimizer.step()
    return w.detach().tolist()


def _checkpoint(f):
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    out = f(model, optimizer, 7)
    if not isinstance(out, dict):
        return _plain(out)
    return sorted(out), out.get("step"), sorted(out.get("model", {}))


EXERCISES = {
    "rms_norm": [
        Case("ch20.rms_norm.values", _rms, _rms_reference(), tolerance=1e-5, hint="ch20.rms_norm.hint"),
    ],
    "rotate": [
        Case("ch20.rotate.quarter", lambda f: _plain(f(1.0, 0.0, np.pi / 2)), (0.0, 1.0), tolerance=1e-9, hint="ch20.rotate.hint"),
        Case("ch20.rotate.half", lambda f: _plain(f(2.0, 3.0, np.pi)), (-2.0, -3.0), tolerance=1e-9, hint="ch20.rotate.hint"),
        Case("ch20.rotate.none", lambda f: _plain(f(0.6, -0.8, 0.0)), (0.6, -0.8), tolerance=1e-9, hint="ch20.rotate.hint"),
    ],
    "swiglu": [
        Case("ch20.swiglu.layers", _swiglu, True, hint="ch20.swiglu.hint"),
    ],
    "adamw_step": [
        Case("ch20.adamw_step.one", lambda f: _adamw(f, 1), _adamw_pytorch(1), tolerance=1e-6, hint="ch20.adamw_step.hint"),
        Case("ch20.adamw_step.five", lambda f: _adamw(f, 5), _adamw_pytorch(5), tolerance=1e-6, hint="ch20.adamw_step.hint"),
    ],
    "learning_rate": [
        Case("ch20.learning_rate.warmup", lambda f: f(50, 1000, 100, 0.002), 0.001, tolerance=1e-12, hint="ch20.learning_rate.hint"),
        Case("ch20.learning_rate.peak", lambda f: f(100, 1000, 100, 0.002), 0.002, tolerance=1e-12, hint="ch20.learning_rate.hint"),
        Case("ch20.learning_rate.middle", lambda f: f(550, 1000, 100, 0.002), 0.0011, tolerance=1e-12, hint="ch20.learning_rate.hint"),
        Case("ch20.learning_rate.end", lambda f: f(1000, 1000, 100, 0.002), 0.0002, tolerance=1e-12, hint="ch20.learning_rate.hint"),
    ],
    "checkpoint": [
        Case("ch20.checkpoint.parts", _checkpoint, (["model", "optimizer", "step"], 7, ["bias", "weight"]), hint="ch20.checkpoint.hint"),
    ],
}
