import numpy as np
import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain as _plain_np
from learn_llm.checks.ch10 import _plain


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


def _hand(f, decay):
    """The page's first step: w = 2, slope 0.5, rate 0.1."""
    out = f(np.array([2.0]), np.zeros(1), np.zeros(1), np.array([0.5]), 1, 0.1, decay)
    return _plain_np(out[0]) if isinstance(out, tuple) and len(out) == 3 else _plain_np(out)


def _resumed(f):
    """A fresh model and optimizer, filled from a checkpoint taken after two steps."""
    torch.manual_seed(2)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        torch.sum(model(torch.ones(1, 2))).backward()
        optimizer.step()
    saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": 2}
    fresh = nn.Linear(2, 1)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=0.1)
    step = f(fresh, fresh_optimizer, saved)
    same_knobs = bool(torch.equal(fresh.weight, model.weight))
    same_averages = len(fresh_optimizer.state_dict()["state"]) == len(optimizer.state_dict()["state"])
    return step, same_knobs, same_averages


EXERCISES = {
    "adamw_step": [
        Case("ch21.adamw_step.hand", lambda f: _hand(f, 0.5), [1.8], tolerance=1e-6, hint="ch21.adamw_step.hint"),
        Case("ch21.adamw_step.no_decay", lambda f: _hand(f, 0.0), [1.9], tolerance=1e-6, hint="ch21.adamw_step.hint"),
        Case("ch21.adamw_step.one", lambda f: _adamw(f, 1), _adamw_pytorch(1), tolerance=1e-6, hint="ch21.adamw_step.hint"),
        Case("ch21.adamw_step.five", lambda f: _adamw(f, 5), _adamw_pytorch(5), tolerance=1e-6, hint="ch21.adamw_step.hint"),
    ],
    "learning_rate": [
        Case("ch21.learning_rate.warmup", lambda f: f(50, 1000, 100, 0.002), 0.001, tolerance=1e-12, hint="ch21.learning_rate.hint"),
        Case("ch21.learning_rate.peak", lambda f: f(100, 1000, 100, 0.002), 0.002, tolerance=1e-12, hint="ch21.learning_rate.hint"),
        Case("ch21.learning_rate.middle", lambda f: f(550, 1000, 100, 0.002), 0.0011, tolerance=1e-12, hint="ch21.learning_rate.hint"),
        Case("ch21.learning_rate.end", lambda f: f(1000, 1000, 100, 0.002), 0.0002, tolerance=1e-12, hint="ch21.learning_rate.hint"),
    ],
    "checkpoint": [
        Case("ch21.checkpoint.parts", _checkpoint, (["model", "optimizer", "step"], 7, ["bias", "weight"]), hint="ch21.checkpoint.hint"),
    ],
    "resume": [
        Case("ch21.resume.all", _resumed, (2, True, True), hint="ch21.resume.hint"),
    ],
}
