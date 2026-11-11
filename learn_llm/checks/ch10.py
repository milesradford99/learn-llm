import torch
from torch import nn

from learn_llm.checking import Case
from learn_llm.checks.ch05 import _plain as _plain_np

MIA_T = [14, 17, 20, 22, 25, 28, 31, 33]
MIA_S = [63, 90, 137, 148, 193, 234, 257, 288]


def _plain(value):
    """Tensor results as plain numbers and lists, so feedback reads like the lesson."""
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if isinstance(value, (tuple, list)):
        return type(value)(_plain(v) for v in value)
    return _plain_np(value)


def _tiny_knobs():
    X = torch.tensor([[1.0, 2.0], [0.0, -1.0]], dtype=torch.float64)
    W1 = torch.tensor([[1.0, -1.0, 0.5], [2.0, 0.0, -1.0]], dtype=torch.float64)
    b1 = torch.tensor([0.0, 1.0, -3.0], dtype=torch.float64)
    W2 = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]], dtype=torch.float64)
    b2 = torch.tensor([0.5, -0.5], dtype=torch.float64)
    return X, W1, b1, W2, b2


def _scores(f):
    return _plain(f(*_tiny_knobs()))


def _reference_scores():
    X, W1, b1, W2, b2 = _tiny_knobs()
    return (torch.relu(X @ W1 + b1) @ W2 + b2).tolist()


def _kinds(f):
    """The layers of a model, by name; anything else is reported as it is."""
    model = f(32)
    return [type(layer).__name__ for layer in model] if isinstance(model, nn.Sequential) else _plain(model)


def _knob_count(f):
    return sum(p.numel() for p in f(32).parameters())


def _layer_sizes(f):
    a, _, b = f(32)
    return [a.in_features, a.out_features, b.in_features, b.out_features]


def _tiny_task():
    torch.manual_seed(1)
    return torch.rand(6, 4), torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)


def _tiny_model():
    torch.manual_seed(2)
    return nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))


def _steps(f, steps):
    """Weights after `steps` of plain SGD; old blame left in .grad shows up as a wrong result."""
    model, (X, labels) = _tiny_model(), _tiny_task()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    out = None
    for _ in range(steps):
        out = f(model, optimizer, X, labels)
    return out, model


def _returns_number(f):
    return type(_steps(f, 1)[0]).__name__


def _moved(f):
    return round(sum(p.sum().item() for p in _steps(f, 3)[1].parameters()), 5)


def _losses(f):
    model, (X, labels) = _tiny_model(), _tiny_task()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    first = f(model, optimizer, X, labels)
    for _ in range(30):
        last = f(model, optimizer, X, labels)
    return last < first


def _reference_moved():
    from torch.nn.functional import cross_entropy

    model, (X, labels) = _tiny_model(), _tiny_task()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        cross_entropy(model(X), labels).backward()
        optimizer.step()
    return round(sum(p.sum().item() for p in model.parameters()), 5)


EXERCISES = {
    "torch_gradient": [
        Case("ch10.torch_gradient.tiny", lambda f: _plain(f(1.0, 0.0, [1, 2], [2, 4])), (-5, -3), tolerance=1e-4, hint="ch10.torch_gradient.hint"),
        Case("ch10.torch_gradient.mia", lambda f: _plain(f(10.0, -60.0, MIA_T, MIA_S)), (-94, 2.5), tolerance=1e-3, hint="ch10.torch_gradient.hint"),
    ],
    "scores": [
        Case("ch10.scores.tiny", _scores, _reference_scores(), tolerance=1e-9, hint="ch10.scores.hint"),
    ],
    "make_model": [
        Case("ch10.make_model.layers", _kinds, ["Linear", "ReLU", "Linear"], hint="ch10.make_model.hint"),
        Case("ch10.make_model.sizes", _layer_sizes, [784, 32, 32, 10], hint="ch10.make_model.hint"),
        Case("ch10.make_model.knobs", _knob_count, 784 * 32 + 32 + 32 * 10 + 10, hint="ch10.make_model.hint"),
    ],
    "train_step": [
        Case("ch10.train_step.number", _returns_number, "float", hint="ch10.train_step.number_hint"),
        Case("ch10.train_step.steps", _moved, _reference_moved(), tolerance=1e-4, hint="ch10.train_step.hint"),
        Case("ch10.train_step.learns", _losses, True, hint="ch10.train_step.hint"),
    ],
}
