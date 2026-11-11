import torch

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain

Q = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
K = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
V = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])


def _reference_attend():
    A = torch.softmax(Q @ K.transpose(1, 2), dim=2)
    return (A @ V).tolist(), A.tolist()


def _attend(f):
    return _plain(f(Q, K, V))


def _sums_to_one(f):
    return _plain(torch.sum(f(torch.tensor([1.0, 2.0]), K[0])))


def _shape(f):
    blended, A = f(torch.zeros(2, 4, 3), torch.zeros(2, 5, 3), torch.zeros(2, 5, 7))
    return list(blended.shape), list(A.shape)


def _reference_with():
    blended, A = _reference_attend_on(Q, K, K)
    return torch.cat([Q, blended], dim=2).tolist(), A.tolist()


def _reference_attend_on(Q, K, V):
    A = torch.softmax(Q @ K.transpose(1, 2), dim=2)
    return A @ V, A


def _with_shape(f):
    joined, A = f(torch.zeros(2, 4, 3), torch.zeros(2, 5, 3))
    return list(joined.shape), list(A.shape)


def _keeps_states(f):
    return _plain(f(Q, K)[0][:, :, :2])


EXERCISES = {
    "shift_right": [
        Case("ch15.shift_right.page", lambda f: _plain(f(torch.tensor([[7, 8, 9], [4, 5, 6]]))), [[0, 7, 8], [0, 4, 5]], hint="ch15.shift_right.hint"),
    ],
    "weighted_average": [
        Case("ch15.weighted_average.pick", lambda f: _plain(f(torch.tensor([1.0, 0.0]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))), [1.0, 2.0], tolerance=1e-6, hint="ch15.weighted_average.hint"),
        Case("ch15.weighted_average.blend", lambda f: _plain(f(torch.tensor([0.5, 0.5]), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))), [2.0, 3.0], tolerance=1e-6, hint="ch15.weighted_average.hint"),
    ],
    "attention_shares": [
        Case("ch15.attention_shares.equal", lambda f: _plain(f(torch.tensor([0.0, 0.0]), K[0])), [1 / 3, 1 / 3, 1 / 3], tolerance=1e-6, hint="ch15.attention_shares.hint"),
        Case("ch15.attention_shares.sum", _sums_to_one, 1.0, tolerance=1e-6, hint="ch15.attention_shares.hint"),
        Case("ch15.attention_shares.tiny", lambda f: _plain(f(torch.tensor([1.0, 0.0]), K[0])), torch.softmax(torch.tensor([1.0, 0.0, 1.0]), dim=0).tolist(), tolerance=1e-6, hint="ch15.attention_shares.hint"),
    ],
    "attend": [
        Case("ch15.attend.shapes", _shape, ([2, 4, 7], [2, 4, 5]), hint="ch15.attend.hint"),
        Case("ch15.attend.values", _attend, _reference_attend(), tolerance=1e-5, hint="ch15.attend.hint"),
    ],
    "with_attention": [
        Case("ch15.with_attention.shapes", _with_shape, ([2, 4, 6], [2, 4, 5]), hint="ch15.with_attention.hint"),
        Case("ch15.with_attention.keeps", _keeps_states, Q.tolist(), tolerance=1e-6, hint="ch15.with_attention.hint"),
        Case("ch15.with_attention.values", lambda f: _plain(f(Q, K)), _reference_with(), tolerance=1e-5, hint="ch15.with_attention.hint"),
    ],
}
