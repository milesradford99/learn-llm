import torch
import torch.nn.functional as F

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain


def _logprob(f, seed):
    torch.manual_seed(seed)
    scores, answers = torch.randn(3, 2, 12, dtype=torch.float64), torch.randint(10, (3, 2))
    return _plain(f(scores, answers))


def _logprob_expected(seed):
    torch.manual_seed(seed)
    scores, answers = torch.randn(3, 2, 12, dtype=torch.float64), torch.randint(10, (3, 2))
    logp = F.log_softmax(scores, dim=-1)
    rows = torch.arange(3)
    return (logp[rows, 0, answers[:, 0]] + logp[rows, 1, answers[:, 1]]).tolist()


def _dpo(f, better, worse, ref_better, ref_worse, beta):
    return _plain(f(*[torch.tensor(v, dtype=torch.float64) for v in (better, worse, ref_better, ref_worse)], beta))


def _adv(f, rows):
    return _plain(f(torch.tensor(rows, dtype=torch.float64)))


EXERCISES = {
    "reward": [
        Case("ch26.reward.right", lambda f: f("3+4=", "07"), 1.0, hint="ch26.reward.hint"),
        Case("ch26.reward.wrong", lambda f: f("3+4=", "08"), 0.0, hint="ch26.reward.hint"),
        Case("ch26.reward.carry", lambda f: f("9+9=", "18"), 1.0, hint="ch26.reward.hint"),
        Case("ch26.reward.digits", lambda f: f("5+2=", "52"), 0.0, hint="ch26.reward.hint"),
    ],
    "answer_logprob": [
        Case("ch26.answer_logprob.even", lambda f: _plain(f(torch.zeros(2, 2, 12, dtype=torch.float64), torch.tensor([[0, 7], [1, 8]]))), [-4.969813299576001] * 2, tolerance=1e-6, hint="ch26.answer_logprob.hint"),
        Case("ch26.answer_logprob.random", lambda f: _logprob(f, 0), _logprob_expected(0), tolerance=1e-6, hint="ch26.answer_logprob.hint"),
    ],
    "dpo_loss": [
        Case("ch26.dpo_loss.start", lambda f: _dpo(f, [-2.0], [-2.0], [-2.0], [-2.0], 1.0), 0.6931471805599453, tolerance=1e-6, hint="ch26.dpo_loss.hint"),
        Case("ch26.dpo_loss.hand", lambda f: _dpo(f, [-1.0], [-3.0], [-2.0], [-2.0], 1.0), 0.1269280110429726, tolerance=1e-6, hint="ch26.dpo_loss.hint"),
        Case("ch26.dpo_loss.beta", lambda f: _dpo(f, [-1.0], [-3.0], [-2.0], [-2.0], 0.5), 0.3132616875182228, tolerance=1e-6, hint="ch26.dpo_loss.hint"),
        Case("ch26.dpo_loss.mean", lambda f: _dpo(f, [-2.0, -1.0], [-2.0, -3.0], [-2.0, -2.0], [-2.0, -2.0], 1.0), 0.41003759580145895, tolerance=1e-6, hint="ch26.dpo_loss.hint"),
    ],
    "advantages": [
        Case("ch26.advantages.hand", lambda f: _adv(f, [[1, 0, 0, 1]]), [[1.0, -1.0, -1.0, 1.0]], tolerance=1e-4, hint="ch26.advantages.hint"),
        Case("ch26.advantages.scale", lambda f: _adv(f, [[10, 0, 0, 10]]), [[1.0, -1.0, -1.0, 1.0]], tolerance=1e-4, hint="ch26.advantages.hint"),
        Case("ch26.advantages.alike", lambda f: _adv(f, [[1, 1, 1, 1]]), [[0.0, 0.0, 0.0, 0.0]], tolerance=1e-4, hint="ch26.advantages.hint"),
        Case("ch26.advantages.rows", lambda f: _adv(f, [[1, 0, 0, 1], [0, 0, 0, 0]]), [[1.0, -1.0, -1.0, 1.0], [0.0, 0.0, 0.0, 0.0]], tolerance=1e-4, hint="ch26.advantages.hint"),
    ],
    "policy_loss": [
        Case("ch26.policy_loss.two", lambda f: _plain(f(torch.tensor([-1.0, -2.0]), torch.tensor([1.0, -1.0]))), -0.5, tolerance=1e-6, hint="ch26.policy_loss.hint"),
        Case("ch26.policy_loss.none", lambda f: _plain(f(torch.tensor([-1.0, -2.0]), torch.tensor([0.0, 0.0]))), 0.0, tolerance=1e-6, hint="ch26.policy_loss.hint"),
    ],
}
