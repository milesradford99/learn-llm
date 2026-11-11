"""Reference solutions for chapter 18. Peek only after trying!"""

import numpy as np
import torch
import torch.nn.functional as F


def rms_norm(x, gain):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    return gain * x / rms


def rotate(a, b, angle):
    return a * np.cos(angle) - b * np.sin(angle), a * np.sin(angle) + b * np.cos(angle)


def swiglu(x, W1, W2, W3):
    return W2(F.silu(W1(x)) * W3(x))


def adamw_step(w, m, v, grad, t, rate, decay):
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad * grad
    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)
    return w - rate * (m_hat / (np.sqrt(v_hat) + 1e-8) + decay * w), m, v


def learning_rate(step, total, warmup, peak):
    if step < warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))


def checkpoint(model, optimizer, step):
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
