import json
import tempfile
from pathlib import Path

import torch

from learn_llm.checking import Case
from learn_llm.checks.ch10 import _plain


def _file() -> Path:
    """A tiny weight file in the real layout: a 16-bit vector, then a 32-bit table."""
    header = json.dumps({
        "__metadata__": {"format": "pt"},
        "gain": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]},
        "table": {"dtype": "F32", "shape": [2, 2], "data_offsets": [6, 22]},
    }).encode("utf-8")
    gain = torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
    table = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32).numpy().tobytes()
    path = Path(tempfile.mkdtemp()) / "tiny.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + gain + table)
    return path


def _header(f):
    path = _file()
    out = f(path)
    if not isinstance(out, (tuple, list)) or len(out) != 2 or not isinstance(out[0], dict):
        return _plain(out)
    return sorted(out[0]), out[0]["table"]["shape"], out[1] == 8 + int.from_bytes(path.read_bytes()[:8], "little")


def _tensor(f, name):
    path = _file()
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    out = f(path, json.loads(raw[8:8 + n])[name], 8 + n)
    return (_plain(out), str(out.dtype)) if isinstance(out, torch.Tensor) else _plain(out)


def _share(f):
    x = torch.arange(6.0).reshape(1, 3, 2, 1)
    out = f(x, 2)
    return (list(out.shape), out[0, :, 0, 0].tolist()) if isinstance(out, torch.Tensor) else out


def _generate(f, limit):
    def next_scores(ids):
        scores = torch.zeros(10)
        scores[(ids[-1] + 1) % 10] = 1.0
        return scores

    start = [3]
    return f(next_scores, start, limit, 7), start


EXERCISES = {
    "header_length": [
        Case("ch27.header_length.hand", lambda f: f(bytes([2, 1, 0, 0, 0, 0, 0, 0])), 258, hint="ch27.header_length.hint"),
        Case("ch27.header_length.real", lambda f: f(bytes([224, 138, 0, 0, 0, 0, 0, 0])), 35552, hint="ch27.header_length.hint"),
    ],
    "read_header": [
        Case("ch27.read_header.tiny", _header, (["__metadata__", "gain", "table"], [2, 2], True), hint="ch27.read_header.hint"),
    ],
    "read_tensor": [
        Case("ch27.read_tensor.table", lambda f: _tensor(f, "table"), ([[1.5, 2.5], [3.5, 4.5]], "torch.float32"), hint="ch27.read_tensor.hint"),
        Case("ch27.read_tensor.gain", lambda f: _tensor(f, "gain"), ([1.0, -2.0, 0.5], "torch.float32"), hint="ch27.read_tensor.hint"),
    ],
    "share_heads": [
        Case("ch27.share_heads.three", _share, ([1, 6, 2, 1], [0.0, 0.0, 2.0, 2.0, 4.0, 4.0]), hint="ch27.share_heads.hint"),
    ],
    "top_k_chances": [
        Case("ch27.top_k_chances.hand", lambda f: _plain(f(torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05], dtype=torch.float64)), 1.0, 2)), [0.625, 0.375, 0.0, 0.0], tolerance=1e-6, hint="ch27.top_k_chances.hint"),
        Case("ch27.top_k_chances.all", lambda f: _plain(f(torch.log(torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)), 1.0, 3)), [0.5, 0.25, 0.25], tolerance=1e-6, hint="ch27.top_k_chances.hint"),
        Case("ch27.top_k_chances.cold", lambda f: _plain(f(torch.log(torch.tensor([0.1, 0.6, 0.3], dtype=torch.float64)), 0.5, 2)), [0.0, 0.8, 0.2], tolerance=1e-6, hint="ch27.top_k_chances.hint"),
    ],
    "generate": [
        Case("ch27.generate.stop", lambda f: _generate(f, 10), ([3, 4, 5, 6], [3]), hint="ch27.generate.hint"),
        Case("ch27.generate.limit", lambda f: _generate(f, 2), ([3, 4, 5], [3]), hint="ch27.generate.hint"),
    ],
}
