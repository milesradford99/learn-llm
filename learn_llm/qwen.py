"""Plumbing for chapter 27: fetch a real published model, speak its tokenizer, name its knobs.

The model is Qwen3-0.6B (Apache-2.0, see data/QWEN-NOTICE.md), pinned to one revision and
checked by SHA-256 after download. Nothing here runs the model: the lesson does that, with the
learner's own reader and the pieces they wrote in chapter 20.
"""

import contextlib
import hashlib
import json
import os
import re
import unicodedata
import urllib.request
from pathlib import Path

import torch

from learn_llm.course import WORK

REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BASE = f"https://huggingface.co/Qwen/Qwen3-0.6B/resolve/{REVISION}/"
FILES = {  # name -> (bytes, SHA-256)
    "config.json": (726, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "model.safetensors": (1503300328, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
}
# Set LEARN_LLM_QWEN to a folder that already holds the three files to use it instead.
FOLDER = Path(os.environ["LEARN_LLM_QWEN"]) if os.environ.get("LEARN_LLM_QWEN") else WORK / ".models" / "qwen3-0.6b"
# A weight file's names for kinds of numbers, and PyTorch's.
TYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64, "I64": torch.int64}
END_OF_TURN = 151645  # <|im_end|>


def _pieces() -> re.Pattern:
    """The tokenizer's own splitting rule, with \\p{L} and \\p{N} spelled for Python's `re`:
    `\\d` misses numbers like ½ and ², so those are listed."""
    more = re.escape("".join(c for c in map(chr, range(0x110000)) if unicodedata.category(c)[0] == "N" and not c.isdecimal()))
    letter, number, other = rf"[^\W\d_{more}]", rf"[\d{more}]", r"(?:[^\s\w]|_)"
    return re.compile(
        rf"(?i:'s|'t|'re|'ve|'m|'ll|'d)|(?:[^\r\n\w]|_)?{letter}+|{number}| ?{other}+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def missing(folder: Path = FOLDER) -> list[str]:
    """The model's files that aren't in `folder` yet, or aren't complete."""
    return [name for name, (size, _) in FILES.items() if not (folder / name).exists() or (folder / name).stat().st_size != size]


def _bar(name: str, size: int):
    """A progress bar on the lesson's page, counting megabytes. Outside a notebook, nothing."""
    import marimo as mo

    if not mo.running_in_notebook():
        return contextlib.nullcontext()
    return mo.status.progress_bar(total=max(size >> 20, 1), title=name, subtitle="MB", show_rate=False, show_eta=False, remove_on_exit=True)


def fetch(folder: Path = FOLDER, report=None) -> Path:
    """Download the model's files into `folder` (1.5 GB the first time), continuing a download
    that was cut off, and check each file against its SHA-256. The page shows a progress bar,
    and `report(name, done, size)` is called as bytes arrive."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in missing(folder):
        size, sha = FILES[name]
        part = folder / (name + ".part")
        have = part.stat().st_size if part.exists() else 0
        if have < size:
            request = urllib.request.Request(BASE + name, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(request, timeout=60) as response, open(part, "ab" if response.status == 206 else "wb") as out, _bar(name, size) as bar:
                done = shown = have if response.status == 206 else 0
                if bar and done >> 20:
                    bar.update(increment=done >> 20)
                while chunk := response.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    if bar and (done >> 20) > (shown >> 20):
                        bar.update(increment=(done >> 20) - (shown >> 20))
                        shown = done
                    if report:
                        report(name, done, size)
        if _sha256(part) != sha:
            part.unlink()
            raise OSError(f"{name} arrived damaged; run the download again")
        part.rename(folder / name)
    return folder


def config(folder: Path = FOLDER) -> dict:
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def layer_knobs(knobs: dict, layer: int) -> dict:
    """The knobs of one block under short names: norm1, q, k, v, q_norm, k_norm, out, norm2,
    gate, up, down. `knobs` maps the file's full names to tensors."""
    prefix = f"model.layers.{layer}."
    names = {
        "norm1": "input_layernorm", "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
        "q_norm": "self_attn.q_norm", "k_norm": "self_attn.k_norm", "out": "self_attn.o_proj",
        "norm2": "post_attention_layernorm", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj",
    }
    return {short: knobs[prefix + long + ".weight"] for short, long in names.items()}


class QwenTokenizer:
    """The model's byte-level BPE tokenizer: chapter 19's method with about 151,000 merges."""

    def __init__(self, spec: dict):
        self.ids = spec["model"]["vocab"]
        self.rank = {tuple(pair): i for i, pair in enumerate(spec["model"]["merges"])}
        self.special = {t["content"]: t["id"] for t in spec["added_tokens"]}
        self.texts = {i: t for t, i in {**self.ids, **self.special}.items()}
        # Byte-level BPE writes each byte as one printable character; this is that table.
        shown = [b for b in range(256) if 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255]
        extra = iter(range(256, 512))
        self.byte_char = {b: chr(b) if b in shown else chr(next(extra)) for b in range(256)}
        self.char_byte = {c: b for b, c in self.byte_char.items()}
        self._special_split = re.compile("(" + "|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True)) + ")")
        self._cache: dict[str, list[int]] = {}
        self._pieces = _pieces()

    @classmethod
    def load(cls, folder: Path = FOLDER) -> "QwenTokenizer":
        return cls(json.loads((folder / "tokenizer.json").read_text(encoding="utf-8")))

    def _piece(self, piece: str) -> list[int]:
        if piece not in self._cache:
            parts = [self.byte_char[b] for b in piece.encode("utf-8")]
            while len(parts) > 1:
                best = min(range(len(parts) - 1), key=lambda i: self.rank.get((parts[i], parts[i + 1]), 1 << 30))
                if (parts[best], parts[best + 1]) not in self.rank:
                    break
                parts[best:best + 2] = [parts[best] + parts[best + 1]]
            self._cache[piece] = [self.ids[p] for p in parts]
        return self._cache[piece]

    def encode(self, text: str) -> list[int]:
        out = []
        for chunk in self._special_split.split(unicodedata.normalize("NFC", text)):
            if chunk in self.special:
                out.append(self.special[chunk])
            else:
                for piece in self._pieces.findall(chunk):
                    out.extend(self._piece(piece))
        return out

    def decode(self, ids: list[int]) -> str:
        out, held = [], bytearray()
        for i in ids:
            if i in self.ids.values() or self.texts[i] not in self.special:
                held.extend(self.char_byte[c] for c in self.texts[i])
            else:
                out.append(held.decode("utf-8", errors="replace") + self.texts[i])
                held = bytearray()
        return "".join(out) + held.decode("utf-8", errors="replace")

    def chat(self, question: str) -> list[int]:
        """A question wrapped the way the model was trained to see one, ready for its answer."""
        return self.encode(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
