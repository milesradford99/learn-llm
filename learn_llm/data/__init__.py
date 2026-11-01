"""Datasets bundled with the course, so every chapter works offline."""

import lzma
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def digits():
    """The bundled MNIST subset: (train_images, train_labels, test_images, test_labels).

    Images are uint8 arrays of shape (n, 28, 28); see MNIST-NOTICE.md for source and licence.
    """
    with np.load(HERE / "mnist-12k.npz", allow_pickle=False) as data:
        return data["train_images"], data["train_labels"], data["test_images"], data["test_labels"]


def stories():
    """The bundled TinyStories subset: (train_stories, test_stories), two lists of strings.

    The same text for every language edition; see TINYSTORIES-NOTICE.md for source and licence.
    """
    text = lzma.decompress((HERE / "tinystories-13k.txt.xz").read_bytes()).decode("utf-8")
    all_stories = text.strip().split("\n<|endoftext|>\n")
    return all_stories[:12000], all_stories[12000:]
