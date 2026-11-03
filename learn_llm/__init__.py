"""Shared code for the learn-llm course: checks, translations and the launcher."""


def todo():
    """Placeholder in exercise stubs. Replace the `todo()` line with your own code."""
    raise NotImplementedError("not written yet")


def laptop_speed():
    """Tune PyTorch for a laptop CPU; lessons call this once, in their hidden setup."""
    import os

    import torch

    # One thread per core, not per hyperthread: more makes convolutions several times slower.
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    # Fading values in deep networks would otherwise crawl through the CPU's slow path.
    torch.set_flush_denormal(True)
