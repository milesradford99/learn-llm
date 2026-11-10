"""Real data-parallel training on separate CPU processes, for chapter 20.

The work function lives here, not in a notebook cell, because each new process starts fresh
and must be able to import it. Every process holds a full copy of chapter 09's digit network,
takes its own slice of each batch, and averages its slopes with the others' before each step.
"""

import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn

PORT = 29531


def digit_model(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))


def _data():
    from learn_llm.data import digits

    images, labels, _, _ = digits()
    return torch.tensor(images.reshape(-1, 784) / 255, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def _slice(batch: torch.Tensor, rank: int, workers: int) -> torch.Tensor:
    size = len(batch) // workers
    return batch[rank * size:(rank + 1) * size]


def train_one_process(steps: int, batch_size: int, rate: float, seed: int):
    """The reference: one process, whole batches."""
    X, Y = _data()
    model = digit_model(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        batch = torch.randperm(len(X), generator=generator)[:batch_size]
        optimizer.zero_grad()
        F.cross_entropy(model(X[batch]), Y[batch]).backward()
        optimizer.step()
    return [p.detach().clone() for p in model.parameters()]


def _worker(rank, workers, steps, batch_size, rate, seed, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(PORT)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=workers)
    X, Y = _data()
    model = digit_model(seed)  # same seed in every process: identical starting copies
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    generator = torch.Generator().manual_seed(seed)  # same order everywhere, so slices don't overlap
    sent = 0
    for step in range(steps):
        batch = _slice(torch.randperm(len(X), generator=generator)[:batch_size], rank, workers)
        optimizer.zero_grad()
        F.cross_entropy(model(X[batch]), Y[batch]).backward()
        for p in model.parameters():
            dist.all_reduce(p.grad)  # every process ends up with the sum of all slopes
            p.grad /= workers
            sent += p.grad.numel() * 4
        optimizer.step()
    if rank == 0:
        # Plain NumPy arrays: a tensor sent between processes would share memory with this
        # process, which is about to exit.
        results.put(([p.detach().numpy().copy() for p in model.parameters()], sent))
    dist.destroy_process_group()


def train_in_processes(workers: int, steps: int, batch_size: int, rate: float, seed: int):
    """Train on `workers` separate processes. Gives back the final knobs, the bytes of slopes each
    process put into all-reduce, and the seconds taken."""
    context = mp.get_context("spawn")
    results = context.Queue()
    started = time.perf_counter()
    processes = [
        context.Process(target=_worker, args=(rank, workers, steps, batch_size, rate, seed, results))
        for rank in range(workers)
    ]
    for process in processes:
        process.start()
    knobs, sent = results.get(timeout=600)
    for process in processes:
        process.join()
    return [torch.tensor(k) for k in knobs], sent, time.perf_counter() - started
