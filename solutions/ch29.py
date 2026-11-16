"""Reference solutions for chapter 29. Peek only after trying!"""

import torch


def blocks_for(length, block_size):
    return (length + block_size - 1) // block_size


def slot(table, position, block_size):
    return table[position // block_size], position % block_size


def read_cache(pool, table, length):
    joined = torch.cat([pool[b] for b in table], dim=1)
    return joined[:, :length][None]


def admit(waiting, running, free_blocks, max_running, block_size):
    while len(waiting) > 0 and len(running) < max_running and blocks_for(len(waiting[0]["ids"]), block_size) <= free_blocks:
        free_blocks = free_blocks - blocks_for(len(waiting[0]["ids"]), block_size)
        running = running + [waiting[0]]
        waiting = waiting[1:]
    return waiting, running


def chunk_sizes(unfed, budget):
    sizes = []
    for n in unfed:
        size = min(n, budget)
        sizes.append(size)
        budget = budget - size
    return sizes
