import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="23 · Many devices")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits
    from learn_llm.diagrams import ring
    from learn_llm.parallel import digit_model, train_in_processes, train_one_process

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 23 · Many devices

    A large model is trained on thousands of GPUs at once, for weeks. No single device could do it: it would take centuries, and
    the model often wouldn't even fit in one device's memory. This chapter is about how the work is split up.

    You don't have thousands of GPUs, and you don't need them to understand the ideas. Your computer's processor has several
    **cores**, and the operating system can run several **processes**, separate running programs that don't share memory, exactly like separate
    GPUs. By the end, you'll have trained a network on several processes at once, and checked that it learns exactly what one process would.

    There are two problems to solve, and they have different answers:

    - **Too slow:** split the *data*, so each device does part of every batch.
    - **Too big:** split the *model*, so each device holds part of the knobs.

    ## Splitting the data

    Recall what one step does: compute the loss on a batch, which is a **mean** over its examples, then the slopes, then move the knobs.
    Now give 4 devices a full copy of the model each, and a quarter of the batch each. Each computes the slopes of the mean over *its* quarter.

    Here's the key fact. When the quarters are the same size, the mean of the four quarter-means is the mean of the whole batch: each quarter-mean is its total divided by 32,
    and $\frac{1}{4}\big(\frac{a}{32} + \frac{b}{32} + \frac{c}{32} + \frac{d}{32}\big) = \frac{a + b + c + d}{128}$, where $a, b, c, d$ are the quarters' totals. Slopes follow the loss (chapter 02's sum and scale rules), so **averaging the four devices' slopes gives exactly the slopes of the
    whole batch**. If every device then applies the same step to the same knobs, the copies stay identical, step after step.

    This is **data parallelism**. The only communication: after each backward pass, every device must end up with the average of all the slopes.

    ## By hand: two halves and the whole

    A batch of 4 examples has the losses 2, 4, 6 and 12. Device A takes the first two and device B the last two. What's the mean loss of each device's half? What's the mean of those two means?
    And what's the mean loss of the whole batch?
    """)
    return


@app.cell(hide_code=True)
def _():
    mean_boxes = hand.boxes(["A's mean", "B's mean", 'mean of the two means', 'mean of the whole batch'])
    return (mean_boxes,)


@app.cell(hide_code=True)
def _(mean_boxes):
    hand.marked(mean_boxes, {"A's mean": 3, "B's mean": 9, 'mean of the two means': 6, 'mean of the whole batch': 6}, {
        "A's mean": r"""$(2 + 4) / 2$""",
        "B's mean": r"""$(6 + 12) / 2$""",
        'mean of the two means': r"""add your first two answers and halve""",
        'mean of the whole batch': r"""$(2 + 4 + 6 + 12) / 4$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same number, because the halves are the same size. Had A taken three examples and B one, the mean of the two means would be 8, not 6: the lone example would count three times too much.

    ## Exercise 1: average_slopes

    Write `average_slopes(slopes)`. `slopes` is a list with one entry per device, and each entry is itself a list of tensors, one per knob: `slopes[0][1]` is device 0's slopes for knob 1.
    All devices have the same knobs, so the inner lists have the same length, and matching tensors have the same shape. It gives back one list of tensors: for each knob, the mean over the devices.

    With 2 devices and 2 knobs, `[[[1, 2], [[0]]], [[3, 6], [[4]]]]` as tensors gives `[[2, 4], [[2]]]`.

    You have the tools: `torch.stack(tensors, dim=0)` (chapter 14) piles tensors of the same shape up along a new first dimension, and `torch.mean(..., dim=0)` takes the mean along it.
    """)
    return


@app.function
def average_slopes(slopes):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "average_slopes", average_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Loop over the knobs, not the devices: `for k in range(len(slopes[0])):`. For knob `k`, collect each device's tensor in a list, stack the list, and take the mean. Append the result to a list that you give back at the end.""",
        r"""It's 4 lines. The list of knob `k` from every device is `[device[k] for device in slopes]`. Both `torch.stack` and `torch.mean` need `dim=0`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's check the key fact on chapter 10's digit network, in one process. One copy computes the slopes of a batch of 128 pictures. Then 4
    "devices" each take 32 of those pictures, and we average their slopes. The cell shows the largest difference between the two.
    (It needs Exercise 1, and shows an error until that passes. That's normal, and the same goes for later cells that use your functions.)
    """)
    return


@app.cell
def _():
    images, labels, _test_images, _test_labels = digits()
    X = torch.tensor(images[:128].reshape(128, 784) / 255, dtype=torch.float32)
    Y = torch.tensor(labels[:128], dtype=torch.long)

    def slopes_of(model, X_part, Y_part):
        model.zero_grad()
        F.cross_entropy(model(X_part), Y_part).backward()
        return [p.grad.clone() for p in model.parameters()]

    model = digit_model(0)
    whole = slopes_of(model, X, Y)
    quarters = [slopes_of(model, X[i * 32:(i + 1) * 32], Y[i * 32:(i + 1) * 32]) for i in range(4)]
    max([torch.max(torch.abs(a - b)).item() for a, b in zip(whole, average_slopes(quarters))])
    return X, Y, model, slopes_of


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A difference under $10^{-8}$ (it prints as something like `5.6e-09`, which means $5.6 \times 10^{-9}$): rounding, nothing more. The four quarters together *are* the whole batch.

    ## Gradient accumulation: the same trick on one device

    The same fact helps on a single device that can't fit a big batch in memory: run the quarters one after another, and add up their
    slopes in `.grad`. PyTorch adds there by itself, remember chapter 10's trap: this time the trap is what we want. One thing is left: adding four quarter-means gives four times too much. So each quarter's loss is divided by 4 *before* going backward, and
    the slopes that pile up are those of the mean of the four. This is called **gradient accumulation**, and it makes any
    batch size possible on any device, at the cost of time.

    /// admonition | 🧰 Toolbox: a slice with a name
    `slice(a, b)` is a slice kept as a value: with `part = slice(2, 5)`, `X[part]` is `X[2:5]`. It's handy when the same slice is taken from two arrays, like the pictures and their labels.
    ///

    ## Exercise 2: accumulated_slopes

    Write `accumulated_slopes(model, X, Y, pieces)`. `model` is a PyTorch model, `X` a batch of inputs, `Y` their labels, and `pieces` divides the batch size evenly.
    It gives back the slopes of the whole batch's `F.cross_entropy` loss, as a list with a copy of each knob's `.grad`, without ever running the model on more than one piece:

    1. Clear the model's slopes once, with `model.zero_grad()`.
    2. For each piece in turn, take that piece's rows of `X` and `Y`, compute its loss, divide the loss by `pieces`, and go backward.
    3. Give back `[p.grad.clone() for p in model.parameters()]`: `.clone()` makes a copy that later steps can't change.
    """)
    return


@app.function
def accumulated_slopes(model, X, Y, pieces):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "accumulated_slopes", accumulated_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The size of a piece is `len(X) // pieces`. Piece `i` starts at `i * size` and stops before `(i + 1) * size`: keep that slice in a name, and use it on both `X` and `Y`.""",
        r"""It's 6 lines. Inside the loop, the loss needs brackets around it before `.backward()`: `(F.cross_entropy(model(X[part]), Y[part]) / pieces).backward()`. If the check fails by a factor of `pieces`, the division is missing. If `zero_grad` is inside the loop, only the last piece counts.""",
    ], LOCALE)
    return


@app.cell
def _(X, Y, model, slopes_of):
    one_go = slopes_of(model, X, Y)
    in_four = accumulated_slopes(model, X, Y, 4)
    max([torch.max(torch.abs(a - b)).item() for a, b in zip(one_go, in_four)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Again a rounding-sized difference.

    ## Talking: all-reduce

    On real devices, "average the slopes" means sending numbers between them. The standard operation is called **all-reduce**: every device starts with
    its own tensor, and every device ends with the sum (or the mean) of all of them.

    The obvious way, everyone sending everything to one device, makes that device a bottleneck. The usual way is a **ring**. Picture the devices in a circle, each passing only to its neighbour:
    """)
    return


@app.cell(hide_code=True)
def _():
    ring(['device A', 'device B', 'device C'], 'Three devices in a circle. A passes to B, B passes to C, and C passes to A.')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each device cuts its slopes into as many chunks as there are devices, $n$ of them. All devices pass at the same moment, one chunk each, a different chunk each,
    and each adds what it receives to its own copy of that chunk. After $n - 1$ such passes, the first lap, each device holds the full sum of one chunk. In a second lap of $n - 1$ passes, the finished chunks travel round so everyone gets them all.

    Here's the first lap for 3 devices whose slopes are 3 numbers, one number per chunk. A starts with $[1, 2, 3]$, B with $[10, 20, 30]$ and C with $[100, 200, 300]$:

    | pass | A sends to B | B sends to C | C sends to A |
    |---|---|---|---|
    | 1 | its chunk 0, which is 1 | its chunk 1, which is 20 | its chunk 2, which is 300 |
    | 2 | the chunk 2 it has just added up | the chunk 0 it has just added up | the chunk 1 it has just added up |

    ## By hand: one lap of the ring

    Follow the table. After pass 1, what does B hold in chunk 0? After pass 2, what does C hold in chunk 0, A in chunk 1, and B in chunk 2?
    The second lap then takes 2 more passes, each device sending one finished chunk along each time. Over all four passes, how many numbers does each device send?
    """)
    return


@app.cell(hide_code=True)
def _():
    ring_boxes = hand.boxes(["B's chunk 0 after pass 1", "C's chunk 0 after pass 2", "A's chunk 1 after pass 2", "B's chunk 2 after pass 2", 'numbers each device sends'])
    return (ring_boxes,)


@app.cell(hide_code=True)
def _(ring_boxes):
    hand.marked(ring_boxes, {"B's chunk 0 after pass 1": 11, "C's chunk 0 after pass 2": 111, "A's chunk 1 after pass 2": 222, "B's chunk 2 after pass 2": 333, 'numbers each device sends': 4}, {
        "B's chunk 0 after pass 1": r"""B's own chunk 0 is 10, and A sends it 1""",
        "C's chunk 0 after pass 2": r"""C's own chunk 0 is 100, and B sends on the 11 it has just made""",
        "A's chunk 1 after pass 2": r"""after pass 1, C's chunk 1 is $200 + 20$. In pass 2 C sends that to A, whose own chunk 1 is 2.""",
        "B's chunk 2 after pass 2": r"""after pass 1, A's chunk 2 is $3 + 300$. In pass 2 A sends that to B, whose own chunk 2 is 30.""",
        'numbers each device sends': r"""one chunk of one number in each of the 4 passes""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After the first lap, the full sums 111, 222 and 333 exist, each on one device. The second lap hands them round. Every device sent 4 numbers, to share slopes of 3 numbers among 3 devices.

    Count the cost in general. In each pass, a device sends one chunk, which is a fraction $1/n$ of the slopes. There are $2(n-1)$ passes in all, so each device sends

    > $$\text{bytes sent} = 2 \times \frac{n - 1}{n} \times \text{size of the slopes}$$

    For the hand example that's $2 \times \tfrac{2}{3} \times 3 = 4$, as you counted. It's always less than twice the slopes, however many devices there are. Adding devices doesn't make each one's talking grow, which is why rings scale so well.

    ## Exercise 3: ring_bytes

    Write `ring_bytes(size, devices)`, that formula: how many bytes each device sends in a ring all-reduce of slopes that take `size` bytes. `ring_bytes(3, 3)` gives `4.0`, and `ring_bytes(1000, 4)` gives `1500.0`.
    """)
    return


@app.function
def ring_bytes(size, devices):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "ring_bytes", ring_bytes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line. The fraction is the number of devices less one, over the number of devices.""",
        r"""Put brackets around `devices - 1`, or Python divides only the 1.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Real processes

    Now for real. The function `train_in_processes` (it's in the course's own code, `learn_llm/parallel.py`, because a new process must be able to load it) starts several separate
    processes. Each one builds the digit network with the same seed, so all copies start identical. On every step, each takes its own slice of the batch, computes its slopes,
    and calls PyTorch's `dist.all_reduce`, which adds the slopes across processes, then divides by the number of processes, and then takes the step. The processes share no memory at all: they talk
    only through all-reduce, over the computer's internal network, the way GPUs in a machine talk over their links.

    The button trains for 200 steps of 128 pictures in one process, and then in 2 and in 4 processes, and compares their final knobs with the one-process knobs.
    The first row is one process on its own: the processes, the seconds. Each later row shows: the processes, the largest difference in any knob, the bytes of slopes each process handed to all-reduce, and the seconds.
    It takes up to a minute, and the page may look frozen meanwhile. The result may show as a list you can unfold.
    """)
    return


@app.cell(hide_code=True)
def _():
    run_button = mo.ui.run_button(label="Train on several processes (up to a minute)")
    run_button
    return (run_button,)


@app.cell
def _(run_button):
    mo.stop(not run_button.value, mo.md("*Press the button to run.*"))
    started = time.perf_counter()
    alone = train_one_process(200, 128, 0.5, 0)
    runs = [(1, round(time.perf_counter() - started, 1))]
    for count in [2, 4]:
        knobs, sent, seconds = train_in_processes(count, 200, 128, 0.5, 0)
        biggest = max([torch.max(torch.abs(a - b)).item() for a, b in zip(alone, knobs)])
        runs.append((count, biggest, sent, round(seconds, 1)))
    runs
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The differences are a few times $10^{-7}$: after 200 steps on separate processes, the knobs are the same as from one process, up to rounding. (Adding
    numbers in a different order changes the last digits, so exact zeros are not to be expected.) Data parallelism changes *where* the work is done, not *what* is learned.

    The byte count is about 40 million either way: $200 \text{ steps} \times 50{,}890 \text{ slopes} \times 4 \text{ bytes}$. That's what each process hands to all-reduce. In a ring, each would send about 41 million bytes with 2 processes and 61 million with 4 (PyTorch may choose another method on one computer).

    The seconds are the honest part. On one computer, processes compete for the same cores and the same memory, and starting a process takes a moment, so more
    processes may not be faster at all, and for a model this small they usually aren't. The idea pays off when each device is a separate GPU with its own memory, and
    the model is big enough that computing dwarfs talking.

    ## Too big: splitting the model

    Data parallelism gives every device a full copy of everything. Chapter 22 counted what that is during training with Adam: the knobs, their slopes, and Adam's two
    averages, 4 numbers per knob, so 16 bytes per knob at 32 bits. For a model of 7 billion knobs, that's 112 GB on every device, more than most GPUs hold.

    But look at what each copy does with Adam's averages: each device updates *all* knobs, identically, in every step. That's wasted memory. **Sharding** splits the
    optimizer's averages between devices: device 0 keeps the averages for the first quarter of the knobs and updates only those, device 1 the second quarter, and so on.
    After the step, the devices swap their updated pieces so all copies are whole again. The same idea can shard the slopes, and even the knobs themselves.
    These levels are called **ZeRO** stages 1, 2 and 3.

    Each device needs to know which knobs are its own. Devices are numbered from 0, and a device's number is called its **rank**. When the knobs don't divide evenly, the first devices take one extra each.

    /// admonition | 🧰 Toolbox: what's left over
    `a % b` is the **remainder** of dividing `a` by `b`, the partner of `//`: `10 // 4` is 2 and `10 % 4` is 2, because $10 = 4 \times 2 + 2$. `12 % 4` is 0.
    ///

    ## By hand: 10 knobs on 4 devices

    Share out 10 knobs, numbered 0 to 9, among 4 devices, in order. How many does each device get? Then for device 2: which knob is its first, and what is its stop, the first knob that isn't its own any more (as in a slice)?
    """)
    return


@app.cell(hide_code=True)
def _():
    shard_boxes = hand.boxes(['device 0 gets', 'device 1 gets', 'device 2 gets', 'device 3 gets', "device 2's start", "device 2's stop"])
    return (shard_boxes,)


@app.cell(hide_code=True)
def _(shard_boxes):
    hand.marked(shard_boxes, {'device 0 gets': 3, 'device 1 gets': 3, 'device 2 gets': 2, 'device 3 gets': 2, "device 2's start": 6, "device 2's stop": 8}, {
        'device 0 gets': r"""everyone gets `10 // 4`, and `10 % 4` knobs are left over: the first devices take one each""",
        'device 1 gets': r"""it's one of the first two devices, so it takes an extra one too""",
        'device 2 gets': r"""the two extras are gone, so just `10 // 4`""",
        'device 3 gets': r"""the same as device 2""",
        "device 2's start": r"""devices 0 and 1 took 3 each before it""",
        "device 2's stop": r"""its start plus the number it gets""",
    }, LOCALE, per_row=4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: shard

    Write `shard(count, devices, rank)`. It gives back the pair: the start and the stop of device `rank`'s share of `count` knobs, numbered from 0.

    > `shard(10, 4, 2)` gives `(6, 8)`, and `shard(10, 4, 0)` gives `(0, 3)`

    Every device gets `count // devices` knobs, and the first `count % devices` devices get one more. A device's start is everything the devices before it took:
    their plain shares, plus one for each of them that took an extra. `min(a, b)` (chapter 12) and `a if test else b` (chapter 18) are useful here.
    """)
    return


@app.function
def shard(count, devices, rank):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "shard", shard, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Name the two numbers first: `size = count // devices` and `extra = count % devices`. Before device `rank` come `rank` devices. How many of them took an extra? All of them if `rank` is small, but never more than `extra`: that's `min(rank, extra)`.""",
        r"""It's 5 lines. The start is `rank * size + min(rank, extra)`. The stop is the start, plus `size`, plus 1 if this device is one of the first `extra` devices, which is the test `rank < extra`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How much does sharding save?

    With 32-bit numbers, a knob costs 4 bytes for itself, 4 for its slope, and 8 for Adam's two averages. Whatever is sharded costs each device only its share, the amount divided by the number of devices:

    | stage | the knobs | the slopes | Adam's averages |
    |---|---|---|---|
    | 0, plain data parallelism | whole | whole | whole |
    | 1 | whole | whole | sharded |
    | 2 | whole | sharded | sharded |
    | 3 | sharded | sharded | sharded |

    ## By hand: the four stages

    Take 1,000 knobs on 4 devices. Whole, the knobs take 4,000 bytes, the slopes 4,000 and the averages 8,000. How many bytes does one device need at each stage?
    """)
    return


@app.cell(hide_code=True)
def _():
    stage_boxes = hand.boxes(['stage 0', 'stage 1', 'stage 2', 'stage 3'])
    return (stage_boxes,)


@app.cell(hide_code=True)
def _(stage_boxes):
    hand.marked(stage_boxes, {'stage 0': 16000, 'stage 1': 10000, 'stage 2': 7000, 'stage 3': 4000}, {
        'stage 0': r"""$4000 + 4000 + 8000$""",
        'stage 1': r"""the averages shrink to $8000 / 4$""",
        'stage 2': r"""the slopes shrink to $4000 / 4$ as well""",
        'stage 3': r"""all three are divided by 4""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: bytes_per_device

    Write `bytes_per_device(knobs, devices, stage)`: the bytes one device needs for a model of `knobs` knobs shared among `devices` devices, at stage 0, 1, 2 or 3 of the table.
    Your hand answers are the check: `bytes_per_device(1000, 4, 2)` gives `7000.0`.

    Note that each stage shards everything the stage before it does, and one thing more. `a >= b` means "a is at least b".

    The table after the check uses your function for 7 billion knobs on 8 devices, written `7e9` as in chapter 22.
    """)
    return


@app.function
def bytes_per_device(knobs, devices, stage):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "bytes_per_device", bytes_per_device, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Start with the three whole amounts in three names. Then three separate `if`s, one for each column of the table, each dividing one amount by `devices`. Add the three at the end.""",
        r"""The `if`s are lined up with each other, not one inside the other, and they test `stage >= 1`, `stage >= 2` and `stage >= 3`. At stage 3 all three tests hold, so all three amounts are divided.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    table = ["| stage | GB per device, 7 billion knobs on 8 devices |", "|---|---|"]
    for stage in range(4):
        table.append(f"| {stage} | {bytes_per_device(7e9, 8, stage) / 1e9:.0f} |")
    mo.md("\n".join(table))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    From 112 GB to 14 GB per device (stage 2 is really 38.5, shown rounded), just by not storing the same thing eight times.
    Activations, the values kept for the backward pass, come on top of this, and they grow with the batch, not with the knobs.

    Sharding isn't free. In stage 3 the knobs of a layer must be gathered from the other devices just before that layer runs, every step. That's more talking, traded for memory.

    ## Splitting one layer

    For the very largest models, even one layer's matrix is split. Recall that `X @ W` computes each column of the result from one column of `W`. So split `W` by columns:
    with two devices, device 0 holds the left half and computes the left half of the result, device 1 the right half, and the halves are joined side by side. This is **tensor parallelism**.

    ## Exercise 6: split_product

    Write `split_product(X, W, devices)`. `X` has shape `(n, k)` and `W` has shape `(k, m)`. It gives back `X @ W`, of shape `(n, m)`, but computed in `devices` pieces:
    each device multiplies `X` by its own share of `W`'s columns, and the pieces are joined in order.

    Your `shard` says which columns belong to which device: there are `W.shape[1]` columns to share out. `W[:, start:stop]` takes columns (chapter 14's `:` means all the rows),
    and `torch.cat(pieces, dim=1)` (chapter 14) joins a list of tables side by side.

    The cell after the check compares your result with the plain product, on random matrices split over 3 devices.
    """)
    return


@app.function
def split_product(X, W, devices):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "split_product", split_product, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""An empty list, then a loop over the ranks, `for rank in range(devices):`. Each turn asks `shard` for that device's start and stop, multiplies, and appends the piece. After the loop, join the pieces.""",
        r"""It's 6 lines. `shard` gives back a pair, so name both halves: `start, stop = shard(W.shape[1], devices, rank)`. The piece is `X @ W[:, start:stop]`.""",
    ], LOCALE)
    return


@app.cell
def _():
    torch.manual_seed(0)
    X_demo = torch.rand(5, 8)
    W_demo = torch.rand(8, 7)
    torch.max(torch.abs(split_product(X_demo, W_demo, 3) - X_demo @ W_demo)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Zero, or something like $10^{-7}$: the same numbers, perhaps added in a different order, so that at most the last digits differ. Real systems combine all of these: data parallelism across groups of machines, sharding within them, and tensor parallelism across the GPUs inside one machine,
    where links are fastest. Choosing the mix is a large part of engineering a training run.

    ## What's next

    You can now say how much a model costs to train and how the work is split. **Chapter 24** turns this into planning: if you have a budget for a training run,
    how big should the model be, and how much data should it read? The answer follows a law you'll measure from your own small runs.

    ## What you can do now

    - explain why averaging slopes from equal slices gives the whole batch's slopes
    - use gradient accumulation to reach any batch size on one device
    - explain all-reduce, follow a ring by hand, and count its bytes
    - train on several real processes and check the result matches one process
    - share knobs out among devices, and work out memory per device for each level of sharding
    - split a matrix product across devices

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** PyTorch's `DistributedDataParallel` does what our processes do, overlapping talking with the backward pass. ZeRO is from Rajbhandari et al. (2019),
    and *FSDP* is PyTorch's version of stage 3. Tensor parallelism is described in the *Megatron-LM* paper, and splitting a model's layers into groups on different devices is called
    *pipeline parallelism*.
    """)
    return


if __name__ == "__main__":
    app.run()
