import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="20 · Many devices")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits
    from learn_llm.parallel import digit_model, train_in_processes, train_one_process

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 · Many devices

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

    ## Exercise 1: average_slopes

    Write `average_slopes(slopes)`. `slopes` is a list with one entry per device, and each entry is a list of tensors, one per knob.
    Give back one list of tensors: for each knob, the mean over devices. `torch.stack(..., dim=0)` (chapter 13) piles the devices' tensors up, and
    `torch.mean(..., dim=0)` averages over them.

    1. `averaged = []`
    2. `for k in range(len(slopes[0])):`, and inside it
       `averaged.append(torch.mean(torch.stack([device[k] for device in slopes], dim=0), dim=0))`
    3. After the loop, `return averaged`
    """)
    return


@app.function
def average_slopes(slopes):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "average_slopes", average_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's check the key fact on chapter 09's digit network, in one process. One copy computes the slopes of a batch of 128 pictures. Then 4
    "devices" each take 32 of those pictures, and we average their slopes. The cell shows the largest difference between the two.
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

    The same fact helps on a single device that can't fit a big batch in memory: run the quarters one after another, add up their
    slopes in `.grad` (PyTorch adds, remember chapter 09's trap), and divide by 4 before stepping. This is called **gradient accumulation**, and it makes any
    batch size possible on any device, at the cost of time.

    ## Exercise 2: accumulated_slopes

    Write `accumulated_slopes(model, X, Y, pieces)`. Cut the batch into `pieces` equal parts, run backward on each part's loss divided by `pieces`, and
    let `.grad` add them up. Give back a copy of the slopes.

    1. `model.zero_grad()`, and `size = len(X) // pieces`, on two lines
    2. `for i in range(pieces):`, and inside it two lines:
       `part = slice(i * size, (i + 1) * size)`, then
       `(F.cross_entropy(model(X[part]), Y[part]) / pieces).backward()`
    3. After the loop: `return [p.grad.clone() for p in model.parameters()]`

    (`slice(a, b)` is a slice kept in a name: `X[slice(a, b)]` is `X[a:b]`.)
    """)
    return


@app.function
def accumulated_slopes(model, X, Y, pieces):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "accumulated_slopes", accumulated_slopes, LOCALE)
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

    The obvious way, everyone sending everything to one device, makes that device a bottleneck. The usual way is a **ring**. Picture the devices in a circle,
    each cutting its slopes into as many chunks as there are devices, $n$ of them. In a first lap, all devices pass at the same moment, each one chunk to its neighbour, a different chunk each,
    and each adds what it receives to its own copy of that chunk. After $n - 1$ such passes, each device holds the full sum of one chunk. In a second lap of $n - 1$ passes, the finished chunks travel round so everyone gets them all.

    Count the cost. In each pass, a device sends one chunk, which is a fraction $1/n$ of the slopes. There are $2(n-1)$ passes in all, so each device sends

    > $$\text{bytes sent} = 2 \times \frac{n - 1}{n} \times \text{size of the slopes}$$

    That's less than twice the slopes, however many devices there are. Adding devices doesn't make each one's talking grow, which is why rings scale so well.

    ## Exercise 3: ring_bytes

    Write `ring_bytes(size, devices)`: how many bytes each device sends in a ring all-reduce of `size` bytes:

    ```
    return 2 * (devices - 1) / devices * size
    ```
    """)
    return


@app.function
def ring_bytes(size, devices):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "ring_bytes", ring_bytes, LOCALE)
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

    Data parallelism gives every device a full copy of everything. Chapter 19 counted what that is during training with Adam: the knobs, their slopes, and Adam's two
    averages, 4 numbers per knob, so 16 bytes per knob at 32 bits. For a model of 7 billion knobs, that's 112 GB on every device, more than most GPUs hold.

    But look at what each copy does with Adam's averages: each device updates *all* knobs, identically, in every step. That's wasted memory. **Sharding** splits the
    optimizer's averages between devices: device 0 keeps the averages for the first quarter of the knobs and updates only those, device 1 the second quarter, and so on.
    After the step, the devices swap their updated pieces so all copies are whole again. The same idea can shard the slopes, and even the knobs themselves.
    These levels are called **ZeRO** stages 1, 2 and 3.

    ## Exercise 4: shard

    Each device needs to know which knobs are its own. Write `shard(count, devices, rank)`, giving back the start and the stop of device `rank`'s share of
    `count` knobs, numbered from 0. `stop` is the first knob that isn't yours, as in a slice. When `count` doesn't divide evenly, the first devices take one extra each. `%` gives the remainder of a division: `10 % 4` is 2.

    1. Two lines: `size = count // devices` and `extra = count % devices`
    2. `start = rank * size + min(rank, extra)`
    3. `stop = start + size + (1 if rank < extra else 0)`
    4. `return start, stop`
    """)
    return


@app.function
def shard(count, devices, rank):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "shard", shard, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: bytes_per_device

    How much does sharding save? With 32-bit numbers, a knob costs 4 bytes for itself, 4 for its slope, and 8 for Adam's two averages. Write
    `bytes_per_device(knobs, devices, stage)`. Stage 0 is plain data parallelism, stage 1 shards Adam's averages, stage 2 also the slopes, and stage 3 also the knobs.

    1. `weights = 4 * knobs`, `slopes = 4 * knobs` and `adam = 8 * knobs`, on three lines
    2. `if stage >= 1:` with `adam = adam / devices` inside
    3. `if stage >= 2:` with `slopes = slopes / devices` inside
    4. `if stage >= 3:` with `weights = weights / devices` inside
    5. `return weights + slopes + adam`

    (Each `if` is lined up with the others, not inside the one before.)
    """)
    return


@app.function
def bytes_per_device(knobs, devices, stage):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "bytes_per_device", bytes_per_device, LOCALE)
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
    From 112 GB to 14 GB per device (stage 2 is really 38.5, shown rounded), just by not storing the same thing eight times. (The table uses `bytes_per_device(7e9, 8, stage)`, where `7e9` means $7 \times 10^9$.)
    Activations, the values kept for the backward pass, come on top of this, and they grow with the batch, not with the knobs.

    Sharding isn't free. In stage 3 the knobs of a layer must be gathered from the other devices just before that layer runs, every step. That's more talking, traded for memory.

    ## Splitting one layer

    For the very largest models, even one layer's matrix is split. Recall that `X @ W` computes each column of the result from one column of `W`. So split `W` by columns:
    device 0 holds the left half and computes the left half of the result, device 1 the right half, and the halves are joined. This is **tensor parallelism**.
    The cell below checks the idea on random matrices:
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    X_demo = torch.rand(5, 8)
    W_demo = torch.rand(8, 6)
    left = X_demo @ W_demo[:, :3]
    right = X_demo @ W_demo[:, 3:]
    torch.max(torch.abs(torch.cat([left, right], dim=1) - X_demo @ W_demo)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About $10^{-7}$: the same numbers, added in a different order, so only the last digits differ. Real systems combine all of these: data parallelism across groups of machines, sharding within them, and tensor parallelism across the GPUs inside one machine,
    where links are fastest. Choosing the mix is a large part of engineering a training run.

    ## What's next

    You can now say how much a model costs to train and how the work is split. **Chapter 21** turns this into planning: if you have a budget for a training run,
    how big should the model be, and how much data should it read? The answer follows a law you'll measure from your own small runs.

    ## What you can do now

    - explain why averaging slopes from equal slices gives the whole batch's slopes
    - use gradient accumulation to reach any batch size on one device
    - explain all-reduce and count a ring's bytes
    - train on several real processes and check the result matches one process
    - work out memory per device for each level of sharding
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
