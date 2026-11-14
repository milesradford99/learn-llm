import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="22 · Making it fast")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 22 · Making it fast

    Training a model means doing an enormous amount of arithmetic. Making it fast isn't about clever tricks first: it's about **measuring**, so you know where
    the time actually goes, then **counting**, so you know how fast it could possibly be. This chapter does both, then tries two speed-ups and finds that one of them
    may make things slower, as it did on the computer this chapter was tested on.

    ## Measuring time

    Measuring one run of anything is unreliable: the first run is often slow while the computer gets ready,
    and every run varies a little. So we run once to warm up, then time several runs and take the mean.

    /// admonition | 🧰 Toolbox: a clock
    `time.perf_counter()` gives the time in seconds, counted from some arbitrary start. One reading means nothing, but the difference between two readings is
    how long the lines between them took: `started = time.perf_counter()`, then the work, then `time.perf_counter() - started`.
    ///

    ## Exercise 1: time_it

    Write `time_it(f, repeats)`. `f` is a function with no inputs and `repeats` a whole number. It calls `f` once to warm up, without timing that call,
    then calls it `repeats` more times, and gives back the mean number of seconds one of those calls took.

    One of the checks hands in a function whose first call takes 0.1 seconds and later ones 0.01: your answer must be 0.01.
    """)
    return


@app.function
def time_it(f, repeats):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "time_it", time_it, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Four parts: the warm-up call, a reading of the clock, a loop that calls `f` again and again, and the `return`. The clock is read after the warm-up, so the slow first call isn't counted.""",
        r"""It's 5 lines. The last gives back the time that passed since `started`, divided by `repeats`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Counting arithmetic

    The work in a network is almost all matrix products. Each number of `A @ B`, for `A` of shape `(n, k)` and `B` of shape `(k, m)`, is a dot product of length $k$:
    $k$ multiplications and about $k$ additions. A multiplication or an addition of two such numbers is one **floating-point operation**, or **FLOP** (a
    "floating-point number" is a computer number with a point, the kind we've been using).

    ## By hand: one small product

    Take `A` of shape `(2, 3)` and `B` of shape `(3, 4)`. How many numbers does `A @ B` have? How many FLOPs does each one cost, counting $k$ multiplications and $k$ additions? And how many FLOPs is the whole product?
    """)
    return


@app.cell(hide_code=True)
def _():
    flop_boxes = hand.boxes(['numbers in the result', 'FLOPs for each', 'FLOPs in all'])
    return (flop_boxes,)


@app.cell(hide_code=True)
def _(flop_boxes):
    hand.marked(flop_boxes, {'numbers in the result': 8, 'FLOPs for each': 6, 'FLOPs in all': 48}, {
        'numbers in the result': r"""the result has shape `(2, 4)`""",
        'FLOPs for each': r"""a dot product of length 3: 3 multiplications and 3 additions""",
        'FLOPs in all': r"""your first answer times your second""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In general there are $n \times m$ numbers, each costing $2k$:

    > $$\text{FLOPs of a product} \approx 2 \times n \times k \times m$$

    ## Exercise 2: matmul_flops

    Write `matmul_flops(n, k, m)`, that formula: the FLOPs of multiplying a matrix of shape `(n, k)` by one of shape `(k, m)`. Your hand answer is the check: `matmul_flops(2, 3, 4)` gives `48`.

    The cell after the check uses it to measure your computer: it times the product of two $1024 \times 1024$ matrices and divides the FLOPs by the seconds. A **gigaFLOP** is a billion FLOPs.
    (This cell needs Exercises 1 and 2, and shows an error until they pass. That's normal, and the same goes for later cells that use your functions.)
    """)
    return


@app.function
def matmul_flops(n, k, m):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "matmul_flops", matmul_flops, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line: the formula, with `*` for each $\times$.""",
    ], LOCALE)
    return


@app.cell
def _():
    A = torch.rand(1024, 1024)
    B = torch.rand(1024, 1024)
    def product():
        return A @ B

    seconds_32 = time_it(product, 5)
    machine_speed = matmul_flops(1024, 1024, 1024) / seconds_32
    round(machine_speed / 1e9, 1)
    return A, B, machine_speed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That's your computer's speed at its best, in gigaFLOPs per second. Yours will differ from anyone else's. (`product` is a function with no inputs, so `time_it` can call it again and again.)
    On the machine this chapter was tested on it was about 180, a modern GPU does about 1,000 times more, and the largest training runs use thousands of those GPUs together.

    ## The cost of training

    For a model with $N$ knobs, here's a rule that's close enough to plan with. In the forward pass, each knob takes part in about one multiplication and one addition per token:
    2 FLOPs. The backward pass needs about twice that: one product for the slopes of the knobs, and one for the blame passed back to the layer's input (chapter 08's
    two products). That makes 4 more.

    ## By hand: a tiny training run

    A model has 1,000 knobs, and reads 10 tokens once. By the rule, how many FLOPs does the forward pass take? The backward pass? And both together?
    """)
    return


@app.cell(hide_code=True)
def _():
    cost_boxes = hand.boxes(['forward', 'backward', 'together'])
    return (cost_boxes,)


@app.cell(hide_code=True)
def _(cost_boxes):
    hand.marked(cost_boxes, {'forward': 20000, 'backward': 40000, 'together': 60000}, {
        'forward': r"""2 FLOPs for each knob, for each token""",
        'backward': r"""twice the forward pass""",
        'together': r"""that's 6 FLOPs for each knob, for each token""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So:

    > $$\text{training FLOPs} \approx 6 \times N \times \text{tokens}$$

    It ignores the attention scores themselves, which matter only for long windows.

    ## Exercise 3: training_flops

    Write `training_flops(knobs, tokens)`, that rule. `tokens` is the number of tokens read in the whole run, counting each reading. `training_flops(1000, 10)` gives `60000`.
    """)
    return


@app.function
def training_flops(knobs, tokens):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "training_flops", training_flops, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line: 6, times the knobs, times the tokens.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From FLOPs to seconds

    A computer never works at its best speed all the time. The share of its best speed that a run achieves is called the run's **utilisation**: 0.5 means half. So:

    > $$\text{seconds} = \frac{\text{FLOPs}}{\text{speed} \times \text{utilisation}}$$

    ## By hand: how long?

    A run needs 30 teraFLOPs, which is $30 \times 10^{12}$ FLOPs, on a computer whose best speed is 100 gigaFLOPs per second, $100 \times 10^9$. How many seconds would it take at full speed? And at a utilisation of 0.5?
    """)
    return


@app.cell(hide_code=True)
def _():
    seconds_boxes = hand.boxes(['seconds at full speed', 'seconds at utilisation 0.5'])
    return (seconds_boxes,)


@app.cell(hide_code=True)
def _(seconds_boxes):
    hand.marked(seconds_boxes, {'seconds at full speed': 300, 'seconds at utilisation 0.5': 600}, {
        'seconds at full speed': r"""$30 \times 10^{12} / (100 \times 10^9)$: the 12 zeros lose 9, and 30,000 divided by 100 is left""",
        'seconds at utilisation 0.5': r"""half the speed takes twice as long""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: run_seconds

    Write `run_seconds(flops, speed, utilisation)`, the formula above. `flops` is the arithmetic of a run, `speed` the computer's best speed in FLOPs per second, and `utilisation` a share between 0 and 1.
    `run_seconds(30e12, 100e9, 0.5)` gives `600.0`.

    /// admonition | 🧰 Toolbox: big numbers, written short
    `30e12` means $30 \times 10^{12}$: the number after the `e` says how many places to move the point to the right. So `7e9` is 7 billion. A minus sign moves the point left, as in chapter 09's `1e-8`.
    ///

    Chapter 21's run read about 2 million tokens twice with about 1.17 million knobs. The cell after the check works out its arithmetic in teraFLOPs (a teraFLOP is $10^{12}$), and how many seconds it would take at your computer's best speed.
    """)
    return


@app.function
def run_seconds(flops, speed, utilisation):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "run_seconds", run_seconds, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line: a division. The bottom is two things multiplied, so it needs its own brackets.""",
        r"""If `run_seconds(30e12, 100e9, 0.5)` gives 150 and not 600, the utilisation ended up on top: without brackets, Python divides by the speed and then multiplies by 0.5.""",
    ], LOCALE)
    return


@app.cell
def _(machine_speed):
    flops_run = training_flops(1.17e6, 2 * 2.03e6)
    round(flops_run / 1e12, 1), round(run_seconds(flops_run, machine_speed, 1.0))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 28 teraFLOPs, which at full speed would take a couple of minutes. Compare the second number with how long chapter 21's button really took on your computer: divide the first by the second in your head, and you have your run's utilisation.
    On the test machine the real run took 4 or 5 minutes, so it used roughly half of
    the computer's best speed. The rest goes to small operations, such as softmax, normalisation, and moving numbers about, which do little arithmetic but take time.
    For a 7-billion-knob model and a trillion tokens, the same formula gives $4.2 \times 10^{22}$ FLOPs: about a thousand GPUs, each doing around 200 trillion FLOPs per second, for about a week at half their best speed.

    ## Counting memory

    During training with Adam, each knob needs four numbers: itself, its slope, and Adam's two averages. At 32 bits, each number takes 4 bytes. On top of that
    come the **activations**: every value the forward pass keeps for the backward pass, which grow with the batch and the window.

    ## By hand: bytes for one model

    How many bytes does one knob need during training with Adam? And how many megabytes, MB, does a model of 1 million knobs need for its knobs, slopes and averages? (A megabyte is a million bytes.)
    """)
    return


@app.cell(hide_code=True)
def _():
    byte_boxes = hand.boxes(['bytes for one knob', 'MB for a million knobs'])
    return (byte_boxes,)


@app.cell(hide_code=True)
def _(byte_boxes):
    hand.marked(byte_boxes, {'bytes for one knob': 16, 'MB for a million knobs': 16}, {
        'bytes for one knob': r"""4 numbers of 4 bytes each""",
        'MB for a million knobs': r"""a million times your first answer, in millions of bytes""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: adam_bytes

    Write `adam_bytes(knobs)`: the bytes that training with Adam needs for a model's knobs, slopes and averages, at 32 bits. `adam_bytes(1000000)` gives `16000000`.
    """)
    return


@app.function
def adam_bytes(knobs):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "adam_bytes", adam_bytes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line: the bytes for one knob, times the number of knobs.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For chapter 20's model, with 1.17 million knobs, that's under 19 MB. For a 7-billion-knob model, 112 GB, which doesn't fit on most GPUs. Chapter 23 is about what to do then.

    ## Speed-up 1: smaller numbers

    A number at 32 bits keeps about 7 correct digits (chapter 10). Two 16-bit formats halve the memory and, on hardware built for them, more than double the speed:

    - **float16** keeps about 3 digits, but can't hold numbers above 65,504;
    - **bfloat16** keeps fewer digits, about 2 or 3, but reaches as high as float32.

    /// admonition | 🧰 Toolbox: asking about a format
    `torch.finfo(dtype)` describes a number format: its `max` is the largest number it can hold, and its `eps` is the gap between 1 and the next number it can hold, a measure of how precise it is.
    An `eps` of about 0.001 is 1 part in 1,000, so about 3 correct digits.
    ///

    ## By hand: will it fit?

    Before running the cell below, predict. In float16, where nothing above 65,504 fits: does $200 \times 200$ fit? Does $300 \times 300$? Answer 1 for yes and 0 for no.
    And how many MB do a million numbers take in a 16-bit format?
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_boxes = hand.boxes(['200 × 200 fits', '300 × 300 fits', 'MB for a million numbers'])
    return (fit_boxes,)


@app.cell(hide_code=True)
def _(fit_boxes):
    hand.marked(fit_boxes, {'200 × 200 fits': 1, '300 × 300 fits': 0, 'MB for a million numbers': 2}, {
        '200 × 200 fits': r"""$200 \times 200 = 40{,}000$. Is that above 65,504?""",
        '300 × 300 fits': r"""$300 \times 300 = 90{,}000$. Is that above 65,504?""",
        'MB for a million numbers': r"""16 bits is 2 bytes""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The cell shows each format's `max` and `eps`, then what float16 makes of 70,000, and what bfloat16 makes of $1 + 0.001$:
    """)
    return


@app.cell
def _():
    formats = []
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        formats.append((dtype, torch.finfo(dtype).max, torch.finfo(dtype).eps))
    formats, torch.tensor(70000.0, dtype=torch.float16), (torch.tensor(1.0, dtype=torch.bfloat16) + 0.001).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    70,000 in float16 becomes `inf`, infinity: too big. And 1 + 0.001 in bfloat16 is still 1: too small a change to register. Training in 16 bits alone would lose small updates, so
    real training uses **mixed precision**: the matrix products run in 16 bits, while the knobs, Adam's averages and the sums that need precision stay in 32 bits.

    Does it make *your* computer faster? Only if its processor has hardware for 16-bit arithmetic. The cell times the same product in each format, in gigaFLOPs per second, so higher is faster. It may take several seconds.
    """)
    return


@app.cell
def _(A, B):
    speeds = []
    for dtype_try in [torch.float32, torch.bfloat16]:
        A_try, B_try = A.to(dtype_try), B.to(dtype_try)

        def product_try():
            return A_try @ B_try

        speeds.append((dtype_try, round(matmul_flops(1024, 1024, 1024) / time_it(product_try, 3) / 1e9, 1)))
    speeds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On the test machine, bfloat16 ran **over 100 times slower** than float32, because its processor has no 16-bit hardware and imitates it step by step. Your numbers may differ:
    some newer processors and Apple chips do have it. That's the lesson: a speed-up is a claim about particular hardware, and the only way to know is to measure. On GPUs, mixed precision is the norm.

    ## Speed-up 2: don't build what you don't need

    Attention builds a table of scores, one for every pair of words: for a window of 1,024 tokens, 4 heads and a batch of 8, that's about 33.5 million numbers, 134 MB, written to
    memory and read back several times (mask, softmax, product). Memory, not arithmetic, becomes the limit: a processor computes far faster than it can fetch numbers
    from main memory. Its small **cache**, the fast memory right next to it, is where speed lives.

    **Tiled attention** never builds the whole table. For each query, it walks along the keys in small tiles, keeping a running result. The trick is in softmax: it needs the
    maximum and the total of all the scores, which we don't know until the end. But both can be kept as running values and fixed up as we go:

    - keep a running maximum `best`, a running total `total` of $e^{\text{score} - \text{best}}$, and a running weighted sum `acc` of values;
    - when a new tile brings a bigger maximum, everything accumulated so far was computed against the old one, so multiply `total` and `acc` by $e^{\text{old best} - \text{new best}}$ (chapter 07: $e^{a-b}$ rescales correctly);
    - at the end, `acc / total` is exactly softmax's weighted average.

    This is the idea of **FlashAttention**.

    Here it is on a small case: 1 number per vector, scores $[1, 3, 2]$, values $[10, 20, 30]$ and tiles of 1.

    | tile | `best` | fix | `total` | `acc` |
    |---|---|---|---|---|
    | 1 | 1 | 0 | 1 | 10 |
    | 2 | 3 | $e^{1 - 3} \approx 0.135$ | $1 \times 0.135 + 1 = 1.135$ | $10 \times 0.135 + 20 = 21.35$ |
    | 3 | 3 | $e^{3 - 3} = 1$ | $1.135 + e^{2 - 3} = 1.503$ | $21.35 + 0.368 \times 30 = 32.39$ |

    And $32.39 / 1.503 \approx 21.55$, the same as softmax over all three scores at once. In the second tile the maximum grew, so everything before it was scaled down. In the third it didn't, so the fix was 1.

    ## By hand: four scores in two tiles

    One query, with 1 number per vector, so a score is just a key times the query, and we take the query to be 1 and skip the $\sqrt{d}$, which is 1.
    The scores are $[0, 1, 2, 0]$ and the values $[10, 20, 30, 40]$, in tiles of 2: first scores $[0, 1]$ with values $[10, 20]$, then scores $[2, 0]$ with values $[30, 40]$.
    You'll need $e^0 = 1$, $e^{-1} = 0.368$ and $e^{-2} = 0.135$.

    For each tile: the new `best` is the largest score so far. Each weight is $e^{\text{score} - \text{best}}$. What was accumulated before gets multiplied by $e^{\text{old best} - \text{new best}}$, and then the tile's weights are added to `total`, and its weights times its values to `acc`.
    In the first tile nothing was accumulated before, so just add.
    """)
    return


@app.cell(hide_code=True)
def _():
    tile_boxes = hand.boxes(['best after tile 1', 'total after tile 1', 'acc after tile 1', 'best after tile 2', 'total after tile 2', 'acc after tile 2', 'answer'], step=0.001)
    return (tile_boxes,)


@app.cell(hide_code=True)
def _(tile_boxes):
    hand.marked(tile_boxes, {'best after tile 1': 1, 'total after tile 1': 1.368, 'acc after tile 1': 23.68, 'best after tile 2': 2, 'total after tile 2': 1.638, 'acc after tile 2': 44.11, 'answer': 26.93}, {
        'best after tile 1': r"""the larger of the scores 0 and 1""",
        'total after tile 1': r"""the weights are $e^{0 - 1}$ and $e^{1 - 1}$: add them""",
        'acc after tile 1': r"""$0.368 \times 10 + 1 \times 20$""",
        'best after tile 2': r"""the largest score so far is now 2""",
        'total after tile 2': r"""the old total times $e^{1 - 2}$, plus the new weights $e^{2 - 2}$ and $e^{0 - 2}$""",
        'acc after tile 2': r"""the old `acc` times $e^{1 - 2}$, plus $1 \times 30 + 0.135 \times 40$""",
        'answer': r"""`acc` divided by `total`""",
    }, LOCALE, tolerance=0.05, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Softmax over all four scores at once gives 26.93 too: the two tiles never needed to be in memory together.

    ## Exercise 6: tiled_attention

    Write `tiled_attention(q, K, V, tile)` for one query `q` of shape `(d,)`, keys `K` of shape `(n, d)` and values `V` of shape `(n, v)`, with no mask, and scores scaled by $\sqrt{d}$ as in chapter 16. `tile` is how many keys to take at a time.
    It gives back a tensor of shape `(v,)`, the same answer as `torch.softmax(K @ q / np.sqrt(d), dim=0) @ V`, without ever holding more than `tile` scores.

    Start with three running values: `best` as `-np.inf` (minus infinity, so that any score beats it, and $e^{-\infty}$ is 0, which makes the first tile work like the others), `total` as `0.0`, and `acc` as a tensor of `v` zeros.
    The loop over tiles is given, since slicing by steps is chapter 09's tool:

    ```
    for start in range(0, len(K), tile):
    ```

    Inside it, the tile's keys are `K[start:start + tile]` and its values `V[start:start + tile]`. Do what you did by hand:

    | step | in math |
    |---|---|
    | the tile's scores | $\text{keys} \cdot q / \sqrt{d}$ |
    | the new best | the larger of `best` and the tile's largest score |
    | the fix for what came before | $e^{\text{best} - \text{new best}}$ |
    | the tile's weights | $e^{\text{scores} - \text{new best}}$ |
    | `total` | $\text{total} \times \text{fix} + \text{the sum of the weights}$ |
    | `acc` | $\text{acc} \times \text{fix} + \text{weights} \cdot \text{values}$ |
    | `best` | becomes the new best |

    After the loop, the answer is `acc` divided by `total`. `torch.max(scores).item()` and `torch.sum(weights).item()` give plain numbers, `torch.exp` works on tensors, and `np.exp` on plain numbers.
    """)
    return


@app.function
def tiled_attention(q, K, V, tile):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "tiled_attention", tiled_attention, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Seven lines inside the loop, one for each row of the table, in that order. `best` must be updated last: the fix needs the old `best` and the new one. The dot product of a tile's keys with `q` is `@`, and so is the weights with the tile's values.""",
        r"""The first two lines inside the loop are `scores = K[start:start + tile] @ q / np.sqrt(len(q))` and `new_best = max(best, torch.max(scores).item())`. If the check with big scores fails, you're taking $e$ to a score without subtracting the best first.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Written in Python, one query at a time, this is slow: the point is the *method*. In real systems the same steps run inside one compiled function on the processor, keeping tiles
    in the cache. PyTorch has one: `F.scaled_dot_product_attention`. The cell below compares it with chapter 16's way, building the whole masked table, at two window lengths.
    Each row: window, seconds the table way, seconds the fused way, the largest difference in the answers (something like `5e-07`, which is 0.0000005, only rounding), and the MB the table needs.
    """)
    return


@app.cell
def _():
    rows = []
    for window in [256, 1024]:
        q_all = torch.rand(8, 4, window, 32)
        k_all = torch.rand(8, 4, window, 32)
        v_all = torch.rand(8, 4, window, 32)
        mask = torch.triu(torch.ones(window, window, dtype=torch.bool), diagonal=1)

        def table_way():
            scores = (q_all @ k_all.transpose(-2, -1) / np.sqrt(32)).masked_fill(mask, -torch.inf)
            return torch.softmax(scores, dim=-1) @ v_all

        def fused_way():
            return F.scaled_dot_product_attention(q_all, k_all, v_all, is_causal=True)

        difference = torch.max(torch.abs(table_way() - fused_way())).item()
        rows.append((window, round(time_it(table_way, 3), 4), round(time_it(fused_way, 3), 4), difference, 8 * 4 * window * window * 4 / 1e6))
    rows
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On the test machine the fused version was about 7 times faster at 256 and about 13 times faster at 1,024, with the same answers up to rounding. The longer the window, the bigger the win,
    because the table, and the time spent writing and reading it, grow with the *square* of the window, and the fused way never builds it. Doubling the window quadruples the table: that's what "grows with the square" means, and why long windows are expensive.

    ## Profiling: find out what's slow

    Before speeding anything up, find out what's slow. A **profiler** times every operation inside a run. PyTorch's is `torch.profiler`. The cell below profiles one training step of a small
    transformer block, PyTorch's own version of chapter 17's `Block`, and lists the five most expensive operations:
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0, batch_first=True)
    x_profile = torch.rand(32, 64, 128)
    with torch.profiler.profile() as profile:
        layer(x_profile).sum().backward()
    print(profile.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The table is dense, so read only the names and the `CPU total` column. Matrix products (names with `mm` in them, such as `aten::mm` and `AddmmBackward0`) usually lead. If something else does, that's
    where to look first. A row's total includes the operations it calls, so the same time can appear on several rows. (`print` shows text as it is, with its columns lined up, which is why it's used here.)

    ## By hand: read a profile

    Your table depends on your computer, so here's a made-up one to practise on. `aten::linear` is a whole `nn.Linear` layer, and it calls `aten::addmm`, the matrix product with its bias. `aten::softmax` calls neither.

    | name | CPU total |
    |---|---|
    | `aten::linear` | 9.1 ms |
    | `aten::addmm` | 8.0 ms |
    | `aten::softmax` | 1.2 ms |

    How many milliseconds did the linear layers spend outside their matrix products? And how long did everything take, counting no time twice?
    """)
    return


@app.cell(hide_code=True)
def _():
    profile_boxes = hand.boxes(['ms in linear outside addmm', 'ms in all'], step=0.1)
    return (profile_boxes,)


@app.cell(hide_code=True)
def _(profile_boxes):
    hand.marked(profile_boxes, {'ms in linear outside addmm': 1.1, 'ms in all': 10.3}, {
        'ms in linear outside addmm': r"""the 9.1 of `aten::linear` already includes the 8.0 of `aten::addmm`""",
        'ms in all': r"""add `aten::linear` and `aten::softmax`, and leave out `aten::addmm`, which is inside the first""",
    }, LOCALE, tolerance=0.01)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Adding all three rows would give 18.3 ms, nearly twice the truth. And 8.0 of the 10.3 ms is one matrix product: that's the line worth making faster.

    ## What's next

    One computer can only go so fast. **Chapter 23** spreads training over several devices, here several processes on your computer, and checks they learn exactly what one would.

    ## What you can do now

    - time code fairly, with a warm-up and repeats
    - count the FLOPs of a matrix product and of a training run, by hand and in code
    - estimate a run's time from your computer's speed, and its utilisation
    - count the memory that training with Adam needs
    - explain float16, bfloat16 and mixed precision, and test whether they help on your hardware
    - explain why attention's score table is the bottleneck, trace tiled attention by hand, and write it
    - read a profile without counting time twice

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** FlashAttention is from Dao et al. (2022). The `6N` rule is from *Scaling Laws for Neural Language Models* (Kaplan et al., 2020), which chapter 24 builds on.
    *Kernel fusion* and `torch.compile` are how many small operations become one, and *activation checkpointing* trades recomputation for activation memory.
    """)
    return


if __name__ == "__main__":
    app.run()
