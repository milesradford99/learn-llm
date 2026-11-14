import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="22 · Making it fast")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from learn_llm import laptop_speed, todo
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

    `time.perf_counter()` (chapter 09) gives a clock in seconds. Measuring one run is unreliable: the first run of anything is often slow while the computer gets ready,
    and every run varies a little. So we run once to warm up, then time several runs and take the mean.

    ## Exercise 1: time_it

    Write `time_it(f, repeats)`, the mean seconds of one call of `f`, after one warm-up call:

    1. `f()`
    2. `started = time.perf_counter()`
    3. `for i in range(repeats):`, with `f()` inside
    4. `return (time.perf_counter() - started) / repeats`
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
    mo.md(r"""
    ## Counting arithmetic

    The work in a network is almost all matrix products. Each number of `A @ B`, for `A` of shape `(n, k)` and `B` of shape `(k, m)`, is a dot product of length `k`:
    `k` multiplications and about `k` additions. There are $n \times m$ such numbers, so a product costs about $2nkm$ **floating-point operations**, or **FLOPs** (a
    "floating-point number" is a computer number with a point, the kind we've been using).

    ## Exercise 2: matmul_flops

    Write `matmul_flops(n, k, m)`:

    ```
    return 2 * n * k * m
    ```

    The cell after the check uses it to measure your computer: it times the product of two $1024 \times 1024$ matrices and divides. A **gigaFLOP** is a billion FLOPs.
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

    For a model with `N` knobs, here's a rule that's close enough to plan with. In the forward pass, each knob takes part in about one multiplication and one addition per token:
    2 FLOPs. The backward pass needs about twice that: one product for the slopes of the knobs, and one for the blame passed back to the layer's input (chapter 08's
    `X.T @ G` and `G @ W.T`). That makes 4 more. So:

    > $$\text{training FLOPs} \approx 6 \times N \times \text{tokens}$$

    It ignores the attention scores themselves, which matter only for long windows.

    ## Exercise 3: training_flops

    Write `training_flops(knobs, tokens)`:

    ```
    return 6 * knobs * tokens
    ```
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
    mo.md(r"""
    Chapter 21's run read about 2 million tokens twice with about 1.17 million knobs. The cell below works out the arithmetic, and how long it would take at your computer's best speed.
    The cell shows two numbers: the teraFLOPs of arithmetic, and the seconds at full speed. Compare those seconds with how long chapter 21's button really took. The share of the best possible speed that a run achieves is called its **utilisation**.
    """)
    return


@app.cell
def _(machine_speed):
    flops_ch18 = training_flops(1.17e6, 2 * 2.03e6)
    round(flops_ch18 / 1e12, 1), round(flops_ch18 / machine_speed)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 28 teraFLOPs (a teraFLOP is $10^{12}$), which at full speed would take a couple of minutes. The real run took 4 or 5 minutes on the test machine, so it used roughly half of
    the computer's best speed. The rest goes to small operations, such as softmax, normalisation, and moving numbers about, which do little arithmetic but take time.
    For a 7-billion-knob model and a trillion tokens, the same formula gives $4.2 \times 10^{22}$ FLOPs: about a thousand GPUs, each doing around 200 trillion FLOPs per second, for about a week at half their best speed.

    ## Counting memory

    During training with Adam, each knob needs four numbers: itself, its slope, and Adam's two averages. At 32 bits, that's 4 bytes each, 16 bytes per knob. On top of that
    come the **activations**: every value the forward pass keeps for the backward pass, which grow with the batch and the window.

    ## Exercise 4: adam_bytes

    Write `adam_bytes(knobs)`, the bytes for knobs, slopes and Adam's averages at 32 bits:

    ```
    return 16 * knobs
    ```
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
    mo.md(r"""
    For chapter 20's model, that's under 19 MB. For a 7-billion-knob model, 112 GB, which doesn't fit on most GPUs. Chapter 23 is about what to do then.

    ## Speed-up 1: smaller numbers

    A number at 32 bits keeps about 7 correct digits (chapter 10). Two 16-bit formats halve the memory and, on hardware built for them, more than double the speed:

    - **float16** keeps about 3 digits, but can't hold numbers above 65,504;
    - **bfloat16** keeps fewer digits, about 2 or 3, but reaches as high as float32.

    `torch.finfo` describes a format: `max` is its largest number, and `eps` is the gap between 1 and the next number it can hold, a measure of how precise it is. An `eps` of about 0.001 is 1 part in 1,000, so about 3 correct digits:
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

    ## Exercise 5: tiled_attention

    Write `tiled_attention(q, K, V, tile)` for one query `q` of shape `(d,)`, keys `K` and values `V` of shape `(n, d)`, no mask, scaled by $\sqrt{d}$. It must give the same
    answer as `torch.softmax(K @ q / np.sqrt(d), dim=0) @ V`.

    1. Three lines: `best = -np.inf`, `total = 0.0` and `acc = torch.zeros(V.shape[1])`
    2. `for start in range(0, len(K), tile):`, and inside it:
    3. `scores = K[start:start + tile] @ q / np.sqrt(len(q))`
    4. `new_best = max(best, torch.max(scores).item())`
    5. `fix = np.exp(best - new_best)`
    6. `weights = torch.exp(scores - new_best)`
    7. `total = total * fix + torch.sum(weights).item()`
    8. `acc = acc * fix + weights @ V[start:start + tile]`
    9. `best = new_best`
    10. After the loop, `return acc / total`

    (`np.inf` is infinity, like `torch.inf`, and `np.exp(-np.inf)` is 0, so the very first tile works too.)

    Here it is by hand, for 1 number per vector, keys `[1, 3, 2]`, values `[10, 20, 30]` and tiles of 1 (we skip the $\sqrt{d}$, which is 1):

    | tile | `best` | `fix` | `total` | `acc` |
    |---|---|---|---|---|
    | 1 | 1 | 0 | 1 | 10 |
    | 2 | 3 | $e^{-2} \approx 0.135$ | 1.135 | 21.35 |
    | 3 | 3 | 1 | 1.503 | 32.39 |

    And $32.39 / 1.503 \approx 21.55$, the same as softmax over all three scores at once.
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

    ## What's next

    One computer can only go so fast. **Chapter 23** spreads training over several devices, here several processes on your computer, and checks they learn exactly what one would.

    ## What you can do now

    - time code fairly, with a warm-up and repeats
    - count the FLOPs of a matrix product and of a training run
    - estimate a run's time from your computer's speed, and its utilisation
    - count the memory that training with Adam needs
    - explain float16, bfloat16 and mixed precision, and test whether they help on your hardware
    - explain why attention's score table is the bottleneck, and write tiled attention
    - read a profile

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** FlashAttention is from Dao et al. (2022). The `6N` rule is from *Scaling Laws for Neural Language Models* (Kaplan et al., 2020), which chapter 24 builds on.
    *Kernel fusion* and `torch.compile` are how many small operations become one, and *activation checkpointing* trades recomputation for activation memory.
    """)
    return


if __name__ == "__main__":
    app.run()
