import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="09 · Hello PyTorch")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits

    LOCALE = "en"
    laptop_speed()
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 · Hello PyTorch

    So far you have written every piece of a training run yourself: the forward pass, an engine
    that keeps the books, backward steps through matrices, batches, Adam. Nobody writes all that
    afresh for each model. They use a **library**, a toolbox of code other people have written
    and tested, as you already do with NumPy.

    The one most of the field uses is **PyTorch**. This chapter isn't a tour of a new subject:
    you already know what PyTorch does, because you built it. So we'll go through it one piece at
    a time, and each time compare its answer with yours. If the numbers agree, you can trust the
    tool, and you know what it's doing under the hood.

    ## Tensors: PyTorch's arrays

    PyTorch's version of a NumPy array is called a **tensor**. The word sounds grand, but a tensor
    is just an array of numbers with a shape. Most of what you know carries over: `@`, `*`, `.T`,
    `.shape`, indexing and slices all work the same.
    """)
    return


@app.cell
def _():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    u = torch.tensor([10.0, 20.0])
    A @ u, A.shape, torch.mean(A)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `torch.tensor(...)` makes a tensor from lists, as `np.array(...)` does. `torch.mean` is
    `np.mean`, and `torch.sum`, `torch.exp` and `torch.log` are what you'd guess. One name differs:
    where NumPy says `axis=1`, PyTorch says `dim=1` ("dimension"). It means the same thing.

    **A word on precision.** A computer stores each number in a fixed amount of memory, measured in
    **bits**. NumPy normally uses 64 bits per number, which keeps about 16 digits correct.
    PyTorch normally uses 32, which keeps about 7: half the memory, and plenty for training. So
    the same sum can differ in its last digits. Here's 0.1 × 3 both ways. (Neither is exactly 0.3.
    That's normal: computers store numbers in binary, and 0.1 can't be stored exactly.)
    """)
    return


@app.cell
def _():
    (torch.tensor(0.1) * 3).item(), 0.1 * 3
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The kind of number stored is called the **dtype** (data type), and there are three you'll meet:

    | dtype | holds | used for |
    |---|---|---|
    | `torch.float32` | numbers with a point, 32 bits | almost everything |
    | `torch.float64` | numbers with a point, 64 bits | exact comparisons, like the ones below |
    | `torch.long` | whole numbers | labels and positions |

    `torch.tensor` takes the dtype as an input, as in `torch.tensor(labels, dtype=torch.long)`.
    A tensor made from a NumPy array keeps that array's dtype, usually 64 bits. PyTorch refuses to
    mix two dtypes in `@`, so if you ever see an error saying `must have the same dtype`, that's
    the cause.

    ## Your engine, built in

    Here is chapter 04's job: Mia's day 1 (`t = 14`, `s = 63`) with `w = 10` and `b = -60`, whose
    blames you found to be 476 and 34.
    """)
    return


@app.cell
def _():
    w = torch.tensor(10.0, requires_grad=True)
    b = torch.tensor(-60.0, requires_grad=True)
    e = w * 14 + b - 63
    L = e * e
    L.backward()
    w.grad.item(), b.grad.item()
    return b, w


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    476 and 34, the same as yours. Compare it with what you wrote:

    | your engine (chapter 04) | PyTorch |
    |---|---|
    | `Value(10)` | `torch.tensor(10.0, requires_grad=True)` |
    | `add(a, b, book)`, `mul(a, b, book)` | `a + b`, `a * b` |
    | `book`, filled in by you | filled in automatically, hidden from you |
    | `backward(loss, book)` | `loss.backward()` |
    | `v.data`, `v.grad` | `v.item()`, `v.grad` |

    `requires_grad=True` means "this is a knob: keep the books on everything calculated from it".
    Every `+` and `*` on such a tensor is written down, with its parents and local slopes, just as
    your `mul` did. `backward()` then reads the book in reverse order and fills in `.grad`.
    `w.grad` is a tensor holding a single number, and `.item()` turns it into a plain Python number.

    Your engine had one trap: blame is *added* to a box, so it must start at 0. PyTorch does the
    same thing, on purpose. Below, we build the loss again and go backward a second time, without
    clearing `.grad`:
    """)
    return


@app.cell
def _(b, w):
    L_again = (w * 14 + b - 63) ** 2
    L_again.backward()
    w.grad.item(), b.grad.item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Doubled: 952 and 68. (Run this cell again and it grows again; run the cell with
    `L.backward()` to start over.) The new blame was added to the old. So every training step must clear the
    old blame first. Forgetting that is the classic PyTorch mistake, and there's a tool for it
    further down.

    (Notice that we had to calculate `L_again` from scratch. PyTorch throws its book away after
    going backward, to save memory, so each step needs a fresh forward pass, as in chapter 04.)

    ## Exercise 1: torch_gradient

    Back in chapter 04 your `engine_gradient` took about twenty lines. Write the same thing with
    tensors. `torch_gradient(w, b, xs, ys)` gets plain numbers `w` and `b` and two lists, and gives
    back the two slopes of the mean squared error as plain numbers:

    1. Two lines: `w = torch.tensor(w, requires_grad=True)`, and the same for `b`
    2. Two lines: `x = torch.tensor(xs, dtype=torch.float32)`, and `y` from `ys` the same way
    3. `e = w * x + b - y` (the single number `w` broadcasts over the whole vector `x`)
    4. `L = torch.mean(e * e)`
    5. `L.backward()`
    6. `return w.grad.item(), b.grad.item()`
    """)
    return


@app.function
def torch_gradient(w, b, xs, ys):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "torch_gradient", torch_gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Three ways, one answer

    Now something bigger: chapter 07's network. Below are your `softmax`, `forward` and `backward`
    from chapters 07 and 08, in NumPy, finished. We'll compute the slopes of the same network, on
    the same 100 pictures, with your hand-derived formulas and with PyTorch.
    """)
    return


@app.function
def softmax(S):
    S = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=1, keepdims=True)


@app.function
def forward(X, W1, b1, W2, b2):
    Z = X @ W1 + b1
    H = np.maximum(Z, 0)
    return Z, H, H @ W2 + b2


@app.function
def backward(X, Y, Z, H, S, W2):
    G = (softmax(S) - Y) / len(X)
    dZ = (G @ W2.T) * (Z > 0)
    return X.T @ dZ, np.sum(dZ, axis=0), H.T @ G, np.sum(G, axis=0)


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X_np = train_images.reshape(10000, 784) / 255
    X_test_np = test_images.reshape(2000, 784) / 255
    rng = np.random.default_rng(0)
    W1_np = rng.normal(0, np.sqrt(2 / 784), (784, 64))
    b1_np = np.zeros(64)
    W2_np = rng.normal(0, np.sqrt(2 / 64), (64, 10))
    b2_np = np.zeros(10)
    X_np.shape, W1_np.shape
    return (
        W1_np,
        W2_np,
        X_np,
        X_test_np,
        b1_np,
        b2_np,
        test_labels,
        train_labels,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: scores

    Write the forward pass in tensors. `scores(X, W1, b1, W2, b2)` gives back the scores in one
    line, with `torch.relu` for the bend:

    ```
    return torch.relu(X @ W1 + b1) @ W2 + b2
    ```

    Compared with chapter 07's `forward`, there's no need to keep `Z` and `H` this time. You kept
    them so that `backward` could use them, and now the book keeps them for you.
    """)
    return


@app.function
def scores(X, W1, b1, W2, b2):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "scores", scores, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The loss needs one new tool. `F.cross_entropy(S, labels)` is chapter 06's cross-entropy, with
    two conveniences. It takes the raw **scores** (it does softmax itself), and it takes the
    **labels** as whole numbers instead of one-hot rows. `F` is the usual short name for a part of
    PyTorch called `torch.nn.functional`. It's already loaded for you, as are `torch`, `nn` and `np`.

    The cell below does the whole comparison on 100 pictures. The first half is your hand-written
    version. The second half puts the same numbers in tensors (`torch.tensor` on a NumPy array keeps
    its 64-bit dtype, so both sides have the same precision) and lets PyTorch find the slopes. The
    last line shows the largest difference between any of the 50,890 slopes.

    Two new forms appear in the cell. `[torch.tensor(k, requires_grad=True) for k in (…)]` builds a
    list by running the expression once for each item: a short way to write a loop with `append`.
    And `.numpy()` turns a tensor back into a NumPy array. `max` is Python's own: the larger of two numbers.
    """)
    return


@app.cell
def _(W1_np, W2_np, X_np, b1_np, b2_np, train_labels):
    X_100, labels_100 = X_np[:100], train_labels[:100]
    Y_100 = np.eye(10)[labels_100]
    Z_np, H_np, S_np = forward(X_100, W1_np, b1_np, W2_np, b2_np)
    mine = backward(X_100, Y_100, Z_np, H_np, S_np, W2_np)

    knobs_t = [torch.tensor(k, requires_grad=True) for k in (W1_np, b1_np, W2_np, b2_np)]
    S_t = scores(torch.tensor(X_100), *knobs_t)
    F.cross_entropy(S_t, torch.tensor(labels_100, dtype=torch.long)).backward()

    biggest = 0
    for m, k in zip(mine, knobs_t):
        biggest = max(biggest, np.max(np.abs(m - k.grad.numpy())).item())
    biggest
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The number ends in about `e-17`. As in chapter 08, that means the point moves 17 places left:
    0.00000000000000003, which is just rounding. Your formulas from chapter 07,
    derived one entry at a time, are exactly what PyTorch's book-keeping computes. Those slopes are
    the ones you nudge-checked, so that's three independent ways to one answer: nudging, your
    formulas, and PyTorch.

    ## Layers as objects

    Writing `W1, b1, W2, b2` by hand gets tiresome with many layers. PyTorch bundles the knobs of a
    layer into an object:
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    layer = nn.Linear(784, 64)
    layer.weight.shape, layer.bias.shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (`torch.Size` is just how PyTorch prints a shape.)

    `nn.Linear(784, 64)` makes a layer with 784 inputs and 64 outputs, holding a matrix of weights
    and a vector of biases. They're already random, with `requires_grad=True`.
    (`torch.manual_seed(0)` makes the random numbers repeatable, like `default_rng(0)`.)

    One thing is the other way round from our habit: our `W1` was `(784, 64)`, with a column for each
    detector, while PyTorch stores `(64, 784)`, with a row for each detector. It's the same numbers on
    their side. Calling `layer(X)` computes `X @ layer.weight.T + layer.bias`, our `Z`. Calling
    an object like a function looks odd, but it's only a shortcut for running its calculation.

    `nn.Sequential(...)` chains layers: each one's output is the next one's input.
    `nn.ReLU()` is the bend, as a layer with no knobs.

    ## Exercise 3: make_model

    Write `make_model(hidden)`, giving back chapter 07's network with `hidden` detectors:

    ```
    return nn.Sequential(nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10))
    ```

    The cell after the check asks the model for its knobs. `model.parameters()` goes through them,
    and `p.numel()` is how many numbers one of them holds.
    """)
    return


@app.function
def make_model(hidden):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "make_model", make_model, LOCALE)
    return


@app.cell
def _():
    torch.manual_seed(0)
    model_demo = make_model(64)
    knob_count = 0
    for p in model_demo.parameters():
        knob_count = knob_count + p.numel()
    model_demo, knob_count
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PyTorch prints the model layer by layer (`bias=True` just says the layer has biases), then our count:
    50,890 knobs: 784 × 64 weights, then 64 biases, then 64 × 10 weights, then 10 biases.

    ## Your Adam, built in

    An **optimizer** is the object that holds the knobs and applies a step rule. Here is your
    `adam_step` from chapter 08 and `torch.optim.Adam` on Mia's problem, side by side: raw
    temperatures, the same start `[6, 0]`, rate 1.0, 500 steps.
    """)
    return


@app.function
def adam_step(w, m, v, grad, t, rate):
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad * grad
    m_hat = m / (1 - 0.9 ** t)
    v_hat = v / (1 - 0.999 ** t)
    return w - rate * m_hat / (np.sqrt(v_hat) + 1e-8), m, v


@app.function
def mia_gradient(k):
    t = np.array(temps)
    s = np.array(sales)
    e = k[0] * t + k[1] - s
    return np.array([np.mean(2 * e * t), np.mean(2 * e)])


@app.cell
def _():
    k_mine = np.array([6.0, 0.0])
    m_mine = np.zeros(2)
    v_mine = np.zeros(2)
    for t_mine in range(1, 501):
        k_mine, m_mine, v_mine = adam_step(k_mine, m_mine, v_mine, mia_gradient(k_mine), t_mine, 1.0)

    k_torch = torch.tensor([6.0, 0.0], dtype=torch.float64, requires_grad=True)
    t_days = torch.tensor(temps, dtype=torch.float64)
    s_days = torch.tensor(sales, dtype=torch.float64)
    optimizer_mia = torch.optim.Adam([k_torch], lr=1.0)
    for _step in range(500):
        optimizer_mia.zero_grad()
        e_days = k_torch[0] * t_days + k_torch[1] - s_days
        torch.mean(e_days * e_days).backward()
        optimizer_mia.step()

    k_mine, k_torch.detach().numpy()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same knobs, to all the digits shown. (`.detach()` is needed first because PyTorch won't
    turn a tensor it's keeping books on into a NumPy array. It gives the same numbers without the
    book.) PyTorch's Adam uses the same 0.9, 0.999 and `1e-8` you typed.

    Three calls in that loop are the heart of every PyTorch training loop:

    - `optimizer.zero_grad()` clears the old blame from every knob (that's the tool for the trap
      from before);
    - `loss.backward()` fills in `.grad` on every knob;
    - `optimizer.step()` moves every knob, using its `.grad`.

    With the forward pass between the first two, that's the whole recipe: clear, forward, backward, step.

    ## Exercise 4: train_step

    Write `train_step(model, optimizer, X, labels)`, one step of training on one batch, giving back
    the loss as a plain number:

    1. `optimizer.zero_grad()`
    2. `S = model(X)`
    3. `loss = F.cross_entropy(S, labels)`
    4. `loss.backward()`
    5. `optimizer.step()`
    6. `return loss.item()`
    """)
    return


@app.function
def train_step(model, optimizer, X, labels):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "train_step", train_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The whole run

    Here's chapter 08's training function, rewritten with what we've just built. `torch.randperm`
    is PyTorch's `rng.permutation`. The loop over epochs and batches is the same as yours, and
    everything else has gone into `train_step`.
    """)
    return


@app.function
def train(model, X, labels, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            loss = train_step(model, optimizer, X[batch], labels[batch])
    return loss


@app.function
def accuracy(S, labels):
    return torch.mean((torch.argmax(S, dim=1) == labels).float()).item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `accuracy` needs a small change for tensors: `.float()` turns the `True`/`False` values into 1s
    and 0s, because `torch.mean` only averages numbers with a point.

    The cell below turns the data into 32-bit tensors, trains for 10 epochs with Adam at rate 0.002,
    and measures on the test pictures. Two things are new:

    - `torch.no_grad()`: the indented lines under `with torch.no_grad():` run without keeping the
      books. Measuring needs no blame, so this saves time and memory.
    - The second number is the seconds taken, as in chapter 08.
    """)
    return


@app.cell
def _(X_np, X_test_np, test_labels, train_labels):
    X = torch.tensor(X_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    labels = torch.tensor(train_labels, dtype=torch.long)
    labels_test = torch.tensor(test_labels, dtype=torch.long)

    torch.manual_seed(0)
    model = make_model(64)
    started = time.perf_counter()
    train(model, X, labels, 10, 100, 0.002)
    seconds = time.perf_counter() - started
    with torch.no_grad():
        test_score = accuracy(model(X_test), labels_test)
    test_score, seconds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 95% on the test pictures, in a few seconds at most, and you wrote only a handful of lines.
    It's the same design as chapter 08's network, on the same pictures, with PyTorch doing the
    book-keeping and the step rule.

    ## What's a GPU?

    Every number so far was computed on your computer's main processor, the **CPU**. A CPU has a
    few very capable workers. A **GPU** (graphics processor, built to colour millions of screen dots at
    once) has thousands of simple ones, and that suits our work: a matrix product is thousands of
    identical small sums. For large models, a GPU can be 10 to 100 times faster.

    This course installs the small, CPU-only version of PyTorch, so that every chapter runs on any
    laptop. Nothing in the course needs a GPU; chapters that would benefit from one say so.
    The cell below asks PyTorch what it can see here:
    """)
    return


@app.cell
def _():
    torch.cuda.is_available()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `False` is expected (on Macs with Apple chips PyTorch has a similar option). If you ever do have
    one, the change is small: tensors and models live on a **device**, and everything in one
    calculation must live on the same one. `model.to("cuda")` and `X.to("cuda")` move them
    (`cuda` is the name of the usual GPU system), and the rest of your code stays the same.

    ## What's next

    Your network sees a picture as 784 separate numbers: move a digit a little, and to the network
    it's a different picture. In **Chapter 10** you'll fix that with an idea built for pictures:
    **convolutions**, small detectors that slide over the image. PyTorch will do the book-keeping,
    which leaves you free to think about the design.

    ## What you can do now

    - make tensors, choose a dtype, and read a `must have the same dtype` error
    - use `requires_grad`, `backward()` and `.grad`, and say what each one does in your engine's terms
    - explain why blame must be cleared before each step
    - build networks with `nn.Linear`, `nn.ReLU` and `nn.Sequential`
    - train with an optimizer: clear, forward, backward, step
    - measure with `torch.no_grad()`
    - say what a GPU adds, and what a device is

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** PyTorch's documentation lives at https://pytorch.org/docs/. The
    engine behind `backward()` is called *autograd*, and `nn.Module`, which you'll meet next
    chapter, is the base of every layer. `nn.Linear` starts its knobs with a rule of its own,
    in the same spirit as chapter 08's He start.
    """)
    return


if __name__ == "__main__":
    app.run()
