import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="09 · Training that works")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np

    from learn_llm import hand, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits

    LOCALE = "en"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 · Training that works

    Chapter 08's network reached about 94%, but it left loose ends. We started its knobs at
    sizes 0.05 and 0.18 without saying why. Every step looked at all 10,000 pictures, so 300 steps
    took about 10 seconds. And back in chapter 02, Mia's `b` crawled because one learning rate
    can't suit every knob; we've been working around that by rescaling inputs ever since.

    This chapter ties those ends off, one at a time: a reasoned start, small batches, two smarter
    step rules (**momentum** and **Adam**), and a way to spot a model that memorises instead of
    learning. These are the habits every modern training run relies on, including the language
    models at the end of this course.

    Here are chapter 08's functions, ready to use:
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


@app.function
def accuracy(S, labels):
    return np.mean(np.argmax(S, axis=1) == labels)


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    Y = np.eye(10)[train_labels]
    X.shape, Y.shape
    return X, X_test, Y, test_labels, train_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (`np.eye(10)[train_labels]` is a shortcut for chapter 07's `one_hot`: `np.eye(10)` is the
    $10 \times 10$ table with 1s on its diagonal, so row $k$ of it is the one-hot row for digit $k$, and
    giving a list of labels picks one such row per picture; picking several rows with a list is
    explained under Small batches.)

    ## How spread out? Variance

    To choose starting sizes we need a way to measure how spread out a bunch of numbers is. Take
    their mean, then for each number its distance from the mean, squared (so negatives don't
    cancel, as in chapter 01), and average those:

    > $$\textbf{variance} = \text{mean of } (\text{number} - \text{mean})^2$$

    For `[1, 3]`: the mean is 2, the distances are −1 and 1, squared 1 and 1, so the variance
    is 1. Its square root is the **typical size** of the distances (in NumPy, `np.sqrt`; the
    square root of 9 is 3, since $3 \times 3 = 9$). For numbers centred on 0, the mean is 0 and the
    variance is simply the mean of the squares.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: a variance

    Take the four numbers `[2, 4, 4, 6]`. Work out their mean, then their variance.
    """)
    return


@app.cell(hide_code=True)
def _():
    variance_boxes = hand.boxes(['mean', 'variance'])
    return (variance_boxes,)


@app.cell(hide_code=True)
def _(variance_boxes):
    hand.marked(variance_boxes, {'mean': 4, 'variance': 2}, {
        'mean': r"""add the four numbers and divide by 4""",
        'variance': r"""the distances from the mean are $-2, 0, 0, 2$. Square them and take the mean""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: variance

    Write `variance(v)` for a NumPy array `v`, following the formula above. Arithmetic on an array works position by position, and `** 2` squares.

    For `[1, 3]` it must give 1, and for your `[2, 4, 4, 6]` it must give 2.
    """)
    return


@app.function
def variance(v):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "variance", variance, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three steps: the mean, each number's distance from it squared, and the mean of those.""",
        r"""One line. No `np.sqrt`: that would give the typical size, not the variance.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A reasoned start

    **Why not zeros?** If every knob starts at 0, every hidden detector computes exactly the same
    thing (all zeros), receives exactly the same blame, and gets exactly the same update. They
    stay identical forever. Worse: with `W2` at 0 too, chapter 08's `dH = G @ W2.T` is all zeros,
    so no blame reaches `W1` at all. The cell below trains a network started at zero for 100
    steps:
    """)
    return


@app.cell
def _(X, X_test, Y, test_labels):
    W1z, b1z, W2z, b2z = np.zeros((784, 64)), np.zeros(64), np.zeros((64, 10)), np.zeros(10)
    for _step in range(100):
        Zz, Hz, Sz = forward(X, W1z, b1z, W2z, b2z)
        g1, gb1, g2, gb2 = backward(X, Y, Zz, Hz, Sz, W2z)
        W1z, b1z, W2z, b2z = W1z - 0.5 * g1, b1z - 0.5 * gb1, W2z - 0.5 * g2, b2z - 0.5 * gb2
    np.max(np.abs(W1z)), accuracy(forward(X_test, W1z, b1z, W2z, b2z)[2], test_labels)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The largest first-layer knob is still exactly 0 (`np.abs` removes the minus signs, so
    `np.max(np.abs(...))` is the biggest size), and the test accuracy is about 12%: only `b2`,
    the scores' starting heights, learned (no blame reaches `b1` either), so it answers the same digit for every picture, about as good as guessing. Random starting knobs break the tie: every detector starts out different.

    **How big?** Look at one detector's sum for one picture, $z = w_0 \times x_0 + w_1 \times x_1 + \ldots$ over
    $n$ inputs, with random knobs of typical size $s$ (so variance $s^2$; `rng.normal(0, 0.05, …)` gives
    typical size 0.05, variance $0.05^2$). The knobs are **independent**: each is drawn on its own,
    without looking at the others. A useful fact about sums of independent random parts: **their
    variances add**. Each part $w \times x$ (the picture is fixed; only the knobs are random)
    has variance $s^2 \times x^2$ (scaling a random number by $x$
    scales its spread by $x$, so the variance by $x^2$). Adding the $n$ parts gives
    $s^2 \times (x_0^2 + x_1^2 + \ldots)$, and that sum of $n$ squares is $n$ times their mean:

    > $$\text{variance of } z \approx n \times s^2 \times (\text{mean of } x^2)$$

    Let's test that on the first picture, with 2,000 different random choices of 784 knobs of size
    0.05:
    """)
    return


@app.cell
def _(X):
    rng = np.random.default_rng(0)
    x = X[0]
    z_samples = []
    for _trial in range(2000):
        z_samples.append(rng.normal(0, 0.05, 784) @ x)
    z_samples = np.array(z_samples)
    variance(z_samples), 784 * 0.05 ** 2 * np.mean(x * x)
    return (z_samples,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.14 measured, 0.135 predicted: the rule holds (the small difference is the luck of
    2,000 draws). Then the bend. The `z`s are centred on 0 (each knob is as
    likely to be + as −), so about half of them are negative, and their mean square is their
    variance. ReLU sets the negative half to 0 and keeps the positive half, so the mean of the
    squares should be cut in half:
    """)
    return


@app.cell
def _(z_samples):
    np.mean(np.maximum(z_samples, 0) ** 2), np.mean(z_samples ** 2) / 2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    0.073 against 0.071: half. So after one layer, the mean square is
    $n \times s^2 \times (\text{mean of } x^2) / 2$. This layer's output is the next layer's input, so this mean
    square becomes the next layer's "mean of $x^2$". For it to stay the same from layer to layer,
    the factor in front must be 1: $n \times s^2 / 2 = 1$, that is

    > $s^2 = 2 / n$, so the typical starting size is $s = \sqrt{2 / n}$

    For the first layer, $n = 784$ inputs: $\sqrt{2 / 784} \approx 0.05$. For the second, $n = 64$:
    $\sqrt{2 / 64} \approx 0.18$. Chapter 08's two numbers. This starting rule is called **He
    initialisation**, after the researcher Kaiming He.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: a starting size

    A layer takes 200 inputs. By the rule above, what variance $s^2$ should its starting knobs have? And what typical size $s$?
    """)
    return


@app.cell(hide_code=True)
def _():
    he_boxes = hand.boxes(['s squared', 's'], step=0.01)
    return (he_boxes,)


@app.cell(hide_code=True)
def _(he_boxes):
    hand.marked(he_boxes, {'s squared': 0.01, 's': 0.1}, {
        's squared': r"""$2 / n$ with $n = 200$""",
        's': r"""the square root of your first answer""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Why it matters shows up in deeper networks. Below, pictures pass through 6 layers of 64 ReLU
    detectors, with three choices of starting size. The numbers are $\sqrt{\text{mean of the squares}}$
    of the values after each layer (the input pixels start at about 0.33). As in chapter 08,
    Python writes very small numbers with `e`: `1.18e-05` means $1.18 \times 0.00001$, that is
    0.0000118; the number after `e-` says how many places the point moves left.
    """)
    return


@app.cell(hide_code=True)
def _(X):
    _rows = ["| starting size | layer 1 | 2 | 3 | 4 | 5 | 6 |", "|---|---|---|---|---|---|---|"]
    for _name, _size in (("0.01", lambda n: 0.01), ("0.3", lambda n: 0.3), (r"$\sqrt{2 / n}$", lambda n: np.sqrt(2 / n))):
        _r = np.random.default_rng(0)
        _h = X[:1000]
        _cells = []
        for _layer in range(6):
            _n = _h.shape[1]
            _h = np.maximum(_h @ _r.normal(0, _size(_n), (_n, 64)), 0)
            _cells.append(f"{np.sqrt(np.mean(_h ** 2)):.3g}")
        _rows.append(f"| {_name} | " + " | ".join(_cells) + " |")
    mo.md("\n".join(_rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Too small and the values fade to 0 within a few layers: no signal, and (since blame is
    multiplied by the same small knobs on the way back) no blame either. Too big and they grow layer after layer
    (from 1.8 after layer 1 to 24 after layer 6, about 13 times bigger).
    $\sqrt{2 / n}$ keeps them steady. That's why deep networks need a reasoned start.

    ## Small batches

    Each step in chapter 08 used all 10,000 pictures to compute one slope. But the slope is a
    **mean** over pictures, and a mean of 100 randomly chosen pictures is already a decent
    estimate of the mean over all 10,000: rougher, but 100 times cheaper. So we take many small,
    rough steps instead of a few big, exact ones:

    1. Shuffle the pictures into a random order.
    2. Cut that order into **batches** of, say, 100 pictures.
    3. Take one step per batch, using the slope from just those 100.

    One pass through all the batches is called an **epoch**: 10,000 pictures in batches of 100
    gives 100 steps per epoch, for the price of one full step.
    """)
    return





@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: shuffling and cutting
    - `rng.permutation(n)` gives the numbers $0, 1, \ldots, n - 1$ in a random order.
    - `range(0, n, size)` counts from 0 in jumps of `size`: `range(0, 10, 4)` gives 0, 4, 8.
    - `a[start:stop]` is a **slice**: the items at positions `start`, `start + 1`, …, up to but not
      including `stop`. `a[4:8]` is 4 items. If `stop` is past the end, the slice just stops at the end.
    - An array can be indexed with a whole list of positions: `X[[3, 0, 7]]` is a matrix made of rows
      3, 0 and 7 of `X`. So if `batch` holds positions, `X[batch]` picks that batch's pictures.
    ///

    ## By hand: cutting into batches

    Ten pictures are cut into batches of 4. How many batches is that? And how many pictures are in the last one?
    """)
    return


@app.cell(hide_code=True)
def _():
    cut_boxes = hand.boxes(['batches', 'in the last batch'])
    return (cut_boxes,)


@app.cell(hide_code=True)
def _(cut_boxes):
    hand.marked(cut_boxes, {'batches': 3, 'in the last batch': 2}, {
        'batches': r"""the slices start at positions 0, 4 and 8""",
        'in the last batch': r"""positions 8 and 9 are all that's left""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: batches

    Write `batches(n, size, rng)`. It shuffles the positions $0$ to $n - 1$ once, cuts that shuffled order into consecutive slices of `size` positions, and gives back the slices as a list of arrays.

    For `n = 10` and `size = 4` the list must hold three arrays, with 4, 4 and 2 positions, and together they must hold every position from 0 to 9 exactly once.
    """)
    return


@app.function
def batches(n, size, rng):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "batches", batches, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Shuffle once, before the loop. Then a `for` loop over the starting positions `range(0, n, size)`, appending one slice of the shuffled order each time.""",
        r"""Slice the shuffled order, not `range(n)`. The `return` comes after the loop, lined up with `for`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is a full training function built from these pieces. It starts with He sizes, and for
    each epoch takes one step per batch. (Ignore `decay` for now; it's 0 until the last section.
    `len(X)` on a matrix is its number of rows.)

    The cell after it trains for 5 epochs and times it with `time.perf_counter()`, a clock in
    seconds. `forward(X_test, *knobs)` passes the four knobs in `knobs` as four separate inputs;
    the `*` unpacks them.
    """)
    return


@app.function
def train(X, Y, epochs, batch_size, rate, decay, seed):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2 / 784), (784, 64))
    b1 = np.zeros(64)
    W2 = rng.normal(0, np.sqrt(2 / 64), (64, 10))
    b2 = np.zeros(10)
    for epoch in range(epochs):
        for batch in batches(len(X), batch_size, rng):
            Z, H, S = forward(X[batch], W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward(X[batch], Y[batch], Z, H, S, W2)
            W1 = W1 - rate * (dW1 + decay * W1)
            b1 = b1 - rate * db1
            W2 = W2 - rate * (dW2 + decay * W2)
            b2 = b2 - rate * db2
    return W1, b1, W2, b2


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: how many steps

    Read `train`. The next cell calls it with 5 epochs and batches of 100, on the 10,000 pictures. How many steps will it take, that is, how many times will the knobs change? And how many pictures will it look at in all?
    """)
    return


@app.cell(hide_code=True)
def _():
    step_boxes = hand.boxes(['steps', 'pictures looked at'])
    return (step_boxes,)


@app.cell(hide_code=True)
def _(step_boxes):
    hand.marked(step_boxes, {'steps': 500, 'pictures looked at': 50000}, {
        'steps': r"""the inner loop runs once per batch: 100 batches in each of 5 epochs""",
        'pictures looked at': r"""every epoch goes through all 10,000 pictures once""",
    }, LOCALE)
    return


@app.cell
def _(X, X_test, Y, test_labels):
    started = time.perf_counter()
    knobs = train(X, Y, 5, 100, 0.5, 0.0, 0)
    seconds = time.perf_counter() - started
    accuracy(forward(X_test, *knobs)[2], test_labels), seconds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Five epochs reach about **94%**, as chapter 08's 300 full steps did. The second number is
    the seconds it took: well under one, against about 10 for chapter 08. 500 small steps look at as many pictures as 5 full ones. Each small step is noisier,
    but there are 100 times more of them. This is how every large model is trained: nobody
    computes a slope over a whole dataset of billions of examples.

    ## Momentum: remember the direction

    Back to chapter 02's problem. On Mia's raw temperatures the slope for `w` is huge (each error is
    multiplied by the temperature, which is in the tens) and the slope for `b` is small, so a learning rate small enough for `w` makes
    `b` crawl. Let's pack both knobs into one vector `k = [w, b]` and watch plain steps from
    chapter 02's start, `[6, 0]`, at about the largest learning rate that doesn't blow up (0.0016):
    """)
    return


@app.function
def mia_gradient(k):
    t = np.array(temps)
    s = np.array(sales)
    e = k[0] * t + k[1] - s
    return np.array([np.mean(2 * e * t), np.mean(2 * e)])


@app.function
def mia_loss(k):
    e = k[0] * np.array(temps) + k[1] - np.array(sales)
    return np.mean(e * e)


@app.cell
def _():
    k_plain = np.array([6.0, 0.0])
    for _step in range(500):
        k_plain = k_plain - 0.0016 * mia_gradient(k_plain)
    k_plain, mia_loss(k_plain)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After 500 steps, `b` has only reached about −11 (the best line has `b` about −107) and the
    loss is still about 640.

    Each step for `b` is small, but they all point the **same way**. Momentum takes advantage of
    that: it keeps a running **velocity** that adds up recent slopes, each step forgetting a bit
    of the past:

    > $\text{velocity} = 0.9 \times \text{velocity} + \text{slope}$, then $\text{knob} = \text{knob} - \text{rate} \times \text{velocity}$

    If the slope stays the same, the velocity grows until it stops changing, that is until
    $V = 0.9 \times V + \text{slope}$. Take $0.9 \times V$ from both sides: $0.1 \times V = \text{slope}$, so
    $V = 10 \times \text{slope}$: steady progress becomes up to 10 times faster.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: a velocity building up

    Start with a velocity of 0, and say the slope is 1 at every step. What is the velocity after 1, 2 and 3 steps?
    """)
    return


@app.cell(hide_code=True)
def _():
    momentum_boxes = hand.boxes(['after 1 step', 'after 2 steps', 'after 3 steps'], step=0.01)
    return (momentum_boxes,)


@app.cell(hide_code=True)
def _(momentum_boxes):
    hand.marked(momentum_boxes, {'after 1 step': 1, 'after 2 steps': 1.9, 'after 3 steps': 2.71}, {
        'after 1 step': r"""$0.9 \times 0 + 1$""",
        'after 2 steps': r"""$0.9 \times$ your first answer, plus 1""",
        'after 3 steps': r"""$0.9 \times$ your second answer, plus 1""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It climbs towards 10, the steady value worked out above, quickly at first and then more and more slowly.

    ## Exercise 3: momentum_step

    Write `momentum_step(w, velocity, grad, rate)`, doing the two formulas above: first the new velocity from the old one and the slope `grad`, then the new knobs.
    Give back the new knobs and the new velocity. It works for a single knob or for a whole array of knobs at once, position by position.
    """)
    return


@app.function
def momentum_step(w, velocity, grad, rate):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "momentum_step", momentum_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two lines: one for the velocity, one to give back the pair.""",
        r"""The knobs move by `rate` times the *new* velocity, so update the velocity first.""",
    ], LOCALE)
    return


@app.cell
def _():
    k_momentum = np.array([6.0, 0.0])
    velocity = np.zeros(2)
    for _step in range(500):
        k_momentum, velocity = momentum_step(k_momentum, velocity, mia_gradient(k_momentum), 0.003)
    k_momentum, mia_loss(k_momentum)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same 500 steps (at 0.003, about the largest rate momentum takes here without blowing up): the
    loss is down to about 41, against 640 for plain steps.

    ## Adam: a step size for every knob

    Momentum helps, but $w$ and $b$ still share one learning rate. **Adam** gives each knob its
    own scale. It keeps an **average** of recent slopes, $m = 0.9 \times m + 0.1 \times \text{slope}$. This is
    momentum's velocity times 0.1, which makes it a true average: if the slope stays the same,
    $m$ settles at the slope itself (take $0.9 \times m$ from both sides of $m = 0.9 \times m + 0.1 \times \text{slope}$: $0.1 \times m = 0.1 \times \text{slope}$), not 10 times it. Next to it, Adam
    keeps an average of the slope **squared**, and divides by its square root, the knob's recent
    typical slope size:

    > $$\text{step} = \text{rate} \times \frac{\text{average slope}}{\sqrt{\text{average squared slope}}}$$

    For a knob with huge slopes ($w$), the division makes the step smaller; for a knob with small
    slopes ($b$), bigger. Every knob ends up moving at a similar pace: on the very first step,
    each moves by exactly `rate` (then the average slope is just the slope, and $\sqrt{\text{slope}^2}$ is its
    size, so their ratio is +1 or −1). After that the averages settle and the steps shrink as
    the knobs near the best line. The cell below shows only the end, after 500 steps at rate
    1.0; on the way there, $b$ passes about −42 at step 100 and −99 at step 300. That is exactly what we did by hand when we rescaled temperatures in chapters 04 and
    05, now done automatically, knob by knob.

    One detail: both averages start at 0, so in the first steps they're too small (after one step
    the average slope is only $0.1 \times \text{slope}$). Adam corrects this by dividing by $1 - 0.9^t$ and
    $1 - 0.999^t$ (in code, powers are `**`: `0.9 ** t`), where $t$ counts the steps from 1: after step 1 that divides by 0.1 and 0.001,
    which undoes the shrinking exactly; as $t$ grows, the correction fades to dividing by 1. (The
    `1e-8`, that is 0.00000001, is there so we never divide by zero.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is one whole step of Adam, with $g$ for the slope and $t$ for the number of the step:

    > $$m = 0.9 \times m + 0.1 \times g \qquad v = 0.999 \times v + 0.001 \times g^2$$

    > $$\hat{m} = \frac{m}{1 - 0.9^t} \qquad \hat{v} = \frac{v}{1 - 0.999^t}$$

    > $$w \leftarrow w - \text{rate} \times \frac{\hat{m}}{\sqrt{\hat{v}} + 0.00000001}$$

    $\hat{m}$ and $\hat{v}$ (read "m hat" and "v hat") are the corrected averages.

    ## By hand: Adam's first step

    A knob's very first step: $m$ and $v$ start at 0, $t = 1$, the slope is $g = 4$, and the rate is 0.5. Work through the formulas. The last box is how far the knob moves. (Leave out the tiny 0.00000001.)
    """)
    return


@app.cell(hide_code=True)
def _():
    adam_boxes = hand.boxes(['m', 'v', 'm hat', 'v hat', 'the knob moves by'], step=0.001)
    return (adam_boxes,)


@app.cell(hide_code=True)
def _(adam_boxes):
    hand.marked(adam_boxes, {'m': 0.4, 'v': 0.016, 'm hat': 4, 'v hat': 16, 'the knob moves by': 0.5}, {
        'm': r"""$0.9 \times 0 + 0.1 \times 4$""",
        'v': r"""$0.999 \times 0 + 0.001 \times 4^2$""",
        'm hat': r"""$m / (1 - 0.9)$""",
        'v hat': r"""$v / (1 - 0.999)$""",
        'the knob moves by': r"""$\text{rate} \times \hat{m} / \sqrt{\hat{v}}$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The knob moves by exactly the rate, 0.5, as promised. Try it in your head with a slope of 4,000: the answer is still 0.5.

    ## Exercise 4: adam_step

    Write `adam_step(w, m, v, grad, t, rate)`, doing the formulas above in order. Give back the new knobs and both averages, the uncorrected `m` and `v`,
    because those are what the next step needs. In code, a power is `**`, and `1e-8` is 0.00000001.

    After the check, a cell runs 500 Adam steps on Mia's problem. `range(1, 501)` counts
    1, 2, …, 500, so `t` starts at 1.
    """)
    return


@app.function
def adam_step(w, m, v, grad, t, rate):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("09", "adam_step", adam_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Five lines, one per formula: `m`, `v`, the two corrected averages under new names, and the `return`.""",
        r"""Don't store the corrected averages back into `m` and `v`: the uncorrected ones are given back. If you get `nan`, a bracket is probably missing around `np.sqrt(...) + 1e-8`.""",
    ], LOCALE)
    return


@app.cell
def _():
    k_adam = np.array([6.0, 0.0])
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, 501):
        k_adam, m, v = adam_step(k_adam, m, v, mia_gradient(k_adam), t, 1.0)
    k_adam, mia_loss(k_adam)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After the same 500 steps, Adam sits on the best possible line: `w` about 11.9, `b` about
    −107, loss about 26.8, on raw temperatures, without any rescaling. Adam (with small changes) trains almost every
    large model today, including the transformers later in this course.

    ## Learning or memorising?

    Last loose end. Train the same network on only **500** pictures, for 200 epochs, and compare
    its accuracy on those 500 with its accuracy on the 2,000 test pictures. (The cell after the
    text runs it twice: with `decay` 0 and with `decay` 0.003, explained below. It shows, in
    order: training accuracy, test accuracy, and test accuracy with decay.)
    """)
    return


@app.cell
def _(X, X_test, Y, test_labels, train_labels):
    small = train(X[:500], Y[:500], 200, 50, 0.5, 0.0, 0)
    small_decay = train(X[:500], Y[:500], 200, 50, 0.5, 0.003, 0)
    [
        accuracy(forward(X[:500], *small)[2], train_labels[:500]),
        accuracy(forward(X_test, *small)[2], test_labels),
        accuracy(forward(X_test, *small_decay)[2], test_labels),
    ]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    100% right on its 500 training pictures, but about 87% on new ones. The network has enough
    knobs to **memorise** its 500 pictures, scratches and smudges included, and memorised details
    don't carry over to new handwriting. This is called **overfitting**: the training score
    looks perfect and says nothing about new data. It's why we always keep test data aside.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: reading the two scores

    The network got 100% of its 500 training pictures right, and 87.25% of the 2,000 test pictures. How many pictures did it get wrong in each set?
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_boxes = hand.boxes(['training pictures wrong', 'test pictures wrong'])
    return (fit_boxes,)


@app.cell(hide_code=True)
def _(fit_boxes):
    hand.marked(fit_boxes, {'training pictures wrong': 0, 'test pictures wrong': 255}, {
        'training pictures wrong': r"""100% right leaves none""",
        'test pictures wrong': r"""$100 - 87.25 = 12.75$ percent of 2,000""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Weight decay** is a standard counter-measure. Add a small penalty for big knobs to the
    loss: $(\text{decay} / 2) \times (\text{sum of every weight squared})$, over the weights in `W1` and `W2`
    (the $b$s, usually called *biases*, are left alone, hence the name *weight* decay). Nudging one weight $w$ moves
    only its own square; the others don't change. By chapter 02's rule, the slope of $w^2$ is
    $2 \times w$, so the slope of $(\text{decay} / 2) \times w^2$ is $(\text{decay} / 2) \times 2 \times w = \text{decay} \times w$, and the
    step rule becomes

    > $\text{weight} = \text{weight} - \text{rate} \times (\text{slope} + \text{decay} \times \text{weight})$

    which is the `+ decay * W1` in `train`: every step also shrinks every weight a little towards
    0, so a weight stays big only if the data keeps asking for it. Here the effect is tiny: the
    last number above is 87.4% against 87.25%, three more pictures out of 2,000. Weight decay
    matters more in big models trained for a long time. The strongest cure is simpler: **more data**. The same network on all 10,000 pictures got about
    94%.

    ## What's next

    You've now built, by hand, everything a deep-learning library does: the forward pass, the
    backward pass through matrices, a good start, mini-batches, momentum and Adam. In
    **Chapter 10** you'll meet **PyTorch**, the library most of the field uses, and see it do all
    of this for you, then check that its answers match yours.

    ## What you can do now

    - measure spread with the variance, and its square root, the typical size
    - explain why starting at zero fails, and derive the start $\sqrt{2 / n}$
    - train with shuffled mini-batches and epochs, using slices and `rng.permutation`
    - speed up training with momentum (a velocity that adds up recent slopes)
    - give every knob its own pace with Adam, including its start-up correction
    - recognise overfitting from training and test scores, and counter it with weight decay or
      more data

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** mini-batch training is *stochastic gradient descent* (SGD). The
    start $\sqrt{2 / n}$ is *He* (or *Kaiming*) initialisation. *Dropout* and *batch normalisation*
    are two more tools in this family; the transformer chapters use *layer normalisation*.
    """)
    return


if __name__ == "__main__":
    app.run()
