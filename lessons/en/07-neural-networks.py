import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="07 · Neural networks")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.plot import chart

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 · Neural networks

    Chapter 06's model got about 90% of the test digits right. Train it longer and it creeps up
    to about 91%, and stops there. The problem isn't training: it's what the model **can**
    express. Each digit has one template, and a picture's score is a weighted sum of its pixels.
    A weighted sum can say "ink here adds, ink there subtracts". It can't say "a 4 has its top
    either open **or** closed": two different shapes, both a 4. A single template has to blend
    both into one blurry 4 that fits neither well, and a picture with ink in both places would
    score highest of all, though no 4 looks like that.

    In this chapter you'll add the missing ingredient, a **bend**, and build a **neural network**.

    ## A bend

    Recall Mia's heatwave from chapter 04: sales climb with the temperature until about 33 °C,
    then fall. A straight line can't follow that. Now meet the simplest possible bend, called
    **ReLU**:

    > $\operatorname{relu}(z) = z$ when $z$ is positive, and $0$ otherwise

    For example $\operatorname{relu}(-3) = 0$, $\operatorname{relu}(0) = 0$ and $\operatorname{relu}(2) = 2$. It's flat at 0, then turns a
    corner at $z = 0$ and rises like a line. On its own that's
    dull. But **add** it to a line and you get a line that changes direction at a point you
    choose. Here is a line with one bend at 33 °C (the numbers −105, 12 and −20 are picked by
    hand):

    > $$\text{guess} = -105 + 12 \times t - 20 \times \operatorname{relu}(t - 33)$$

    Below 33 °C, $t - 33$ is negative, the ReLU gives 0, and it's the line $-105 + 12 \times t$. Above
    33 °C, the ReLU adds $t - 33$, and $- 20 \times$ it bends the line down: from then on each degree
    adds $12 - 20 = -8$.
    """)
    return


@app.cell(hide_code=True)
def _():
    _temps = [14, 17, 20, 22, 25, 28, 31, 33, 35, 37, 39]
    _sales = [63, 90, 137, 148, 193, 234, 257, 288, 280, 262, 236]
    _bent = [(t / 10, -105 + 12 * t / 10 - 20 * max(0, t / 10 - 33)) for t in range(120, 411)]
    _loss = np.mean([(-105 + 12 * t - 20 * max(0, t - 33) - s) ** 2 for t, s in zip(_temps, _sales)])
    mo.vstack([
        chart((10, 42), (0, 320), "temperature (°C)", "ice creams sold",
              dots=list(zip(_temps, _sales)), curve=_bent),
        mo.md(f"Loss of this bent line on the 11 days: **{_loss:.0f}**. "
              "The best straight line has about 810; chapter 04's curve had about 283."),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One bend already beats chapter 04's curve. With more bends, each at its own place and with
    its own size, you can follow any shape, one straight piece at a time. That's the whole idea
    of a neural network: many simple pieces, each with a bend, added up.

    In NumPy, `np.maximum(Z, 0)` compares each number with 0, position by position, and keeps
    the larger: exactly ReLU for a whole array at once.

    ## Exercise 1: relu

    Write `relu(Z)`: `return np.maximum(Z, 0)`.
    """)
    return


@app.function
def relu(Z):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "relu", relu, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A hidden layer

    Now the digits. Instead of going straight from pixels to ten scores, we go through a middle
    step of 64 numbers:

    1. `Z = X @ W1 + b1`: 64 weighted sums of the pixels, exactly like chapter 06's scores but 64
       of them. `W1` has shape `(784, 64)`: 64 templates. `Z` has shape `(n, 64)`.
    2. `H = relu(Z)`: bend each one, just as $\operatorname{relu}(t - 33)$ bent the line above ($b_1$ sets where
       each bend sits). A template that matches poorly (negative sum) gives 0; one
       that matches well gives how well. Each column of $H$ is a little **detector**: it says
       how strongly its pattern is present in the picture, and stays silent otherwise.
    3. `S = H @ W2 + b2`: ten scores, each a weighted sum of the 64 detectors. `W2` has shape
       `(64, 10)`.

    `H` is called a **hidden layer** (hidden because it's neither the input nor the answer: it
    only exists in between), and the whole thing is a **neural network**. Now the score for 4 can
    say, for example, "detector 12 (open top) **or** detector 40 (closed top)" (made-up numbers):
    each detector that fires adds its share, and a silent one adds nothing.

    The scores then go through chapter 06's softmax and cross-entropy, unchanged. Here they are
    again, with chapter 06's `accuracy` and `one_hot`. They are finished; don't change them:
    """)
    return


@app.function
def softmax(S):
    S = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S)
    return E / np.sum(E, axis=1, keepdims=True)


@app.function
def cross_entropy(P, Y):
    return -np.mean(np.sum(Y * np.log(P), axis=1))


@app.function
def accuracy(S, labels):
    return np.mean(np.argmax(S, axis=1) == labels)


@app.function
def one_hot(labels):
    Y = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: forward

    Write `forward(X, W1, b1, W2, b2)`, returning all three steps (we'll need `Z` and `H` again
    for the slopes):

    1. `Z = X @ W1 + b1`
    2. `H = relu(Z)`
    3. `S = H @ W2 + b2`
    4. `return Z, H, S`
    """)
    return


@app.function
def forward(X, W1, b1, W2, b2):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "forward", forward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why the bend matters

    Without the ReLU, step 3 would compute `(X @ W1 + b1) @ W2 + b2`. Leave the `b`s aside for a
    moment: `@` can be grouped either way, like ordinary multiplication, so
    `(X @ W1) @ W2 = X @ (W1 @ W2)`, and `W1 @ W2` is a single `(784, 10)` matrix: one template per
    digit again. Two layers without a bend are no better than one. Check the grouping with small
    random matrices (the new tools in this cell are explained right after it):
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (3, 4))
    B = rng.normal(0, 1, (4, 5))
    C = rng.normal(0, 1, (5, 2))
    (A @ B) @ C - A @ (B @ C)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    All (almost) zeros: the same matrix either way. Numbers like `2.22044605e-16` mean $2.2 \times 0.0000000000000001$, rounding in the last digit, and `e+00` on the zeros means "times 1":
    just 0.

    This cell used something new: **random numbers**. `np.random.default_rng(0)` makes a
    random-number generator; the `0` is a starting point (a *seed*) that makes it produce the
    same numbers every time you run the page. `rng.normal(0, 1, (3, 4))` gives a `(3, 4)` array
    of random numbers spread around 0, most of them between −2 and 2 (`0` is the centre, `1` the
    typical size).

    We need random numbers to start the network. If every knob started at 0, every hidden
    detector would be identical and stay identical forever. So we start with small random knobs:
    typical size 0.05 for `W1` and 0.18 for `W2`. (Chapter 08 explains both: why not zero, and
    why these sizes.) The next cell uses its own generator with seed 1, so your start is the
    same as the one described here.

    ## The data, and a first forward pass
    """)
    return


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    Y = one_hot(train_labels)
    start_rng = np.random.default_rng(1)
    W1_start = start_rng.normal(0, 0.05, (784, 64))
    b1_start = np.zeros(64)
    W2_start = start_rng.normal(0, 0.18, (64, 10))
    b2_start = np.zeros(10)
    Z0, H0, S0 = forward(X, W1_start, b1_start, W2_start, b2_start)
    Z0.shape, H0.shape, S0.shape, cross_entropy(softmax(S0), Y)
    return (
        H0, S0, W1_start, W2_start, X, X_test, Y, Z0, b1_start, b2_start,
        test_labels, train_labels,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Shapes `(10000, 64)`, `(10000, 64)` and `(10000, 10)`, and a loss of about 2.41, a little
    worse than the 2.303 of knowing nothing: random knobs make random guesses.

    ## Slopes through two layers

    Now the gradient. The loss depends on the knobs through a chain of steps
    (`W1` → `Z` → `H` → `S` → loss), and chapter 03 tells us what to do with a chain: walk it
    backwards, passing blame. We take it one step at a time, starting from the end.

    We name each blame with a `d` in front: `dH` is the blame of `H`, `dW1` the blame (slope) of
    `W1`, and so on.

    **Scores.** The loss is chapter 06's cross-entropy of `softmax(S)`, so the blame of the scores
    is chapter 06's $G = (P - Y) / n$, with $P = \operatorname{softmax}(S)$.

    **`W2` and `b2`.** `S = H @ W2 + b2` is exactly chapter 06's `X @ W + b`, with `H` in place of
    `X`. So the same result holds: `dW2 = H.T @ G` and `db2 = np.sum(G, axis=0)`.

    **$H$.** This one is new: chapter 06 never needed the slope for its *inputs*. Take one hidden
    value, `H[i, j]` (picture $i$, detector $j$). It appears in all ten scores of picture $i$:
    written out, $S[i, c] = H[i, 0] \times W_2[0, c] + \ldots + H[i, 63] \times W_2[63, c] + b_2[c]$, so by chapter
    03's rule for a product (in $a \times c$, nudging $a$ moves the result by $c$), nudging `H[i, j]` moves `S[i, c]` by `W2[j, c]` per unit: that's the local
    slope. One value feeding ten arrows: **blames add** (chapter 03). So

    > blame of `H[i, j]` = $G[i, 0] \times W_2[j, 0] + G[i, 1] \times W_2[j, 1] + \ldots + G[i, 9] \times W_2[j, 9]$

    That's row `i` of `G` dotted with row `j` of `W2`, which is row `i` of `G` dotted with
    **column** `j` of `W2.T`. Since entry `[i, j]` of `A @ B` is row `i` of `A` dotted with
    column `j` of `B`, that's entry `[i, j]` of `G @ W2.T`. For every `i` and `j` at once:
    `dH = G @ W2.T`, with shape
    `(n, 10) @ (10, 64) = (n, 64)`, the same shape as `H`.

    **The bend.** `H = relu(Z)`, position by position. The slope of $\operatorname{relu}(z)$: where $z$ is
    positive, $\operatorname{relu}(z) = z$, slope 1; where $z$ is negative, $\operatorname{relu}(z) = 0$, slope 0. (Exactly at
    0 there's a corner; we take slope 0 there.) So the bend passes blame straight through where
    $Z > 0$ and blocks it everywhere else:

    > `dZ = dH * (Z > 0)`

    $Z > 0$ gives `True`/`False` for each position, and multiplying by it counts `True` as 1 and
    `False` as 0. An array like this, used to let some values through and block others, is called
    a **mask**. A detector that was silent for a picture takes no blame for it: nudging its
    knobs a tiny bit wouldn't change anything.

    **`W1` and `b1`.** `Z = X @ W1 + b1` is chapter 06's step once more, with `dZ` as the blame:
    `dW1 = X.T @ dZ`, `db1 = np.sum(dZ, axis=0)`.

    That's backpropagation through a whole network. Every step is chapter 03's rule; the
    matrices only do it for all pictures and all knobs at once.

    ## Exercise 3: backward

    Write `backward(X, Y, Z, H, S, W2)`, returning `dW1, db1, dW2, db2`:

    1. `G = (softmax(S) - Y) / len(X)`
    2. `dW2 = H.T @ G` and `db2 = np.sum(G, axis=0)`
    3. `dH = G @ W2.T`
    4. `dZ = dH * (Z > 0)`
    5. `dW1 = X.T @ dZ` and `db1 = np.sum(dZ, axis=0)`
    6. `return dW1, db1, dW2, db2`
    """)
    return


@app.function
def backward(X, Y, Z, H, S, W2):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "backward", backward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checking it by nudging

    A whole network's worth of new formulas deserves a check. At the start, nudge one knob of the
    first layer, `W1[406, 0]` (from pixel 406, row 14 and column 14: the middle of the picture,
    into detector 0),
    and see how the loss moves; then do the same for a knob of the second layer, `W2[5, 3]`.
    (This cell needs Exercises 2 and 3.)
    """)
    return


@app.cell
def _(H0, S0, W1_start, W2_start, X, Y, Z0, b1_start, b2_start):
    h = 0.0001
    dW1, db1, dW2, db2 = backward(X, Y, Z0, H0, S0, W2_start)
    base = cross_entropy(softmax(S0), Y)

    nudge1 = np.zeros((784, 64))
    nudge1[406, 0] = h
    S1 = forward(X, W1_start + nudge1, b1_start, W2_start, b2_start)[2]

    nudge2 = np.zeros((64, 10))
    nudge2[5, 3] = h
    S2 = forward(X, W1_start, b1_start, W2_start + nudge2, b2_start)[2]

    [(cross_entropy(softmax(S1), Y) - base) / h, dW1[406, 0],
     (cross_entropy(softmax(S2), Y) - base) / h, dW2[5, 3]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Items 0 and 1 are the nudge and the formula for `W1[406, 0]`; items 2 and 3 for `W2[5, 3]`.
    They agree in their first three or four digits after the zeros (−0.01332… and −0.01333…;
    0.0029739… twice): a nudge is never exact. (A corner pixel would give slope exactly 0 and
    test nothing, so we picked one in the middle.)

    ## Training

    Same loop as chapter 06, with four sets of knobs instead of two. 300 steps over all 10,000
    pictures take a while (up to a minute), so the training waits for the button. The first line
    of the training cell, `mo.stop(...)`, only makes it wait for the button.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the network (may take up to a minute)")
    train_button
    return (train_button,)


@app.cell
def _(W1_start, W2_start, X, X_test, Y, b1_start, b2_start, test_labels, train_button, train_labels):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    W1, b1, W2, b2 = W1_start, b1_start, W2_start, b2_start
    for step in range(300):
        Z, H, S = forward(X, W1, b1, W2, b2)
        gW1, gb1, gW2, gb2 = backward(X, Y, Z, H, S, W2)
        W1 = W1 - 0.5 * gW1
        b1 = b1 - 0.5 * gb1
        W2 = W2 - 0.5 * gW2
        b2 = b2 - 0.5 * gb2
    S_train = forward(X, W1, b1, W2, b2)[2]
    S_test = forward(X_test, W1, b1, W2, b2)[2]
    cross_entropy(softmax(S_train), Y), accuracy(S_train, train_labels), accuracy(S_test, test_labels)
    return (W1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The three numbers are the training loss, the training accuracy and the test accuracy. The
    last, about 0.944, means about **94%** of the test pictures right, against about 91% for a
    single-layer model trained for as long: the hidden layer cut the mistakes by about 40%.

    ## What the detectors look for

    Each column of `W1` is one detector's 784 knobs, drawn as a $28 \times 28$ map (blue: ink here
    makes it fire; red: ink here silences it). Here are the first 16:
    """)
    return


@app.cell(hide_code=True)
def _(W1):
    images.weight_maps([W1[:, j] for j in range(16)], [f"detector {j}" for j in range(16)], size=72)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    They look much noisier than chapter 06's templates, and the noise is telling. Near the edges,
    almost every picture is white paper: many of those pixels are 0 in every picture, and the
    rest almost always are. Their knobs take little or no blame (the slope `X.T @ dZ` multiplies
    by those zeros) and keep (nearly) their random starting values. Towards the middle, where the ink is,
    you can make out smoother patches: pieces of strokes that each detector learned to respond
    to, without anyone telling it what to look for. The second layer learns which mix of
    detectors makes each digit.

    ## What's next

    It works, but training was slow: every one of the 300 steps looked at all 10,000 pictures.
    And we chose the starting sizes 0.05 and 0.18 without saying why. In **Chapter 08** you'll
    learn the tricks that make training fast and reliable: small batches, a smarter step rule,
    a reasoned start, and how to keep a model honest on new data.

    ## What you can do now

    - explain why a weighted sum can't express "this or that", and how a bend fixes it
    - build a neural network with one hidden layer: `X @ W1 + b1`, ReLU, `H @ W2 + b2`
    - explain why two layers without a bend are no better than one
    - backpropagate through a network with matrices: `H.T @ G`, `G @ W2.T`, the ReLU mask, `X.T @ dZ`
    - check a network's gradient by nudging, and train it

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** this network is a *multilayer perceptron* (MLP); ReLU stands for
    *rectified linear unit*. The fact that sums of bends can follow any shape is the *universal
    approximation theorem*.
    """)
    return


if __name__ == "__main__":
    app.run()
