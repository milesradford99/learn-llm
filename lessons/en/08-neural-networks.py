import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="08 · Neural networks")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.diagrams import grids, network, pipeline
    from learn_llm.plot import chart

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 08 · Neural networks

    Chapter 07's model got about 90% of the test digits right. Train it longer and it creeps up
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
    corner at $z = 0$ and rises like a line:
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-4, 4), (-1, 4), "z", "relu(z)", curve=[(-4, 0), (0, 0), (4, 4)], dots=[(-3, 0), (0, 0), (2, 2)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Think of a ramp that starts at a wall: anything left of the wall is on the floor, anything right
    of it climbs. On its own that's
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
    One bend already beats chapter 04's curve. Move the bend yourself: where it sits, and how
    much the slope changes there. The loss tells you how well each choice fits.
    """)
    return


@app.cell(hide_code=True)
def _():
    bend_at = mo.ui.slider(24, 39, value=33, label="bend at (°C)", show_value=True)
    bend_turn = mo.ui.slider(-30, 0, step=2, value=-20, label="slope change after the bend", show_value=True)
    mo.vstack([bend_at, bend_turn])
    return bend_at, bend_turn


@app.cell(hide_code=True)
def _(bend_at, bend_turn):
    _temps = [14, 17, 20, 22, 25, 28, 31, 33, 35, 37, 39]
    _sales = [63, 90, 137, 148, 193, 234, 257, 288, 280, 262, 236]
    _a, _k = bend_at.value, bend_turn.value
    _loss = np.mean([(-105 + 12 * t + _k * max(0, t - _a) - s) ** 2 for t, s in zip(_temps, _sales)])
    mo.vstack([
        chart((10, 42), (0, 320), "temperature (°C)", "ice creams sold", dots=list(zip(_temps, _sales)),
              curve=[(t / 10, -105 + 12 * t / 10 + _k * max(0, t / 10 - _a)) for t in range(120, 411)]),
        mo.md(r"guess $= -105 + 12 \times t {sign} {size} \times \operatorname{{relu}}(t - {at})$, loss **{loss:.0f}**".format(
            sign="-" if _k < 0 else "+", size=abs(_k), at=_a, loss=_loss)),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Set the slope change to 0 and the bend disappears: a straight line again. Put the bend at
    24 °C and the line turns down far too early. Of all the slider settings, 33 °C with a change
    of −20 gives the lowest loss, about 37. Gradient descent will do this search for us, for
    every bend at once.

    With more bends, each at its own place and with
    its own size, you can follow any shape, one straight piece at a time. That's the whole idea
    of a neural network: many simple pieces, each with a bend, added up.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: the bent line

    Use the bent line above, $\text{guess} = -105 + 12 \times t - 20 \times \operatorname{relu}(t - 33)$. What does it guess at 30 °C, and at 40 °C?
    """)
    return


@app.cell(hide_code=True)
def _():
    bend_boxes = hand.boxes(['guess at 30', 'guess at 40'])
    return (bend_boxes,)


@app.cell(hide_code=True)
def _(bend_boxes):
    hand.marked(bend_boxes, {'guess at 30': 255, 'guess at 40': 235}, {
        'guess at 30': r"""below the bend, $\operatorname{relu}(30 - 33) = 0$, so it's just $-105 + 12 \times 30$""",
        'guess at 40': r"""above the bend, $\operatorname{relu}(40 - 33) = 7$: $-105 + 12 \times 40 - 20 \times 7$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: relu

    /// admonition | 🧰 Toolbox: the larger of two
    `np.maximum(A, 2)` compares each number of an array with 2, position by position, and keeps the larger:
    `np.maximum(np.array([1, 5, -3]), 2)` gives `[2, 5, 2]`. The result has the shape of `A`.

    Don't mix it up with `np.max`, which picks out the single largest number of an array.
    ///

    Write `relu(Z)` for an array `Z` of any shape: each positive number stays, and everything else becomes 0.

    For example, `[[-2, 0, 3], [1.5, -0.5, 0]]` must give `[[0, 0, 3], [1.5, 0, 0]]`.
    """)
    return


@app.function
def relu(Z):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("08", "relu", relu, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""ReLU keeps the larger of each number and 0.""",
        r"""One line, with `np.maximum`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A hidden layer

    Now the digits. Instead of going straight from pixels to ten scores, we go through a middle
    step of 64 numbers:

    1. $Z = X W_1 + b_1$: 64 weighted sums of the pixels, exactly like chapter 06's scores but 64
       of them. `W1` has shape `(784, 64)`: 64 templates. `Z` has shape `(n, 64)`.
    2. $H = \operatorname{relu}(Z)$: bend each one, just as $\operatorname{relu}(t - 33)$ bent the line above ($b_1$ sets where
       each bend sits). A template that matches poorly (negative sum) gives 0; one
       that matches well gives how well. Each column of $H$ is a little **detector**: it says
       how strongly its pattern is present in the picture, and stays silent otherwise.
    3. $S = H W_2 + b_2$: ten scores, each a weighted sum of the 64 detectors. `W2` has shape
       `(64, 10)`.

    `H` is called a **hidden layer** (hidden because it's neither the input nor the answer: it
    only exists in between), and the whole thing is a **neural network**. Now the score for 4 can
    say, for example, "detector 12 (open top) **or** detector 40 (closed top)" (made-up numbers):
    each detector that fires adds its share, and a silent one adds nothing.

    Here are the three steps as a picture, with the shape of each value for `n` pictures:
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([('X', '(n, 784)'), ('Z', '(n, 64)'), ('H', '(n, 64)'), ('S', '(n, 10)')],
             ['@ W1 + b1', 'relu', '@ W2 + b2'],
             'X goes through W1 and b1 to give Z, relu gives H, and W2 and b2 give the scores S')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### "This or that" in a tiny world

    Here is the opening problem on $3 \times 3$ pictures. Say a "bar" can be drawn in the **left**
    column or in the **right** column: two shapes, one answer. A picture with ink in **both**
    columns is not a bar.

    A single template, made by averaging (chapter 06), puts 0.5 on both outer columns. Each bar
    lands 3 inked pixels on its half of the template: score 1.5. But the picture with both
    columns lands 6: score 3, the highest of all. A weighted sum can only add up ink, so more ink
    in the right places always means a higher score.

    Now two detectors. Detector A has knobs $+1$ on the left column, $-1$ on the right column
    and $b = -2$; detector B is its mirror image:
    """)
    return


@app.cell(hide_code=True)
def _():
    _a = [[1, 0, -1]] * 3
    _b = [[-1, 0, 1]] * 3
    _marks = lambda grid: {(r, c): "blue" if v > 0 else "red" for r in range(3) for c, v in enumerate(grid[r]) if v}
    mo.vstack([
        grids([
            {"name": "left bar", "values": [[1, 0, 0]] * 3, "marks": {(r, 0): "blue" for r in range(3)}}, "   ",
            {"name": "right bar", "values": [[0, 0, 1]] * 3, "marks": {(r, 2): "blue" for r in range(3)}}, "   ",
            {"name": "both (not a bar)", "values": [[1, 0, 1]] * 3,
             "marks": {(r, c): "blue" for r in range(3) for c in (0, 2)}},
        ], "Three pictures: a bar in the left column, a bar in the right column, and ink in both columns"),
        grids([
            {"name": "knobs of detector A", "values": _a, "marks": _marks(_a)}, "   ",
            {"name": "knobs of detector B", "values": _b, "marks": _marks(_b)},
        ], "Detector A: plus 1 on the left column, minus 1 on the right; detector B the other way round"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The bar's score is $\operatorname{relu}(z_A) + \operatorname{relu}(z_B)$. Work it through for each picture:

    | picture | $z_A$ | $z_B$ | $h_A$ | $h_B$ | network's score | template's score |
    |---|---|---|---|---|---|---|
    | left bar | $3 - 0 - 2 = 1$ | $0 - 3 - 2 = -5$ | 1 | 0 | **1** | 1.5 |
    | right bar | $-5$ | $1$ | 0 | 1 | **1** | 1.5 |
    | both | $3 - 3 - 2 = -2$ | $-2$ | 0 | 0 | **0** | 3 |

    Each detector fires for its own shape and stays silent otherwise, and the ReLU turns
    "silent" into a clean 0 instead of a negative number that would drag the score around. The
    network says "left **or** right, but not both", which no single template can say. That's what
    the hidden layer buys.

    The scores then go through chapter 07's softmax and cross-entropy, unchanged. Here they are
    again, with `accuracy` from chapter 06 and `one_hot` from chapter 07. They are finished; don't change them:
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
    ## By hand: one picture through a tiny network

    Before writing it for 784 pixels, do it for 2. There is one picture, `x = [1, 2]`, with two detectors and two scores. Every `b` is 0, and

    > $$W_1 = \begin{bmatrix} 1 & -3 \\ 1 & 1 \end{bmatrix} \qquad W_2 = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$$

    Column 0 of $W_1$ holds the knobs of detector 0, so $z_0 = 1 \times 1 + 2 \times 1 = 3$. Work out the rest: the other weighted sum, both hidden values, and both scores.

    Here is the same network as a picture. Each arrow carries one knob: the number on it
    multiplies the value at the arrow's start, and a box adds up everything arriving. Follow the
    arrows into a box and you have its weighted sum.
    """)
    return


@app.cell(hide_code=True)
def _():
    tiny_edges = [(0, 0, 0, "1"), (0, 0, 1, "-3"), (0, 1, 0, "1"), (0, 1, 1, "1"),
                  (1, 0, 0, "relu"), (1, 1, 1, "relu"),
                  (2, 0, 0, "1"), (2, 0, 1, "0"), (2, 1, 0, "2"), (2, 1, 1, "1")]
    tiny_heads = ["pixels x", "sums z", "detectors h", "scores s"]
    network([["x0 = 1", "x1 = 2"], ["z0 = 3", "z1 = ?"], ["h0 = ?", "h1 = ?"], ["s0 = ?", "s1 = ?"]],
            tiny_edges, "A network with 2 pixels, 2 detectors and 2 scores; the knobs are written on the arrows", tiny_heads)
    return tiny_edges, tiny_heads


@app.cell(hide_code=True)
def _():
    forward_boxes = hand.boxes(['z1', 'h0', 'h1', 's0', 's1'])
    return (forward_boxes,)


@app.cell(hide_code=True)
def _(forward_boxes):
    hand.marked(forward_boxes, {'z1': -1, 'h0': 3, 'h1': 0, 's0': 3, 's1': 0}, {
        'z1': r"""`x` dotted with column 1 of $W_1$: $1 \times (-3) + 2 \times 1$""",
        'h0': r"""$\operatorname{relu}(z_0)$, and $z_0$ is 3""",
        'h1': r"""$\operatorname{relu}(z_1)$: a negative sum gives 0""",
        's0': r"""`h` dotted with column 0 of $W_2$: $h_0 \times 1 + h_1 \times 2$""",
        's1': r"""`h` dotted with column 1 of $W_2$: $h_0 \times 0 + h_1 \times 1$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _(forward_boxes, tiny_edges, tiny_heads):
    mo.stop(not hand.done(forward_boxes, {'z1': -1, 'h0': 3, 'h1': 0, 's0': 3, 's1': 0}),
            mo.md("*When all five boxes are right, the picture fills in here.*"))
    network([["x0 = 1", "x1 = 2"], ["z0 = 3", "z1 = -1"], ["h0 = 3", "h1 = 0"], ["s0 = 3", "s1 = 0"]],
            tiny_edges, "The same network with every value filled in", tiny_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: forward

    Write `forward(X, W1, b1, W2, b2)`. It does the three steps for all the pictures at once, and gives back all three results, `Z`, `H` and `S`, in that order.
    (We'll need `Z` and `H` again for the slopes.) Use your `relu`.

    On the tiny network above, with `X = [[1, 2]]`, a matrix with one row for the one picture, it must give `[[3, -1]]`, `[[3, 0]]` and `[[3, 0]]`.
    """)
    return


@app.function
def forward(X, W1, b1, W2, b2):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("08", "forward", forward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line for each step, in the order of the picture: weighted sums, bend, weighted sums.""",
        r"""`@` makes the weighted sums and `+` adds the `b`s. Give back three things, separated by commas.""",
    ], LOCALE)
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
    typical size 0.05 for `W1` and 0.18 for `W2`. (Chapter 09 explains both: why not zero, and
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

    Now the gradient. The loss depends on the weights through a chain of steps
    (`W1` → `Z` → `H` → `S` → loss), and chapter 03 tells us what to do with a chain: walk it
    backwards, passing blame. We take it one step at a time, starting from the end.

    We name each blame with a `d` in front: `dH` is the blame of `H`, `dW1` the blame (slope) of
    `W1`, and so on.

    **Scores.** The loss is chapter 07's cross-entropy of `softmax(S)`, so the blame of the scores
    is chapter 07's $G = (P - Y) / n$, with $P = \operatorname{softmax}(S)$.

    **`W2` and `b2`.** `S = H @ W2 + b2` is exactly chapter 07's `X @ W + b`, with `H` in place of
    `X`. (Quick recap of why chapter 07 gave `X.T @ G`: each weight's slope pairs its input column
    across all pictures with the score's blame column across all pictures, and transposing turns
    columns into rows so `@` computes every dot product at once.) So the same result holds here:
    $dW_2 = H^\top G$ (in code, `H.T @ G`), and $db_2$ is the sum down each column of $G$.

    **$H$.** This one is new: chapter 07 never needed the slope for its *inputs*. Take one hidden
    value, `H[i, j]` (picture $i$, detector $j$). It appears in all ten scores of picture $i$:
    written out, $S[i, c] = H[i, 0] \times W_2[0, c] + \ldots + H[i, 63] \times W_2[63, c] + b_2[c]$, so by chapter
    03's rule for a product (in $a \times c$, nudging $a$ moves the result by $c$), nudging `H[i, j]` moves `S[i, c]` by `W2[j, c]` per unit: that's the local
    slope. One value feeding ten arrows: **blames add** (chapter 03). So

    > blame of `H[i, j]` = $G[i, 0] \times W_2[j, 0] + G[i, 1] \times W_2[j, 1] + \ldots + G[i, 9] \times W_2[j, 9]$

    That's row `i` of `G` dotted with row `j` of `W2`, which is row `i` of `G` dotted with
    **column** `j` of `W2.T`. Since entry `[i, j]` of `A @ B` is row `i` of `A` dotted with
    column `j` of `B`, that's entry `[i, j]` of `G @ W2.T`. For every `i` and `j` at once:
    $dH = G W_2^\top$ (`G @ W2.T`), with shape `(n, 10) @ (10, 64) = (n, 64)`, the same shape as `H`.

    Here it is on a tiny example with 2 pictures, 2 detectors and 2 digits (using the tiny network's
    $W_2 = \begin{bmatrix} 1 & 0 \\ 2 & 1 \end{bmatrix}$, so $W_2^\top = \begin{bmatrix} 1 & 2 \\ 0 & 1 \end{bmatrix}$).
    Row 0 of `G` (blue) dotted with column 1 of `W2.T` (red) lands in `dH[0, 1]`: $1 \times 2 + (-1) \times 1 = 1$.
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "G (2, 2)", "values": [[1, -1], [0, 2]],
         "rows": ["picture 0", "picture 1"], "cols": ["digit 0", "digit 1"],
         "marks": {(0, 0): "blue", (0, 1): "blue"}},
        "@",
        {"name": "W2.T (2, 2)", "values": [[1, 2], [0, 1]],
         "rows": ["digit 0", "digit 1"], "cols": ["det 0", "det 1"],
         "marks": {(0, 1): "red", (1, 1): "red"}},
        "=",
        {"name": "dH (2, 2)", "values": [[1, 1], [0, 2]],
         "rows": ["picture 0", "picture 1"], "cols": ["det 0", "det 1"],
         "marks": {(0, 1): "box"}},
    ], "Row 0 of G (blue) dotted with column 1 of W2.T (red) gives dH[0, 1]: 1 times 2 plus -1 times 1 equals 1")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **The bend (ReLU).** `H = relu(Z)`, position by position. The slope of $\operatorname{relu}(z)$: where $z$ is
    positive, $\operatorname{relu}(z) = z$, slope 1; where $z$ is negative, $\operatorname{relu}(z) = 0$, slope 0. (Exactly at
    0 there's a corner; we take slope 0 there.) So the bend passes blame straight through where
    $Z > 0$ and blocks it everywhere else:

    > `dZ = dH * (Z > 0)`

    Notice that `*` here is **element-by-element** multiplication, not matrix multiplication `@`.
    `Z > 0` gives `True`/`False` at each position, and multiplying treats `True` as `1` and
    `False` as `0` automatically. An array like this, used to let some values through and block
    others, is called a **mask**. A detector that was silent for a picture takes no blame for it:
    nudging its weights a tiny bit wouldn't change anything. For two pictures and three detectors:
    """)
    return


@app.cell(hide_code=True)
def _():
    _z = [[2.0, -1.0, 0.5], [-0.3, 1.2, -2.0]]
    _m = [[1 if v > 0 else 0 for v in row] for row in _z]
    _dh = [[0.4, -0.6, 0.1], [0.3, -0.2, 0.5]]
    _blocked = {(r, c): "red" for r in range(2) for c in range(3) if not _m[r][c]}
    grids([
        {"name": "Z", "values": _z, "rows": ["picture 0", "picture 1"], "marks": _blocked},
        "→",
        {"name": "mask Z > 0", "values": _m, "marks": _blocked},
        "   ",
        {"name": "dH", "values": _dh},
        "×",
        {"name": "mask", "values": _m, "marks": _blocked},
        "=",
        {"name": "dZ", "values": [[d if m else 0 for d, m in zip(dr, mr)] for dr, mr in zip(_dh, _m)], "marks": _blocked},
    ], "Where Z is negative the mask is 0 (red), and multiplying dH by the mask blocks the blame there")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Where a sum in `Z` was negative (red), its detector was silent, the mask holds 0, and the
    blame in `dH` stops there. Everywhere else it passes through unchanged.

    **`W1` and `b1`.** `Z = X @ W1 + b1` is chapter 07's step once more, with `dZ` as the blame:
    $dW_1 = X^\top dZ$ (`X.T @ dZ`), and $db_1$ is the sum down each column of $dZ$.

    That's backpropagation through a whole network. Every step is chapter 03's rule; the
    matrices only do it for all pictures and all weights at once.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: blame through the tiny network

    Back to the tiny network, where `x = [1, 2]`, `Z = [3, -1]` and `H = [3, 0]`. Say the blames of the two scores came out as `G = [1, -1]` (round numbers, to keep the arithmetic easy).
    Pass the blame back, one step at a time. With a single picture, each formula is short:

    - $dW_2[j, c] = H[j] \times G[c]$
    - $dH[j]$ is $G$ dotted with row $j$ of $W_2$
    - $dZ$ is $dH$ where $Z$ is positive, and 0 elsewhere
    - $dW_1[i, j] = x[i] \times dZ[j]$
    """)
    return


@app.cell(hide_code=True)
def _():
    backward_boxes = hand.boxes(['dW2[0, 0]', 'dW2[1, 0]', 'dH[0]', 'dH[1]', 'dZ[0]', 'dZ[1]', 'dW1[1, 0]'])
    return (backward_boxes,)


@app.cell(hide_code=True)
def _(backward_boxes):
    hand.marked(backward_boxes, {'dW2[0, 0]': 3, 'dW2[1, 0]': 0, 'dH[0]': 1, 'dH[1]': 1, 'dZ[0]': 1, 'dZ[1]': 0, 'dW1[1, 0]': 2}, {
        'dW2[0, 0]': r"""$H[0] \times G[0]$""",
        'dW2[1, 0]': r"""$H[1] \times G[0]$, and detector 1 was silent""",
        'dH[0]': r"""row 0 of $W_2$ is `[1, 0]`: $1 \times 1 + (-1) \times 0$""",
        'dH[1]': r"""row 1 of $W_2$ is `[2, 1]`: $1 \times 2 + (-1) \times 1$""",
        'dZ[0]': r"""$Z[0]$ is positive, so the blame passes through unchanged""",
        'dZ[1]': r"""$Z[1]$ is negative, so the blame is blocked""",
        'dW1[1, 0]': r"""$x[1] \times dZ[0]$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _(backward_boxes, tiny_edges, tiny_heads):
    mo.stop(not hand.done(backward_boxes, {'dW2[0, 0]': 3, 'dW2[1, 0]': 0, 'dH[0]': 1, 'dH[1]': 1, 'dZ[0]': 1, 'dZ[1]': 0, 'dW1[1, 0]': 2}),
            mo.md("*When all seven boxes are right, the blames appear on the picture here.*"))
    network([["x0 = 1", "x1 = 2"], [["z0 = 3", "blame 1"], ["z1 = -1", "blame 0"]],
             [["h0 = 3", "blame 1"], ["h1 = 0", "blame 1"]], [["s0 = 3", "blame 1"], ["s1 = 0", "blame -1"]]],
            tiny_edges, "The tiny network with each value's blame under it, in red", tiny_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Look at detector 1. It was silent for this picture, so the knobs leading out of it get no blame (`dW2[1, 0]` is 0). And the mask stops its blame from travelling further back (`dZ[1]` is 0),
    so its own knobs in $W_1$ get none either. For this picture, only detector 0 learns.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: backward

    Write `backward(X, Y, Z, H, S, W2)`, giving back the four slopes in the order `dW1, db1, dW2, db2`. Walk backwards, as in the text:

    | step | formula | shape |
    |---|---|---|
    | blame of the scores | $G = (P - Y) / n$, with $P = \operatorname{softmax}(S)$ and $n$ the number of pictures, `len(X)` | `(n, 10)` |
    | second layer | $dW_2 = H^\top G$, and $db_2$ is the sum of $G$ down each column | `(64, n) @ (n, 10)` → `(64, 10)`; `(10,)` |
    | hidden values | $dH = G W_2^\top$ | `(n, 10) @ (10, 64)` → `(n, 64)` |
    | the bend (ReLU) | $dZ$ is $dH$ times the mask `Z > 0` (element-by-element `*`, not `@`) | `(n, 64) * (n, 64)` → `(n, 64)` |
    | first layer | $dW_1 = X^\top dZ$, and $db_1$ is the sum of $dZ$ down each column | `(784, n) @ (n, 64)` → `(784, 64)`; `(64,)` |

    In code, matrix multiplication $A B$ is `A @ B`, transpose $M^\top$ is `M.T` (so $H^\top G$ is `H.T @ G` and $G W_2^\top$ is `G @ W2.T`), and the sum down each column is `np.sum(M, axis=0)` (chapter 07). Notice how the middle dimensions always cancel in `@`: `(64, n) @ (n, 10)` gives `(64, 10)`, while `(n, 10) @ (10, 64)` gives `(n, 64)`—an easy self-check for which matrix gets `.T` and in what order.
    """)
    return


@app.function
def backward(X, Y, Z, H, S, W2):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("08", "backward", backward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One or two lines for each row of the table, in the table's order. Check each shape against the table as you go.""",
        r"""The order inside each `@` matters: `H.T @ G`, but `G @ W2.T`. If the nudge check below disagrees only for `W1`, look at the mask.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checking it by nudging

    A whole network's worth of new formulas deserves a check. At the start, nudge one weight of the
    first layer, `W1[406, 0]` (from pixel 406, row 14 and column 14: the middle of the picture,
    into detector 0),
    and see how the loss moves; then do the same for a weight of the second layer, `W2[5, 3]`.
    (This cell needs all three exercises. Since `forward` returns `(Z, H, S)` and Python counts
    from 0, `forward(...)[2]` below picks out the third return value, the scores `S`.)
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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: how many knobs

    Before training, count what will be trained. `W1` has shape `(784, 64)`, `b1` has 64 numbers, `W2` has shape `(64, 10)`, and `b2` has 10.
    How many knobs are in `W1`? And how many in the whole network?
    """)
    return


@app.cell(hide_code=True)
def _():
    count_boxes = hand.boxes(['W1', 'whole network'])
    return (count_boxes,)


@app.cell(hide_code=True)
def _(count_boxes):
    hand.marked(count_boxes, {'W1': 50176, 'whole network': 50890}, {
        'W1': r"""$784 \times 64$""",
        'whole network': r"""`W1`, plus 64, plus $64 \times 10$, plus 10""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Chapter 07's model had 7,850 knobs. This one has over six times as many, and almost all of them are in the first layer.

    ## Training

    Same loop as chapter 07, with four sets of knobs instead of two. 300 steps over all 10,000
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
    They look much noisier than chapter 07's templates, and the noise is telling. Near the edges,
    almost every picture is white paper: many of those pixels are 0 in every picture, and the
    rest almost always are. Their knobs take little or no blame (the slope `X.T @ dZ` multiplies
    by those zeros) and keep (nearly) their random starting values. Towards the middle, where the ink is,
    you can make out smoother patches: pieces of strokes that each detector learned to respond
    to, without anyone telling it what to look for. The second layer learns which mix of
    detectors makes each digit.

    ## What's next

    It works, but training was slow: every one of the 300 steps looked at all 10,000 pictures.
    And we chose the starting sizes 0.05 and 0.18 without saying why. In **Chapter 09** you'll
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
