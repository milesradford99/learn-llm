import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="07 · A loss for choices")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
    from learn_llm.diagrams import bars, grids, network, pipeline
    from learn_llm.plot import chart

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 · A loss for choices

    In chapter 06 we scored each picture with `S = X @ W + b` (784 pixels per row, 10 columns of knobs in `W`, and 10 biases in `b`), and built `W` by simply averaging each digit's pictures. That gave 62% accuracy on the test digits, and guessed 8 for a quarter of all pictures — because those knobs were never adjusted to tell digits apart.

    In this chapter, gradient descent will tune `W` and `b` directly, just as it tuned Mia's line. What it needs is a loss, and building one takes two new math tools: $e^x$ and $\log x$. You'll work out their slopes step by step, just as you did in chapter 02.

    Here are your two functions from chapter 06, finished. After them come the pictures, flattened into rows as before, with all knobs starting at 0, and the accuracy of that starting model.
    """)
    return


@app.function
def guess(S):
    return np.argmax(S, axis=1)


@app.function
def accuracy(S, labels):
    return np.mean(guess(S) == labels)


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    W_zero = np.zeros((784, 10))
    b_zero = np.zeros(10)
    S_zero = X @ W_zero + b_zero
    accuracy(S_zero, train_labels)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    0.0969: with every score at 0, the model guesses 0 for every picture, and is only right on the 969 zeros.

    ## Why accuracy can't be the loss

    We want gradient descent to improve the knobs, so we need a loss with useful slopes.
    Accuracy doesn't give us any: nudge one knob a tiny bit and the scores change a tiny bit, but the
    biggest score almost always stays the biggest. No guess changes, so accuracy doesn't budge:
    its slope is 0 almost everywhere, and gradient descent never knows which way to move.

    See it on one picture — a 1 — with only two digits (0 and 1). Keep digit 0's score fixed at 1,
    and slowly slide digit 1's score from -1 to 3. On the left is accuracy on this picture; on the
    right is a smooth measure of how sure the model is that it's a 1 (which we'll build next):
    """)
    return


@app.cell(hide_code=True)
def _():
    _xs = [x / 50 for x in range(-50, 151)]
    mo.hstack([
        chart((-1, 3), (0, 1), "score for digit 1", "accuracy",
              curve=[(x, 1.0 if x > 1 else 0.0) for x in _xs]),
        chart((-1, 3), (0, 1), "score for digit 1", "probability for digit 1",
              curve=[(x, float(np.exp(x) / (np.exp(x) + np.exp(1)))) for x in _xs]),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Accuracy behaves like a **light switch**: flat at 0, a sudden jump the instant digit 1's score
    passes digit 0's, then flat at 1. On a flat stretch, a small nudge changes nothing, so the slope
    is 0 and gradient descent has no hint which way to go. The curve on the right behaves like a
    **dimmer**: every tiny nudge moves it a little, so every point has a slope saying "this way is better".

    So we need a smooth measure: not just *which* digit wins, but **how confident** the model is
    in each digit. If we turn raw scores into probabilities (shares that add up to 1) — like
    "0.7 for 3, 0.2 for 8, 0.1 for the rest" — then nudging a knob changes those probabilities
    smoothly, giving gradient descent clear slopes to follow.

    ## From scores to probabilities: $e^s$

    Raw scores can be negative, while probabilities must be positive. So first we turn each score
    into a positive number while keeping their order (a higher score stays higher). The tool is
    raising a special number called **$e$** (about 2.718) to the power of the score.

    You know powers from chapter 00: `2 ** 3` is $2 \times 2 \times 2$. Powers obey a handy rule:
    $2^{3 + 2} = 2^3 \times 2^2$ (five 2s multiplied either way). Math extends powers to 0, negative
    numbers, and decimals in the one way that keeps this rule true:

    - $e^3 \times e^0 = e^{3 + 0} = e^3 \implies e^0 = 1$
    - $e^1 \times e^{-1} = e^0 = 1 \implies e^{-1} = 1 / e \approx 0.37$
    - $e^{0.5} \times e^{0.5} = e^1 \implies e^{0.5} = \sqrt{e} \approx 1.65$
    - And for subtraction: $e^{a - b} \times e^b = e^a \implies e^{a - b} = e^a / e^b$

    We write this as $e^s$, and in NumPy it's `np.exp(s)`. Here is the curve for scores from -3 to 3:
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-3, 3), (0, 20), "score s", "eˢ",
          curve=[(x / 20, float(np.exp(x / 20))) for x in range(-60, 61)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A few concrete values:

    | score $s$ | −2 | −1 | 0 | 1 | 2 | 3 |
    |---|---|---|---|---|---|---|
    | $e^s$ | 0.14 | 0.37 | 1 | 2.72 | 7.39 | 20.09 |

    Each step to the right multiplies by 2.718; each step to the left divides by 2.718. No matter
    how negative the score gets, $e^s$ only gets closer to 0 — never reaching 0 or going negative.

    Three things to notice: $e^s$ is always positive; a higher score always gives a larger $e^s$;
    and every +1 in the score multiplies $e^s$ by about 2.718. (Why $e$ instead of 2? Because $e$
    gives the simplest possible slope — you'll see why in a moment.)

    Next, **divide each $e^s$ by their total** so they add up to 1. For scores `[2, 1, 0.1]`:
    $e^2 = 7.39$, $e^1 = 2.72$, $e^{0.1} = 1.11$, total 11.21, so the probabilities are
    $7.39 / 11.21 = 0.659$, $0.242$, and $0.099$. This two-step recipe — compute $e^s$, then divide
    by the sum — is called **softmax**. Its outputs are the model's **probabilities** (how sure it
    is of each digit).

    Try dragging the three scores and watch $e^s$ and the probabilities respond:
    """)
    return


@app.cell(hide_code=True)
def _():
    play_scores = mo.ui.array([mo.ui.slider(-3, 3, step=0.5, value=v, label=f"score for digit {i}", show_value=True)
                               for i, v in enumerate([2, 1, 0])])
    play_scores.vstack()
    return (play_scores,)


@app.cell(hide_code=True)
def _(play_scores):
    _s = np.array(play_scores.value, dtype=float)
    _e = np.exp(_s)
    mo.hstack([
        mo.vstack([mo.md(r"$e^s$"), bars([(f"digit {i}", v) for i, v in enumerate(_e)], "e to the power of each score", digits=2)]),
        mo.vstack([mo.md(r"probabilities: each $e^s$ divided by the total, " + f"{_e.sum():.2f}"),
                   bars([(f"digit {i}", v) for i, v in enumerate(_e / _e.sum())], "the probability of each digit", top=1.0)]),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Things to try:

    - Make all three scores equal. Every probability is $1/3$: the model can't decide.
    - Push one score up by 1. Its $e^s$ multiplies by 2.72, pulling probability away from the others.
    - Set the scores to `2, 1, 0`, note the probabilities, then change them to `3, 2, 1`. The
      probabilities don't change at all: adding the same constant to every score multiplies every
      $e^s$ by the same factor, which cancels out when dividing by the total. Only the **differences**
      between scores matter.

    This property solves a practical computer problem: `np.exp(1000)` is far too big for a computer
    to store, so it returns `inf` (infinity), and `inf / inf` produces `nan` ("not a number").
    The fix is simple: before computing $e^s$, subtract the largest score $m$ from every score.
    By the rule $e^{s - m} = e^s / e^m$, every term and the total are divided by the same $e^m$,
    so the probabilities stay identical. After subtracting $m$, the largest score is 0 (so $e^0 = 1$)
    and every other score is negative, so nothing can overflow. (Without this step, NumPy prints an
    "overflow" warning.)

    For example, `[1000, 999, 998]` minus its largest value `1000` becomes `[0, -1, -2]` — the exact
    same gaps as `[2, 1, 0]`, giving the same probabilities `0.67, 0.24, 0.09`, using tiny safe
    numbers ($e^0 = 1$, $e^{-1} = 0.37$, $e^{-2} = 0.14$).

    ### Rows, columns, and `keepdims`

    For a matrix of scores `S`, we apply softmax row by row so each picture gets its own probabilities.
    `np.max(S, axis=1)` gives the largest score in each row, and `np.sum(E, axis=1)` gives each row's
    sum — one number per picture. Passing `keepdims=True` keeps the result as a tall **column** of
    shape `(10000, 1)` instead of a flat 1D vector `(10000,)`.

    Why does the column shape matter? Recall broadcasting from chapter 05: a row `(3,)` stretches
    vertically across every row of `(2, 3)`, whereas a column `(2, 1)` stretches horizontally across
    every column of `(2, 3)`. So with `m = np.max(S, axis=1, keepdims=True)`, `S - m` subtracts each
    picture's own maximum from all of its scores:
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "S  (2, 3)", "values": [[2, 1, 0], [5, 7, 6]], "rows": ["picture 0", "picture 1"],
         "marks": {(0, 0): "box", (1, 1): "box"}},
        "−",
        {"name": "m  (2, 1)", "values": [[2], [7]], "marks": {(0, 0): "blue", (1, 0): "red"}},
        "=",
        {"name": "S − m  (2, 3)", "values": [[0, -1, -2], [-2, 0, -1]],
         "marks": {**{(0, c): "blue" for c in range(3)}, **{(1, c): "red" for c in range(3)}}},
    ], "Each row's largest score, kept as a column m, is subtracted from every score in its own row")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `m` holds each row's largest score (outlined in `S`) as a column. Picture 0's `2` is subtracted
    across row 0 (blue), and picture 1's `7` across row 1 (red). Without `keepdims=True`, `m` would
    be `[2, 7]` of shape `(2,)`; NumPy would treat it as a **row** and try to match its 2 entries
    against each 3-entry row of `S`, crashing with a shape error (or worse, silently computing the
    wrong answer if the number of pictures happened to equal the number of digits!).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: probabilities

    Three scores: `[2, 1, 0]`. To two decimals, $e^2 = 7.39$, $e^1 = 2.72$, and $e^0 = 1$. Work out their total, then the probability of the first score and the last score (to two decimals).
    """)
    return


@app.cell(hide_code=True)
def _():
    share_boxes = hand.boxes(['total', 'first share', 'last share'], step=0.01)
    return (share_boxes,)


@app.cell(hide_code=True)
def _(share_boxes):
    hand.marked(share_boxes, {'total': 11.11, 'first share': 0.67, 'last share': 0.09}, {
        'total': r"""add the three numbers""",
        'first share': r"""$7.39 \div \text{total}$""",
        'last share': r"""$1 \div \text{total}$""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: softmax

    Write `softmax(S)` for a matrix of scores, one row per picture. Each row gets its own probabilities:

    1. Subtract from every score the largest score in its row.
    2. Apply $e^s$ to the result using `np.exp`.
    3. Divide each entry by the sum of its row.

    Steps 1 and 3 both need one number per row that broadcasts across that row — that's what `keepdims=True` is for.

    The cell after the check tests your function on one picture's scores, `[[2, 1, 0.1]]`, which should give about `[[0.659, 0.242, 0.099]]`. Notice the double brackets: a 1-row matrix (one picture), because softmax works row by row.
    """)
    return


@app.function
def softmax(S):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "softmax", softmax, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`np.max(S, axis=1, keepdims=True)` is each row's largest score, as a column. The total of each row is an `np.sum` with the same two settings.""",
        r"""If the check with two pictures fails while the one with a single picture passes, a `keepdims=True` is missing. If `[[1000, 1000]]` gives `nan`, step 1 is missing.""",
    ], LOCALE)
    return


@app.cell
def _():
    softmax(np.array([[2, 1, 0.1]]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The loss: surprise

    How should we score probabilities like `[0.659, 0.242, 0.099]` when the right answer is the first
    one? We want a loss that is 0 when the model gives the right digit probability 1.0, and grows
    without limit as that probability shrinks toward 0: a model that's very confident in a wrong
    answer should pay a heavy penalty.

    **$\log$** does exactly this. $\log$ undoes $e^a$: $\log(e^a) = a$, so $\log x$ answers "$e$ to
    what power gives $x$?" (in NumPy, `np.log(x)`). $\log 1 = 0$ (since $e^0 = 1$), and for any
    probability below 1 the log is negative (only a negative power of $e$ is below 1), plunging
    downward as $p$ approaches 0. Flipping the sign gives a positive loss called **surprise**:

    | probability $p$ of the right digit | 1 | 0.9 | 0.5 | 0.1 | 0.01 |
    |---|---|---|---|---|---|
    | surprise, $-\log p$ | 0 | 0.105 | 0.693 | 2.303 | 4.605 |

    Notice the last two columns: dropping $p$ by a factor of 10 (from `0.1` to `0.01`) adds a fixed
    `2.303` to the surprise (from `2.303` to `4.605`). Dropping another factor of 10 to `0.001`
    adds another `2.303`, giving `6.908`. Every 10x drop in confidence costs the same extra penalty.
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((0, 1), (0, 5), "probability p of the right digit", "surprise −log p",
          curve=[(x / 200, float(-np.log(x / 200))) for x in range(2, 201)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The loss is the mean surprise over all pictures. It's called the **cross-entropy**.

    To pick out the right digit's share in a matrix of shares, we use a matrix `Y` of the same
    shape with a 1 in the right digit's column and 0 everywhere else (a **one-hot** row). Then
    `Y * P` keeps only the right digit's share in each row, and `np.sum(Y * P, axis=1)` gives
    one share per picture. With three digits, a picture of a 0 and a picture of a 2:
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "P", "values": [[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]], "rows": ["picture 0", "picture 1"],
         "cols": ["0", "1", "2"]},
        "×",
        {"name": "Y", "values": [[1, 0, 0], [0, 0, 1]], "cols": ["0", "1", "2"],
         "marks": {(0, 0): "blue", (1, 2): "blue"}},
        "=",
        {"name": "Y × P", "values": [[0.7, 0, 0], [0, 0, 0.4]], "cols": ["0", "1", "2"],
         "marks": {(0, 0): "blue", (1, 2): "blue"}},
        "→",
        {"name": "row sums", "values": [[0.7], [0.4]], "marks": {(0, 0): "box", (1, 0): "box"}},
    ], "Multiplying the shares P by the one-hot rows Y position by position keeps only the right digit's share in each row; adding along each row gives 0.7 and 0.4")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The 0s in `Y` wipe out every wrong digit's share, so adding along a row just picks out the one
    that's left. Here is a helper that builds `Y` from the labels: a matrix of zeros,
    then a 1 at row `i`, column `labels[i]`, for each picture.
    """)
    return


@app.function
def one_hot(labels):
    Y = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        Y[i, labels[i]] = 1
    return Y


@app.cell
def _(test_labels, train_labels):
    Y = one_hot(train_labels)
    train_labels[0], Y[0]
    return (Y,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The first picture is a 6, so its one-hot row has its single 1 at position 6 (counting from
    0: the seventh number). The dots, as in `1.`, mean the numbers are stored as decimals.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: the loss of two pictures

    Two pictures. The model gives the right digit of the first a share of 0.5, and the right digit of the second a share of 0.1. Look up the two surprises in the table above. Then work out the loss: the mean of the two.
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_boxes = hand.boxes(['first surprise', 'second surprise', 'loss'], step=0.001)
    return (loss_boxes,)


@app.cell(hide_code=True)
def _(loss_boxes):
    hand.marked(loss_boxes, {'first surprise': 0.693, 'second surprise': 2.303, 'loss': 1.498}, {
        'first surprise': r"""the table's entry for a share of 0.5""",
        'second surprise': r"""the table's entry for a share of 0.1""",
        'loss': r"""add the two surprises and divide by 2""",
    }, LOCALE, tolerance=0.002)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: cross_entropy

    Write `cross_entropy(P, Y)`: the mean surprise. `P` holds the shares, one row per picture, and `Y` holds the one-hot rows.

    1. `np.log` works position by position, so `np.log(P)` is a matrix of the same shape as `P`.
    2. Multiplying by `Y` keeps only the right digit's entry in each row. The sum along each row then gives one number per picture: $\log p$ for its right digit.
    3. The loss is minus the mean of those numbers.

    For your two pictures above, written as `P = [[0.5, 0.5], [0.1, 0.9]]` and `Y = [[1, 0], [1, 0]]`, it must give 1.498.
    """)
    return


@app.function
def cross_entropy(P, Y):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "cross_entropy", cross_entropy, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`np.sum(M, axis=1)` adds along each row, giving one number per picture.""",
        r"""One line: a minus sign, then `np.mean` of the row sums of `Y * np.log(P)`. The minus matters: logs of shares are negative, and a loss should be positive.""",
    ], LOCALE)
    return


@app.cell
def _(S_zero, Y):
    cross_entropy(softmax(S_zero), Y)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With all scores 0, every digit gets the same probability, $1/10$, so every picture's surprise is
    $-\log 0.1 = 2.303$. That's the loss of a model that knows nothing. Training should bring it down.

    ## The slopes of $e^x$ and $\log x$

    To run gradient descent we need the loss's slopes, and the loss is built from $e^x$ and $\log x$.
    Just as in chapter 02, we can work out their slopes by nudging the input by a tiny amount $h$.

    **$e^x$.** Nudge $x$ by a tiny $h$. By the power rule, $e^{x + h} = e^x \times e^h$, so the
    change is $e^x \times e^h - e^x$. Factoring out $e^x$ gives $e^x \times (e^h - 1)$, so the
    ratio (change $\div h$) is:

    > $$e^x \times \frac{e^h - 1}{h}$$

    Notice that $(e^h - 1) / h$ doesn't depend on $x$ at all. The same algebra works for any base
    (for base 2 it's $(2^h - 1) / h$). Look at what this number closes in on as $h$ shrinks:

    | base | `h = 0.01` | `h = 0.0001` |
    |---|---|---|
    | 2 | 0.6956 | 0.6932 |
    | 3 | 1.1047 | 1.0987 |
    | $e$ | 1.0050 | 1.0001 |

    As $h$ shrinks, the ratio approaches 0.693 for base 2, 1.099 for base 3, and **1** for base $e$.
    In fact, $e$ is *defined* as the base where this factor equals 1. So the ratio closes in on
    $e^x \times 1$:

    > The slope of $e^x$ is $e^x$ itself.

    In a picture: wherever you stand on the curve, its height and its steepness are the exact same
    number. At $x = 1$ the curve is 2.72 high and climbs 2.72 per unit; at $x = 2$ it is 7.39 high
    and climbs 7.39 per unit.
    """)
    return


@app.cell(hide_code=True)
def _():
    _e = float(np.exp(1))
    chart((-1, 2.6), (0, 14), "x", "eˣ",
          curve=[(x / 50, float(np.exp(x / 50))) for x in range(-50, 131)],
          lines=[(_e, 0.0), (_e ** 2, -(_e ** 2))],
          dots=[(1, _e), (2, _e ** 2)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **$\log x$.** Let's try a concrete number first. Since $e^2 \approx 7.39$, we have $\log 7.39 \approx 2$.
    Now nudge $y = 2$ up by a tiny $k = 0.01$ to $2.01$. We just saw that $e^y$ has slope $e^y$, so at
    $y = 2$ its slope is $7.39$. That means $x = e^y$ grows by about $7.39 \times 0.01 = 0.0739$.

    Now flip your viewpoint: when $x$ grows by $0.0739$, $\log x$ grows by $0.01$. So the slope
    (change in $\log x \div$ change in $x$) is $0.01 / 0.0739 = 1 / 7.39$.

    The same works for any $x$ (with $y = \log x$, so $x = e^y$): nudging $y$ by a tiny $k$ makes
    $x$ grow by about $e^y \times k = x \times k$, so:

    > The slope of $\log x$ is $\frac{k}{x \times k} = \frac{1}{x}$.

    ## The slope of the loss for each score

    To keep things small, imagine only three digits (0, 1, 2) and one picture whose true digit is 0.
    Call its three scores $s_0, s_1, s_2$, and write $T = e^{s_0} + e^{s_1} + e^{s_2}$ for the sum.
    Then the right digit's probability is $p_0 = e^{s_0} / T$, and the surprise is $L = -\log(e^{s_0} / T)$.

    First, one quick rule for $\log$: **$\log(x / y) = \log x - \log y$**. Why? Write $x = e^a$ and
    $y = e^b$ (so $a = \log x$ and $b = \log y$). By the power rule, $x / y = e^a / e^b = e^{a - b}$
    (for example, $e^5 / e^2 = e^3$, so $\log(e^5 / e^2) = 3 = 5 - 2$). Applying this rule splits
    our surprise into two simple terms:

    > $$L = -(\log(e^{s_0}) - \log T) = -\log(e^{s_0}) + \log T = -s_0 + \log T$$

    (The minus sign flips both terms inside the brackets, and $\log(e^{s_0}) = s_0$ because $\log$
    undoes $e^s$.)

    ### Quick reminder: the two rules of blame (chapter 03)

    Draw the calculation as boxes and arrows, where each arrow carries its **local slope** (how much
    the box at the tip moves when the box at the tail is nudged by 1). The loss starts with blame 1.
    Walking backwards:

    - **Along a road, multiply.** If $u$ feeds $v$ with local slope 3 and $v$'s blame is 2, then $u$
      gets $3 \times 2 = 6$ along that arrow.
    - **Where roads split, add.** If $u$ feeds two boxes and gets 3 along one arrow and 4 along the
      other, its total blame is $3 + 4 = 7$.

    Now draw $L = -s_0 + \log T$ with those exact local slopes on every arrow:
    """)
    return


@app.cell(hide_code=True)
def _():
    network(
        [[["s0"], ["s1"], ["s2"]], [["-s0"], ["T", "total"]], [["L", "-s0 + log T"]]],
        [(0, 0, 0, "-1"), (0, 0, 1, "exp(s0)"), (0, 1, 1, "exp(s1)"), (0, 2, 1, "exp(s2)"),
         (1, 0, 0, "1"), (1, 1, 0, "1/T")],
        "Three scores feed -s0 and T, which add to L = -s0 + log T. s0 has two paths to L (slopes -1 and exp(s0)/T); s1 and s2 each have one path through T.",
        ["scores", "two parts", "surprise"],
        at=0.45,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Just follow the arrows backwards from $L$ (which starts with blame 1):

    - **For $s_0$** (the right digit), two roads leave $s_0$, so we multiply along each road and add:
      - Top road ($s_0 \to -s_0 \to L$): $(-1) \times 1 = -1$.
      - Bottom road ($s_0 \to T \to L$): $e^{s_0} \times \frac{1}{T} = \frac{e^{s_0}}{T} = p_0$.
      - Total: **$p_0 - 1$**.
    - **For $s_1$** (a wrong digit), only one road leaves $s_1$: $e^{s_1} \times \frac{1}{T} = p_1$.
      The same holds for $s_2$: **$p_2$**.

    In one sentence: **probability minus 1 for the right digit; probability itself for every other
    digit.** With a one-hot row `y`, that's simply `p - y`. Gradient descent steps *against* the
    slope, so it pushes the right digit's score **up** (its slope is negative) and every wrong
    digit's score **down**, in proportion to how much probability it wrongly took.

    Let's verify this by nudging, the chapter 02 way, on scores `[2, 1, 0.1]`:
    """)
    return


@app.cell
def _():
    def surprise(s):
        return -np.log(np.exp(s[0]) / np.sum(np.exp(s)))

    s3 = np.array([2.0, 1.0, 0.1])
    h = 0.0001
    nudged = [
        (surprise(s3 + np.array([h, 0, 0])) - surprise(s3)) / h,
        (surprise(s3 + np.array([0, h, 0])) - surprise(s3)) / h,
        (surprise(s3 + np.array([0, 0, h])) - surprise(s3)) / h,
    ]
    nudged
    return (h,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About `-0.341`, `0.242`, and `0.099`: the probabilities `[0.659, 0.242, 0.099]` with 1 subtracted
    from the right digit's. The formula matches to every decimal place.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: the slopes for one picture

    A picture of a 1. The model's probabilities for digits 0, 1, and 2 are `[0.7, 0.2, 0.1]`: fairly
    confident, and wrong. What are the slopes of its surprise for the three scores?
    """)
    return


@app.cell(hide_code=True)
def _():
    slope_boxes = hand.boxes(['score 0', 'score 1', 'score 2'], step=0.1)
    return (slope_boxes,)


@app.cell(hide_code=True)
def _(slope_boxes):
    hand.marked(slope_boxes, {'score 0': 0.7, 'score 1': -0.8, 'score 2': 0.1}, {
        'score 0': r"""a wrong digit: the slope is its probability""",
        'score 1': r"""the right digit: its probability minus 1""",
        'score 2': r"""a wrong digit: the slope is its probability""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The largest slope (in magnitude) belongs to the right digit: gradient descent pulls its score up
    hard. Meanwhile, the wrong digit that got 0.7 is pushed down much harder than the one that got 0.1.

    Think of it as a tug of war. The right digit is pulled up by $1 - p$ (how much probability it's
    still missing). Every wrong digit is pushed down by $p$ (how much probability it wrongly took).
    Once the right digit gets all the probability, every pull drops to 0. Watch it happen step by
    step (learning rate 1):
    """)
    return


@app.cell(hide_code=True)
def _():
    pull_steps = mo.ui.slider(0, 6, value=0, label="steps taken", show_value=True)
    pull_steps
    return (pull_steps,)


@app.cell(hide_code=True)
def _(pull_steps):
    _s = np.log(np.array([0.7, 0.2, 0.1]))
    _y = np.array([0.0, 1.0, 0.0])
    for _ in range(pull_steps.value):
        _p = np.exp(_s) / np.exp(_s).sum()
        _s = _s - (_p - _y)
    _p = np.exp(_s) / np.exp(_s).sum()
    mo.hstack([
        mo.vstack([mo.md("probabilities"), bars([(f"digit {i}", v) for i, v in enumerate(_p)], "probabilities", top=1.0)]),
        mo.vstack([mo.md("slopes: probability minus one-hot"),
                   bars([(f"digit {i}", v) for i, v in enumerate(_p - _y)], "slopes", top=1.0)]),
        mo.md(rf"surprise: **{-np.log(_p[1]):.3f}**"),
    ], widths=[2, 2, 1])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After one step, digit 1's probability jumps from 0.2 to about 0.5; after six steps it passes 0.9,
    and the surprise drops from 1.609 toward 0. As the right probability grows, every slope shrinks:
    the closer the model gets to right, the gentler the pull.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From the scores to the knobs

    Gradient descent doesn't edit scores directly — it adjusts the knobs (`W`, with $784 \times 10$
    numbers, and `b`, with 10 numbers). Here is the full pipeline for all $n$ pictures at once, with
    each matrix's shape:
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("X", "(n, 784)"), ("S", "(n, 10)"), ("P", "(n, 10)"), ("L", "one number")],
             ["@ W + b", "softmax", "cross-entropy"],
             "X, times W plus b, gives the scores S; softmax gives the probabilities P; cross-entropy gives the loss L")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The last section carried blame from **one** picture's surprise back to its scores. Two steps
    remain: the loss $L$ is the **mean** of all pictures' surprises (not just one), and we need
    slopes for the **knobs** `W` and `b` (not just the scores `S`). Using the exact same two rules —
    *multiply along a road, add where roads split* — let's trace both steps on a tiny world where we
    can see every single number.

    ### A world small enough to see

    Two pictures of only 3 pixels each, and only two digits (0 and 1). Ink is 1, blank is 0.
    Picture 0 is a 0, with ink in pixels 0 and 2; picture 1 is a 1, with ink in pixels 1 and 2.
    Suppose the model gives picture 0 probabilities `[0.6, 0.4]` (right, not very sure) and
    picture 1 probabilities `[0.8, 0.2]` (confident, and wrong).

    Here is the end of the calculation as a graph: four scores, two surprises, one loss. Score
    `S[i, c]` is picture $i$'s score for digit $c$. The loss is the mean of the two surprises,
    $L = (\text{surprise}_0 + \text{surprise}_1) / 2$. Local slopes sit on the arrows; blames are in red.
    """)
    return


@app.cell(hide_code=True)
def _():
    blame_edges = [(0, 0, 0, "-0.4"), (0, 1, 0, "0.4"), (0, 2, 1, "0.8"), (0, 3, 1, "-0.8"),
                   (1, 0, 0, "1/2"), (1, 1, 0, "1/2")]
    blame_heads = ["scores", "surprises", "loss"]
    network([[["S[0,0]", "blame -0.2"], ["S[0,1]", "blame 0.2"], ["S[1,0]", "blame ?"], ["S[1,1]", "blame ?"]],
             [["surprise 0", "blame 1/2"], ["surprise 1", "blame 1/2"]], [["L", "blame 1"]]],
            blame_edges, "Four scores feed two surprises, which feed the loss L. Each arrow into L has local slope one half; "
            "the arrows from picture 0's scores have local slopes -0.4 and 0.4, from picture 1's 0.8 and -0.8",
            blame_heads)
    return blame_edges, blame_heads


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Step 1: from the loss to each surprise

    $L = (\text{surprise}_0 + \text{surprise}_1) / 2$. Nudge surprise 0 by a tiny $h$ and $L$ moves by
    $h / 2$: the local slope of each arrow into $L$ is $\frac{1}{2}$. $L$'s blame is 1, so along a road,
    each surprise gets $\frac{1}{2} \times 1 = \frac{1}{2}$. With $n$ pictures, $L$ is a sum divided by $n$,
    and each surprise gets $\frac{1}{n}$.

    ### Step 2: from each surprise to its scores

    The last section found the local slopes from one picture's surprise to its own scores:
    share minus one-hot, $p - y$. Picture 0 is a 0, with shares `[0.6, 0.4]`, so its one-hot
    row is `[1, 0]` and the slopes are $0.6 - 1 = -0.4$ towards `S[0,0]` and $0.4 - 0 = 0.4$
    towards `S[0,1]`: the numbers on its two arrows. Along a road, multiply by the blame at the
    arrow's end, $\frac{1}{2}$:

    > $$\text{blame of } S[0, 0] = -0.4 \times \tfrac{1}{2} = -0.2 \qquad \text{blame of } S[0, 1] = 0.4 \times \tfrac{1}{2} = 0.2$$

    Each score feeds only its own picture's surprise, so there's nothing to add. Your turn:
    picture 1's two scores.
    """)
    return


@app.cell(hide_code=True)
def _():
    score_blame_boxes = hand.boxes(["S[1, 0]", "S[1, 1]"], step=0.1)
    return (score_blame_boxes,)


@app.cell(hide_code=True)
def _(score_blame_boxes):
    hand.marked(score_blame_boxes, {"S[1, 0]": 0.4, "S[1, 1]": -0.4}, {
        "S[1, 0]": r"""picture 1 is a 1, so its one-hot row is `[0, 1]`: the local slope is $0.8 - 0$, then multiply by its surprise's blame""",
        "S[1, 1]": r"""the local slope is $0.2 - 1$, then multiply by its surprise's blame""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _(blame_edges, blame_heads, score_blame_boxes):
    mo.stop(not hand.done(score_blame_boxes, {"S[1, 0]": 0.4, "S[1, 1]": -0.4}),
            mo.md("*When both boxes are right, the graph fills in here.*"))
    network([[["S[0,0]", "blame -0.2"], ["S[0,1]", "blame 0.2"], ["S[1,0]", "blame 0.4"], ["S[1,1]", "blame -0.4"]],
             [["surprise 0", "blame 1/2"], ["surprise 1", "blame 1/2"]], [["L", "blame 1"]]],
            blame_edges, "The same graph with every score's blame filled in", blame_heads)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### The blame of every score, as one table

    Every score got the same treatment: (share minus one-hot), times $\frac{1}{n}$. So put the four
    blames in a table shaped like the scores, one row per picture and one column per digit. With
    `P` the table of shares and `Y` the one-hot rows, the whole table is

    > $$G = (P - Y) / n$$

    `G[i, c]` is the blame of score `S[i, c]`. In our world:
    """)
    return


@app.cell(hide_code=True)
def _():
    _cols = ["digit 0", "digit 1"]
    grids([
        {"name": "P", "values": [[0.6, 0.4], [0.8, 0.2]], "rows": ["picture 0", "picture 1"], "cols": _cols},
        "−",
        {"name": "Y", "values": [[1, 0], [0, 1]], "cols": _cols},
        "=",
        {"name": "P − Y", "values": [[-0.4, 0.4], [0.8, -0.8]], "cols": _cols},
        "÷ 2 =",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]], "cols": _cols,
         "marks": {(r, c): "red" for r in range(2) for c in range(2)}},
    ], "The shares P minus the one-hot rows Y, divided by the number of pictures, 2, give the blames G of the four scores")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That's the blame of every score, and it's where the hard part ends: from here to the knobs
    is one more step of the same two rules.

    ### Step 3: from the scores to one knob

    Each digit has one knob per pixel: `W[j, c]` is the knob for pixel $j$ and digit $c$, so
    here `W` has shape `(3, 2)`. Here are the pictures' pixels again, `X`, one row per picture:
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]],
         "rows": ["picture 0", "picture 1"], "cols": ["pixel 0", "pixel 1", "pixel 2"]},
    ], "The pixels X: picture 0 has ink in pixels 0 and 2, picture 1 in pixels 1 and 2")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Take the knob `W[2, 0]`. The score of picture $i$ for digit 0 is

    > $$S[i, 0] = X[i, 0] \times W[0, 0] + X[i, 1] \times W[1, 0] + X[i, 2] \times W[2, 0] + b[0]$$

    So `W[2, 0]` is in the digit-0 score of **both** pictures: one box with two arrows leaving
    it. The local slope of each arrow is the pixel the knob multiplies there, `X[i, 2]` (in
    $x \times w$, nudging $w$ moves the result by $x$ per unit). Both pictures have ink in pixel 2,
    so both slopes are 1. As a graph, with the score blames you just found:
    """)
    return


@app.cell(hide_code=True)
def _():
    network([[["W[2,0]", "blame ?"]], [["S[0,0]", "blame -0.2"], ["S[1,0]", "blame 0.4"]]],
            [(0, 0, 0, "X[0,2] = 1"), (0, 0, 1, "X[1,2] = 1")],
            "The knob W[2,0] feeds the digit-0 score of both pictures, each arrow with local slope 1",
            ["knob", "scores"], at=0.55)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Roads split, so add; along each road, multiply:

    > $$\text{slope of } W[2, 0] = 1 \times (-0.2) + 1 \times 0.4 = 0.2$$

    And a pixel without ink? Take `W[1, 0]`. Picture 0 has no ink in pixel 1, so its arrow has
    local slope 0 and carries no blame at all; only picture 1 counts: $0 \times (-0.2) + 1 \times 0.4 = 0.4$.
    A knob only learns from the pictures that have ink where it looks.
    """)
    return


@app.cell(hide_code=True)
def _():
    network([[["W[1,0]", "blame 0.4"]], [["S[0,0]", "blame -0.2"], ["S[1,0]", "blame 0.4"]]],
            [(0, 0, 0, "X[0,1] = 0"), (0, 0, 1, "X[1,1] = 1")],
            "The knob W[1,0]: its arrow to picture 0's score has local slope 0, its arrow to picture 1's has 1",
            ["knob", "scores"], at=0.55)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the matrices, the same knob `W[2, 0]` looks like this: the knob is outlined, the pixel it
    multiplies is blue, and the scores it touches, with their blames, are red.
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]], "rows": ["picture 0", "picture 1"],
         "cols": ["pixel 0", "pixel 1", "pixel 2"], "marks": {(0, 2): "blue", (1, 2): "blue"}},
        "@",
        {"name": "W", "values": [["W[0,0]", "W[0,1]"], ["W[1,0]", "W[1,1]"], ["W[2,0]", "W[2,1]"]],
         "cols": ["digit 0", "digit 1"], "marks": {(2, 0): "box"}},
        "→",
        {"name": "S", "values": [["S[0,0]", "S[0,1]"], ["S[1,0]", "S[1,1]"]],
         "cols": ["digit 0", "digit 1"], "marks": {(0, 0): "red", (1, 0): "red"}},
        "→",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]],
         "cols": ["digit 0", "digit 1"], "marks": {(0, 0): "red", (1, 0): "red"}},
    ], "The knob W[2,0] multiplies pixel 2 of each picture and is part of each picture's digit-0 score")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: three more knobs

    Work out the slopes of three more knobs the same way. For the knob `W[j, c]`: the arrows go
    to the digit-$c$ score of each picture; each arrow's local slope is that picture's pixel $j$;
    multiply each by that score's blame, and add.
    """)
    return


@app.cell(hide_code=True)
def _():
    knob_boxes = hand.boxes(["W[0, 0]", "W[1, 1]", "W[2, 1]"], step=0.1)
    return (knob_boxes,)


@app.cell(hide_code=True)
def _(knob_boxes):
    hand.marked(knob_boxes, {"W[0, 0]": -0.2, "W[1, 1]": -0.4, "W[2, 1]": -0.2}, {
        "W[0, 0]": r"""pixel 0 and digit 0: $X[0, 0] \times G[0, 0] + X[1, 0] \times G[1, 0]$, and only picture 0 has ink in pixel 0""",
        "W[1, 1]": r"""pixel 1 and digit 1: only picture 1 has ink in pixel 1, so only the blame of `S[1, 1]` counts""",
        "W[2, 1]": r"""pixel 2 and digit 1: both pictures have ink in pixel 2, so add both blames of digit 1""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Read the signs as the step rule will, moving each knob against its slope:

    - `W[0, 0]` goes **up**. Only picture 0 has ink in pixel 0, and picture 0 is a 0: from now
      on, ink there counts more for a 0.
    - `W[1, 0]` (the worked example, slope 0.4) goes **down**. Only picture 1 has ink in pixel 1,
      and the model gave that 1 a share of 0.8 for being a 0: ink there will count less for a 0.
    - `W[1, 1]` goes **up**: the same pixel, and the same picture, which got only 0.2 for being
      the 1 it is: ink there will count more for a 1.
    - `W[2, 1]` goes up a little. Both pictures have ink in pixel 2, so both their blames
      count, $0.2 + (-0.4)$, and they partly cancel.

    ### The pattern: two columns

    Look at what you multiplied for the knob `W[j, c]`. Going down the pictures, you took
    pixel $j$ of each one, which is **column $j$ of $X$**, and the blame of digit $c$ for each
    one, which is **column $c$ of $G$**. You multiplied them in pairs and added up: that is the
    dot product of the two columns. For `W[2, 1]`:

    | picture | pixel 2 (column 2 of `X`) | blame of digit 1 (column 1 of `G`) | product |
    |---|---|---|---|
    | 0 | 1 | 0.2 | 0.2 |
    | 1 | 1 | −0.4 | −0.4 |
    | **sum** | | | **−0.2** |
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]], "rows": ["picture 0", "picture 1"],
         "cols": ["pixel 0", "pixel 1", "pixel 2"], "marks": {(0, 2): "blue", (1, 2): "blue"}},
        "·",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]],
         "cols": ["digit 0", "digit 1"], "marks": {(0, 1): "red", (1, 1): "red"}},
        "=",
        {"name": "slope of W[2,1]", "values": [[-0.2]], "marks": {(0, 0): "box"}},
    ], "Column 2 of X dotted with column 1 of G gives the slope of the knob W[2,1]: 1 times 0.2 plus 1 times -0.4")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Columns into rows: the transpose

    We want that dot product for every pixel $j$ and every digit $c$: a matrix of slopes with
    the same shape as `W`. `matrix @ matrix` (chapter 06) dots every **row** of the first matrix
    with every column of the second, so the columns of $X$ would have to be rows. Turning every
    column into a row is called the **transpose**, written `X.T`: row $j$ of `X.T` is column $j$
    of `X`. Our `(2, 3)` matrix becomes `(3, 2)`:
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X", "values": [[1, 0, 1], [0, 1, 1]], "rows": ["picture 0", "picture 1"],
         "cols": ["pixel 0", "pixel 1", "pixel 2"], "marks": {(0, 2): "blue", (1, 2): "blue"}},
        "→",
        {"name": "X.T", "values": [[1, 0], [0, 1], [1, 1]], "rows": ["pixel 0", "pixel 1", "pixel 2"],
         "cols": ["picture 0", "picture 1"], "marks": {(2, 0): "blue", (2, 1): "blue"}},
    ], "Transposing X: column 2 of X becomes row 2 of X.T")
    return


@app.cell
def _():
    X_tiny = np.array([[1, 0, 1], [0, 1, 1]])
    X_tiny.T
    return (X_tiny,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now `X.T @ G` dots row $j$ of `X.T` (pixel $j$ in every picture) with column $c$ of `G`
    (digit $c$'s blame in every picture), for every $j$ and every $c$: all the knob slopes at
    once. Shapes: `(3, 2) @ (2, 2)` gives `(3, 2)`, the shape of `W`.
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X.T", "values": [[1, 0], [0, 1], [1, 1]], "rows": ["pixel 0", "pixel 1", "pixel 2"],
         "cols": ["picture 0", "picture 1"], "marks": {(2, 0): "blue", (2, 1): "blue"}},
        "@",
        {"name": "G", "values": [[-0.2, 0.2], [0.4, -0.4]], "cols": ["digit 0", "digit 1"],
         "marks": {(0, 1): "red", (1, 1): "red"}},
        "=",
        {"name": "slopes for W", "values": [[-0.2, 0.2], [0.4, -0.4], [0.2, -0.2]],
         "rows": ["pixel 0", "pixel 1", "pixel 2"], "cols": ["digit 0", "digit 1"],
         "marks": {(2, 1): "box"}},
    ], "X.T times G gives one slope per knob, in the shape of W: row 2 of X.T dotted with column 1 of G is the slope in row 2, column 1")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Your three answers are in it, at `[0, 0]`, `[1, 1]` and `[2, 1]`, and the worked ones at `[2, 0]` and `[1, 0]`. In NumPy:
    """)
    return


@app.cell
def _(X_tiny):
    G_tiny = np.array([[-0.2, 0.2], [0.4, -0.4]])
    X_tiny.T @ G_tiny
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And `b`? `b[c]` is added, times 1, to the digit-$c$ score of every picture, so its slope is
    the sum of the blames in column $c$ of $G$: `np.sum(G, axis=0)`, which adds **down** each
    column. Here $-0.2 + 0.4 = 0.2$ for digit 0 and $0.2 + (-0.4) = -0.2$ for digit 1.

    ### The same, full size

    Nothing changes for the real pictures but the sizes. `X` is `(n, 784)`, so `X.T` is
    `(784, n)`: one row per pixel, one column per picture. `G` is `(n, 10)`. Then

    > `X.T @ G` has shape `(784, n) @ (n, 10)` = `(784, 10)`, the shape of `W`

    and its entry `[j, c]` is the slope of the knob for pixel $j$ and digit $c$. Each column, like
    a column of `W`, has one number per pixel: once you've written the exercise, you'll see the
    columns drawn as $28 \times 28$ pictures.
    """)
    return


@app.cell(hide_code=True)
def _():
    grids([
        {"name": "X.T  (784, n)", "values": [["·", "·", "⋯", "·"], ["·", "·", "⋯", "·"],
                                             ["⋮", "⋮", "", "⋮"], ["·", "·", "⋯", "·"]],
         "rows": ["pixel 0", "pixel 1", "⋮", "pixel 783"], "cols": ["pic 0", "pic 1", "⋯", "pic n-1"],
         "marks": {(1, c): "blue" for c in range(4)}},
        "@",
        {"name": "G  (n, 10)", "values": [["·", "·", "⋯", "·"], ["·", "·", "⋯", "·"],
                                          ["⋮", "⋮", "", "⋮"], ["·", "·", "⋯", "·"]],
         "rows": ["pic 0", "pic 1", "⋮", "pic n-1"], "cols": ["digit 0", "digit 1", "⋯", "digit 9"],
         "marks": {(r, 3): "red" for r in range(4)}},
        "=",
        {"name": "slopes for W  (784, 10)", "values": [["·", "·", "⋯", "·"], ["·", "·", "⋯", "·"],
                                                       ["⋮", "⋮", "", "⋮"], ["·", "·", "⋯", "·"]],
         "rows": ["pixel 0", "pixel 1", "⋮", "pixel 783"], "cols": ["digit 0", "digit 1", "⋯", "digit 9"],
         "marks": {(1, 3): "box"}},
    ], "At full size: X.T of shape 784 by n, times G of shape n by 10, gives the slopes for W, 784 by 10")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: gradient

    Write `gradient(X, Y, W, b)`, giving back the slopes for `W` and the slopes for `b`, as a pair:

    1. The shares `P`: softmax of the scores.
    2. The blames of the scores, $G = (P - Y) / n$. The number of pictures $n$ is `len(X)`, the number of rows of `X`.
    3. The slopes for `W`, made from `X.T` and `G`, and the slopes for `b`, the sum down each column of `G`.

    The slopes for `W` must have the shape of `W`, and those for `b` the shape of `b`.
    """)
    return


@app.function
def gradient(X, Y, W, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("07", "gradient", gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The scores are `X @ W + b`, as in chapter 06.""",
        r"""`X.T @ G` has shape `(784, n) @ (n, 10)`. `np.sum(G, axis=0)` adds down the columns. Give both back, separated by a comma.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A last check by nudging, on the real pictures: the knob for pixel 300 and digit 3, at the
    all-zero start. `nudge` is all zeros except `h` in row 300, column 3, so `W_zero + nudge`
    nudges only that one knob. (It uses Exercises 1 to 3.)
    """)
    return


@app.cell
def _(W_zero, X, Y, b_zero, h):
    nudge = np.zeros((784, 10))
    nudge[300, 3] = h
    before = cross_entropy(softmax(X @ W_zero + b_zero), Y)
    after = cross_entropy(softmax(X @ (W_zero + nudge) + b_zero), Y)
    dW_zero, db_zero = gradient(X, Y, W_zero, b_zero)
    (after - before) / h, dW_zero[300, 3]
    return (dW_zero,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both about 0.0108. The formula and the nudge agree.

    ### The slopes as pictures

    Column $c$ of `dW_zero` has one slope per pixel for digit $c$, so it can be drawn as a
    $28 \times 28$ picture, like the columns of `W` in chapter 06. Here is minus the slope for
    each digit, the direction the first step will move its knobs: **blue** knobs go up, **red**
    ones go down.
    """)
    return


@app.cell(hide_code=True)
def _(dW_zero):
    images.weight_maps([-dW_zero[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A blurry copy of each digit, after a single step. At the all-zero start every share is 0.1,
    so a knob goes up where its digit has more ink than pictures usually have, and down where it
    has less: the matrix `X.T @ G` holds all ten of these pictures at once.

    ## Training

    Gradient descent, as in chapter 05: start at zero, take 100 steps with learning rate 0.5.
    Afterwards we measure the loss and the accuracy on the training pictures, and then on the
    **test** pictures, which the model never saw while learning. The test accuracy is the honest
    one: it says how well the model does on new handwriting. (This cell takes a few seconds, and
    needs all three exercises.)
    """)
    return


@app.cell
def _(X, X_test, Y, test_labels, train_labels):
    W = np.zeros((784, 10))
    b = np.zeros(10)
    for step in range(100):
        dW, db = gradient(X, Y, W, b)
        W = W - 0.5 * dW
        b = b - 0.5 * db
    train_loss = cross_entropy(softmax(X @ W + b), Y)
    train_acc = accuracy(X @ W + b, train_labels)
    test_acc = accuracy(X_test @ W + b, test_labels)
    train_loss, train_acc, test_acc
    return W, b


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The loss falls from 2.303 to about 0.38, and the model now gets about 0.90 (**90%**) of the
    test pictures right, up from 10%. It reads handwriting it has never seen. The averaged templates of chapter 06 got 62%. Here training and
    test accuracy are almost equal: a model this simple can't memorise its training pictures.
    Bigger models can, and chapter 09 shows what that looks like.

    ## What the model learned

    Each column of $W$ is a digit's 784 knobs, one per pixel, so we can draw it as a $28 \times 28$
    picture: **blue** where ink raises that digit's score, **red** where ink lowers it. Chapter 06's templates were blue only. These have also learned where ink counts *against* a digit.
    """)
    return


@app.cell(hide_code=True)
def _(W):
    images.weight_maps([W[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can see rough outlines of the digits: the 0 is a blue ring with red in the hole, where a
    0 has no ink. The 1 is a blue stroke down the middle. Each column is a **template**, and a
    picture's score is how well it matches.

    Here are some test pictures the model got wrong, with its guess:
    """)
    return


@app.cell(hide_code=True)
def _(W, X_test, b, test_images, test_labels):
    _guesses = np.argmax(X_test @ W + b, axis=1)
    _wrong = np.nonzero(_guesses != test_labels)[0][:8]
    images.digits(test_images[_wrong], [f"{test_labels[i]} → {_guesses[i]}" for i in _wrong])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (Each caption is "true digit → guess".) Some are honestly hard; others are digits written
    off-centre or at a slant, which a single template can't cover.

    ## What's next

    A template can only say "ink here is good, ink there is bad", and add it up. It can't say
    "a 7 has a stroke here **or** there, but not both". In **Chapter 08** you'll stack a hidden
    layer of simple pieces in between, and the model becomes a **neural network**.

    ## What you can do now

    - turn scores into probabilities with softmax, safely
    - say why accuracy can't be a loss, and measure a classifier with cross-entropy (mean surprise)
    - derive the slopes of $e^x$ (itself) and $\log x$ ($1 / x$) by nudging
    - derive the gradient of cross-entropy, $p - y$, and compute it for all knobs with `X.T @ G`
    - train a digit classifier and test it on pictures it has never seen

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** this model is called *softmax regression* or *multinomial logistic
    regression*; $\log$ here is the *natural logarithm*, often written `ln`.
    """)
    return


if __name__ == "__main__":
    app.run()
