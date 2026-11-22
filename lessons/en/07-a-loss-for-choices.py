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

    Chapter 06 ended with templates made by averaging: 62% of the test digits right, and an 8 guessed for a quarter of all pictures.
    Those knobs were never adjusted to tell digits apart. In this chapter gradient descent does that, as it did for Mia's line. What it needs is a loss,
    and building one takes two new pieces of maths, $e^x$ and $\log$. You'll derive their slopes exactly as you did in chapter 02.

    Here are your two functions from chapter 06, finished. After them come the pictures, unrolled into rows as before, with knobs that are all 0, and the accuracy of that model.
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
    0.0969: with every score at 0, the model guesses 0 for every picture, and is right for the 969 zeros.

    ## Why accuracy can't be the loss

    We want gradient descent to improve the knobs, so we need a loss with useful slopes.
    Accuracy doesn't have them: nudge one knob a tiny bit and the scores move a tiny bit, but the
    biggest score almost always stays the biggest. No guess changes, so the accuracy doesn't move
    at all: its slope is 0 almost everywhere, and the step rule never changes the knobs.

    See it on one picture, a 1, with only two digits. The score for digit 0 stays at 1. Turn the
    knob that sets the score for digit 1 slowly from -1 to 3. On the left, the accuracy on this
    picture; on the right, a smooth measure of how sure the model is that it's a 1 (built in the
    next section):
    """)
    return


@app.cell(hide_code=True)
def _():
    _xs = [x / 50 for x in range(-50, 151)]
    mo.hstack([
        chart((-1, 3), (0, 1), "score for digit 1", "accuracy",
              curve=[(x, 1.0 if x > 1 else 0.0) for x in _xs]),
        chart((-1, 3), (0, 1), "score for digit 1", "share for digit 1",
              curve=[(x, float(np.exp(x) / (np.exp(x) + np.exp(1)))) for x in _xs]),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The accuracy is a **light switch**: flat at 0, a sudden jump where the score for 1 passes the
    score for 0, then flat at 1. On a flat stretch, a tiny nudge changes nothing, so the slope is
    0 and gradient descent has no hint which way to go. The curve on the right is a **dimmer**:
    every nudge of the knob moves it a little, so at every point it has a slope that says "this
    way is better".

    We need a smooth measure instead: not just *which* digit wins, but **how sure** the model
    is of each. If scores could be turned into shares that add up to 1, like "0.7 for 3, 0.2
    for 8, 0.1 for the rest", a tiny nudge would change the shares a tiny bit, and that gives
    slopes.

    ## From scores to shares: $e^s$

    Scores can be negative, and a share can't. So first we turn each score into a positive
    number, in a way that keeps the order (a higher score must stay higher). The tool for this
    is a power of a special number called **e**, about 2.718.

    You know powers from chapter 00: `2 ** 3` is $2 \times 2 \times 2$. Powers have a handy rule:
    $2^{3 + 2} = 2^3 \times 2^2$ (five 2s multiplied either way). Powers are extended to
    every number, including 0, negatives and fractions, in the one way that keeps this rule
    true. For example: $e^3 \times e^0$ must be $e^{3 + 0} = e^3$, so $e^0 = 1$. $e^1 \times e^{-1}$ must be
    $e^0 = 1$, so $e^{-1} = 1 / e$. $e^{0.5} \times e^{0.5}$ must be $e^1$, so $e^{0.5}$ is the number that
    times itself gives e (about 1.65). One more rule follows the same way:
    $e^{a - b} \times e^b = e^a$, so **$e^{a - b} = e^a / e^b$**. We write $e^s$; in NumPy it's
    `np.exp(s)`. Here it is for scores from -3 to 3:
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
    A few values, to make it concrete:

    | score $s$ | −2 | −1 | 0 | 1 | 2 | 3 |
    |---|---|---|---|---|---|---|
    | $e^s$ | 0.14 | 0.37 | 1 | 2.72 | 7.39 | 20.09 |

    Each step to the right multiplies by 2.718; each step to the left divides by it. However
    negative the score, $e^s$ only gets closer to 0, never reaching it or going below.

    Three things to notice: $e^s$ is always positive; a higher score always gives a higher
    $e^s$; and every +1 in the score multiplies $e^s$ by about 2.718. (Why e and not 2? Because
    with e the slopes come out simplest; you'll see exactly why below.)

    Then **divide each by their total**, so they add up to 1. For scores `[2, 1, 0.1]`:
    $e^2 = 7.39$, $e^1 = 2.72$, $e^{0.1} = 1.11$, total 11.21, so the shares are
    $7.39 / 11.21 = 0.659$, $0.242$ and $0.099$. This recipe, $e^s$ then divide by the total,
    is called **softmax**. Its shares are the model's **probabilities**: how sure it is of each
    digit.

    Try it. Drag the three scores and watch the $e^s$ values and the shares:
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
        mo.vstack([mo.md(r"shares: each $e^s$ divided by the total, " + f"{_e.sum():.2f}"),
                   bars([(f"digit {i}", v) for i, v in enumerate(_e / _e.sum())], "the share of each digit", top=1.0)]),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Things to try:

    - Make all three scores equal. Every share is $1/3$: the model can't decide.
    - Push one score up by 1. Its $e^s$ is multiplied by 2.72, and it takes share from the others.
    - Set the scores to 2, 1, 0, note the shares, then to 3, 2, 1. The shares don't change at all:
      adding the same number to every score multiplies every $e^s$ by the same amount, and the
      division by the total cancels it. Only the **differences** between scores matter.

    One practical problem: `np.exp(1000)` is far too big for the computer, so it gives `inf`
    (infinity), and `inf / inf` is not a number (NumPy shows that as `nan`, short for "not a
    number"). The fix: before $e^{\square}$, subtract the largest score $m$ from every score. This doesn't
    change the shares at all: by the rule above, $e^{s - m} = e^s / e^m$, so every $e^{\square}$ is divided
    by the same $e^m$, and dividing every part and the total by the same number leaves the shares
    unchanged. After subtracting, the largest score is 0, so $e^{\square}$ of it is 1 and nothing gets too
    big for the computer. (If you leave this step out, NumPy also prints a warning saying
    "overflow": that's this problem.)

    It's the last thing you tried above, the other way round. The scores `[1000, 999, 998]` minus
    their largest, 1000, are `[0, -1, -2]`: the same differences as `[2, 1, 0]`, so the same shares,
    0.67, 0.24 and 0.09. And $e^0 = 1$, $e^{-1} = 0.37$, $e^{-2} = 0.14$ are small, friendly numbers.

    ### Rows, columns and `keepdims`

    For a matrix of scores we do this row by row: each picture gets its own shares.
    `np.max(S, axis=1)` gives the largest number of each row and `np.sum(E, axis=1)` the total of
    each row, one number per picture. Add `keepdims=True` and the result keeps a column shape,
    `(10000, 1)` instead of `(10000,)`. A column like that broadcasts across each row: with
    `m = np.max(S, axis=1, keepdims=True)`, `S - m` subtracts each picture's own largest score
    from all ten of its scores (chapter 05 used a row
    against every row; this is a column against every column). With two pictures and three
    digits:
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
    `m` holds each row's largest score (outlined in `S`), standing as a column. Picture 0's 2 is
    used against all of row 0 (blue), picture 1's 7 against all of row 1 (red). Without
    `keepdims=True`, `m` would be the vector `[2, 7]` of shape `(2,)`; NumPy would treat it as a
    **row** and try to use it against every row of `S`, whose rows have 3 numbers, not 2: an error
    (or, with as many pictures as digits, a silently wrong answer).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: shares

    Three scores: `[2, 1, 0]`. To two decimals, $e^2 = 7.39$, $e^1 = 2.72$ and $e^0 = 1$. Work out their total. Then work out the share of the first score and the share of the last, to two decimals.
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

    Write `softmax(S)` for a matrix of scores, one row per picture. Each row gets its own shares:

    1. Subtract from every score the largest score of its row.
    2. Take $e^{\square}$ of the results, with `np.exp`.
    3. Divide each by the total of its row.

    Steps 1 and 3 need one number per row that broadcasts across that row. That's what `keepdims=True` is for.

    The cell after the check tries your function on one picture's scores, `[[2, 1, 0.1]]`. It must give about `[[0.659, 0.242, 0.099]]`. Note the two pairs of
    brackets: a matrix with one row (one picture), because softmax works row by row.
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

    How should we score shares like `[0.659, 0.242, 0.099]` when the right answer is the first
    one? We want a loss that is 0 when the model gives the right digit all of its share (1.0),
    and that grows, without limit, as that share shrinks towards 0: a model that's very sure of
    a wrong answer should pay a lot.

    **log** does this. $\log$ undoes $e^{\square}$: $\log(e^a) = a$, so $\log x$ answers "e to the power of
    what gives x?" (in NumPy, `np.log(x)`). $\log 1 = 0$ (because $e^0 = 1$), and for shares below 1 the log is negative
    (only a negative power of e is below 1), heading down without limit as the share approaches
    0. So we use **minus** the log of the
    right digit's share, called the **surprise**:

    | share of the right digit | 1 | 0.9 | 0.5 | 0.1 | 0.01 |
    |---|---|---|---|---|---|
    | surprise, $-\log p$ | 0 | 0.105 | 0.693 | 2.303 | 4.605 |
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((0, 1), (0, 5), "share p of the right digit", "surprise −log p",
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
    With all scores 0, every digit gets the same share, 1/10, so every picture's surprise is
    $-\log 0.1 = 2.303$. That's the loss of a model that knows nothing. Training should bring it
    down.

    ## The slopes of $e^x$ and $\log x$

    To run gradient descent we need the loss's slopes, and the loss is built from $e^{\square}$ and $\log$.
    So, as in chapter 02, we work out their slopes by nudging.

    **$e^x$.** Nudge $x$ by a tiny $h$. By the power rule, $e^{x + h} = e^x \times e^h$, so the
    change is $e^x \times e^h - e^x$. Both parts contain $e^x$ (the second is $e^x \times 1$), so the
    change is $e^x \times (e^h - 1)$, and the ratio (change $\div$ $h$) is

    > $$e^x \times \frac{e^h - 1}{h}$$

    The part $(e^h - 1) / h$ doesn't depend on $x$ at all. The same steps work for any base:
    for base 2 the part is $(2^h - 1) / h$. Here it is for bases 2, 3 and e, as $h$ shrinks:

    | base | `h = 0.01` | `h = 0.0001` |
    |---|---|---|
    | 2 | 0.6956 | 0.6932 |
    | 3 | 1.1047 | 1.0987 |
    | e | 1.0050 | 1.0001 |

    As in chapter 02, as $h$ shrinks the ratio closes in on a fixed number: about 0.693 for base
    2, 1.099 for base 3, and exactly **1** for base e. That's what makes e special; it's in fact
    *chosen* as the base where this number is 1. So the ratio closes in on $e^x \times 1$:

    > the slope of $e^x$ is $e^x$ itself.

    In a picture: wherever you stand on the curve, its height and its steepness are the same
    number. The two straight lines touch the curve at $x = 1$ and $x = 2$ and go in its direction
    there. At $x = 1$ the curve is 2.72 high and the line climbs 2.72 per unit; at $x = 2$ it is
    7.39 high and the line climbs 7.39 per unit.
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
    **$\log x$.** Say $y = \log x$, which means $x = e^y$. Nudge $y$ by a tiny $k$: by the slope
    we just found (change $\approx$ slope $\times$ nudge), $x$ grows by about $e^y \times k = x \times k$. So when $x$ grows by $x \times k$, $\log x$
    grows by $k$. Change of $\log x$ per unit change of $x$:

    > the slope of $\log x$ is $k / (x \times k) = 1 / x$.

    ## The slope of the loss for each score

    To keep it short, pretend there are only three possible digits, 0, 1 and 2. Take one picture
    with scores `s0, s1, s2`, where 0 is the right digit. Write
    $T = e^{s_0} + e^{s_1} + e^{s_2}$ for the total, so the right share is $p_0 = e^{s_0} / T$, and the
    surprise is $L = -\log(e^{s_0} / T)$.

    One more log rule first. Take any two positive numbers $x$ and $y$, and write them as powers
    of e: $x = e^a$ and $y = e^b$, so $a = \log x$ and $b = \log y$. Then
    $x / y = e^a / e^b = e^{a - b}$ (the rule from earlier), and "e to the power of what gives
    $x / y$?" is $a - b$. So $\log(x / y) = a - b = \log x - \log y$. Applied to the surprise:

    > $L = -(\log(e^{s_0}) - \log T) = -\log(e^{s_0}) + \log T = -s_0 + \log T$

    (the minus sign in front flips both parts, and $\log(e^{s_0}) = s_0$ because log undoes $e^{\square}$).

    Now the slopes, with the rules from chapters 02 and 03 (slopes of a sum add; a chain
    multiplies, blame times local slope):

    - **For $s_0$** (the right digit): the $-s_0$ part has slope −1. The $\log T$ part: the slope
      of $\log T$ for $T$ is $1 / T$, and the slope of $T$ for $s_0$ is $e^{s_0}$ (the other two
      parts, $e^{s_1}$ and $e^{s_2}$, don't contain $s_0$: slope 0), so together $e^{s_0} / T = p_0$.
      Total: **$p_0 - 1$**.
    - **For $s_1$** (a wrong digit): $-s_0$ doesn't contain $s_1$, slope 0. $\log T$ gives
      $e^{s_1} / T = p_1$. Total: **$p_1$**. The same for $s_2$: $p_2$.

    In words: **share minus 1 for the right digit, share for every other digit.** With a one-hot
    row `y`, that's simply `p - y`. Gradient descent steps against the slope, so it pushes the
    right digit's score **up** (its slope is negative) and every other score **down**, each in
    proportion to the share it was wrongly given.

    Let's check it by nudging, the chapter 02 way, for scores `[2, 1, 0.1]`:
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
    About `-0.341`, `0.242` and `0.099`: the shares `[0.659, 0.242, 0.099]` with 1 taken off the
    right digit's. The formula holds.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: the slopes for one picture

    A picture of a 1. The model's shares for the digits 0, 1 and 2 are `[0.7, 0.2, 0.1]`: it's fairly sure, and wrong. What are the slopes of its surprise for the three scores?
    """)
    return


@app.cell(hide_code=True)
def _():
    slope_boxes = hand.boxes(['score 0', 'score 1', 'score 2'], step=0.1)
    return (slope_boxes,)


@app.cell(hide_code=True)
def _(slope_boxes):
    hand.marked(slope_boxes, {'score 0': 0.7, 'score 1': -0.8, 'score 2': 0.1}, {
        'score 0': r"""a wrong digit: the slope is its share""",
        'score 1': r"""the right digit: its share minus 1""",
        'score 2': r"""a wrong digit: the slope is its share""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The biggest slope, in size, belongs to the right digit: gradient descent will push its score up hard. And the wrong digit that got 0.7 will be pushed down much harder than the one that got 0.1.

    Think of it as a tug of war. The right digit is pulled up by $1 - p$, what it's still missing.
    Every wrong digit is pushed down by its share $p$, what it wrongly took. The pulls stop when
    the right digit has everything. Watch it happen to this picture: each step moves every score
    against its slope (learning rate 1), and the shares are worked out again.
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
        mo.vstack([mo.md("shares"), bars([(f"digit {i}", v) for i, v in enumerate(_p)], "shares", top=1.0)]),
        mo.vstack([mo.md("slopes, share minus one-hot"),
                   bars([(f"digit {i}", v) for i, v in enumerate(_p - _y)], "slopes", top=1.0)]),
        mo.md(rf"surprise: **{-np.log(_p[1]):.3f}**"),
    ], widths=[2, 2, 1])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After one step, digit 1's share jumps from 0.2 to about 0.5; after six, it's over 0.9, and the
    surprise has fallen from 1.609 towards 0. As the right share grows, every slope shrinks:
    the closer the model is to right, the gentler the pull.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## From the scores to the knobs

    Gradient descent doesn't move scores, though: it moves knobs, the $784 \times 10$ numbers in
    $W$ and the 10 in $b$. Here is the whole recipe for all the pictures at once, with the shape
    of each value:
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("X", "(n, 784)"), ("S", "(n, 10)"), ("P", "(n, 10)"), ("L", "one number")],
             ["@ W + b", "softmax", "cross-entropy"],
             "X, times W plus b, gives the scores S; softmax gives the shares P; cross-entropy gives the loss L")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Blame flows backwards, as in chapter 03. The last section took it from **one** picture's
    surprise back to that picture's scores. Two steps are still missing: the loss $L$ is the
    **mean** of all the pictures' surprises, not one surprise; and gradient descent moves knobs,
    not scores. We'll fill in both on a world small enough to see every number, with nothing but
    chapter 03's rules.

    ### A reminder: the two rules of blame

    Draw the calculation as boxes and arrows. Each arrow has a **local slope**: how much the box
    at its end moves, per unit, when the box at its start is nudged a tiny bit. The loss has
    blame 1. Walking backwards:

    - **Along a road, multiply.** If $u$ feeds $v$ with local slope 3, and $v$'s blame is 2,
      then $u$ gets $3 \times 2 = 6$ along that arrow.
    - **Where roads split, add.** If $u$ feeds several boxes, its blame is the sum of what it
      gets along each arrow. (In chapter 04, $a$ in $v = (a + b) \times a$ got 7 along one road and
      2 along the other: 9.)

    That's all this section uses.

    ### A world small enough to see

    Two pictures of only 3 pixels each, and only two digits, 0 and 1. A pixel is 1 where there's
    ink and 0 where there isn't. Picture 0 is a 0, with ink in pixels 0 and 2; picture 1 is a 1,
    with ink in pixels 1 and 2. The model gives picture 0 the shares `[0.6, 0.4]` (right, but
    not sure) and picture 1 the shares `[0.8, 0.2]` (sure, and wrong).

    Here is the end of the calculation as a graph: four scores, two surprises, one loss. Score
    `S[i, c]` is picture $i$'s score for digit $c$. The loss is the mean of the two surprises,
    $L = (\text{surprise}_0 + \text{surprise}_1) / 2$. Local slopes are on the arrows; blames are in red.
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
