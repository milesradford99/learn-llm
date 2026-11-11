import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="07 · A loss for choices")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits
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
    Three things to notice: $e^s$ is always positive; a higher score always gives a higher
    $e^s$; and every +1 in the score multiplies $e^s$ by about 2.718. (Why e and not 2? Because
    with e the slopes come out simplest; you'll see exactly why below.)

    Then **divide each by their total**, so they add up to 1. For scores `[2, 1, 0.1]`:
    $e^2 = 7.39$, $e^1 = 2.72$, $e^{0.1} = 1.11$, total 11.21, so the shares are
    $7.39 / 11.21 = 0.659$, $0.242$ and $0.099$. This recipe, $e^s$ then divide by the total,
    is called **softmax**. Its shares are the model's **probabilities**: how sure it is of each
    digit.

    One practical problem: `np.exp(1000)` is far too big for the computer, so it gives `inf`
    (infinity), and `inf / inf` is not a number (NumPy shows that as `nan`, short for "not a
    number"). The fix: before $e^{\square}$, subtract the largest score $m$ from every score. This doesn't
    change the shares at all: by the rule above, $e^{s - m} = e^s / e^m$, so every $e^{\square}$ is divided
    by the same $e^m$, and dividing every part and the total by the same number leaves the shares
    unchanged. After subtracting, the largest score is 0, so $e^{\square}$ of it is 1 and nothing gets too
    big for the computer. (If you leave this step out, NumPy also prints a warning saying
    "overflow": that's this problem.)

    ### Rows, columns and `keepdims`

    For a matrix of scores we do this row by row: each picture gets its own shares.
    `np.max(S, axis=1)` gives the largest number of each row and `np.sum(E, axis=1)` the total of
    each row, one number per picture. Add `keepdims=True` and the result keeps a column shape,
    `(10000, 1)` instead of `(10000,)`. A column like that broadcasts across each row: with
    `m = np.max(S, axis=1, keepdims=True)`, `S - m` subtracts each picture's own largest score
    from all ten of its scores (chapter 05 used a row
    against every row; this is a column against every column).
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
    one share per picture. Here is a helper that builds `Y` from the labels: a matrix of zeros,
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
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## All pictures, all knobs

    For the mean loss over `n` pictures, each picture's slopes are divided by `n`. So the
    slopes of the loss for all the scores at once are

    > `G = (P - Y) / n`

    a matrix of shape `(n, 10)`, one row per picture, one column per digit: `G[i, c]` is the blame
    of score `S[i, c]`.

    Now from scores to knobs, with the same idea as in chapter 05 (there the blame vector came
    first, `(2 * e) @ X`; here the shapes call for a different arrangement, worked out below). The knob `W[j, c]` (pixel $j$, digit $c$)
    multiplies pixel $j$ in digit $c$'s score, for every picture. So its slope is: for each
    picture, the score's blame `G[i, c]` times the local slope `X[i, j]` (in $X[i, j] \times W[j, c]$,
    nudging `W[j, c]` moves the score by `X[i, j]` per unit), all added up, because `W[j, c]`
    feeds $n$ scores and their blames add. That
    is the dot product of **column $j$ of $X$** with **column $c$ of $G$**.

    We want that for every `j` and every `c`: a `(784, 10)` matrix, the same shape as `W`. If
    `X`'s columns were rows, it would be `matrix @ matrix`. Turning columns into rows is called
    the **transpose**, written `X.T`: row `j` of `X.T` is column `j` of `X`, so `X.T` has shape
    `(784, n)`. Then

    > `X.T @ G` has shape `(784, n) @ (n, 10)` = `(784, 10)`, with entry `[j, c]` = column `j` of
    > `X` dotted with column `c` of `G`

    which is exactly the slope of every knob. And the slope for `b[c]` is the sum of column $c$
    of $G$ (each $b$ adds 1 $\times$ itself to its score, for every picture): `np.sum(G, axis=0)`, which
    adds **down** each column.
    """)
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
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Both about 0.0108. The formula and the nudge agree.

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
