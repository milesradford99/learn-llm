import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="06 · Recognizing digits")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np

    from learn_llm import hand, images, todo
    from learn_llm.checks import check
    from learn_llm.data import digits as load_digits

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 06 · Recognizing digits

    So far every model guessed one number: ice creams sold. In this chapter the model looks at a
    picture of a handwritten digit and says which digit it is: 0, 1, 2, … or 9. That's
    **classification**: choosing one answer from a fixed list. In this chapter you'll turn pictures into
    numbers, score them, and build a first classifier by hand. In chapter 07 the machine learns a better one.

    ## The pictures

    The course comes with 12,000 handwritten digits from a famous collection called MNIST.
    `load_digits()` hands them over in four parts: 10,000 pictures to learn from, their
    correct digits (the **labels**), and 2,000 more pictures with labels that we keep aside to
    test the model at the end.
    """)
    return


@app.cell
def _():
    train_images, train_labels, test_images, test_labels = load_digits()
    train_images.shape
    return test_images, test_labels, train_images, train_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The shape has three numbers, `(10000, 28, 28)` (shown above as a list of three numbers:
    10000, 28 and 28). That's 10,000 pictures, each a grid of 28
    rows by 28 columns. Each number in the grid is one dot of the picture (a **pixel**): 0 is
    white paper, 255 is black ink, and the numbers in between are greys.

    /// admonition | 🧰 Toolbox: picking items out
    To pick one item out of an array (or a list), put its position in square brackets, counting
    from 0: `train_images[0]` is the first picture, `train_labels[0]` its label. For a matrix,
    give the row and then the column: for a matrix $M$, `M[2, 5]` is the number in row 2,
    column 5. One position on a matrix gives a whole row: `train_images[0]` is the first picture
    (a $28 \times 28$ matrix), so `train_images[0][14]` is row 14 of it, a horizontal line across its
    middle:
    ///
    """)
    return


@app.cell
def _(train_images, train_labels):
    train_labels[0], train_images[0][14]
    return


@app.cell(hide_code=True)
def _(train_images, train_labels):
    mo.vstack([
        mo.md(r"""
    Label 6 (NumPy shows a single number with its kind around it: `np.uint8(6)` is a whole number
    from 0 to 255, and later `np.float64(...)` is a decimal; only the number inside matters), and
    a row that is white (0) except where the pen crossed it. Here are the first 8 pictures, drawn
    from their numbers, with their labels:
    """),
        images.digits(train_images[:8], train_labels[:8]),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One long row per picture

    Our models work on vectors, so we unroll each $28 \times 28$ grid into one row of 784 numbers
    ($28 \times 28 = 784$): row 0 of the picture, then row 1 after it, and so on. `reshape` does that:
    it keeps the same numbers in the same order and only changes the shape. We also divide by
    255, so every pixel is between 0 (white) and 1 (black), for the same reason we rescaled
    temperatures in chapters 04 and 05: small numbers of similar size suit one learning rate.
    The shapes below are `(10000, 784)` and `(2000, 784)`: one row of 784 numbers per picture.
    """)
    return


@app.cell
def _(test_images, train_images):
    X = train_images.reshape(10000, 784) / 255
    X_test = test_images.reshape(2000, 784) / 255
    X.shape, X_test.shape
    return X, X_test


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Ten scores, one per digit

    In chapter 05 one vector of knobs turned a day's inputs into one guess. Now we want ten
    numbers, one **score** per digit, so we give each digit its own vector of 784 knobs (plus
    its own `b`). The score for digit 3 is the dot product of the picture with digit 3's knobs:
    if those knobs are big where a 3 usually has ink, a 3 gets a high score.

    Put the ten knob vectors side by side as the ten **columns** of a matrix `W` of shape
    `(784, 10)`, and the ten `b`s in a vector `b` of shape `(10,)`. Then

    > `S = X @ W + b`

    gives all the scores for all the pictures at once. `matrix @ matrix` takes the dot product of
    **every row** of the first with **every column** of the second: the score of picture $i$ for
    digit $c$ is row $i$ of $X$ dotted with column $c$ of $W$. The shape rule is the one from
    chapter 05: `(10000, 784) @ (784, 10)` gives `(10000, 10)`; the two numbers in the middle
    (784 and 784) match and disappear. Then $+ b$ adds each digit's $b$ to its column, for every
    picture (broadcasting, as in chapter 05).

    /// admonition | 🧰 Toolbox: arrays of zeros
    Let's start with every knob at 0. `np.zeros((784, 10))` makes an array of that shape filled
    with zeros (the shape is written as a pair, in brackets), and `np.zeros(10)` a vector of 10
    zeros.
    ///
    """)
    return


@app.cell
def _(X):
    W_zero = np.zeros((784, 10))
    b_zero = np.zeros(10)
    S_zero = X @ W_zero + b_zero
    S_zero.shape
    return S_zero, W_zero, b_zero


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: the shape rule

    Before going on, try the rule on a smaller case. Say `X` holds 5 pictures, so its shape is `(5, 784)`, and `W` is as above, `(784, 10)`.
    How many rows and how many columns does `X @ W` have? And how many knobs does `W` hold in all?
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['rows', 'columns', 'knobs in W'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'rows': 5, 'columns': 10, 'knobs in W': 7840}, {
        'rows': r"""one row of scores for each picture""",
        'columns': r"""one score for each digit""",
        'knobs in W': r"""784 rows of 10 knobs""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The guess

    The model's guess is the digit with the **highest** score. Here are made-up scores for 4 pictures, with only 3 digits to keep it small:

    | picture | score for 0 | score for 1 | score for 2 | label |
    |---|---|---|---|---|
    | 0 | 1 | 3 | 2 | 1 |
    | 1 | 0 | 0 | 5 | 2 |
    | 2 | 4 | 1 | 1 | 1 |
    | 3 | 2 | 6 | 3 | 1 |

    Picture 0 has its highest score, 3, in the column of digit 1, so the model guesses 1. Its label is 1 too, so that guess is right.

    ## By hand: guesses and accuracy

    Fill in the guesses for pictures 1, 2 and 3. Then the **accuracy**: the share of the 4 pictures whose guess equals the label, as a number between 0 and 1.
    """)
    return


@app.cell(hide_code=True)
def _():
    table_boxes = hand.boxes(['picture 1', 'picture 2', 'picture 3', 'accuracy'], step=0.01)
    return (table_boxes,)


@app.cell(hide_code=True)
def _(table_boxes):
    hand.marked(table_boxes, {'picture 1': 2, 'picture 2': 0, 'picture 3': 1, 'accuracy': 0.75}, {
        'picture 1': r"""the column with the biggest number in its row""",
        'picture 2': r"""the column with the biggest number in its row""",
        'picture 3': r"""the column with the biggest number in its row""",
        'accuracy': r"""count the pictures whose guess equals the label (picture 0 is one of them), and divide by 4""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: guess

    /// admonition | 🧰 Toolbox: where is the biggest?
    `np.argmax(v)` gives the *position* of the biggest number in a vector: `np.argmax(np.array([3, 9, 4]))` is 1.

    For a matrix `M`, `np.argmax(M, axis=1)` works **along each row**, and gives one position per row. `axis=0` would work down each column instead.
    When two numbers tie for the biggest, `np.argmax` takes the first.
    ///

    Write `guess(S)`. `S` is a matrix of scores, with one row per picture and one column per digit. Give back one guessed digit per picture.

    On the table above it must give `[1, 2, 0, 1]`.
    """)
    return


@app.function
def guess(S):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "guess", guess, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The position of the biggest score in a row *is* the guessed digit, because column 3 holds the scores of digit 3.""",
        r"""One line, with `np.argmax` along each row.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: accuracy

    /// admonition | 🧰 Toolbox: comparing arrays
    `==` (two equals signs) asks "is it equal?", while a single `=` stores. On arrays it compares position by position:
    `np.array([1, 2, 0]) == np.array([1, 5, 0])` gives `[True, False, True]`.

    `np.mean` of such an array counts each `True` as 1 and each `False` as 0, so it gives the share of `True`s: here $2 / 3$.
    ///

    Write `accuracy(S, labels)`: the share of pictures whose guess equals their label. Use your `guess`.

    On the table above, with the labels `[1, 2, 1, 1]`, it must give 0.75.
    """)
    return


@app.function
def accuracy(S, labels):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "accuracy", accuracy, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three steps: the guesses, the comparison with the labels, the mean.""",
        r"""It fits in one line: the mean of `guess(S) == labels`.""",
    ], LOCALE)
    return


@app.cell
def _(S_zero, train_labels):
    accuracy(S_zero, train_labels)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With all scores 0 there's a tie everywhere, and `np.argmax` picks the first position: the
    model guesses 0 for every picture. It's right exactly when the picture *is* a 0, which is
    0.0969 of the time (969 of the 10,000 pictures). About 1 in 10, as you'd expect from guessing.

    ## Knobs made by hand: templates

    Before any learning, can we choose knobs that do better than guessing? Remember what a score is: the dot product of a picture with a digit's knobs.
    It's big when the picture has ink where the knobs are big. So here's a simple idea. Make the knobs of digit 3 a **typical 3**: the mean of all the 3s among the training pictures, pixel by pixel.
    A pixel where most 3s have ink gets a big knob, and a pixel where 3s are blank gets a knob near 0. Such a column of knobs is called a **template**.

    ## Exercise 3: template

    /// admonition | 🧰 Toolbox: picking rows with True and False
    Put an array of `True` and `False` in the square brackets of a matrix, one for each row, and you get only the rows marked `True`.
    With `M` holding the rows `[1, 2]`, `[3, 4]` and `[5, 6]`, `M[np.array([True, False, True])]` holds `[1, 2]` and `[5, 6]`.

    The array usually comes from a comparison. With `kinds = np.array([7, 9, 7])`, `M[kinds == 7]` gives those same two rows.

    `np.mean(M, axis=0)` is the mean **down each column**: one number per column. For the two rows just picked, it's `[3, 4]`.
    ///

    Write `template(X, labels, c)`: the mean of the rows of `X` whose label is `c`. `X` has one row per picture, so the result is one vector, with as many numbers as a picture has pixels.

    For `X` with the rows `[1, 2]`, `[3, 4]`, `[5, 6]` and the labels `[7, 9, 7]`, `template(X, labels, 7)` must give `[3, 4]`.
    """)
    return


@app.function
def template(X, labels, c):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("06", "template", template, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two steps: pick the rows whose label is `c`, then take the mean down each column.""",
        r"""`labels == c` marks the rows. `axis=0` is what makes the mean go down the columns: without it you get a single number for everything.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell (its code is hidden) builds a template for each of the ten digits with your function, and puts them side by side as the ten columns of `W`. Every `b` stays 0.
    It shows two accuracies: on the training pictures, and on the **test** pictures, the 2,000 that the templates were *not* made from.
    (This cell and the two after it need exercises 1 to 3.)
    """)
    return


@app.cell(hide_code=True)
def _(X, X_test, test_labels, train_labels):
    W_hand = np.stack([template(X, train_labels, c) for c in range(10)], axis=1)
    b_hand = np.zeros(10)
    accuracy(X @ W_hand + b_hand, train_labels), accuracy(X_test @ W_hand + b_hand, test_labels)
    return W_hand, b_hand


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.62 on both: 62% right, up from 10%, with no learning at all, only averaging. The test accuracy is the honest number: it says how the model does on handwriting
    it has never seen. Here are the ten templates as pictures, blue where the knobs are big:
    """)
    return


@app.cell(hide_code=True)
def _(W_hand):
    images.weight_maps([W_hand[:, c] for c in range(10)], list(range(10)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    They look like blurred digits, which is what they are.

    Now the weak spot. The next cell counts how often each digit is guessed on the training pictures, from digit 0 on the left to digit 9 on the right. Each digit has about 1,000 pictures.
    """)
    return


@app.cell(hide_code=True)
def _(W_hand, X, b_hand):
    [int(n) for n in np.bincount(guess(X @ W_hand + b_hand), minlength=10)]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The model says 8 for about 2,750 pictures and 0 for about 2,100, but 5 only once. The templates of 0 and 8 hold the most ink, spread wide, so they give a high score to almost anything.
    Most of the 5s end up as 0s and 8s: the fainter template of the 5 is outscored nearly every time. Averaging looks at each digit on its own. It never asks the question that matters:
    which knobs tell a 5 *apart from* an 8?

    ## What's next

    Good knobs would raise the right digit's score *and* lower the others'. Nobody can set 7,850 knobs that way by hand. Gradient descent can, if we give it a loss to go down.
    Accuracy can't be that loss. **Chapter 07** shows why, builds a loss that works, and lets the machine find its own templates. They reach 90%.

    ## What you can do now

    - pick items from arrays with `[i]` and `[i, j]`, and reshape a grid into one row
    - multiply a matrix by a matrix, and check the shapes
    - turn scores into guesses, and measure them with accuracy
    - pick rows of a matrix with `True` and `False`
    - build a classifier from templates, and test it on pictures it has never seen
    - say why averaged templates fall short

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the pictures are from *MNIST* (LeCun, Cortes and Burges). A classifier made from one mean per class is called a *nearest-centroid* or *template-matching* classifier.
    """)
    return


if __name__ == "__main__":
    app.run()
