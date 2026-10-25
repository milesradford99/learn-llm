import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="05 · Many numbers at once")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav

    LOCALE = "en"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    rain = [0, 1, 0, 1, 0, 0, 1, 0]
    weekend = [0, 0, 1, 0, 0, 1, 0, 0]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 05 · Many numbers at once

    Your engine from chapter 04 handles one number at a time. Real models work with thousands or
    millions of numbers at once. In this chapter you'll learn to handle whole lists of numbers
    in one go, and to compute a gradient for many knobs with a couple of lines.

    ## Mia's notebook has more columns

    Mia didn't only write down the temperature. She also noted whether it **rained** that day
    (1 for yes, 0 for no) and whether it was a **weekend** (1 for yes, 0 for no). Here are her 8
    days again, with the best possible line's guess (the line `w × t + b` with the lowest loss,
    about 27) and by how much it missed. (Days are numbered from 0, the way Python counts.)
    """)
    return


@app.cell(hide_code=True)
def _():
    _w, _b = 11.9445, -107.4326  # the best line for these 8 days
    _lines = ["| day | temperature | rain | weekend | sold | line's guess | miss |", "|---|---|---|---|---|---|---|"]
    for _i, (_t, _r, _k, _s) in enumerate(zip(temps, rain, weekend, sales)):
        _g = _w * _t + _b
        _lines.append(f"| {_i} | {_t} °C | {_r} | {_k} | {_s} | {_g:.0f} | {_g - _s:+.0f} |")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Look at the "miss" column. On the three rainy days (days 1, 3 and 6) the line guessed **too
    many**: people buy less in the rain. On the two weekend days (2 and 5) it guessed **too
    few**. A line only knows the temperature, so it can't do better. Let's give the model the other
    columns too.

    ## One knob per input

    Give each input its own knob:

    > guess = w_t × temperature + w_r × rain + w_k × weekend + b

    `w_t` is how many more ice creams per degree, `w_r` how many more (or fewer) on a rainy day,
    `w_k` how many more on a weekend (k for wee**k**end, since `w` is already taken), and `b` the starting height, as before. Try knobs
    `w_t = 12`, `w_r = -8`, `w_k = 4`, `b = -105` on day 1 (17 °C, rain, not a weekend):

    > 12 × 17 + (-8) × 1 + 4 × 0 + (-105) = 204 − 8 + 0 − 105 = 91

    She sold 90: closer than the line's guess of 96.

    Look at how that sum is built: the knobs `[12, -8, 4]` and the day's inputs `[17, 1, 0]` are
    multiplied **in pairs**, first with first, second with second, third with third, and the
    products are added up. Then `b` is added.

    A list of numbers like `[12, -8, 4]` is called a **vector**. Multiplying two vectors in pairs
    and adding up the products is called their **dot product**:

    > dot([12, -8, 4], [17, 1, 0]) = 12 × 17 + (-8) × 1 + 4 × 0 = 196

    You already know this from shopping. If apples cost 2, bread 3 and milk 1, and you buy 4
    apples, 1 bread and 2 milks, the bill is the dot product of prices and amounts:
    2 × 4 + 3 × 1 + 1 × 2 = 13. A model's guess works the same way: each knob is a "price" per
    unit of its input.

    ## Exercise 1: dot

    Write `dot(ws, xs)`: the dot product of two vectors of the same length. It's like chapter
    01's loss loop: start a `total` at 0, go through both lists in pairs with
    `for w, x in zip(ws, xs):`, and add `w * x` to the total each time. Return the total.
    """)
    return


@app.function
def dot(ws, xs):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "dot", dot, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## All the days at once: a table

    Each day is a vector of three inputs. Stack all 8 days and you get a table with one **row**
    per day and one **column** per input. A table of numbers like this is called a **matrix**.
    This one has 8 rows and 3 columns, so we say its **shape** is 8 by 3, written `(8, 3)`: rows
    first, then columns.

    In plain Python, a matrix is a list of lists: one inner list per row.
    """)
    return


@app.cell
def _():
    rows = [
        [14, 0, 0],
        [17, 1, 0],
        [20, 0, 1],
        [22, 1, 0],
        [25, 0, 0],
        [28, 0, 1],
        [31, 1, 0],
        [33, 0, 0],
    ]
    len(rows)
    return (rows,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `len(rows)` is 8: the outer list holds 8 values, and each of them is itself a list of 3
    numbers. A loop `for row in rows:` gives you one day's inputs at a time.

    ## Exercise 2: predict_all

    Write `predict_all(rows, ws, b)`: the list of guesses, one per row. Start with an empty list
    `guesses = []`. For each `row` in `rows`, the guess is `dot(ws, row) + b`; add it to the end
    of the list with `guesses.append(...)` (chapter 04). Return `guesses`.
    """)
    return


@app.function
def predict_all(rows, ws, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "predict_all", predict_all, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With the knobs from above, `predict_all(rows, [12, -8, 4], -105)` guesses
    63, 91, 139, 151, 195, 235, 259 and 291. She sold 63, 90, 137, 148, 193, 234, 257 and 288.

    ## NumPy: the same loops, written once

    Loops like these are so common that there's a ready-made tool for them: **NumPy**, a Python
    library (a collection of ready-made code) for working with whole vectors and matrices. This
    page has already loaded it with the line

    ```python
    import numpy as np
    ```

    which means "load NumPy, and let me call it `np` for short". After that, everything NumPy
    offers is reached with `np.` in front, like `np.array`.

    `np.array(...)` turns a list (or a list of lists) into a NumPy **array**: a vector or matrix
    that can do maths on all its numbers at once. An array has a `shape` slot that says how many
    rows and columns it has. (As in earlier chapters, values separated by commas on the last line
    show up as a numbered list; the shape `(8, 3)` itself shows up as a list of its two
    numbers, 8 and 3, and `y`'s shape `(8,)` as a list with just 8.)
    """)
    return


@app.cell
def _(rows):
    X = np.array(rows)
    y = np.array(sales)
    X.shape, y.shape
    return X, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `X` has shape `(8, 3)`: 8 rows, 3 columns. `y`, the sales, has shape `(8,)`: a vector of 8
    numbers. (A shape with one number, written with a trailing comma, is a vector.) By habit, a
    matrix gets a capital letter.

    ### Maths on every number at once

    With arrays, `+`, `-`, `*` and `/` work **position by position**. This is different from
    lists, so compare:
    """)
    return


@app.cell
def _():
    list_sum = [1, 2, 3] + [10, 20, 30]
    array_sum = np.array([1, 2, 3]) + np.array([10, 20, 30])
    array_times = np.array([1, 2, 3]) * np.array([10, 20, 30])
    list_sum, array_sum, array_times
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `+` on two lists **glues** them together into one longer list. On two arrays it adds
    position by position: `[11, 22, 33]`. And `*` on arrays multiplies position by position:
    `[10, 40, 90]`. (Inside a numbered list like this one, NumPy shows an array without commas,
    as `[11 22 33]`.)

    A single number works against every position:
    """)
    return


@app.cell
def _():
    np.array([1, 2, 3]) * 10 + 5
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### `@`: dot products

    The dot product has its own sign in NumPy: `@`. Between two vectors, `@` is exactly your
    `dot`:
    """)
    return


@app.cell
def _():
    np.array([12, -8, 4]) @ np.array([17, 1, 0])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    196, as before. (NumPy may show a single number as `np.int64(196)` or `np.float64(...)`:
    read it as just the number inside the brackets.)

    Between a matrix and a vector, `matrix @ vector` takes the dot product of **every row** with
    the vector, and gives back one number per row. That is exactly your `predict_all` without
    the `+ b`:
    """)
    return


@app.cell
def _(X):
    X @ np.array([12, -8, 4]) - 105
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same 8 guesses, in one line: `X @ w` gives one dot product per day, and `- 105` works
    against every one of them.

    ### Shapes must fit

    For `matrix @ vector`, each row must be as long as the vector: here, 3 inputs and 3 knobs.
    In shapes: `(8, 3) @ (3,)` gives `(8,)`. The two numbers **in the middle** (3 and 3) must
    match, and they disappear; the number on the outside (8) is what's left. If they don't match, NumPy stops with an
    error. The next cell does this on purpose, with 2 knobs for 3 inputs, so you know what that
    error looks like:
    """)
    return


@app.cell
def _(X):
    X @ np.array([12, -8])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The error mentions `matmul` (short for *matrix multiplication*, NumPy's name for `@`) and the
    sizes that don't fit. Ignore the middle part (`gufunc signature ...`); the useful bit is at
    the end: `size 2 is different from 3`. When you see it, check the shapes.

    ## Exercise 3: predict_np

    Write `predict_np(X, w, b)`: the guesses for every row of `X`, using `@`. Here `X` is a
    matrix and `w` a vector, both NumPy arrays. One line: `return X @ w + b`.
    """)
    return


@app.function
def predict_np(X, w, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "predict_np", predict_np, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: the loss, all at once

    The loss is the mean squared error from chapter 01: for every day the error
    `e = guess − sold`, then the mean of `e × e`. With arrays, every step works on all 8 days at
    once:

    1. `e = X @ w + b - y`: the 8 errors, as one vector.
    2. `e * e`: the 8 squared errors (position by position).
    3. `np.mean(...)`: NumPy's mean of all the numbers in an array.

    Write `mse_np(X, y, w, b)` and return `np.mean(e * e)`.
    """)
    return


@app.function
def mse_np(X, y, w, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "mse_np", mse_np, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With knobs `w = [10, 0, 0]` and `b = -60`, rain and weekend are switched off and the model is
    chapter 01's line `10 × t − 60`. Its loss is 177.5, the same number you found in chapter 01.
    With `[12, -8, 4]` and `-105` the loss is 4.

    ## The gradient for many knobs

    Now the slopes. Chapter 03 worked them out for one day of the line: with
    `e = w × t + b − s` and day loss `e × e`, the blame of `e` is `2 × e`, and

    - the slope for `w` is `2 × e × t` (the local slope from `e` to `w` is `t`, the input `w`
      multiplies);
    - the slope for `b` is `2 × e` (the local slope from `e` to `b` is 1).

    The gradient of the mean loss was the **mean** of these over all days.

    With three inputs, one day's error is `e = w_t × t + w_r × r + w_k × k + b − s`. Nothing new
    happens. Nudge `w_r` while the other knobs stay still: only the `w_r × r` part of the sum
    moves, by `r` per unit of `w_r`, so the local slope from `e` to `w_r` is `r`. The same goes
    for every knob: each one multiplies its own input, so by the same rule

    - slope for `w_t` = `2 × e × t`, slope for `w_r` = `2 × e × r`, slope for `w_k` = `2 × e × k`,
    - slope for `b` = `2 × e`.

    So **each knob's slope is 2 × e times that knob's own input**, averaged over the days. For
    `w_r` that is: take the vector of 8 values `2 × e`, multiply it in pairs with the rain
    column, add up, divide by 8. "Multiply in pairs and add up" is a dot product: the dot
    product of `2 × e` with the **rain column**.

    ### `vector @ matrix`: one dot product per column

    `@` also works the other way round. `vector @ matrix` takes the dot product of the vector with
    **every column** of the matrix, and gives back one number per column:
    """)
    return


@app.cell
def _():
    np.array([1, 1]) @ np.array([[1, 2, 3], [4, 5, 6]])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The columns of `[[1, 2, 3], [4, 5, 6]]` are `[1, 4]`, `[2, 5]` and `[3, 6]`. Dotted with
    `[1, 1]` they give 5, 7 and 9. In shapes: `(2,) @ (2, 3)` gives `(3,)`; again the two numbers in
    the middle match and disappear.

    So `(2 * e) @ X` gives, in one go, the three dot products we need: `2 × e` with the
    temperature column, with the rain column and with the weekend column. Shapes: `(8,) @ (8, 3)`
    gives `(3,)`, one number per knob. Divide by the number of days to get the means.

    The order matters. `X @ (2 * e)` would be `(8, 3) @ (8,)`: the two sizes in the middle, 3 and
    8, don't match, and you'd get the matmul error. The vector goes first.

    ## Exercise 5: gradient_np

    Write `gradient_np(X, y, w, b)`, returning the slopes for all knobs in `w`, and the slope for
    `b`:

    1. `e = X @ w + b - y`
    2. `grad_w = (2 * e) @ X / len(y)`: a vector, one slope per knob in `w`. (`len(y)` is 8, the
       number of days: `len` works on arrays too.)
    3. `grad_b = np.mean(2 * e)`: the slope for `b`.
    4. `return grad_w, grad_b`
    """)
    return


@app.function
def gradient_np(X, y, w, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("05", "gradient_np", gradient_np, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    At the line's knobs (`[10, 0, 0]`, `-60`), the slopes for `w_t` and `b` are -94 and 2.5:
    exactly chapter 03's gradient, because the extra knobs are switched off. The two new slopes
    say where the extra knobs should go: the slope for `w_r` is 6.25 (positive, so the step rule
    `knob − rate × slope` moves `w_r` **down**: rain lowers sales) and the slope for `w_k` is
    -2.75 (negative, so `w_k` moves **up**: weekends raise them).

    ## Checking it by nudging

    New formula, so let's check it the chapter 02 way: nudge one knob by a tiny `h`, see how much
    the loss moves, divide by `h`. Adding `[h, 0, 0]` to `w` nudges only `w_t` (position by
    position); `[0, h, 0]` nudges only `w_r`; `[0, 0, h]` only `w_k`. This cell uses your
    `mse_np`, so it shows an error until Exercise 4 passes:
    """)
    return


@app.cell
def _(X, y):
    h = 0.0001
    w_line = np.array([10, 0, 0])
    base = mse_np(X, y, w_line, -60)
    nudge_t = (mse_np(X, y, w_line + np.array([h, 0, 0]), -60) - base) / h
    nudge_r = (mse_np(X, y, w_line + np.array([0, h, 0]), -60) - base) / h
    nudge_k = (mse_np(X, y, w_line + np.array([0, 0, h]), -60) - base) / h
    nudge_t, nudge_r, nudge_k
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About -94, 6.25 and -2.75, the same as `gradient_np` (the tiny leftovers come from `h`, as
    in chapter 02).

    ## How much faster?

    Here are 200,000 made-up days (the computer invents temperatures, rain and weekends, and
    sales that roughly follow them). The cell below times your plain-Python `predict_all` and
    your NumPy `predict_np` on them. (It needs Exercises 1 to 3.)
    """)
    return


@app.cell(hide_code=True)
def _():
    _rng = np.random.default_rng(5)
    _many = np.column_stack([_rng.uniform(10, 38, 200_000), _rng.integers(0, 2, 200_000), _rng.integers(0, 2, 200_000)])
    _w = np.array([12.0, -8.0, 4.0])
    try:
        _start = time.perf_counter()
        predict_all(_many.tolist(), _w.tolist(), -105)
        _loops = time.perf_counter() - _start
        _start = time.perf_counter()
        predict_np(_many, _w, -105)
        _numpy = time.perf_counter() - _start
        _out = mo.md(
            f"Plain loops: **{_loops:.3f} seconds**. NumPy: **{_numpy:.4f} seconds**, "
            f"about **{_loops / _numpy:,.0f} times faster**."
        )
    except Exception:
        _out = mo.md("*Finish Exercises 1 to 3 to see the timing.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same guesses, same maths. NumPy runs the loops inside its own fast machinery instead of one
    Python step at a time. That speed is what makes training on lots of data possible.

    ## The payoff: fitting all four knobs

    Now gradient descent, as in chapters 02 and 03, but for all knobs at once: we **fit** the
    model, which means finding good values for its knobs. One detail first,
    which you met in chapter 04: temperatures (14 to 33) are much bigger than the rain and
    weekend values (0 or 1), so one learning rate wouldn't suit all knobs. We measure temperature
    from 25 °C in tens of degrees again, and leave the other two columns alone:

    > `Xs = (X - np.array([25, 0, 0])) / np.array([10, 1, 1])`

    This uses one more NumPy habit. `X` has shape `(8, 3)`, but `np.array([25, 0, 0])` has shape
    `(3,)`: a single row. NumPy uses that row against **every** row of `X`: from each day it
    subtracts 25 from the temperature and 0 from the rest. Then `/ np.array([10, 1, 1])` divides
    each day's temperature by 10 and the rest by 1. Using a smaller array against every row (or
    a single number against every position) is called **broadcasting**.

    The cell below starts all knobs at 0 and takes 500 steps of the step rule. `w - 0.1 * grad_w`
    updates all three `w` knobs at once, position by position. (It uses your `gradient_np` and
    `mse_np`.)
    """)
    return


@app.cell
def _(X, y):
    Xs = (X - np.array([25, 0, 0])) / np.array([10, 1, 1])
    fit_w = np.array([0.0, 0.0, 0.0])
    fit_b = 0.0
    for step in range(500):
        grad_w, grad_b = gradient_np(Xs, y, fit_w, fit_b)
        fit_w = fit_w - 0.1 * grad_w
        fit_b = fit_b - 0.1 * grad_b
    fit_w, fit_b, mse_np(Xs, y, fit_w, fit_b)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The loss drops to about **0.54**. The best possible line had about **27**. The knobs, read
    back in Mia's units:

    - `w_t` ≈ 119 per 10 degrees: about **11.9 more ice creams per degree**;
    - `w_r` ≈ -8.4: about **8 fewer** on a rainy day;
    - `w_k` ≈ 4.2: about **4 more** on a weekend;
    - `b` ≈ 193: the guess for a dry weekday at 25 °C (where the measured temperature is 0).

    The model found the rain and weekend effects you spotted in the table by eye, and measured
    them.

    ## What's next

    You can now compute guesses, loss and gradient for many inputs and many days in a few lines.
    In **Chapter 06** you'll teach a model to recognise handwritten digits: each image is a
    vector of hundreds of numbers, and instead of one guess the model makes ten, one per digit.

    ## What you can do now

    - explain a vector, a dot product, a matrix and its shape
    - write a dot product and a whole table of guesses with plain loops
    - use NumPy arrays: position-by-position maths, `@` for dot products, `np.mean`
    - check that shapes fit: `(8, 3) @ (3,)` gives `(8,)`, `(8,) @ (8, 3)` gives `(3,)`
    - compute the gradient for many knobs at once, and check it by nudging
    - use broadcasting to rescale columns, and fit a model with several inputs

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** a model like this is called *linear regression with several
    features*. `vector @ matrix` is the same as `matrix.T @ vector`, where `.T` is the
    *transpose* (rows and columns swapped); you'll see it written as `Xᵀ e` in books.
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("05", LOCALE)
    return


if __name__ == "__main__":
    app.run()
