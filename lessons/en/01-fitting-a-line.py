import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="01 · Fitting a line")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav
    from learn_llm.plot import chart

    LOCALE = "en"
    BEST_B = -108  # the starting height used for the one-knob picture at the end


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 01 · Fitting a line

    In this chapter a program learns its first pattern from data. There's no magic in it:
    you'll build every piece yourself, with the tools from chapter 00.

    ## The question

    Mia runs an ice-cream stand. For eight days she wrote down the temperature at noon and how
    many ice creams she sold that day. Tomorrow's forecast says 30 °C. How many should she
    prepare?

    Here is her notebook, as two lists. Position matters: the first temperature goes with the
    first sales number (that day was 14 °C and she sold 63), the second with the second, and so on.
    """)
    return


@app.cell
def _():
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]
    return sales, temps


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Walking two lists side by side

    To look at one day at a time, you need the temperature *and* the sales of that day
    together. `zip` does that: it pairs up two lists like the two sides of a zipper.

    `for t, s in zip(temps, sales):` takes one pair per round of the loop: `t` is that day's
    temperature and `s` that day's sales. (Two names before `in`, separated by a comma,
    because each round hands you two values.)

    `print` shows values as a line of text under the cell. You need it inside a loop: normally
    a cell only shows its last line's result, but `print` shows something every time it runs.
    """)
    return


@app.cell
def _(sales, temps):
    for t, s in zip(temps, sales):
        print(t, s)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Picture it

    /// admonition | 🧰 Toolbox: reading a chart
    A chart turns pairs of numbers into dots. The line along the bottom (the **x-axis**)
    measures the first number, here the temperature: further right means hotter. The line up
    the left side (the **y-axis**) measures the second number, here the sales: higher up
    means more sold. Each dot is one day: go right to its temperature and up to its sales.
    ///

    Find the dot for the day at 14 °C: it sits above 14 on the bottom line, at the height of 63.
    """)
    return


@app.cell(hide_code=True)
def _(sales, temps):
    chart((10, 36), (0, 320), "temperature (°C)", "ice creams sold", dots=list(zip(temps, sales)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The dots climb from left to right: hotter days, more ice creams. And they climb fairly
    evenly, almost along a straight line. If we can describe that line with numbers, we can
    read off any temperature, including 30 °C.

    ## A line, in numbers

    Here is a recipe that turns a temperature into a guess for the sales:

    > guess = w × temperature + b

    `w` and `b` are two numbers we get to choose. Let's see what each one does:

    - **`w`** is the **steepness**: how many more ice creams each extra degree brings. With
      `w = 12`, every degree warmer adds 12 sales. A bigger `w` makes the line steeper.
    - **`b`** is the **starting height**: the line's height at 0 °C, because at 0 °C the
      recipe gives `w × 0 + b`, and `w × 0` is 0. The big dot at the left edge of the chart
      marks it. Mia never sells ice cream at 0 °C, so it doesn't matter that `b` turns out
      negative; only the part of the line over her dots matters.

    Drag the two sliders and watch the red line. Try to lay it along the dots. Notice what `w`
    does: the line swings around the big dot at 0 °C, like a door on its hinge. That's why,
    after changing `w`, you usually have to move `b` again.
    """)
    return


@app.cell(hide_code=True)
def _():
    w_slider = mo.ui.slider(0, 20, step=0.5, value=6, label="w (steepness)", show_value=True)
    b_slider = mo.ui.slider(-200, 100, step=1, value=-100, label="b (starting height)", show_value=True)
    return b_slider, w_slider


@app.cell(hide_code=True)
def _(b_slider, sales, temps, w_slider):
    mo.vstack([
        mo.hstack([w_slider, b_slider], justify="start", gap=2),
        chart((0, 36), (-200, 320), "temperature (°C)", "ice creams sold",
              dots=list(zip(temps, sales)), lines=[(w_slider.value, b_slider.value)],
              marker=(0, b_slider.value)),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A recipe like this, which turns an input into a guess, is called a **model**. Its
    adjustable numbers, here `w` and `b`, are its **parameters**: think of them as the model's
    knobs. "Learning", in this whole course, means one thing: **finding good values for the
    knobs**. Today you turn them by hand; next chapter, the computer turns them itself.

    ## Exercise 1: the model's guess

    Write `predict(w, b, x)`: it gives back the guess for the input `x`, which in Python is
    `w * x + b`.
    (The recipe works for any input, so we call the input `x` rather than "temperature".)
    """)
    return


@app.function
def predict(w, b, x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("01", "predict", predict, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How wrong is a line?

    By eye, some lines are clearly better than others. To let a computer compare them, we need
    a **number** that says how wrong a line is.

    Start with one day. The **error** is the guess minus the truth. (A different meaning from
    chapter 00's red error box: here it's a number saying how far off a guess is.)

    - guess 150, truth 137: error = 150 − 137 = **13** (guessed 13 too many)
    - guess 120, truth 137: error = 120 − 137 = **−17** (guessed 17 too few)

    On the chart, each error is the dashed gap between a dot and the line. The table lists them
    for the line your sliders set. (These are the same two sliders as above: moving any copy
    moves them all.)
    """)
    return


@app.cell(hide_code=True)
def _(b_slider, sales, temps, w_slider):
    _w, _b = w_slider.value, b_slider.value
    _rows = "\n".join(
        f"| {_t} | {_s} | {_w * _t + _b:g} | {_w * _t + _b - _s:g} |" for _t, _s in zip(temps, sales)
    )
    mo.vstack([
        mo.hstack([w_slider, b_slider], justify="start", gap=2),
        mo.hstack([
            chart((0, 36), (-200, 320), "temperature (°C)", "ice creams sold",
                  dots=list(zip(temps, sales)), lines=[(_w, _b)], marker=(0, _b),
                  segments=[(_t, _s, _t, _w * _t + _b) for _t, _s in zip(temps, sales)]),
            mo.md("| temp | sold | guess | error |\n|---|---|---|---|\n" + _rows),
        ], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now we want one number for the whole line. The obvious idea is to add up the errors. But
    look what happens with errors of 13 and −13: they add to 0, as if the line were perfect,
    while it's wrong on both days. Too-high and too-low misses cancel out.

    The fix is to **square** each error before adding: multiply it by itself.

    /// admonition | 🧰 Toolbox: squaring
    Squaring a number means multiplying it by itself: 3² = 3 × 3 = 9. In Python, `3 ** 2`.
    `**` happens before `+` and `-`, so the square of a difference needs parentheses:
    `(10 - 7) ** 2` is 9, while `10 - 7 ** 2` is 10 − 49.

    A negative number squared is positive. Why is (−3) × (−3) = 9? Follow the pattern as the
    first number goes down by 1: 2 × (−3) = −6, 1 × (−3) = −3, 0 × (−3) = 0. Each time the result
    goes **up** by 3. One more step: (−1) × (−3) = 3, then (−2) × (−3) = 6, (−3) × (−3) = 9.
    So every squared error is zero or more, and nothing cancels.

    Squaring also makes big misses count much more than small ones: an error of 2 becomes 4,
    but an error of 10 becomes 100. A line that is badly wrong on one day gets punished hard.
    ///

    ## Exercise 2: squared error

    Write `squared_error(guess, truth)`: work out the error (guess minus truth), then give
    back its square.
    """)
    return


@app.function
def squared_error(guess, truth):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("01", "squared_error", squared_error, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One number for the whole line: the loss

    Take the squared error of every day, then their **mean** (the average from chapter 00:
    add them up, divide by how many). That is the **mean squared error**. Lower is better;
    0 would mean the line passes exactly through every dot.

    A number that measures how wrong a model is has a general name: the **loss**. The mean
    squared error is the loss we use in this chapter. To get a feel for its size: if a line
    missed by about 5 ice creams every day, each squared error would be about 25, and so would
    their mean.

    ## Exercise 3: the loss of a line

    Write `mse(w, b, xs, ys)`, the mean squared error of the line with knobs `w` and `b` on
    the inputs `xs` and true answers `ys`. You have everything you need:

    1. Start a running total at 0: `running = 0`.
    2. Loop over the pairs with `for x, y in zip(xs, ys):`. For each pair, use your `predict`
       to get the guess, and your `squared_error` to compare it with `y`. Add the result to
       the running total the way chapter 00 counted with `big = big + 1`. It's easiest in two
       lines: first `guess = predict(w, b, x)`, then add `squared_error(guess, y)` to
       the running total: `running = running + ...`. Python works out the right side first, using the old
       `running`, then stores the result back in `running`.
    3. After the loop, return the running total divided by how many pairs there are
       (`len(xs)`).

    Finish Exercises 1 and 2 first: this one calls both.
    """)
    return


@app.function
def mse(w, b, xs, ys):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("01", "mse", mse, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The hunt

    Now use your loss as a scoreboard. Move the sliders to make the loss as small as you can.
    Get it **below 100** and Mia can trust your line.

    A strategy that works:

    1. Estimate the steepness from the data: the dots rise from 63 to 288 (225 more ice creams)
       while the temperature goes from 14 to 33 (19 degrees warmer). That's about 225 ÷ 19 ≈ 12
       more ice creams per degree, so set `w` near 12.
    2. Move `b` while watching the loss number (not just the picture), and stop where the
       loss is smallest.
    3. Try `w` one notch up or down, then redo step 2. If the loss got smaller, keep going in
       that direction; if not, go back. Stop when changing `w` no longer helps.

    For fine control, click a slider and use the **arrow keys**: each press moves it one notch.
    """)
    return


@app.cell(hide_code=True)
def _(b_slider, sales, temps, w_slider):
    try:
        _loss = mse(w_slider.value, b_slider.value, temps, sales)
        _verdict = (
            mo.callout(mo.md(f"**Loss: {_loss:,.1f}.** Below 100: a good line! For 30 °C it predicts "
                             f"**{predict(w_slider.value, b_slider.value, 30):g}** ice creams."), kind="success")
            if _loss < 100 else mo.callout(mo.md(f"**Loss: {_loss:,.1f}.** Keep going: get it below 100."), kind="warn")
        )
    except Exception:
        _verdict = mo.callout(mo.md("*Finish Exercise 3 to see the loss here.*"), kind="neutral")
    mo.vstack([
        mo.hstack([w_slider, b_slider], justify="start", gap=2),
        chart((0, 36), (-200, 320), "temperature (°C)", "ice creams sold",
              dots=list(zip(temps, sales)), lines=[(w_slider.value, b_slider.value)],
              marker=(0, b_slider.value)),
        _verdict,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## The shape of the loss

    How did you find a good line? Probably by nudging one slider, checking whether the loss
    went down, and nudging again. Let's draw that search.

    To draw a picture, keep just one knob: fix the starting height at `b = {BEST_B}` (close to
    the best value) and only change `w`. This chart has new axes: along the bottom is `w`, not
    the temperature, and the height is the loss. So each point on the curve stands for one whole
    line, with that steepness, and how wrong it is. It was worked out with **your**
    `mse`. Drag the slider to move the big dot along the curve.
    """)
    return


@app.cell(hide_code=True)
def _():
    valley_w = mo.ui.slider(10, 14, step=0.1, value=10.5, label="w", show_value=True)
    return (valley_w,)


@app.cell(hide_code=True)
def _(sales, temps, valley_w):
    try:
        _curve = [(10 + i * 0.05, mse(10 + i * 0.05, BEST_B, temps, sales)) for i in range(81)]
        _w = valley_w.value
        _out = mo.vstack([
            valley_w,
            chart((10, 14), (0, 2700), "w (steepness), with b fixed at -108", "loss",
                  curve=_curve, marker=(_w, mse(_w, BEST_B, temps, sales))),
        ])
    except Exception:
        _out = mo.md("*Finish Exercise 3 to see this picture.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The loss forms a **valley**. Lines that are too flat (left) or too steep (right) have a high
    loss; the best `w` sits at the bottom. Finding good knobs means finding the bottom of the
    valley.

    With one or two knobs you can do that by eye. The language models at the end of this
    course have *billions* of knobs, and nobody can drag billions of sliders. In **Chapter 02**
    the computer will find the bottom of the valley by itself, by always stepping downhill.

    ## What you can do now

    - read a chart of dots, and walk two lists side by side with `zip`
    - describe a straight line with two knobs: steepness `w` and starting height `b`
    - measure how wrong a model is with a loss: the mean squared error
    - find good knobs by making the loss small

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** this method is called *linear regression*, one of the oldest
    tools in statistics.
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("01", LOCALE)
    return


if __name__ == "__main__":
    app.run()
