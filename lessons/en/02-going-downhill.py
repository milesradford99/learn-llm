import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="02 · Going downhill")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav
    from learn_llm.plot import chart

    LOCALE = "en"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]
    B_FIXED = -108


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 02 · Going downhill

    At the end of chapter 01 you saw that the loss forms a valley, and that good knobs sit at
    the bottom. You found the bottom by dragging sliders. In this chapter the computer finds
    it by itself, and you'll understand every number it uses to get there.

    ## Where we left off

    Mia's data is the same as before:
    `temps = [14, 17, 20, 22, 25, 28, 31, 33]` and `sales = [63, 90, 137, 148, 193, 234, 257, 288]`.

    Here are the three functions you wrote in chapter 01, ready to use: the guess of a line,
    the squared error of one guess, and the loss (mean squared error) of a whole line.
    """)
    return


@app.function
def predict(w, b, x):
    return w * x + b


@app.function
def squared_error(guess, truth):
    return (guess - truth) ** 2


@app.function
def mse(w, b, xs, ys):
    running = 0
    for x, y in zip(xs, ys):
        running = running + squared_error(predict(w, b, x), y)
    return running / len(xs)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To start, we use just one knob, like the valley picture at the end of chapter 01: the
    starting height stays fixed at `b = -108`, and only the steepness `w` changes. Then the
    loss depends on `w` alone:
    """)
    return


@app.function
def loss_of_w(w):
    return mse(w, -108, temps, sales)


@app.cell
def _():
    loss_of_w(10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Which way is downhill?

    Imagine standing somewhere on the valley in fog: you can't see the bottom, only the ground
    under your feet. A simple way to find out which way is down: **take a tiny step and see
    whether you went down or up.**

    For our knob, a tiny step means changing `w` by a tiny amount, say `0.0001`, and comparing
    the loss before and after. We call that tiny change a **nudge**. Move the slider and
    compare the two losses:
    """)
    return


@app.cell(hide_code=True)
def _():
    w_here = mo.ui.slider(10, 14, step=0.1, value=10.5, label="w", show_value=True)
    return (w_here,)


@app.cell(hide_code=True)
def _(w_here):
    _w = w_here.value
    _before, _after = loss_of_w(_w), loss_of_w(_w + 0.0001)
    _after2 = loss_of_w(_w + 0.0002)
    _change = _after - _before
    mo.vstack([
        w_here,
        mo.md(f"""
| | w | loss |
|---|---|---|
| here | {_w:.4f} | {_before:.4f} |
| nudged right | {_w + 0.0001:.4f} | {_after:.4f} |
| nudged twice as far | {_w + 0.0002:.4f} | {_after2:.4f} |

The loss changed by **{_change:+.4f}**: nudging right goes **{"down" if _change < 0 else "up"}hill**,
so the bottom is to the **{"right" if _change < 0 else "left"}**.
"""),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How steep? The slope

    The change in the loss tells us the direction. It also tells us how steep the ground is,
    but only once we compare it with the size of the nudge: a nudge twice as big gives a change
    about twice as big (look at the last row of the table above: nudging by 0.0002 instead of
    0.0001 roughly doubles the change).
    The change on its own depends on how big a nudge we chose; the change **per unit** of nudge
    doesn't.

    /// admonition | 🧰 Toolbox: "per", dividing two changes
    Dividing one amount by another tells you "how much of this **per** one of that". If 3 kg of
    apples cost 6 euros, the price is 6 ÷ 3 = 2 euros **per** kg. Buy twice as much and you pay
    twice as much, but the price per kg stays 2.
    ///

    Here we divide the change in the loss by the change in `w` (the nudge):

    > slope = (loss after the nudge − loss before) ÷ nudge

    The result is the **slope**: how much the loss changes **per** unit of `w`, right where you
    stand. It's the same idea as the steepness `w` of Mia's line in chapter 01: there, `w` said
    how many more ice creams per extra degree; here, the slope says how much more loss per
    extra unit of `w`.

    (Short notation you'll see from here on: `x²` means `x × x`, in Python `x ** 2`; and `2x`
    means `2 × x`. Maths books leave out the × sign; in Python you must always write `*`.)

    Read the slope like this:

    - its **sign** gives the direction: negative means "the loss goes down if `w` grows", positive
      means "the loss goes down if `w` shrinks";
    - its **size** gives the steepness: far from the bottom the ground is steep (big slope), near the
      bottom it's almost flat (slope near 0).

    In the picture, the dashed line has exactly the slope at the big dot: it's the straight line that
    just touches the curve there.
    """)
    return


@app.cell(hide_code=True)
def _(w_here):
    _w = w_here.value
    try:
        _s = nudge_slope(loss_of_w, _w)
        _note = f"Your `nudge_slope` says the slope at w = {_w:g} is **{_s:,.1f}**."
    except Exception:
        _s = (loss_of_w(_w + 0.0001) - loss_of_w(_w)) / 0.0001
        _note = f"The slope at w = {_w:g} is **{_s:,.1f}**. (After Exercise 1, your own function computes this.)"
    _y = loss_of_w(_w)
    mo.vstack([
        w_here,
        chart((10, 14), (0, 2700), "w (steepness of Mia's line)", "loss",
              curve=[(10 + i * 0.05, loss_of_w(10 + i * 0.05)) for i in range(81)],
              marker=(_w, _y), segments=[(_w - 0.6, _y - 0.6 * _s, _w + 0.6, _y + 0.6 * _s)]),
        mo.md(_note),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: measure a slope by nudging

    Write `nudge_slope(f, x)`: the slope of the function `f` at the input `x`, measured with a
    nudge of `0.0001`.

    One new idea first. **A function is a value too**, just like a number or a list: you can
    hand it to another function as an input. Then, inside that other function, you call it
    like any function. For example:
    """)
    return


@app.function
def square(x):
    return x * x


@app.function
def call_with_three(f):
    return f(3)


@app.cell
def _():
    call_with_three(square)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `call_with_three(square)` hands the function `square` itself to `call_with_three` (no
    parentheses after `square`: we pass the recipe, not an answer). Inside, `f` stands for
    `square`, so `f(3)` is `square(3)`, which is `9`.

    In `nudge_slope(f, x)`, `f` is whichever function you want the slope of. Steps:

    1. store the nudge: `h = 0.0001`
    2. work out `f(x + h) - f(x)`: how much `f` changed
    3. return that change divided by `h`

    You can write it in one `return` line, but parentheses matter: the subtraction must happen
    before the division (remember, `/` goes before `-` otherwise).
    """)
    return


@app.function
def nudge_slope(f, x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("02", "nudge_slope", nudge_slope, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Taking a step

    Now we know which way is downhill and how steep it is. Here is the rule for one step:

    > new w = w − rate × slope

    In Python: `w - rate * slope`. Why a **minus**? Check all the cases:

    | slope | means | − rate × slope | so w... |
    |---|---|---|---|
    | negative (e.g. −2000) | downhill is to the right | positive | grows: moves right ✓ |
    | positive (e.g. +2000) | downhill is to the left | negative | shrinks: moves left ✓ |
    | near 0 | almost at the bottom | near 0 | barely moves ✓ |

    In every case, the sign of the slope tells you which side is uphill, and the minus sends `w`
    the opposite way: downhill.

    Why multiply by the slope instead of always stepping the same amount? Far from the bottom
    the ground is steep, so the step is big; near the bottom it's flat, so steps get small by
    themselves and the walk slows down as it arrives.

    The **learning rate** (`rate` in the code) is a small number you choose that sets how big the
    steps are overall. It has nothing to do with the "per" division above; it's just the name
    everyone uses. Here, the slopes are in the thousands (about −2,400 at `w = 10`), so the learning
    rate has to be tiny, like `0.0008`, or a single step would fly far past the valley.

    ## Exercise 2: one step

    Write `step(w, slope, rate)`: it gives back the new `w` after one step downhill. Type the
    Python form, `w - rate * slope`, with `-` and `*` from the keyboard; the `−` and `×` in the
    rule above are maths symbols that Python doesn't accept.
    """)
    return


@app.function
def step(w, slope, rate):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("02", "step", step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: walk downhill

    One step isn't enough: we repeat the same two moves, measure the slope and take a step,
    many times. This repeating is called **gradient descent**: walking downhill one step at a
    time.

    To repeat something a given number of times, use `range`:
    `for i in range(5):` runs the indented lines 5 times (`i` counts 0, 1, 2, 3, 4; we
    don't need it here, it just has to have a name).

    Write `descend(f, w, rate, steps)`: start at `w`, and `steps` times: measure the slope of `f`
    at the current `w` with your `nudge_slope`, then use your `step` to replace `w` with the new
    value (`w = step(...)`). After the loop, return `w`. Exercises 1 and 2 come first.

    Store the result in `w` itself, not in a new name like `step` or `slope`: `step` is already
    the name of your function, and reusing it would hide the function from Python.
    """)
    return


@app.function
def descend(f, w, rate, steps):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("02", "descend", descend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Watch it walk

    Here is your walk on Mia's valley, starting at `w = 10`, 15 steps. Each dot is where one step
    landed. Try the learning rates in the menu:
    """)
    return


@app.cell(hide_code=True)
def _():
    rate_choice = mo.ui.dropdown(
        {"0.0001 (small)": 0.0001, "0.0008": 0.0008, "0.0015": 0.0015, "0.0018 (big)": 0.0018},
        value="0.0001 (small)", label="learning rate",
    )
    return (rate_choice,)


@app.cell(hide_code=True)
def _(rate_choice):
    try:
        _path = [10]
        for _i in range(15):
            _path.append(step(_path[-1], nudge_slope(loss_of_w, _path[-1]), rate_choice.value))
        _end = _path[-1]
        _out = mo.vstack([
            rate_choice,
            chart((6, 18), (0, 12000), "w", "loss",
                  curve=[(6 + i * 0.1, loss_of_w(6 + i * 0.1)) for i in range(121)],
                  dots=[(p, loss_of_w(p)) for p in _path if 6 <= p <= 18]),
            mo.md(f"After 15 steps: w = **{_end:.3f}**, loss = **{loss_of_w(_end):,.1f}**."),
        ])
    except Exception:
        _out = mo.md("*Finish Exercises 1 and 2 to watch the walk.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    What you should see:

    - **0.0001**: every step is safe but short; after 15 steps it's still on its way.
    - **0.0008**: a few steps and it's at the bottom.
    - **0.0015**: the steps are so big they jump over the bottom to the other side, back and forth,
      but the jumps shrink, so it still settles.
    - **0.0018**: each jump lands higher up than the last, and the walk flies out of the valley.
      (Dots outside the picture aren't drawn.)

    Choosing the learning rate is a balance: too small is slow, too big never arrives. You'll
    meet this choice in every model in this course.

    ## Both knobs at once

    Mia's line has two knobs. Nudging works the same way for each: measure the slope for `w`
    while holding `b` still, measure the slope for `b` while holding `w` still, then step both.
    The cell below does exactly that with your `nudge_slope` and `step`, starting from `w = 6`,
    `b = 0`. Pick how many steps:
    """)
    return


@app.cell(hide_code=True)
def _():
    steps_choice = mo.ui.dropdown(["100", "1000", "5000", "20000"], value="100", label="steps")
    return (steps_choice,)


@app.cell(hide_code=True)
def _(steps_choice):
    try:
        _w, _b = 6, 0
        for _i in range(int(steps_choice.value)):
            _sw = nudge_slope(lambda v: mse(v, _b, temps, sales), _w)
            _sb = nudge_slope(lambda v: mse(_w, v, temps, sales), _b)
            _w, _b = step(_w, _sw, 0.001), step(_b, _sb, 0.001)
        _out = mo.vstack([
            steps_choice,
            chart((10, 36), (0, 320), "temperature (°C)", "ice creams sold",
                  dots=list(zip(temps, sales)), lines=[(_w, _b)]),
            mo.md(f"w = **{_w:.2f}**, b = **{_b:.1f}**, loss = **{mse(_w, _b, temps, sales):,.1f}**"),
        ])
    except Exception:
        _out = mo.md("*Finish Exercises 1 and 2 to run this.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The computer finds Mia's line by itself: after 20,000 steps the loss is about 31, close to
    the best possible (about 27). Nobody dragged a slider. (The one-knob walk above got lower
    because we'd already handed it a good `b = -108`; here the computer had to find `b` too,
    starting from 0.)

    You may notice `b` creeps along much more slowly than `w`. The loss reacts far more strongly
    to `w` (every degree multiplies it) than to `b`, so a learning rate that is safe for `w` is
    timid for `b`. Later chapters show the standard fixes; for now, many steps are enough.

    ## The price of nudging

    Every slope measured by nudging costs an extra loss calculation **per knob**. Two knobs: two
    extra calculations per step, no problem. The language models at the end of this course have
    billions of knobs, and each loss calculation reads lots of text. Billions of extra loss
    calculations for every single step would take forever.

    So we want a way to get the **exact** slope from a formula, with no nudging. For that, let's
    look closely at what happens during a nudge, starting with the simplest curve there is.

    ## The slope of x², exactly

    Take `square(x) = x × x`. Nudge `x` by a small amount `h`. The new value is `(x + h) × (x + h)`.
    How much bigger is that than `x × x`? A picture answers it: a square with side `x + h` is made
    of four pieces.
    """)
    return


@app.cell(hide_code=True)
def _():
    _x, _h = 200, 60  # drawing sizes, not values
    _o = 20
    mo.Html(f"""
<svg viewBox="0 0 330 330" width="100%" style="max-width:330px" role="img"
     aria-label="A square of side x plus h, split into x squared, two rectangles x times h, and h squared"
     xmlns="http://www.w3.org/2000/svg" font-size="16" text-anchor="middle">
  <rect x="{_o}" y="{_o}" width="{_x}" height="{_x}" fill="var(--blue-4, #dbeafe)" stroke="var(--slate-9, #64748b)"/>
  <rect x="{_o + _x}" y="{_o}" width="{_h}" height="{_x}" fill="var(--crimson-4, #fee2e2)" stroke="var(--slate-9, #64748b)"/>
  <rect x="{_o}" y="{_o + _x}" width="{_x}" height="{_h}" fill="var(--crimson-4, #fee2e2)" stroke="var(--slate-9, #64748b)"/>
  <rect x="{_o + _x}" y="{_o + _x}" width="{_h}" height="{_h}" fill="var(--amber-4, #fef3c7)" stroke="var(--slate-9, #64748b)"/>
  <text x="{_o + _x / 2}" y="{_o + _x / 2 + 6}" fill="var(--slate-12, #1e293b)">x × x</text>
  <text x="{_o + _x + _h / 2}" y="{_o + _x / 2 + 6}" fill="var(--slate-12, #1e293b)">x × h</text>
  <text x="{_o + _x / 2}" y="{_o + _x + _h / 2 + 6}" fill="var(--slate-12, #1e293b)">x × h</text>
  <text x="{_o + _x + _h / 2}" y="{_o + _x + _h / 2 + 6}" fill="var(--slate-12, #1e293b)" font-size="13">h × h</text>
  <text x="{_o + _x / 2}" y="{_o + _x + _h + 22}" fill="var(--slate-11, #475569)">x</text>
  <text x="{_o + _x + _h / 2}" y="{_o + _x + _h + 22}" fill="var(--slate-11, #475569)">h</text>
</svg>""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The big blue square is the old value, `x × x`. The nudge adds two strips of `x × h` each,
    plus a tiny corner `h × h`. So:

    > change = 2 × x × h + h × h

    Now divide by the nudge `h` to get the slope, exactly as `nudge_slope` does:

    > slope = (2 × x × h + h × h) ÷ h = 2 × x + h

    Dividing a sum divides each part: (6 + 4) ÷ 2 = 3 + 2. Each part here has an `h` in it, so
    dividing by `h` removes one `h` from each: `2 × x × h ÷ h` is `2 × x`, and `h × h ÷ h` is `h`.

    So the measured slope is always `2 × x + h`: the true steepness plus a small extra that comes
    from the nudge's own size. Watch what happens at `x = 3` as the nudge shrinks:

    | nudge h | measured slope 2 × 3 + h |
    |---|---|
    | 0.1 | 6.1 |
    | 0.01 | 6.01 |
    | 0.001 | 6.001 |
    | 0.0001 | 6.0001 |

    We can't simply set `h = 0`: the change would be 0 and we'd be dividing 0 by 0, which means
    nothing. But the measurements clearly **close in** on one number, 6, getting as close as we
    like when `h` is small enough. That number is what we call the **exact** slope. Since the
    measured slope is always `2 × x + h`, the number it closes in on is **`2 × x`**.

    Check it against your `nudge_slope`, which still carries its `h = 0.0001`:
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        _out = mo.md(
            f"`nudge_slope(square, 3)` = **{nudge_slope(square, 3):.6f}**, and the formula says "
            f"`2 × 3` = **6**. The difference is the leftover `h` = 0.0001."
        )
    except Exception:
        _out = mo.md("*Finish Exercise 1 to compare.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That exact slope has a name: the **derivative**. Mathematicians write it like this:

    > d/dx (x²) = 2x

    Read it as: "the derivative of x² with respect to x is 2x", or simply, "the slope of x² at any
    `x` is `2x`". The `d/dx` is shorthand for "the exact slope, as `x` gets a vanishing nudge". From
    now on, **slope** and **derivative** mean the same thing in this course.

    ## More rules, the same way

    Every rule below comes from nudging, just like `x²` did. One fact about multiplication first:
    **multiplying a sum multiplies each part**. Seven bags of (3 apples + 2 pears) hold 7 × 3 apples
    and 7 × 2 pears: 7 × (3 + 2) = 7 × 3 + 7 × 2 = 35.

    - **A constant**, like `5`. It doesn't depend on `x` at all: nudging `x` changes nothing.
      Change 0, so the slope is **0**.
    - **`c × x`**, like `7 × x`. Nudging gives `7 × (x + h)` = `7 × x + 7 × h`. The change is
      `7 × h`; divide by `h`: the slope is **7**. In numbers at `x = 3`: 7 × 3.0001 − 7 × 3
      = 0.0007, and 0.0007 ÷ 0.0001 = 7. In general, the slope of `c × x` is **c**.
    - **`x × x`**: slope **2 × x**, from the square picture. (The picture uses lengths, so
      positive numbers, but the multiplying-out it shows, `(x + h) × (x + h)` = `x × x + 2 × x × h + h × h`,
      works for any numbers, negative ones too.)
    - **`c` times a function.** After the nudge, the function's value is its old value plus some
      change. So `c` times it is `c × (old + change)` = `c × old + c × change`: the change got
      multiplied by `c`. Dividing by `h` then gives **c times the function's slope**. In numbers,
      with `c = 4` and `x × x` at `x = 3`: 4 × 3.0001² − 4 × 3² = 4 × 0.00060001 = 0.00240004, and
      0.00240004 ÷ 0.0001 ≈ 24 = 4 × 6. So `4 × x × x` has slope `4 × 2 × x = 8 × x`.
    - **One function plus another.** If one function rises by 0.3 and the other by 0.5 for the same
      nudge, their sum rises by 0.3 + 0.5 = 0.8: the rises stack. So the slope of a sum is **the two
      slopes added**.

    | function | slope |
    |---|---|
    | a constant | 0 |
    | `c × x` | c |
    | `x × x` | 2 × x |
    | `c` × a function | c × that function's slope |
    | a function + another | the two slopes added |

    ## Exercise 4: an exact slope

    Take `f(x) = 3 × x × x + 5 × x + 2`. (This `f` is a new function with its own name; it has
    nothing to do with the input called `f` inside `nudge_slope`.) Using the rules, work out its
    slope **on paper** first:
    take each of the three parts, find its slope, add them up. Then write `slope_f(x)` to return
    that formula. In Python write `6 * x`, never `6x`: without the `*`, the cell shows a red
    error instead of the checks. The checks compare it with the right formula; under them, your `nudge_slope`
    measures the same slope as a second opinion.
    """)
    return


@app.function
def f(x):
    return 3 * x * x + 5 * x + 2


@app.function
def slope_f(x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("02", "slope_f", slope_f, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    try:
        _rows = "\n".join(f"| {x} | {slope_f(x):g} | {nudge_slope(f, x):.4f} |" for x in (-2, 0, 1, 4))
        _out = mo.md("| x | your formula | nudging |\n|---|---|---|\n" + _rows)
    except Exception:
        _out = mo.md("*Finish Exercises 1 and 4 to compare your formula with nudging.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Your formula and the nudge agree (up to the nudge's small leftover), and the formula needs no
    extra calculations at all. That's the trade we wanted.

    ## What's still missing

    Mia's loss is an average of things like `(w × 14 + b − 63)²`: a square, but of something that
    itself depends on `w`. None of our rules covers "a square of a recipe that contains `w`" yet.
    In **Chapter 03** you'll split such recipes into small steps, find each step's slope, and see
    how the slopes combine, passing blame backwards from the loss to every knob.

    ## What you can do now

    - find the downhill direction and steepness by nudging: the slope
    - walk downhill with `new w = w − rate × slope`, and pick a learning rate
    - pass a function to another function, and repeat with `range`
    - derive exact slopes (derivatives) with the nudge rules, and check them by nudging

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** in maths textbooks, the "smaller and smaller nudge" idea is called
    a *limit*, and the nudge method is the *finite difference*.
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("02", LOCALE)
    return


if __name__ == "__main__":
    app.run()
