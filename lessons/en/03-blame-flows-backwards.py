import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="03 · Blame flows backwards")

with app.setup(hide_code=True):
    import time

    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav
    from learn_llm.diagrams import one_day
    from learn_llm.plot import chart

    LOCALE = "en"
    temps = [14, 17, 20, 22, 25, 28, 31, 33]
    sales = [63, 90, 137, 148, 193, 234, 257, 288]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 03 · Blame flows backwards

    Chapter 02 ended with a gap. Mia's loss is built from pieces like `(w × 14 + b − 63)²` (the `²` means "times itself"):
    a square of something that itself depends on `w`. Our slope rules don't cover that yet. In
    this chapter you'll find the exact slope of anything built from small steps, and you'll see
    the idea that trains every neural network, including the language models at the end of this
    course.

    Here are the functions from earlier chapters that you'll reuse: the loss of a line
    (chapter 01) and slope by nudging (chapter 02).
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


@app.function
def nudge_slope(f, x):
    h = 0.0001
    return (f(x + h) - f(x)) / h


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One day, step by step

    Take Mia's first day: `t = 14` degrees, `s = 63` ice creams sold. Take the line with knobs
    `w = 10` and `b = -60`. The squared error for that day is worked out in four small steps.
    Each step does **one** thing, and gives its result a name. (In the code, the names end in `1`
    for "day 1"; in the text and pictures we just say `p`, `g`, `e`, `L`.)
    """)
    return


@app.cell
def _():
    p1 = 10 * 14      # p = w × t: the steepness part of the guess
    g1 = p1 + (-60)   # g = p + b: the guess
    e1 = g1 - 63      # e = g − s: the error
    L1 = e1 * e1      # L = e × e: the squared error
    p1, g1, e1, L1
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (The last line asks for four values at once, separated by commas. The page shows them as a
    numbered list of items, counting from 0: item 0 is `p` = 140, item 1 is `g` = 80, item 2 is
    `e` = 17, item 3 is `L` = 289.)

    Here are the same steps as a picture. Each box holds one value. An arrow means "this value
    is used to make that one", and the symbol on the arrow says how. Data you can't change sits
    on the left (`t`, and `s` lower down); the knobs `w` and `b` are the values we want to adjust.
    """)
    return


@app.cell(hide_code=True)
def _():
    one_day({"w": 10, "t": 14, "p": 140, "b": -60, "g": 80, "s": 63, "e": 17, "L": 289})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A picture like this, of how values are made from other values, is called a
    **computation graph**.

    ## Each step on its own is easy

    Look at one arrow at a time and ask the chapter 02 question: if the value at the start of the
    arrow got a tiny nudge, how much would the value at the end move, **per unit** of the nudge?
    (Everything else stays still.) That's the **local slope** of the arrow. Every one of them is a
    rule you already have:

    | step | nudge this | the result moves, per unit | why (chapter 02 rule) | here |
    |---|---|---|---|---|
    | `L = e × e` | `e` | `2 × e` | slope of x × x is 2 × x | 2 × 17 = **34** |
    | `e = g − s` | `g` | `1` | `g` goes up by h, so `e` goes up by h | **1** |
    | `g = p + b` | `p` | `1` | `p` goes up by h, so `g` goes up by h | **1** |
    | `g = p + b` | `b` | `1` | same, for `b` | **1** |
    | `p = w × t` | `w` | `t` | slope of c × x is c (here c is `t`, which stays fixed) | **14** |

    ## Exchange rates multiply

    /// admonition | 🧰 Toolbox: chaining exchange rates
    Suppose 1 euro buys 8 yuan, and 1 yuan buys 20 yen. How many yen does 1 euro buy? Change the
    euro into 8 yuan, then each of those 8 yuan into 20 yen: 8 × 20 = **160** yen. When you convert
    through a chain, the rates **multiply**.
    ///

    The local slopes are exactly such rates: "per unit of this, that many units of that". Follow a
    tiny nudge `h` of `w` all the way to `L`:

    | value | moves by |
    |---|---|
    | `w` | `h` |
    | `p` | `14 × h` (rate 14) |
    | `g` | `1 × 14 × h` = `14 × h` (rate 1) |
    | `e` | `1 × 14 × h` = `14 × h` (rate 1) |
    | `L` | about `34 × 14 × h` = `476 × h` (rate 34; the tiny leftover is explained below) |

    So `L` moves 476 times as much as `w`: the slope of `L` with respect to `w` is
    **14 × 1 × 1 × 34 = 476**. Multiplying local slopes along the path is called the
    **chain rule**.

    Is it really exact? Look at where a leftover could come from. The first three steps are
    "straight-line" steps (`× 14`, `+ b`, `− s`): for those, the change is **exactly** the rate
    times the nudge, with nothing left over (like `7 × (x + h) = 7 × x + 7 × h` in chapter 02).
    So `e` moves by exactly `14 × h`. Only the last step, the square, leaves something over. Use
    chapter 02's square picture with side `e = 17`, nudged by `14 × h` instead of `h`: it grows by
    two strips of `17 × 14 × h` (together `2 × 17 × 14 × h` = `476 × h`) plus the small corner
    `(14 × h) × (14 × h)` = `196 × h × h`. Divide by `h`:

    > measured slope = 476 + 196 × h

    The same pattern as `2 × x + h` in chapter 02: as the nudge shrinks, the measurements close in
    on **476**. (What if a bending step sits in the middle, with a straight-line step after it?
    Take `3 × (x × x)`. The square leaves a corner `h × h`; the `× 3` afterwards makes it `3 × h × h`;
    divided by `h` that's `3 × h`, which still vanishes as `h` shrinks. Straight-line steps only
    scale a leftover, they never stop it vanishing.) Check it by nudging:
    """)
    return


@app.cell
def _():
    def loss_day1(w):
        return (w * 14 + (-60) - 63) ** 2

    nudge_slope(loss_day1, 10)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Nudging with `h = 0.0001` gives about `476.0196` (the long tail of digits after that is the
    computer's rounding noise), which is `476 + 196 × 0.0001`: exactly the chain rule's 476, plus
    the leftover we just predicted. Smaller nudges get closer still:

    | nudge h | 0.1 | 0.01 | 0.001 | 0.0001 |
    |---|---|---|---|---|
    | measured slope | 495.6 | 477.96 | 476.196 | 476.0196 |

    ## Working backwards: blame

    We could follow a nudge forward from `w`, then again from `b`, and so on for every knob. But
    there's a smarter order. Start at the end and give every box a **blame**: how much `L` moves
    per unit that this box moves. Then walk backwards, arrow by arrow. ("Per unit" here is a rate
    for a tiny nudge, as in chapter 02, not a jump of a whole 1: moving `e` from 17 to 18 would
    move `L` by 35, the 34 plus the square's leftover.)

    - `L` itself: moving `L` by 1 moves `L` by 1. Blame **1**.
    - `e`: one unit of `e` moves `L` by 34 (the local slope), and each unit of `L` is worth 1.
      Blame 34 × 1 = **34**.
    - `g`: one unit of `g` moves `e` by 1, and each unit of `e` is worth 34. Blame 1 × 34 = **34**.
    - `p`: one unit of `p` moves `g` by 1. Blame 1 × 34 = **34**.
    - `b`: one unit of `b` also moves `g` by 1. Blame 1 × 34 = **34**.
    - `w`: one unit of `w` moves `p` by 14, and each unit of `p` is worth 34. Blame 14 × 34 = **476**.

    (Blames can be negative too. A negative blame means raising that box *lowers* the loss; the
    step rule's minus then pushes the knob up, which is downhill.)

    One rule did all of it:

    > blame of a box = local slope of its arrow × blame of the box the arrow points to

    Here are the blames, in red, under each box:
    """)
    return


@app.cell(hide_code=True)
def _():
    one_day({"w": 10, "t": 14, "p": 140, "b": -60, "g": 80, "s": 63, "e": 17, "L": 289},
            {"w": "blame 476", "p": "blame 34", "b": "blame 34", "g": "blame 34", "e": "blame 34", "L": "blame 1"})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The blame of `w` is its slope, 476, the same as the chain rule's answer. That's no coincidence:
    forwards we multiplied 14 × 1 × 1 × 34, backwards 34 × 1 × 1 × 14, the same numbers in the
    other order, and the order of a multiplication doesn't change its result. The blame of `b` is
    its slope, 34. This backward walk is called **backpropagation**: the blame for the loss flows
    backwards through the graph to every knob.

    Why backwards? Notice that `w` and `b` share the end of the path (`g` → `e` → `L`). Going
    backwards, that shared part is worked out **once**, and every knob reuses it. The backward walk
    visits each box once, one multiplication per arrow, just like the forward walk does. So a model
    with a billion knobs gets all billion slopes for about the cost of working out the loss once
    more. Compare that with nudging: two loss calculations for **every** knob.

    The slopes of the loss for all the knobs together are called the **gradient**. That's where
    *gradient descent*, the walk downhill from chapter 02, gets its name.

    ## Exercise 1: fill in the table by hand

    Now you do one. Take Mia's third day: `t = 20`, `s = 137`, with the same knobs `w = 10` and
    `b = -60`. First work **forwards** (the values), then **backwards** (the blames). Type each
    answer in its box; each one is checked as you go.
    """)
    return


@app.cell(hide_code=True)
def _():
    hand = mo.ui.dictionary({
        name: mo.ui.number(label=name, step=1)
        for name in ("p", "g", "e", "L", "blame e", "blame g", "blame p", "blame b", "blame w")
    })
    return (hand,)


@app.cell(hide_code=True)
def _(hand):
    _answers = {"p": 200, "g": 140, "e": 3, "L": 9,
                "blame e": 6, "blame g": 6, "blame p": 6, "blame b": 6, "blame w": 120}
    _hints = {
        "p": "p = w × t", "g": "g = p + b (b is negative)", "e": "e = g − s", "L": "L = e × e",
        "blame e": "local slope of L = e × e is 2 × e; times the blame of L (1)",
        "blame g": "local slope 1, times the blame of e",
        "blame p": "local slope 1, times the blame of g",
        "blame b": "local slope 1, times the blame of g",
        "blame w": "local slope t (here 20), times the blame of p",
    }
    _lines = []
    for _name, _right in _answers.items():
        _v = hand.value[_name]
        if _v is None:
            _lines.append(f"- ⬜ **{_name}**: {_hints[_name]}")
        elif abs(_v - _right) < 1e-9:
            _lines.append(f"- ✅ **{_name}** = {_v:g}")
        else:
            _lines.append(f"- ❌ **{_name}** = {_v:g} is not right yet. Hint: {_hints[_name]}")
    _done = all(hand.value[n] is not None and abs(hand.value[n] - r) < 1e-9 for n, r in _answers.items())
    mo.vstack([
        mo.hstack([hand["p"], hand["g"], hand["e"], hand["L"]], justify="start"),
        mo.hstack([hand["blame e"], hand["blame g"], hand["blame p"], hand["blame b"], hand["blame w"]],
                  justify="start", wrap=True),
        mo.callout(mo.md("\n".join(_lines)), kind="success" if _done else "neutral"),
        one_day({"w": 10, "t": 20, "b": -60, "s": 137,
                 **({k: _answers[k] for k in ("p", "g", "e", "L")} if _done else {})},
                {k.split()[1]: f"blame {_answers[k]}" for k in _answers if k.startswith("blame")} if _done else {}),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## When a value is used twice, blames add

    One more case. Take a step `a × c` where **both** sides are values that can move. Nudging only
    `a` (holding `c` still) moves the result by `c` per unit; nudging only `c` moves it by `a` per
    unit. That's the `c × x` rule from chapter 02, once for each side.

    What if both sides move at the same time, each by `h`? Picture a rectangle with sides `a` and
    `c`, like chapter 02's square. Growing both sides by `h` adds a strip `c × h` along one side, a
    strip `a × h` along the other, and a tiny corner `h × h`:

    > (a + h) × (c + h) = a × c + **c × h** + **a × h** + h × h

    The strip `c × h` is the effect through the first arrow (from `a`), and `a × h` is the effect
    through the second arrow (from `c`). They sit side by side in the total, so they **add**. The
    corner `h × h` is the usual leftover that vanishes as `h` shrinks.

    Now take `x × x`: the **same** box feeds both sides, so `a` and `c` are both `x` and both move
    together. The rule gives:

    > blame of x = (through the first arrow) + (through the second arrow) = x + x = **2 × x**

    That's exactly the slope of `x × x` from chapter 02, as it must be: with `a = c = x`, this
    rectangle *is* chapter 02's square. The general rule: when a value feeds several arrows, the
    blames arriving through each of them add up.

    The same thing happens in Mia's full loss. The loss is the **mean** over 8 days, and `w` is
    used on every day. So `w` collects blame from all 8 days, added up, then divided by 8 (the
    mean divides by 8, and "c times a function" gives c times its slope). In short:

    > the slope of the mean loss = the mean of the slopes of each day

    ## Exercise 2: one day's slopes in code

    Write `day_slopes(w, b, t, s)`, doing exactly what you did by hand: the forward steps, then the
    backward steps. It gives back **two** values, the slope for `w` and the slope for `b`.

    To give back two values, separate them with a comma: `return blame_w, blame_b`. Whoever calls
    the function catches both the same way: `dw, db = day_slopes(10, -60, 14, 63)` stores 476 in
    `dw` and 34 in `db`. (Like `for t, s in zip(...)`: two names, two values.)

    The cell lists the steps as comments, one box per line: three forward, then the blames
    backwards, then the `return`. To turn a comment into code, delete the `#` **and the space
    after it**, so the line starts in the same column as `todo()`. Do that for every line, then
    delete `todo()`. (If you see `IndentationError: unexpected indent`, a space after `#` is
    still there.)

    `L` itself needs no line: its blame is 1, so `blame_e = 2 * e * 1`, which is just `2 * e`.
    """)
    return


@app.function
def day_slopes(w, b, t, s):
    # p = w * t
    # g = p + b
    # e = g - s
    # blame_e = 2 * e
    # blame_g = blame_e
    # blame_p = blame_g
    # blame_b = blame_g
    # blame_w = t * blame_p
    # return blame_w, blame_b
    todo()


@app.cell(hide_code=True)
def _():
    check("03", "day_slopes", day_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: the gradient for all days

    Write `gradient(w, b, xs, ys)`: the slopes of the **mean** loss, for `w` and for `b`. As we
    worked out, that's the mean of each day's slopes:

    1. Two running totals at 0: `total_w = 0` and `total_b = 0`.
    2. Loop over the days with `for t, s in zip(xs, ys):`. Get that day's slopes with
       `dw, db = day_slopes(w, b, t, s)` and add each to its running total.
    3. After the loop (press Backspace so `return` lines up with `for`):
       `return total_w / len(xs), total_b / len(xs)`.
    """)
    return


@app.function
def gradient(w, b, xs, ys):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("03", "gradient", gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checking the gradient against nudging

    A gradient written by hand can hide a mistake that still gives plausible numbers. Professionals
    always compare it against nudging, a habit called a **gradient check**. Here is yours, at a few
    different knob settings:
    """)
    return


@app.cell(hide_code=True)
def _():
    try:
        _rows = []
        for _w, _b in ((10, -60), (12, -108), (6, 0), (15, -150)):
            _dw, _db = gradient(_w, _b, temps, sales)
            _nw = nudge_slope(lambda v: mse(v, _b, temps, sales), _w)
            _nb = nudge_slope(lambda v: mse(_w, v, temps, sales), _b)
            _rows.append(f"| {_w}, {_b} | {_dw:,.3f} | {_nw:,.3f} | {_db:,.3f} | {_nb:,.3f} |")
        _out = mo.md("| w, b | your slope for w | nudging | your slope for b | nudging |\n"
                     "|---|---|---|---|---|\n" + "\n".join(_rows))
    except Exception:
        _out = mo.md("*Finish Exercises 2 and 3 to run the gradient check.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Once your exercises pass, each pair matches, up to the nudge's small leftover. For `w` the nudging column is about 0.06
    off: that's the leftover again, here the average of `t × t` over Mia's days (about 603.5)
    times `h = 0.0001`. (If your columns disagree by much more, your checks above will show which
    exercise to look at.)

    ## Exercise 4: gradient descent with exact slopes

    Write `fit(w, b, xs, ys, rate, steps)`: gradient descent on both knobs, with your exact
    gradient instead of nudging. Repeat `steps` times with `for i in range(steps):`, and inside
    the loop:

    - `dw, db = gradient(w, b, xs, ys)`
    - `w = w - rate * dw` and `b = b - rate * db` (the step rule from chapter 02, once per knob)

    After the loop, `return w, b`.
    """)
    return


@app.function
def fit(w, b, xs, ys, rate, steps):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("03", "fit", fit, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Race: nudging against backpropagation

    Both methods below fit Mia's line from `w = 6`, `b = 0`, with 20,000 steps and the same
    learning rate. The left one measures slopes by nudging (as in chapter 02); the right one uses
    your `fit`. Press the button to race them.
    """)
    return


@app.cell(hide_code=True)
def _():
    race = mo.ui.run_button(label="Race")
    race
    return (race,)


@app.cell(hide_code=True)
def _(race):
    mo.stop(not race.value, mo.md("*Press **Race** to start.*"))
    try:
        _start = time.perf_counter()
        _w, _b = 6, 0
        for _i in range(20000):
            _sw = nudge_slope(lambda v: mse(v, _b, temps, sales), _w)
            _sb = nudge_slope(lambda v: mse(_w, v, temps, sales), _b)
            _w, _b = _w - 0.001 * _sw, _b - 0.001 * _sb
        _t_nudge = time.perf_counter() - _start
        _start = time.perf_counter()
        _fw, _fb = fit(6, 0, temps, sales, 0.001, 20000)
        _t_back = time.perf_counter() - _start
        _out = mo.md(f"""
| | nudging | backpropagation |
|---|---|---|
| loss calculations per step | 4 (2 per knob) | about 1 (one forward + backward pass) |
| time | {_t_nudge:.2f} s | {_t_back:.2f} s |
| final w, b | {_w:.2f}, {_b:.1f} | {_fw:.2f}, {_fb:.1f} |
| final loss | {mse(_w, _b, temps, sales):.1f} | {mse(_fw, _fb, temps, sales):.1f} |
""")
    except Exception:
        _out = mo.md("*Finish Exercises 2 to 4 to run the race.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Same answer, and backpropagation is faster even with only two knobs. With two knobs the gap
    is small; with a billion knobs, nudging would need a billion extra loss calculations per step,
    while backpropagation still needs one backward walk.

    ## What's next

    You found the backward steps for this one recipe by hand. Every time the recipe changes (a
    curve instead of a line, more inputs, a neural network), the backward steps would have to be
    worked out by hand again. In **Chapter 04** you'll build a small engine that records every
    step as it's computed and walks backwards by itself, so the computer keeps the books.

    ## What you can do now

    - break a calculation into a computation graph of small steps
    - find each step's local slope with the chapter 02 rules
    - chain local slopes by multiplying, and add blame where a value is used twice
    - run backpropagation by hand and in code, and check a gradient against nudging
    - fit a model with exact gradients

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** in textbooks, the blame of a value is its *partial derivative*,
    written ∂L/∂w ("the slope of L with respect to w, with everything else held still").
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("03", LOCALE)
    return


if __name__ == "__main__":
    app.run()
