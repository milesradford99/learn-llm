import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="04 · Keeping the books")

with app.setup(hide_code=True):
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
    # 04 · Keeping the books

    In chapter 03 you worked out the backward steps for Mia's line by hand, then wrote them as
    code in `day_slopes`. That works, but only for that one recipe. Change the recipe (a curve
    instead of a line, a neural network) and you'd have to work out every backward step by hand
    again.

    In this chapter you'll build a small **engine** that does it for you. While a calculation
    runs, the engine writes down every step in a book, like a shopkeeper keeping the books. Then
    it reads the book backwards and hands out the blame by itself, using exactly the rules from
    chapter 03.

    ## What the engine has to do

    Recall the rule from chapter 03. Every box gets a blame (how much the loss moves per unit
    this box moves, for a tiny nudge), and walking backwards:

    > blame of a box = the sum, over every arrow leaving it, of
    > (local slope of that arrow × blame of the box the arrow points to)

    So for every box, the engine must remember three things: its **value**, the boxes it was
    **made from**, and the **local slope** towards each of them. That calls for a new kind of
    Python value.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A new kind of value: classes

    So far you've used numbers, lists and functions. Python lets you invent your own kind of
    value, with named slots for whatever it needs to remember. The recipe for a new kind of value
    is called a **class**. Here is ours:
    """)
    return


@app.class_definition
class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self.parents = []
        self.local_slopes = []


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Line by line:

    - `class Value:` says "here is the recipe for a new kind of value, called `Value`".
    - `def __init__(self, data):` is the setup function Python runs every time you make a new
      `Value`. The name `__init__` (two underscores on each side, short for *initialise*) is fixed
      by Python. It has two inputs: `self` is the new value being set up, and `data` is the
      number you give it.
    - `self.data = data` stores the number in a slot called `data`, inside this value. A slot like
      this is called an **attribute**. The word `data` appears twice here, meaning two different
      things: on the left, `self.data` is the slot inside the value; on the right, `data` is the
      input from the `def` line.
    - `self.grad = 0` starts its blame at 0. ("grad" is short for gradient: the blame from
      chapter 03. It starts at 0 because blame is collected later, by adding.)
    - `self.parents = []` and `self.local_slopes = []` start as empty lists: the boxes this one
      was made from, and the local slope towards each, filled in later.

    Making one looks like calling a function, but with only **one** input: `Value(3)`. You never
    pass `self` yourself. `Value(3)` does three things:

    1. Python makes a new, empty value.
    2. It runs `__init__`, passing that new value in as `self` and `3` in as `data`. So
       `self.data = data` fills the new value's `data` slot with 3.
    3. It gives the filled-in value back, and `v = Value(3)` stores it under the name `v`.

    A dot reads a slot: `v.data` means "the `data` slot of `v`". Each value has its own slots, so
    below, `v` and `u` hold different numbers. (As in chapter 03, several values separated by
    commas on the last line show up as a numbered list of items, counting from 0.)
    """)
    return


@app.cell
def _():
    v = Value(3)
    u = Value(10)
    v.data, u.data, v.grad, v.parents
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can also store into a slot with the dot: `v.grad = 5` would replace its blame with 5.

    ## The book

    The book is an ordinary list. Every time the engine does one step, it adds the result to the
    end of the book. Lists have a built-in way to do that: `book.append(x)` adds `x` to the end
    of `book`. (The dot again: `append` is a function that belongs to every list.)
    """)
    return


@app.cell
def _():
    shelf = []
    shelf.append(5)
    shelf.append(8)
    shelf
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One step: multiply

    Here is how the engine multiplies two values. It works out the result like normal maths,
    then fills in the result's memory and writes it in the book:
    """)
    return


@app.function
def mul(a, b, book):
    out = Value(a.data * b.data)
    out.parents = [a, b]
    out.local_slopes = [b.data, a.data]
    book.append(out)
    return out


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - `out = Value(a.data * b.data)`: a new box holding the product of the two values.
    - `out.parents = [a, b]`: it was made from `a` and `b`.
    - `out.local_slopes = [b.data, a.data]`: the local slopes from chapter 03. In `a × b`, nudging
      `a` moves the result by `b` per unit, and nudging `b` moves it by `a` per unit. The order
      matches `parents`: the first slope goes with the first parent.
    - `book.append(out)`: write the step in the book.
    - `return out`: hand back the new box, so it can be used in the next step.

    Try it:
    """)
    return


@app.cell
def _():
    shop_book = []
    x = Value(3)
    y = Value(4)
    z = mul(x, y, shop_book)
    z.data, z.local_slopes, len(shop_book)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `z` holds 12; its local slopes are 4 (towards `x`) and 3 (towards `y`); the book has one entry.

    ## Exercise 1: add

    Write `add(a, b, book)` the same way. Think first: in `a + b`, if `a` gets a tiny nudge, how
    much does the result move per unit? And for `b`? (Chapter 03's table has the answer.)
    """)
    return


@app.function
def add(a, b, book):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "add", add, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: subtract

    Write `sub(a, b, book)` for `a - b`. Careful with the second local slope: if `b` grows by a
    tiny `h`, the result `a - b` **shrinks** by `h`. So per unit of `b`, the result moves by `-1`.
    """)
    return


@app.function
def sub(a, b, book):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "sub", sub, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: square

    Write `square(a, book)` for `a × a`. It has only **one** parent, `a` (so `out.parents = [a]`),
    and one local slope: chapter 02's slope of `x × x`, which is `2 × x`. Here `x` is `a.data`.
    (On the keyboard, `×` is `*`: Python doesn't know the `×` sign.)
    """)
    return


@app.function
def square(a, book):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "square", square, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reading the book backwards

    Now the heart of the engine. After the calculation, the book lists every step in the order it
    happened, and the last entry is the loss. To hand out blame:

    1. The loss blames itself 1: `loss.grad = 1`.
    2. Go through the book **from the last entry to the first**. For each entry, pass blame to each
       of its parents: add (local slope × this entry's blame) to the parent's blame.

    We **add** to the parent's blame because blames add up when a box feeds several arrows
    (chapter 03): each arrow out of a box brings its own share.

    Why from last to first? A box is always written in the book **after** the boxes it was made
    from: it can't be worked out before them. So when we read backwards, we reach every box that
    *uses* a value before we reach the value itself. By the time we get to a box, all its users
    have already passed their share of blame to it: its blame is complete, and it's safe to pass
    it on.

    Here it is on Mia's day 1 from chapter 03. Recording `p = w × t`, `g = p + b`, `e = g − s`,
    `L = e × e` writes the book in this order: entry 0 is `p`, then `g`, then `e`, then `L`. With
    day 1's numbers, `p = 140`, `g = 80`, `e = 17` and `L = 289`, so the local slope of `e × e`
    is `2 × 17 = 34`. Reading the book backwards:

    | step | entry | its blame is complete because… | it passes on |
    |---|---|---|---|
    | start | `L` | it is the loss: blame 1 | |
    | 1 | `L` | | `e` gets 34 × 1 = 34 |
    | 2 | `e` | its only user, `L`, is done | `g` gets 1 × 34, `s` gets −1 × 34 |
    | 3 | `g` | its only user, `e`, is done | `p` gets 1 × 34, `b` gets 1 × 34 |
    | 4 | `p` | its only user, `g`, is done | `w` gets 14 × 34 = 476, `t` gets 10 × 34 |

    Here is what goes wrong in the other order. Take `x × x + x` at `x = 3`. Its slope is
    `2 × 3 + 1 = 7` (slope of `x × x` plus slope of `x`). The cell below records it, then reads the
    book **front to back** instead. (It uses your `add`, so it shows an error until Exercise 1
    passes.)
    """)
    return


@app.cell
def _():
    wrong_book = []
    x3 = Value(3)
    product = mul(x3, x3, wrong_book)
    total3 = add(product, x3, wrong_book)

    total3.grad = 1
    for entry in wrong_book:  # front to back: the wrong order
        for parent, slope in zip(entry.parents, entry.local_slopes):
            parent.grad = parent.grad + slope * entry.grad
    x3.grad
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It says `1`, not `7`. (Below, `x` is the cell's `x3`, the product is `product` and the total
    is `total3`.) The book holds two entries: first the product (`mul(x3, x3)`, whose parents
    list is `[x3, x3]`: the **same** box twice, once per side of the `×`), then the total.
    Front to back:

    1. The product comes first. Its blame is still 0, because the total hasn't passed it any
       yet. So `x` gets 3 × 0 from the first side and 3 × 0 from the second: nothing.
    2. Then the total (blame 1) passes 1 × 1 to the product and 1 × 1 to `x`. `x` now has 1. The
       product finally has its blame, 1, but its entry has already been read, so that blame is
       never passed on.

    Backwards, the total comes first: the product gets 1 and `x` gets 1. Then the product, now
    holding its complete blame 1, passes 3 × 1 + 3 × 1 = 6 more to `x`. Total: 1 + 6 = 7.

    To go through a list from last to first, Python has `reversed`: `for entry in reversed(book):`.

    ## Exercise 4: backward

    Write `backward(loss, book)`. Your four lines replace `todo()`, so they all sit inside the
    `def`. The first two start at the same place, where `todo()` was. Each of the last two is
    pushed in one more step than the line above it, because it sits inside that line's loop.
    Here is the whole function, so you can see where everything goes:

    ```python
    def backward(loss, book):
        loss.grad = 1
        for entry in reversed(book):
            for parent, slope in zip(entry.parents, entry.local_slopes):
                parent.grad = parent.grad + slope * entry.grad
    ```

    Type only the four lines under `def`: the `def` line is already there in your cell. It
    doesn't need to `return` anything: the blames are stored in the boxes themselves.
    """)
    return


@app.function
def backward(loss, book):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "backward", backward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Mia's first day, through the engine

    Here is chapter 03's day 1 again (`t = 14`, `s = 63`, knobs `w = 10`, `b = -60`), this time
    written with your engine. Nobody tells it the backward steps. (This cell and the picture
    below it show an error until Exercises 1 to 4 all pass.)
    """)
    return


@app.cell
def _():
    day_book = []
    w = Value(10)
    b = Value(-60)
    t = Value(14)
    s = Value(63)
    p = mul(w, t, day_book)
    g = add(p, b, day_book)
    e = sub(g, s, day_book)
    L = square(e, day_book)
    backward(L, day_book)
    w.grad, b.grad
    return L, b, e, g, p, s, t, w


@app.cell(hide_code=True)
def _(L, b, e, g, p, s, t, w):
    one_day({"w": w.data, "t": t.data, "p": p.data, "b": b.data, "g": g.data, "s": s.data, "e": e.data, "L": L.data},
            {n: f"blame {v.grad:g}" for n, v in (("w", w), ("p", p), ("b", b), ("g", g), ("e", e), ("L", L))})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    476 for `w` and 34 for `b`: the same numbers you found by hand in chapter 03, and every box
    in the picture got its blame. The data boxes `t` and `s` were made with `Value` too, so they collect blame as
    well; it's simply never used, because we can't change the data.

    ## Exercise 5: the gradient of the whole loss

    Write `engine_gradient(w_num, b_num, xs, ys)`: the slopes of Mia's **mean** loss for both
    knobs, computed entirely by the engine. It's chapter 03's `gradient`, but you only write the
    forward calculation:

    1. `book = []`, then the knobs as values: `w = Value(w_num)` and `b = Value(b_num)`. (`w_num`
       is a plain number; `w` is the box made from it.) From here on, combine boxes only with
       `add`, `sub`, `mul` and `square`, never with `+`, `-` or `*`: only those functions write
       the step in the book.
    2. A running total that is itself a value: `total = Value(0)`.
    3. Loop with `for x, y in zip(xs, ys):`. Inside: `guess = add(mul(w, Value(x), book), b, book)`,
       then `e = sub(guess, Value(y), book)`, then `total = add(total, square(e, book), book)`.
    4. After the loop (these lines line up with `for`, not inside it), the mean is the total
       times `1 / len(xs)`:
       `mean = mul(total, Value(1 / len(xs)), book)`.
    5. `backward(mean, book)`, then `return w.grad, b.grad`.
    """)
    return


@app.function
def engine_gradient(w_num, b_num, xs, ys):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("04", "engine_gradient", engine_gradient, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    At `w = 10`, `b = -60` it gives `(-94, 2.5)`, exactly chapter 03's hand-built gradient.

    ## Fresh boxes every time

    One trap. Because `backward` **adds** to each blame, running it twice on the same boxes adds
    the blame twice. The cell below records day 1 once, then reads the book backwards twice. You
    might expect double, 952. (Like the cells above, it needs Exercises 1 to 4.)
    """)
    return


@app.cell
def _():
    twice_book = []
    w2 = Value(10)
    loss2 = square(sub(add(mul(w2, Value(14), twice_book), Value(-60), twice_book), Value(63), twice_book), twice_book)
    backward(loss2, twice_book)
    first = w2.grad
    backward(loss2, twice_book)
    first, w2.grad
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The second time, `w`'s blame is 2380, five times 476. On the second pass every box still
    holds its blame from the first pass, and the new shares are added on top of it, box after box
    down the book:

    - `L` is set to 1 again (that line stores, it doesn't add).
    - `e` had 34 and gets 34 × 1 more: 68.
    - `g` had 34 and gets 1 × 68 more: 102.
    - `p` had 34 and gets 1 × 102 more: 136.
    - `w` had 476 and gets 14 × 136 = 1904 more: 2380.

    Each box passes on its whole, already-wrong blame, so the mistake grows at every step. That's
    why `engine_gradient` makes **fresh** `Value` boxes, with a fresh book, every
    time it's called: every blame starts at 0. (Real deep-learning tools have the same rule;
    you'll meet it again later in the course.)

    ## The payoff: a new recipe, no new backward steps

    Mia kept selling through a heatwave. Above about 33 °C, sales started to **drop**: it's too hot
    to go out. Here are all 11 days:
    """)
    return


@app.cell(hide_code=True)
def _():
    hot_temps = temps + [35, 37, 39]
    hot_sales = sales + [280, 262, 236]
    mo.md(f"`temps` = {hot_temps}  \n`sales` = {hot_sales}")
    return hot_sales, hot_temps


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A straight line can't go up and then come down. A **curve** can:

    > guess = a × x × x + w × x + b

    with three knobs `a`, `w` and `b`. With `a` negative, the curve bends downwards, like a hill.

    One practical detail: here `x` is the temperature measured **from 25 °C, in tens of degrees**:
    `x = (t - 25) / 10`, so 14 °C becomes -1.1 and 39 °C becomes 1.4. Why? In chapter 02 you saw
    `b` creep along because the loss reacts more strongly to a knob that gets multiplied by big
    numbers, so one learning rate was safe for `w` but timid for `b`. Here `a` gets multiplied by
    `x × x`:

    | `x` | `x × x` runs from | to |
    |---|---|---|
    | raw temperature `t` | 14 × 14 = 196 | 39 × 39 = 1521 |
    | `t / 10` | 1.4 × 1.4 = 1.96 | 3.9 × 3.9 = 15.21 |
    | `(t - 25) / 10` | 0 | 1.4 × 1.4 = 1.96 |

    With raw temperatures, `a` would be multiplied by numbers up to 1521 and `b` by 1: a learning
    rate small enough for `a` would leave `b` almost frozen. Dividing by 10 helps, but `x × x`
    still reaches 15. Subtracting 25 first puts the middle of the
    temperatures at 0, so `x` runs from about -1 to 1.5 and `x × x` from 0 to 2. Now all three knobs
    are multiplied by numbers of similar size (`x × x`, `x`, and 1), and one learning rate suits
    all of them.

    For a curve, chapter 03 would have meant working out new backward steps by hand. With your
    engine, you only write the forward recipe. Press the button to fit the curve with 500 steps of
    gradient descent, every gradient computed by your engine:
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_curve = mo.ui.run_button(label="Fit the curve")
    fit_curve
    return (fit_curve,)


@app.cell(hide_code=True)
def _(fit_curve, hot_sales, hot_temps):
    mo.stop(not fit_curve.value, mo.md("*Press **Fit the curve** to start.*"))
    try:
        _xs = [(t - 25) / 10 for t in hot_temps]
        _knobs = [0.0, 0.0, 0.0]
        for _i in range(500):
            _book = []
            _a, _w, _b = Value(_knobs[0]), Value(_knobs[1]), Value(_knobs[2])
            _total = Value(0)
            for _x, _y in zip(_xs, hot_sales):
                _xv = Value(_x)
                _guess = add(add(mul(_a, square(_xv, _book), _book), mul(_w, _xv, _book), _book), _b, _book)
                _total = add(_total, square(sub(_guess, Value(_y), _book), _book), _book)
            _mean = mul(_total, Value(1 / len(_xs)), _book)
            backward(_mean, _book)
            _knobs = [k - 0.1 * v.grad for k, v in zip(_knobs, (_a, _w, _b))]
        _ka, _kw, _kb = _knobs
        _curve = [(t, _ka * ((t - 25) / 10) ** 2 + _kw * (t - 25) / 10 + _kb) for t in range(12, 42)]
        _out = mo.vstack([
            chart((10, 42), (0, 320), "temperature (°C)", "ice creams sold",
                  dots=list(zip(hot_temps, hot_sales)), curve=_curve),
            mo.md(f"a = **{_ka:.1f}**, w = **{_kw:.1f}**, b = **{_kb:.1f}**, loss = **{_mean.data:,.1f}**. "
                  f"(The best possible straight line on these 11 days has loss about 810.)"),
        ])
    except Exception:
        _out = mo.md("*Finish Exercises 1 to 4 to fit the curve.*")
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The curve rises with the temperature and then turns down in the heatwave, and its loss is far
    lower than any straight line's. Because `x` is measured from 25 °C in tens of degrees, the
    knobs read differently from the line's `w = 10`, `b = -60`: `b` is the guess at 25 °C
    (where `x = 0`), about 205 ice creams; `w` is how fast sales climb there, about 101 more per
    10 degrees; and the negative `a` bends the curve back down as it gets hotter. You never worked out a backward step for it: the engine kept
    the books and did it.

    ## What's next

    Your engine handles one number at a time: every multiplication is its own box in the book.
    Real models work with thousands or millions of numbers at once, and a book entry per number
    would be far too slow. In **Chapter 05** you'll learn to work with whole lists of numbers in
    one go (vectors and matrices), and then teach the same backward rules to them.

    ## What you can do now

    - define a class with `__init__`, and read and store attributes with a dot
    - add to a list with `append`, and walk it backwards with `reversed`
    - build an engine that records a calculation and backpropagates through it automatically
    - explain why the book must be read backwards, and why each calculation needs fresh boxes
    - fit a new model by writing only its forward recipe

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** this kind of engine is called *automatic differentiation* (reverse
    mode), or *autograd*. The order we get from the book is a *topological order*.
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("04", LOCALE)
    return


if __name__ == "__main__":
    app.run()
