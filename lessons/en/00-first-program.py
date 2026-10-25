import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="00 · Your first program")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.nav import nav

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 00 · Your first program

    Welcome! By the end of this course you will have built your own language model, the
    kind of program behind chatbots, from the ground up. Every piece starts here, with a few
    lines of Python.

    You don't need to know anything yet. When a new idea is needed, this page explains it
    right where you need it.

    ## How this page works

    This page is a **notebook**. It is made of **cells**. Some cells hold text, like this one.
    Others hold Python code, with the code's result shown just below the code.

    - To **run** a code cell, click inside it and press **Shift + Enter**. This also moves you
      to the next cell; click back in to keep editing. (A ▶ run button also appears next to a
      cell when you point at it.)
    - While typing code: after a line that ends with `:`, pressing **Enter** pushes the next
      line in by four spaces for you. **Backspace** moves it back out by four.
    - Your changes are saved automatically, in your own copy of the lesson. You can't break
      the course; experiment freely.
    - Sooner or later you'll see a red **error** box under a cell. That's normal, even for
      experts: it means Python couldn't follow an instruction. Read its last line, fix the
      code, and run the cell again.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Python as a calculator

    The cell below holds a tiny Python program: `2 + 3`. Its result appears under it.
    Change the numbers and run it again.
    """)
    return


@app.cell
def _():
    2 + 3
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Python knows the usual arithmetic. Some symbols differ from what you write on paper:

    | On paper | In Python | Meaning | Result |
    |---|---|---|---|
    | 7 × 6 | `7 * 6` | multiply | `42` |
    | 10 ÷ 4 | `10 / 4` | divide | `2.5` |
    | 2¹⁰ | `2 ** 10` | power: ten 2s multiplied together | `1024` |

    Try a few in the cell below, one at a time: Python shows the result of the **last** line
    only. Python doesn't understand × or ÷, so type `*` and `/`. As on paper, `*` and `/`
    happen before `+` and `-`: `1 + 2 * 3` is `7`. Parentheses go first: `(1 + 2) * 3` is `9`.
    """)
    return


@app.cell
def _():
    7 * 6
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Giving values a name

    Typing the same number again and again gets tiring. You can give a value a **name**,
    then use the name instead. A name that stores a value is called a **variable**.

    `price = 12` means "from now on, `price` stands for 12". The `=` here means *store*,
    not *equals*. The next cell stores 12 in `price` and 3 in `count`. Storing shows no
    result under the cell; that's expected.

    You choose the names. Use letters, digits and `_`, with no spaces: `new_price` works,
    `new price` doesn't.
    """)
    return


@app.cell
def _():
    price = 12
    count = 3
    return count, price


@app.cell
def _(count, price):
    price * count
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the cell that says `price = 12`, change it to `price = 15` and press
    **Shift + Enter**. The result `36` turns into `45` by itself: the cell that uses
    `price` runs again whenever `price` changes.

    /// note | One rule of this notebook
    Each name gets its value (with `=`) in **one** cell only. If you write `price = 20` in a
    second cell, the notebook shows the error *"The variable 'price' was defined by another
    cell"*. Pick a new name, such as
    `new_price`, or change the original cell. (Names used *inside* a function, which you'll
    meet next, don't count.)
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Functions: a recipe with a name

    A **function** is a small recipe. You give it some input, it follows the steps, and it
    gives back an answer. Here is a function that doubles a number:

    ```python
    def double(x):
        return x * 2
    ```

    Reading it line by line:

    - `def double(x):` says "I'm defining a recipe called `double`. Its input will be called `x`."
      The colon `:` at the end means "the recipe's steps come next".
    - The next line is pushed to the right (**indented**) by four spaces. That's how Python
      knows the line belongs to the recipe.
    - `return x * 2` says "the answer is `x * 2`". `return` **gives the answer back** to
      wherever the function was used.

    Using a function is called **calling** it. You write its name, then the input in
    parentheses: `double(5)` runs the recipe with `x` standing for `5`.
    """)
    return


@app.function
def double(x):
    return x * 2


@app.cell
def _():
    double(5)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: Celsius to Fahrenheit

    Your turn. Most of the world measures temperature in degrees Celsius (°C), but the US
    uses Fahrenheit (°F). The rule to convert is:

    > multiply the Celsius temperature by 9, divide by 5, then add 32.

    In the cell below, the first line names the input `celsius` (not `x` this time), so
    use `celsius` wherever the rule says "the Celsius temperature".

    Two things you'll see in the cell:

    - A line starting with `#` is a **comment**: a note for humans. Python ignores it.
    - `todo()` is a placeholder that means "not written yet". Select just the word `todo()` and
      type your `return` line over it, followed by the conversion. Leave the four spaces in
      front of it alone.

    Then press **Shift + Enter**. The checks below the cell tell you how you did, every time
    you run it. If a red error appears and the checks disappear, there's a typing slip in
    your cell: look for a missing `:`, a missing `(` or `)`, or spaces that don't line up
    (that's what `IndentationError` means). Fix it and the checks come back.
    """)
    return


@app.function
def to_fahrenheit(celsius):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("00", "to_fahrenheit", to_fahrenheit, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    temperature = mo.ui.slider(-40, 100, value=20, label="Celsius", show_value=True)
    return (temperature,)


@app.cell(hide_code=True)
def _(temperature):
    try:
        _answer = f"**{temperature.value} °C = {to_fahrenheit(temperature.value):g} °F**"
    except Exception:
        _answer = "*Finish Exercise 1, then drag the slider to try your function.*"
    mo.hstack([temperature, mo.md(_answer)], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Many values at once: lists

    A **list** holds several values in order, between square brackets, separated by commas:
    `[3, 1, 4, 1, 5]`. Pictures, words and whole books are all stored in computers as long
    lists of numbers, so you'll use lists everywhere in this course.

    `len` (short for *length*) is a function that comes with Python. It gives back how many
    values a list holds:
    """)
    return


@app.cell
def _():
    scores = [3, 1, 4, 1, 5]
    len(scores)
    return (scores,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Doing something for each value: loops

    A **loop** repeats some steps once for every value in a list. The next cell counts how
    many scores are bigger than 2. Read it line by line first:

    - `big = 0` stores `0` in `big`, the count so far.
    - `for s in scores:` means "take each value of `scores` in turn, call it `s`, and run
      the indented lines below for it". With 5 scores, those lines run 5 times. (`s` is just
      a name we picked; any name works.)
    - `if s > 2:` means "only when `s` is bigger than 2, run the line indented below me".
      (`>` means *bigger than*, `<` means *smaller than*.)
    - `big = big + 1` means "work out `big + 1`, then store it back in `big`". It looks
      odd as maths, but remember `=` means *store*. So `big` goes up by one.

    Look at the spaces in front of each line: the `if` line is inside the loop (4 spaces), and
    the counting line is inside the `if` (8 spaces). Each time a line ends with `:`, the
    lines that belong to it move four spaces further right.
    """)
    return


@app.cell
def _(scores):
    big = 0
    for s in scores:
        if s > 2:
            big = big + 1
    big
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: add up a list

    Write `total(numbers)`, which gives back the sum of all the numbers in the list.
    The plan is already written in the cell as comments:

    1. Store `0` in a name for the running total, for example `running = 0`.
    2. Loop over `numbers` (for example `for n in numbers:`), and add **each number**
       (not 1) to the running total.
    3. After the loop, `return` the running total.

    Watch the spaces: inside the function, lines start 4 spaces in; the line inside the
    loop starts 8 spaces in. After that line, press **Enter**, then **Backspace** once to go
    back to 4 spaces, and write the `return` there, so it runs once, **after** the loop has
    finished. If your function gives back only the first number, your `return` is still
    inside the loop.
    """)
    return


@app.function
def total(numbers):
    # Plan: start a running total at 0; for each number, add it to the
    # running total; after the loop, return the running total.
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("00", "total", total, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: the average

    /// admonition | 🧰 Toolbox: the average (mean)
    The **average** of some numbers, also called the **mean**, is their total divided by
    how many there are. The average of 2, 4 and 9 is (2 + 4 + 9) ÷ 3 = 15 ÷ 3 = 5.
    ///

    Write `average(numbers)`. Finish Exercise 2 first, because you can reuse it: a function
    can call another function. Inside `average`, `total(numbers)` gives the sum and
    `len(numbers)` gives how many numbers there are. Don't forget the parentheses: `total`
    on its own is the recipe, `total(numbers)` is its answer.
    """)
    return


@app.function
def average(numbers):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("00", "average", average, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What you can do now

    - run Python code and read its result
    - store values in variables
    - write functions that take inputs and `return` an answer
    - hold many values in a list and loop over them

    That is enough to start teaching a computer. In **Chapter 01** you'll use exactly these
    tools to make a program learn a pattern from data, then predict numbers it has
    never seen.

    ---

    <!-- go-deeper -->
    **Go deeper (optional, not needed for the course):** the official
    [Python tutorial](https://docs.python.org/3/tutorial/) covers these ideas in more detail.
    """)
    return


@app.cell(hide_code=True)
def _():
    nav("00", LOCALE)
    return


if __name__ == "__main__":
    app.run()
