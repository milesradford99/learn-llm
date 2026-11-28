import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="24 · Scaling laws")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.plot import chart
    from learn_llm.scaling import knobs_of, small_run, story_tokens

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 24 · Scaling laws

    Suppose you're given a budget for one big training run: so many GPUs for so many weeks. Two questions come first. How big should the model be?
    And how much text should it read? Guess wrong, and the budget is spent on a model that's worse than it could have been. And you can't find the answer by trial and error:
    each try *is* the whole big run.

    The way out is one of the most useful habits in the field. Run many **small** experiments, cheap enough to repeat. Find the rule their results follow.
    Then use that rule to predict the big run before paying for it. Such a rule is called a **scaling law**, and in this chapter you'll measure one yourself, on your own
    machine, and use it to plan a run.

    ## Power laws

    Nearly every scaling law has the same shape, the **power law**:

    > $$y = a \times x^k$$

    Here $x$ is what you spend (amount of text or number of knobs), $y$ is the result you care about (the test loss), and $a$ and $k$ are two fixed numbers. When $k$ is negative, $y$ falls as $x$ grows:
    a minus sign in the exponent flips it into a fraction ($x^{-1} = \frac{1}{x}$), and a $0.5$ exponent is a square root, so $x^{-0.5} = \frac{1}{\sqrt{x}}$.

    A power law has one defining property. Multiply $x$ by 2, and $y$ is multiplied by $2^k$, **wherever you start**: $a \times (2x)^k = 2^k \times (a \times x^k)$.
    Going from 1 to 2 gives the same *proportional* drop as going from a million to two million. Each doubling costs twice as much as the last one, yet buys the same percentage improvement.

    ## By hand: a law that halves

    Take $y = 64 \times x^{-0.5}$, which is $\frac{64}{\sqrt{x}}$. At $x = 1$ it gives 64. What does it give at 4, at 16 and at 64?
    """)
    return


@app.cell(hide_code=True)
def _():
    law_boxes = hand.boxes(['y at x = 4', 'y at x = 16', 'y at x = 64'])
    return (law_boxes,)


@app.cell(hide_code=True)
def _(law_boxes):
    hand.marked(law_boxes, {'y at x = 4': 32, 'y at x = 16': 16, 'y at x = 64': 8}, {
        'y at x = 4': r"""$\sqrt{4} = 2$, and $64 / 2$""",
        'y at x = 16': r"""$\sqrt{16} = 4$""",
        'y at x = 64': r"""$\sqrt{64} = 8$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every time $x$ is multiplied by 4, $y$ halves. Here's the law on an ordinary chart:
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((0, 68), (0, 70), "x", "y", curve=[(x, 64 / np.sqrt(x)) for x in np.linspace(0.85, 64, 200)], dots=[(1, 64), (4, 32), (16, 16), (64, 8)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It's hard to read. The first three dots are squeezed against the left edge, and by eye you couldn't tell this curve apart from many others that fall and flatten.

    ## Logs make it a line

    Notice the pattern between neighbouring dots: they aren't evenly spaced by adding a fixed step, they're evenly spaced by *multiplying* — each dot is 4 times further right and half as high. So instead of raw values, count the multiplications. That's what a logarithm does: turning multiplication into addition, $\log(a \times b) = \log a + \log b$. Apply that to $x^k$ ($x$ multiplied by itself $k$ times) and you get $\log(x^k) = k \times \log x$ (which holds for any $k$, not just whole numbers). Now take the log of both sides of the power law:

    > $$\log y = \log a + k \times \log x$$

    Read this as the equation of a line. With $\log x$ along the horizontal axis and $\log y$ along the vertical axis, **every power law becomes a straight line whose slope is $k$ and whose intercept is $\log a$.** A chart drawn with log axes is called a **log-log plot**.
    If measured points fall on a straight line there, they follow a power law, and the line's slope $k$ is the single number worth remembering.

    Our logs so far were natural logs ($\ln$), which count multiplications by $e$. By hand it's easier to count multiplications by 2 — the doublings. Our four dots become:

    | $x$ | doublings in $x$ | $y$ | doublings in $y$ |
    |---:|---:|---:|---:|
    | 1 | 0 | 64 | 6 |
    | 4 | 2 | 32 | 5 |
    | 16 | 4 | 16 | 4 |
    | 64 | 6 | 8 | 3 |

    For example, $64 = 2 \times 2 \times 2 \times 2 \times 2 \times 2$: six doublings from 1. Plot the two "doublings" columns against each other:
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-0.5, 6.5), (2, 7), "doublings in x", "doublings in y", dots=[(0, 6), (2, 5), (4, 4), (6, 3)], lines=[(-0.5, 6)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A straight line, with evenly spaced dots. It drops 1 for every 2 across, so its slope is $-0.5$ — that's $k$. It starts at 6, and six doublings make 64 — that's $a$.

    Does the choice of log base change the slope? **No.** In $\log y = \log a + k \times \log x$, switching between base-2 log and the natural log $\ln$ scales both sides by the same constant, so the slope $k$ stays identical. That's why we can count doublings by hand and still pair `np.log` with `np.exp` in code.

    ## The best line, in one go

    Real measurements never sit exactly on a line, so we need the line that fits them best. Earlier in the course we measured a line's error with mean squared error (chapter 01) and walked downhill with gradient descent to find the best one (chapter 02).
    For a single straight line there's a one-step shortcut. The bottom of that error valley is where the slopes with respect to both line parameters are zero, and those two equations can be solved once and for all. We'll skip the algebra and use the result directly.
    Write $\bar{x}$ for the mean of the $x$ values and $\bar{y}$ for the mean of the $y$ values. Then:

    > $$\text{slope} = \frac{\text{sum of } (x - \bar{x}) \times (y - \bar{y})}{\text{sum of } (x - \bar{x})^2} \qquad\qquad \text{intercept} = \bar{y} - \text{slope} \times \bar{x}$$

    The top of the fraction checks how $x$ and $y$ move together: when $x$ is above its mean, is $y$ above its mean too (a positive product) or below (a negative one)? The bottom scales that by how spread out the $x$ values are.
    The intercept then pins the line so it passes through the centre point $(\bar{x}, \bar{y})$. This direct formula is called a **least-squares fit**.

    ## By hand: the line through three points

    Use the first three rows of the table: the points $(0, 6)$, $(2, 5)$ and $(4, 4)$. Find the two means, the top and the bottom of the fraction, then the slope and the intercept.
    """)
    return


@app.cell(hide_code=True)
def _():
    line_boxes = hand.boxes(['mean of x', 'mean of y', 'top', 'bottom', 'slope', 'intercept'], step=0.1)
    return (line_boxes,)


@app.cell(hide_code=True)
def _(line_boxes):
    hand.marked(line_boxes, {'mean of x': 2, 'mean of y': 5, 'top': -4, 'bottom': 8, 'slope': -0.5, 'intercept': 6}, {
        'mean of x': r"""$(0 + 2 + 4) / 3$""",
        'mean of y': r"""$(6 + 5 + 4) / 3$""",
        'top': r"""the distances from the means are $-2, 0, 2$ for $x$ and $1, 0, -1$ for $y$: multiply them in pairs and add""",
        'bottom': r"""square $-2$, $0$ and $2$, and add""",
        'slope': r"""top divided by bottom""",
        'intercept': r"""mean of $y$, minus the slope times the mean of $x$; mind the two minus signs""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same $-0.5$ and 6 you read off the chart.

    ## Exercise 1: fit_line

    Write `fit_line(xs, ys)`. `xs` and `ys` are NumPy arrays of the same length, one entry per point. It gives back the pair (slope, intercept) of the least-squares line, from the two formulas above.

    For `xs = [0, 1, 2]` and `ys = [1, 3, 5]` it must give `(2, 1)`: those points are on the line $y = 2x + 1$.
    """)
    return


@app.function
def fit_line(xs, ys):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "fit_line", fit_line, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Arrays do the work without a loop: `xs - np.mean(xs)` is every distance from the mean at once, `*` multiplies two arrays position by position, and `np.sum` adds up the result.""",
        r"""It's 3 or 4 lines. Keep the distances `xs - np.mean(xs)` in a name, since the top and the bottom both use them. Work out the slope first: the intercept needs it.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: logs of a whole array
    `np.log` takes the natural log of every number in an array, and `np.exp` undoes it: `np.exp(np.log(np.array([2.0, 5.0])))` is `[2, 5]` again.
    ///

    ## Exercise 2: fit_power_law

    Write `fit_power_law(xs, ys)`. `xs` and `ys` are NumPy arrays of positive numbers. It gives back the pair $(a, k)$ of the power law $y = a \times x^k$ that fits them best on a log-log plot.
    The recipe is the section "Logs make it a line", with your `fit_line` doing the fitting.

    For our dots, `xs = [1, 4, 16]` and `ys = [64, 32, 16]`, it must give `(64, -0.5)`.
    """)
    return


@app.function
def fit_power_law(xs, ys):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "fit_power_law", fit_power_law, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Fit a line to the logs of both arrays. Which of the line's two numbers is $k$? And the other one is $\log a$, not $a$.""",
        r"""It's 2 lines. `fit_line` gives back a pair, so name both halves on the left of the `=`. The natural log is fine here: the slope is the same whichever log you count with, as long as `np.exp` undoes the same one.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Your own small runs

    Time to measure. The function `small_run(d, tokens, …)` runs one complete training job and returns one number:

    - it builds a 2-block transformer with `d` numbers per token (the same architecture from chapter 20);
    - it trains it with AdamW, warmup and cosine decay (chapter 21) on `tokens` tokens of the stories, reading each token once;
    - it returns the mean surprise per token on a fixed slice of test stories — that's the loss, and $e^{\text{loss}}$ is the perplexity.

    Its code lives in `learn_llm/scaling.py`, and nothing in it is new. We'll test four widths, from 16 to 128, and four amounts of text, from 50,000 to 400,000 tokens: 16 runs in total. The cell below loads the tokens and counts each width's knobs.
    """)
    return


@app.cell
def _():
    train_tokens, test_tokens, vocab_size = story_tokens()
    widths = [16, 32, 64, 128]
    token_counts = np.array([50000, 100000, 200000, 400000])
    sizes = np.array([knobs_of(d, vocab_size) for d in widths])
    sizes
    return sizes, test_tokens, token_counts, train_tokens, vocab_size, widths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    From about 78,000 knobs to nearly a million. The biggest run reads eight times more text than the smallest with a model twelve times bigger, so it costs about a hundred times more compute.
    The button runs all 16 experiments. It takes 1 to 3 minutes, and until it finishes, the cells below that depend on it say `Ancestor stopped`. That's normal.
    """)
    return


@app.cell(hide_code=True)
def _():
    sweep_button = mo.ui.run_button(label="Do the 16 small runs (1 to 3 minutes)")
    sweep_button
    return (sweep_button,)


@app.cell
def _(sweep_button, test_tokens, token_counts, train_tokens, vocab_size, widths):
    mo.stop(not sweep_button.value, mo.md("*Press the button to do the runs.*"))
    loss_table = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            loss_table[i, j] = small_run(widths[i], token_counts[j], train_tokens, test_tokens, vocab_size)
    loss_table.round(3)
    return (loss_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each row is one model size, smallest first. Each column is one amount of text, least first. Read across a row: more text, lower loss. Read down a column: more knobs, lower loss.
    Nothing surprising yet. The real question is *how fast* it falls.

    Take the third row — the model with 64 numbers per token — and fit a power law to its four losses using your function:
    """)
    return


@app.cell
def _(loss_table, token_counts):
    a_text, k_text = fit_power_law(token_counts, loss_table[2])
    np.round([a_text, k_text], 3)
    return a_text, k_text


@app.cell(hide_code=True)
def _(a_text, k_text, loss_table, token_counts):
    chart((10.5, 13.2), (1.2, 1.7), "log of tokens", "log of loss", dots=list(zip(np.log(token_counts), np.log(loss_table[2]))), lines=[(k_text, np.log(a_text))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To read the horizontal axis without computing $e^{12}$ in your head: a natural log of `11` is about 60,000 tokens, `12` is about 160,000, and `13` is about 440,000.

    The four dots sit close to the line: over this range, the loss follows a power law in the amount of text. You should see a $k$ near $-0.14$. By the doubling rule, each doubling of the text multiplies the loss by
    $2^{-0.14} \approx 0.91$ — shaving off 9% each time, while each doubling costs twice as much.

    The same holds down a column. Here is the last column (400,000 tokens) against the number of knobs:
    """)
    return


@app.cell
def _(loss_table, sizes):
    np.round(fit_power_law(sizes, loss_table[:, 3]), 3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A $k$ near $-0.1$: doubling the model size takes about 7% off the loss.

    A pure power law can't be the whole truth, though. It drops towards 0 forever, whereas no model can reach a loss of 0 on stories: what happens next in a story is partly the writer's free choice.
    There is an irreducible **floor** under the loss — the surprise that even a perfect model would still face. We'll build it into the law in a moment.

    ## What a budget buys

    Chapter 22 gave a simple rule for training cost: about $6 \times N \times D$ FLOPs (floating-point operations), for a model with $N$ knobs reading $D$ tokens. A **compute budget** is a fixed total number of FLOPs you're allowed to spend, so it fixes the product $N \times D$ and nothing else.
    You choose how to split it: a model twice as big can afford only half as much text.

    ## By hand: spend a budget

    Your budget is $6 \times 10^{12}$ FLOPs. How many tokens can a model with 1,000,000 knobs read? And one with 4,000,000 knobs?
    """)
    return


@app.cell(hide_code=True)
def _():
    budget_boxes = hand.boxes(['tokens for 1,000,000 knobs', 'tokens for 4,000,000 knobs'])
    return (budget_boxes,)


@app.cell(hide_code=True)
def _(budget_boxes):
    hand.marked(budget_boxes, {'tokens for 1,000,000 knobs': 1000000, 'tokens for 4,000,000 knobs': 250000}, {
        'tokens for 1,000,000 knobs': r"""$6 \times 10^{12} = 6 \times 10^6 \times D$, so $D$ is $10^{12} / 10^6$""",
        'tokens for 4,000,000 knobs': r"""four times the knobs, a quarter of the tokens""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: tokens_for

    Write `tokens_for(budget, knobs)`: the number of tokens a model with `knobs` knobs can read for `budget` FLOPs. It needn't be a whole number.

    `tokens_for(6e12, 1e6)` must give a million, as in your hand calculation.
    """)
    return


@app.function
def tokens_for(budget, knobs):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "tokens_for", tokens_for, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Start from $\text{budget} = 6 \times \text{knobs} \times \text{tokens}$ and get the tokens alone on one side.""",
        r"""It's one line: a division. The 6 and the knobs both go under the budget, so they need brackets around them.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One law for both

    To choose the split we need one law that accounts for both the knobs and the text. The form that has worked best for language models adds three parts:

    > $$\text{loss}(N, D) = E + \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}}$$

    - $E$ is the floor: the loss a perfect model would still have after reading endless text.
    - $\frac{a}{N^{\alpha}}$ is the extra loss from the model being **too small**. It's a power law in $N$ (dividing by $N^{\alpha}$ is the same as multiplying by $N^{-\alpha}$), and it fades as the model grows.
    - $\frac{b}{D^{\beta}}$ is the extra loss from reading **too little text**. It fades as the text grows.

    $\alpha$ and $\beta$ are the Greek letters alpha and beta. Five numbers in all — $E$, $a$, $\alpha$, $b$ and $\beta$ — to be measured from runs.

    Now the trade-off is clear. With a fixed budget, a bigger model leaves less compute for text, shrinking the middle term while enlarging the last term. A smaller model does the reverse. Somewhere in between sits a best size.

    ## Exercise 4: joint_law

    Write `joint_law(knobs, tokens, floor, a, alpha, b, beta)` using the formula above. It must work on plain numbers as well as arrays or tensors of knobs and tokens, which happens automatically if you stick to `+`, `*` or `/`, and `**`.

    With 100 knobs, 10,000 tokens, a floor of 2, $a = 10$, $b = 300$ and both powers 0.5, it must give $2 + \frac{10}{10} + \frac{300}{100} = 6$.
    """)
    return


@app.function
def joint_law(knobs, tokens, floor, a, alpha, b, beta):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "joint_law", joint_law, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three parts, added. In Python a power is `**`, and the worked example has $100^{0.5} = 10$ and $10000^{0.5} = 100$ to test each part on its own.""",
        r"""It's one line. Either divide by the power, or multiply by the power with a minus sign: `x ** -k` is $x^{-k}$.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: the best split

    Here's a toy law with easy numbers: $\text{loss} = 1 + \frac{100}{N} + \frac{400}{D}$. The budget is 60,000 FLOPs, so $N \times D = 10{,}000$.
    Work out the loss for four model sizes. First find the $D$ each one can afford.
    """)
    return


@app.cell(hide_code=True)
def _():
    split_boxes = hand.boxes(['loss with N = 10', 'loss with N = 50', 'loss with N = 100', 'loss with N = 200'], step=0.1)
    return (split_boxes,)


@app.cell(hide_code=True)
def _(split_boxes):
    hand.marked(split_boxes, {'loss with N = 10': 11.4, 'loss with N = 50': 5, 'loss with N = 100': 6, 'loss with N = 200': 9.5}, {
        'loss with N = 10': r"""$D = 1000$, so $1 + 10 + 0.4$""",
        'loss with N = 50': r"""$D = 200$, so $1 + 2 + 2$""",
        'loss with N = 100': r"""$D = 100$, so $1 + 1 + 4$""",
        'loss with N = 200': r"""$D = 50$, so $1 + 0.5 + 8$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The smallest model is starved of knobs and the biggest is starved of text. The best of the four is $N = 50$, where the two parts balance. A model chosen this way is called **compute-optimal** for its budget.

    /// admonition | 🧰 Toolbox: where is the smallest?
    `np.argmin` is `np.argmax`'s twin: it returns the index of the smallest number. `np.argmin([7, 3, 9])` is 1.
    ///

    ## Exercise 5: best_size

    Write `best_size(budget, sizes, law)`. `sizes` is a list of knob counts to choose from, and `law` is a function that takes `(knobs, tokens)` and returns a predicted loss (remember from chapter 02 that functions can be passed as arguments just like numbers).
    Each size gets the tokens the budget can pay for, via your `tokens_for`. Return the size with the lowest predicted loss.

    On your hand exercise, with a budget of 60,000, the sizes `[10, 50, 100, 200]` and the toy law, it must return 50.
    """)
    return


@app.function
def best_size(budget, sizes, law):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "best_size", best_size, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Build a list with one predicted loss per size, find the position of the smallest with `np.argmin`, and use that position on `sizes`.""",
        r"""It's 2 lines. The first is a list built with `for n in sizes` inside square brackets, calling `law` on `n` and on the tokens `n` can afford. The second gives back one item of `sizes`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fitting the law to your runs

    The joint law isn't a straight line in any plot, so `fit_line` can't find its five numbers. Fortunately we already know a method that can fit any smooth formula: gradient descent. The function below is the same PyTorch training loop from chapter 10, where `joint_law` is the "model" and its five numbers are the parameters Adam tunes.

    Three small tricks make it work cleanly:

    - **Keeping $a$ and $b$ positive with `torch.exp`**: `a` and `b` must never turn negative, so Adam optimizes `p[1]` and `p[3]` freely and passes `torch.exp(p[1])` and `torch.exp(p[3])` into `joint_law`. Even if Adam pushes `p[1]` to a negative number like $-10$, $e^{-10} \approx 0.000045$ is still strictly positive.
    - **Measuring percentage error in log space**: the loss is the mean squared difference of `torch.log(predicted)` and `torch.log(measured)`. Because $\log(1.01 y) - \log y = \log 1.01 \approx 0.01$, a difference in logs directly reflects a percentage error — so a 1% miss counts the same everywhere in the table.
    - **Filling the $4 \times 4$ grid in one shot**: `[:, None]` turns the 4 sizes into a $4 \times 1$ column and `[None, :]` turns the 4 token counts into a $1 \times 4$ row, so broadcasting evaluates all 16 predictions at once.
    """)
    return


@app.function
def fit_joint(sizes, token_counts, loss_table):
    n = torch.tensor(sizes, dtype=torch.float64)[:, None]
    d = torch.tensor(token_counts, dtype=torch.float64)[None, :]
    measured = torch.tensor(loss_table)
    p = torch.tensor([1.0, 3.0, 0.3, 3.0, 0.3], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([p], lr=0.02)
    for step in range(4000):
        optimizer.zero_grad()
        predicted = joint_law(n, d, p[0], torch.exp(p[1]), p[2], torch.exp(p[3]), p[4])
        torch.mean((torch.log(predicted) - torch.log(measured)) ** 2).backward()
        optimizer.step()
    return p[0].item(), torch.exp(p[1]).item(), p[2].item(), torch.exp(p[3]).item(), p[4].item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: read the fit

    Look at `fit_joint`. How many numbers is Adam adjusting? How many measured losses does it have to fit them to? And how many times does it compute the whole table of predictions?
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_boxes = hand.boxes(['numbers adjusted', 'measured losses', 'tables computed'])
    return (fit_boxes,)


@app.cell(hide_code=True)
def _(fit_boxes):
    hand.marked(fit_boxes, {'numbers adjusted': 5, 'measured losses': 16, 'tables computed': 4000}, {
        'numbers adjusted': r"""count the entries of `p`""",
        'measured losses': r"""the size of `loss_table`""",
        'tables computed': r"""once per step of the loop""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Five numbers from sixteen measurements: enough, but not by a wide margin. Keep that in mind when reading the result, which is $(E, a, \alpha, b, \beta)$, rounded:
    """)
    return


@app.cell
def _(loss_table, sizes, token_counts):
    law_numbers = fit_joint(sizes, token_counts, loss_table)
    np.round(law_numbers, 3)
    return (law_numbers,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You should see powers $\alpha$ and $\beta$ near 0.25 to 0.3, and a floor $E$ somewhere near 0.3. Don't trust the floor. The lowest loss you measured is about 3.5 —
    ten times higher — so these 16 runs tell us almost nothing about where the curve will eventually bottom out. A fit is only trustworthy within the range where it was measured.

    ## Predict, then run

    Now use the law the way it's used in practice: choose a run before paying for it. The budget is $1.86 \times 10^{12}$ FLOPs, slightly less than the largest run in your table. Which of the four sizes should get it?

    (In the cell below, `*law_numbers` unpacks the 5 numbers inside `law_numbers` and passes them in order as the 5 arguments `floor`, `a`, `alpha`, `b`, `beta` to `joint_law`.)
    """)
    return


@app.cell
def _(law_numbers, sizes):
    budget = 1.86e12

    def fitted_law(knobs, tokens):
        return joint_law(knobs, tokens, *law_numbers)

    choice = best_size(budget, sizes, fitted_law)
    choice, np.round([fitted_law(n, tokens_for(budget, n)) for n in sizes], 3)
    return budget, fitted_law


@app.cell(hide_code=True)
def _(budget, fitted_law):
    _ns = np.exp(np.linspace(np.log(60000), np.log(1500000), 120))
    _curve = [(np.log(n), fitted_law(n, tokens_for(budget, n))) for n in _ns]
    _best = min(_curve, key=lambda point: point[1])
    chart((11, 14.3), (3.4, 4.1), "log of knobs", "predicted loss", curve=_curve, marker=_best)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here on the horizontal axis, `11` is about 60,000 knobs, `12` about 160,000, `13` about 440,000, and `14` about 1.2 million.

    The first number printed is the law's choice: the third size, 387,392 knobs (the model with 64 numbers per token). The chart plots the predicted loss across all sizes at this budget: the same valley you found by hand, with a reassuringly flat bottom — being slightly off the exact optimum barely hurts.

    The chosen model will read 800,000 tokens, twice as much text as any run in your table; the size below it would read 1.8 million tokens, more than four times as much. So this is a genuine prediction outside the measured range.
    The button runs the three largest sizes at this budget and prints each prediction alongside its measured loss. (The smallest size is skipped: it could afford 3.9 million tokens, whereas our stories only contain 2 million — running out of data is a real-world bottleneck too.) Expect 1 to 2 minutes.
    """)
    return


@app.cell(hide_code=True)
def _():
    test_button = mo.ui.run_button(label="Run the three candidates (1 to 2 minutes)")
    test_button
    return (test_button,)


@app.cell
def _(budget, fitted_law, sizes, test_button, test_tokens, train_tokens, vocab_size, widths):
    mo.stop(not test_button.value, mo.md("*Press the button to run the candidates.*"))
    outcome = np.zeros((3, 4))
    for row in range(3):
        affordable = tokens_for(budget, sizes[row + 1])
        measured = small_run(widths[row + 1], affordable, train_tokens, test_tokens, vocab_size)
        outcome[row] = [widths[row + 1], affordable, fitted_law(sizes[row + 1], affordable), measured]
    outcome.round(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each row shows (numbers per token, tokens read, predicted loss, measured loss). On our machine the measured losses were about 3.60, 3.52 and 3.64: the middle size wins, just as predicted, and every prediction lands within 0.1 of the measurement.
    Your numbers will differ slightly.

    That is the whole method: sixteen cheap runs told you how to spend a larger budget before you spent it.

    ## What the real laws say, and what they don't

    Frontier labs follow this exact playbook at a million times the scale. Two findings stand out:

    - The exponents are small, just like yours. Ten times more compute buys a modest, steady drop in loss, never a sudden cliff. Progress from scale is real — and expensive.
    - For large models, the compute-optimal ratio turned out to be roughly **20 tokens of text per knob**. Before this was measured, labs trained huge models on too little text; smaller models trained on more text beat them for the same compute.

    Your own optimum is around 2 tokens per knob, not 20. That's not a bug — it's the most important caveat in this chapter: **the constants in a scaling law belong to the exact setup they were measured on.**
    Your models are tiny, most of their knobs live in the token embedding tables, and the stories are simple. Change the architecture, the dataset, or the training recipe, and the constants shift.
    The method carries over; the numbers must always be re-measured.

    And a scaling law is only trustworthy near the range where it was measured. Ours was checked at twice the text. Nobody should trust its prediction at a thousand times the text, where the poorly pinned-down floor $E$ would dominate the answer.

    ## What's next

    More text lowers the loss, but all the text here was clean. On the open internet, most text is duplicated, broken or junk. **Chapter 25** is about turning a raw pile of text into data worth training on.

    ## What you can do now

    - recognise a power law, and say what each doubling buys
    - turn a power law into a straight line with logs, and read $k$ and $a$ off it
    - fit the best line through points in one step without gradient descent
    - measure a scaling law from your own training runs
    - work out what a compute budget can pay for, and choose the compute-optimal model size
    - explain why a law's constants don't carry over to another setup, or far beyond the measured range

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** The first large study is Kaplan et al., *Scaling Laws for Neural Language Models* (2020). The three-part law and the 20 tokens per knob rule come from Hoffmann et al.,
    *Training Compute-Optimal Large Language Models* (2022), known as the *Chinchilla* paper. NumPy's `np.polyfit(xs, ys, 1)` does what your `fit_line` does.
    """)
    return


if __name__ == "__main__":
    app.run()
