import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="14 · Memory: RNNs")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.diagrams import pipeline
    from learn_llm.leaderboard import record, show
    from learn_llm.plot import chart

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 14 · Memory: RNNs

    Chapter 13's counting models remember one or two words, and treat `dog` and `cat` as total
    strangers. You read differently. You go through a sentence word by word, and at each word you
    update what you have in mind. By the end of a paragraph you still know who it's about, without
    having stored any exact word sequence.

    A **recurrent neural network** (RNN) does the same. It keeps a vector of numbers, its **memory**, and
    updates it with each word it reads. This chapter builds one, finds out why it forgets, and fixes that with the
    **LSTM**, the design that led language modelling through most of the 2010s.

    ## A new bend: tanh

    Memory passes through the same step again and again, so it must not grow without limit. ReLU has no ceiling. For
    memory we use a bend that squashes every number into the range −1 to 1. It's called **tanh** (say "tanch"):
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-4, 4), (-1, 1), "x", "tanh(x)",
          curve=[(x / 10, float(np.tanh(x / 10))) for x in range(-40, 41)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    What matters is its shape: close to $x$ near 0, and flattening out towards
    +1 and −1 for large numbers. In PyTorch it's `torch.tanh`. Below, 0.1 stays at about 0.1, while 5 is squashed to almost 1:
    """)
    return


@app.cell
def _():
    torch.tanh(torch.tensor([-5.0, -1.0, 0.0, 0.1, 1.0, 5.0]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## One step of memory

    Say $x$ is the vector of the current word (chapter 12's idea), and $h$ is the memory so far, of size $m$. The new memory combines the two:

    > $$\text{new } h = \tanh(x \cdot W_x + h \cdot W_h + b)$$

    $W_x$ turns a word into memory-sized numbers, $W_h$ carries the old memory forward, and $b$ is the usual
    bias. It's chapter 08's layer, with one twist: part of its input is its own previous output. "Recurrent" just means it comes back round.

    Here's the step used three times, to read `the dog ran`. The memory starts as zeros, and each arrow is the same step with the same knobs,
    fed a different word:
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([('h', 'all zeros'), ('new h', 'has read: the'), ('new h', 'the dog'), ('new h', 'the dog ran')],
             ['step, x = the', 'step, x = dog', 'step, x = ran'],
             'The memory h starts as zeros. One step with the word the gives a new h, the same step with dog gives the next, and the same step with ran gives the last.')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The **same** knobs are used at every word, like chapter 11's kernels: a kernel slides over positions in a picture,
    and this step slides over positions in a text. So the network has the same number of knobs for a text of 3 words or 3,000.

    ## By hand: two steps of memory

    Take the smallest case: a word is one number, the memory is one number, and the knobs are $W_x = 2$, $W_h = 1$ and $b = -1$.
    The memory starts at 0. The first word is $x = 1$, the second is $x = 0$. You'll need these:

    | $z$ | −1 | −0.24 | 0 | 0.24 | 1 | 2 |
    |---|---|---|---|---|---|---|
    | $\tanh(z)$ | −0.76 | −0.24 | 0 | 0.24 | 0.76 | 0.96 |

    For each word, work out the number that goes into tanh, then the new memory.
    """)
    return


@app.cell(hide_code=True)
def _():
    step_boxes = hand.boxes(['into tanh, word 1', 'memory after word 1', 'into tanh, word 2', 'memory after word 2'], step=0.01)
    return (step_boxes,)


@app.cell(hide_code=True)
def _(step_boxes):
    hand.marked(step_boxes, {'into tanh, word 1': 1, 'memory after word 1': 0.76, 'into tanh, word 2': -0.24, 'memory after word 2': -0.24}, {
        'into tanh, word 1': r"""$x \times W_x + h \times W_h + b$, with $x = 1$ and $h = 0$""",
        'memory after word 1': r"""look your first answer up in the table""",
        'into tanh, word 2': r"""the same sum, now with $x = 0$ and $h$ equal to the memory after word 1""",
        'memory after word 2': r"""look your third answer up in the table""",
    }, LOCALE, tolerance=0.011)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The second word was 0, so it brought nothing. Still the memory changed, and its new value depends on the first word:
    with a different first word you'd have a different number here. That's the memory at work.

    ## Exercise 1: rnn_step

    Write `rnn_step(x, h, Wx, Wh, b)`, the formula above for a whole batch of texts at once.

    | input | shape | what it is |
    |---|---|---|
    | `x` | `(batch, d)` | the current word of each text, as a vector of `d` numbers |
    | `h` | `(batch, m)` | each text's memory so far |
    | `Wx` | `(d, m)` | knobs |
    | `Wh` | `(m, m)` | knobs |
    | `b` | `(m,)` | knobs |

    It gives back the new memory, of shape `(batch, m)`. Your first hand step is one of the checks: with every tensor holding one number,
    `x = [[1.0]]`, `h = [[0.0]]`, `Wx = [[2.0]]`, `Wh = [[1.0]]` and `b = [-1.0]` give `[[0.76]]`.
    """)
    return


@app.function
def rnn_step(x, h, Wx, Wh, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "rnn_step", rnn_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""It's chapter 08's layer with two inputs instead of one. The $\cdot$ between a batch and a matrix of knobs is `@`.""",
        r"""One line: a `return` with `torch.tanh` around the whole sum. The bias goes inside the tanh.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reading a whole text

    Now the picture above, in code. The words arrive as `X` of shape `(batch, length, d)`: for each text in the batch, one vector per word.
    We need to take out word number $t$ of every text, and at the end put the memories back together.

    /// admonition | 🧰 Toolbox: taking a dimension apart, and putting one together
    - A `:` alone in an index means "all of this dimension". For `X` of shape `(batch, length, d)`, `X[:, 0]` is the first
      word of every text, of shape `(batch, d)`.
    - `torch.stack(tensors, dim=1)` joins a list of tensors of the same shape along a **new** dimension 1.
      Three tensors of shape `(batch, m)` become one of shape `(batch, 3, m)`.
    ///

    The cell below takes two words out of a tensor of shape `(2, 5, 4)`, then stacks them, and shows the two shapes.
    """)
    return


@app.cell
def _():
    words_demo = torch.rand(2, 5, 4)
    words_demo[:, 0].shape, torch.stack([words_demo[:, 0], words_demo[:, 4]], dim=1).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: run_rnn

    Write `run_rnn(X, Wx, Wh, b)`. `X` has shape `(batch, length, d)` and the knobs are as in Exercise 1. It gives back the memory
    after every word, of shape `(batch, length, m)`:

    1. The memory starts as zeros, of shape `(batch, m)`. Both sizes can be read off the inputs: `X.shape[0]` and `Wh.shape[0]`.
    2. For each position `t`, from the first word to the last, one `rnn_step` turns word `t` and the memory into the new memory. Keep every new memory in a list.
    3. Stack the list along dimension 1.

    Your hand exercise again: one text of two words, `X = [[[1.0], [0.0]]]`, with the same knobs gives `[[[0.76], [-0.24]]]`.
    """)
    return


@app.function
def run_rnn(X, Wx, Wh, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "run_rnn", run_rnn, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Before the loop: `h` from `torch.zeros`, and an empty list. The loop is `for t in range(X.shape[1]):`. Inside it, the result of `rnn_step` is stored back into `h`, so the next word sees it, and `h` is appended to the list.""",
        r"""It's 6 lines. Word `t` of every text is `X[:, t]`. The last line gives back `torch.stack` of the list with `dim=1`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    PyTorch's `nn.RNN` is the same thing. Let's check, chapter 10 style: take a random `nn.RNN`, copy its
    knobs into your function, and compare. (This cell needs exercise 2.) `nn.RNN(4, 6, batch_first=True)` takes 4 numbers per word and keeps a memory of 6;
    `batch_first=True` says the batch comes first in the shape, as in ours. PyTorch stores its weights on their side, as `nn.Linear` does, and
    keeps two biases, which simply add. Calling it gives back a pair, the memories after every word and the last memory, so `[0]` takes the first.
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    rnn = nn.RNN(4, 6, batch_first=True)
    X_demo = torch.rand(2, 5, 4)
    Wx_demo = rnn.weight_ih_l0.T
    Wh_demo = rnn.weight_hh_l0.T
    b_demo = rnn.bias_ih_l0 + rnn.bias_hh_l0
    with torch.no_grad():
        difference = run_rnn(X_demo, Wx_demo, Wh_demo, b_demo) - rnn(X_demo)[0]
    torch.max(torch.abs(difference)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About `1e-07`, which is 0.0000001: the point moves 7 places left, as in chapter 09. That's only rounding, so
    the two agree.

    ## From text to examples

    To train, we need inputs and answers. For a language model, the answer at each position is
    simply the next word. For the text `[5, 2, 9, 7, 1]`, cut into pieces of length 2:

    ```
    X: 5 2      Y: 2 9
       9 7         7 1
    ```

    At every position, `Y` is `X` moved along by one. So one short piece holds many training examples.
    The last id, 1, is only ever an answer: nothing comes after it, so it can't be an input.

    ## By hand: cut a text

    Cut the 8 ids `[4, 8, 3, 6, 1, 9, 2, 5]` into pieces of length 3. How many whole rows fit, given that
    the last answer needs one id more than the inputs use? What's the last id of `X`, and the first and the last id of `Y`?
    """)
    return


@app.cell(hide_code=True)
def _():
    window_boxes = hand.boxes(['rows', 'last id of X', 'first id of Y', 'last id of Y'])
    return (window_boxes,)


@app.cell(hide_code=True)
def _(window_boxes):
    hand.marked(window_boxes, {'rows': 2, 'last id of X': 9, 'first id of Y': 8, 'last id of Y': 2}, {
        'rows': r"""3 rows would use 9 ids as inputs, and there are only 8""",
        'last id of X': r"""`X` holds the first $2 \times 3 = 6$ ids""",
        'first id of Y': r"""the answer to the first input, 4, is the id after it""",
        'last id of Y': r"""the answer to the last input is the id after it""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The last id, 5, isn't used at all: it doesn't fill a row.

    ## Exercise 3: windows

    Write `windows(ids, length)`. `ids` is a NumPy array of ids and `length` a whole number. It gives back two arrays `X, Y`,
    each of shape `(n, length)`, where `n` is the number of whole rows that fit:

    > `windows(np.array([5, 2, 9, 7, 1]), 2)` gives `[[5, 2], [9, 7]]` and `[[2, 9], [7, 1]]`

    `X` is the first $n \times \text{length}$ ids, and `Y` is the same number of ids, starting one place later. `//` is whole-number division
    (chapter 11) and `reshape` (chapter 06) folds a run of numbers into rows.
    """)
    return


@app.function
def windows(ids, length):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "windows", windows, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""First the number of rows. One id is set aside for the last answer, so it's `len(ids) - 1` ids, divided by `length` with `//`. Then two slices of `ids`, each reshaped to `(n, length)`.""",
        r"""It's 4 lines. The slice for `X` stops at `n * length`. The slice for `Y` starts at 1 and stops one later than `X`'s does.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Why plain RNNs forget

    A memory is only useful if the network can *learn* what to keep. To learn that the first word
    mattered, blame from the end of the text must travel all the way back to the first word.
    That's chapter 03's chain rule again, and chapter 11's problem again: blame gets multiplied at each
    step, and a long chain of multiplications either fades or explodes.

    Let's measure it. We run a fresh, untrained `nn.RNN` over random word vectors, take the
    last memory, and go backward. Then we look at how much blame reached the very first
    word's vector, for texts of different lengths. `out[:, -1]` is the last memory. Any single number made from it
    would do as something to blame; we square and add. (`X.grad[:, 0]` is the blame on word 0. Setting
    `requires_grad=True` on the input keeps the books on it too.)
    """)
    return


@app.function
def first_word_blame(layer, length):
    torch.manual_seed(0)
    X = torch.rand(4, length, 16, requires_grad=True)
    out = layer(X)[0]
    torch.sum(out[:, -1] ** 2).backward()
    return torch.mean(torch.abs(X.grad[:, 0])).item()


@app.cell
def _():
    torch.manual_seed(0)
    plain = nn.RNN(16, 64, batch_first=True)
    plain_blame = []
    for length_try in [2, 10, 30, 60]:
        plain_blame.append((length_try, first_word_blame(plain, length_try)))
    plain_blame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each pair is a length and the blame on its first word: about 0.1 after 2 words, 0.0007 after 10,
    0.00000002 after 30, and about `4e-15` after 60, which means the point moves 15 places left. To the last memory of a 60-word text, the first word might as well
    not exist. So a plain RNN can't learn to carry something across a long text.

    ## The LSTM: a memory with gates

    The trouble is that the plain step *replaces* the memory every time, through a tanh. The **LSTM** ("long short-term memory")
    keeps a second memory vector $c$, the **cell**, and changes it only by **adding**. That's chapter 11's $x + f(x)$:
    addition keeps a clear road for blame. (The LSTM is the older idea: it dates from 1997.)

    To control the cell it uses **gates**. A gate is a vector of numbers between 0 and 1, made with chapter 12's sigmoid,
    multiplied position by position with something else: 0 blocks, 1 lets through. There are three gates, and one more
    value, the candidate new content $n$. All four look at the same thing, the current word and the previous $h$ side by side,
    which we'll call "both". Each has its own layer of knobs:

    | value | made by | question it answers |
    |---|---|---|
    | forget $f$ | $\sigma(L_0(\text{both}))$ | how much of each cell number should be kept? |
    | input $i$ | $\sigma(L_1(\text{both}))$ | how much of the candidate should be written? |
    | output $o$ | $\sigma(L_2(\text{both}))$ | how much of the cell should show in $h$? |
    | candidate $n$ | $\tanh(L_3(\text{both}))$ | what could be written? |

    Then one step changes the cell and the memory, with $\times$ position by position:

    > $$\text{new } c = f \times c + i \times n$$

    > $$\text{new } h = o \times \tanh(\text{new } c)$$

    The first line is the key. With $f$ near 1 and $i$ near 0, the cell passes through untouched, and so could the blame.
    The word "could" matters, as we'll see.

    ## By hand: what the gates do

    One cell number, $c = 0.8$, and a candidate $n = -0.5$. Work out the new $c$ for three settings of the gates:
    $f = 1, i = 0$; then $f = 0, i = 1$; then $f = 0.5, i = 0.5$. Last, whatever the cell holds, what is $h$ when the output gate is $o = 0$?
    """)
    return


@app.cell(hide_code=True)
def _():
    gate_boxes = hand.boxes(['c with f = 1, i = 0', 'c with f = 0, i = 1', 'c with f = 0.5, i = 0.5', 'h with o = 0'], step=0.01)
    return (gate_boxes,)


@app.cell(hide_code=True)
def _(gate_boxes):
    hand.marked(gate_boxes, {'c with f = 1, i = 0': 0.8, 'c with f = 0, i = 1': -0.5, 'c with f = 0.5, i = 0.5': 0.15, 'h with o = 0': 0}, {
        'c with f = 1, i = 0': r"""$1 \times 0.8 + 0 \times (-0.5)$: keep everything, write nothing""",
        'c with f = 0, i = 1': r"""$0 \times 0.8 + 1 \times (-0.5)$: forget everything, write the candidate""",
        'c with f = 0.5, i = 0.5': r"""half of 0.8, plus half of $-0.5$""",
        'h with o = 0': r"""$o$ multiplies everything in the second formula""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So the gates can keep a cell number, replace it, or blend the two, and they can hide the cell from $h$ without erasing it.
    Each position of the cell has its own gate numbers, so one position can be kept while another is replaced.

    /// admonition | 🧰 Toolbox: side by side, and layers in a list
    - `torch.cat([a, b], dim=1)` joins tensors along a dimension they **already have**. Tables of shape `(2, 3)` and `(2, 4)`
      go side by side into one of shape `(2, 7)`. (`torch.stack` makes a new dimension; `torch.cat` makes an existing one longer.)
    - A layer is a value like any other, so layers can sit in a list. If `layers` is a list of `nn.Linear` layers, `layers[0](a)`
      runs the first one on `a`.
    - `torch.sigmoid(a)` is $\sigma$ for every number of a tensor.
    ///
    """)
    return


@app.cell
def _():
    side_demo = torch.cat([torch.zeros(2, 3), torch.ones(2, 4)], dim=1)
    layers_demo = [nn.Linear(7, 5), nn.Linear(7, 1)]
    side_demo.shape, layers_demo[0](side_demo).shape, layers_demo[1](side_demo).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: lstm_step

    Write `lstm_step(x, h, c, L)`. `x` has shape `(batch, d)`, and `h` and `c` have shape `(batch, m)`. `L` is a list of four `nn.Linear` layers in the table's
    order, $L_0$ to $L_3$, each taking `(batch, d + m)` and giving `(batch, m)`. It gives back the pair: new $h$, new $c$.

    "Both" is `x` and `h` side by side, `x` first. Then follow the table and the two formulas. The order matters: the new $h$ is made from the *new* $c$.

    One of the checks compares your step with PyTorch's own LSTM, using the same knobs.
    """)
    return


@app.function
def lstm_step(x, h, c, L):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "lstm_step", lstm_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Eight lines: one for "both", one for each row of the table, one for each formula, and the `return`. The three gates use `torch.sigmoid`, and the candidate uses `torch.tanh`. `*` multiplies position by position.""",
        r"""The first two lines are `both = torch.cat([x, h], dim=1)` and `f = torch.sigmoid(L[0](both))`. Store the new cell back into `c` before you compute `h`, and give back `h, c` in that order.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now the same measurement on a fresh `nn.LSTM`, as before: length, then blame on the first word.
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    lstm_demo = nn.LSTM(16, 64, batch_first=True)
    lstm_blame = []
    for length_lstm in [2, 10, 30, 60]:
        lstm_blame.append((length_lstm, first_word_blame(lstm_demo, length_lstm)))
    lstm_blame
    return (lstm_demo,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Just as bad: about `8e-14` at 60 words. That's worth a moment's thought. A fresh LSTM has random knobs, so
    its forget gates sit near 0.5: at every word, the cell is cut roughly in half, and half of a half of a half soon fades to nothing.

    The difference is that an LSTM has a dial for this, and a plain RNN doesn't. A big bias in the forget gate makes $f$ close to 1. Let's set it by hand.
    PyTorch stacks the four gates' biases in one vector. With 64 numbers per gate, the forget gate's are positions 64 to 127.
    A bias of 3 gives $\sigma(3)$, about 0.95. (Storing into a knob must happen under `torch.no_grad()`, so it isn't written in the book.)
    """)
    return


@app.cell
def _(lstm_demo):
    with torch.no_grad():
        lstm_demo.bias_ih_l0[64:128] = 3.0
    open_blame = []
    for length_open in [2, 10, 30, 60]:
        open_blame.append((length_open, first_word_blame(lstm_demo, length_open)))
    open_blame
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.04 at 60 words: the blame no longer fades, and it's hundreds of billions of times stronger than before. A plain RNN has no simple dial like this, because its
    memory is rebuilt through a tanh at every word. In an LSTM, training can open the gate for whatever is worth keeping, and leave it closed for the rest.
    Gates learn, like any other knobs.

    ## Training a language model

    Here's the plan for the whole model. Word ids go in. A table of vectors turns them into
    numbers (chapter 12's `nn.Embedding`), an `nn.LSTM` reads them, and an `nn.Linear` gives a score for every
    word in the vocabulary, at every position:

    | step | result | shape |
    |---|---|---|
    | the ids | | `(batch, length)` |
    | `table` looks up each id | one vector per word | `(batch, length, d)` |
    | `lstm` reads them, starting from `memory` | the memory after every word, and the final memory | `(batch, length, m)`, and the memory |
    | `out` scores every word of the vocabulary | scores at every position | `(batch, length, vocab)` |

    Calling an `nn.LSTM` takes two inputs, the vectors and the memory to start from, and gives back a pair: the memories after every word,
    and the final memory (itself a pair, $h$ and $c$, which we just pass along). If the starting memory is `None`, Python's word for "nothing", it starts from zeros.

    ## Exercise 5: reader_forward

    Write `reader_forward(ids, memory, table, lstm, out)`, following the table. `table`, `lstm` and `out` are the three layers, and `memory` is
    either `None` or the final memory from an earlier call. It gives back a pair: the scores, and the final memory.
    """)
    return


@app.function
def reader_forward(ids, memory, table, lstm, out):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "reader_forward", reader_forward, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three calls, each on the result of the one before: `table`, `lstm`, `out`. The middle one gives back a pair, so name both halves, as in `a, b = ...`.""",
        r"""It's 2 lines. The first is `states, memory = lstm(table(ids), memory)`. The second gives back the scores and the memory.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the model as a PyTorch module. It holds the three layers, and its `forward` is your function.
    `memory=None` in the first line of `forward` is a **default**: a caller may leave that input out, and it's then `None`. `write`, further down, hands in the memory from the call before.
    The data is chapter 13's: same vocabulary, same ids. The cell that loads it takes a few seconds.
    """)
    return


@app.cell
def _():
    class Reader(nn.Module):
        def __init__(self, vocab_size, d, m):
            super().__init__()
            self.table = nn.Embedding(vocab_size, d)
            self.lstm = nn.LSTM(d, m, batch_first=True)
            self.out = nn.Linear(m, vocab_size)

        def forward(self, ids, memory=None):
            return reader_forward(ids, memory, self.table, self.lstm, self.out)
    return (Reader,)


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = {}
    for word in to_words(" ".join(train_stories)):
        counts[word] = counts.get(word, 0) + 1
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000] + ["<unk>", "<end>"]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    train_ids = story_ids(train_stories, index)
    test_ids = story_ids(test_stories, index)
    len(vocab), len(train_ids), len(test_ids)
    return index, test_ids, train_ids, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `train_reader` is chapter 10's loop with two additions, flattening and clipping, and `score_reader` measures on the test stories with chapter 13's `perplexity`.

    /// admonition | 🧰 Toolbox: three small tools
    - In `reshape`, one size may be written `-1`, meaning "whatever number fits": a tensor of 24 numbers reshaped
      to `(-1, 4)` gets 6 rows, and reshaped to `(-1)` becomes one long vector.
    - `F.cross_entropy(S, answers, reduction="none")` gives each example's surprise, instead of their mean.
    - `np.concatenate(arrays)` joins a list of NumPy arrays end to end into one.
    ///

    **Flattening.** `F.cross_entropy` wants a table of scores and a vector of answers, but we have one
    table per text. `S.reshape(-1, size)` lays them end to end, one row per word.

    **Clipping.** Blame in a long chain can also explode, and one huge slope would throw the knobs far away. `nn.utils.clip_grad_norm_(knobs, 1.0)`
    measures the length of all the slopes together, as one long vector, and if it's over 1, scales them all down
    so that it's 1. The direction is kept, only the size is limited. This is **gradient clipping**.

    In `score_reader`, the minus sign turns surprises into the log chances that `perplexity` expects.
    `S, _memory = ...` keeps the scores and ignores the memory.
    """)
    return


@app.function
def perplexity(log_probs):
    return np.exp(-np.mean(log_probs))


@app.function
def train_reader(model, X, Y, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    size = model.out.out_features
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            S, _memory = model(X[batch])
            loss = F.cross_entropy(S.reshape(-1, size), Y[batch].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return loss.item()


@app.function
def score_reader(model, X, Y):
    size = model.out.out_features
    log_probs = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            S, _memory = model(X[start:start + 64])
            surprise = F.cross_entropy(S.reshape(-1, size), Y[start:start + 64].reshape(-1), reduction="none")
            log_probs.append(-surprise.numpy())
    return perplexity(np.concatenate(log_probs))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: the shape that reaches the loss

    Read the line of `train_reader` that computes `loss`. We'll train with batches of 64 texts, each 32 words long, and the vocabulary has 2,002 entries,
    so `S` has shape `(64, 32, 2002)`. After `S.reshape(-1, size)`, how many rows and columns does the table of scores have? And how many answers are in `Y[batch].reshape(-1)`?
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['rows of scores', 'columns of scores', 'answers'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'rows of scores': 2048, 'columns of scores': 2002, 'answers': 2048}, {
        'rows of scores': r"""one row per word: 64 texts of 32 words""",
        'columns of scores': r"""`size` is the number of scores per word, one for each vocabulary entry""",
        'answers': r"""one answer for each row of scores""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So every step of training is 2,048 next-word questions at once, each scored like one digit in chapter 10.

    The button trains on all 1.9 million words, twice, with a memory of 128 numbers. This is the first
    training in this course that takes real time: about 2 to 5 minutes on a laptop. Until it
    finishes, the cells that need its result say `Ancestor stopped`. That's normal. If you re-run the training cell, press the button again.
    The cell then shows the last training loss, and the test perplexity.
    We train on pieces of 32 words, and score on longer pieces of 64.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the LSTM (2 to 5 minutes)")
    train_button
    return (train_button,)


@app.cell
def _(Reader, test_ids, train_button, train_ids, vocab):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    X_np, Y_np = windows(train_ids, 32)
    X_test_np, Y_test_np = windows(test_ids, 64)
    torch.manual_seed(0)
    model = Reader(len(vocab), 64, 128)
    last_loss = train_reader(model, torch.tensor(X_np), torch.tensor(Y_np), 2, 64, 0.003)
    lstm_score = score_reader(model, torch.tensor(X_test_np), torch.tensor(Y_test_np))
    last_loss, lstm_score
    return lstm_score, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The second number, the test perplexity, should come out at about 31, well below the trigram's 38.
    (The first is the mean surprise on the last training batch, not on the test stories.) Here's the leaderboard, with your LSTM on it:
    """)
    return


@app.cell(hide_code=True)
def _(lstm_score):
    record("lstm", lstm_score)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Writing with temperature

    To write text, we feed the model its own last word, carry its memory along, and sample as in chapter 13. There's
    one new dial. Divide the scores by a number called the **temperature** before softmax:

    - temperature 1 gives the model's real chances;
    - below 1, the gaps between scores grow, and the likely words get even likelier: safer, duller text;
    - above 1, the gaps shrink, and the chances flatten out: wilder text.

    ## By hand: turn the dial

    Two words have the scores $[1, 0]$. Chapter 12 showed that with two scores, softmax gives the first one the share $\sigma$ of the
    gap between them, and here are three values: $\sigma(0.5) = 0.62$, $\sigma(1) = 0.73$, $\sigma(2) = 0.88$.
    What's the first word's share at temperature 1? At temperature 0.5? At temperature 2?
    """)
    return


@app.cell(hide_code=True)
def _():
    heat_boxes = hand.boxes(['share at temperature 1', 'share at temperature 0.5', 'share at temperature 2'], step=0.01)
    return (heat_boxes,)


@app.cell(hide_code=True)
def _(heat_boxes):
    hand.marked(heat_boxes, {'share at temperature 1': 0.73, 'share at temperature 0.5': 0.88, 'share at temperature 2': 0.62}, {
        'share at temperature 1': r"""the scores stay $[1, 0]$, so the gap is 1""",
        'share at temperature 0.5': r"""dividing by 0.5 doubles the scores: $[2, 0]$""",
        'share at temperature 2': r"""dividing by 2 halves the scores: $[0.5, 0]$""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 6: sample_word

    Write `sample_word(scores, temperature)`. `scores` is a tensor with one score per word, of shape `(vocab,)`, and `temperature` a number above 0.
    It gives back one id as a plain Python number, drawn with the chances that softmax gives to the scores divided by the temperature.

    /// admonition | 🧰 Toolbox: softmax, ready-made
    `torch.softmax(a, dim=0)` is chapter 07's softmax for a tensor: it turns a vector of scores into chances that add up to 1.
    `dim=0` says along which dimension, and a vector has only that one. `torch.softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0)`
    is about `[0.67, 0.24, 0.09]`.
    ///

    You know the other tools: `torch.multinomial(p, 1)` draws 1 id with the chances in `p` (chapter 12),
    and `.item()` makes a tensor of one number a plain number (chapter 10). With the scores `[1.0, 0.0]` and temperature 0.1, nearly every call should give 0.
    """)
    return


@app.function
def sample_word(scores, temperature):
    # ← replace todo() below with your code
    todo()


@app.function
def write(model, start, count, temperature, index, vocab):
    words = [index[start]]
    ids = torch.tensor([[index[start]]])
    memory = None
    with torch.no_grad():
        for step in range(count):
            S, memory = model(ids, memory)
            words.append(sample_word(S[0, -1], temperature))
            ids = torch.tensor([[words[-1]]])
    return " ".join([vocab[i] for i in words])


@app.cell(hide_code=True)
def _():
    check("14", "sample_word", sample_word, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two steps: the chances, then the draw. Divide by the temperature before softmax, not after.""",
        r"""It's 2 lines. The first is `p = torch.softmax(scores / temperature, dim=0)`. The second draws 1 id from `p` and ends with `.item()`.""",
    ], LOCALE)
    return


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    write(model, "once", 40, 0.8, index, vocab)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Compare that with chapter 13's bigram text. Stretches of it now read like sentences, and the people in the story stay around
    for a while. Change `0.8` to `0.3` or `1.5` in the cell above and run it again to see the difference.

    Notice what `write` does with `memory`: everything the model knows about the story so far has to fit in
    those 128 numbers, however long the story gets.

    ## What's next

    That squeeze is the LSTM's limit. Asking one fixed vector to hold a whole text is like reading a book,
    then trying to translate it from memory. In **Chapter 15** you'll let a model look back at any word it wants, when it needs it.
    This idea, **attention**, is the heart of every modern language model.

    ## What you can do now

    - explain how a recurrent step reuses the same knobs at every word
    - work out an RNN step by hand, write it and run it over a text, and check it against `nn.RNN`
    - turn text into inputs and next-word answers
    - measure how blame fades through time
    - explain how an LSTM's cell and gates protect memory and blame, and write its step from the formulas
    - use clipping to limit exploding slopes
    - train a language model, put it on the leaderboard, and generate text with a temperature

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the LSTM is from Hochreiter and Schmidhuber (1997). Starting forget gates with a positive bias is common practice. The *GRU* is a
    simpler relative with two gates. Going backward through a long text is called *backpropagation through time*. Our training cuts the text into
    pieces of 32 words, so blame travels at most 32 steps; this is called *truncation*.
    """)
    return


if __name__ == "__main__":
    app.run()
