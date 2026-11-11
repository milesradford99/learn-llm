import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="15 · Attention")

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
    from learn_llm.diagrams import looking
    from learn_llm.images import maps

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 15 · Attention

    Chapter 14's LSTM squeezes everything it has read into one vector of 128 numbers. For predicting the next word
    in a story, that goes a long way. Now take a harder job: translation. A model must read a whole
    sentence, then write it out in another language. It's natural to build that from two LSTMs: an **encoder** reads
    the sentence into a memory, and a **decoder** starts from that memory and writes.

    The trouble is the single memory: all of a long sentence has to pass through one small vector. Think about how you'd translate
    a long sentence yourself. You wouldn't memorise it and then turn away. You'd keep looking back at the
    words you need, as you need them.

    That looking back is called **attention**, and it's the idea behind the T in ChatGPT, which stands for Transformer (chapters 16 and 17).

    ## A task we can check

    We have no translated text in the course, so we'll use a task with the same shape, where every
    answer can be checked at a glance: **write the text backwards**.

    > in: `the dog ran home` → out: `home ran dog the`

    It's a good test, because the first word to write is the last one read, and the last to write is the very first one read. No
    understanding is needed, only remembering a whole sequence exactly. The cell below prepares 500 common words, plus a new `<start>` word explained below, and a
    function that draws random stretches of text, of a chosen length, with their reversal. `torch.flip` reverses along a dimension,
    `rng.integers` can draw a whole batch of starting points at once, and `np.stack` makes a table from a list of rows.
    (`to_words` is chapter 12's text cutter.)
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def draw_batch(ids, length, size, rng):
    starts = rng.integers(0, len(ids) - length, size)
    X = torch.tensor(np.stack([ids[s:s + length] for s in starts]))
    return X, torch.flip(X, dims=[1])


@app.cell
def _():
    train_stories = stories()[0]
    words = to_words(" ".join(train_stories[:3000]))
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    vocab = ["<start>"] + sorted(counts, key=counts.get, reverse=True)[:500]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    ids = np.array([index[word] for word in words if word in index])
    X_demo, Y_demo = draw_batch(ids, 6, 1, np.random.default_rng(0))
    [vocab[i] for i in X_demo[0].tolist()], [vocab[i] for i in Y_demo[0].tolist()]
    return Y_demo, ids, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## How the decoder learns

    The decoder writes one word at a time, and each word it writes is its input for the next step. In training, we don't feed it
    its own guesses, which would be mostly wrong at first and would throw off every step after them. Instead, at each step we give it the
    **correct** previous word. This is called **teacher forcing**, and it lets all the steps train at once. So the decoder's input is the answer, moved
    one place to the right, with a `<start>` word (id 0) at the front:

    ```
    answer Y:       home ran  dog  the
    decoder input:  <start> home ran dog
    ```

    ## Exercise 1: shift_right

    /// admonition | 🧰 Toolbox: columns of a table
    Take a table `T` with the rows `[1, 2, 3]` and `[4, 5, 6]`.

    - `T[:, :1]` keeps every row, and only the first column: `[[1], [4]]`. It's still a table, with one column.
    - `T[:, :-1]` keeps every row, and drops the last column: `[[1, 2], [4, 5]]`.
    - `torch.zeros_like(T)` is a table of zeros with the same shape and the same kind of number as `T`.
    - Chapter 14's `torch.cat([A, B], dim=1)` joins two tables side by side.
    ///

    Write `shift_right(Y)`, for a table `Y` of shape `(batch, length)`. Every row loses its last word and gets a 0, the id of `<start>`, at the
    front, so the shape stays the same:

    > `[[7, 8, 9], [4, 5, 6]]` becomes `[[0, 7, 8], [0, 4, 5]]`

    Build it from two pieces: a column of zeros, and `Y` without its last column.
    """)
    return


@app.function
def shift_right(Y):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "shift_right", shift_right, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Every tool you need is in the Toolbox box: a slice that keeps one column, `torch.zeros_like`, a slice that drops the last column, and `torch.cat`.""",
        r"""It fits in one line. Inside `torch.cat([..., ...], dim=1)` the column of zeros comes first. To get a column of zeros, take the first column of `Y` and ask for zeros like it.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the model. It's chapter 14's `Reader` in two halves. The encoder gives back its states `K`, one
    vector for each word read, and its final memory. The decoder starts from that final memory and gives its own states `Q`, one for each word to write.

    `use` is a function we hand in, as with `step` in chapter 11's `Block`: it says what to do with `Q` and `K`. It gives back two things, the vectors to score and the shares (more on those soon),
    and `[0]` takes the first. `no_attention` returns `Q` untouched and no shares, ignoring `K` entirely, so
    the final memory is all that passes from encoder to decoder. `size` is the memory size, and `width` is how many numbers go into the last layer.
    """)
    return


@app.function
def no_attention(Q, K):
    return Q, None


@app.cell
def _():
    class Translator(nn.Module):
        def __init__(self, vocab_size, size, use, width):
            super().__init__()
            self.table = nn.Embedding(vocab_size, 64)
            self.encoder = nn.LSTM(64, size, batch_first=True)
            self.decoder = nn.LSTM(64, size, batch_first=True)
            self.use = use
            self.out = nn.Linear(width, vocab_size)

        def read(self, X):
            return self.encoder(self.table(X))

        def forward(self, X, decoder_in):
            K, memory = self.read(X)
            Q, _memory = self.decoder(self.table(decoder_in), memory)
            return self.out(self.use(Q, K)[0])
    return (Translator,)


@app.function
def train_translator(model, ids, steps, rng):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for step in range(steps):
        length = rng.integers(3, 21).item()
        X, Y = draw_batch(ids, length, 64, rng)
        S = model(X, shift_right(Y))
        optimizer.zero_grad()
        F.cross_entropy(S.reshape(-1, S.shape[2]), Y.reshape(-1)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


@app.function
def translate(model, X):
    K, memory = model.read(X)
    word = torch.zeros(len(X), 1, dtype=torch.long)
    written = []
    shares = []
    for t in range(X.shape[1]):
        Q, memory = model.decoder(model.table(word), memory)
        mixed, A = model.use(Q, K)
        word = torch.argmax(model.out(mixed), dim=2)
        written.append(word)
        shares.append(A)
    return torch.cat(written, dim=1), shares


@app.function
def share_correct(model, ids, length, rng):
    X, Y = draw_batch(ids, length, 500, rng)
    with torch.no_grad():
        return torch.mean((translate(model, X)[0] == Y).float()).item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Predict the shapes

    Before training anything, follow one batch through the model on paper. Say `X` holds 64 texts of 6 words each, the memory size is 128, and the
    vocabulary has 501 words. Read `forward` above, line by line, and fill in the sizes. An `nn.LSTM` gives one state for every word it reads.
    """)
    return



@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['K: words', 'K: numbers', 'Q: words', 'out: scores'])
    return (shape_boxes,)



@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'K: words': 6, 'K: numbers': 128, 'Q: words': 6, 'out: scores': 501}, {
        'K: words': r"""`K` has shape `(64, ?, 128)`: the encoder leaves one state for each word it read""",
        'K: numbers': r"""each state is one memory, and `size` says how many numbers that is""",
        'Q: words': r"""the decoder's input is `shift_right(Y)`, which has the same shape as `Y`""",
        'out: scores': r"""`self.out` gives one score for every word in the vocabulary""",
    }, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So the encoder hands over 6 states of 128 numbers, but `no_attention` throws `K` away. Only the final memory, one vector of 128 numbers, reaches the decoder.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `train_translator` trains on stretches of 3 to 20 words. `translate` writes for real, with no teacher: each word comes from
    the decoder's own last guess, taking the likeliest word (`torch.argmax`). `share_correct` scores 500 examples of one length: the share of positions with the right word.

    The button trains for 1,500 batches, half a minute or so, and shows pairs of length and share correct, for lengths 5, 10, 15 and 20: 0.988 would mean 98.8%.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train without attention")
    train_button
    return (train_button,)


@app.cell
def _(Translator, ids, train_button, vocab):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    torch.manual_seed(0)
    rng_plain = np.random.default_rng(0)
    plain_model = Translator(len(vocab), 128, no_attention, 128)
    train_translator(plain_model, ids, 1500, rng_plain)
    plain_scores = []
    for plain_length in [5, 10, 15, 20]:
        plain_scores.append((plain_length, round(share_correct(plain_model, ids, plain_length, rng_plain), 3)))
    plain_scores
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You should see about 99% at length 5, about 78% at 10, about 42% at 15, and about 23% at 20. The longer the
    sentence, the worse it gets, because 128 numbers can't hold it all. That's the bottleneck.

    ## Weighted averages

    The fix starts with a small idea. Say we have three vectors, and we want a mix of them: half of the first, 30% of the second,
    20% of the third. That's a **weighted average**: multiply each vector by its share, and add up.

    | row | vector | share | share $\times$ vector |
    |---|---|---|---|
    | 0 | `[10, 0]` | 0.5 | `[5, 0]` |
    | 1 | `[0, 10]` | 0.3 | `[0, 3]` |
    | 2 | `[10, 10]` | 0.2 | `[2, 2]` |
    | | | **sum** | **`[7, 5]`** |

    The shares add up to 1, so the result stays among the vectors: it's pulled most towards the one with the biggest share.
    With shares `[1, 0, 0]`, you get exactly the first vector. So a weighted average is a kind of soft lookup: it can pick out one row, or blend a few.

    ## By hand: a blend

    Your turn. The rows are `[4, 0]`, `[0, 8]` and `[2, 2]`, and the shares are 0.25, 0.25 and 0.5. Work out the two numbers of the weighted average.
    """)
    return


@app.cell(hide_code=True)
def _():
    blend_boxes = hand.boxes(['first number', 'second number'])
    return (blend_boxes,)



@app.cell(hide_code=True)
def _(blend_boxes):
    hand.marked(blend_boxes, {'first number': 2, 'second number': 3}, {
        'first number': r"""use the first number of each row: $0.25 \times 4 + 0.25 \times 0 + 0.5 \times 2$""",
        'second number': r"""use the second number of each row: $0.25 \times 0 + 0.25 \times 8 + 0.5 \times 2$""",
    }, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: weighted_average

    Write `weighted_average(a, V)`, for shares `a` of shape `(n,)` and a matrix `V` of shape `(n, d)` whose rows are the vectors. It gives back one
    vector of `d` numbers. You don't need a loop: chapter 05 has an operation that multiplies each row of a matrix by one number of a vector and adds up the results.

    On the table above, the shares `[0.5, 0.3, 0.2]` must give `[7, 5]`.
    """)
    return


@app.function
def weighted_average(a, V):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "weighted_average", weighted_average, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Chapter 05: a vector times a matrix, written with `@`, takes each row times its number, and adds them up.""",
        r"""One line, with the shares on the left of the `@`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where to look: query, key, value

    Now the decoder can look back. At each step of writing, it has a state, and the encoder has left one state per word read.
    Attention takes three steps, and you know every piece of them:

    1. **Compare.** Take the dot product of the decoder's state with each encoder state. A big dot product means "this one is relevant"
       (chapter 12: dot products measure how alike vectors are).
    2. **Share out.** Softmax turns those scores into shares that add up to 1 (chapter 07).
    3. **Blend.** Take the weighted average of the encoder states with those shares.

    **One step, with numbers.** Real states have 128 numbers. Here are three with 2 numbers each, for three words read, and a decoder state `q = [2, 1]`:

    | word read | its state | 1. compare: dot product with `q` | 2. share out: softmax |
    |---|---|---|---|
    | the | `[1, 0]` | $2 \times 1 + 1 \times 0 = 2$ | 0.24 |
    | dog | `[0, 1]` | $2 \times 0 + 1 \times 1 = 1$ | 0.09 |
    | ran | `[1, 1]` | $2 \times 1 + 1 \times 1 = 3$ | 0.67 |

    Step 3 blends the three states with those shares:

    > $$0.24 \times [1, 0] + 0.09 \times [0, 1] + 0.67 \times [1, 1] = [0.91, 0.76]$$

    The result is close to the state of `ran`, with a little of the others mixed in. Here's the same step as a picture: the thicker the line, the bigger the share.
    """)
    return


@app.cell(hide_code=True)
def _():
    looking(['the [1, 0]', 'dog [0, 1]', 'ran [1, 1]'], [0.24, 0.09, 0.67], 'q = [2, 1]',
            'The decoder state q looks at three encoder states with shares 0.24, 0.09 and 0.67')
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Notice how sharp softmax is. The score of `ran` is only 1 more than the score of `the`, yet its share is about 2.7 times as big, because
    each extra point of score multiplies the share by $e$ (chapter 07).

    The result is a fresh summary of the input, made for this one step of writing. The terms for the three parts come from looking
    things up. The **query** `q` is what you're looking for, the **keys** `K` are the labels you compare it with, and
    the **values** `V` are what you get back. In a library, the query is your question, the keys are the titles on the spines, and the values are the books.
    Here the keys and the values are the same encoder states, but they needn't be, and in the next chapter they won't be.

    Why would a decoder state and an encoder state have a big dot product? At first, they don't. Training makes the useful ones large, as you'll see below.

    ## By hand: compare

    A new query, `q = [1, 3]`, and three keys: `[2, 0]`, `[0, 1]` and `[1, 1]`. Work out the three scores. Then say which key will get the
    biggest share: 0, 1 or 2.
    """)
    return



@app.cell(hide_code=True)
def _():
    score_boxes = hand.boxes(['score 0', 'score 1', 'score 2', 'biggest share'])
    return (score_boxes,)



@app.cell(hide_code=True)
def _(score_boxes):
    hand.marked(score_boxes, {'score 0': 2, 'score 1': 3, 'score 2': 4, 'biggest share': 2}, {
        'score 0': r"""the dot product of `[1, 3]` and `[2, 0]`: $1 \times 2 + 3 \times 0$""",
        'score 1': r"""the dot product of `[1, 3]` and `[0, 1]`""",
        'score 2': r"""the dot product of `[1, 3]` and `[1, 1]`""",
        'biggest share': r"""softmax keeps the order: the biggest score gets the biggest share. Type the number of that key""",
    }, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: attention_shares

    Write `attention_shares(q, K)`, doing steps 1 and 2 for one query `q` of shape `(d,)` and keys `K` of shape `(n, d)`, one key per row. It gives back `n` shares.

    `K @ q` is chapter 05's matrix times vector: one dot product for each row of `K`. `torch.softmax` needs to be told along which dimension the
    shares should add up to 1, and a vector has only one, `dim=0`.

    On the table above, `q = [2, 1]` must give about `[0.24, 0.09, 0.67]`.
    """)
    return


@app.function
def attention_shares(q, K):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "attention_shares", attention_shares, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two steps in order: the scores, then softmax of the scores.""",
        r"""One line: `torch.softmax(...)` around the dot products, with `dim=0`.""",
    ], LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here are the shares for your hand example, from your own function:
    """)
    return



@app.cell
def _():
    attention_shares(torch.tensor([1.0, 3.0]), torch.tensor([[2.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.09, 0.24 and 0.67. Those are the same three shares as in the table, though the scores were 2, 3, 4 there and 2, 1, 3 here. Softmax only sees the
    gaps between scores, as chapter 07 showed: adding the same number to every score changes nothing.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: attend

    Now everything at once: a batch of texts, with many queries in each. Every word being written has its own query, so the shares form a table:
    one row for each word written, one column for each word read.

    /// admonition | 🧰 Toolbox: a batch of matrices
    A tensor of shape `(5, 2, 3)` can be read as 5 matrices of shape `(2, 3)`, one for each text.

    - `@` works text by text, on the last two dimensions: `(5, 2, 3) @ (5, 3, 4)` gives `(5, 2, 4)`.
    - `M.transpose(1, 2)` swaps dimensions 1 and 2. It's `.T` for each text: `(5, 2, 3)` becomes `(5, 3, 2)`.
    - `torch.softmax(M, dim=2)` makes the numbers add up to 1 along the last dimension, that is along each row of each matrix.
    ///

    In one text, with the queries as the rows of $Q$, the keys as the rows of $K$ and the values as the rows of $V$, the three steps are:

    > $$\text{scores} = Q K^\top \qquad A = \text{softmax}(\text{scores}) \qquad \text{blended} = A V$$

    $K^\top$ is $K$ with rows and columns swapped (chapter 07), so that row $i$ of the scores holds the dot products of query $i$ with every key.

    Write `attend(Q, K, V)`. It gives back two things, the blended vectors and the shares `A`:

    | | shape |
    |---|---|
    | `Q` | `(batch, writes, d)` |
    | `K` | `(batch, reads, d)` |
    | `V` | `(batch, reads, e)`: values may have a size of their own |
    | scores, and `A` | `(batch, writes, reads)` |
    | blended | `(batch, writes, e)` |

    Each written word shares out its attention over the words read, so the shares must add up to 1 along `reads`.
    """)
    return


@app.function
def attend(Q, K, V):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "attend", attend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three lines, one for each formula. Follow the shapes in the table: `(batch, writes, d) @ (batch, d, reads)` gives `(batch, writes, reads)`.""",
        r"""`K` has shape `(batch, reads, d)`, so it needs its last two dimensions swapped before the `@`. The softmax runs along dimension 2. The blend uses `V`, not `K`, and the function gives back a pair.""",
    ], LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: with_attention

    One piece is missing: a `use` function to hand to the `Translator`, in place of `no_attention`. Look at how `forward` and `translate` call `use(Q, K)`, and at what `no_attention` gives back.

    Write `with_attention(Q, K)`:

    1. Run your `attend`. The queries are the decoder states `Q`. The encoder states `K` serve as the keys and as the values.
    2. Join each decoder state with its blended summary, side by side, so the last layer sees both. With a memory of 128 numbers that's 256, which is why the last layer's width doubles.
    3. Give back the joined vectors and the shares.

    | | shape |
    |---|---|
    | `Q` | `(batch, writes, 128)` |
    | `K` | `(batch, reads, 128)` |
    | joined | `(batch, writes, 256)` |
    | shares | `(batch, writes, reads)` |
    """)
    return



@app.function
def with_attention(Q, K):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("15", "with_attention", with_attention, LOCALE)
    return



@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`attend(Q, K, K)` gives the blended vectors and the shares. `torch.cat` joins two tensors along a dimension you choose.""",
        r"""Two lines. The states have three dimensions, `(batch, writes, 128)`, and the joining happens along the last one, `dim=2`. The decoder state goes first.""",
    ], LOCALE)
    return



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Nothing else changes. Same model, same data, same 1,500 batches, same memory size.
    Until you press the button, the cells after it say `Ancestor stopped`. That's normal.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button_2 = mo.ui.run_button(label="Train with attention")
    train_button_2
    return (train_button_2,)


@app.cell
def _(Translator, ids, train_button_2, vocab):
    mo.stop(not train_button_2.value, mo.md("*Press the button to train.*"))
    torch.manual_seed(0)
    rng_attention = np.random.default_rng(0)
    attention_model = Translator(len(vocab), 128, with_attention, 256)
    train_translator(attention_model, ids, 1500, rng_attention)
    attention_scores = []
    for attention_length in [5, 10, 15, 20]:
        attention_scores.append((attention_length, round(share_correct(attention_model, ids, attention_length, rng_attention), 3)))
    attention_scores
    return attention_model, rng_attention


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You should see nearly 100% up to length 15, and about 95% at length 20, against 23% before. The memory is no bigger.
    It just doesn't have to carry everything any more: whatever the decoder needs, it fetches.

    ## Watching it look

    The shares aren't only useful, they're readable. For one sentence of 12 words, here are the shares: one row for
    each word written, one column for each word read. The stronger the blue, the bigger the share. Under the picture are the text read and the text written.
    To write the first word, a good model should look at the last word read; to write the last, at the first.
    """)
    return


@app.cell
def _(attention_model, ids, rng_attention, vocab):
    X_look, Y_look = draw_batch(ids, 12, 1, rng_attention)
    with torch.no_grad():
        written_look, shares_look = translate(attention_model, X_look)
    A_look = torch.cat(shares_look, dim=1)[0].numpy()
    mo.vstack([
        maps([A_look], ["written (down) against read (across)"], 280),
        " ".join([vocab[i] for i in X_look[0].tolist()]),
        " ".join([vocab[i] for i in written_look[0].tolist()]),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It should look like a line of blue from the top right to the bottom left. Nobody told the model where to look: the
    blame from each wrong word taught the decoder's states and the encoder's states to line up.

    Attention also helps blame. There's now a short road from every written word straight to every read word, which is
    the same trick as chapter 11's skip connections and chapter 14's cell.

    ## What's next

    Here's a question. The LSTMs are still there, reading word by word, and they're slow, since each step must wait for the one before it. If attention is what
    does the real work, do we need the LSTMs at all? What if each word simply attended to the other words, all at once?

    That's the **Transformer**. In **Chapter 16** you'll build its one new layer from attention alone. In chapter 17 you'll stack that layer into a model, put it on the leaderboard, and watch it take first place.

    ## What you can do now

    - explain the encoder-decoder design, and why one fixed memory is a bottleneck
    - train a decoder with teacher forcing
    - compute a weighted average with `@`
    - explain query, key and value
    - work out one step of attention by hand: scores, shares, blend
    - write attention for a whole batch, in three steps: compare, share out, blend
    - plug attention into an encoder-decoder model
    - read an attention picture

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** attention for translation is from Bahdanau, Cho and Bengio (2014); this dot-product form, added after the decoder step, is
    close to Luong and colleagues (2015). Real translation systems need a *beam search* instead of always
    taking the likeliest word, and measure quality with scores such as *BLEU*.
    """)
    return


if __name__ == "__main__":
    app.run()
