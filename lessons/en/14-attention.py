import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="14 · Attention")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.images import maps

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 14 · Attention

    Chapter 13's LSTM squeezes everything it has read into one vector of 128 numbers. For predicting the next word
    in a story, that goes a long way. Now take a harder job: translation. A model must read a whole
    sentence, then write it out in another language. It's natural to build that from two LSTMs: an **encoder** reads
    the sentence into a memory, and a **decoder** starts from that memory and writes.

    The trouble is the single memory: all of a long sentence has to pass through one small vector. Think about how you'd translate
    a long sentence yourself. You wouldn't memorise it and then turn away. You'd keep looking back at the
    words you need, as you need them.

    That looking back is called **attention**, and it's the idea behind the T in ChatGPT, which stands for Transformer (chapter 15).

    ## A task we can check

    We have no translated text in the course, so we'll use a task with the same shape, where every
    answer can be checked at a glance: **write the text backwards**.

    > in: `the dog ran home` → out: `home ran dog the`

    It's a good test, because the first word to write is the last one read, and the last to write is the very first one read. No
    understanding is needed, only remembering a whole sequence exactly. The cell below prepares 500 common words, plus a new `<start>` word explained below, and a
    function that draws random stretches of text, of a chosen length, with their reversal. `torch.flip` reverses along a dimension,
    `rng.integers` can draw a whole batch of starting points at once, and `np.stack` makes a table from a list of rows.
    (`to_words` is chapter 11's text cutter.)
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

    Write `shift_right(Y)`, for a table `Y` of shape `(batch, length)`. `Y[:, :-1]` drops the last column. `Y[:, :1]` is the first column, and
    `torch.zeros_like` gives zeros of the same shape and kind. Chapter 13's `torch.cat` joins the two:

    ```
    return torch.cat([torch.zeros_like(Y[:, :1]), Y[:, :-1]], dim=1)
    ```
    """)
    return


@app.function
def shift_right(Y):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "shift_right", shift_right, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the model. It's chapter 13's `Reader` in two halves. The encoder gives back its states `K`, one
    vector for each word read, and its final memory. The decoder starts from that final memory and gives its own states `Q`, one for each word to write.

    `use` is a function we hand in, as with `step` in chapter 10's `Block`: it says what to do with `Q` and `K`. It gives back two things, the vectors to score and the shares (more on those soon),
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

    The fix starts with a small idea. Say we have three vectors, and we want a mix of them: 70% of the first, 20% of the second,
    10% of the third. That's a **weighted average**: multiply each vector by its share, and add up. If the vectors are the rows of a matrix `V`
    and the shares are a vector `a`, it's just `a @ V`, chapter 05's vector times matrix.

    With shares `[1, 0, 0]`, you get exactly the first vector. So a weighted average is a kind of soft lookup: it can pick out one row, or blend a few.

    ## Exercise 2: weighted_average

    Write `weighted_average(a, V)`, for shares `a` of shape `(n,)` and rows `V` of shape `(n, d)`:

    ```
    return a @ V
    ```
    """)
    return


@app.function
def weighted_average(a, V):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "weighted_average", weighted_average, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where to look: query, key, value

    Now the decoder can look back. At each step of writing, it has a state, and the encoder has left one state per word read.
    Attention takes three steps, and you know every piece of them:

    1. **Compare.** Take the dot product of the decoder's state with each encoder state. A big dot product means "this one is relevant"
       (chapter 11: dot products measure how alike vectors are).
    2. **Share out.** Softmax turns those scores into shares that add up to 1 (chapter 06).
    3. **Blend.** Take the weighted average of the encoder states with those shares.

    The result is a fresh summary of the input, made for this one step of writing. The terms for these come from looking
    things up. The **query** `q` is what you're looking for, the **keys** `K` are the labels you compare it with, and
    the **values** `V` are what you get back. Here the keys and the values are the same encoder states, but they needn't be, and in
    the next chapter they won't be.

    Why would a decoder state and an encoder state have a big dot product? At first, they don't. Training makes the useful ones large, as you'll see below.

    ## Exercise 3: attention_shares

    Write `attention_shares(q, K)` for one query `q` of shape `(d,)`, and keys `K` of shape `(n, d)`. `K @ q` gives one dot product per key:

    ```
    return torch.softmax(K @ q, dim=0)
    ```
    """)
    return


@app.function
def attention_shares(q, K):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "attention_shares", attention_shares, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: attend

    Now everything at once: a batch of texts, with many queries in each. `Q` has shape `(batch, writes, d)`, `K` has shape
    `(batch, reads, d)`, and `V` has shape `(batch, reads, e)`: values may have a size of their own. On tensors with three dimensions, `@` works text by text, on the last two.
    `K.transpose(1, 2)` swaps dimensions 1 and 2, like `.T` for each text.

    Write `attend(Q, K, V)`, giving back the blended vectors and the shares:

    1. `scores = Q @ K.transpose(1, 2)`, shape `(batch, writes, reads)`: every query against every key
    2. `A = torch.softmax(scores, dim=2)`, so each query's shares add up to 1
    3. `return A @ V, A`, the weighted averages, of shape `(batch, writes, e)`, and the shares
    """)
    return


@app.function
def attend(Q, K, V):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("14", "attend", attend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To plug it in, `with_attention` joins each decoder state to its blended summary, side by side, which is why the last layer's
    width is twice the memory size. Nothing else changes. Same model, same data, same 1,500 batches, same memory size.
    Until you press the button, the cells after it say `Ancestor stopped`. That's normal.
    """)
    return


@app.function
def with_attention(Q, K):
    blended, A = attend(Q, K, K)
    return torch.cat([Q, blended], dim=2), A


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
    the same trick as chapter 10's skip connections and chapter 13's cell.

    ## What's next

    Here's a question. The LSTMs are still there, reading word by word, and they're slow, since each step must wait for the one before it. If attention is what
    does the real work, do we need the LSTMs at all? What if each word simply attended to the other words, all at once?

    That's the **Transformer**, and in **Chapter 15** you'll build one from attention alone, put it on the leaderboard, and watch it take first place.

    ## What you can do now

    - explain the encoder-decoder design, and why one fixed memory is a bottleneck
    - train a decoder with teacher forcing
    - compute a weighted average with `@`
    - explain query, key and value
    - write attention for a whole batch, in three steps: compare, share out, blend
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
