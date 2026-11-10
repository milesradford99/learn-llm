import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="16 · The Transformer")

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
    from learn_llm.leaderboard import record, show

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 16 · The Transformer

    Chapter 15 ended with a question: if attention does the real work, why keep the LSTM? An LSTM has to read one word
    at a time, so every step waits for the one before it, and blame has a long way to travel.

    The **Transformer** drops recurrence altogether. Every word looks directly at the words before it, all at once. The
    design was published in 2017, and every large language model today is built on it. By the end of this chapter you'll have built one, a tiny **GPT**, with
    about a million knobs, and it will take first place on your leaderboard.

    Every idea in it, you've met: attention (chapter 15), `x + f(x)` (chapter 11), layers and Adam (chapter 10). This chapter adds
    four small pieces, and then puts them together.

    ## Self-attention

    In chapter 15, the decoder's states asked questions of the encoder's states. In **self-attention** the words of one text ask
    questions of each other. Each word's vector goes through three separate `nn.Linear` layers, giving it three roles:

    - a **query**: what am I looking for?
    - a **key**: what do I offer?
    - a **value**: what do I pass on if chosen?

    Then attention runs as before: compare, share out, blend. So the new vector for each word is a blend of the words it found relevant.

    ## Piece 1: no peeking

    A language model predicts the next word, so a word mustn't look at the words after it, or the model would just
    copy the answer. We need a table of positions to block. For 4 words, `True` means "blocked":

    ```
    False  True  True  True      word 0 sees only itself
    False False  True  True      word 1 sees words 0 and 1
    False False False  True
    False False False False      word 3 sees everything before it
    ```

    `torch.ones(n, n, dtype=torch.bool)` is a table of `True`, and `torch.triu(table, diagonal=1)` keeps what's above the
    main diagonal and sets the rest to `False` ("triu" is short for "upper triangle").

    ## Exercise 1: future_mask

    Write `future_mask(n)`:

    ```
    return torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    ```
    """)
    return


@app.function
def future_mask(n):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "future_mask", future_mask, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Piece 2: attention, kept in scale

    To apply the mask, we set blocked scores to minus infinity before softmax. Since $e^{-\infty}$ is 0, a blocked word gets a
    share of exactly 0, and the other shares still add up to 1. `scores.masked_fill(mask, -torch.inf)` stores that number wherever `mask` is `True`.

    There's one more change from chapter 15. A dot product of two vectors of $d$ random numbers sums $d$ parts. By chapter 09's rule,
    variances add, so for numbers of ordinary size its typical size grows with $\sqrt{d}$. Big scores make softmax give nearly all the share to one word,
    and then almost no blame flows to the rest. So we divide the scores by $\sqrt{d}$ to keep them in scale.

    ## Exercise 2: attend

    Write `attend(Q, K, V, mask)`. The shapes are as in chapter 15, and `d` is the last size of `K`. Using `-1` for "the last
    dimension" and `-2` for the one before it means the function will work for any number of dimensions, which we'll need soon.
    It gives back the blend and the shares.

    1. `scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])`
    2. `scores = scores.masked_fill(mask, -torch.inf)`
    3. `A = torch.softmax(scores, dim=-1)`
    4. `return A @ V, A`
    """)
    return


@app.function
def attend(Q, K, V, mask):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "attend", attend, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Piece 3: many heads

    One round of attention gives each word one blend. But a word may want several things at once: who is the subject? what
    tense are we in? So we run several rounds side by side, called **heads**, each with its own queries, keys and values.

    It costs nothing extra in size. With `d = 128` and 4 heads, each vector is cut into 4 pieces of 32 numbers, each head works on
    its own piece, and the 4 results are joined again at the end. The shapes go:

    > `(batch, words, 128)` → `(batch, words, 4, 32)` → `(batch, 4, words, 32)`

    The last swap puts `heads` in the batch-like position, so that `@` and the same `attend` treat each head as a separate text.

    ## Exercise 3: split_heads

    Write `split_heads(x, heads)` for `x` of shape `(batch, words, d)`:

    1. `batch, words, d = x.shape`
    2. `return x.reshape(batch, words, heads, d // heads).transpose(1, 2)`
    """)
    return


@app.function
def split_heads(x, heads):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "split_heads", split_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's the whole layer. `reshape` at the end undoes `split_heads`, and a last `nn.Linear` mixes what the heads found.
    `self.shares` keeps the latest shares, so we can look at them later.
    """)
    return


@app.cell
def _():
    class SelfAttention(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.heads = heads
            self.query = nn.Linear(d, d, bias=False)
            self.key = nn.Linear(d, d, bias=False)
            self.value = nn.Linear(d, d, bias=False)
            self.mix = nn.Linear(d, d)

        def forward(self, x):
            batch, words, d = x.shape
            Q = split_heads(self.query(x), self.heads)
            K = split_heads(self.key(x), self.heads)
            V = split_heads(self.value(x), self.heads)
            blended, self.shares = attend(Q, K, V, future_mask(words))
            return self.mix(blended.transpose(1, 2).reshape(batch, words, d))
    return (SelfAttention,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Attention can't count

    Try something. Take three word vectors and run attention with the mask off. Then reverse the words, run attention again,
    and reverse the answer back. Each word asked the same questions of the same words, so it should get the same blend. The cell
    prints the biggest difference between the two answers:
    """)
    return


@app.cell
def _(SelfAttention):
    torch.manual_seed(0)
    layer_demo = SelfAttention(8, 2)
    x_demo = torch.rand(1, 3, 8)
    Q_demo = split_heads(layer_demo.query(x_demo), 2)
    K_demo = split_heads(layer_demo.key(x_demo), 2)
    V_demo = split_heads(layer_demo.value(x_demo), 2)
    no_mask = torch.zeros(3, 3, dtype=torch.bool)
    forwards = attend(Q_demo, K_demo, V_demo, no_mask)[0]
    backwards = attend(torch.flip(Q_demo, dims=[2]), torch.flip(K_demo, dims=[2]), torch.flip(V_demo, dims=[2]), no_mask)[0]
    torch.max(torch.abs(forwards - torch.flip(backwards, dims=[2]))).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Something like `6e-08`, that is 0.00000006: only rounding, so they're the same. To attention, `the dog bit the man` and `the man bit the dog` are the same bag of words.
    An LSTM got order for free, by reading in order. Here we have to supply it.

    ## Piece 4: positions and a steady scale

    **Positions.** The simplest fix: a second `nn.Embedding`, with one learned vector for each position. Position 0 has a vector,
    position 1 has another, and so on. Add it to the word's vector, and the word now knows where it stands.
    (Chapter 20 shows a smarter way.) It also sets a limit: with a table of 32 positions, our model can see only the last 32 words.

    **A steady scale.** Stacking many layers lets values fade or explode, the trouble from chapter 09's depth table.
    **Layer normalisation** rescales each word's vector to mean 0 and typical size 1, using chapter 09's variance. Real layers
    add two learned knobs on top, which `nn.LayerNorm` does for us.

    ## Exercise 4: layer_norm

    Write `layer_norm(x)`, working across the last dimension. `keepdim=True` is PyTorch's `keepdims=True`, and `unbiased=False` asks for
    plain variance, as in chapter 09. The `1e-5` guards against dividing by 0.

    1. `mean = torch.mean(x, dim=-1, keepdim=True)`
    2. `variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)`
    3. `return (x - mean) / torch.sqrt(variance + 1e-5)`
    """)
    return


@app.function
def layer_norm(x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "layer_norm", layer_norm, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The block

    Attention lets words talk to each other. Then each word needs time to think about what it heard, so a block has a second
    part: a small network from chapter 08, applied to each word separately, with a hidden layer four times as wide.

    Both parts sit on chapter 11's road for blame, `x + f(x)`, with the normalising done just before `f`:

    > `x = x + attention(norm1(x))`

    > `x = x + think(norm2(x))`

    ## Exercise 5: block_step

    Write `block_step(x, norm1, attention, norm2, think)`:

    1. `x = x + attention(norm1(x))`
    2. `x = x + think(norm2(x))`
    3. `return x`
    """)
    return


@app.function
def block_step(x, norm1, attention, norm2, think):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "block_step", block_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The whole model

    Word table plus position table, a stack of blocks, one last normalisation, and an `nn.Linear` for a score per word.
    `torch.arange(n)` is the tensor $0, 1, \ldots, n - 1$. The data is chapter 13's, with the same vocabulary and the same test
    stories, so the score goes on the same leaderboard. The cell after these shows the vocabulary size, the training words and the test words.
    """)
    return


@app.cell
def _(SelfAttention):
    class Block(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.attention = SelfAttention(d, heads)
            self.norm2 = nn.LayerNorm(d)
            self.think = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

        def forward(self, x):
            return block_step(x, self.norm1, self.attention, self.norm2, self.think)

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size, longest, d, heads, layers):
            super().__init__()
            self.words = nn.Embedding(vocab_size, d)
            self.positions = nn.Embedding(longest, d)
            self.blocks = nn.Sequential(*[Block(d, heads) for i in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, vocab_size)

        def forward(self, ids):
            x = self.words(ids) + self.positions(torch.arange(ids.shape[1]))
            return self.out(self.norm(self.blocks(x)))
    return (TinyGPT,)


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


@app.function
def windows(ids, length):
    n = (len(ids) - 1) // length
    return ids[:n * length].reshape(n, length), ids[1:n * length + 1].reshape(n, length)


@app.function
def perplexity(log_probs):
    return np.exp(-np.mean(log_probs))


@app.function
def train_gpt(model, X, Y, epochs, batch_size, rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    size = model.out.out_features
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X[batch]).reshape(-1, size), Y[batch].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return loss.item()


@app.function
def score_gpt(model, X, Y):
    size = model.out.out_features
    log_probs = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            S = model(X[start:start + 64])
            log_probs.append(-F.cross_entropy(S.reshape(-1, size), Y[start:start + 64].reshape(-1), reduction="none").numpy())
    return perplexity(np.concatenate(log_probs))


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
    The button builds a model with 128 numbers per word, 4 heads and 3 blocks, and trains on all 1.9 million words,
    twice. It takes about 3 to 7 minutes on a laptop. Until it finishes, the cells that need its result say `Ancestor stopped`. That's
    normal. If you re-run the training cell, press the button again. The cell shows three numbers: the knobs, the last training loss, and the test perplexity.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the transformer (3 to 7 minutes)")
    train_button
    return (train_button,)


@app.cell
def _(TinyGPT, test_ids, train_button, train_ids, vocab):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    X_np, Y_np = windows(train_ids, 32)
    X_test_np, Y_test_np = windows(test_ids, 32)
    torch.manual_seed(0)
    model = TinyGPT(len(vocab), 32, 128, 4, 3)
    knobs = sum([p.numel() for p in model.parameters()])
    last_loss = train_gpt(model, torch.tensor(X_np), torch.tensor(Y_np), 2, 64, 0.002)
    gpt_score = score_gpt(model, torch.tensor(X_test_np), torch.tensor(Y_test_np))
    knobs, last_loss, gpt_score.item()
    return gpt_score, model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 1.1 million knobs, and a test perplexity of about 23. Here's the leaderboard:
    """)
    return


@app.cell(hide_code=True)
def _(gpt_score):
    record("transformer", gpt_score)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    First place, if you've run chapter 14: 23 against the LSTM's 31, in a similar training time. Chapter 13's counting models are far behind.

    ## Writing, and watching it look

    `write` is chapter 14's loop with one change: no memory is carried along. Instead, the model gets the latest 32 words again at each step (`words[-32:]`). Expect fluent sentences that wander: that's what a 23 gets you.
    """)
    return


@app.function
def write(model, start, count, temperature, index, vocab):
    words = [index[start]]
    with torch.no_grad():
        for step in range(count):
            S = model(torch.tensor([words[-32:]]))
            p = torch.softmax(S[0, -1] / temperature, dim=0)
            words.append(torch.multinomial(p, 1).item())
    return " ".join([vocab[i] for i in words])


@app.cell
def _(index, model, vocab):
    torch.manual_seed(1)
    write(model, "once", 50, 0.8, index, vocab)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    And here are the shares of one head in the last block, as in chapter 15: one row for each word, one column for each earlier word it looks at. The upper right is blank, because that's the future.
    """)
    return


@app.cell
def _(index, model, vocab):
    sentence = "once upon a time there was a little girl named lily she loved her dog".split()
    with torch.no_grad():
        model(torch.tensor([[index[word] for word in sentence]]))
    looks = model.blocks[-1].attention.shares[0, 0].numpy()
    mo.vstack([maps([looks], ["word (down) looks at word (across)"], 320), " ".join(sentence)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (This is the first of the 4 heads.) Each row adds up to 1. Don't expect each head to mean something tidy: patterns like these are often hard to read, and that's normal.

    ## What's next

    You've built the core of a modern language model, and it's small enough to understand completely. The difference between this and a famous model is mostly
    size and data. Big models have billions of knobs, and were trained on far more text.

    One thing you've seen but not yet questioned: how do we know a model is any good at a task, beyond one score? **Chapter 18** looks at how
    models are reused on new tasks, and at the many ways to fool yourself when you measure them.

    ## What you can do now

    - explain query, key and value in self-attention
    - block the future with a mask, and scale scores by $\sqrt{d}$
    - split vectors into heads, and say why that costs no extra size
    - explain why attention alone ignores word order, and how positions fix that
    - normalise a layer, and build a block from two `x + f(x)` steps
    - train a tiny GPT and put it on the leaderboard

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the Transformer is from *Attention Is All You Need* (Vaswani et al., 2017), and GPT from OpenAI's papers from 2018 on. The original used fixed
    sine waves for positions. The order "norm, then attention" is called *pre-norm*. Real models also use *dropout*.
    """)
    return


if __name__ == "__main__":
    app.run()
