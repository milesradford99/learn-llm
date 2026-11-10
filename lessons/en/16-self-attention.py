import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="16 · Self-attention")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 16 · Self-attention

    Chapter 15 ended with a question: if attention does the real work, why keep the LSTM? An LSTM has to read one word
    at a time, so every step waits for the one before it, and blame has a long way to travel.

    The **Transformer** drops recurrence altogether. Every word looks directly at the words before it, all at once. The
    design was published in 2017, and every large language model today is built on it.

    It takes two chapters. This one builds its one new part, the **self-attention** layer, piece by piece. Chapter 17 stacks
    that layer into a tiny **GPT**, trains it, and puts it on your leaderboard. Every idea in the layer, you've met: it's chapter 15's attention,
    with four changes.

    ## Words asking each other

    In chapter 15, the decoder's states asked questions of the encoder's states. In self-attention the words of one text ask
    questions of each other: each word is a query, and each word is also a key and a value.

    The simplest way would be to use each word's vector for all three. Let's try it on three words with vectors of 2 numbers.
    The table below holds every score: row $i$, column $j$ is the dot product of word $i$ with word $j$, that is, how strongly word $i$ looks at word $j$.
    """)
    return


@app.cell
def _():
    X_words = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]])
    X_words @ X_words.T
    return (X_words,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Two things are wrong with this table.

    - **Every word looks mostly at itself.** A vector's dot product with itself is the sum of its squares, which is usually the biggest in its row:
      4 in row 0, and 9 in row 2.
    - **Looking is mutual.** Row 1, column 2 holds 3, and so does row 2, column 1. Word 1 looks at word 2 exactly as strongly as word 2 looks at word 1,
      because a dot product doesn't care about the order of its two vectors.

    Language doesn't work like that. In `the dog ran`, the word `ran` needs to find its subject, `dog`. That doesn't mean `dog` needs `ran` just as much.

    So each word gets separate vectors for separate roles. Its vector goes through three different `nn.Linear` layers (with no bias, so each is a matrix), giving it:

    - a **query**: what am I looking for?
    - a **key**: what do I offer to others who are looking?
    - a **value**: what do I pass on, if I'm chosen?

    Here are the same three words, with a made-up matrix for the queries and another for the keys. Row $i$, column $j$ is now the query of word $i$ times the key of word $j$:
    """)
    return


@app.cell
def _(X_words):
    to_query_demo = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    to_key_demo = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    (X_words @ to_query_demo) @ (X_words @ to_key_demo).T
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The table is no longer the same when you swap rows and columns: word 2 looks at word 0 with a score of 6, while word 0 looks at word 2 with a score of $-6$.
    And the diagonal is no longer the biggest. What a word looks for and what it offers have come apart. In a real layer, these matrices are knobs, and training finds them.

    After that, attention runs as before: compare, share out, blend the values. So the new vector of each word is a blend of what the words it found relevant passed on.

    ## By hand: who asks whom

    Two words. `dog` has the query `[1, 0]` and the key `[0, 2]`. `ran` has the query `[0, 3]` and the key `[1, 1]`.
    A score is always the asker's query times the other word's key. Work out both.
    """)
    return


@app.cell(hide_code=True)
def _():
    ask_boxes = hand.boxes(['dog asks ran', 'ran asks dog'])
    return (ask_boxes,)


@app.cell(hide_code=True)
def _(ask_boxes):
    hand.marked(ask_boxes, {'dog asks ran': 1, 'ran asks dog': 6}, {
        'dog asks ran': r"""the query of `dog`, `[1, 0]`, times the key of `ran`, `[1, 1]`""",
        'ran asks dog': r"""the query of `ran`, `[0, 3]`, times the key of `dog`, `[0, 2]`""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Different, as intended. That was change 1: three roles. Three more follow.

    ## Change 2: no peeking

    A language model predicts the next word, so a word mustn't look at the words after it, or the model would just
    copy the answer. We need a table that says which looks are blocked. For 4 words, `True` means "blocked":

    ```
    False  True  True  True      word 0 sees only itself
    False False  True  True      word 1 sees words 0 and 1
    False False False  True
    False False False False      word 3 sees everything up to itself
    ```

    Everything above the main diagonal is `True`. This table is called a **mask**.

    ## By hand: the mask

    For a text of 5 words, how many looks does the mask block? And how many words can word 2 see, itself included? (Words are numbered from 0.)
    """)
    return


@app.cell(hide_code=True)
def _():
    mask_boxes = hand.boxes(['blocked', 'word 2 sees'])
    return (mask_boxes,)


@app.cell(hide_code=True)
def _(mask_boxes):
    hand.marked(mask_boxes, {'blocked': 10, 'word 2 sees': 3}, {
        'blocked': r"""count the `True`s row by row: word 0 is blocked from 4 words, word 1 from 3, and so on""",
        'word 2 sees': r"""words 0, 1 and 2""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: future_mask

    /// admonition | 🧰 Toolbox: tables of True and False
    - `torch.ones(2, 3, dtype=torch.bool)` is a table of 2 rows and 3 columns, all `True`. With `torch.bool`, 1 is stored as `True` and 0 as `False`.
    - `torch.triu(T, diagonal=1)` keeps what lies above the main diagonal of `T`, and sets the rest to 0, or `False`. ("triu" is short for "upper triangle".)
      For a table with the rows `[1, 2, 3]`, `[4, 5, 6]` and `[7, 8, 9]` it gives `[0, 2, 3]`, `[0, 0, 6]`, `[0, 0, 0]`.
    - `S.masked_fill(mask, 99.0)` is a copy of `S` with 99 stored wherever `mask` is `True`.
    ///

    Write `future_mask(n)`, giving the table above for `n` words: `n` rows, `n` columns, `True` exactly where a word would look at a later word.
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
    hand.stuck([
        r"""Start from a table that is `True` everywhere, then keep only the part above the main diagonal.""",
        r"""One line: `torch.triu(...)` around a table of `n` by `n` ones with `dtype=torch.bool`. With `diagonal=1` the main diagonal itself is not kept, since a word may look at itself.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    To apply the mask, we set the blocked scores to minus infinity before softmax. Since $e^{-\infty}$ is 0, a blocked word gets a
    share of exactly 0, and the other shares still add up to 1. PyTorch writes minus infinity as `-torch.inf`.

    ## Change 3: keeping the scores in scale

    A dot product of two vectors of $d$ numbers is a sum of $d$ parts. By chapter 09's rule,
    variances add, so for numbers of ordinary size its typical size grows with $\sqrt{d}$. With 128 numbers per word, scores are far bigger than with 4.

    Why does that matter? Softmax turns big gaps between scores into shares near 0 and 1. The cell below measures it. For each size `d`, it makes a random query and 8 random keys,
    200 times over, and notes the largest of the 8 shares: once with plain scores, once with the scores divided by $\sqrt{d}$. Each row shows `d`, then the two averages.
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    scale_table = []
    for d_demo in [4, 32, 128, 512]:
        plain = 0.0
        scaled = 0.0
        for trial in range(200):
            q_demo = torch.rand(d_demo) * 2 - 1
            K_demo = torch.rand(8, d_demo) * 2 - 1
            plain = plain + torch.max(torch.softmax(K_demo @ q_demo, dim=0)).item() / 200
            scaled = scaled + torch.max(torch.softmax(K_demo @ q_demo / np.sqrt(d_demo), dim=0)).item() / 200
        scale_table.append((d_demo, round(plain, 2), round(scaled, 2)))
    scale_table
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With plain scores, the biggest share climbs from under 0.3 to nearly 0.9: at 512 numbers, one key takes nearly everything before any training has happened.
    That's bad for learning. When one share is near 1, softmax is flat, and almost no blame reaches the other words (chapter 06).

    Divided by $\sqrt{d}$, the biggest share stays near 0.2 at every size. So that's what self-attention does:

    > $$\text{scores} = \frac{Q K^\top}{\sqrt{d}}$$

    ## Exercise 2: attend

    /// admonition | 🧰 Toolbox: counting dimensions from the end
    `dim=-1` means the last dimension, and `-2` the one before it. `M.transpose(-2, -1)` swaps the last two dimensions, however many come before them,
    and `M.shape[-1]` is the last size. Code written this way works on `(batch, words, d)` and also on tensors with an extra dimension in the middle, which we'll need for change 4.
    ///

    Write `attend(Q, K, V, mask)`. It's chapter 15's `attend` with changes 2 and 3, and it gives back the blend and the shares, as before:

    1. The scores: every query against every key, divided by $\sqrt{d}$, where $d$ is the last size of `K`. `np.sqrt` works on a plain number.
    2. Wherever `mask` is `True`, the score becomes `-torch.inf`.
    3. The shares `A`: softmax along the last dimension.
    4. The blend of the values with those shares.

    Count every dimension from the end, so that it also works for `Q`, `K` and `V` of shape `(batch, heads, words, d)`.
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
    hand.stuck([
        r"""Four lines. Lines 1, 3 and 4 are chapter 15's three lines with `-2` and `-1` in place of `1` and `2`. Line 2 uses `masked_fill` from the Toolbox box of exercise 1.""",
        r"""The mask goes on the scores, before softmax. If you mask after softmax, the shares no longer add up to 1. If the check says only the sizes of the numbers are off, look at the division by `np.sqrt(K.shape[-1])`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Change 4: many heads

    One round of attention gives each word one blend. But a word may want several things at once: who is the subject? what
    tense are we in? So we run several rounds side by side, called **heads**, each with its own queries, keys and values.

    It costs nothing extra in size. With 128 numbers per word and 4 heads, each vector is cut into 4 pieces of 32 numbers, each head works on
    its own piece, and the 4 results are joined again at the end.

    The cutting is done with shapes alone. Follow a tiny example: 1 text, 2 words, 4 numbers per word, cut for 2 heads.
    `torch.arange(8.0)` is the tensor of the numbers 0.0, 1.0, up to 7.0, like `range`. The cell shows the tensor, then the
    tensor after `reshape`, then after `transpose`:
    """)
    return


@app.cell
def _():
    x_trace = torch.arange(8.0).reshape(1, 2, 4)
    cut = x_trace.reshape(1, 2, 2, 2)
    swapped = cut.transpose(1, 2)
    x_trace.tolist(), cut.tolist(), swapped.tolist()
    return (x_trace,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Read the three results one after the other.

    1. At the start, the shape is `(1, 2, 4)`. Word 0 is `[0, 1, 2, 3]` and word 1 is `[4, 5, 6, 7]`.
    2. `reshape(1, 2, 2, 2)` cuts each word in two. The shape is now `(batch, words, heads, piece)`: word 0 holds the pieces `[0, 1]` and `[2, 3]`.
    3. `transpose(1, 2)` swaps `words` and `heads`. The shape is now `(batch, heads, words, piece)`: head 0 holds `[0, 1]` from word 0 and `[4, 5]` from word 1,
       which are the first halves of both words. Head 1 holds the second halves.

    Why the swap? `@` works on the last two dimensions and treats all the ones before them as a batch. With `heads` moved up front, next to `batch`,
    your `attend` runs each head as if it were a separate text, with no change at all. That's why exercise 2 counted dimensions from the end.

    ## By hand: the shapes

    A tensor of shape `(1, 6, 12)`, one text of 6 words with 12 numbers each, is split for 3 heads. Fill in the four sizes of the result, in order.
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['size 0', 'size 1', 'size 2', 'size 3'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'size 0': 1, 'size 1': 3, 'size 2': 6, 'size 3': 4}, {
        'size 0': r"""the batch stays in front""",
        'size 1': r"""after the swap, `heads` comes second""",
        'size 2': r"""then `words`""",
        'size 3': r"""each piece holds 12 numbers divided among 3 heads""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: split_heads

    Write `split_heads(x, heads)` for `x` of shape `(batch, words, d)`, giving back shape `(batch, heads, words, d // heads)`, as in the example.
    (`//` is chapter 11's division that gives a whole number.)

    `batch, words, d = x.shape` gives the three sizes their names.

    On the example above, `split_heads(x_trace, 2)` must give the third result.
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
    hand.stuck([
        r"""Two steps, the same as in the example: `reshape` to `(batch, words, heads, piece)`, then swap dimensions 1 and 2.""",
        r"""Reshape first, then transpose. Reshaping straight to `(batch, heads, words, piece)` gives the right shape with the numbers in the wrong heads.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: join_heads

    After attention, the heads have to be joined again. Write `join_heads(x)`, which undoes `split_heads`: it takes shape
    `(batch, heads, words, piece)` and gives back `(batch, words, heads * piece)`.

    Undo the two steps in the opposite order: first the swap, then the cut. `batch, heads, words, piece = x.shape` names the four sizes.

    On the example, `join_heads` of the third result must give back the first.
    """)
    return


@app.function
def join_heads(x):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "join_heads", join_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`transpose(1, 2)` undoes itself: swapping the same two dimensions again puts them back.""",
        r"""Two lines: name the four sizes, then transpose and reshape to `(batch, words, heads * piece)`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: the whole layer

    Now put the four changes together. Write `self_attention_step(x, to_query, to_key, to_value, mix, heads)`.
    `x` has shape `(batch, words, d)`. The next four inputs are layers, and you call a layer like a function: `to_query(x)`.

    1. Make the queries, keys and values: each is its layer applied to `x`, then split into heads.
    2. Run `attend` on them, with the mask for this number of words. The number of words is `x.shape[1]`.
    3. Join the heads of the blend, and pass the result through `mix`, a last layer that lets the heads' findings combine.
    4. Give back that result and the shares.

    | | shape |
    |---|---|
    | `x` | `(batch, words, d)` |
    | queries, keys, values | `(batch, heads, words, d // heads)` |
    | shares | `(batch, heads, words, words)` |
    | result | `(batch, words, d)` |
    """)
    return


@app.function
def self_attention_step(x, to_query, to_key, to_value, mix, heads):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("16", "self_attention_step", self_attention_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Every piece is a function you've written in this chapter: `split_heads`, `future_mask`, `attend`, `join_heads`. Five lines will do: three for the roles, one for `attend`, one to give back.""",
        r"""`attend` gives back a pair, the blend and the shares. Only the blend is joined and mixed. The shares are given back as they are.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is the layer as a PyTorch module. Its knobs are the three role matrices and `mix`, and its `forward` is your function. It keeps the latest shares in
    `self.shares`, so we can look at them in chapter 17. The cell after it runs a fresh layer on one text of 5 words with 8 numbers each, split over 2 heads, and shows two shapes.
    Until all five exercises pass, the cells after it show an error. That's normal.
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
            out, self.shares = self_attention_step(x, self.query, self.key, self.value, self.mix, self.heads)
            return out
    return (SelfAttention,)


@app.cell
def _(SelfAttention):
    torch.manual_seed(0)
    layer_demo = SelfAttention(8, 2)
    x_demo = torch.rand(1, 5, 8)
    with torch.no_grad():
        out_demo = layer_demo(x_demo)
    out_demo.shape, layer_demo.shares.shape
    return layer_demo, x_demo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `[1, 5, 8]` and `[1, 2, 5, 5]`: the output has the shape of the input, so layers can be stacked. The shares are one table of 5 by 5 for each of the 2 heads.

    ## Attention can't count

    One thing is missing, and it's easy to overlook. Take those five word vectors and run attention with the mask off. Then reverse the words, run attention again,
    and reverse the answer back. Each word asked the same questions of the same words, so it should get the same blend. The cell
    prints the biggest difference between the two answers:
    """)
    return


@app.cell
def _(layer_demo, x_demo):
    Q_demo = split_heads(layer_demo.query(x_demo), 2)
    K_demo2 = split_heads(layer_demo.key(x_demo), 2)
    V_demo = split_heads(layer_demo.value(x_demo), 2)
    no_mask = torch.zeros(5, 5, dtype=torch.bool)
    forwards = attend(Q_demo, K_demo2, V_demo, no_mask)[0]
    backwards = attend(torch.flip(Q_demo, dims=[2]), torch.flip(K_demo2, dims=[2]), torch.flip(V_demo, dims=[2]), no_mask)[0]
    torch.max(torch.abs(forwards - torch.flip(backwards, dims=[2]))).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Something like `1e-07`, that is 0.0000001: only rounding, so they're the same. To attention, `the dog bit the man` and `the man bit the dog` are the same bag of words.
    An LSTM got order for free, by reading in order. Here we'll have to supply it.

    ## What's next

    You have the layer. **Chapter 17** tells it where each word stands, keeps its numbers steady through many layers, stacks it into
    a tiny GPT with about a million knobs, and trains it on the stories.

    ## What you can do now

    - say why a word needs a separate query, key and value
    - block the future with a mask
    - say why scores are divided by $\sqrt{d}$
    - split vectors into heads and join them again, and say why that costs no extra size
    - build a self-attention layer from those pieces
    - explain why attention alone ignores word order

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the Transformer is from *Attention Is All You Need* (Vaswani et al., 2017), where this layer is called
    *scaled dot-product attention* with *multi-head attention*. A layer whose words may look both ways, with no mask, is the kind used in BERT (chapter 18).
    """)
    return


if __name__ == "__main__":
    app.run()
