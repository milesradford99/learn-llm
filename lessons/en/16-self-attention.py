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

    Chapter 15 ended with a question: if attention is doing the real work, why keep the LSTM at all? An LSTM has to read one word
    at a time, so every step waits for the previous one, and gradients (error signals) have to travel a long chain back to earlier words.

    The **Transformer** drops recurrence altogether. Every word looks directly at the words before it, all at once. Published in 2017,
    this design is the backbone of every large language model today.

    We'll build it across two chapters. This chapter builds its one new component, the **self-attention** layer, piece by piece. Chapter 17 stacks
    that layer into a tiny **GPT**, trains it, and puts it on your leaderboard. Every core idea is one you've met before: it's chapter 15's attention
    (score, softmax into weights, blend values), with four upgrades.

    ## Words asking each other

    In chapter 15, the decoder's states asked questions of the encoder's states. In self-attention, the words of a single text ask
    questions of each other: every word acts as a query, a key, and a value all at once.

    The simplest idea would be to use each word's raw vector for all three roles. Let's try it on three words with 2-number vectors.
    The table below holds every score: row $i$, column $j$ is the dot product of word $i$ with word $j$ — how strongly word $i$ attends to word $j$.
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
    Two things go wrong with this table.

    - **Every word attends mostly to itself.** A vector's dot product with itself is the sum of its squares, which is usually the largest entry in its row:
      4 in row 0, and 9 in row 2.
    - **Attention is forced to be symmetric.** Row 1, column 2 holds 3, and so does row 2, column 1. Word 1 attends to word 2 with the exact same score as word 2 attends to word 1,
      because a dot product doesn't care which vector comes first.

    Language doesn't work like that. In `the dog ran`, the verb `ran` needs to find its subject `dog`, but `dog` doesn't need `ran` in the same way.

    So each word gets separate vectors for separate roles. Its vector goes through three different `nn.Linear` layers (with no bias, so each is just a matrix), giving it:

    - a **query**: what am I looking for?
    - a **key**: what do I offer to others who are looking?
    - a **value**: what information do I pass along if I'm chosen?

    Here are the same three words with one made-up matrix for queries and another for keys. Row $i$, column $j$ is now word $i$'s query dotted with word $j$'s key:
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
    The table is no longer symmetric: word 2 scores 6 on word 0, while word 0 scores $-6$ on word 2.
    And the diagonal is no longer the largest. What a word searches for and what it advertises have been separated. In a real layer, these matrices are trainable parameters learned from data.

    After that, attention runs just as before: compute scores, turn each row into weights (probabilities) with softmax, and take a weighted blend of the values. Each word's new vector is a blend of what the relevant words passed along.

    ## By hand: who asks whom

    Two words: `dog` has query `[1, 0]` and key `[0, 2]`; `ran` has query `[0, 3]` and key `[1, 1]`.
    A score is always the asker's query dotted with the other word's key. Work out both.
    """)
    return


@app.cell(hide_code=True)
def _():
    ask_boxes = hand.boxes(['dog asks ran', 'ran asks dog'])
    return (ask_boxes,)


@app.cell(hide_code=True)
def _(ask_boxes):
    hand.marked(ask_boxes, {'dog asks ran': 1, 'ran asks dog': 6}, {
        'dog asks ran': r"""the query of `dog`, `[1, 0]`, dotted with the key of `ran`, `[1, 1]`""",
        'ran asks dog': r"""the query of `ran`, `[0, 3]`, dotted with the key of `dog`, `[0, 2]`""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Different, as intended. That was change 1: three separate roles. Three more follow.

    ## Change 2: no peeking

    A language model predicts the next word, so a word must not look at words that come after it — otherwise the model would just
    copy the answer. We need a table marking which positions are blocked. For 4 words, `True` means blocked:

    ```
    False  True  True  True      word 0 sees only itself
    False False  True  True      word 1 sees words 0 and 1
    False False False  True
    False False False False      word 3 sees everything up to itself
    ```

    Everything strictly above the main diagonal is `True`. This table is called a **mask**.

    ## By hand: the mask

    For a 5-word text, how many future positions does the mask block in total? And how many words can word 2 see, itself included? (Words are numbered from 0.)
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

    Write `future_mask(n)`, returning the mask table above for `n` words: `n` rows, `n` columns, `True` exactly where a word would look at a later word.
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
    To apply the mask, we set every blocked score to minus infinity before softmax. Since $e^{-\infty} = 0$, a blocked word gets
    weight 0, and the remaining weights still add up to 1. PyTorch writes minus infinity as `-torch.inf`.

    ## Change 3: scaling scores by $\sqrt{d}$

    A dot product of two vectors of length $d$ adds $d$ products together. By chapter 09's rule,
    variances add when independent terms are summed (variance is the square of typical size: adding $d$ terms makes the variance $d$ times larger, so taking the square root makes the typical size $\sqrt{d}$ times larger). With 128 numbers per word, dot-product scores are far larger than with 4.

    Why does that matter? When the gaps between scores grow large, softmax pushes the weights toward 0 and 1. The cell below measures this directly. For each vector size `d`, it generates a random query and 8 random keys
    200 times, and records the largest of the 8 attention weights: once with raw scores, and once with scores divided by $\sqrt{d}$. Each row shows `d`, then the two averages.
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
    With raw scores, the largest weight climbs from under 0.3 to nearly 0.9: at 512 numbers, one random key grabs almost all the attention before any training has even happened.
    That hurts learning. Remember chapter 07: when one probability is close to 1 (and the rest are near 0), the softmax curve flattens out — nudging a score barely changes the probabilities, so almost no gradient reaches the other words.

    Divided by $\sqrt{d}$, the largest weight stays near 0.2 at every size. So self-attention scales every score by $\sqrt{d}$:

    > $$\text{scores} = \frac{Q K^\top}{\sqrt{d}}$$

    ## Exercise 2: attend

    /// admonition | 🧰 Toolbox: counting dimensions from the end
    `dim=-1` means the last dimension, and `-2` the one before it. `M.transpose(-2, -1)` swaps the last two dimensions, however many come before them,
    and `M.shape[-1]` is the last size. Code written this way works on 3D `(batch, words, d)` tensors and also on 4D tensors with an extra `heads` dimension in the middle, which we'll need for change 4.
    ///

    Write `attend(Q, K, V, mask)`. It's chapter 15's `attend` with changes 2 and 3, returning the blended values and the attention weights:

    1. **Scores:** every query dotted with every key, divided by $\sqrt{d}$, where $d$ is the last size of `K` (`np.sqrt` works on a plain number).
    2. **Mask:** wherever `mask` is `True`, replace the score with `-torch.inf`.
    3. **Weights `A`:** softmax along the last dimension.
    4. **Blend:** multiply the weights `A` by `V`, and return both the blend and `A`.

    Count every dimension from the end so that `attend` works for both 3D `(batch, words, d)` and 4D `(batch, heads, words, d)` inputs.

    **Note on mask shape (2D to 4D broadcasting):** `future_mask(n)` returns a 2D table of shape `(words, words)`. In multi-head mode, `scores` is 4D with shape `(batch, heads, words, words)`. You don't need to reshape the mask: `masked_fill` aligns dimensions from the right and automatically **broadcasts** that 2D mask across every sequence in the batch and every head.
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
        r"""The mask goes on the scores, before softmax. If you mask after softmax, the weights no longer add up to 1. If the check says only the sizes of the numbers are off, look at the division by `np.sqrt(K.shape[-1])`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Change 4: multiple heads

    One round of attention gives each word a single blend. Yet a word often wants several kinds of information at once: who is the subject? what
    tense are we in? So we run several rounds side by side, called **heads**, each working on its own slice of queries, keys, and values.

    This costs nothing extra in vector size. With 128 numbers per word and 4 heads, each word's vector is sliced into 4 pieces of 32 numbers, each head attends using
    its own piece, and the 4 results are stitched back together at the end.

    The slicing is done with tensor shapes alone. Follow a tiny example: 1 text, 2 words, 4 numbers per word, split across 2 heads.
    `torch.arange(8.0)` creates the tensor of numbers 0.0, 1.0, up to 7.0, like `range`. The cell shows the original tensor, then the
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
    2. `reshape(1, 2, 2, 2)` slices each word in two. The shape is now `(batch, words, heads, piece)`: word 0 holds `[0, 1]` and `[2, 3]`, and word 1 holds `[4, 5]` and `[6, 7]`.
    3. `transpose(1, 2)` swaps `words` and `heads`. The shape is now `(batch, heads, words, piece)`: head 0 holds `[0, 1]` from word 0 and `[4, 5]` from word 1
       (the first half of each word), while head 1 holds `[2, 3]` and `[6, 7]` (the second half).

    **Why can't we skip `transpose` and `reshape` straight to `(batch, heads, words, piece)`?** Because `reshape` never reorders numbers — it only regroups them in sequence.
    If you jump straight to `reshape(1, 2, 2, 2)` and treat axis 1 as `heads`, head 0 swallows the entire word 0 (`[0, 1, 2, 3]`) and head 1 swallows word 1 (`[4, 5, 6, 7]`), instead of head 0 getting the first half of every word (`[0, 1]` and `[4, 5]`)!
    So you must always slice each word first (`reshape` to `(batch, words, heads, piece)`), then move `heads` forward (`transpose(1, 2)`).

    Why move `heads` forward at all? Matrix multiplication (`@`) acts on the last two dimensions and treats all leading dimensions as a batch. With `heads` moved up front next to `batch`,
    your `attend` runs every head in parallel as if it were a separate text — and broadcasts the 2D mask to every head automatically — with no change at all. That's why exercise 2 counted dimensions from the end.

    ## By hand: the shapes

    A tensor of shape `(1, 6, 12)` — one text of 6 words with 12 numbers each — is split for 3 heads. Fill in the four sizes of the result, in order.
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

    Write `split_heads(x, heads)` for `x` of shape `(batch, words, d)`, returning shape `(batch, heads, words, d // heads)`, as in the example.
    (`//` is integer division, e.g. `12 // 3` is `4`.)

    `batch, words, d = x.shape` unpacks and names the three sizes.

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

    After attention, the heads have to be stitched back together. Write `join_heads(x)`, which undoes `split_heads`: it takes shape
    `(batch, heads, words, piece)` and returns `(batch, words, heads * piece)`.

    Undo the two steps in reverse order: first undo the swap, then undo the slice. `batch, heads, words, piece = x.shape` unpacks and names the four sizes.

    On the example, `join_heads` on the third result must give back the first.
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

    Now put all four changes together. Write `self_attention_step(x, to_query, to_key, to_value, mix, heads)`.
    `x` has shape `(batch, words, d)`. The next four inputs are layers, and you call a layer like a function: `to_query(x)`.

    1. Build the queries, keys, and values: pass `x` through each layer, then split into heads.
    2. Run `attend` on them with the mask for this number of words (`x.shape[1]`).
    3. Rejoin the heads of the blended values, and pass the result through `mix`, a final layer that combines what the different heads found.
    4. Return that result and the attention weights.

    | | shape |
    |---|---|
    | `x` | `(batch, words, d)` |
    | queries, keys, values | `(batch, heads, words, d // heads)` |
    | attention weights | `(batch, heads, words, words)` |
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
        r"""Every piece is a function you've written in this chapter: `split_heads`, `future_mask`, `attend`, `join_heads`. Five lines will do: three for the roles, one for `attend`, one to return.""",
        r"""`attend` returns a pair: the blend and the attention weights. Only the blend is joined and passed through `mix`; the weights are returned as they are.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here is the layer wrapped as a PyTorch module. Its trainable parameters are the three role matrices and `mix`, and its `forward` simply calls your function. It saves the latest attention weights in
    `self.shares` so we can inspect them in chapter 17. The cell after it runs a fresh layer on one text of 5 words with 8 numbers each, split across 2 heads, and shows two shapes.
    Until all five exercises pass, the cells below will show an error — that's normal.
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
    `[1, 5, 8]` and `[1, 2, 5, 5]`: the output has the exact same shape as the input, so layers can be stacked one on top of another. The attention weights form one 5-by-5 table for each of the 2 heads.

    ## Attention doesn't know word order

    One thing is still missing, and it's easy to overlook. Take those five word vectors and run attention with the mask turned off. Then reverse the order of the words, run attention again,
    and reverse the output back. Each word asked the same questions of the same words, so it should get the same blend. The cell
    prints the largest difference between the two outputs:
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
    Something like `1e-07` (that is, 0.0000001): just floating-point rounding, so the two outputs are identical. Without a mask, `the dog bit the man` and `the man bit the dog` look like the exact same bag of words to attention.
    An LSTM got word order for free by reading one word after another. Here we'll have to supply position information explicitly.

    ## What's next

    You have the layer. **Chapter 17** tells it where each word stands, keeps its numbers steady through many layers, stacks it into
    a tiny GPT with about a million parameters, and trains it on the stories.

    ## What you can do now

    - explain why each word needs a separate query, key, and value
    - block future words with a mask and understand how a 2D mask broadcasts to 4D scores
    - explain why scores are divided by $\sqrt{d}$
    - split vectors into heads and rejoin them (and why a single-step `reshape` scrambles the numbers)
    - assemble a full multi-head self-attention layer from those pieces
    - explain why attention alone cannot tell word order

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the Transformer is from *Attention Is All You Need* (Vaswani et al., 2017), where this layer is called
    *scaled dot-product attention* with *multi-head attention*. A layer whose words may look both ways, with no mask, is the kind used in BERT (chapter 18).
    """)
    return


if __name__ == "__main__":
    app.run()
