import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="20 · A modern transformer")

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
    from learn_llm.diagrams import circle
    from learn_llm.tokenizer import Tokenizer, story_words

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 · A modern transformer

    Chapter 17's model is the 2017 design. Large models today keep its skeleton, blocks of attention and thinking on the road of $x + f(x)$, but
    almost every part has been swapped for a better one since. This chapter makes three of those swaps inside the block:

    1. **RMSNorm**, a simpler normalisation;
    2. **rotary positions** (RoPE), which replace the position table;
    3. **SwiGLU**, a gated thinking layer.

    You'll build each piece, put them together into a block, and check the new model's shapes and size. **Chapter 21** then trains it, with two more swaps
    that belong to the training loop. And the model reads **tokens** now, from chapter 19's method.

    ## Tokens, and a fair leaderboard

    The course comes with 2,000 merges trained on the stories exactly as in chapter 19. Training them your way would take a while on this much text, so `Tokenizer`
    makes the same merges with two standard tricks to go faster: it counts words instead of rereading the text, and remembers each word once it has encoded it. Merges never cross the start of a word, so `the` and ` cat` are never glued into one piece.

    One rule keeps the leaderboard fair. Earlier models read the 2,000 commonest words, with every other word as `<unk>`. If a token model had to spell out those rare words while the word models
    just wrote `<unk>`, the comparison would be rigged against it: we measured this, and it costs about 40% more perplexity. So the token model reads exactly the same text,
    rare words included as `<unk>`, and its score is still total surprise divided by the number of **words**, as chapter 19 promised.

    /// admonition | 🧰 Toolbox: sets
    `set(items)` makes a **set** from a list: a collection that answers `word in known` instantly, like a dictionary's keys with no values.
    `"dog" in set(["dog", "cat"])` is `True`.
    ///

    The cell below counts the words exactly as before (`re.findall` lists each run of letters and apostrophes, as chapter 12's `to_words` did), and takes a few seconds.
    """)
    return


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = {}
    for story in train_stories:
        for word in re.findall(r"[a-z']+", story.lower()):
            counts[word] = counts.get(word, 0) + 1
    known = set(sorted(counts, key=counts.get, reverse=True)[:2000])
    tokenizer = Tokenizer.load()
    train_tokens = np.array(tokenizer.encode(story_words(train_stories, known)))
    test_words = story_words(test_stories, known)
    test_tokens = np.array(tokenizer.encode(test_words))
    word_count = sum([len(words) for words in test_words]) + len(test_words)
    tokenizer.size, len(train_tokens), len(test_tokens), word_count
    return test_tokens, tokenizer, train_tokens, word_count


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    2,258 token ids (256 bytes, 2,000 merges, `<end>` and `<unk>`). The test stories become about 167,000 tokens for 159,000 words (each story's `<end>` counts as a word, as before): a little over one token per word,
    because the 2,000 words are mostly single tokens and only a few get cut into pieces.

    ## 1. RMSNorm

    Chapter 17's layer norm subtracts the mean, then divides by the typical size. Researchers found the subtraction does little, so **RMSNorm** skips it: divide each vector by its
    **root mean square**, then multiply by a learned **gain** for each number in the vector:

    > $$\text{rms} = \sqrt{\text{mean of } x^2 + 0.000001} \qquad \text{rms\_norm}(x) = \text{gain} \times \frac{x}{\text{rms}}$$

    The tiny number only guards against dividing by zero. It's one step less than layer norm, so a little faster, and works as well.

    ## By hand: one vector

    Normalise $x = [1, 7]$ with a gain of 1 for both numbers. Work out the mean of the squares, the root mean square, and the two results. (Leave out the tiny number.)
    """)
    return


@app.cell(hide_code=True)
def _():
    rms_boxes = hand.boxes(['mean of the squares', 'rms', 'first result', 'second result'], step=0.01)
    return (rms_boxes,)


@app.cell(hide_code=True)
def _(rms_boxes):
    hand.marked(rms_boxes, {'mean of the squares': 25, 'rms': 5, 'first result': 0.2, 'second result': 1.4}, {
        'mean of the squares': r"""$(1 \times 1 + 7 \times 7) / 2$""",
        'rms': r"""the square root of your first answer""",
        'first result': r"""$1 / \text{rms}$""",
        'second result': r"""$7 / \text{rms}$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The results, 0.2 and 1.4, have a root mean square of exactly 1: that's what the division is for. The signs and the proportions of the vector are kept, and its mean is not moved to 0.

    ## Exercise 1: rms_norm

    Write `rms_norm(x, gain)`. `x` is a tensor of any shape whose last dimension has `d` numbers, and `gain` has shape `(d,)`. Each vector along the last dimension
    is normalised by the formula above, so the result has the shape of `x`. In code the tiny number is `1e-6`, and it goes inside the square root, `torch.sqrt`.

    Your hand example is one of the checks: `x = [[1.0, 7.0]]` with `gain = [1.0, 1.0]` gives `[[0.2, 1.4]]`.
    """)
    return


@app.function
def rms_norm(x, gain):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rms_norm", rms_norm, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""It's chapter 17's `layer_norm` with less in it: one `torch.mean` along the last dimension, of `x * x` this time, and no subtraction. The mean needs `dim=-1` and `keepdim=True`, so that each vector is divided by its own number.""",
        r"""It's 2 lines. The first makes `rms` with `torch.sqrt` around the mean plus `1e-6`. The second gives back the gain times `x`, divided by `rms`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Rotations

    Chapter 17 gave each position a learned vector, which has two weaknesses: the model knows nothing about position 33 if it only trained on 32, and "word 7 looking at word 5"
    has to be learned separately from "word 27 looking at word 25", though it's the same relationship: two words back. Rotary positions fix both. They need one idea from geometry.

    **Angles, cos and sin.** Draw a circle of radius 1 around a point. Start at the right-hand edge, the point $(1, 0)$, and walk anticlockwise round the circle. How far round you've
    gone is an **angle**, written $\theta$ (the Greek letter theta). Your position then has two coordinates, measured from the centre: how far right, called the **cosine** of the angle, and how far up, its **sine**. Negative means left or down.
    """)
    return


@app.cell(hide_code=True)
def _():
    circle(0.9, 'A circle of radius 1. A point has walked the angle theta round from (1, 0). Its distance to the right of the centre is cos theta, and its height above the centre is sin theta.')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So the point at angle $\theta$ is $(\cos\theta, \sin\theta)$. At angle 0 you're at $(1, 0)$. A quarter of the way round, you're at the top, $(0, 1)$. We measure angles in **radians**: a full
    circle is $2\pi$ (where $\pi$ is about 3.14), about 6.28, so a quarter turn is $\pi / 2$, about 1.57.

    /// admonition | 🧰 Toolbox: cos, sin and π in NumPy
    `np.cos(angle)` and `np.sin(angle)` compute the two, for one angle or for a whole array of them, and `np.pi` is $\pi$.
    ///
    """)
    return


@app.cell
def _():
    [(round(np.cos(angle).item(), 3), round(np.sin(angle).item(), 3)) for angle in [0, np.pi / 2, np.pi, 0.9]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The four points are the right edge, the top, the left edge, and the point in the picture, 0.9 radians round. Here they are as a table to use below:

    | turn | angle $\theta$ | $\cos\theta$ | $\sin\theta$ |
    |---|---|---|---|
    | none | 0 | 1 | 0 |
    | a quarter | $\pi / 2$ | 0 | 1 |
    | a half | $\pi$ | −1 | 0 |

    **Rotating a pair.** To turn any point $(a, b)$ about the centre by an angle $\theta$, the rule from geometry is:

    > $$(a, b) \;\to\; (a\cos\theta - b\sin\theta,\;\; a\sin\theta + b\cos\theta)$$

    ## By hand: turn three points

    Use the rule and the table. Turn $(1, 0)$ by a quarter turn. Turn $(0, 1)$ by a quarter turn. And turn $(3, 4)$ by a half turn.
    """)
    return


@app.cell(hide_code=True)
def _():
    turn_boxes = hand.boxes(['(1, 0): new first', '(1, 0): new second', '(0, 1): new first', '(0, 1): new second', '(3, 4): new first', '(3, 4): new second'])
    return (turn_boxes,)


@app.cell(hide_code=True)
def _(turn_boxes):
    hand.marked(turn_boxes, {'(1, 0): new first': 0, '(1, 0): new second': 1, '(0, 1): new first': -1, '(0, 1): new second': 0, '(3, 4): new first': -3, '(3, 4): new second': -4}, {
        '(1, 0): new first': r"""$1 \times \cos\theta - 0 \times \sin\theta$, with $\cos\theta = 0$""",
        '(1, 0): new second': r"""$1 \times \sin\theta + 0 \times \cos\theta$, with $\sin\theta = 1$""",
        '(0, 1): new first': r"""$0 \times 0 - 1 \times 1$""",
        '(0, 1): new second': r"""$0 \times 1 + 1 \times 0$""",
        '(3, 4): new first': r"""$3 \times (-1) - 4 \times 0$""",
        '(3, 4): new second': r"""$3 \times 0 + 4 \times (-1)$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Check them against the picture. The right edge went to the top. The top went to the left edge, a quarter turn ahead of it. And a half turn sends any point to the opposite side of the centre.

    ## Exercise 2: rotate

    Write `rotate(a, b, angle)`, the rule above with NumPy's `np.cos` and `np.sin`. It gives back the turned pair: the new first number, then the new second.
    `a` and `b` may be single numbers or arrays of the same shape.

    > `rotate(1.0, 0.0, np.pi / 2)` gives `(0.0, 1.0)`, up to rounding
    """)
    return


@app.function
def rotate(a, b, angle):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rotate", rotate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two expressions, one for each half of the rule, given back together with a comma between them. Both use the old `a` and the old `b`.""",
        r"""Mind the minus sign: it's in the first half only. If you store the new first number in `a` before working out the second, the second uses the wrong `a`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now the property that makes rotations perfect for positions. Recall that an attention score is the dot product of a query and a key. Take two pairs, `q` and `k`. Turn `q` by one angle and `k` by another, and take their dot product.
    The result depends only on the **difference** between the angles, not on the angles themselves. Turn both by an extra 1.5 and nothing changes:
    """)
    return


@app.cell
def _():
    q = (0.8, -0.3)
    k = (0.5, 0.9)

    def turned_dot(angle_q, angle_k):
        qa, qb = rotate(q[0], q[1], angle_q)
        ka, kb = rotate(k[0], k[1], angle_k)
        return round((qa * ka + qb * kb).item(), 6)

    turned_dot(0.7, 0.2), turned_dot(2.2, 1.7)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same number, because turning both pairs together doesn't change how they sit relative to each other.

    ## Rotary positions

    So: cut each query and key into pairs of numbers, and turn the pairs of the word at position $p$ by $p$ times some angle. When
    word 7 asks word 5, the scores depend on $7 - 5 = 2$ steps of angle. Word 27 asking word 25 gives exactly the same. Attention now sees relative positions, with no position table at all, and positions beyond the training windows can at least be computed (models still work best on lengths like those they trained on).

    Each pair turns at its own **speed**, in radians per position: fast pairs tell near positions apart, and slow pairs tell far ones apart. For a vector of 4 numbers $[x_0, x_1, x_2, x_3]$ there are 2 pairs.
    The first half holds the first numbers of the pairs and the second half the second numbers, so the pairs are $(x_0, x_2)$ and $(x_1, x_3)$:

    | pair | its two numbers | turned by |
    |---|---|---|
    | 0 | $(x_0, x_2)$ | $\text{position} \times \text{speed}_0$ |
    | 1 | $(x_1, x_3)$ | $\text{position} \times \text{speed}_1$ |

    ## Exercise 3: rope_one

    Write `rope_one(x, position, speeds)` for one word's vector. `x` is a NumPy vector with an even number of entries, `position` is a whole number, and `speeds` is a NumPy vector half as long as `x`.
    It gives back a vector of the same size as `x`: all the turned first numbers, then all the turned second numbers.

    Your `rotate` does the turning, for all the pairs at once: hand it the first half of `x` as `a`, the second half as `b`, and an array of angles. `np.concatenate` (chapter 14) joins the two halves again.

    > `rope_one(np.array([1.0, 0.0, 0.0, 1.0]), 1, np.array([np.pi / 2, np.pi]))` gives `[0, 0, 1, -1]`, up to rounding

    There, pair 0 is $(1, 0)$, turned a quarter turn to $(0, 1)$, and pair 1 is $(0, 1)$, turned a half turn to $(0, -1)$.
    """)
    return


@app.function
def rope_one(x, position, speeds):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rope_one", rope_one, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The number of pairs is `len(x) // 2`. The first half of `x` is a slice up to that number, and the second half a slice from it. The angles are `position * speeds`: one angle for each pair.""",
        r"""It can be 3 lines: `half = len(x) // 2`, then `a, b = rotate(...)` with the two slices and the angles, then give back `np.concatenate([a, b])`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The cell below checks the property on your function, with the speeds the real thing uses: from 1 radian per position for the fastest pair down to about 1/10,000 for the slowest. (The 10,000 is the conventional choice.)
    It turns a query and a key of 8 numbers, and takes their dot product twice: for positions 7 and 5, and for positions 27 and 25.
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(0)
    q_demo, k_demo = rng.random(8), rng.random(8)
    speeds_demo = 1 / 10000 ** (np.arange(4) / 4)
    near = rope_one(q_demo, 7, speeds_demo) @ rope_one(k_demo, 5, speeds_demo)
    far = rope_one(q_demo, 27, speeds_demo) @ rope_one(k_demo, 25, speeds_demo)
    round(near.item(), 6), round(far.item(), 6)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same score: two words back is two words back, wherever it happens.

    `rope` below is your `rope_one` for whole tensors: every position of every text at once, in PyTorch, with the positions and speeds worked out inside. You don't need to write it, but you can read it with these two tools:

    /// admonition | 🧰 Toolbox: two indexing tools
    - `x[..., :half]` means "the first `half` entries of the last dimension, whatever dimensions come before it". The three dots stand for all of those.
    - `v[:, None]` turns a vector of shape `(n,)` into a column of shape `(n, 1)`, and `v[None, :]` into a row of shape `(1, n)`. A column times a row gives a table of shape `(n, m)`:
      here, every position meets every speed.
    ///
    """)
    return


@app.function
def rope(x):
    words, size = x.shape[-2], x.shape[-1]
    half = size // 2
    speeds = 1 / 10000 ** (torch.arange(half) / half)
    angles = torch.arange(words)[:, None] * speeds[None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * torch.cos(angles) - b * torch.sin(angles), a * torch.sin(angles) + b * torch.cos(angles)], dim=-1)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: read rope

    Suppose `x` has shape `(2, 4, 16, 32)`: 2 texts, 4 heads, 16 words, 32 numbers per head. How many pairs does each vector have? What are the two sizes of `angles`, its rows and its columns?
    And which position's vectors does `rope` leave exactly as they were?
    """)
    return


@app.cell(hide_code=True)
def _():
    rope_boxes = hand.boxes(['pairs', 'rows of angles', 'columns of angles', 'position left unchanged'])
    return (rope_boxes,)


@app.cell(hide_code=True)
def _(rope_boxes):
    hand.marked(rope_boxes, {'pairs': 16, 'rows of angles': 16, 'columns of angles': 16, 'position left unchanged': 0}, {
        'pairs': r"""`half` is the last size divided by 2""",
        'rows of angles': r"""one row for each position: `words` is `x.shape[-2]`""",
        'columns of angles': r"""one column for each speed, and there are `half` speeds""",
        'position left unchanged': r"""which position gives an angle of 0 for every speed?""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. SwiGLU: thinking with a gate

    Chapter 17's thinking layer is $W_2(\text{relu}(W_1 x))$. Two changes make it better. First, a smoother bend: **SiLU**, $x \times \sigma(x)$ with chapter 12's sigmoid.
    It's about $x$ for large $x$ and about 0 for very negative $x$, like ReLU, but with no sharp corner. In PyTorch it's `F.silu`. Second, something like chapter 14's gates: a third layer $W_3$ decides,
    number by number, how much of the first passes. (It isn't squeezed between 0 and 1, so it's gate-like rather than a strict gate.) The $\times$ here is position by position:

    > $$\text{think}(x) = W_2\big(\text{silu}(W_1 x) \times W_3 x\big)$$

    Three matrices instead of two would mean more knobs, so the hidden layer is made $\tfrac{8}{3}$ as wide as the vector instead of 4 times, to keep the size about the same.

    ## By hand: count the knobs

    Our model's vectors have $d = 128$ numbers. Chapter 17's thinking layer had two matrices, $128 \times 512$ and $512 \times 128$. The new hidden width is $8 \times 128 / 3$, rounded down to a whole number.
    What is it? How many knobs do the three new matrices have together, with no biases? And how many did chapter 17's two have, also leaving out biases?
    """)
    return


@app.cell(hide_code=True)
def _():
    knob_boxes = hand.boxes(['hidden width', 'knobs in the three matrices', 'knobs in the old two'])
    return (knob_boxes,)


@app.cell(hide_code=True)
def _(knob_boxes):
    hand.marked(knob_boxes, {'hidden width': 341, 'knobs in the three matrices': 130944, 'knobs in the old two': 131072}, {
        'hidden width': r"""$1024 / 3$ is 341.33…""",
        'knobs in the three matrices': r"""each of the three has $128 \times 341$ knobs""",
        'knobs in the old two': r"""each of the two has $128 \times 512$ knobs""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Within a tenth of a percent of each other: a gate for the same price.

    ## Exercise 4: swiglu

    Write `swiglu(x, W1, W2, W3)`, the formula above. The three are `nn.Linear` layers: `W1` and `W3` take `d` numbers and give the hidden width, and `W2` takes the hidden width and gives `d`.
    `x` has `d` numbers in its last dimension, and so does the result.
    """)
    return


@app.function
def swiglu(x, W1, W2, W3):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "swiglu", swiglu, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""A layer is called like a function, as in `W1(x)`. Read the formula from the inside out: two layers look at `x`, one result goes through `F.silu`, the two are multiplied with `*`, and `W2` takes the product.""",
        r"""It's one line, and it starts `return W2(`. The SiLU goes on `W1(x)` only: `W3(x)` is the gate, and stays as it is.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The modern block

    Now the three swaps go into chapter 17's block. Its plan doesn't change: two parts, each on a road that passes it by, each with a normalisation first.

    | step | chapter 17 | now |
    |---|---|---|
    | normalise | layer norm | `rms_norm`, with the block's first gain |
    | look | attention, with a position table before the first block | attention, with `rope` on the queries and keys |
    | add | $x + \text{look}$ | the same |
    | normalise | layer norm | `rms_norm`, with the block's second gain |
    | think | $W_2(\text{relu}(W_1 x))$ | `swiglu` |
    | add | $x + \text{think}$ | the same |

    ## Exercise 5: modern_block_step

    Write `modern_block_step(x, gain1, look, gain2, W1, W2, W3)`, the right-hand column. `x` has shape `(batch, words, d)`. `look` is a function that takes a tensor of that shape and gives back
    attention's result, of the same shape: call it like any function. The gains have shape `(d,)`, and the three layers are SwiGLU's. It gives back the block's output, of the shape of `x`.

    As in chapter 17, the second half starts from the result of the first: what gets normalised the second time is the *new* `x`.
    """)
    return


@app.function
def modern_block_step(x, gain1, look, gain2, W1, W2, W3):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "modern_block_step", modern_block_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two lines, one for each half of the table. Each is `x` plus something: first `look` of the normalised `x`, then `swiglu` of the normalised `x`. Use your `rms_norm` and `swiglu`.""",
        r"""The first line is `x = x + look(rms_norm(x, gain1))`. The second gives back the new `x` plus `swiglu` of `rms_norm(x, gain2)` with the three layers.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here are the block and the model as PyTorch modules. `ModernBlock` holds the knobs, its `look` is chapter 17's attention over several heads with `rope` applied to the queries and keys, and its `forward` is your function.
    `attend` is chapter 16's, with the mask built in. Three things to notice:

    - `nn.Parameter(tensor)` makes a tensor into a knob of its own, so that training moves it. Here each gain is one, starting as all 1s.
    - The model has no position table: compare `ModernGPT` with chapter 17's `TinyGPT`.
    - Its last layer and its attention layers have no biases, which modern models usually drop.

    Until all five exercises pass, the cell after these shows an error. That's normal.
    """)
    return


@app.function
def attend(Q, K, V):
    words = Q.shape[-2]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones(words, words, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


@app.cell
def _():
    class ModernBlock(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.heads = heads
            self.gain1 = nn.Parameter(torch.ones(d))
            self.gain2 = nn.Parameter(torch.ones(d))
            self.query = nn.Linear(d, d, bias=False)
            self.key = nn.Linear(d, d, bias=False)
            self.value = nn.Linear(d, d, bias=False)
            self.mix = nn.Linear(d, d, bias=False)
            hidden = 8 * d // 3
            self.W1 = nn.Linear(d, hidden, bias=False)
            self.W2 = nn.Linear(hidden, d, bias=False)
            self.W3 = nn.Linear(d, hidden, bias=False)

        def look(self, y):
            batch, words, d = y.shape

            def heads_of(layer):
                return layer(y).reshape(batch, words, self.heads, d // self.heads).transpose(1, 2)

            blended = attend(rope(heads_of(self.query)), rope(heads_of(self.key)), heads_of(self.value))
            return self.mix(blended.transpose(1, 2).reshape(batch, words, d))

        def forward(self, x):
            return modern_block_step(x, self.gain1, self.look, self.gain2, self.W1, self.W2, self.W3)

    class ModernGPT(nn.Module):
        def __init__(self, vocab_size, d, heads, layers):
            super().__init__()
            self.words = nn.Embedding(vocab_size, d)
            self.blocks = nn.Sequential(*[ModernBlock(d, heads) for i in range(layers)])
            self.gain = nn.Parameter(torch.ones(d))
            self.out = nn.Linear(d, vocab_size, bias=False)

        def forward(self, ids):
            return self.out(rms_norm(self.blocks(self.words(ids)), self.gain))
    return (ModernGPT,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Does it fit together?

    No training yet: first make sure the pieces fit. The cell builds a model of the same size as chapter 17's, with 128 numbers per token, 4 heads and 3 blocks,
    and feeds it the first 64 test tokens twice over, as a batch of 2. It shows the shape of the scores, the number of knobs, and the loss.

    Before you look: an untrained model knows nothing, so it should be about as unsure as choosing among all 2,258 tokens. By chapter 13, the surprise of that is $\log 2258$, about 7.7.
    """)
    return


@app.cell
def _(ModernGPT, test_tokens, tokenizer):
    torch.manual_seed(0)
    model = ModernGPT(tokenizer.size, 128, 4, 3)
    ids_demo = torch.tensor(np.array([test_tokens[:64], test_tokens[:64]]))
    with torch.no_grad():
        S_demo = model(ids_demo)
        loss_demo = F.cross_entropy(S_demo[:, :-1].reshape(-1, tokenizer.size), ids_demo[:, 1:].reshape(-1))
    S_demo.shape, sum([p.numel() for p in model.parameters()]), round(loss_demo.item(), 2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The scores have shape `(2, 64, 2258)`: for each of 2 texts, at each of 64 positions, a score for every token. There are about 1.17 million knobs, nearly the same as chapter 17's model.
    And the loss is about 7.9, close to the predicted 7.7. (It's a little above, because random scores aren't perfectly even.) That's a good sign that nothing is wired up wrongly: a much bigger number at the start usually means a mistake.

    ## What's next

    The model is built, and it has never seen a story. **Chapter 21** trains it properly: with **AdamW**, a cleaner way to shrink knobs, a **schedule** for the learning rate,
    and **checkpoints**, so that a run can be saved and picked up later. Then it goes on the leaderboard.

    ## What you can do now

    - explain why the token model must read the same text as the word models
    - normalise with RMSNorm, by hand and in code
    - explain angles, cosine and sine, and rotate a pair of numbers
    - explain why rotary positions give attention relative positions, and apply them to a vector
    - write a gated SwiGLU layer, and count its knobs
    - put the pieces together into a modern block, and check an untrained model's shape and loss

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** RMSNorm is from Zhang and Sennrich (2019), RoPE from Su et al. (*RoFormer*, 2021), and SwiGLU from Shazeer (*GLU Variants Improve Transformer*, 2020).
    Models such as LLaMA combine exactly these choices. Many models also *tie* the first and last layers, sharing one table of token vectors.
    """)
    return


if __name__ == "__main__":
    app.run()
