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

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.course import WORK
    from learn_llm.data import stories
    from learn_llm.leaderboard import record, show
    from learn_llm.plot import chart
    from learn_llm.tokenizer import Tokenizer, story_words

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 · A modern transformer

    Chapter 17's model is the 2017 design. Large models today keep its skeleton, blocks of attention and thinking on the road of `x + f(x)`, but
    almost every part has been swapped for a better one since. This chapter makes five of those swaps, each small, each measured:

    1. **RMSNorm**, a simpler normalisation;
    2. **rotary positions** (RoPE), which replace the position table;
    3. **SwiGLU**, a gated thinking layer;
    4. **AdamW**, a cleaner way to shrink knobs;
    5. a **schedule** for the learning rate.

    Then it adds what every long run needs: **checkpoints**, to save a run and pick it up later. And the model reads **tokens** now, from chapter 19's method.

    ## Tokens, and a fair leaderboard

    The course comes with 2,000 merges trained on the stories exactly as in chapter 19. Training them your way would take a while on this much text, so `Tokenizer`
    makes the same merges with two standard tricks to go faster: it counts words instead of rereading the text, and remembers each word once it has encoded it. Merges never cross the start of a word, so `the` and ` cat` are never glued into one piece.

    One rule keeps the leaderboard fair. Earlier models read the 2,000 commonest words, with every other word as `<unk>`. If a token model had to spell out those rare words while the word models
    just wrote `<unk>`, the comparison would be rigged against it: we measured this, and it costs about 40% more perplexity. So the token model reads exactly the same text,
    rare words included as `<unk>`, and its score is still total surprise divided by the number of **words**, as chapter 19 promised.

    The cell below counts the words exactly as before (`re.findall` lists each run of letters and apostrophes, as chapter 12's `to_words` did). `set(list)` makes a **set**: a collection
    that answers `word in known` instantly, like a dictionary's keys with no values.
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
    **root mean square**, $\sqrt{\text{mean of } x^2}$, then multiply by a learned gain for each number in the vector. (The `1e-6` only guards against dividing by zero.) It's one step less, so a little faster, and works as well.

    ## Exercise 1: rms_norm

    Write `rms_norm(x, gain)` over the last dimension:

    1. `rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)`
    2. `return gain * x / rms`
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
    mo.md(r"""
    ## 2. Rotations

    Chapter 17 gave each position a learned vector, which has two weaknesses: the model knows nothing about position 33 if it only trained on 32, and "word 7 looking at word 5"
    has to be learned separately from "word 27 looking at word 25", though it's the same relationship: two words back. Rotary positions fix both. They need one idea from geometry.

    **Angles, cos and sin.** Draw a circle of radius 1 around a point. Start at the right-hand edge, and walk anticlockwise round the circle. How far round you've
    gone is an **angle**. Your position then has two coordinates, measured from the centre: how far right, called the **cosine** of the angle, and how far up, its **sine**. Negative means left or down. So the point at angle $\theta$
    is $(\cos\theta, \sin\theta)$. At angle 0 you're at $(1, 0)$. A quarter of the way round, you're at the top, $(0, 1)$. We measure angles in **radians**: a full
    circle is $2\pi$ (where $\pi$ is about 3.14), about 6.28, so a quarter turn is $\pi / 2$, about 1.57. NumPy computes them, as `np.cos` and `np.sin`:
    """)
    return


@app.cell
def _():
    [(round(np.cos(angle).item(), 3), round(np.sin(angle).item(), 3)) for angle in [0, np.pi / 2, np.pi, 1.0]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `np.pi` is $\pi$. The four points are the right edge, the top, the left edge, and a point one radian round.

    **Rotating a pair.** To turn a point $(a, b)$ about the centre by an angle $\theta$, the rule from geometry is:

    > $$(a, b) \;\to\; (a\cos\theta - b\sin\theta,\;\; a\sin\theta + b\cos\theta)$$

    Check it on $(1, 0)$: you get $(\cos\theta, \sin\theta)$, the point at angle $\theta$, as it should be. And $(0, 1)$, the top, goes to $(-\sin\theta, \cos\theta)$: a quarter turn ahead of it, as it should.

    ## Exercise 2: rotate

    Write `rotate(a, b, angle)`, giving back the turned pair:

    ```
    return a * np.cos(angle) - b * np.sin(angle), a * np.sin(angle) + b * np.cos(angle)
    ```
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

    **Rotary positions.** So: cut each query and key into pairs of numbers, and turn the pairs of the word at position $p$ by $p$ times some angle. When
    word 7 asks word 5, the scores depend on $7 - 5 = 2$ steps of angle. Word 27 asking word 25 gives exactly the same. Attention now sees relative positions, with no position table at all, and positions beyond the training windows can at least be computed (models still work best on lengths like those they trained on).
    Each pair turns at its own speed, from 1 radian per position down to about 1/10,000 radian, so that both near and far distances can be told apart. (The 10,000 is the conventional choice.) `rope` below does that for whole tensors. It's
    `rotate`, applied to many pairs at once, with the first half of each vector as the `a`s and the second half as the `b`s.
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
    (`x[..., :half]` means "the first half of the last dimension, whatever comes before it". `[:, None]` turns a vector into a column, so that every position meets every speed.)

    ## 3. SwiGLU: thinking with a gate

    Chapter 17's thinking layer is `W2(relu(W1(x)))`. Two changes make it better. First, a smoother bend: **SiLU**, $x \times \sigma(x)$ with chapter 12's sigmoid.
    It's about $x$ for large $x$ and about 0 for very negative $x$, like ReLU, but with no sharp corner. Second, something like chapter 14's gates: a second layer `W3` decides,
    number by number, how much of the first passes. (It isn't squeezed between 0 and 1, so it's gate-like rather than a strict gate.) The $\times$ here is position by position:

    > $$\text{think}(x) = W_2\big(\text{silu}(W_1 x) \times W_3 x\big)$$

    Three matrices instead of two would mean more knobs, so the hidden layer is made $\tfrac{8}{3}$ as wide as the vector instead of 4 times, to keep the size the same.

    ## Exercise 3: swiglu

    Write `swiglu(x, W1, W2, W3)`, where the three are `nn.Linear` layers and `F.silu` is SiLU:

    ```
    return W2(F.silu(W1(x)) * W3(x))
    ```
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
    mo.md(r"""
    ## 4. AdamW

    In chapter 09 you added weight decay to the slope: `slope + decay * weight`. With plain steps, that shrinks every weight a little. But Adam then divides
    the whole thing by each knob's typical slope size, so a knob with big slopes barely shrinks, and one with small slopes shrinks a lot. That's not what decay is meant to do.

    **AdamW** keeps the decay out of Adam's averages and applies it directly to the knob, at the same rate for every knob. This one change is the "W".

    ## Exercise 4: adamw_step

    Write `adamw_step(w, m, v, grad, t, rate, decay)`. It's chapter 09's `adam_step` with a decay term in the step:

    1. `m = 0.9 * m + 0.1 * grad`
    2. `v = 0.999 * v + 0.001 * grad * grad`
    3. Two lines: `m_hat = m / (1 - 0.9 ** t)` and `v_hat = v / (1 - 0.999 ** t)`
    4. `return w - rate * (m_hat / (np.sqrt(v_hat) + 1e-8) + decay * w), m, v`

    The cell after the check compares your step with PyTorch's `torch.optim.AdamW` on chapter 09's two knobs.
    """)
    return


@app.function
def adamw_step(w, m, v, grad, t, rate, decay):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "adamw_step", adamw_step, LOCALE)
    return


@app.cell
def _():
    w_mine, m_mine, v_mine = np.array([1.0, -2.0]), np.zeros(2), np.zeros(2)
    w_torch = torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True)
    optimizer_demo = torch.optim.AdamW([w_torch], lr=0.1, weight_decay=0.5)
    for t_demo in range(1, 6):
        slope = np.array([0.5, -3.0]) * t_demo
        w_mine, m_mine, v_mine = adamw_step(w_mine, m_mine, v_mine, slope, t_demo, 0.1, 0.5)
        w_torch.grad = torch.tensor(slope)
        optimizer_demo.step()
    w_mine, w_torch.detach().numpy()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The two arrays should be identical: the same knobs after 5 steps.

    ## 5. A schedule for the learning rate

    A good learning rate early in training isn't a good one late. At the very start, the knobs are random and the slopes are wild, so big steps can throw training off.
    Near the end, small steps settle into a low point instead of jumping about it. So the rate follows a **schedule**:

    - **warmup**: grow the rate in a straight line from 0 to its peak over the first steps;
    - then **cosine decay**: let it fall smoothly towards a floor, following half a wave of cosine. That cosine starts at 1 and ends at −1, so $\tfrac{1}{2}(1 + \cos(\pi \times \text{progress}))$ goes from 1 to 0 as progress goes from 0 to 1.

    > $$\text{rate} = \text{peak} \times \Big(\text{floor} + (1 - \text{floor}) \times \tfrac{1}{2}\big(1 + \cos(\pi \times \text{progress})\big)\Big)$$

    where progress is the share of the steps after warmup that are done. We use a floor of 0.1: the rate never falls below a tenth of its peak.

    ## Exercise 5: learning_rate

    Write `learning_rate(step, total, warmup, peak)`. `if ...: return ...` leaves the function early.

    1. `if step < warmup:` with `return peak * step / warmup` inside
    2. `progress = (step - warmup) / (total - warmup)`
    3. `return peak * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))` (the 0.9 is $1 - 0.1$, from the formula)

    The chart after the check shows the schedule for 1,000 steps: it rises to 0.003 by step 100, then falls smoothly to 0.0003.
    """)
    return


@app.function
def learning_rate(step, total, warmup, peak):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "learning_rate", learning_rate, LOCALE)
    return


@app.cell
def _():
    chart((0, 1000), (0, 0.0032), "step", "learning rate",
          curve=[(s, learning_rate(s, 1000, 100, 0.003)) for s in range(0, 1001, 5)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Checkpoints

    A real training run lasts weeks, and machines fail. So every so often a run writes a **checkpoint**: everything needed to carry on as if nothing had happened.
    That's more than the knobs. AdamW's averages are part of the state too (chapter 18's `state_dict()` gives them, for an optimizer as for a model), and so is the step count, which the schedule needs.

    ## Exercise 6: checkpoint

    Write `checkpoint(model, optimizer, step)`, giving back a dictionary with all three:

    ```
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    ```

    The cell after the check proves it works. It trains a small model for 20 steps, saves a checkpoint to a file with `torch.save`, carries on for 20 more,
    then makes a brand-new model, loads the checkpoint with `torch.load`, and runs the same last 20 steps. The two models should end identical: the number shown is the biggest gap
    between any pair of matching knobs. (`torch.rand` and `torch.randint` make random practice data, and `folder.mkdir` creates the folder if it's missing.)
    """)
    return


@app.function
def checkpoint(model, optimizer, step):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "checkpoint", checkpoint, LOCALE)
    return


@app.cell
def _():
    folder = WORK / ".models"
    folder.mkdir(parents=True, exist_ok=True)
    X_demo = torch.rand(64, 10)
    Y_demo = torch.randint(0, 3, (64,))

    def small_run(model, optimizer, first, last):
        for s in range(first, last):
            optimizer.zero_grad()
            F.cross_entropy(model(X_demo), Y_demo).backward()
            optimizer.step()

    torch.manual_seed(0)
    straight = nn.Linear(10, 3)
    straight_opt = torch.optim.AdamW(straight.parameters(), lr=0.01)
    small_run(straight, straight_opt, 0, 20)
    torch.save(checkpoint(straight, straight_opt, 20), folder / "demo.pt")
    small_run(straight, straight_opt, 20, 40)

    saved = torch.load(folder / "demo.pt")
    resumed = nn.Linear(10, 3)
    resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    resumed.load_state_dict(saved["model"])
    resumed_opt.load_state_dict(saved["optimizer"])
    small_run(resumed, resumed_opt, saved["step"], 40)
    torch.max(torch.abs(straight.weight - resumed.weight)).item()
    return (folder,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Exactly 0. (The checkpoint lives in `my-work/.models`, next to your copies of the lessons.)

    ## The modern model

    Here are all five swaps together. Compare `ModernBlock` with chapter 17's `Block`: the same two `x + f(x)` steps, with `rms_norm` and `swiglu` in place of layer norm and ReLU, and `rope`
    applied to queries and keys. `nn.Parameter` makes a tensor into a knob of its own, here each gain. The model has no position table. Its last layer and attention layers have no biases, which modern models usually drop.
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

        def forward(self, x):
            batch, words, d = x.shape
            y = rms_norm(x, self.gain1)

            def heads_of(layer):
                return layer(y).reshape(batch, words, self.heads, d // self.heads).transpose(1, 2)

            blended = attend(rope(heads_of(self.query)), rope(heads_of(self.key)), heads_of(self.value))
            x = x + self.mix(blended.transpose(1, 2).reshape(batch, words, d))
            return x + swiglu(rms_norm(x, self.gain2), self.W1, self.W2, self.W3)

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


@app.function
def windows(ids, length):
    n = (len(ids) - 1) // length
    return ids[:n * length].reshape(n, length), ids[1:n * length + 1].reshape(n, length)


@app.function
def train_modern(model, X, Y, epochs, batch_size, peak):
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak, weight_decay=0.1)
    total = epochs * ((len(X) + batch_size - 1) // batch_size)
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), batch_size):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, total, 100, peak)
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            S = model(X[batch])
            F.cross_entropy(S.reshape(-1, S.shape[-1]), Y[batch].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step = step + 1
    return optimizer, step


@app.function
def per_word_perplexity(model, X, Y, word_count):
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(X), 64):
            S = model(X[start:start + 64])
            total = total + F.cross_entropy(S.reshape(-1, S.shape[-1]), Y[start:start + 64].reshape(-1), reduction="sum").item()
    return np.exp(total / word_count)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `train_modern` is chapter 17's loop with AdamW and the schedule: before every step it sets the rate for that step (`optimizer.param_groups` is where the rate lives).
    `per_word_perplexity` adds up the surprise over all test tokens and divides by the **words**, as promised. Windows are 64 tokens long, a little over 60 words, since a token
    is nearly a word.

    One honest detail: cutting the text into windows drops a few tokens at the very end, so the words are counted over the tokens actually scored. The difference is a few tokens out of 167,000.

    Finish all six exercises first: the training uses them all. The button trains a modern model of the same size as chapter 17's: 128 numbers per token, 4 heads, 3 blocks. It takes about 3 to 7 minutes,
    then saves a checkpoint to `my-work/.models/ch20.pt`, so the run can be loaded again later. Until it finishes, the cells that need it say `Ancestor stopped`. That's normal.
    The cell shows the knobs and the per-word perplexity.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the modern transformer (3 to 7 minutes)")
    train_button
    return (train_button,)


@app.cell
def _(ModernGPT, folder, test_tokens, tokenizer, train_button, train_tokens, word_count):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    X_np, Y_np = windows(train_tokens, 64)
    X_test_np, Y_test_np = windows(test_tokens, 64)
    torch.manual_seed(0)
    model = ModernGPT(tokenizer.size, 128, 4, 3)
    optimizer, steps_done = train_modern(model, torch.tensor(X_np), torch.tensor(Y_np), 2, 32, 0.003)
    torch.save(checkpoint(model, optimizer, steps_done), folder / "ch20.pt")
    scored_words = word_count * Y_test_np.size / (len(test_tokens) - 1)
    modern_score = per_word_perplexity(model, torch.tensor(X_test_np), torch.tensor(Y_test_np), scored_words)
    sum([p.numel() for p in model.parameters()]), round(modern_score.item(), 2)
    return model, modern_score


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 1.17 million knobs, nearly the same as chapter 17's model, and a per-word perplexity of about 19. Here's the leaderboard:
    """)
    return


@app.cell(hide_code=True)
def _(modern_score):
    record("modern", modern_score)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 19 against chapter 17's 23. How much of that is the tokens, and how much the five swaps? We trained chapter 17's design on these same tokens, with the same time and
    the same text: it scored about 23.4, just like on words. So the tokens alone change little here, since with `<unk>` kept this text is almost one token per word. The gain comes from the swaps.

    And without `<unk>`? We also trained both designs on the full text, with every rare word spelled out in tokens. Both scored about 40% worse per word, about 27 for the modern one: spelling out
    `xylophone` costs far more than saying "some rare word". That's the honest price of a model that can write any word, and why our leaderboard keeps `<unk>`.

    Here's some writing from the new model, all lower case and without punctuation, like the text it learned from. `tokenizer.decode` turns tokens back into text, and `[:-1]` drops the `<end>` after the starting words:
    """)
    return


@app.cell
def _(model, tokenizer):
    torch.manual_seed(1)
    written = tokenizer.encode([["once", "upon", "a", "time"]])[:-1]
    with torch.no_grad():
        for step_write in range(60):
            S_write = model(torch.tensor([written[-64:]]))
            written.append(torch.multinomial(torch.softmax(S_write[0, -1] / 0.8, dim=0), 1).item())
    tokenizer.decode(written)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What's next

    We've made the model better for the same cost. Next, the cost itself. **Chapter 22** measures where the time and memory go when a model trains, counts the arithmetic
    in a training run, and makes attention several times faster with one change.

    ## What you can do now

    - normalise with RMSNorm
    - explain angles, cosine and sine, and rotate a pair of numbers
    - explain why rotary positions give attention relative positions
    - write a gated SwiGLU layer
    - explain AdamW's decay, and match PyTorch's AdamW
    - write a warmup-and-cosine schedule
    - save and resume a run exactly with a checkpoint

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** RMSNorm is from Zhang and Sennrich (2019), RoPE from Su et al. (*RoFormer*, 2021), SwiGLU from Shazeer (*GLU Variants Improve Transformer*, 2020),
    and AdamW from Loshchilov and Hutter (2017). Models such as LLaMA combine exactly these choices. Many models also *tie* the first and last layers, sharing one table of token vectors.
    """)
    return


if __name__ == "__main__":
    app.run()
