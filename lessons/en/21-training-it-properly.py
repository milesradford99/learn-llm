import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="21 · Training it properly")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
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
    # 21 · Training it properly

    Chapter 20 built a modern transformer and stopped before training it. A good design isn't enough: how a model is trained matters as much as what it's made of.
    This chapter adds the three things every serious training run has:

    1. **AdamW**, a cleaner way to shrink knobs;
    2. a **schedule** for the learning rate;
    3. **checkpoints**, to save a run and pick it up later.

    Then you'll train chapter 20's model with all three, and put it on the leaderboard.

    **From earlier chapters.** The cells below hold chapter 20's model and data, with their code folded away: `rms_norm`, `rope`, `swiglu`, the block and the model are the ones you wrote there,
    and the tokens are loaded as before, which takes a few seconds. Nothing in them is new.
    """)
    return


@app.function(hide_code=True)
def rms_norm(x, gain):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    return gain * x / rms


@app.function(hide_code=True)
def rope(x):
    words, size = x.shape[-2], x.shape[-1]
    half = size // 2
    speeds = 1 / 10000 ** (torch.arange(half) / half)
    angles = torch.arange(words)[:, None] * speeds[None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * torch.cos(angles) - b * torch.sin(angles), a * torch.sin(angles) + b * torch.cos(angles)], dim=-1)


@app.function(hide_code=True)
def swiglu(x, W1, W2, W3):
    return W2(F.silu(W1(x)) * W3(x))


@app.function(hide_code=True)
def attend(Q, K, V):
    words = Q.shape[-2]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones(words, words, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


@app.cell(hide_code=True)
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
            x = x + self.look(rms_norm(x, self.gain1))
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


@app.cell(hide_code=True)
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
    ## 1. AdamW

    In chapter 09 you added weight decay to the slope: $\text{slope} + \text{decay} \times w$. With plain steps, that shrinks every weight a little. But Adam then divides
    the whole thing by each knob's typical slope size, so a knob with big slopes barely shrinks, and one with small slopes shrinks a lot. That's not what decay is meant to do.

    **AdamW** keeps the decay out of Adam's averages and applies it directly to the knob, at the same rate for every knob. This one change is the "W".
    Chapter 09's Adam is unchanged in its first two lines, and only the step is new:

    > $$m = 0.9 \times m + 0.1 \times g \qquad v = 0.999 \times v + 0.001 \times g^2$$

    > $$\hat{m} = \frac{m}{1 - 0.9^t} \qquad \hat{v} = \frac{v}{1 - 0.999^t}$$

    > $$w \leftarrow w - \text{rate} \times \Big(\frac{\hat{m}}{\sqrt{\hat{v}} + 0.00000001} + \text{decay} \times w\Big)$$

    ## By hand: one step, three ways

    A knob sits at $w = 2$. It's the very first step, the slope is $g = 0.5$, the rate is 0.1 and the decay is 0.5. Chapter 09 showed that on the first step $\hat{m} / \sqrt{\hat{v}}$ is
    exactly $+1$ for a positive slope, whatever its size. Work out the knob after one step, three ways:

    - plain **Adam**, with no decay at all;
    - Adam with decay **added to the slope**, chapter 09's way: the slope becomes $g + \text{decay} \times w$, and then Adam runs as usual;
    - **AdamW**, by the formula above.
    """)
    return


@app.cell(hide_code=True)
def _():
    decay_boxes = hand.boxes(['Adam', 'decay added to the slope', 'AdamW'], step=0.01)
    return (decay_boxes,)


@app.cell(hide_code=True)
def _(decay_boxes):
    hand.marked(decay_boxes, {'Adam': 1.9, 'decay added to the slope': 1.9, 'AdamW': 1.8}, {
        'Adam': r"""$2 - 0.1 \times 1$""",
        'decay added to the slope': r"""the slope becomes $0.5 + 0.5 \times 2 = 1.5$. It's still positive, so $\hat{m} / \sqrt{\hat{v}}$ is still $+1$.""",
        'AdamW': r"""$2 - 0.1 \times (1 + 0.5 \times 2)$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Added to the slope, the decay vanished: Adam's division swallowed it, and the knob landed exactly where plain Adam put it. In AdamW the decay
    does its job, pulling the knob 0.1 closer to zero on top of Adam's step.

    ## Exercise 1: adamw_step

    Write `adamw_step(w, m, v, grad, t, rate, decay)`, the three formulas above. It's chapter 09's `adam_step` with one more input and one more term.
    `w`, `m`, `v` and `grad` are NumPy arrays of the same shape, and `t` counts the steps from 1. Like `adam_step`, it gives back three things: the new knobs, and the
    uncorrected `m` and `v`, because those are what the next step needs. In code, a power is `**` and the tiny number is `1e-8`.

    Your hand example is one of the checks: `w = [2.0]` and `grad = [0.5]` at `t = 1`, with rate 0.1 and decay 0.5, give new knobs `[1.8]`.
    The cell after the check compares your step with PyTorch's `torch.optim.AdamW` over five steps.
    """)
    return


@app.function
def adamw_step(w, m, v, grad, t, rate, decay):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "adamw_step", adamw_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Five lines, in the order of the formulas: the new `m`, the new `v`, the two corrected averages under other names (so that `m` and `v` stay uncorrected), then the `return` with three things.""",
        r"""The decay term goes inside the brackets that `rate` multiplies, and outside the division: `rate * (m_hat / (np.sqrt(v_hat) + 1e-8) + decay * w)`. If your result is 1.9 for the hand example, the decay is missing or in the wrong place.""",
    ], LOCALE)
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
    The two arrays should be identical: the same knobs after 5 steps. From here on we use PyTorch's.

    ## 2. A schedule for the learning rate

    A good learning rate early in training isn't a good one late. At the very start, the knobs are random and the slopes are wild, so big steps can throw training off.
    Near the end, small steps settle into a low point instead of jumping about it. So the rate follows a **schedule**. Here's the one we'll use, for a run of 1,000 steps with a peak of 0.003:
    """)
    return


@app.cell(hide_code=True)
def _():
    _curve = [(s, 0.003 * s / 100 if s < 100 else 0.003 * (0.1 + 0.45 * (1 + float(np.cos(np.pi * (s - 100) / 900))))) for s in range(0, 1001, 5)]
    chart((0, 1000), (0, 0.0032), "step", "learning rate", curve=_curve)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It has two parts:

    - **warmup**: for the first steps, here 100, the rate grows in a straight line from 0 to its peak:

    > $$\text{rate} = \text{peak} \times \frac{\text{step}}{\text{warmup}}$$

    - then **cosine decay**: it falls smoothly towards a floor, following half a wave of cosine. Call **progress** the share of the steps after warmup that are done, from 0 to 1.
      As progress goes from 0 to 1, $\cos(\pi \times \text{progress})$ goes from 1 to −1 (chapter 20's table), so $\tfrac{1}{2}(1 + \cos(\pi \times \text{progress}))$ goes from 1 to 0:

    > $$\text{progress} = \frac{\text{step} - \text{warmup}}{\text{total} - \text{warmup}} \qquad \text{rate} = \text{peak} \times \Big(0.1 + 0.9 \times \tfrac{1}{2}\big(1 + \cos(\pi \times \text{progress})\big)\Big)$$

    The 0.1 is the floor: the rate never falls below a tenth of its peak. The 0.9 is the rest, $1 - 0.1$.

    ## By hand: four points on the curve

    For the chart's run (1,000 steps, warmup 100, peak 0.003), work out the rate at steps 50, 100, 550 and 1,000. Halfway round the half wave, $\cos(\pi / 2) = 0$.
    Check each answer against the chart.
    """)
    return


@app.cell(hide_code=True)
def _():
    rate_boxes = hand.boxes(['step 50', 'step 100', 'step 550', 'step 1000'], step=0.00005)
    return (rate_boxes,)


@app.cell(hide_code=True)
def _(rate_boxes):
    hand.marked(rate_boxes, {'step 50': 0.0015, 'step 100': 0.003, 'step 550': 0.00165, 'step 1000': 0.0003}, {
        'step 50': r"""still in warmup: half of the 100 steps are done, so half the peak""",
        'step 100': r"""warmup is over, and progress is 0: $\cos 0 = 1$, so the bracket is $0.1 + 0.9 = 1$""",
        'step 550': r"""progress is $450 / 900 = 0.5$, the cosine is 0, and the bracket is $0.1 + 0.9 \times 0.5$""",
        'step 1000': r"""progress is 1, $\cos\pi = -1$, and the bracket is 0.1""",
    }, LOCALE, tolerance=1e-7)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: leaving a function early
    A `return` ends a function at once, wherever it stands. So an `if` near the top can deal with one case and leave, and the lines below it only run in the other case:

    ```
    def size_word(n):
        if n < 10:
            return "small"
        return "big"
    ```
    ///

    ## Exercise 2: learning_rate

    Write `learning_rate(step, total, warmup, peak)`, the function that draws the chart. All four inputs are numbers: the step, the total number of steps, the number of warmup steps, and the peak rate.
    It gives back the rate for that step. While `step` is below `warmup`, that's the warmup formula; from there on, the cosine one. `np.cos` and `np.pi` are chapter 20's.

    > `learning_rate(550, 1000, 100, 0.003)` gives `0.00165`

    The cell after the check draws your function's curve. It should match the chart above.
    """)
    return


@app.function
def learning_rate(step, total, warmup, peak):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "learning_rate", learning_rate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""An `if` for the warmup case, with a `return` inside it, as in the Toolbox. Below it, a line for `progress` and a `return` for the cosine formula.""",
        r"""It's 4 lines. The last is `peak` times a bracket: 0.1, plus 0.9 times 0.5 times `(1 + np.cos(np.pi * progress))`. Count your brackets.""",
    ], LOCALE)
    return


@app.cell
def _():
    chart((0, 1000), (0, 0.0032), "step", "learning rate",
          curve=[(s, learning_rate(s, 1000, 100, 0.003)) for s in range(0, 1001, 5)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Checkpoints

    A real training run lasts weeks, and machines fail. So every so often a run writes a **checkpoint**: everything needed to carry on as if nothing had happened.
    That's more than the knobs. AdamW's averages $m$ and $v$ are part of the state too, and so is the step count, which the schedule needs.

    /// admonition | 🧰 Toolbox: state, and files
    - An optimizer has a `state_dict()` and a `load_state_dict(...)`, exactly like a model (chapter 18). Its state holds the averages.
    - A dictionary can be written with its pairs inside the curly brackets: `{"name": "ann", "age": 30}`.
    - `torch.save(thing, path)` writes a dictionary of tensors and numbers to a file, and `torch.load(path)` reads it back.
    ///

    ## Exercise 3: checkpoint

    Write `checkpoint(model, optimizer, step)`. It gives back a dictionary with three keys: `"model"` holds the model's state, `"optimizer"` the optimizer's state, and `"step"` the step number.

    ## Exercise 4: resume

    Write `resume(model, optimizer, saved)`, the way back. `saved` is a dictionary made by `checkpoint`. It loads the two saved states into the model and the optimizer it's given,
    and gives back the saved step number, so the caller knows where to carry on.
    """)
    return


@app.function
def checkpoint(model, optimizer, step):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "checkpoint", checkpoint, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""One line: a `return` with a dictionary written out in curly brackets, three pairs with commas between them.""",
        r"""The values are `model.state_dict()`, `optimizer.state_dict()` and `step`. Mind the brackets after `state_dict`: without them you store the tool, not its result.""",
    ], LOCALE)
    return


@app.function
def resume(model, optimizer, saved):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "resume", resume, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three lines: load into the model, load into the optimizer, give back the step. Each reads one key of `saved`.""",
        r"""The first line is `model.load_state_dict(saved["model"])`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell proves they work. It trains a small model for 20 steps, saves a checkpoint to a file, and carries on for 20 more. Then it makes a brand-new model,
    loads the checkpoint with your `resume`, and runs the same last 20 steps. If nothing was lost, the two models end identical. It does the same once more, but
    "forgets" the optimizer, loading only the knobs. The two numbers shown are the biggest gap between matching knobs, for the full resume and for the forgetful one.
    (`torch.rand` and `torch.randint` make random practice data, and `folder.mkdir` creates the folder if it's missing.)
    """)
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
    small_run(resumed, resumed_opt, resume(resumed, resumed_opt, saved), 40)

    forgetful = nn.Linear(10, 3)
    forgetful_opt = torch.optim.AdamW(forgetful.parameters(), lr=0.01)
    forgetful.load_state_dict(saved["model"])
    small_run(forgetful, forgetful_opt, 20, 40)
    torch.max(torch.abs(straight.weight - resumed.weight)).item(), torch.max(torch.abs(straight.weight - forgetful.weight)).item()
    return (folder,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Exactly 0 for the full resume, and a small gap for the forgetful one: with its averages reset to zero, AdamW takes different steps, and the run is no longer the same run.
    (The checkpoint lives in `my-work/.models`, next to your copies of the lessons.)

    ## The training loop

    `train_modern` is chapter 17's loop with AdamW and the schedule.

    /// admonition | 🧰 Toolbox: changing the rate during a run
    An optimizer keeps its settings in `optimizer.param_groups`, a list of dictionaries, usually with one entry. Storing into `group["lr"]` changes the learning rate from the next step on.
    ///
    """)
    return


@app.function(hide_code=True)
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
    Before every step, `train_modern` sets the rate for that step from your `learning_rate`.
    `per_word_perplexity` adds up the surprise over all test tokens (`reduction="sum"` gives the total instead of the mean) and divides by the **words**, as chapter 20 promised. Windows are 64 tokens long, a little over 60 words, since a token
    is nearly a word.

    ## By hand: read the training loop

    Find the line of `train_modern` that computes `total`, the number of steps in the whole run. The `+ batch_size - 1` makes the division round up, so a last batch that isn't full still counts.
    Suppose there are 1,000 windows, the batch size is 32, and we train for 2 epochs. How many steps does one epoch take? What's `total`? And what's the learning rate at the very first step, `step = 0`?
    """)
    return


@app.cell(hide_code=True)
def _():
    loop_boxes = hand.boxes(['steps in one epoch', 'total', 'rate at step 0'])
    return (loop_boxes,)


@app.cell(hide_code=True)
def _(loop_boxes):
    hand.marked(loop_boxes, {'steps in one epoch': 32, 'total': 64, 'rate at step 0': 0}, {
        'steps in one epoch': r"""$(1000 + 31)$ whole-divided by 32: 31 full batches and one of 8""",
        'total': r"""2 epochs of that""",
        'rate at step 0': r"""the warmup formula with `step = 0`""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So the very first step doesn't move the knobs at all, and the second moves them by a hundredth of the peak rate. That's warmup doing its job.

    One honest detail: cutting the text into windows drops a few tokens at the very end, so the words are counted over the tokens actually scored. The difference is a few tokens out of 167,000.

    ## The run

    Finish all four exercises first: the training uses them. The button trains chapter 20's model: 128 numbers per token, 4 heads, 3 blocks, the same size as chapter 17's. It takes about 3 to 7 minutes,
    then saves a checkpoint to `my-work/.models/ch21.pt`, so the run can be loaded again later. Until it finishes, the cells that need it say `Ancestor stopped`. That's normal.
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
    torch.save(checkpoint(model, optimizer, steps_done), folder / "ch21.pt")
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
    About 19 against chapter 17's 23. How much of that is the tokens, and how much the five swaps of these two chapters? We trained chapter 17's design on these same tokens, with the same time and
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

    - explain AdamW's decay, work out one step by hand, and match PyTorch's AdamW
    - read a warmup-and-cosine schedule off its chart, and write it
    - save a run with a checkpoint and resume it exactly, and say what goes wrong if the optimizer's state is left out
    - read a full training loop: schedule, clipping, steps and epochs
    - score a token model per word, and put it on the leaderboard

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** AdamW is from Loshchilov and Hutter (2017), who also proposed cosine schedules. Long runs usually keep several checkpoints and the best one by a held-out score,
    and big labs treat the schedule's peak, warmup length and decay as hyperparameters to search over.
    """)
    return


if __name__ == "__main__":
    app.run()
