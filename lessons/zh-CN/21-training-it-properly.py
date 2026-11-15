import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="21 · 好好训练它")

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

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 21 · 好好训练它

    第 20 章造出了一个现代 Transformer，却在训练之前停了下来。光有好的设计还不够：模型怎么训练，和它由什么组成一样重要。这一章加上每次认真的训练都有的三样东西：

    1. **AdamW**，更干净地缩小旋钮；
    2. 学习率的**调整计划**（schedule）；
    3. **检查点**（checkpoint），用来保存一次训练，以后接着跑。

    然后你会用这三样训练第 20 章的模型，并把它放上排行榜。

    **来自前面的章节。** 下面的单元格里是第 20 章的模型和数据，代码折叠起来了：`rms_norm`、`rope`、`swiglu`、块和模型都是你在那一章写过的，词元也和以前一样载入，要花几秒钟。里面没有新东西。
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

    在第 09 章，你把权重衰减加到斜率上：$\text{斜率} + \text{衰减} \times w$。用普通的梯度下降步时，这会让每个权重都缩小一点。但 Adam 接着会把整个结果除以每个旋钮的典型斜率大小，于是斜率大的旋钮几乎不缩小，斜率小的却缩小很多。这不是衰减本来的意思。

    **AdamW** 让衰减不进入 Adam 的平均值，而是直接作用在旋钮上，对每个旋钮的比例都一样。这一处改动就是那个“W”。第 09 章的 Adam 前两行不变，只有步进是新的：

    > $$m = 0.9 \times m + 0.1 \times g \qquad v = 0.999 \times v + 0.001 \times g^2$$

    > $$\hat{m} = \frac{m}{1 - 0.9^t} \qquad \hat{v} = \frac{v}{1 - 0.999^t}$$

    > $$w \leftarrow w - \text{学习率} \times \Big(\frac{\hat{m}}{\sqrt{\hat{v}} + 0.00000001} + \text{衰减} \times w\Big)$$

    ## 动手算：同一步，三种算法

    一个旋钮在 $w = 2$。这是它的第一步，斜率 $g = 0.5$，学习率 0.1，衰减 0.5。第 09 章说过，第一步时只要斜率是正的，不管它多大，$\hat{m} / \sqrt{\hat{v}}$ 都正好是 $+1$。用三种算法算出走一步之后的旋钮：

    - 普通的 **Adam**，完全没有衰减；
    - 把衰减**加到斜率上**的 Adam，也就是第 09 章的做法：斜率变成 $g + \text{衰减} \times w$，然后照常运行 Adam；
    - **AdamW**，按上面的公式。
    """)
    return


@app.cell(hide_code=True)
def _():
    decay_boxes = hand.boxes(['Adam', '衰减加到斜率上', 'AdamW'], step=0.01)
    return (decay_boxes,)


@app.cell(hide_code=True)
def _(decay_boxes):
    hand.marked(decay_boxes, {'Adam': 1.9, '衰减加到斜率上': 1.9, 'AdamW': 1.8}, {
        'Adam': r"""$2 - 0.1 \times 1$""",
        '衰减加到斜率上': r"""斜率变成 $0.5 + 0.5 \times 2 = 1.5$。它还是正的，所以 $\hat{m} / \sqrt{\hat{v}}$ 还是 $+1$。""",
        'AdamW': r"""$2 - 0.1 \times (1 + 0.5 \times 2)$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    加到斜率上的衰减不见了：Adam 的除法把它吞掉了，旋钮落在和普通 Adam 完全相同的地方。在 AdamW 里，衰减起了作用：在 Adam 那一步之外，又把旋钮往零拉近了 0.1。

    ## 练习 1：adamw_step

    写 `adamw_step(w, m, v, grad, t, rate, decay)`，也就是上面的三个公式。它就是第 09 章的 `adam_step`，多一个输入，多一项。`w`、`m`、`v` 和 `grad` 是形状相同的 NumPy 数组，`t` 从 1 开始数步数。和 `adam_step` 一样，它返回三样东西：新的旋钮，以及没有修正过的 `m` 和 `v`，因为下一步需要的是它们。在代码里，乘方写作 `**`，那个很小的数写作 `1e-8`。

    你动手算的例子就是其中一项检查：`w = [2.0]`、`grad = [0.5]`、`t = 1`，学习率 0.1，衰减 0.5，得到新的旋钮 `[1.8]`。检查之后的单元格把你写的这一步和 PyTorch 的 `torch.optim.AdamW` 比较五步。
    """)
    return


@app.function
def adamw_step(w, m, v, grad, t, rate, decay):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "adamw_step", adamw_step, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""五行，顺序和公式一样：新的 `m`，新的 `v`，两个修正后的平均值（换个名字存，这样 `m` 和 `v` 保持没修正的样子），最后是返回三样东西的 `return`。""",
        r"""衰减项放在 `rate` 所乘的括号里面、除法的外面：`rate * (m_hat / (np.sqrt(v_hat) + 1e-8) + decay * w)`。如果动手算的那个例子你得到 1.9，就是衰减漏了，或者放错了地方。""",
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
    两个数组应该完全相同：走 5 步之后，旋钮一模一样。从这里开始，我们用 PyTorch 的。

    ## 2. 学习率的调整计划

    训练早期合适的学习率，到后期就不合适了。刚开始时旋钮是随机的，斜率很乱，大步子可能把训练带偏。快结束时，小步子才能稳稳落在低点，而不是在它周围跳来跳去。所以学习率按一个**调整计划**变化。下面是我们要用的这一个，对应一次 1,000 步、峰值为 0.003 的训练：
    """)
    return


@app.cell(hide_code=True)
def _():
    _curve = [(s, 0.003 * s / 100 if s < 100 else 0.003 * (0.1 + 0.45 * (1 + float(np.cos(np.pi * (s - 100) / 900))))) for s in range(0, 1001, 5)]
    chart((0, 1000), (0, 0.0032), "步", "学习率", curve=_curve)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    它分两段：

    - **预热**（warmup）：在最初的若干步里（这里是 100 步），学习率从 0 沿直线增长到峰值：

    > $$\text{学习率} = \text{峰值} \times \frac{\text{步数}}{\text{预热步数}}$$

    - 然后是**余弦衰减**（cosine decay）：沿着半个余弦波，平滑地降向一个下限。把预热之后的那些步里已经走完的比例叫作**进度**，它从 0 到 1。进度从 0 到 1 时，$\cos(\pi \times \text{进度})$ 从 1 到 −1（见第 20 章的表），所以 $\tfrac{1}{2}(1 + \cos(\pi \times \text{进度}))$ 从 1 降到 0：

    > $$\text{进度} = \frac{\text{步数} - \text{预热步数}}{\text{总步数} - \text{预热步数}} \qquad \text{学习率} = \text{峰值} \times \Big(0.1 + 0.9 \times \tfrac{1}{2}\big(1 + \cos(\pi \times \text{进度})\big)\Big)$$

    0.1 是下限：学习率永远不低于峰值的十分之一。0.9 是剩下的部分，$1 - 0.1$。

    ## 动手算：曲线上的四个点

    按图里的这次训练（1,000 步，预热 100 步，峰值 0.003），算出第 50、100、550 和 1,000 步的学习率。半个波走到一半时，$\cos(\pi / 2) = 0$。每算出一个答案，都和图对一对。
    """)
    return


@app.cell(hide_code=True)
def _():
    rate_boxes = hand.boxes(['第 50 步', '第 100 步', '第 550 步', '第 1000 步'], step=0.00005)
    return (rate_boxes,)


@app.cell(hide_code=True)
def _(rate_boxes):
    hand.marked(rate_boxes, {'第 50 步': 0.0015, '第 100 步': 0.003, '第 550 步': 0.00165, '第 1000 步': 0.0003}, {
        '第 50 步': r"""还在预热：100 步走了一半，所以是峰值的一半""",
        '第 100 步': r"""预热结束，进度是 0：$\cos 0 = 1$，所以括号里是 $0.1 + 0.9 = 1$""",
        '第 550 步': r"""进度是 $450 / 900 = 0.5$，余弦是 0，括号里是 $0.1 + 0.9 \times 0.5$""",
        '第 1000 步': r"""进度是 1，$\cos\pi = -1$，括号里是 0.1""",
    }, LOCALE, tolerance=1e-7)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：提前离开函数
    `return` 不管写在哪里，都会让函数立刻结束。所以可以在靠近开头的地方用一个 `if` 处理一种情况并离开，它下面的行就只在另一种情况下运行：

    ```
    def size_word(n):
        if n < 10:
            return "small"
        return "big"
    ```
    ///

    ## 练习 2：learning_rate

    写 `learning_rate(step, total, warmup, peak)`，也就是画出那张图的函数。四个输入都是数：当前的步数、总步数、预热的步数、峰值学习率。它返回这一步的学习率。`step` 小于 `warmup` 时，用预热的公式；从那以后，用余弦的公式。`np.cos` 和 `np.pi` 是第 20 章的。

    > `learning_rate(550, 1000, 100, 0.003)` 得到 `0.00165`

    检查之后的单元格会画出你的函数的曲线。它应该和上面的图一样。
    """)
    return


@app.function
def learning_rate(step, total, warmup, peak):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "learning_rate", learning_rate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""用一个 `if` 处理预热的情况，里面写 `return`，和工具箱里的例子一样。它下面是算 `progress` 的一行，和按余弦公式返回的一行。""",
        r"""一共 4 行。最后一行是 `peak` 乘以一个括号：0.1，加上 0.9 乘 0.5 乘 `(1 + np.cos(np.pi * progress))`。数一数你的括号。""",
    ], LOCALE)
    return


@app.cell
def _():
    chart((0, 1000), (0, 0.0032), "步", "学习率",
          curve=[(s, learning_rate(s, 1000, 100, 0.003)) for s in range(0, 1001, 5)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. 检查点

    真实的训练要跑好几周，而机器会出故障。所以训练每隔一段时间就写一个**检查点**：让训练能接着跑、就像什么都没发生过一样所需要的一切。这不只是旋钮。AdamW 的平均值 $m$ 和 $v$ 也是状态的一部分，步数也是，调整计划需要它。

    /// admonition | 🧰 工具箱：状态和文件
    - 优化器也有 `state_dict()` 和 `load_state_dict(...)`，和模型一模一样（第 18 章）。它的状态里存着那些平均值。
    - 写字典时，可以把键值对直接写在花括号里：`{"name": "ann", "age": 30}`。
    - `torch.save(thing, path)` 把一个由张量和数字组成的字典写进文件，`torch.load(path)` 把它读回来。
    ///

    ## 练习 3：checkpoint

    写 `checkpoint(model, optimizer, step)`。它返回一个有三个键的字典：`"model"` 下面是模型的状态，`"optimizer"` 下面是优化器的状态，`"step"` 下面是步数。

    ## 练习 4：resume

    写 `resume(model, optimizer, saved)`，也就是反过来走。`saved` 是 `checkpoint` 造出的字典。它把保存的两份状态分别载入传给它的模型和优化器，并返回保存的步数，这样调用的人就知道该从哪一步接着跑。
    """)
    return


@app.function
def checkpoint(model, optimizer, step):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "checkpoint", checkpoint, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行：`return` 后面是一个写在花括号里的字典，三个键值对，中间用逗号隔开。""",
        r"""三个值是 `model.state_dict()`、`optimizer.state_dict()` 和 `step`。留意 `state_dict` 后面的括号：没有括号，存进去的是这个工具本身，而不是它的结果。""",
    ], LOCALE)
    return


@app.function
def resume(model, optimizer, saved):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("21", "resume", resume, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三行：载入模型，载入优化器，返回步数。每一行读 `saved` 的一个键。""",
        r"""第一行是 `model.load_state_dict(saved["model"])`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下一个单元格证明它们管用。它把一个小模型训练 20 步，把检查点存进文件，再接着训练 20 步。然后新建一个模型，用你的 `resume` 载入检查点，跑同样的最后 20 步。如果什么都没丢，两个模型最后应该完全相同。它还会再做一遍，但这次“忘了”优化器，只载入旋钮。显示的两个数是对应旋钮之间的最大差别：一个是完整恢复的，一个是忘了优化器的。（`torch.rand` 和 `torch.randint` 造出随机的练习数据，`folder.mkdir` 在文件夹不存在时创建它。）
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
    完整恢复时正好是 0，忘了优化器时有一点差别：平均值被清零之后，AdamW 走的步子就不一样了，这次训练也就不再是原来那一次。（检查点存在 `my-work/.models` 里，就在你的课程副本旁边。）

    ## 训练循环

    `train_modern` 是第 17 章的循环，加上 AdamW 和调整计划。

    /// admonition | 🧰 工具箱：在训练途中改学习率
    优化器把它的设置存在 `optimizer.param_groups` 里：这是一个由字典组成的列表，通常只有一项。往 `group["lr"]` 里存一个数，从下一步起学习率就变了。
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
    每一步之前，`train_modern` 先用你的 `learning_rate` 设好这一步的学习率。`per_word_perplexity` 把所有测试词元的意外程度加起来（`reduction="sum"` 给出总和而不是均值），再除以**词**数，正如第 20 章承诺的那样。窗口长 64 个词元，略多于 60 个词，因为一个词元差不多就是一个词。

    ## 动手算：读一读训练循环

    在 `train_modern` 里找到计算 `total` 的那一行，它是整次训练的总步数。`+ batch_size - 1` 让除法向上取整，这样最后一个没装满的批也算一步。假设有 1,000 个窗口，批大小是 32，训练 2 轮。一轮有多少步？`total` 是多少？最开始那一步，也就是 `step = 0` 时，学习率是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    loop_boxes = hand.boxes(['一轮的步数', 'total', '第 0 步的学习率'])
    return (loop_boxes,)


@app.cell(hide_code=True)
def _(loop_boxes):
    hand.marked(loop_boxes, {'一轮的步数': 32, 'total': 64, '第 0 步的学习率': 0}, {
        '一轮的步数': r"""$(1000 + 31)$ 整除 32：31 个装满的批，加一个只有 8 个窗口的批""",
        'total': r"""这样的 2 轮""",
        '第 0 步的学习率': r"""把 `step = 0` 代进预热的公式""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所以第一步根本不会移动旋钮，第二步也只按峰值的百分之一来移动。这就是预热在起作用。

    一个诚实的细节：把文本切成窗口时，最末尾会丢掉几个词元，所以词数是按实际打分的词元来折算的。差别只是 167,000 个词元里的几个。

    ## 这一次训练

    先完成全部四个练习：训练会用到它们。按下按钮，会训练第 20 章的模型：每个词元 128 个数字，4 个头，3 个块，和第 17 章的一样大。大约要 3 到 7 分钟，然后把检查点存到 `my-work/.models/ch21.pt`，以后可以重新载入这次训练。完成之前，需要它的单元格会显示 `Ancestor stopped`，这是正常的。单元格显示旋钮数和按词计算的困惑度。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练现代 Transformer（3 到 7 分钟）")
    train_button
    return (train_button,)


@app.cell
def _(ModernGPT, folder, test_tokens, tokenizer, train_button, train_tokens, word_count):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
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
    大约 117 万个旋钮，和第 17 章的模型差不多，按词计算的困惑度大约 19。下面是排行榜：
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
    大约 19，而第 17 章是 23。这里面有多少来自词元，多少来自这两章的五处替换？我们用同样的词元、同样的时间和同样的文本训练了第 17 章的设计：它得了大约 23.4，和用词训练时一样。所以在这里，光换成词元，成绩变化不大，因为保留 `<unk>` 之后，这份文本差不多一个词就是一个词元。进步来自那五处替换。

    如果不用 `<unk>` 呢？我们也在完整文本上训练了两种设计，每个少见的词都用词元拼出来。两者按词计算都差了大约 40%，现代模型大约是 27：拼出 `xylophone`（木琴）比只说“某个少见的词”代价高得多。这是一个能写出任何词的模型要付出的真实代价，也是我们的排行榜保留 `<unk>` 的原因。

    下面是新模型写的一段文字，全是小写、没有标点，和它学习的文本一样。`tokenizer.decode` 把词元变回文本，`[:-1]` 去掉开头几个词后面的 `<end>`：
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
    ## 接下来

    我们用同样的代价让模型变得更好了。接下来看代价本身。**第 22 章**测量训练时时间和内存都花在哪里，数一数一次训练要做多少算术，并用一处改动让注意力快好几倍。

    ## 你现在会做的事

    - 解释 AdamW 的衰减，亲手算出一步，并和 PyTorch 的 AdamW 对上
    - 从图上读出预热加余弦的调整计划，并把它写出来
    - 用检查点保存一次训练并精确地恢复它，并说出漏掉优化器的状态会出什么问题
    - 读懂一个完整的训练循环：调整计划、裁剪、步数和轮数
    - 按词给词元模型打分，并把它放上排行榜

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** AdamW 出自 Loshchilov 和 Hutter（2017），余弦调整计划也是他们提出的。长时间的训练通常会保留好几个检查点，并按留出数据上的分数留下最好的那个；大实验室会把调整计划的峰值、预热长度和衰减方式都当作超参数来搜索。
    """)
    return


if __name__ == "__main__":
    app.run()
