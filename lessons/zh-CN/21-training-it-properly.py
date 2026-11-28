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

    第 20 章搭好了一个现代 Transformer，还没来得及训练它。好架构只是一半，怎么训练同样决定最终效果。这一章我们补上正式训练中缺一不可的三件套：

    1. **AdamW**：把权重衰减（缩小旋钮的力）和 Adam 的步长调整分开处理，互不干扰；
    2. **学习率调度**（learning-rate schedule）：让学习率先热身、再平滑下降；
    3. **检查点**（checkpoint）：随时保存训练进度，中断后也能原样续跑。

    写完这三样后，我们就用它们来训练第 20 章的模型，并把成绩送上排行榜。

    **复用前章代码：** 下面的折叠单元格里放着第 20 章写过的模型（`rms_norm`、`rope`、`swiglu`、模块和整网）以及分词数据（载入需要几秒钟），没有新内容。
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

    第 09 章里，我们曾把权重衰减直接加在斜率上：$\text{斜率} + \text{衰减} \times w$。对普通梯度下降来说，这样确实能让每个旋钮按相同比例往 0 收缩；但 Adam 会把整项除以每个旋钮平时斜率的大小，结果斜率大的旋钮几乎没缩，斜率小的却被缩掉一大截——这违背了权重衰减的初衷。

    **AdamW**（名字里的 **W** 就是 **Weight decay**，权重衰减的缩写）把衰减从 Adam 的滑动平均里拆出来，直接按相同比例作用在每个旋钮上。

    *符号速记：$g$ 是当前斜率，$m$ 和 $v$ 分别是斜率与斜率平方的滑动平均（初始为 0），$t$ 是从 1 开始的步数。*

    前两行和第 09 章的 Adam 完全一样，只有最后一步更新不同：

    > $$m = 0.9 \times m + 0.1 \times g \qquad v = 0.999 \times v + 0.001 \times g^2$$

    > $$\hat{m} = \frac{m}{1 - 0.9^t} \qquad \hat{v} = \frac{v}{1 - 0.999^t}$$

    > $$w \leftarrow w - \text{学习率} \times \Big(\frac{\hat{m}}{\sqrt{\hat{v}} + 0.00000001} + \text{衰减} \times w\Big)$$

    ## 动手算：走一步，比三种做法

    假设一个旋钮当前在 $w = 2$，这是第 1 步（$t = 1$），斜率 $g = 0.5$，学习率 0.1，衰减 0.5。顺便看一眼为什么第 1 步 $\hat{m} / \sqrt{\hat{v}}$ 总是 $\pm 1$：因为初始 $m = v = 0$，第 1 步算出 $m = 0.1g$、$v = 0.001g^2$，偏差修正刚好把前面的系数除掉，得到 $\hat{m} = g$、$\hat{v} = g^2$，所以只要斜率为正，$\hat{m} / \sqrt{\hat{v}} = g / |g| = +1$。请用下面三种做法分别算出走一步后的 $w$：

    - 普通的 **Adam**，完全不加衰减；
    - 把衰减**加到斜率上**的 Adam（第 09 章的做法）：先把斜率改成 $g + \text{衰减} \times w$，再按普通 Adam 算；
    - **AdamW**，直接套上面的公式。
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
    看出来了吗？加在斜率里的衰减彻底失效了——Adam 的除法把它抵消得干干净净，旋钮落点和普通 Adam 一模一样。换成 AdamW 后，衰减才真正起效：在 Adam 原本那一步的基础上，额外把旋钮往 0 拉近了 0.1。

    ## 练习 1：adamw_step

    实现 `adamw_step(w, m, v, grad, t, rate, decay)`，把上面三行公式写成代码。它就是在第 09 章 `adam_step` 的基础上多一个参数、多一项衰减。`w`、`m`、`v`、`grad` 是形状相同的 NumPy 数组，`t` 是从 1 开始计数的步数。和 `adam_step` 一样，返回三个值：更新后的旋钮，以及**未做偏差修正**的 `m` 和 `v`（下一步迭代还要用它们）。代码里乘方用 `**`，分母里那个极小数写作 `1e-8`。

    你刚手算的例子就在自动检查里：`w = [2.0]`、`grad = [0.5]`、`t = 1`，学习率 0.1，衰减 0.5，更新后的旋钮应为 `[1.8]`。通过检查后，下一个单元格会把你的实现和 PyTorch 自带的 `torch.optim.AdamW` 连跑 5 步做对比。
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
        r"""共五行，顺序和公式一一对应：更新 `m`、更新 `v`、用新变量名存两个偏差修正后的平均值（别覆盖未修正的 `m` 和 `v`），最后用 `return` 返回三个值。""",
        r"""衰减项放在 `rate` 所乘的括号里面、除法的外面：`rate * (m_hat / (np.sqrt(v_hat) + 1e-8) + decay * w)`。如果手算那个例子得出 1.9，多半是漏了衰减项或括号放错了位置。""",
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
    两份结果应该完全一致：连走 5 步后，旋钮数值一模一样。接下来的训练里，我们就直接用 PyTorch 自带的 `AdamW`。

    ## 2. 学习率调度

    训练初期好用的学习率，到了后期往往太大。刚开始时，旋钮全是随机数，斜率大起大落，步子迈太猛容易把训练带跑偏；快收尾时，步子得收小才能稳稳落进谷底，而不是在低点附近来回震荡。因此，我们会给学习率安排一条**变化曲线**（调度计划，schedule）。下面这条曲线对应一次 1,000 步、峰值 0.003 的训练：
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
    整条曲线分两段：

    - **预热**（warmup）：前 100 步里，学习率从 0 沿直线升到峰值：

    > $$\text{学习率} = \text{峰值} \times \frac{\text{步数}}{\text{预热步数}}$$

    - **余弦衰减**（cosine decay）：预热结束后，学习率沿着半个余弦波平滑降到一个下限。记预热后已走完的比例为**进度**（从 0 到 1）。为了方便心算，我们把公式拆成两步：先算一个从 1 平滑降到 0 的**余弦系数**，再把它映射到 $[0.1, 1.0]$ 区间后乘以峰值学习率：

    > $$\text{进度} = \frac{\text{步数} - \text{预热步数}}{\text{总步数} - \text{预热步数}} \qquad \text{余弦系数} = \frac{1}{2}\big(1 + \cos(\pi \times \text{进度})\big)$$

    > $$\text{学习率} = \text{峰值} \times (0.1 + 0.9 \times \text{余弦系数})$$

    当进度从 0 走到 1 时，$\cos(\pi \times \text{进度})$ 从 $1$ 降到 $-1$（见第 20 章的表），于是余弦系数正好从 1 平滑降到 0。式子里的 0.1 是下限（学习率最低不低于峰值的十分之一），0.9 则是剩余的浮动区间（$1 - 0.1$）。

    ## 动手算：曲线上的四个点

    按图里的这次训练（总步数 1,000，预热 100 步，峰值 0.003），心算第 50、100、550 和 1,000 步的学习率。衰减走到一半时，$\cos(\pi / 2) = 0$，余弦系数正好是 0.5。每算完一个，对照上面的曲线检查一下。
    """)
    return


@app.cell(hide_code=True)
def _():
    rate_boxes = hand.boxes(['第 50 步', '第 100 步', '第 550 步', '第 1000 步'], step=0.00005)
    return (rate_boxes,)


@app.cell(hide_code=True)
def _(rate_boxes):
    hand.marked(rate_boxes, {'第 50 步': 0.0015, '第 100 步': 0.003, '第 550 步': 0.00165, '第 1000 步': 0.0003}, {
        '第 50 步': r"""还在预热阶段：100 步走了一半，学习率是峰值的一半""",
        '第 100 步': r"""预热刚结束，进度是 0：$\cos 0 = 1$，余弦系数为 1，乘数是 $0.1 + 0.9 = 1$""",
        '第 550 步': r"""进度是 $450 / 900 = 0.5$，$\cos(\pi / 2) = 0$，余弦系数为 0.5，乘数是 $0.1 + 0.9 \times 0.5 = 0.55$""",
        '第 1000 步': r"""进度是 1，$\cos\pi = -1$，余弦系数为 0，乘数是 0.1""",
    }, LOCALE, tolerance=1e-7)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：提前离开函数
    `return` 无论写在哪一行，都会立刻结束函数。因此可以在开头用 `if` 先处理一种情况并直接返回，后面的代码就只会在另一种情况下执行：

    ```
    def size_word(n):
        if n < 10:
            return "small"
        return "big"
    ```
    ///

    ## 练习 2：learning_rate

    实现 `learning_rate(step, total, warmup, peak)`，也就是画出上面那条曲线的函数。四个输入都是数值：当前步数、总步数、预热步数和峰值学习率，返回这一步对应的学习率。当 `step` 小于 `warmup` 时按预热公式算；否则按余弦衰减公式算。`np.cos` 和 `np.pi` 在第 20 章用过。

    > `learning_rate(550, 1000, 100, 0.003)` 应返回 `0.00165`

    通过检查后，下一个单元格会画出你写的函数曲线，应该和上面的图完全重合。
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
        r"""先像工具箱那样用 `if` 处理预热情况并提前 `return`；下面依次算出 `progress`、余弦系数，最后返回映射后的学习率。""",
        r"""先算 `wave = 0.5 * (1 + np.cos(np.pi * progress))`，再返回 `peak * (0.1 + 0.9 * wave)`（也可以合写在一行里）。""",
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

    真实的模型训练动辄跑上几周，中途难免遇到断电或机器故障。因此每隔一段时间，程序都会存一份**检查点**（checkpoint）——把中途续跑所需的全部现场打包存盘，等恢复后就像从未中断过一样。需要保存的不只有模型旋钮：AdamW 记下的滑动平均 $m$ 和 $v$ 也是训练状态的一部分，当前步数也得存下来，学习率调度还要靠它接着算。

    /// admonition | 🧰 工具箱：状态和文件
    - 和模型一样（见第 18 章），优化器也有 `state_dict()` 和 `load_state_dict(...)`，用来读写内部存着的滑动平均。
    - 字典可以直接用花括号写出键值对：`{"name": "ann", "age": 30}`。
    - `torch.save(thing, path)` 把包含张量和数字的字典存成文件，`torch.load(path)` 再把它读出来。
    ///

    ## 练习 3：checkpoint

    实现 `checkpoint(model, optimizer, step)`，返回一个包含三个键的字典：`"model"` 对应模型的状态，`"optimizer"` 对应优化器的状态，`"step"` 对应当前步数。

    ## 练习 4：resume

    实现 `resume(model, optimizer, saved)`，完成相反的恢复操作。`saved` 是 `checkpoint` 返回的字典：把其中保存的两份状态分别载入传入的 `model` 和 `optimizer`，然后返回保存的步数，好让调用方知道从哪一步接着跑。
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
        r"""三个值是 `model.state_dict()`、`optimizer.state_dict()` 和 `step`。留意 `state_dict` 后面的括号：不加括号存进去的是函数本身，而不是调用后的结果。""",
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
        r"""三行：载入模型状态，载入优化器状态，返回步数。每一行各读 `saved` 的一个键。""",
        r"""第一行是 `model.load_state_dict(saved["model"])`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的单元格用来验证你的实现：它先把一个小模型训练 20 步，存盘后再接着跑 20 步作为对照；然后新建一个模型，用你的 `resume` 读回检查点并补跑后 20 步——如果状态完整无缺，两个模型最终的旋钮应该分毫不差。接着它再试一次“偷懒”版本：只恢复模型旋钮，丢掉优化器状态。最后输出的两个数，分别是完整恢复和丢掉优化器状态时，对应旋钮的最大差值。（`torch.rand` 和 `torch.randint` 用来生成随机练习数据，`folder.mkdir` 会在目录不存在时自动创建。）
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
    完整恢复时误差正好是 0，而漏掉优化器时出现了偏差：滑动平均一被清零，AdamW 后续每一步的步长都会走样，续跑出来的就已经不是原来的那次训练了。（检查点文件保存在 `my-work/.models` 目录下，就在你的课程副本旁边。）

    ## 训练循环

    `train_modern` 在第 17 章的训练循环上加上了 AdamW 和学习率调度。

    /// admonition | 🧰 工具箱：训练途中动态改学习率
    优化器的各项设置存在 `optimizer.param_groups` 里（一个由字典组成的列表，通常只有一项）。只要改写 `group["lr"]`，下一步就会按新的学习率更新。
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
    每走一步之前，`train_modern` 会先调用你的 `learning_rate` 设好当前步的学习率。在 `optimizer.step()` 之前，还多了一行 `nn.utils.clip_grad_norm_(model.parameters(), 1.0)`，叫作**梯度裁剪**（gradient clipping）：把整套斜率的总长度限制在 1.0 以内，防止偶尔冒出超大斜率把训练带崩。

    `per_word_perplexity` 把所有测试词元的意外程度全部加总（`reduction="sum"` 表示求和而非求平均），再除以**词数**，兑现第 20 章的约定。每个窗口长 64 个词元，略多于 60 个词，因为平均下来一个词元差不多就是一个词。

    ## 动手算：读懂训练循环

    在 `train_modern` 里找到计算 `total`（总步数）的那一行。`+ batch_size - 1` 的作用是让整数除法向上取整，这样最后剩下那个没装满的批也能算作一步。假设一共有 1,000 个窗口，批大小是 32，一共训练 2 轮：一轮要走多少步？`total` 是多少？最开始那一步（`step = 0`）的学习率又是多少？
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
    所以第 0 步根本不会挪动旋钮，第 1 步也只用峰值的百分之一去微调——这就是预热在起作用。

    补一个求实的细节：把测试集切成 64 词元的固定窗口时，最末尾凑不够一个窗口的几个词元会被截掉。下面训练单元格里的 `scored_words = word_count * Y_test_np.size / (len(test_tokens) - 1)`，就是按“实际参与打分的词元占全部测试词元的比例”把总词数等比例折算一下，让分母和真正打分的词元对齐。（当然，167,000 个词元里也就差了末尾几个。）

    ## 正式训练

    先把前面四个练习全部通过，训练过程会直接调用它们。点击下方按钮，开始训练第 20 章的模型：每个词元 128 维，4 个注意力头，3 个模块，规模和第 17 章相当。全程大约需要 3 到 7 分钟，跑完后会自动把检查点存到 `my-work/.models/ch21.pt`，方便以后重新载入。训练结束前，依赖它的后续单元格会显示 `Ancestor stopped`，这是正常现象。跑完后会输出模型的旋钮总数和按词计算的困惑度。
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
    大约 117 万个旋钮，和第 17 章的模型规模相近，但按词计算的困惑度降到了约 19。来看看排行榜：
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
    约 19，对比第 17 章的 23。这 4 分的进步里，有多少归功于换成词元，又有多少来自这两章的五处架构与训练升级？我们拿第 17 章的旧架构在同样的数据和词元上、用同样的时间重新训了一遍，结果约 23.4，和之前按词训练时几乎一样。因为保留 `<unk>` 后，这份语料差不多本来就是一个词对应一个词元，单换词元本身拉不开差距——真正的提升来自这五处升级。

    那如果去掉 `<unk>` 呢？我们也试过在完整文本上训练两个架构，把所有生僻词都拆成词元拼出来。结果两者的按词困惑度都变差了约 40%，现代模型降到约 27：逐段拼出 `xylophone`（木琴）这类词，比笼统地猜一句“某个生僻词”要难得多。这就是模型具备“拼出任意生词”能力所要付出的代价，也是排行榜继续保留 `<unk>` 以便公平对比的原因。

    下面看看新模型续写的一段故事（和训练语料一样，全小写、无标点）。`tokenizer.decode` 负责把词元还原成文字，`[:-1]` 用来去掉开头提示词末尾自动附带的 `<end>`：
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
    ## 下一章

    同样的模型规模，我们已经把效果提了一大截；接下来该直面“花销”本身了。**第 22 章**会拆解训练时的时间和显存都花在了哪里，算清楚一次训练要做多少次运算，并用一处改动让注意力提速好几倍。

    ## 你现在掌握了什么

    - 说清 AdamW 的权重衰减原理，能手算一步，并与 PyTorch 的 `AdamW` 对齐结果
    - 看懂“预热 + 余弦衰减”的学习率曲线，并亲手写出调度函数
    - 用检查点完整保存并无损恢复训练，明白漏存优化器状态会导致什么问题
    - 读懂完整的现代训练循环：学习率调度、梯度裁剪、步数与轮数
    - 按词计算词元模型的困惑度，并把模型送上排行榜

    ---

    <!-- go-deeper -->
    **延伸阅读（选读）：** AdamW 和余弦学习率调度均由 Loshchilov 与 Hutter（2017）提出。实际的长周期训练通常会保留多个检查点，并按验证集分数挑出表现最好的那份；大模型团队也会把峰值学习率、预热步数和衰减方式作为超参数来系统调优。
    """)
    return


if __name__ == "__main__":
    app.run()
