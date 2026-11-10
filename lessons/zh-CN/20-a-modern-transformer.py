import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="20 · 现代 Transformer")

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

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 · 现代 Transformer

    第 16 章的模型是 2017 年的设计。今天的大模型保留了它的骨架，也就是在 `x + f(x)` 这条路上交替的注意力和思考，但几乎每个部件后来都换成了更好的。这一章做其中五处替换，每处都不大，每处都经过测量：

    1. **RMSNorm**，更简单的归一化；
    2. **旋转位置编码**（RoPE），取代位置表；
    3. **SwiGLU**，带门的思考层；
    4. **AdamW**，更干净地缩小旋钮；
    5. 学习率的**调整计划**（schedule）。

    然后加上每次长时间训练都需要的东西：**检查点**（checkpoint），用来保存一次训练，以后接着跑。另外，模型现在读的是**词元**（token），用的是第 19 章的方法。

    ## 词元，以及公平的排行榜

    课程自带 2,000 次合并，是完全按第 19 章的方法在这些故事上训练出来的。在这么多文本上用你的写法训练要花不少时间，所以 `Tokenizer` 用两个标准技巧得到同样的合并，只是更快：它统计词而不是反复重读全文，并且每个词编码一次之后就记住结果。合并从不跨过一个词的开头，所以 `the` 和 ` cat` 绝不会粘成一块。

    有一条规则让排行榜保持公平。之前的模型读的是最常见的 2,000 个词，其他词都写成 `<unk>`。如果词元模型得把这些少见的词拼出来，而词模型只写一个 `<unk>`，比较就对它不公平：我们测过，这会让困惑度（perplexity）高出大约 40%。所以词元模型读的是完全相同的文本，少见的词同样是 `<unk>`，分数仍然是总意外程度除以**词**数，正如第 19 章承诺的那样。

    下面的单元格和以前完全一样地数词（`re.findall` 列出每一串字母和撇号，就像第 12 章的 `to_words`）。`set(list)` 造出一个**集合**（set）：它能一下子回答 `word in known`，就像只有键、没有值的字典。
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
    共 2,258 个词元编号（256 个字节、2,000 次合并、`<end>` 和 `<unk>`）。测试故事变成大约 167,000 个词元，对应 159,000 个词（和以前一样，每个故事的 `<end>` 算一个词）：平均每个词略多于一个词元，因为这 2,000 个词大多是单个词元，只有少数被切成几段。

    ## 1. RMSNorm

    第 16 章的层归一化（layer normalisation）先减去均值，再除以典型大小。研究者发现减均值的作用不大，所以 **RMSNorm**（均方根归一化）省掉了这一步：把每个向量除以它的**均方根**，$\sqrt{x^2 \text{ 的均值}}$，再乘以向量里每个数各自学出来的增益。（`1e-6` 只是防止除以零。）少一步，所以快一点，效果一样好。

    ## 练习 1：rms_norm

    写 `rms_norm(x, gain)`，沿最后一维计算：

    1. `rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)`
    2. `return gain * x / rms`
    """)
    return


@app.function
def rms_norm(x, gain):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rms_norm", rms_norm, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. 旋转

    第 16 章给每个位置一个学出来的向量，这有两个弱点：如果只在 32 个位置上训练过，模型对第 33 个位置一无所知；而且“第 7 个词看第 5 个词”和“第 27 个词看第 25 个词”得分别学，尽管它们是同一种关系：往回两个词。旋转位置编码解决了这两个问题。它需要几何里的一个想法。

    **角、余弦和正弦。** 以一个点为中心，画一个半径为 1 的圆。从最右边出发，沿着圆逆时针走。走过多少就是一个**角**（angle）。这时你的位置有两个坐标，从圆心量起：往右多远，叫这个角的**余弦**（cosine）；往上多远，叫它的**正弦**（sine）。负数表示往左或往下。所以角 $\theta$ 处的点是 $(\cos\theta, \sin\theta)$。角为 0 时，你在 $(1, 0)$。走过四分之一圈，你在顶上，$(0, 1)$。我们用**弧度**（radian）量角：一整圈是 $2\pi$（$\pi$ 大约是 3.14），大约 6.28，所以四分之一圈是 $\pi / 2$，大约 1.57。NumPy 能算它们，写作 `np.cos` 和 `np.sin`：
    """)
    return


@app.cell
def _():
    [(round(np.cos(angle).item(), 3), round(np.sin(angle).item(), 3)) for angle in [0, np.pi / 2, np.pi, 1.0]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `np.pi` 就是 $\pi$。这四个点分别是最右边、顶上、最左边，以及走过 1 弧度的一个点。

    **旋转一对数。** 把点 $(a, b)$ 绕圆心转过角 $\theta$，几何里的规则是：

    > $$(a, b) \;\to\; (a\cos\theta - b\sin\theta,\;\; a\sin\theta + b\cos\theta)$$

    拿 $(1, 0)$ 验证：得到 $(\cos\theta, \sin\theta)$，正是角 $\theta$ 处的点，理应如此。而顶上的 $(0, 1)$ 变成 $(-\sin\theta, \cos\theta)$：比它多转了四分之一圈，同样理应如此。

    ## 练习 2：rotate

    写 `rotate(a, b, angle)`，返回转过之后的一对数：

    ```
    return a * np.cos(angle) - b * np.sin(angle), a * np.sin(angle) + b * np.cos(angle)
    ```
    """)
    return


@app.function
def rotate(a, b, angle):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "rotate", rotate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在来看让旋转特别适合表示位置的那个性质。回想一下，注意力分数是查询和键的点积。取两对数 `q` 和 `k`，把 `q` 转一个角，把 `k` 转另一个角，再算它们的点积。结果只取决于两个角的**差**，和角本身无关。两个都再多转 1.5，结果不变：
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
    数字相同，因为把两对数一起转动，并不改变它们之间的相对位置。

    **旋转位置编码。** 于是：把每个查询和键切成一对一对的数，第 $p$ 个位置的词，它的每对数都转过 $p$ 乘以某个角。第 7 个词问第 5 个词时，分数取决于 $7 - 5 = 2$ 步的角度。第 27 个词问第 25 个词，结果完全一样。注意力现在看到的是相对位置，完全不需要位置表，而且超出训练窗口的位置至少也能算出来（不过模型在接近训练长度时效果最好）。每一对数有自己的转速，从每个位置 1 弧度，一直慢到大约 1/10,000 弧度，这样远近距离都分得清。（10,000 是约定俗成的取值。）下面的 `rope` 对整个张量做这件事。它就是同时对很多对数做 `rotate`，每个向量的前一半当作 `a`，后一半当作 `b`。
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
    （`x[..., :half]` 表示“最后一维的前一半，前面的维度不管”。`[:, None]` 把一个向量变成一列，这样每个位置都会和每种转速相遇。）

    ## 3. SwiGLU：带门的思考

    第 16 章的思考层是 `W2(relu(W1(x)))`。两处改动让它更好。第一，更平滑的弯折：**SiLU**，即 $x \times \sigma(x)$，用的是第 12 章的 sigmoid。$x$ 很大时它约等于 $x$，$x$ 很负时约等于 0，和 ReLU 一样，只是没有尖角。第二，类似第 14 章的门：第二个层 `W3` 逐个数字地决定第一个层有多少能通过。（它没有被压到 0 到 1 之间，所以只是像门，而不是严格意义上的门。）这里的 $\times$ 是逐个位置相乘：

    > $$\text{思考}(x) = W_2\big(\text{silu}(W_1 x) \times W_3 x\big)$$

    三个矩阵会比两个矩阵多出旋钮，所以隐藏层的宽度改成向量的 $\tfrac{8}{3}$ 倍，而不是 4 倍，让大小保持不变。

    ## 练习 3：swiglu

    写 `swiglu(x, W1, W2, W3)`，三个参数都是 `nn.Linear` 层，`F.silu` 就是 SiLU：

    ```
    return W2(F.silu(W1(x)) * W3(x))
    ```
    """)
    return


@app.function
def swiglu(x, W1, W2, W3):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "swiglu", swiglu, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. AdamW

    在第 09 章，你把权重衰减加到斜率上：`slope + decay * weight`。用普通的梯度下降步时，这会让每个权重都缩小一点。但 Adam 接着会把整个结果除以每个旋钮的典型斜率大小，于是斜率大的旋钮几乎不缩小，斜率小的却缩小很多。这不是衰减本来的意思。

    **AdamW** 让衰减不进入 Adam 的平均值，而是直接作用在旋钮上，对每个旋钮的比例都一样。这一处改动就是那个“W”。

    ## 练习 4：adamw_step

    写 `adamw_step(w, m, v, grad, t, rate, decay)`。它就是第 09 章的 `adam_step`，在步进里多一项衰减：

    1. `m = 0.9 * m + 0.1 * grad`
    2. `v = 0.999 * v + 0.001 * grad * grad`
    3. 两行：`m_hat = m / (1 - 0.9 ** t)` 和 `v_hat = v / (1 - 0.999 ** t)`
    4. `return w - rate * (m_hat / (np.sqrt(v_hat) + 1e-8) + decay * w), m, v`

    检查之后的单元格在第 09 章的两个旋钮上，把你写的这一步和 PyTorch 的 `torch.optim.AdamW` 作比较。
    """)
    return


@app.function
def adamw_step(w, m, v, grad, t, rate, decay):
    # ← 把下面的 todo() 换成你的代码
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
    两个数组应该完全相同：走 5 步之后，旋钮一模一样。

    ## 5. 学习率的调整计划

    训练早期合适的学习率，到后期就不合适了。刚开始时旋钮是随机的，斜率很乱，大步子可能把训练带偏。快结束时，小步子才能稳稳落在低点，而不是在它周围跳来跳去。所以学习率按一个**调整计划**变化：

    - **预热**（warmup）：在最初的若干步里，让学习率从 0 沿直线增长到峰值；
    - 然后是**余弦衰减**：沿着半个余弦波，平滑地降向一个下限。这段余弦从 1 开始、到 −1 结束，所以 $\tfrac{1}{2}(1 + \cos(\pi \times \text{进度}))$ 在进度从 0 到 1 时从 1 降到 0。

    > $$\text{学习率} = \text{峰值} \times \Big(\text{下限} + (1 - \text{下限}) \times \tfrac{1}{2}\big(1 + \cos(\pi \times \text{进度})\big)\Big)$$

    其中进度是预热结束后的那些步数里已经走完的比例。我们取下限为 0.1：学习率永远不低于峰值的十分之一。

    ## 练习 5：learning_rate

    写 `learning_rate(step, total, warmup, peak)`。`if ...: return ...` 让函数提前结束。

    1. `if step < warmup:`，里面写 `return peak * step / warmup`
    2. `progress = (step - warmup) / (total - warmup)`
    3. `return peak * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))`（0.9 就是公式里的 $1 - 0.1$）

    检查之后的图画出 1,000 步的调整计划：到第 100 步升到 0.003，然后平滑地降到 0.0003。
    """)
    return


@app.function
def learning_rate(step, total, warmup, peak):
    # ← 把下面的 todo() 换成你的代码
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
    ## 检查点

    真实的训练要跑好几周，而机器会出故障。所以训练每隔一段时间就写一个**检查点**：让训练能接着跑、就像什么都没发生过一样所需要的一切。这不只是旋钮。AdamW 的平均值也是状态的一部分（第 18 章的 `state_dict()` 能给出它们，优化器和模型都一样），步数也是，调整计划需要它。

    ## 练习 6：checkpoint

    写 `checkpoint(model, optimizer, step)`，返回一个包含这三样东西的字典：

    ```
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}
    ```

    检查之后的单元格证明它管用。它把一个小模型训练 20 步，用 `torch.save` 把检查点存进文件，再接着训练 20 步；然后新建一个模型，用 `torch.load` 读入检查点，跑同样的最后 20 步。两个模型最后应该完全相同：显示的数字是所有对应旋钮之间的最大差别。（`torch.rand` 和 `torch.randint` 造出随机的练习数据，`folder.mkdir` 在文件夹不存在时创建它。）
    """)
    return


@app.function
def checkpoint(model, optimizer, step):
    # ← 把下面的 todo() 换成你的代码
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
    正好是 0。（检查点存在 `my-work/.models` 里，就在你的课程副本旁边。）

    ## 现代模型

    下面把五处替换放到一起。把 `ModernBlock` 和第 16 章的 `Block` 比一比：同样是两步 `x + f(x)`，用 `rms_norm` 和 `swiglu` 取代了层归一化和 ReLU，`rope` 作用在查询和键上。`nn.Parameter` 把一个张量变成一个独立的旋钮，这里是每个增益。模型没有位置表。最后一层和注意力层都没有偏置，现代模型通常不要它们。
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
    `train_modern` 是第 16 章的循环，加上 AdamW 和调整计划：每一步之前，它先设好这一步的学习率（学习率存在 `optimizer.param_groups` 里）。`per_word_perplexity` 把所有测试词元的意外程度加起来，再除以**词**数，正如承诺的那样。窗口长 64 个词元，略多于 60 个词，因为一个词元差不多就是一个词。

    一个诚实的细节：把文本切成窗口时，最末尾会丢掉几个词元，所以词数是按实际打分的词元来折算的。差别只是 167,000 个词元里的几个。

    先完成全部六个练习：训练会用到它们。按下按钮，会训练一个和第 16 章一样大的现代模型：每个词元 128 个数字，4 个头，3 个块。大约要 3 到 7 分钟，然后把检查点存到 `my-work/.models/ch20.pt`，以后可以重新载入这次训练。完成之前，需要它的单元格会显示 `Ancestor stopped`，这是正常的。单元格显示旋钮数和按词计算的困惑度。
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
    torch.save(checkpoint(model, optimizer, steps_done), folder / "ch20.pt")
    scored_words = word_count * Y_test_np.size / (len(test_tokens) - 1)
    modern_score = per_word_perplexity(model, torch.tensor(X_test_np), torch.tensor(Y_test_np), scored_words)
    sum([p.numel() for p in model.parameters()]), round(modern_score.item(), 2)
    return model, modern_score


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 117 万个旋钮，和第 16 章的模型差不多，按词计算的困惑度大约 19。下面是排行榜：
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
    大约 19，而第 16 章是 23。这里面有多少来自词元，多少来自五处替换？我们用同样的词元、同样的时间和同样的文本训练了第 16 章的设计：它得了大约 23.4，和用词训练时一样。所以在这里，光换成词元，成绩变化不大，因为保留 `<unk>` 之后，这份文本差不多一个词就是一个词元。进步来自那五处替换。

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

    - 用 RMSNorm 做归一化
    - 解释角、余弦和正弦，并旋转一对数
    - 解释为什么旋转位置编码让注意力看到相对位置
    - 写一个带门的 SwiGLU 层
    - 解释 AdamW 的衰减，并和 PyTorch 的 AdamW 对上
    - 写一个预热加余弦的调整计划
    - 用检查点保存并精确地恢复一次训练

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** RMSNorm 出自 Zhang 和 Sennrich（2019），RoPE 出自 Su 等（*RoFormer*，2021），SwiGLU 出自 Shazeer（*GLU Variants Improve Transformer*，2020），AdamW 出自 Loshchilov 和 Hutter（2017）。LLaMA 这样的模型正是把这些选择组合在一起。很多模型还会*绑定*（tie）第一层和最后一层，共用一张词元向量表。
    """)
    return


if __name__ == "__main__":
    app.run()
