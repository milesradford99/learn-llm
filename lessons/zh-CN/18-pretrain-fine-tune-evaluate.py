import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="18 · 预训练、微调与评估")

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
    from learn_llm.diagrams import pipeline

    LOCALE = "zh-CN"
    laptop_speed()
    ANIMALS = ["dog", "cat", "bird", "fish"]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 18 · 预训练、微调与评估

    用 190 万个词训练时不需要标签，因为文本本身就是答案。大多数有用的任务不是这样。要给邮件分类或者发现疾病，都得有人手工标注样本，而标签很贵。

    标准做法是**预训练**（pretraining）：先用大量普通文本训练语言模型，再把它用到只有少量标签的任务上。这一章就来试试。但这一章真正讲的是另一件事：**怎样判断一个结果是真的。** 你会故意骗一个模型，见到一个什么也不做却能拿 40% 的“模型”，还会看到一个流行的说法在数字里得不到印证。

    ## 从故事里造一个任务

    我们没有人工标注的数据，所以从故事里造一个任务。取所有只提到下面四种动物之一的故事：`dog`（狗）、`cat`（猫）、`bird`（鸟）、`fish`（鱼）。标签就是那种动物。然后把这个词在故事里全部**遮住**，问模型这个故事讲的是哪种动物。它只能靠 `barked`（叫）、`swam`（游泳）这样的线索推断。（只遮这几个完全相同的词，所以 `dogs` 这样的词还是可能漏过去。）

    /// admonition | 🧰 工具箱：在两个值里选一个
    `a if test else b` 是一个表达式：条件成立时给出 `a`，否则给出 `b`。它可以放进造列表的表达式里，用来决定每一项：`[0 if x < 0 else x for x in [3, -1, 2]]` 得到 `[3, 0, 2]`。
    ///

    ## 练习 1：blank_out

    写 `blank_out(ids, hidden, blank)`。`ids` 是词编号列表，`hidden` 是要遮住的编号列表，`blank` 是替换用的编号。它返回一个长度相同的新列表：

    > `blank_out([5, 7, 5, 9], [5], 0)` 得到 `[0, 7, 0, 9]`
    """)
    return


@app.function
def blank_out(ids, hidden, blank):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "blank_out", blank_out, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""对 `ids` 写一个造列表的表达式。每一项要么是 `blank`，要么是编号本身；条件是这个编号是否 `in hidden`。""",
        r"""只要一行，形状和工具箱里的例子一样：`return [... if ... else ... for i in ids]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 模型，和一个新的最后一层

    **来自前面的章节。** 下面两个单元格里是第 17 章的模型，小一些。它们的代码折叠起来了，因为这些你在那一章全都写过。有两处是新的。模型多了一个开关：`.features(ids)` 给出最后一个块输出的向量，还没有打分；`forward` 就只是 `self.out(self.features(ids))`。另外，注意力的三个层放在一个 `nn.ModuleList` 里：这是一种列表，它让 PyTorch 知道里面的层有旋钮要训练。
    """)
    return


@app.function(hide_code=True)
def attend(Q, K, V, mask):
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    A = torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1)
    return A @ V, A


@app.cell(hide_code=True)
def _():
    class Block(nn.Module):
        def __init__(self, d, heads):
            super().__init__()
            self.heads = heads
            self.norm1 = nn.LayerNorm(d)
            self.attention = nn.ModuleList([nn.Linear(d, d, bias=False) for i in range(3)])
            self.mix = nn.Linear(d, d)
            self.norm2 = nn.LayerNorm(d)
            self.think = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

        def look(self, x):
            batch, words, d = x.shape
            Q, K, V = [layer(x).reshape(batch, words, self.heads, d // self.heads).transpose(1, 2) for layer in self.attention]
            mask = torch.triu(torch.ones(words, words, dtype=torch.bool), diagonal=1)
            return self.mix(attend(Q, K, V, mask)[0].transpose(1, 2).reshape(batch, words, d))

        def forward(self, x):
            x = x + self.look(self.norm1(x))
            return x + self.think(self.norm2(x))

    class TinyGPT(nn.Module):
        def __init__(self, vocab_size, longest, d, heads, layers):
            super().__init__()
            self.words = nn.Embedding(vocab_size, d)
            self.positions = nn.Embedding(longest, d)
            self.blocks = nn.Sequential(*[Block(d, heads) for i in range(layers)])
            self.norm = nn.LayerNorm(d)
            self.out = nn.Linear(d, vocab_size)

        def features(self, ids):
            x = self.words(ids) + self.positions(torch.arange(ids.shape[1]))
            return self.norm(self.blocks(x))

        def forward(self, ids):
            return self.out(self.features(ids))
    return (TinyGPT,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    要给故事分类，我们在这些特征上面加一个新的小层。每个样本是故事的前 64 个词，特征里每个词有 96 个数字。计划是这样的：
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([('ids', '(batch, 64)'), ('特征', '(batch, 64, 96)'), ('一个向量', '(batch, 96)'), ('分数', '(batch, 4)')],
             ['GPT 的主体', '对所有词取平均', '新的一层：头部'],
             '编号经过 GPT 的主体得到特征，对所有词取平均后每个故事得到一个向量，再由叫作头部的新层给出四个分数。')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    GPT 的那些块叫**主体**（body），新的 `nn.Linear(96, 4)` 叫**头部**（head）。对所有词取平均，能把任意长度的故事变成一个向量，这样头部才能给它打分：每种动物一个分数。

    ## 练习 2：classify

    写 `classify(features, head)`，也就是图里的后两个箭头。`features` 的形状是 `(batch, words, d)`，`head` 是一个 `nn.Linear(d, 4)`。它返回分数，形状是 `(batch, 4)`。`torch.mean(x, dim=1)` 沿第 1 维取平均，这一维随后从形状里消失。
    """)
    return


@app.function
def classify(features, head):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "classify", classify, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两步，一步套一步：先对所有词取平均，再对结果调用头部。""",
        r"""只要一行：`return head(...)`，括号里是沿 `dim=1` 对 `features` 取的 `torch.mean`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Classifier` 装着一个主体和一个头部，它的 `forward` 就是你的函数。它后面的单元格准备数据。练习 1 和练习 2 通过之前，用到它们的单元格会显示错误。这是正常的，后面的练习也一样。
    """)
    return


@app.cell
def _():
    class Classifier(nn.Module):
        def __init__(self, body, d):
            super().__init__()
            self.body = body
            self.head = nn.Linear(d, len(ANIMALS))

        def forward(self, ids):
            return classify(self.body.features(ids), self.head)
    return (Classifier,)


@app.function(hide_code=True)
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function(hide_code=True)
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.function
def labelled(story_list, index):
    animal_ids = [index[a] for a in ANIMALS]
    examples = []
    for story in story_list:
        ids = [index.get(word, index["<unk>"]) for word in to_words(story)]
        found = [a for a in range(4) if animal_ids[a] in ids]
        if len(found) == 1:
            examples.append((ids, found[0]))
    return examples


@app.function
def to_tensors(examples, hidden, blank, end):
    rows = []
    for ids, label in examples:
        row = blank_out(ids, hidden, blank)[:64]
        rows.append(row + [end] * (64 - len(row)))
    return torch.tensor(rows), torch.tensor([label for ids, label in examples])


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
    hidden = [index[a] for a in ANIMALS]
    train_examples = labelled(train_stories, index)
    test_examples = labelled(test_stories, index)
    X_test, Y_test = to_tensors(test_examples, hidden, index["<unk>"], index["<end>"])
    len(train_examples), len(test_examples), torch.bincount(Y_test).tolist()
    return (
        X_test,
        Y_test,
        hidden,
        index,
        test_examples,
        train_examples,
        train_stories,
        vocab,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 3,300 篇有标签的故事用于训练，265 篇用于测试：狗 107 篇，猫 55 篇，鸟 79 篇，鱼 24 篇。

    /// admonition | 🧰 工具箱：数标签
    `np.bincount(a)` 统计数组里每个整数出现的次数：结果的位置 0 是 0 的个数，位置 1 是 1 的个数，依此类推。`np.bincount(np.array([0, 2, 2, 1, 2]))` 是 `[1, 1, 3]`。上面用到的 `torch.bincount` 是它在 PyTorch 里的孪生兄弟。
    ///

    ## 先定一条基线

    在用任何模型之前，先问问分数意味着什么。如果 40% 的测试故事讲狗，一个总说 `dog` 的“模型”一个字不读也能拿 40%。这就是**多数类基线**（majority baseline），模型只有超过它才有意思。反过来也一样：如果 99% 的邮件都正常，一个什么都不拦的过滤器也能得 99%。

    ## 练习 3：majority_share

    写 `majority_share(labels)`。`labels` 是一个 NumPy 数组，里面是 0 或更大的整数。它返回最常见的那个标签占全部标签的比例：也就是总给出这个答案时的准确率。

    > `majority_share(np.array([1, 1, 1, 0, 2, 2]))` 得到 `0.5`
    """)
    return


@app.function
def majority_share(labels):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "majority_share", majority_share, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""用 `np.bincount` 数出每个标签的个数，用 `np.max` 取最大的那个，再除以标签的总数。""",
        r"""只要一行。在例子里，各标签的个数是 `[1, 3, 2]`，最大的是 3，一共有 6 个标签。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 比准确率更好的度量

    准确率看不出模型犯的是哪种错。对某一类，比如 `fish`，把预测分成三种计数：

    - **真阳性**（$tp$）：说是鱼，确实是鱼；
    - **假阳性**（$fp$）：说是鱼，其实不是；
    - **假阴性**（$fn$）：其实是鱼，模型说成了别的。

    由此得到两个度量：

    > $$\text{精确率} = \frac{tp}{tp + fp} \qquad \text{召回率} = \frac{tp}{tp + fn}$$

    **精确率**（precision）问的是：它说是鱼的时候，有多少次是对的？**召回率**（recall）问的是：真正的鱼，它找到了多少？

    从不判“垃圾邮件”的过滤器找不到垃圾邮件，召回率是 0；它一次也没猜过，精确率无从算起，我们同样记 0。什么都判成“垃圾邮件”的过滤器，召回率是 1，精确率却很差。这两个数字互相制约。

    ## 动手算：数出这三种情况

    十篇故事，每篇都有模型的猜测和真实的标签。F 是鱼，D 是狗。

    | 故事 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | 猜测 | F | F | F | D | F | D | D | F | D | D |
    | 标签 | F | F | D | D | F | F | D | F | F | D |

    对“鱼”这一类，数出 $tp$、$fp$ 和 $fn$。再算出精确率和召回率，保留两位小数。
    """)
    return


@app.cell(hide_code=True)
def _():
    fish_boxes = hand.boxes(['tp', 'fp', 'fn', '精确率', '召回率'], step=0.01)
    return (fish_boxes,)


@app.cell(hide_code=True)
def _(fish_boxes):
    hand.marked(fish_boxes, {'tp': 4, 'fp': 1, 'fn': 2, '精确率': 0.8, '召回率': 0.67}, {
        'tp': r"""两行都是 F 的那些列""",
        'fp': r"""猜测是 F、标签是 D 的那些列""",
        'fn': r"""猜测是 D、标签是 F 的那些列""",
        '精确率': r"""$tp$ 除以猜成 F 的次数，也就是 $tp + fp$""",
        '召回率': r"""$tp$ 除以标签是 F 的个数，也就是 $tp + fn$""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    模型 10 次里对了 7 次，但光看这一个数，你不会知道它漏掉了三分之一的鱼。

    /// admonition | 🧰 工具箱：用比较来计数
    - 把数组和一个数比较，得到一个由 `True` 和 `False` 组成的数组：`np.array([1, 0, 1]) == 1` 是 `[True, False, True]`。
    - `&` 表示逐个位置的“并且”：两边都是 `True` 的位置才是 `True`。每个比较都要有自己的括号：`(a == 1) & (b != 1)`。
    - 对这样的数组用 `np.sum`，数出的是 `True` 的个数。
    ///

    ## 练习 4：precision_recall

    写 `precision_recall(guesses, labels, c)`。`guesses` 和 `labels` 是长度相同的 NumPy 数组，里面是类别的编号；`c` 是要衡量的那一类。它返回一对：先是精确率，再是召回率。如果某个分数的分母是 0，这个度量就记 0：`max(n, 1)` 是不让分母为 0 的简便写法，因为这时分子也是 0。

    把鱼记作类别 1，狗记作类别 0，上面那张表就是其中一项检查，结果大约是 `(0.8, 0.67)`。
    """)
    return


@app.function
def precision_recall(guesses, labels, c):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "precision_recall", precision_recall, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三个计数，每个都是对两个用 `&` 连起来的比较做 `np.sum`。`tp` 是 `guesses` 和 `labels` 都等于 `c`。`fp` 是猜测等于 `c` 而标签不等于（`!=`）。`fn` 正好反过来。""",
        r"""一共 4 行。第一行是 `tp = np.sum((guesses == c) & (labels == c))`。最后一行返回两个分数，中间用逗号隔开，精确率在前，两个分母都写成 `max(..., 1)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 差别是真的吗？

    只有 265 篇测试故事，分数的噪声不小。想知道噪声有多大，可以设想一个模型，它答对每篇故事的概率正好是 0.6，所以它真实的分数是 60%。现在拿 200 个不同的测试集去测它，每个测试集 265 篇故事。下面的单元格用随机数来做这件事（`rng.random(shape)` 给出 0 到 1 之间的数，所以每个数小于 0.6 的概率是 0.6），并显示这 200 个分数里最低和最高的。
    """)
    return


@app.cell
def _():
    rng = np.random.default_rng(0)
    right = rng.random((200, 265)) < 0.6
    test_scores = np.mean(right, axis=1)
    round(np.min(test_scores).item(), 3), round(np.max(test_scores).item(), 3)
    return (test_scores,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同一个模型，分数从大约 52% 到大约 69% 都有，差别只在于测试集里碰巧放进了哪些故事。所以单独一个分数是真值加上一点运气，我们得知道运气的成分能有多大。

    把每篇故事想成一枚硬币：以概率 $p$ 落成“对”（1），否则是“错”（0）。一次这样的结果方差是 $p \times (1 - p)$，这一点可以先相信。其余的用第 09 章的两条规则：方差相加，所以 $n$ 篇故事的总和方差是 $n \times p \times (1 - p)$；再除以 $n$ 求均值，方差要除以 $n^2$。于是得到 $p \times (1 - p) / n$，它的平方根说明测出的分数通常会偏离真值多远。

    这个大小的两倍能覆盖大约 95% 的情况，这也是可以先相信的事实，所以常用的**误差范围**（margin of error）是：

    > $$\text{误差范围} = 2 \times \sqrt{\frac{p \times (1 - p)}{n}}$$

    ## 动手算：两个误差范围

    一个模型在 $n = 100$ 个样本的测试上得分 $p = 0.5$。误差范围是多少？样本多到四倍，$n = 400$ 时又是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    margin_boxes = hand.boxes(['n = 100 时的误差范围', 'n = 400 时的误差范围'], step=0.01)
    return (margin_boxes,)


@app.cell(hide_code=True)
def _(margin_boxes):
    hand.marked(margin_boxes, {'n = 100 时的误差范围': 0.1, 'n = 400 时的误差范围': 0.05}, {
        'n = 100 时的误差范围': r"""$0.5 \times 0.5 / 100 = 0.0025$，它的平方根是 0.05""",
        'n = 400 时的误差范围': r"""$0.25 / 400 = 0.000625$，它的平方根是 0.025""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    数据多到四倍，误差范围只缩小一半。所以小测试集带来的麻烦很难摆脱。

    ## 练习 5：margin

    写 `margin(p, n)`，也就是上面的公式。`p` 是 0 到 1 之间的分数，`n` 是测试样本的个数。`np.sqrt` 是平方根。

    > `margin(0.5, 100)` 得到 `0.1`

    检查之后的单元格会数一数：模拟出的 200 个分数里，有多少落在离真值 0.6 一个误差范围以内。
    """)
    return


@app.function
def margin(p, n):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "margin", margin, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行。分数写在 `np.sqrt` 里面，结果再乘以 2。""",
        r"""如果 `margin(0.5, 100)` 得到 0.005，就是漏了平方根。如果得到 0.05，就是漏了 2。""",
    ], LOCALE)
    return


@app.cell
def _(test_scores):
    np.mean(np.abs(test_scores - 0.6) < margin(0.6, 265)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 0.95，和前面说的一样：20 个分数里有 19 个落在误差范围以内，1 个落在外面。

    当 $p = 0.6$、$n = 265$ 时，误差范围大约是 0.06，也就是 6 个百分点。比较两个分数时，它们各有各的噪声，方差相加后，差值的误差范围大约是 1.4 倍：这里大约是 8.5 个百分点。所以在这个测试上，58% 和 61% 的两个模型分不出高下。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 预训练

    下面的按钮用全部训练故事训练那个较小的 GPT，不管标签，大约 2 到 4 分钟。完成之前，需要它的单元格会显示 `Ancestor stopped`，这是正常的。（模型见过的是和测试故事同*类型*的文本，而不是测试故事本身：进去的只有 `train_stories`。）
    """)
    return


@app.function
def pretrain(model, ids, epochs):
    n = (len(ids) - 1) // 64
    X = torch.tensor(ids[:n * 64].reshape(n, 64))
    Y = torch.tensor(ids[1:n * 64 + 1].reshape(n, 64))
    size = model.out.out_features
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, 64):
            batch = order[start:start + 64]
            optimizer.zero_grad()
            F.cross_entropy(model(X[batch]).reshape(-1, size), Y[batch].reshape(-1)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return {name: value.clone() for name, value in model.state_dict().items()}


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="预训练（2 到 4 分钟）")
    train_button
    return (train_button,)


@app.cell
def _(TinyGPT, index, train_button, train_stories, vocab):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    torch.manual_seed(0)
    pretrained_model = TinyGPT(len(vocab), 64, 96, 4, 2)
    saved = pretrain(pretrained_model, story_ids(train_stories, index), 2)
    len(saved)
    return pretrained_model, saved


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：保存旋钮
    `model.state_dict()` 给出一个字典，按名字存着模型的全部旋钮；`model.load_state_dict(saved)` 把存下来的一套旋钮放回一个设计相同的模型里。
    ///

    所以 `saved` 让我们能从同一个预训练好的起点启动好几个模型。显示的数字是有名字的旋钮张量的个数。

    ## 完全不用标签

    语言模型不用一个标签也能试这个任务。在每个空白处，问四种动物作为下一个词的概率，把故事里所有空白处的对数加起来，取最大的那个。单元格显示多数类基线、正确比例和它的误差范围。
    """)
    return


@app.function
def guess_without_labels(model, examples, hidden, blank):
    guesses = []
    for ids, label in examples:
        row = torch.tensor([blank_out(ids, hidden, blank)[:64]])
        with torch.no_grad():
            chances = torch.log_softmax(model(row)[0], dim=1)
        totals = np.zeros(4)
        for t in range(1, row.shape[1]):
            if ids[t] in hidden:
                totals = totals + chances[t - 1, hidden].numpy()
        guesses.append(np.argmax(totals).item())
    return np.array(guesses)


@app.cell
def _(Y_test, hidden, index, pretrained_model, test_examples):
    no_label_guesses = guess_without_labels(pretrained_model, test_examples, hidden, index["<unk>"])
    labels_np = Y_test.numpy()
    no_label_score = np.mean(no_label_guesses == labels_np)
    majority_share(labels_np).item(), no_label_score.item(), margin(no_label_score, len(labels_np)).item()
    return (labels_np,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    多数类基线是 40%。预训练模型不用任何标签就拿到大约 51%，比基线高出的部分超过了 6 个百分点的误差范围。预训练确实起了作用。

    ## 微调

    现在用上标签。**微调**（fine-tuning）是从预训练好的旋钮出发，加一个新的最后一层，再用有标签的样本轻轻地训练。这比从随机旋钮出发更好吗？还有个折中的办法：**冻结**（freeze）主体，只让新层学习。对照图里的两个部分，三种做法是这样的：

    | | 主体 | 头部 |
    |---|---|---|
    | **从零开始** | 随机起点，参加训练 | 随机起点，参加训练 |
    | **冻结主体** | 预训练好的，不训练 | 随机起点，参加训练 |
    | **微调** | 预训练好的，轻轻地训练 | 随机起点，参加训练 |

    要冻结一个旋钮，就把它的 `requires_grad` 设为 `False`（第 10 章），这样责任到不了它，优化器也不会动它。

    ## 练习 6：freeze

    写 `freeze(model)`。它把模型每个旋钮的 `requires_grad` 都设为 `False`，什么也不返回：它直接修改传给它的模型，所以不需要 `return`。`model.parameters()` 逐个给出模型的旋钮（第 10 章）。
    """)
    return


@app.function
def freeze(model):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "freeze", freeze, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""对 `model.parameters()` 做一个循环，里面只有一行，存入 `False`。""",
        r"""一共 2 行。存值用一个 `=`，像 `p.requires_grad = False` 这样。两个 `==` 只是比较。""",
    ], LOCALE)
    return


@app.function
def fit(model, X, Y, epochs, rate):
    knobs = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(knobs, lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(X))
        for start in range(0, len(X), 32):
            batch = order[start:start + 32]
            optimizer.zero_grad()
            F.cross_entropy(model(X[batch]), Y[batch]).backward()
            optimizer.step()


@app.function
def share_right(model, X, Y):
    with torch.no_grad():
        return torch.mean((torch.argmax(model(X), dim=1) == Y).float()).item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下一个单元格在 100、300、1,000 篇有标签的故事上试这三种做法，大约要一分钟。每种做法用适合它的设置：冻结主体时只训练一层，所以轮次多一些；微调用很小的学习率，以免破坏预训练好的旋钮。运行之前先猜一猜：哪一列会赢？
    """)
    return


@app.cell(hide_code=True)
def _(
    Classifier,
    TinyGPT,
    X_test,
    Y_test,
    hidden,
    index,
    saved,
    train_examples,
    vocab,
):
    rows = ["| 标签数 | 从零开始 | 冻结主体 | 微调 |", "|---|---|---|---|"]
    best_for_table = {}
    for n in [100, 300, 1000]:
        chosen = [train_examples[i] for i in np.random.default_rng(0).permutation(len(train_examples))[:n].tolist()]
        X_n, Y_n = to_tensors(chosen, hidden, index["<unk>"], index["<end>"])
        cells = []
        for kind, epochs, rate in (("scratch", 15, 0.001), ("frozen", 40, 0.003), ("tuned", 10, 0.0005)):
            torch.manual_seed(1)
            body = TinyGPT(len(vocab), 64, 96, 4, 2)
            if kind != "scratch":
                body.load_state_dict(saved)
            if kind == "frozen":
                freeze(body)
            classifier = Classifier(body, 96)
            fit(classifier, X_n, Y_n, epochs, rate)
            cells.append(f"{share_right(classifier, X_test, Y_test):.1%}")
            best_for_table[(n, kind)] = classifier
        rows.append(f"| {n} | " + " | ".join(cells) + " |")
    mo.md("\n".join(rows))
    return (best_for_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在带着练习 5 的结论来读这张表。（两格数字相同只是巧合。）在 100 和 300 个标签时，差距只有几个百分点，远在大约 8.5 的误差范围以内，所以这几列分不出高下。在 1,000 个标签时，从零开始领先 7 到 8 个百分点，正好在误差范围的边缘：可能是真的，但不能确定。

    **所以预训练在这里没有带来明显的好处。** 写这一章时，我们预期会看到常见的结论，也就是预训练明显胜出。我们试了好几种设置，这个结果始终没有出现。这是小模型在少量文本上预训练的诚实结果，而判断哪种任务它才能发挥优势，也是这门手艺的一部分。当预训练模型远大于你靠手头标签能训练出来的模型，而且标签非常少时，预训练才划算。前面那个 51% 的亮点不在于数字本身，而在于它完全不用标签。

    下面是 300 个标签时的微调模型，每行是（动物，精确率，召回率）。注意最少见的 `fish`，只看准确率是看不出模型在这一类上的表现的。鱼的故事只有 24 篇，所以这两个数字的噪声也很大。
    """)
    return


@app.cell
def _(X_test, best_for_table, labels_np):
    with torch.no_grad():
        tuned_guesses = torch.argmax(best_for_table[(300, "tuned")](X_test), dim=1).numpy()
    report = []
    for c in range(4):
        p, r = precision_recall(tuned_guesses, labels_np, c)
        report.append((ANIMALS[c], round(p.item(), 2), round(r.item(), 2)))
    report
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 泄漏：故意骗自己

    **泄漏**（leakage）是指关于答案的信息偷偷进了模型看到的内容。我们来制造一次。下面我们忘了遮住动物的词，训练和测试故事都没遮，得到一个漂亮的分数。然后在遮住的故事上测量同一个模型，那才是真正的任务。
    """)
    return


@app.cell
def _(Classifier, TinyGPT, X_test, Y_test, index, train_examples, vocab):
    chosen_leak = [train_examples[i] for i in np.random.default_rng(0).permutation(len(train_examples))[:300].tolist()]
    X_leak, Y_leak = to_tensors(chosen_leak, [], index["<unk>"], index["<end>"])
    X_test_leak = torch.tensor([(ids[:64] + [index["<end>"]] * 64)[:64] for ids, label in labelled(stories()[1], index)])
    torch.manual_seed(1)
    leaky = Classifier(TinyGPT(len(vocab), 64, 96, 4, 2), 96)
    fit(leaky, X_leak, Y_leak, 15, 0.001)
    share_right(leaky, X_test_leak, Y_test), share_right(leaky, X_test, Y_test)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 81%，然后大约 45%。第一个数字看着像重大突破，其实毫无价值：模型只是读到了答案。在真正的任务上，它只比总说 `dog` 好一点点。测试集只有在干净时才能保护你。

    ## 更隐蔽的泄漏：同一段文本出现两次

    泄漏通常比刚才那个隐蔽。常见的一种是：文本集合里有重复的内容，把集合切成训练和测试两部分时，同一段文本可能两边各落一份。这样一来，模型被测的正是它训练过的文本，连答案都见过。下面这个切分就有这种泄漏：
    """)
    return


@app.function
def leaky_split(texts, n_test):
    cut = len(texts) - n_test
    return texts[:cut], texts[cut:]


@app.cell
def _():
    leaky_split(["a", "b", "a", "c", "b"], 2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：找出泄漏

    看这个单元格的结果：先是训练文本，再是测试文本。3 个训练文本里，有几个也出现在测试部分？把它们去掉之后，训练文本还剩几个？
    """)
    return


@app.cell(hide_code=True)
def _():
    leak_boxes = hand.boxes(['也在测试部分里的训练文本', '剩下的训练文本'])
    return (leak_boxes,)


@app.cell(hide_code=True)
def _(leak_boxes):
    hand.marked(leak_boxes, {'也在测试部分里的训练文本': 1, '剩下的训练文本': 2}, {
        '也在测试部分里的训练文本': r"""测试部分是 `["c", "b"]`。`"a"`、`"b"`、`"a"` 里哪个在里面？""",
        '剩下的训练文本': r"""3 减去你刚才数出的个数""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 7：clean_split

    写 `clean_split(texts, n_test)`，一个没有这种泄漏的切分。`texts` 是一个字符串列表，`n_test` 是一个整数。和 `leaky_split` 一样，它返回一对：先是训练文本，再是测试文本，而且测试部分是最后 `n_test` 个文本。但训练部分要去掉所有也出现在测试部分里的文本。

    > `clean_split(["a", "b", "a", "c", "b"], 2)` 得到 `(["a", "a"], ["c", "b"])`

    我们从训练这一边去掉，而不是从测试那一边，这样不管训练集是什么，测试集都保持不变。
    """)
    return


@app.function
def clean_split(texts, n_test):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("18", "clean_split", clean_split, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""开头和 `leaky_split` 一样，先得到两部分。然后把训练列表重新造一遍，只留下 `not in` 测试部分的文本：用第 12 章那种末尾带 `if` 的造列表表达式。""",
        r"""可以写成 4 行：切分的位置，测试部分，筛过的训练部分，再加 `return`。筛选的写法是 `[t for t in ... if t not in test]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    还有些泄漏更难发现：一个暴露答案的日期，或者在测试集上反复调整一个数字（第 13 章的 $k$）。把一个测试重复到结果好看为止，本身就是泄漏，共享的**基准**（benchmark），也就是大家都用的标准测试集，也因此会慢慢失效。

    ## 两边都看：BERT

    GPT 从左读到右，因为它的工作是写。要理解已有的文本，为什么不两边都看？**BERT** 是没有未来掩码的 Transformer，训练的任务也不同：遮住 15% 的词，从两边预测它们。拿 `the ▢ barked loudly` 里的空白来说：

    | 模型 | 填空时能用到的词 |
    |---|---|
    | GPT | `the` |
    | BERT | `the`、`barked`、`loudly` |

    我们在这里也试过，本以为它能超过之前的 51%，因为它能看到空白后面的 `barked`。在这份数据上它只得了大约 32%，还低于基线。它需要的数据，比小模型加少量文本能提供的更多。所以这是一条教训，而不是配方：在巨大规模上取胜的想法，不保证在小规模上也能赢。

    ## 接下来

    第二部分到此结束。你已经能表示词、给语言模型打分、搭建 Transformer。第三部分会按真实实验室的做法重建你的小 GPT，从最开始的一步做起。我们的词表遇到没见过的词和任何其他语言都会失灵。**第 19 章**用一个真正的**分词器**（tokenizer）取代它。

    ## 你现在会做的事

    - 解释预训练、微调和冻结，以及各自什么时候合适
    - 给预训练好的主体装上一个新的头部
    - 把任何分数都和基线比较
    - 亲手数出真阳性和假阳性，计算精确率和召回率，并说出它们各防什么
    - 估计误差范围，判断两个分数是否真的不同
    - 识别泄漏，并在切分数据时不让同一段文本落在两边
    - 解释 BERT 和 GPT 的区别

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** BERT 出自 Devlin 等（2018）。精确率和召回率可以合成 *F1 分数*。严格比较两个模型在同一测试上的表现，要用 McNemar 这样的*配对*检验。基准有 *SST-2* 和 *SQuAD* 等，把现成模型用到新任务上叫*迁移学习*（transfer learning）。
    """)
    return


if __name__ == "__main__":
    app.run()
