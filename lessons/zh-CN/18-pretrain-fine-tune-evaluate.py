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

    用 190 万个词训练语言模型时不需要标签，因为文本本身就是答案。但大多数有用的任务并非如此：无论是给邮件分类还是识别疾病，都得靠人工标注样本，而标注成本很高。

    行业里的标准做法叫**预训练**（pretraining）：先用大量普通文本训练一个语言模型，再把它用到只有少量标签的任务上。这一章我们就来试一试。不过，这一章真正想讲的是另一件事：**怎样判断一个结果到底靠不靠谱。** 你会亲手故意骗一个模型，见到一个什么也不做却能拿 40% 分数的“模型”，还会看到一个广为流传的说法在数据里站不住脚。

    ## 从故事里造一个任务

    手头没有人工标注的数据，我们就从故事集里现造一个任务。挑出所有只提到下面四种动物之一的故事：`dog`（狗）、`cat`（猫）、`bird`（鸟）、`fish`（鱼），把对应的动物当作标签。然后把这个动物词在故事里全部**遮住**，让模型猜故事讲的是哪种动物。它只能靠 `barked`（吠叫）、`swam`（游泳）这样的上下文线索来推断。（这里只遮这四个完全匹配的词，所以复数 `dogs` 之类的词偶尔还是会漏网。）

    /// admonition | 🧰 工具箱：在两个值里选一个
    `a if test else b` 是一个条件表达式：条件成立时给出 `a`，否则给出 `b`。它可以放进**列表推导式**（list comprehension，第 12 章学过的用方括号直接造列表的写法）里，用来决定每一项：`[0 if x < 0 else x for x in [3, -1, 2]]` 会得到 `[3, 0, 2]`。
    ///

    ## 练习 1：blank_out

    写 `blank_out(ids, hidden, blank)`。`ids` 是词编号列表，`hidden` 是要遮住的编号列表，`blank` 是用来替换的编号。它返回一个长度相同的新列表：

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
        r"""对 `ids` 写一个列表推导式。每一项要么是 `blank`，要么是编号本身；条件是这个编号是否 `in hidden`。""",
        r"""只要一行，写法和工具箱里的例子一样：`return [... if ... else ... for i in ids]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 模型与新的最后一层

    **来自前面的章节。** 下面两个单元格里放的是第 17 章模型的缩小版。这些代码你在上一章全都写过，这里就折叠起来了。有两处小改动：一是模型多了一个方法 `.features(ids)`，用来取出最后一个块输出的向量（还没进入打分层），而 `forward` 就只是 `self.out(self.features(ids))`；二是注意力的三个线性层放进了一个 `nn.ModuleList` 里——这是一种专门的列表，用来告诉 PyTorch 里面的层带有需要训练的旋钮。
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
    要给故事分类，我们在这些特征上面再接一个新的小层。每个样本取故事的前 64 个词，每个词的特征向量有 96 个数字。整体流程是这样的：
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
    GPT 原有的这些块合起来叫**主体**（body），新接上的 `nn.Linear(96, 4)` 叫**头部**（head）。对所有词的特征取平均，就能把任意长度的故事压缩成一个向量，方便头部给它打分——四种动物各得一个分数。

    ## 练习 2：classify

    写 `classify(features, head)`，对应图里的后两个箭头。`features` 的形状是 `(batch, words, d)`，`head` 是一个 `nn.Linear(d, 4)`。函数返回形状为 `(batch, 4)` 的分数。`torch.mean(x, dim=1)` 会沿第 1 维（也就是词这一维）取平均，取完平均后这一维会从形状里消失。
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
        r"""分两步，一步套一步：先对所有词取平均，再把结果传给头部。""",
        r"""只要一行：`return head(...)`，括号里是沿 `dim=1` 对 `features` 做 `torch.mean` 的结果。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `Classifier` 把一个主体和一个头部装在一起，它的 `forward` 调用的就是你刚写的函数。后面的单元格负责准备数据。在练习 1 和练习 2 通过之前，依赖它们的单元格会报错——这很正常，后面的练习也一样。
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
    训练集中有大约 3,300 篇带标签的故事，测试集有 265 篇：狗 107 篇，猫 55 篇，鸟 79 篇，鱼 24 篇。

    /// admonition | 🧰 工具箱：数标签
    `np.bincount(a)` 统计数组里每个非负整数出现的次数：结果里索引 0 处是 0 的个数，索引 1 处是 1 的个数，依此类推。例如 `np.bincount(np.array([0, 2, 2, 1, 2]))` 会得到 `[1, 1, 3]`。上面用到的 `torch.bincount` 是它在 PyTorch 里的对应版本。
    ///

    ## 先定一条基线

    在跑任何模型之前，先要搞清楚“分数高”到底意味着什么。如果测试集里有 40% 的故事讲的是狗，那一个永远只猜 `dog` 的“模型”连一个字都不读，也能拿到 40% 的准确率。这就是**多数类基线**（majority baseline），真正的模型至少得超过它才有意义。反过来也一样：如果 99% 的邮件都是正常邮件，一个什么都不拦截的过滤器也能考出 99% 的准确率。

    ## 练习 3：majority_share

    写 `majority_share(labels)`。`labels` 是一个由非负整数组成的 NumPy 数组。函数返回出现次数最多的那个标签所占的比例，也就是永远只猜这个类别时的准确率。

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
        r"""先用 `np.bincount` 数出每个标签的个数，再用 `np.max` 找出最大值，最后除以标签总数。""",
        r"""只要一行。在上面的例子里，各标签的个数是 `[1, 3, 2]`，最大的是 3，一共有 6 个标签。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 比准确率更细致的指标

    准确率看不出模型到底犯了哪种错。针对某一个类别（比如 `fish`），我们可以把预测结果分成三类来数：

    - **真阳性**（$tp$）：模型猜是鱼，实际上也是鱼；
    - **假阳性**（$fp$）：模型猜是鱼，实际上不是；
    - **假阴性**（$fn$）：实际上是鱼，模型却猜成了别的。

    据此可以算出两个指标：

    > $$\text{精确率} = \frac{tp}{tp + fp} \qquad \text{召回率} = \frac{tp}{tp + fn}$$

    **精确率**（precision）问的是：模型猜是鱼的时候，猜得有多准？**召回率**（recall）问的是：所有真正的鱼里，模型找出来多少？

    一个从不判“垃圾邮件”的过滤器一封垃圾邮件也抓不到，召回率是 0；它一次也没猜过，精确率无从算起，我们同样记 0。相反，一个把所有邮件都判成“垃圾邮件”的过滤器，召回率是 1，精确率却惨不忍睹。这两个指标互相制约，缺一不可。

    ## 动手算：数出这三种情况

    下面有 10 篇故事，每篇都给出了模型的猜测和真实标签。F 代表鱼，D 代表狗。

    | 故事 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |---|---|---|---|---|---|---|---|---|---|---|
    | 猜测 | F | F | F | D | F | D | D | F | D | D |
    | 标签 | F | F | D | D | F | F | D | F | F | D |

    针对“鱼”这一类，数出 $tp$、$fp$ 和 $fn$，再算出精确率和召回率（保留两位小数）。
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
    模型 10 次里猜对了 7 次，但光看 70% 这一个数字，你根本看不出它漏掉了三分之一的鱼。

    /// admonition | 🧰 工具箱：用比较来计数
    - 把数组和一个数做比较，会得到一个由 `True` 和 `False` 组成的数组：`np.array([1, 0, 1]) == 1` 得到 `[True, False, True]`。
    - `&` 表示逐个位置的“并且”：只有两边都是 `True` 的位置才是 `True`。每个比较都要各自加上括号：`(a == 1) & (b != 1)`。
    - 对这样的数组调用 `np.sum`，就会数出里面 `True` 的个数。
    ///

    ## 练习 4：precision_recall

    写 `precision_recall(guesses, labels, c)`。`guesses` 和 `labels` 是长度相同的 NumPy 数组，存着类别编号；`c` 是要评估的类别。函数返回一对值：先是精确率，再是召回率。如果某个分数的分母是 0，就把该指标记为 0——用 `max(n, 1)` 当分母是个简便写法，因为分母为 0 时分子也一定是 0。

    把鱼记作类别 1、狗记作类别 0，上面那张表就在测试用例里，结果大约是 `(0.8, 0.67)`。
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
        r"""算出三个计数，每个都是对两个用 `&` 连起来的比较做 `np.sum`。`tp` 要求 `guesses` 和 `labels` 都等于 `c`；`fp` 要求猜测等于 `c` 而标签不等于（`!=`）；`fn` 正好反过来。""",
        r"""一共 4 行。第一行是 `tp = np.sum((guesses == c) & (labels == c))`。最后一行返回两个分数，用逗号隔开，精确率在前，两个分母都写成 `max(..., 1)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 差别是真的吗？

    测试集只有 265 篇故事，分数的波动不小。想知道波动有多大，可以假设一个模型答对每篇故事的概率正好是 0.6，也就是说它的真实水平是 60%。现在拿 200 个不同的测试集去测它，每个测试集有 265 篇故事。下面的单元格用随机数来模拟这件事（`rng.random(shape)` 生成 0 到 1 之间的随机数，所以每个数小于 0.6 的概率正好是 0.6），并输出这 200 个分数里的最低值和最高值。
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
    同一个模型，测出来的分数从大约 52% 到 69% 都有，差别全在于测试集里碰巧抽到了哪些故事。所以单看一个分数，其实是“真实水平”混着“运气成分”，我们得弄清楚运气最多能带来多大波动。

    把每篇故事想象成抛一枚硬币：以概率 $p$ 落成“答对”（1），其余情况是“答错”（0）。单次结果的方差是 $p \times (1 - p)$（这个公式直接拿来用就行）。接下来按第 09 章的两条规则推两步：独立随机量的方差可以直接相加，所以 $n$ 篇故事答对总数的方差是 $n \times p \times (1 - p)$；再除以 $n$ 求平均分，方差要除以 $n^2$（因为方差是对偏差求平方，整体除以 $n$，平方后就除以 $n^2$）。约分后剩下 $p \times (1 - p) / n$，开根号就是测出的分数通常会偏离真值多远。

    统计学里有一个常用结论：取这个偏离幅度的 2 倍，就能覆盖大约 95% 的情况。这就是常见的**误差范围**（margin of error）：

    > $$\text{误差范围} = 2 \times \sqrt{\frac{p \times (1 - p)}{n}}$$

    ## 动手算：两个误差范围

    一个模型在 $n = 100$ 个样本的测试上得分 $p = 0.5$。误差范围是多少？样本量翻到四倍、$n = 400$ 时又是多少？
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
    数据量翻到 4 倍，误差范围只缩小一半。这也是为什么小测试集的噪声很难摆脱。

    ## 练习 5：margin

    写 `margin(p, n)`，实现上面的公式。`p` 是 0 到 1 之间的分数，`n` 是测试样本的数量。平方根用 `np.sqrt`。

    > `margin(0.5, 100)` 得到 `0.1`

    检查后面的单元格会统计：刚才模拟出的 200 个分数里，有多少落在真值 0.6 的一个误差范围以内。
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
        r"""只要一行。把分数放进 `np.sqrt` 里，结果再乘以 2。""",
        r"""如果 `margin(0.5, 100)` 得到 0.005，说明漏了平方根；如果得到 0.05，说明漏了乘以 2。""",
    ], LOCALE)
    return


@app.cell
def _(test_scores):
    np.mean(np.abs(test_scores - 0.6) < margin(0.6, 265)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 0.95，和前面说的一致：每 20 个分数里大约有 19 个落在误差范围以内，1 个落在外面。

    当 $p = 0.6$、$n = 265$ 时，误差范围大约是 0.06，也就是 6 个百分点。比较两个模型的分数时，两边各有各的噪声，差值的误差范围要乘以约 1.4 倍（两个独立波动的方差相加变成 2 倍，开根号后误差范围就变成 $\sqrt{2} \approx 1.4$ 倍）：这里大约是 8.5 个百分点。所以在这个测试集上，58% 和 61% 的两个模型根本分不出高下。
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 预训练

    点击下面的按钮，就会用全部训练故事对这个缩小版 GPT 做预训练（完全不用标签），大约需要 2 到 4 分钟。训练完成前，依赖它的单元格会显示 `Ancestor stopped`，这很正常。（模型见过的是和测试故事同*类型*的文本，并没有见过测试故事本身：送进去的只有 `train_stories`。）
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
    `model.state_dict()` 会返回一个字典，按名字存着模型里的全部旋钮；`model.load_state_dict(saved)` 则把存下来的一套旋钮装回结构相同的模型里。
    ///

    有了 `saved`，我们就能从同一个预训练好的起点，分头训练好几个模型。上面显示的数字，就是这个字典里一共存了多少组旋钮。

    ## 完全不用标签

    语言模型甚至不用任何标签就能尝试这个任务。在每个被遮住的空白处，我们让模型给出四种动物作为下一个词的概率，把整篇故事里所有空白处的对数概率加起来，选总分最高的那种动物。读下面代码时留意两点：因为 GPT 是在前一个位置 `t - 1` 预测位置 `t` 的词，所以空白出现在 `t` 时要取 `chances[t - 1]`；而把多个空白处的对数概率相加，等价于把它们的概率相乘（第 13 章）。再下一个单元格会依次显示多数类基线、无标签预测的正确率以及它的误差范围。
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
    多数类基线是 40%。预训练模型不用任何标签就拿到了大约 51%，高出基线的幅度超过了 6 个百分点的误差范围。这说明预训练确实学到了有用的东西。

    ## 微调

    现在用上标签。**微调**（fine-tuning）是指从预训练好的旋钮出发，装上一个新的最后一层，再用带标签的样本轻轻地训练。这样做一定比从随机旋钮从头训练更好吗？还有一种折中做法：**冻结**（freeze）主体，只让新加的头部去学。对照前面的两段结构，三种做法分别是：

    | | 主体 | 头部 |
    |---|---|---|
    | **从零开始** | 随机初始化，参加训练 | 随机初始化，参加训练 |
    | **冻结主体** | 预训练好的，不训练 | 随机初始化，参加训练 |
    | **微调** | 预训练好的，轻轻地训练 | 随机初始化，参加训练 |

    要冻结一个旋钮，只要把它的 `requires_grad` 设为 `False`（第 10 章里用来控制 PyTorch 是否追踪梯度的开关），这样误差的责任传不到它，优化器也不会更新它。

    ## 练习 6：freeze

    写 `freeze(model)`，把模型里每个旋钮的 `requires_grad` 都设为 `False`。这个函数直接在原模型上修改，不需要 `return`。遍历模型所有旋钮可以用 `model.parameters()`（第 10 章）。
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
        r"""用 `for` 循环遍历 `model.parameters()`，循环体里只有一行，把 `requires_grad` 设为 `False`。""",
        r"""一共 2 行。赋值用一个 `=`，就像 `p.requires_grad = False`；两个 `==` 只是做比较。""",
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
    下一个单元格会在 100、300、1,000 篇带标签的故事上测试这三种做法，大约需要一分钟。每种做法都配了适合它的超参数：冻结主体时只训练最后一层，所以训练轮次更多；微调则用很小的学习率，免得把预训练好的旋钮改乱。运行之前先猜一猜：哪一列会赢？
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
    现在结合练习 5 的结论来读这张表。（两格数字相同纯属巧合。）在 100 和 300 个标签时，三者差距只有几个百分点，远在约 8.5 个百分点的误差范围以内，根本分不出高下。在 1,000 个标签时，从零开始领先 7 到 8 个百分点，刚好卡在误差范围的边缘：也许是真的，但还不能下定论。

    **所以在这个任务上，预训练并没有带来明显优势。** 写这一章时，我们原本以为会看到教科书般的结论——预训练明显胜出，还试了好几组参数，但这个结果始终没出现。这正是小模型在少量文本上预训练的真实表现；学会判断预训练在什么任务上才能发挥威力，本身就是做模型的必修课。只有当预训练模型远大于你靠手头标签能独自训练的模型、而且标签又非常稀缺时，预训练才真正划算。回头看，前面那个 51% 的亮点不在于分数有多高，而在于它一个标签都没用。

    下面列出 300 个标签时的微调模型在每个类别上的表现，每行是（动物，精确率，召回率）。留意样本最少的 `fish`——只看总准确率完全看不出模型在它上面的表现。不过鱼的故事只有 24 篇，所以这两个数字本身波动也很大。
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

    **泄漏**（leakage）是指答案的信息偷偷混进了模型的输入里。我们来亲手制造一次：下面故意忘了遮住动物词，训练和测试故事都没遮，于是得到了一个很漂亮的分数；接着再把同一个模型放到遮住动物词的测试集上评估，那才是真正的任务。
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
    先是大约 81%，接着掉到大约 45%。第一个数字看着像重大突破，其实毫无价值——模型只是直接读到了答案。到了真正的任务上，它只比永远猜 `dog` 强一点点。只有干净的测试集，才能帮你守住底线。

    ## 更隐蔽的泄漏：同一段文本出现两次

    现实里的泄漏通常没这么明显。常见的一种情况是：原始文本里混着重复内容，切分训练集和测试集时，同一段文本可能两边各分到一份。这样一来，模型考试时碰到的正是它背过的原题。下面这个切分就有这种泄漏：
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

    看上面单元格的输出：前面是训练集，后面是测试集。3 条训练文本里，有几条也出现在测试集里？把它们剔除后，训练集还剩几条？
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

    写 `clean_split(texts, n_test)`，做一个没有这种泄漏的切分。`texts` 是字符串列表，`n_test` 是整数。和 `leaky_split` 一样，它返回一对列表（训练集在前，测试集在后），测试集取最后 `n_test` 条文本；但训练集要剔除所有在测试集里出现过的文本。

    > `clean_split(["a", "b", "a", "c", "b"], 2)` 得到 `(["a", "a"], ["c", "b"])`

    注意我们是从训练集里剔除，而不是从测试集里剔除——这样无论训练集怎么变，测试集都保持不变。
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
        r"""开头和 `leaky_split` 一样，先切出两部分。再用第 12 章学过的末尾带 `if` 的列表推导式重新构造训练集，只保留 `not in` 测试集的文本。""",
        r"""可以写成 4 行：算出切分位置、切出测试集、过滤出训练集，再 `return`。过滤的写法是 `[t for t in ... if t not in test]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    还有些泄漏更隐蔽：比如数据里带了一个会暴露答案的日期，或者在测试集上反复调参（就像第 13 章的 $k$）。把测试集反复用到结果好看为止，本身就是一种泄漏；大家公用的标准测试集——**基准**（benchmark），也正是因此会慢慢失效。

    ## 两边都看：BERT

    GPT 只能从左往右读，因为它的任务是续写文本。如果要理解一段已经写好的文字，为什么不两边一起看？**BERT** 就是去掉了未来掩码的 Transformer，训练任务也换成了“完形填空”：随机遮住 15% 的词，同时结合左右两边来预测。拿 `the ▢ barked loudly` 里的空白来说：

    | 模型 | 填空时能用到的词 |
    |---|---|
    | GPT | `the` |
    | BERT | `the`、`barked`、`loudly` |

    我们在这里也试过这种做法，本以为它能超过前面的 51%（毕竟它能看到空白后面的 `barked`）。结果在这份数据上它只拿到约 32%，甚至低于基线——它需要的数据量远超小模型加小语料能提供的规模。所以请把这当作一条经验，而不是死板的配方：在大规模上大获全胜的方法，并不保证在小规模上同样奏效。

    ## 接下来

    第二部分到此结束。你已经学会了怎样表示词、给语言模型打分，并亲手搭出了 Transformer。第三部分会照着真实实验室的做法，从第一步开始重建你的小 GPT。现在这套词表一遇到生词或者其他语言就会失灵，**第 19 章**我们会用真正的**分词器**（tokenizer）来取代它。

    ## 你现在会做的事

    - 解释预训练、微调和冻结各自的做法，以及什么时候该用哪一种
    - 给预训练好的主体接上一个新的头部
    - 把任何分数都先和基线比一比
    - 亲手数出真阳性、假阳性和假阴性，算出精确率和召回率，并说明它们各自防什么错
    - 估算误差范围，判断两个分数之间是不是真的有差距
    - 识别数据泄漏，并在切分数据时避免同一段文本落入两边
    - 说明 BERT 和 GPT 的区别

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** BERT 出自 Devlin 等（2018）。精确率和召回率可以综合成一个指标——*F1 分数*。在同一个测试集上严格比较两个模型，通常会用 McNemar 检验这类*配对*检验。常见基准包括 *SST-2* 和 *SQuAD*，而把现成模型用到新任务上的做法统称为*迁移学习*（transfer learning）。
    """)
    return


if __name__ == "__main__":
    app.run()
