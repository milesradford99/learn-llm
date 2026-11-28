import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="12 · 把词变成向量")

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
    from learn_llm.plot import chart

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 12 · 把词变成向量

    从这一章开始，我们进入课程的第二部分：语言。到最后一章为止，我们都会用同一份文本——一批简短的小故事，叫做 **TinyStories**，这样就能把每个新模型和前面的模型放在一起比较。这份数据已经随课程打包好了，所有语言版本用的都是同一份英文原文。下面是 12,000 篇训练故事中第一篇的开头（不用逐句读懂，把它当成一段普通文本就行；另外 1,000 篇 `test_stories` 留到第 13 章做测试用）。
    """)
    return


@app.cell
def _():
    train_stories, test_stories = stories()
    train_stories[0][:300]
    return (train_stories,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    神经网络只能处理数字，所以第一个问题就是怎样把词变成数字。最直接的办法是第 07 章的 one-hot 编码（比如词表只有 4 个词时，第 2 个词表示为 `[0, 1, 0, 0]`）：如果认识 2,000 个词，每个词就是一行 2,000 个数字，其中只有一个 1。但这对网络几乎没有帮助——任意两个不同词的 one-hot 向量点积都是 0，也就是说，在网络眼里，`dog` 和 `cat` 的差别，跟 `dog` 和 `yesterday` 一样大。

    我们想要更好的表示：一个简短的**词向量**（word vector），比如 32 个数字，并且让意思相近的词拥有相近的向量。这些数字靠人工肯定写不出来，但有个巧妙的办法可以只靠文本把它们学出来，核心直觉就一句话：**看一个词常和谁做邻居，就能了解这个词。** 即使遮住下面句子里的一个词，你也能猜出它大概是什么词：

    > The ▢ barked at the mailman and wagged its tail.（这只 ▢ 冲着邮递员叫，还摇着尾巴。）

    经常出现在相同邻居周围的词，意思往往相近。所以我们给网络布置一个任务：根据一个词去预测它的邻居，训练完后，把网络在这个过程中学出来的向量留下来。这就是著名的 **word2vec** 方法。

    ## 从文本到单词

    先把文本切分成单词。下面这个辅助函数会把所有字母转为小写，只保留连续的字母和撇号（比如 `don't` 里的 `'`），其余标点全部丢掉。（`re` 是 Python 的文本匹配工具，这里不用专门学；`[a-z']+` 的意思就是“一个或多个字母或撇号”。）

    /// admonition | 🧰 工具箱：两个处理文本的工具
    - `text.lower()` 把文本里的大写字母全部转为小写：`"The Dog".lower()` 得到 `"the dog"`。
    - `" ".join(items)` 用空格把列表里的字符串拼成一个长字符串：`" ".join(["a", "b", "c"])` 得到 `"a b c"`。
    ///
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.cell
def _(train_stories):
    words = to_words(" ".join(train_stories[:6000]))
    len(words), words[:12]
    return (words,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    6,000 篇故事大约有 965,000 个词，这一章已经够用了。用全部 12,000 篇效果差不多，只是训练时间会翻倍。

    ## 用字典计数

    我们需要统计每个词出现了多少次。列表不适合干这个：列表只能按位置（第几个）查找，而我们想直接按单词来查。Python 为此提供了**字典**（dictionary）：它把一条条**键**（key）和**值**（value）配对存起来，给它一个键，就能立刻查出对应的值。

    /// admonition | 🧰 工具箱：字典
    - `{}` 创建一个空字典。`pets["dog"] = 3` 把键 `"dog"` 对应的值设为 3；如果 `"dog"` 之前已经有值，新值会直接覆盖旧值。
    - `pets["dog"]` 读取 `"dog"` 对应的值。如果读取一个不存在的键，程序会报错停下。
    - `pets.get("fish", 0)` 也是读取，但如果键不存在，不会报错，而是返回你给的默认值（这里是 0）。
    - `for name in pets:` 按键第一次加入的先后顺序，依次遍历字典里的所有键。
    ///

    下面的单元格先记下 3 只狗，再分两次各加上 2 只猫。第一次加猫时，字典里还没有 `"cat"` 这个键，所以 `.get` 返回默认值 0，猫的数量就从 0 开始累加。
    """)
    return


@app.cell
def _():
    pets = {}
    pets["dog"] = 3
    pets["cat"] = pets.get("cat", 0) + 2
    pets["cat"] = pets.get("cat", 0) + 2
    pets, pets["dog"], pets.get("fish", 0)
    return (pets,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：count_words

    写 `count_words(words)`。它接收一个单词列表，返回一个字典，记录每个不同的词各出现了多少次：

    > `count_words(["the", "dog", "the"])` 得到 `{"the": 2, "dog": 1}`

    从一个空字典开始，把单词逐个看一遍。每遇到一个词，就把它对应的计数加 1。想一想：一个词第一次出现时，它原来的计数是多少？
    """)
    return


@app.function
def count_words(words):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "count_words", count_words, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""分三步：创建一个空字典，遍历 `words`，循环结束后返回这个字典。上面工具箱单元格里数猫的写法，演示了怎样给一个可能还不存在的键累加计数。""",
        r"""一共 4 行。第一行是 `counts = {}`。循环里面只有一行：用 `.get` 查出当前计数再加 1，写回键 `word` 对应的值。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 词表

    接下来我们留下最常见的 2,000 个词作为**词表**（vocabulary），并给每个词分配一个数字**编号**（id），也就是它在词表里的位置。这里会用到两个小工具。

    /// admonition | 🧰 工具箱：排序和筛选
    - `sorted(pets)` 把所有键按字母顺序排成一个列表。`sorted(pets, key=pets.get)` 则按 `pets.get` 查出的值来排序（默认从小到大）；加上 `reverse=True`，就是从大到小排。
    - `[name for name in names if name in pets]` 像第 05 章的列表推导式一样生成新列表，但只保留满足 `if` 条件的项。字典里包含这个键时，`name in pets` 为真。
    ///
    """)
    return


@app.cell
def _(pets):
    sorted(pets, key=pets.get, reverse=True), [name for name in ["dog", "owl", "cat"] if name in pets]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下一个单元格会调用你写的 `count_words`，所以在练习 1 通过之前它会报错。这是正常现象，后面依赖它的单元格也一样。它生成三样东西：

    - `vocab`：最常见的 2,000 个词，按出现次数从高到低排列；
    - `index`：反查字典，从单词映射到它的编号；
    - `ids`：把整段文本转成编号列表，不在词表里的词直接跳过。

    最后一行显示总共有多少个不同的词、`vocab` 的前 8 个词，以及 `ids` 的总长度。
    """)
    return


@app.cell
def _(words):
    counts = count_words(words)
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    ids = [index[word] for word in words if word in index]
    len(counts), vocab[:8], len(ids)
    return counts, ids, index, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    语料里一共出现了大约 6,800 个不同的词，而前 2,000 个词就已经覆盖了全文 97% 的内容。小故事的词汇量小，很适合在普通笔记本电脑上跑。

    ## 两个向量有多像？

    第 05 章学过的点积，是把两个向量逐个位置相乘再相加。两个向量方向越一致，点积就越大。但它也会随着向量变长而变大，而衡量“像不像”时，我们只关心方向。所以要把点积除以两个向量各自的长度。

    /// admonition | 🧰 工具箱：向量的长度
    用 $\cdot$ 表示点积（代码里写作 `@`），向量 $a$ 的**长度**是 $\sqrt{a \cdot a}$，也就是各项平方和再开方。对 $[3, 4]$ 来说就是 $\sqrt{9 + 16} = 5$——在坐标图上，它正好是从原点 $(0, 0)$ 到点 $(3, 4)$ 的那条箭头的长度（勾股定理）。
    ///

    这个指标叫**余弦相似度**（cosine similarity）：

    > $$\operatorname{similarity}(a, b) = \frac{a \cdot b}{a \text{ 的长度} \times b \text{ 的长度}}$$

    方向完全相同时结果是 1，互相垂直（毫无共同点）时是 0，方向完全相反时是 −1。拿 $a = [3, 4]$ 和 $b = [6, 8]$ 来算：点积是 $3 \times 6 + 4 \times 8 = 50$，两个长度分别是 $\sqrt{9 + 16} = 5$ 和 $\sqrt{36 + 64} = 10$，所以相似度为 $50 / (5 \times 10) = 1$。因为 $b$ 只是把 $a$ 拉长了一倍，方向并没有变。

    ## 动手算：两个相似度

    先看 $a = [3, 4]$ 和 $b = [4, 3]$：同样的两个数，互换了位置。算出它们的点积、两个长度和相似度。然后算 $c = [0, 2]$ 和 $d = [3, 4]$ 的相似度。
    """)
    return


@app.cell(hide_code=True)
def _():
    cosine_boxes = hand.boxes(['a · b', 'a 的长度', 'b 的长度', 'a 和 b 的相似度', 'c 和 d 的相似度'], step=0.01)
    return (cosine_boxes,)


@app.cell(hide_code=True)
def _(cosine_boxes):
    hand.marked(cosine_boxes, {'a · b': 24, 'a 的长度': 5, 'b 的长度': 5, 'a 和 b 的相似度': 0.96, 'c 和 d 的相似度': 0.8}, {
        'a · b': r"""$3 \times 4 + 4 \times 3$""",
        'a 的长度': r"""$\sqrt{3 \times 3 + 4 \times 4}$""",
        'b 的长度': r"""两个数换了位置，平方和不变""",
        'a 和 b 的相似度': r"""点积除以两个长度的乘积""",
        'c 和 d 的相似度': r"""点积是 $0 \times 3 + 2 \times 4$，长度是 2 和 5""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    互换位置的两个数，指的方向很接近，但并不相同：相似度是 0.96，不是 1。

    ## 练习 2：similarity

    写 `similarity(a, b)`，输入是两个大小相同的 NumPy 向量，按上面的公式返回一个数。`np.sqrt` 用于开平方根（第 09 章）。

    > `similarity(np.array([3.0, 4.0]), np.array([4.0, 3.0]))` 得到 `0.96`
    """)
    return


@app.function
def similarity(a, b):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "similarity", similarity, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""你需要三个数，每个都用一次 `@` 算出：$a \cdot b$、$a \cdot a$ 和 $b \cdot b$。后两个再套上 `np.sqrt`，就得到了两个长度。""",
        r"""可以写成 3 行：两个长度各一行，最后 `return`。第一行是 `length_a = np.sqrt(a @ a)`。两个长度相乘的部分要加括号，否则 Python 会先除以第一个，再乘以第二个。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 任务：猜邻居

    训练样本是一对对单词编号。我们在文本上逐词滑动：每个词轮流当**中心词**（centre word），它左右各 `window` 个位置以内的词都是它的**上下文词**（context word）。比如对文本 `the dog ran fast`，取 `window = 1`：

    | 中心词 | 上下文词 | 词对 |
    |---|---|---|
    | `the` | `dog` | (the, dog) |
    | `dog` | `the`、`ran` | (dog, the)、(dog, ran) |
    | `ran` | `dog`、`fast` | (ran, dog)、(ran, fast) |
    | `fast` | `ran` | (fast, ran) |

    头尾两个词的上下文词少一些，因为窗口超出了文本边界。根据中心词去预测上下文词，这个任务就叫 **skip-gram**。

    ## 动手算：数一数词对

    取一段 5 个词的文本，`window = 2`，也就是每个中心词往左最多看 2 个位置，往右最多看 2 个位置。第一个词有几个上下文词？中间那个词呢？整段文本一共给出多少个词对？
    """)
    return


@app.cell(hide_code=True)
def _():
    pair_boxes = hand.boxes(['第一个词的上下文词', '中间那个词的上下文词', '词对总数'])
    return (pair_boxes,)


@app.cell(hide_code=True)
def _(pair_boxes):
    hand.marked(pair_boxes, {'第一个词的上下文词': 2, '中间那个词的上下文词': 4, '词对总数': 14}, {
        '第一个词的上下文词': r"""它左边什么都没有，所以只算右边的词""",
        '中间那个词的上下文词': r"""它两边各有 2 个词""",
        '词对总数': r"""五个词分别有 2、3、4、3、2 个上下文词""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：别越出列表边界
    - `max(a, b)` 返回较大的数，`min(a, b)` 返回较小的数。用它们可以把位置“卡”在列表范围内（比如长度 10 的列表）：`max(0, 1 - 3)` 是 0，不会变成负数；`min(10, 8 + 3)` 是 10，不会越过末尾。
    - `range(start, stop)` 从 `start` 数到 `stop` 的前一个：`range(2, 5)` 给出 2、3、4。如果想把位置 5 也包含进来，就写 `range(2, 5 + 1)`。
    - `a != b` 表示“a 不等于 b”，正好和 `==` 相反。
    ///

    ## 练习 3：skipgram_pairs

    写 `skipgram_pairs(ids, window)`。`ids` 是一个编号列表，`window` 是一个整数。它返回两个长度相同的列表：中心词列表和上下文词列表，两个列表里同一个位置 $p$ 上的两项合起来就是一个词对。中心词按文本顺序排列，每个中心词的上下文词从左到右排列。把上面表格里的四个词记作编号 0 到 3：

    > `skipgram_pairs([0, 1, 2, 3], 1)` 得到 `([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])`

    对位置 $i$ 上的中心词，它的上下文词落在从 $i - \text{window}$ 到 $i + \text{window}$ 之间、且在列表范围内的位置上，位置 $i$ 自己除外。
    """)
    return


@app.function
def skipgram_pairs(ids, window):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "skipgram_pairs", skipgram_pairs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先造两个空列表，再写两层循环，一层套一层：`i` 走遍每个位置，`j` 走遍 `i` 附近的位置。在两层循环里面，用一个 `if` 排除等于 `i` 的 `j`，再用两行 `append` 给两个列表各加一项。""",
        r"""一共 8 行。里层循环是一个 `range`：起点是 `i - window`，用 `max` 保证不小于 0；终点是 `i + window + 1`，用 `min` 保证不超过 `len(ids)`。加进列表的是 `ids[i]` 和 `ids[j]`，是编号，不是位置。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 真的还是编的？换个更省事的问题

    每个词都有两个向量：当中心词时用一个（$u$），当上下文词时用另一个（$v$），词义最终会沉淀在 $u$ 里。我们希望当 $v$ 确实是 $u$ 的邻居时，点积 $u \cdot v$ 尽量高。

    最直接的做法还是第 07 章那一套：给词表里全部 2,000 个词打分，再做 softmax。但每遇到一对词都要算 2,000 次点积，而训练词对有几百万个，计算量太大了。所以我们换一个便宜得多的问题：*这一对词是真的邻居吗？* 每次拿到一个真实词对，再随机抽几个**噪声词**（noise word）充当假邻居，训练网络把真邻居和假邻居区分开。这种做法叫**负采样**（negative sampling）。

    ## 从一个分数得到概率

    回答“是”或“否”，需要把单个分数 $s$ 转成一个概率。我们可以对两个分数做 softmax：“真”的分数是 $s$，“假”的分数定为 0。（为什么可以定为 0？第 07 章讲过 softmax 的平移不变性——给所有分数同时加减同一个常数，算出的概率不变，所以不妨把其中一个基准定为 0。）因为任何数的 0 次方都是 1（$e^0 = 1$），判为“真”的概率就是：

    > $$\sigma(s) = \frac{e^s}{e^s + e^0} = \frac{e^s}{e^s + 1}$$

    这个函数叫 **sigmoid**，用希腊字母 $\sigma$（读作 sigma）表示。下面是分数从 −6 到 6 时它的曲线：
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-6, 6), (0, 1), "分数 s", "σ(s)",
          curve=[(x / 10, float(1 / (1 + np.exp(-x / 10)))) for x in range(-60, 61)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $s$ 很大时它接近 1，$s = 0$ 时正好是 0.5，$s$ 很负时接近 0。这条曲线左右完全对称：判为“假”的概率是 $1 - \sigma(s)$，而它正好等于 $\sigma(-s)$。

    先看一个具体数字：当 $s = 2$ 时，判为“真”的概率 $\sigma(2) \approx 0.88$，那么判为“假”的概率就是 $1 - 0.88 = 0.12$——对照上面的曲线，这正好等于 $\sigma(-2)$（$\sigma(2)$ 比 1 低多少，$\sigma(-2)$ 就比 0 高多少）。

    用代数验证也很简单。把 $\sigma(-s)$ 的分子分母同乘 $e^s$（记住 $e^{-s} \times e^s = e^0 = 1$）：

    > $$\sigma(-s) = \frac{e^{-s}}{e^{-s} + 1} = \frac{e^{-s} \times e^s}{(e^{-s} + 1) \times e^s} = \frac{1}{1 + e^s}$$

    再把 $1 - \sigma(s)$ 通分：

    > $$1 - \sigma(s) = 1 - \frac{e^s}{e^s + 1} = \frac{(e^s + 1) - e^s}{e^s + 1} = \frac{1}{e^s + 1}$$

    两式完全相等，所以 $1 - \sigma(s) = \sigma(-s)$ 对任意分数 $s$ 都成立。

    ## 一对词的损失

    损失沿用第 07 章的“意外程度”：把模型给正确答案的概率取 $-\log p$（概率越小，$-\log p$ 越大，损失就越高），再把 1 个真实词和 $k$ 个噪声词的损失加在一起。对真实词，正确答案是“真”，概率是 $\sigma(u \cdot v)$；对噪声词 $n$，正确答案是“假”，概率是 $1 - \sigma(u \cdot n) = \sigma(-u \cdot n)$：

    > $$\text{一对词的损失} = -\log \sigma(u \cdot v) - \log \sigma(-u \cdot n_1) - \ldots - \log \sigma(-u \cdot n_k)$$

    留意噪声项里有两个负号，它们各有分工：**内层负号**把噪声词的高分变低分（这样噪声词得分越低，判为“假”的概率 $\sigma(-u \cdot n)$ 就越高）；**外层负号**把原本为负的对数变成正的损失（因为概率在 0 到 1 之间，取 $\log$ 总是负数）。第一项把 $u$ 拉向真邻居，后面各项则把 $u$ 从假邻居旁边推开。下面是几个分数对应的概率和损失 $-\log \sigma(s)$：

    | $s$ | −2 | −1 | 0 | 1 | 2 |
    |---|---|---|---|---|---|
    | $\sigma(s)$ | 0.12 | 0.27 | 0.50 | 0.73 | 0.88 |
    | $-\log \sigma(s)$ | 2.13 | 1.31 | 0.69 | 0.31 | 0.13 |

    ## 动手算：一对词的损失

    取 $u = [1, 2]$，真实的上下文词 $v = [2, 0]$，一个噪声词 $n = [1, -1]$。先算出两个分数 $u \cdot v$ 和 $u \cdot n$，再从表里查对应的损失并相加。留意噪声分数前面的负号。

    然后假设噪声词变成了 $[2, 0]$，和真实的上下文词一模一样。这时损失是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_boxes = hand.boxes(['真实分数', '噪声分数', '损失', '换了噪声词之后的损失'], step=0.01)
    return (loss_boxes,)


@app.cell(hide_code=True)
def _(loss_boxes):
    hand.marked(loss_boxes, {'真实分数': 2, '噪声分数': -1, '损失': 0.44, '换了噪声词之后的损失': 2.26}, {
        '真实分数': r"""$1 \times 2 + 2 \times 0$""",
        '噪声分数': r"""$1 \times 1 + 2 \times (-1)$""",
        '损失': r"""真实词的损失是表里 $s = 2$ 那一格。噪声词的是它的分数取负之后那一格，也就是 $s = 1$。把两个加起来。""",
        '换了噪声词之后的损失': r"""现在噪声分数是 2，所以它的损失是表里 $s = -2$ 那一格""",
    }, LOCALE, tolerance=0.011)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第二个损失是第一个的五倍。噪声词的分数和真邻居一样高，正是损失要重罚的情况；训练会把 $u$ 从这个假邻居旁边推开，让损失降下来。

    ## 一次算一整批

    练习里要一次算出很多对词的损失，所以向量会叠成批次送进来：`u` 和 `v` 的形状是 `(b, d)`，表示 `b` 对词、每个向量有 `d` 个数字；`n` 的形状是 `(b, k, d)`，表示每对词配有 `k` 个噪声向量。（这里的 `n` 指 noise 噪声，不是前面章节里的样本数。）

    /// admonition | 🧰 工具箱：成批算点积
    - 对两个形状为 `(b, d)` 的张量，`torch.sum(x * y, dim=1)` 先逐个位置相乘，再沿第 1 维（每一行内部）相加，得到每行一个点积，形状是 `(b,)`。
    - `x.unsqueeze(1)`（第 11 章）会在第 1 维插入一个长度为 1 的维度，把 `(b, d)` 变成 `(b, 1, d)`。张量相乘时，长度为 1 的维度会自动广播（沿这一维对齐到另一边的长度 `k`），所以 `(b, 1, d)` 和 `(b, k, d)` 可以直接相乘，得到 `(b, k, d)`——相当于让每对的中心词向量分别与属于它的 `k` 个噪声向量逐元素相乘。
    - `F.logsigmoid(s)` 直接对张量里的每个数计算 $\log \sigma(s)$，分数很大或很负时也不会溢出（和第 07 章 softmax 同加减常数防溢出的思路类似）。
    ///

    下面的单元格用 2 行、每行 3 个数字，和一个形状为 `(2, 4, 3)` 的张量演示前两个工具，并输出三个结果的形状。
    """)
    return


@app.cell
def _():
    rows_demo = torch.ones(2, 3)
    stack_demo = torch.ones(2, 4, 3)
    product_demo = stack_demo * rows_demo.unsqueeze(1)
    torch.sum(rows_demo * rows_demo, dim=1).shape, product_demo.shape, torch.sum(product_demo, dim=2).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：pair_loss

    写 `pair_loss(u, v, n)`，输入形状如上。它返回一个数（没有维度的标量张量）：这 `b` 对词各自损失的平均值。

    为了看清三维张量的形状流转，不妨想象一个极简批次：`b = 2`（2 对词）、`k = 3`（每对 3 个噪声词）、`d = 2`（每个向量 2 个数）：

    - `u` 的形状是 `(2, 2)`，经过 `u.unsqueeze(1)` 插入中间维，变成 `(2, 1, 2)`；
    - 与 `n`（形状 `(2, 3, 2)`）逐元素相乘，广播对齐后得到 `(2, 3, 2)`；
    - 沿 `dim=2` 求和：把每组 `d = 2` 个数加成一个点积，得到 `(2, 3)`（即 `(b, k)`，每对词有 `k` 个噪声分数）；
    - 对噪声分数取负并套上 `F.logsigmoid` 后，再沿 `dim=1` 求和：把每对词的 `k = 3` 个噪声词损失加在一起，得到 `(2,)`（即 `(b,)`，正好和真实词损失的形状对齐）。

    整个计算过程经过的形状如下：

    | 算什么 | 形状 |
    |---|---|
    | 真实分数 $u \cdot v$，每对一个 | `(b,)` |
    | 噪声分数 $u \cdot n_j$，每对 `k` 个 | `(b, k)` |
    | 按公式算出的每对词的损失 | `(b,)` |
    | 它们的平均值 | 一个数 |

    你在动手算里得到的结果就是其中一项检查：只有 1 对词的一批，`u = [[1, 2]]`，`v = [[2, 0]]`，`n = [[[1, -1]]]`，结果大约是 `0.44`。
    """)
    return


@app.function
def pair_loss(u, v, n):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "pair_loss", pair_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""四行，表格的每一行对应一行代码。前两行的写法在工具箱单元格里。第三行：对真实分数和取负后的噪声分数分别用 `F.logsigmoid`，把每对的 `k` 个噪声项沿 `dim=1` 加起来，两部分前面都加负号。""",
        r"""第一行是 `real = torch.sum(u * v, dim=1)`。算噪声分数时，`u` 要先 `unsqueeze(1)` 才能和 `n` 相乘，然后沿 `dim=2` 求和。最后一行返回这些损失的 `torch.mean`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 训练

    最后一个 PyTorch 工具：`nn.Embedding(2000, 32)` 是一张 2,000 行、每行 32 个可调参数（旋钮）的表。给它任意形状的编号张量，它就会按编号查出对应的行：`(b,)` 的编号查出 `(b, 32)` 的向量，`(b, k)` 的编号查出 `(b, k, 32)` 的向量。本质上就是一次查表，只不过表里的数字会在训练中不断学习更新。

    /// admonition | 🧰 工具箱：小于 1 的次方
    `x ** 0.5` 是 `x` 的平方根，其他小于 1 的次方效果类似：对大数压得多，对小数压得少。比如 `100 ** 0.75` 大约是 32，`10000 ** 0.75` 是 1,000——原本相差 100 倍的两个次数，取 0.75 次方后只相差大约 32 倍。
    ///

    噪声词按概率表 `noise` 抽取：把每个词的出现次数取 0.75 次方，再归一化使总和为 1。这样高频词不会过于垄断，低频词也有机会被抽作噪声词。`torch.multinomial(noise, m, replacement=True)` 按这些概率抽出 `m` 个编号。

    `train_vectors` 其实就是第 10 章的 `train`，只是把交叉熵损失换成了 `pair_loss`。它的流程很直白：建两张词向量表（用 `nn.init.normal_` 填入很小的随机初值），交给 Adam 优化器一起更新；每一批先查出 `u`、`v` 和 `n`，然后执行标准的训练三步：**清空旧梯度（`zero_grad`）→ 反向传播算新梯度（`backward`）→ 更新参数（`step`）**。训练结束后，返回 NumPy 数组形式的中心词表（每个词一行），上下文词表用完就丢掉。
    """)
    return


@app.function
def train_vectors(centres, contexts, noise, size, epochs, batch_size, rate):
    centre_table = nn.Embedding(len(noise), size)
    context_table = nn.Embedding(len(noise), size)
    nn.init.normal_(centre_table.weight, 0, 0.1)
    nn.init.normal_(context_table.weight, 0, 0.1)
    optimizer = torch.optim.Adam([centre_table.weight, context_table.weight], lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(centres))
        for start in range(0, len(centres), batch_size):
            batch = order[start:start + batch_size]
            noise_ids = torch.multinomial(noise, 5 * len(batch), replacement=True)
            n = context_table(noise_ids.reshape(len(batch), 5))
            optimizer.zero_grad()
            loss = pair_loss(centre_table(centres[batch]), context_table(contexts[batch]), n)
            loss.backward()
            optimizer.step()
    return centre_table.weight.detach().numpy()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：读一读训练循环

    在 `train_vectors` 里找到生成 `noise_ids` 和 `n` 的那两行。调用它时我们设 `size = 32`，每批 1,024 对词。对一个完整的批：一共抽出了多少个噪声编号？`reshape` 之后，每对词分到几个？`n` 里的每个噪声向量有多少个数字？
    """)
    return


@app.cell(hide_code=True)
def _():
    loop_boxes = hand.boxes(['抽出的噪声编号', '每对的噪声词', '每个噪声向量的数字'])
    return (loop_boxes,)


@app.cell(hide_code=True)
def _(loop_boxes):
    hand.marked(loop_boxes, {'抽出的噪声编号': 5120, '每对的噪声词': 5, '每个噪声向量的数字': 32}, {
        '抽出的噪声编号': r"""`torch.multinomial` 的第二个输入是要抽多少个：批大小的 5 倍""",
        '每对的噪声词': r"""编号被 `reshape` 成 `(len(batch), 5)`：每对一行""",
        '每个噪声向量的数字': r"""`context_table` 的每一行有 `size` 个数字""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所以 `n` 的形状是 `(1024, 5, 32)`——正是你的 `pair_loss` 期待的 `(b, k, d)`，其中 $k = 5$。

    按下按钮会先生成词对（`window = 2` 时大约 370 万对），再训练 2 轮，在普通笔记本电脑上大约半分钟。训练完成之前，依赖它的单元格会显示 `Ancestor stopped`，这是正常的。跑完后你会看到 `(2000, 32)`：2,000 个词，每个词 32 个数字。
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="训练词向量")
    train_button
    return (train_button,)


@app.cell
def _(counts, ids, train_button, vocab):
    mo.stop(not train_button.value, mo.md("*按下按钮开始训练。*"))
    centres, contexts = skipgram_pairs(ids, 2)
    freq = torch.tensor([counts[word] for word in vocab], dtype=torch.float32)
    noise = freq ** 0.75 / torch.sum(freq ** 0.75)
    torch.manual_seed(0)
    E = train_vectors(torch.tensor(centres), torch.tensor(contexts), noise, 32, 2, 1024, 0.003)
    E.shape
    return (E,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 它学到了什么？

    没有人告诉过网络任何词的意思，它也从未见过词典，只看过一对对邻居。现在来看看，哪些词的向量和某个词最相似。

    /// admonition | 🧰 工具箱：排好序的位置
    `np.argsort(a)` 返回数组里各个数从小到大排好序后的**下标位置**：`np.argsort(np.array([0.5, 0.1, 0.9]))` 得到 `[1, 0, 2]`。如果想让最大的排最前，只要对取负后的数组排序即可。
    ///

    `nearest` 先把每一行都缩放到长度 1。两个向量长度都是 1 时，余弦相似度公式里的除法就省掉了，光用 `@` 就能一次算出和所有词的相似度。接着它对 `-sims` 排序，让最相似的排在最前面；排第一的总是词本身，所以跳过它。在框里换个词，按回车试试：
    """)
    return


@app.function
def nearest(query, E, vocab, count):
    unit = E / np.sqrt(np.sum(E * E, axis=1, keepdims=True))
    sims = unit @ (query / np.sqrt(query @ query))
    return [vocab[i] for i in np.argsort(-sims)[:count]]


@app.cell(hide_code=True)
def _():
    word_box = mo.ui.text(value="dog", label="词：")
    word_box
    return (word_box,)


@app.cell
def _(E, index, vocab, word_box):
    asked = word_box.value.strip().lower()
    nearest(E[index[asked]], E, vocab, 6)[1:] if asked in index else "这个词不在词表里。"
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    再看几个常见词的近邻表：
    """)
    return


@app.cell(hide_code=True)
def _(E, index, vocab):
    table = ["| 词 | 最近的词 |", "|---|---|"]
    for word in ["dog", "happy", "mom", "red", "ran"]:
        table.append(f"| {word} | " + ", ".join(nearest(E[index[word]], E, vocab, 6)[1:]) + " |")
    mo.md("\n".join(table))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dog`（狗）周围是动物，`happy`（开心）周围是情绪和状态，`mom`（妈妈）周围是家人，`red`（红色）周围是颜色，`ran`（跑）周围是各种移动动作。（每次重新运行，列表可能会略有差异。）这一切语义关系，全都是只靠“谁站在谁旁边”学出来的。

    不过别指望它十全十美。每个词只有 32 个数字，训练文本也只有大约一百万词，所以偶尔会出现奇怪的结果，重跑时也会有波动。而且模型只能学到训练文本里有的内容：用儿童故事训练出来的向量懂狗和妈妈，但不懂化学；文本里自带的习惯甚至偏见（比如谁常做什么事），也会原封不动地进入词向量。

    ## 接下来

    词向量抓住了单个词的意思，但到现在为止，我们还没有用到词的*顺序*。在**第 13 章**，你会造出第一个真正的语言模型：它能预测下一个词，还能续写文本。为了衡量它的好坏，你会认识**困惑度**（perplexity），并开启课程的**排行榜**（leaderboard）——在同一批留出的测试故事上打分，从第 13 章到最后一章，每个新模型都要尝试刷新这个纪录。

    ## 你现在会做的事

    - 解释为什么 one-hot 编码不含词义，而词向量能表达语义相近
    - 用字典统计词频，并把文本转成编号列表
    - 用余弦相似度衡量两个向量有多像
    - 从文本构造 skip-gram 训练词对
    - 从 softmax 推出 sigmoid，并亲手算出负采样损失
    - 把 `nn.Embedding` 用作一张可学习的词向量表
    - 查看词向量的近邻词，并理解它们的局限

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 本章的方法叫 *word2vec*（Mikolov 等，2013）。工业级实现通常还会按概率丢弃一部分超高频词（比如 `the`）；另一种经典方法 *GloVe* 则直接统计词对共现次数来得到类似的词向量。“词义体现在邻居里”这个思想在语言学中叫*分布假说*（distributional hypothesis）。本章故事数据来自 *TinyStories*（Eldan 和 Li，2023）。

    词向量之间的差值也能表达语义关系。如果从 `boy`（男孩）到 `girl`（女孩）的差值大致代表“变为女性”的方向，那么从 `he`（他）沿同一方向走一步，就应该落在 `she`（她）附近。你可以在新单元格里试试：`nearest(E[index["girl"]] - E[index["boy"]] + E[index["he"]], E, vocab, 6)`。输入的三个词本身往往也会排在最前面，记得跳过它们往后看。由于这里只有 32 维和约一百万词，这类**类比**（analogy）有的成立，很多还不成立。
    """)
    return


if __name__ == "__main__":
    app.run()
