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

    课程的第二部分讲语言。从这里到最后一章，我们都用同一份文本：一批简短的小故事，叫 **TinyStories**。这样每个新模型都能和之前的模型比较。它已经打包在课程里，而且所有语言版本用的都是这份英文文本。下面是 12,000 篇故事中第一篇的开头：是英文原文，不用读懂，只要知道这是一段文本就行。（另外 1,000 篇 `test_stories` 留到第 13 章用。）
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
    网络处理的是数字，所以第一个问题是怎样把词变成数字。最直接的办法是第 07 章的 one-hot 行：如果认识 2,000 个词，每个词就是一行 2,000 个数字，其中只有一个 1。但这对网络毫无帮助。任意两个不同的 one-hot 行，点积都是 0，所以 `dog` 和 `cat` 的差别，跟 `dog` 和 `yesterday` 的差别一样大。

    我们想要更好的东西：一个很短的**词向量**（word vector），比如 32 个数字，并且让意思相近的词得到相近的向量。这些数字没人写得出来。但有个办法只靠文本就能把它们学出来，它基于一个观察：**看一个词的邻居，就能了解这个词。** 遮住下面句子里的一个词，你依然能猜出它是哪一类词：

    > The ▢ barked at the mailman and wagged its tail.（这只 ▢ 冲着邮递员叫，还摇着尾巴。）

    出现在相同邻居中间的词，意思往往相近。所以我们给网络一个任务：预测一个词的邻居，然后留下它在这个过程中发明的向量。这就是 **word2vec** 方法。

    ## 从文本到单词

    先把文本切成单词。下面这个辅助函数把所有字母变成小写，保留连续的字母和撇号，其余的都丢掉。（`re` 是 Python 的文本模式工具，不用专门学；`[a-z']+` 的意思就是“一个或多个字母或撇号”。）

    /// admonition | 🧰 工具箱：两个处理文本的工具
    - `text.lower()` 给出把大写字母全部变成小写的文本：`"The Dog".lower()` 是 `"the dog"`。
    - `" ".join(items)` 把字符串列表粘成一个字符串，中间加空格：`" ".join(["a", "b", "c"])` 是 `"a b c"`。
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
    6,000 篇故事大约有 965,000 个词，对这一章足够了。用 12,000 篇只不过多花一倍时间。

    ## 用字典计数

    我们需要知道每个词出现了多少次。列表做不到：列表按位置查找，而我们想按词查找。Python 为此准备了**字典**（dictionary）：它存的是一对对的**键**（key）和**值**（value），能一下子找到某个键。

    /// admonition | 🧰 工具箱：字典
    - `{}` 造出一个空字典。`pets["dog"] = 3` 把 3 存在键 `"dog"` 下面，原来存在那里的东西会被换掉。
    - `pets["dog"]` 把它读出来。读一个从没存过的键，会报错停下。
    - `pets.get("fish", 0)` 也是读取，但键不存在时给出第二个值，这里是 0。
    - `for name in pets:` 按键第一次存入的先后顺序，逐个访问所有的键。
    ///

    下面的单元格先存入 3 只狗，再两次各加 2 只猫。第一次加的时候还没有 `"cat"` 这个键，所以 `.get` 给出 0，猫的数量就从 0 开始加。
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

    写 `count_words(words)`。它接收一个单词列表，返回一个字典，从每个不同的词映射到它出现的次数：

    > `count_words(["the", "dog", "the"])` 得到 `{"the": 2, "dog": 1}`

    从一个空字典开始，把单词一个一个看过去。每看到一个词，就把它自己的次数加 1。想一想：一个词第一次出现时，它的次数是多少？
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
        r"""分三部分：造一个空字典，对 `words` 做循环，循环结束后返回字典。工具箱单元格里的猫，演示了怎样给一个可能还不存在的次数加数。""",
        r"""一共 4 行。第一行是 `counts = {}`。循环里面只有一行：把 `.get` 查到的数加 1，存到键 `word` 下面。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 词表

    现在我们留下最常见的 2,000 个词作为**词表**（vocabulary），并给每个词一个数字，也就是它的**编号**（id）：它在列表里的位置。这还需要两个工具。

    /// admonition | 🧰 工具箱：排序和筛选
    - `sorted(pets)` 把所有键按字母顺序排成一个列表。`sorted(pets, key=pets.get)` 则按 `pets.get` 对每个键给出的结果来排，也就是按存的值从小到大排。加上 `reverse=True`，最大的排在最前面。
    - `[name for name in names if name in pets]` 和第 05 章一样造出一个列表，但只留下通过 `if` 后面那个测试的项。字典里有这个键时，`name in pets` 为真。
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
    下一个单元格用到你写的 `count_words`，所以练习 1 通过之前它会显示错误。这是正常的，后面凡是用到它的单元格也一样。它造出三样东西：

    - `vocab`：最常见的 2,000 个词，最常见的排最前；
    - `index`：反过来的字典，从词到编号；
    - `ids`：用编号表示的全部文本，词表以外的词被跳过。

    它显示不同的词的个数、`vocab` 的前 8 个词，以及 `ids` 的长度。
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
    一共出现了大约 6,800 个不同的词，这 2,000 个就覆盖了文本的 97%。小故事的词汇量小，所以很适合在笔记本上跑。

    ## 两个向量有多像？

    第 05 章的点积是逐个位置相乘再相加。两个向量方向一致时，点积就大。但它也会随长度增长，而衡量相似只关心方向。所以我们除以两个向量的长度。

    /// admonition | 🧰 工具箱：向量的长度
    用 $\cdot$ 表示点积（代码里是 `@`），向量 $a$ 的**长度**是 $\sqrt{a \cdot a}$，也就是各项平方和的平方根。对 $[3, 4]$ 来说是 $\sqrt{9 + 16} = 5$：在图上，这就是从 $(0, 0)$ 到点 $(3, 4)$ 的那支箭头的长度。
    ///

    这个度量叫**余弦相似度**（cosine similarity）：

    > $$\operatorname{similarity}(a, b) = \frac{a \cdot b}{a \text{ 的长度} \times b \text{ 的长度}}$$

    方向相同时是 1，互相垂直（毫无共同点）时是 0，方向相反时是 −1。取 $a = [3, 4]$，$b = [6, 8]$，点积是 $3 \times 6 + 4 \times 8 = 50$。长度是 $\sqrt{9 + 16} = 5$ 和 $\sqrt{36 + 64} = 10$。所以相似度是 $50 / (5 \times 10) = 1$：$b$ 是 $a$ 拉长一倍，方向相同。

    ## 动手算：两个相似度

    先看 $a = [3, 4]$ 和 $b = [4, 3]$：同样两个数，换了位置。算出点积、两个长度和相似度。然后算 $c = [0, 2]$ 和 $d = [3, 4]$ 的相似度。
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
    换了位置的两个数，指的方向很接近，但不相同：是 0.96，不是 1。

    ## 练习 2：similarity

    写 `similarity(a, b)`，输入是两个大小相同的 NumPy 向量。它按上面的公式返回一个数。`np.sqrt` 是平方根（第 09 章）。

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
        r"""你需要三个数，每个都用一次 `@` 算出：$a \cdot b$、$a \cdot a$ 和 $b \cdot b$。后两个再经过 `np.sqrt`，就成了长度。""",
        r"""可以写成 3 行：两个长度各一行，然后是 `return`。第一行是 `length_a = np.sqrt(a @ a)`。两个长度相乘的部分要加括号，否则 Python 会先除以第一个，再乘以第二个。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 任务：猜邻居

    训练样本是一对对编号。沿着文本滑动；每个词轮流当**中心词**（centre word），两边 `window` 个位置以内的每个词都是它的**上下文词**（context word）。对文本 `the dog ran fast`，`window = 1`：

    | 中心词 | 上下文词 | 词对 |
    |---|---|---|
    | `the` | `dog` | (the, dog) |
    | `dog` | `the`、`ran` | (dog, the)、(dog, ran) |
    | `ran` | `dog`、`fast` | (ran, dog)、(ran, fast) |
    | `fast` | `ran` | (fast, ran) |

    两头的词上下文词少一些，因为窗口伸到了文本外面。从中心词预测上下文词，这个任务叫 **skip-gram**。

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
    /// admonition | 🧰 工具箱：别走出列表
    - `max(a, b)` 取两个数里较大的，`min(a, b)` 取较小的。它们能拦住一个位置，不让它走出一个有 10 项的列表：`max(0, 1 - 3)` 是 0，不是 −2；`min(10, 8 + 3)` 是 10，不是 11。
    - `range(start, stop)` 从 `start` 数到 `stop` 的前一个：`range(2, 5)` 给出 2、3、4。想把位置 5 也包括进来，就写 `range(2, 5 + 1)`。
    - `a != b` 表示“a 不等于 b”，是 `==` 的反面。
    ///

    ## 练习 3：skipgram_pairs

    写 `skipgram_pairs(ids, window)`。`ids` 是一个编号列表，`window` 是一个整数。它返回两个长度相同的列表：中心词和上下文词。两个列表里位置 $p$ 上的两项合起来就是一个词对。中心词按文本顺序排列，每个中心词的上下文词从左到右排列。把上面表格里的四个词记作编号 0 到 3：

    > `skipgram_pairs([0, 1, 2, 3], 1)` 得到 `([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])`

    对位置 $i$ 上的中心词，上下文词在从 $i - \text{window}$ 到 $i + \text{window}$ 之间、落在列表之内的那些位置上，$i$ 自己除外。
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
    ## 真的还是编的？换个便宜的问题

    每个词有两个向量，当中心词时用一个（$u$），当上下文词时用另一个（$v$）。词义最后会落在 $u$ 里。我们希望当 $v$ 确实是邻居时，$u \cdot v$ 很高。

    最直接的模型是第 07 章那一套：给全部 2,000 个词打分，再做 softmax。但这意味着每一对都要算 2,000 个点积，而词对有几百万个。我们改问一个便宜得多的问题：*这一对是真的吗？* 对每个真实的词对，再随机抽几个**噪声词**（noise word），训练网络把真的上下文词和假的区分开。这叫**负采样**（negative sampling）。

    ## 从一个分数得到概率

    回答是或否，需要从一个分数 $s$ 得到一个概率。对两个分数做 softmax：“真”的分数是 $s$，“假”的分数是 0。（为什么是 0？第 07 章的平移说明，softmax 只关心分数之间的差，所以其中一个不妨取 0。）任何数的 0 次方都是 1，所以 $e^0 = 1$，“真”的份额是

    > $$\sigma(s) = \frac{e^s}{e^s + e^0} = \frac{e^s}{e^s + 1}$$

    这叫 **sigmoid**，用希腊字母 $\sigma$（读作 sigma）表示。下面是分数从 −6 到 6 时它的样子：
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
    $s$ 很大时它接近 1，$s = 0$ 时正好是 0.5，$s$ 很负时接近 0。这条曲线还有一种对称：“假”的份额是 $1 - \sigma(s)$，它等于 $\sigma(-s)$。在图上，$\sigma(2)$ 比 1 低多少，$\sigma(-2)$ 就比 0 高多少。（验证一下：$\sigma(-s) = e^{-s} / (e^{-s} + 1)$。分子分母同乘 $e^s$，注意 $e^{-s} \times e^s = e^0 = 1$，得到 $1 / (1 + e^s)$，正好等于 $1 - e^s / (e^s + 1)$。）

    ## 一对的损失

    损失就是第 07 章的意外程度，即正确答案所得份额的 $-\log$，把一个真实词和 $k$ 个噪声词加起来。对真实词，正确答案是“真”，份额是 $\sigma(u \cdot v)$。对噪声词 $n$，正确答案是“假”，份额是 $\sigma(-u \cdot n)$：

    > $$\text{一对的损失} = -\log \sigma(u \cdot v) - \log \sigma(-u \cdot n_1) - \ldots - \log \sigma(-u \cdot n_k)$$

    第一项把 $u$ 拉向真实的邻居，其余各项把它从假词那里推开。下面是几个分数对应的意外程度 $-\log \sigma(s)$：

    | $s$ | −2 | −1 | 0 | 1 | 2 |
    |---|---|---|---|---|---|
    | $\sigma(s)$ | 0.12 | 0.27 | 0.50 | 0.73 | 0.88 |
    | $-\log \sigma(s)$ | 2.13 | 1.31 | 0.69 | 0.31 | 0.13 |

    ## 动手算：一对的损失

    取 $u = [1, 2]$，真实的上下文词 $v = [2, 0]$，一个噪声词 $n = [1, -1]$。先算出两个分数 $u \cdot v$ 和 $u \cdot n$，再从表里查意外程度，算出损失。留意噪声分数前面的负号。

    然后假设噪声词是 $[2, 0]$，和真实的上下文词一样。这时损失是多少？
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
        '损失': r"""真实词的意外程度是表里 $s = 2$ 那一格。噪声词的是它的分数取负之后那一格，也就是 $s = 1$。把两个加起来。""",
        '换了噪声词之后的损失': r"""现在噪声分数是 2，所以它的意外程度是表里 $s = -2$ 那一格""",
    }, LOCALE, tolerance=0.011)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第二个损失是第一个的五倍。噪声词的分数和真邻居一样高，正是损失要罚的情况；训练会把 $u$ 从这个词那里推开，让损失降下来。

    ## 一次算一整批

    练习里要一次算出很多对的损失，所以向量是叠在一起送来的：`u` 和 `v` 的形状是 `(b, d)`，表示 `b` 对、每个向量 `d` 个数字；`n` 的形状是 `(b, k, d)`，每对 `k` 个噪声向量。（这里的 `n` 指噪声，不是前面章节里的样本数。）

    /// admonition | 🧰 工具箱：成批的点积
    - 对两个形状为 `(b, d)` 的张量，`torch.sum(x * y, dim=1)` 先逐个位置相乘，再沿每一行相加：每行一个点积，形状是 `(b,)`。
    - `x.unsqueeze(1)`（第 11 章）插入一个大小为 1 的维度：`(b, d)` 变成 `(b, 1, d)`。做算术时，大小为 1 的维度会重复自己去配合另一边，所以 `(b, 1, d)` 乘 `(b, k, d)` 是可以的，结果是 `(b, k, d)`：`x` 的第 0 行，分别和属于第 0 行的那 `k` 个向量相遇。
    - `F.logsigmoid(s)` 对张量里的每个数计算 $\log \sigma(s)$，分数很大时也安全，道理和第 07 章 softmax 的平移相似。
    ///

    下面的单元格用 2 行、每行 3 个数字，和一个形状为 `(2, 4, 3)` 的张量，试了前两个工具，并显示三个形状。
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

    写 `pair_loss(u, v, n)`，形状如上。它返回一个数，也就是一个没有维度的张量：`b` 对各自的损失的平均值。中间要经过：

    | 算什么 | 形状 |
    |---|---|
    | 真实分数 $u \cdot v$，每对一个 | `(b,)` |
    | 噪声分数 $u \cdot n_j$，每对 `k` 个 | `(b, k)` |
    | 按公式算出的每对的损失 | `(b,)` |
    | 它们的平均值 | 一个数 |

    你在动手算里得到的数就是其中一项检查：只有 1 对的一批，`u = [[1, 2]]`，`v = [[2, 0]]`，`n = [[[1, -1]]]`，结果大约是 `0.44`。
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

    最后一个 PyTorch 工具：`nn.Embedding(2000, 32)` 是一张 2,000 行、每行 32 个旋钮的表。给它任意形状的编号张量，它就为每个编号查出一行，所以 `(b,)` 的编号给出 `(b, 32)` 的向量，`(b, k)` 的编号给出 `(b, k, 32)`。就这么简单：一次查表，而表里的数字会学习。

    /// admonition | 🧰 工具箱：小于 1 的次方
    `x ** 0.5` 是 `x` 的平方根，其他小于 1 的次方也差不多：对大数压得多，对小数压得少。`100 ** 0.75` 大约是 32，`10000 ** 0.75` 是 1,000。所以相差 100 倍的两个次数，最后只相差大约 32 倍。
    ///

    噪声词按概率表 `noise` 抽取：每个词的次数取 0.75 次方，再缩放到总和为 1。这样常见词没那么占优势了，少见的词也有机会当噪声词。`torch.multinomial(noise, m, replacement=True)` 按这些概率抽出 `m` 个编号。

    `train_vectors` 就是第 10 章的 `train`，把交叉熵换成了 `pair_loss`。它做的事是：建两张表，交给 Adam 训练；每一批先查出 `u`、`v` 和 `n`，然后清空、反向、走一步。它返回 NumPy 数组形式的中心词表，每个词一行；上下文词表则丢掉。`nn.init.normal_` 给表一个小的随机起点。
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

    在 `train_vectors` 里找到造出 `noise_ids` 和 `n` 的那两行。我们调用它时用 `size = 32`，每批 1,024 对。对一个完整的批：抽出了多少个噪声编号？`reshape` 之后，每对分到几个？`n` 里的每个噪声向量有多少个数字？
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
    所以 `n` 的形状是 `(1024, 5, 32)`：正是你的 `pair_loss` 要的 `(b, k, d)`，其中 $k = 5$。

    按钮先造出词对（`window = 2` 时大约 370 万对），再训练 2 轮：在笔记本上大约半分钟。完成之前，需要它的结果的单元格会显示 `Ancestor stopped`，这是正常的。完成后你会看到 `(2000, 32)`：2,000 个词，每个 32 个数字。
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

    没有人告诉网络任何词的意思，它也没见过词典。它只看到一对对邻居。现在来找找，哪些词的向量和某个词最相似。

    /// admonition | 🧰 工具箱：排好序的位置
    `np.argsort(a)` 给出数组里各个数的位置，从最小的数排到最大的数：`np.argsort(np.array([0.5, 0.1, 0.9]))` 是 `[1, 0, 2]`。想让最大的排最前，就对取负后的数组排序。
    ///

    `nearest` 把每一行都缩放到长度 1。两个长度都是 1，相似度公式里的除法就没有了，所以光用 `@` 就能一次算出和所有词的相似度。然后它对 `-sims` 排序，让最大的排在最前面。排第一的总是词本身，所以跳过它。在框里换一个词，按回车：
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
    再看几个词的表格：
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
    `dog`（狗）周围是动物，`happy`（开心）周围是感受和状态，`mom`（妈妈）周围是家人，`red`（红色）周围是颜色，`ran`（跑）周围是各种移动方式。（你的列表每次运行可能略有不同。）这一切都只来自谁站在谁旁边。

    别指望它完美。它只有每个词 32 个数字，训练文本也只有约一百万个词，所以有些结果很奇怪，重新运行时也会有些不同。它也只学得到文本里有的东西。用儿童故事训练出的向量懂狗和妈妈，不懂化学。文本的各种习惯，包括关于谁该做什么的偏见，也会进到向量里。

    ## 接下来

    词向量抓住了词的意思，但到现在为止，还没有任何东西用到词的*顺序*。在**第 13 章**，你会造出第一个真正的语言模型：它预测下一个词，并能生成文本。为了衡量它，你会认识**困惑度**（perplexity），并开始建立**排行榜**（leaderboard）：在同一批留出的故事上算出的一个分数，从那里到课程结束，每个模型都要尝试打破它。

    ## 你现在会做的事

    - 解释为什么 one-hot 行不含词义，词向量又增加了什么
    - 用字典计数，并把文本变成编号
    - 用余弦相似度衡量两个向量有多像
    - 从文本造出 skip-gram 词对
    - 从 softmax 推出 sigmoid，并亲手算出一个负采样损失
    - 把 `nn.Embedding` 用作一张学出来的向量表
    - 探索最近的词，并说出它们的局限

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 这个方法是 *word2vec*（Mikolov 等，2013）。实际运行时还会丢掉一部分最常见的词，比如 `the`；*GloVe* 则直接统计词对，得到类似的向量。“词义体现在邻居里”这个想法叫*分布假说*（distributional hypothesis）。故事来自 *TinyStories*（Eldan 和 Li，2023）。

    向量之间的差也能表示关系。如果从 `boy`（男孩）到 `girl`（女孩）的这一步大致是“变成女性”的方向，那么从 `he`（他）走同样一步，应该落在 `she`（她）附近。在一个新单元格里试试：`nearest(E[index["girl"]] - E[index["boy"]] + E[index["he"]], E, vocab, 6)`。你放进去的三个词往往也会出现在答案里，所以要跳过它们往后看。只有 32 个数字和一百万个词，这样的**类比**（analogy）有些成立，很多不成立。
    """)
    return


if __name__ == "__main__":
    app.run()
