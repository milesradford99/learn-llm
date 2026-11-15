import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="24 · 缩放定律")

with app.setup(hide_code=True):
    import marimo as mo
    import numpy as np
    import torch

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.plot import chart
    from learn_llm.scaling import knobs_of, small_run, story_tokens

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 24 · 缩放定律

    假设你拿到一笔预算，用来做一次大规模训练：这么多块 GPU，跑这么多周。有两个问题要先回答。模型该多大？它该读多少文本？猜错了，预算就花在一个本可以更好的模型上。而你没法靠试来找答案：每一次试，*都是*那次大训练。

    出路是这个领域里最有用的习惯之一：做很多**小**实验，小到可以反复做，然后找出它们的结果遵循的规律，再用这条规律在花钱之前预测大训练。这样的规律叫**缩放定律**（scaling law）。这一章里，你会在自己的电脑上亲手测出一条，并用它来规划一次训练。

    ## 幂律

    几乎所有缩放定律都有同一种形状，叫**幂律**（power law）：

    > $$y = a \times x^k$$

    $x$ 是你投入的（文本、旋钮），$y$ 是你得到的（损失），$a$ 和 $k$ 是两个固定的数。当 $k$ 是负数时，$y$ 随 $x$ 变大而下降：记住 $x^{-1}$ 就是 $\frac{1}{x}$，$x^{-0.5}$ 就是 $\frac{1}{\sqrt{x}}$。

    幂律有一条决定性的性质。把 $x$ 乘以 2，$y$ 就乘以 $2^k$，**不管你从哪里开始**：$a \times (2x)^k = 2^k \times (a \times x^k)$。从 1 变到 2，按比例算，和从一百万变到两百万效果一样。每翻一倍，花费是上一次的两倍，换来的却一样多。

    ## 动手算：每次减半的定律

    取 $y = 64 \times x^{-0.5}$，也就是 $\frac{64}{\sqrt{x}}$。$x = 1$ 时它等于 64。那么 $x$ 为 4、16 和 64 时，它各等于多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    law_boxes = hand.boxes(['x = 4 时的 y', 'x = 16 时的 y', 'x = 64 时的 y'])
    return (law_boxes,)


@app.cell(hide_code=True)
def _(law_boxes):
    hand.marked(law_boxes, {'x = 4 时的 y': 32, 'x = 16 时的 y': 16, 'x = 64 时的 y': 8}, {
        'x = 4 时的 y': r"""$\sqrt{4} = 2$，再算 $64 / 2$""",
        'x = 16 时的 y': r"""$\sqrt{16} = 4$""",
        'x = 64 时的 y': r"""$\sqrt{64} = 8$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每次 $x$ 乘以 4，$y$ 就减半。下面是画在普通坐标上的这条定律：
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((0, 64), (0, 70), "x", "y", curve=[(x, 64 / np.sqrt(x)) for x in np.linspace(0.85, 64, 200)], dots=[(1, 64), (4, 32), (16, 16), (64, 8)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这张图很难读。前三个点都挤在左边缘，你也分不出这条曲线和许多别的先降后平的曲线有什么不同。

    ## 取对数，就变成直线

    这几个点在“乘法”上是等距的：每个点比前一个向右 4 倍、向下 2 倍。那就来数乘法的次数。这正是对数做的事：第 13 章用过 $\log(a \times b) = \log a + \log b$。把它用在 $x^k$ 上，也就是 $x$ 自乘 $k$ 次，就得到 $\log(x^k) = k \times \log x$。（它对任何 $k$ 都成立，不只是整数。）现在对幂律两边取对数：

    > $$\log y = \log a + k \times \log x$$

    把它读成一条直线。横轴放 $\log x$，纵轴放 $\log y$，**幂律就是一条直线，斜率是 $k$，截距是 $\log a$。**这样画出来的图叫**双对数图**（log-log plot）。如果测量点在这样的图上落在一条直线上，它们就遵循幂律，而这条线的斜率就是唯一值得记住的数。

    我们之前用的对数都是自然对数，它数的是乘以 $e$ 的次数。手算时，数乘以 2 的次数，也就是翻倍的次数，更容易。想法是一样的。我们的四个点是：

    | $x$ | $x$ 翻倍的次数 | $y$ | $y$ 翻倍的次数 |
    |---:|---:|---:|---:|
    | 1 | 0 | 64 | 6 |
    | 4 | 2 | 32 | 5 |
    | 16 | 4 | 16 | 4 |
    | 64 | 6 | 8 | 3 |

    比如 $64 = 2 \times 2 \times 2 \times 2 \times 2 \times 2$：从 1 翻了六倍。把两列“翻倍的次数”互相对着画出来：
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-0.5, 6.5), (2, 7), "x 翻倍的次数", "y 翻倍的次数", dots=[(0, 6), (2, 5), (4, 4), (6, 3)], lines=[(-0.5, 6)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一条直线，点之间距离相等。每向右 2 格就向下 1 格，所以斜率是 $-0.5$：这就是 $k$。它从 6 出发，而翻六倍得到 64：这就是 $a$。

    ## 一步求出最佳直线

    真实的测量不会正好落在一条直线上，所以我们需要最贴合它们的那条线。你做过这件事：第 01 章用均方误差衡量一条线的偏差，第 02 章顺着山坡走下去，找到最好的那条。对一条直线，有个捷径。山谷的谷底，就是两个斜率都为零的地方，这两个方程可以一次性解出来。我们跳过代数，直接用结果。用 $\bar{x}$ 表示 $x$ 值的均值，$\bar{y}$ 表示 $y$ 值的均值。于是：

    > $$\text{斜率} = \frac{(x - \bar{x}) \times (y - \bar{y}) \text{ 之和}}{(x - \bar{x})^2 \text{ 之和}} \qquad\qquad \text{截距} = \bar{y} - \text{斜率} \times \bar{x}$$

    分子问的是：当 $x$ 高于它的均值时，$y$ 是不是也高于它的均值（乘积为正），还是低于（乘积为负）？分母则按 $x$ 值的分散程度来调整答案。截距再让这条线穿过点 $(\bar{x}, \bar{y})$。这叫**最小二乘拟合**（least-squares fit）。

    ## 动手算：过三个点的直线

    用表里的前三行：点 $(0, 6)$、$(2, 5)$ 和 $(4, 4)$。依次求出两个均值、分数的分子和分母，再求斜率和截距。
    """)
    return


@app.cell(hide_code=True)
def _():
    line_boxes = hand.boxes(['x 的均值', 'y 的均值', '分子', '分母', '斜率', '截距'], step=0.1)
    return (line_boxes,)


@app.cell(hide_code=True)
def _(line_boxes):
    hand.marked(line_boxes, {'x 的均值': 2, 'y 的均值': 5, '分子': -4, '分母': 8, '斜率': -0.5, '截距': 6}, {
        'x 的均值': r"""$(0 + 2 + 4) / 3$""",
        'y 的均值': r"""$(6 + 5 + 4) / 3$""",
        '分子': r"""到均值的距离，$x$ 是 $-2, 0, 2$，$y$ 是 $1, 0, -1$：两两相乘，再加起来""",
        '分母': r"""把 $-2$、$0$ 和 $2$ 各自平方，再加起来""",
        '斜率': r"""分子除以分母""",
        '截距': r"""$y$ 的均值，减去斜率乘以 $x$ 的均值；小心那两个负号""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    和你从图上读出来的一样，是 $-0.5$ 和 6。

    ## 练习 1：fit_line

    写 `fit_line(xs, ys)`。`xs` 和 `ys` 是长度相同的 NumPy 数组，每个点占一项。它返回最小二乘直线的（斜率，截距）这一对数，用的是上面那两个公式。

    对 `xs = [0, 1, 2]` 和 `ys = [1, 3, 5]`，它必须得到 `(2, 1)`：这几个点在直线 $y = 2x + 1$ 上。
    """)
    return


@app.function
def fit_line(xs, ys):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "fit_line", fit_line, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""数组可以不用循环就把活干完：`xs - np.mean(xs)` 一次得到所有到均值的距离，`*` 把两个数组按位置相乘，`np.sum` 把结果加起来。""",
        r"""一共 3 到 4 行。把距离 `xs - np.mean(xs)` 存进一个名字，因为分子和分母都要用它。先算斜率：截距要用到它。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：整个数组的对数
    `np.log` 对数组里的每个数取自然对数，`np.exp` 把它还原：`np.exp(np.log(np.array([2.0, 5.0])))` 又是 `[2, 5]`。
    ///

    ## 练习 2：fit_power_law

    写 `fit_power_law(xs, ys)`。`xs` 和 `ys` 是由正数组成的 NumPy 数组。它返回幂律 $y = a \times x^k$ 的一对数 $(a, k)$，这条幂律在双对数图上和它们拟合得最好。做法就是“取对数，就变成直线”那一节，拟合的部分交给你的 `fit_line`。

    对我们的点 `xs = [1, 4, 16]` 和 `ys = [64, 32, 16]`，它必须得到 `(64, -0.5)`。
    """)
    return


@app.function
def fit_power_law(xs, ys):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "fit_power_law", fit_power_law, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""对两个数组的对数拟合一条直线。直线的两个数里，哪一个是 $k$？另一个是 $\log a$，不是 $a$。""",
        r"""一共 2 行。`fit_line` 返回一对数，所以在 `=` 左边给两半各起一个名字。这里用自然对数就行：只要 `np.exp` 还原的是同一种对数，不管用哪种对数，斜率都一样。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 你自己的小规模训练

    该测量了。函数 `small_run(d, tokens, …)` 完整地做一次训练，返回一个数：

    - 它建一个第 20 章的 transformer，每个词元用 `d` 个数表示，有 2 个块；
    - 它用第 21 章的办法（AdamW、预热、余弦衰减）在故事的 `tokens` 个词元上训练它，每个词元只读一遍；
    - 它返回在固定的一段测试故事上，每个词元的平均意外程度。这就是损失，而 $e^{\text{损失}}$ 就是困惑度。

    它的代码在 `learn_llm/scaling.py` 里，其中没有任何新东西。我们试四种宽度，从 16 到 128，和四种文本量，从 50,000 到 400,000 个词元：一共 16 次训练。下面的单元格载入词元，并数出每种宽度的旋钮数。
    """)
    return


@app.cell
def _():
    train_tokens, test_tokens, vocab_size = story_tokens()
    widths = [16, 32, 64, 128]
    token_counts = np.array([50000, 100000, 200000, 400000])
    sizes = np.array([knobs_of(d, vocab_size) for d in widths])
    sizes
    return sizes, test_tokens, token_counts, train_tokens, vocab_size, widths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    从大约 78,000 个旋钮到将近一百万个。最大的一次训练读的文本是最小那次的八倍，模型却大了十二倍，所以花费大约是一百倍。按钮会做完全部 16 次训练。要 1 到 3 分钟，在它完成之前，需要结果的单元格会显示 `Ancestor stopped`。这是正常的。
    """)
    return


@app.cell(hide_code=True)
def _():
    sweep_button = mo.ui.run_button(label="做 16 次小规模训练（1 到 3 分钟）")
    sweep_button
    return (sweep_button,)


@app.cell
def _(sweep_button, test_tokens, token_counts, train_tokens, vocab_size, widths):
    mo.stop(not sweep_button.value, mo.md("*按下按钮开始训练。*"))
    loss_table = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            loss_table[i, j] = small_run(widths[i], token_counts[j], train_tokens, test_tokens, vocab_size)
    loss_table.round(3)
    return (loss_table,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每一行是一种模型大小，最小的在最前。每一列是一种文本量，最少的在最前。顺着一行读：文本越多，损失越低。顺着一列读：旋钮越多，损失越低。目前没什么意外。问题是它降得有*多快*。

    取第三行，也就是每个词元用 64 个数的那个模型，用你的函数对它的四个损失拟合一条幂律：
    """)
    return


@app.cell
def _(loss_table, token_counts):
    a_text, k_text = fit_power_law(token_counts, loss_table[2])
    np.round([a_text, k_text], 3)
    return a_text, k_text


@app.cell(hide_code=True)
def _(a_text, k_text, loss_table, token_counts):
    chart((10.5, 13.2), (1.2, 1.7), "词元数的对数", "损失的对数", dots=list(zip(np.log(token_counts), np.log(loss_table[2]))), lines=[(k_text, np.log(a_text))])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    四个点离这条线很近：在这个范围内，损失随文本量遵循幂律。你应该会看到 $k$ 接近 $-0.14$。按翻倍的规则，文本每翻一倍，损失就乘以 $2^{-0.14} \approx 0.91$。每次都减掉 9%，而每一次翻倍的花费是上一次的两倍。

    顺着一列也一样。下面是最后一列，400,000 个词元，对着旋钮数：
    """)
    return


@app.cell
def _(loss_table, sizes):
    np.round(fit_power_law(sizes, loss_table[:, 3]), 3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $k$ 接近 $-0.1$：模型翻一倍，损失大约减少 7%。

    不过，单纯的幂律不可能是全部真相。它会一直降向 0，而没有哪个模型能在故事上把损失降到 0：故事接下来写什么，一部分是作者的自由选择。损失之下有一个**下限**（floor），也就是即使是完美的模型也还剩下的意外程度。我们稍后会把它放进定律里。

    ## 一笔预算能买到什么

    第 22 章给出了训练的花费：大约 $6 \times N \times D$ 个 FLOP，$N$ 是旋钮数，$D$ 是读过的词元数。**计算预算**（compute budget）是你可以花的 FLOP 总数，所以它只固定了乘积 $N \times D$，别的都没定。怎么分，由你来选。模型大一倍，就只能读一半的文本。

    ## 动手算：花掉一笔预算

    你的预算是 $6 \times 10^{12}$ 个 FLOP。一个有 1,000,000 个旋钮的模型能读多少词元？有 4,000,000 个旋钮的呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    budget_boxes = hand.boxes(['1,000,000 个旋钮能读的词元数', '4,000,000 个旋钮能读的词元数'])
    return (budget_boxes,)


@app.cell(hide_code=True)
def _(budget_boxes):
    hand.marked(budget_boxes, {'1,000,000 个旋钮能读的词元数': 1000000, '4,000,000 个旋钮能读的词元数': 250000}, {
        '1,000,000 个旋钮能读的词元数': r"""$6 \times 10^{12} = 6 \times 10^6 \times D$，所以 $D$ 是 $10^{12} / 10^6$""",
        '4,000,000 个旋钮能读的词元数': r"""旋钮多四倍，词元就少到四分之一""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：tokens_for

    写 `tokens_for(budget, knobs)`：一个有 `knobs` 个旋钮的模型，花 `budget` 个 FLOP 能读的词元数。不必是整数。

    `tokens_for(6e12, 1e6)` 必须得到一百万，和你手算的一样。
    """)
    return


@app.function
def tokens_for(budget, knobs):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "tokens_for", tokens_for, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""从 $\text{预算} = 6 \times \text{旋钮数} \times \text{词元数}$ 出发，把词元数单独留在一边。""",
        r"""只要一行：一次除法。6 和旋钮数都要放在预算下面，所以它们需要用括号括起来。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一条同时管两者的定律

    要选择怎么分，我们需要一条同时了解旋钮和文本的定律。对语言模型效果最好的形式，是把三部分加起来：

    > $$\text{损失}(N, D) = E + \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}}$$

    - $E$ 是下限：一个完美的模型读了无穷多的文本之后的损失。
    - $\frac{a}{N^{\alpha}}$ 是因为模型**太小**而损失的部分。它是 $N$ 的幂律（除以 $N^{\alpha}$ 就是乘以 $N^{-\alpha}$），随模型变大而消退。
    - $\frac{b}{D^{\beta}}$ 是因为模型读得**太少**而损失的部分。它随文本变多而消退。

    $\alpha$ 和 $\beta$ 是希腊字母 alpha 和 beta。一共五个数，$E$、$a$、$\alpha$、$b$ 和 $\beta$，要靠训练来测出。

    现在取舍就显出来了。预算固定时，大模型让中间那部分变小，最后一部分变大，因为它买不起多少文本。小模型则相反。两者之间的某处，有一个最佳大小。

    ## 练习 4：joint_law

    写 `joint_law(knobs, tokens, floor, a, alpha, b, beta)`，也就是上面的公式。它必须既能用在普通的数上，也能用在由旋钮数和词元数组成的数组或张量上；只要你只用 `+`、`*` 或 `/`，以及 `**`，就做得到。

    旋钮数 100、词元数 10,000、下限 2、$a = 10$、$b = 300$、两个幂都是 0.5 时，它必须得到 $2 + \frac{10}{10} + \frac{300}{100} = 6$。
    """)
    return


@app.function
def joint_law(knobs, tokens, floor, a, alpha, b, beta):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "joint_law", joint_law, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三部分，加起来。Python 里的幂是 `**`，题目里的例子有 $100^{0.5} = 10$ 和 $10000^{0.5} = 100$，可以拿来分别检验每一部分。""",
        r"""只要一行。可以除以那个幂，也可以乘以带负号的幂：`x ** -k` 就是 $x^{-k}$。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：最佳的分法

    这是一条数字简单的玩具定律：$\text{损失} = 1 + \frac{100}{N} + \frac{400}{D}$。预算是 60,000 个 FLOP，所以 $N \times D = 10{,}000$。算出四种模型大小各自的损失。先求出每一种能负担的 $D$。
    """)
    return


@app.cell(hide_code=True)
def _():
    split_boxes = hand.boxes(['N = 10 时的损失', 'N = 50 时的损失', 'N = 100 时的损失', 'N = 200 时的损失'], step=0.1)
    return (split_boxes,)


@app.cell(hide_code=True)
def _(split_boxes):
    hand.marked(split_boxes, {'N = 10 时的损失': 11.4, 'N = 50 时的损失': 5, 'N = 100 时的损失': 6, 'N = 200 时的损失': 9.5}, {
        'N = 10 时的损失': r"""$D = 1000$，所以是 $1 + 10 + 0.4$""",
        'N = 50 时的损失': r"""$D = 200$，所以是 $1 + 2 + 2$""",
        'N = 100 时的损失': r"""$D = 100$，所以是 $1 + 1 + 4$""",
        'N = 200 时的损失': r"""$D = 50$，所以是 $1 + 0.5 + 8$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最小的模型旋钮不够，最大的模型文本不够。四个里最好的是 $N = 50$，这时两个部分相等。这样选出来的模型，对它的预算来说叫**计算最优**（compute-optimal）。

    /// admonition | 🧰 工具箱：最小的在哪里？
    `np.argmin` 是 `np.argmax` 的孪生兄弟：最小的那个数的位置。`np.argmin([7, 3, 9])` 是 1。
    ///

    ## 练习 5：best_size

    写 `best_size(budget, sizes, law)`。`sizes` 是一个列表，里面是可供选择的旋钮数，`law` 是一个函数：`law(knobs, tokens)` 返回预测的损失（第 02 章：函数也是值）。每种大小都得到预算能买得起的词元数，用你的 `tokens_for` 算。返回预测损失最低的那个大小。

    在你动手算的那道题上，预算 60,000，大小 `[10, 50, 100, 200]`，用那条玩具定律，它必须得到 50。
    """)
    return


@app.function
def best_size(budget, sizes, law):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("24", "best_size", best_size, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""造一个列表，每种大小对应一个预测损失，用 `np.argmin` 找出最小的那个的位置，再用这个位置去取 `sizes` 里的项。""",
        r"""一共 2 行。第一行是方括号里写 `for n in sizes` 造出来的列表，对 `n` 和 `n` 买得起的词元数调用 `law`。第二行返回 `sizes` 里的一项。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 用你的训练结果拟合这条定律

    联合定律在任何图上都不是直线，所以 `fit_line` 找不出它的五个数。但你知道一种什么都能拟合的办法：顺着山坡走下去。下面的函数就是第 10 章的训练循环。它的“模型”是你的 `joint_law`，它的旋钮是定律的五个数，它的损失是预测损失和测得损失的对数之间的均方偏差，这样，表里任何位置 1% 的偏差都算一样大。

    有两个细节。`a` 和 `b` 必须保持为正，所以拟合时把它们的对数当作旋钮，再把它们的 `torch.exp` 交给定律。另外，把各种大小排成一列，把各种词元数排成一行（第 20 章的 `[:, None]`），这样广播就能一次填满整个 $4 \times 4$ 的预测表。
    """)
    return


@app.function
def fit_joint(sizes, token_counts, loss_table):
    n = torch.tensor(sizes, dtype=torch.float64)[:, None]
    d = torch.tensor(token_counts, dtype=torch.float64)[None, :]
    measured = torch.tensor(loss_table)
    p = torch.tensor([1.0, 3.0, 0.3, 3.0, 0.3], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([p], lr=0.02)
    for step in range(4000):
        optimizer.zero_grad()
        predicted = joint_law(n, d, p[0], torch.exp(p[1]), p[2], torch.exp(p[3]), p[4])
        torch.mean((torch.log(predicted) - torch.log(measured)) ** 2).backward()
        optimizer.step()
    return p[0].item(), torch.exp(p[1]).item(), p[2].item(), torch.exp(p[3]).item(), p[4].item()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：读懂这次拟合

    看 `fit_joint`。Adam 在调整多少个数？它要把这些数拟合到多少个测得的损失上？它一共计算了多少次整张预测表？
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_boxes = hand.boxes(['调整的数', '测得的损失', '算出的表'])
    return (fit_boxes,)


@app.cell(hide_code=True)
def _(fit_boxes):
    hand.marked(fit_boxes, {'调整的数': 5, '测得的损失': 16, '算出的表': 4000}, {
        '调整的数': r"""数一数 `p` 有几项""",
        '测得的损失': r"""`loss_table` 的大小""",
        '算出的表': r"""循环的每一步算一次""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    十六个测量值拟合五个数：够用，但余地不大。读结果时要记住这一点，结果是 $(E, a, \alpha, b, \beta)$，已四舍五入：
    """)
    return


@app.cell
def _(loss_table, sizes, token_counts):
    law_numbers = fit_joint(sizes, token_counts, loss_table)
    np.round(law_numbers, 3)
    return (law_numbers,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    你应该会看到幂 $\alpha$ 和 $\beta$ 在 0.25 到 0.3 附近，下限 $E$ 在 0.3 附近。不要相信这个下限。你测到的最低损失大约是 3.5，是它的十倍，所以你的训练几乎没说明这条下降的曲线到哪里为止。一次拟合，只有在它被测量的那个范围内才靠得住。

    ## 先预测，再运行

    现在像真正用它的人那样用这条定律：在做一次训练之前先决定它。预算是 $1.86 \times 10^{12}$ 个 FLOP，比你表里最大那次训练的花费少一点。四种大小里，该让哪一种得到它？
    """)
    return


@app.cell
def _(law_numbers, sizes):
    budget = 1.86e12

    def fitted_law(knobs, tokens):
        return joint_law(knobs, tokens, *law_numbers)

    choice = best_size(budget, sizes, fitted_law)
    choice, np.round([fitted_law(n, tokens_for(budget, n)) for n in sizes], 3)
    return budget, fitted_law


@app.cell(hide_code=True)
def _(budget, fitted_law):
    _ns = np.exp(np.linspace(np.log(60000), np.log(1500000), 120))
    _curve = [(np.log(n), fitted_law(n, tokens_for(budget, n))) for n in _ns]
    _best = min(_curve, key=lambda point: point[1])
    chart((11, 14.3), (3.4, 4.1), "旋钮数的对数", "预测的损失", curve=_curve, marker=_best)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第一个数是定律的选择：第三种大小，387,392 个旋钮，也就是每个词元用 64 个数的那个模型。图上画出了在这笔预算下每种大小的预测损失：就是你手算时找到的那个山谷，谷底是平的。谷底平是好消息，因为稍微偏离最佳大小，代价很小。

    定律选中的模型会读 800,000 个词元，是你表里任何一次训练的两倍。比它小一档的会读 180 万个词元，是四倍多。所以这是一个真正的预测，落在测量范围之外。按钮会在这笔预算下运行三种最大的尺寸，并把每一种的预测和测量结果放在一起显示。（最小的那种被省掉了：它能负担 390 万个词元，而故事一共只有 200 万个。文本用完，也是一个真实的限制。）要 1 到 2 分钟。
    """)
    return


@app.cell(hide_code=True)
def _():
    test_button = mo.ui.run_button(label="运行三个候选（1 到 2 分钟）")
    test_button
    return (test_button,)


@app.cell
def _(budget, fitted_law, sizes, test_button, test_tokens, train_tokens, vocab_size, widths):
    mo.stop(not test_button.value, mo.md("*按下按钮运行这些候选。*"))
    outcome = np.zeros((3, 4))
    for row in range(3):
        affordable = tokens_for(budget, sizes[row + 1])
        measured = small_run(widths[row + 1], affordable, train_tokens, test_tokens, vocab_size)
        outcome[row] = [widths[row + 1], affordable, fitted_law(sizes[row + 1], affordable), measured]
    outcome.round(3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每一行是（每个词元的数的个数，读过的词元数，预测的损失，测得的损失）。在我们的机器上，测得的损失大约是 3.60、3.52 和 3.64：和预测的一样，中间那种大小胜出，而且每个预测和测量值的差距都在 0.1 以内。你的数字会有一点不同。

    这就是这个方法。十六次便宜的训练，在你花钱之前就告诉了你该怎么花这笔预算。

    ## 真实的定律说了什么，没说什么

    各个实验室做的正是这件事，只是训练规模大了一百万倍。有两个发现值得知道：

    - 幂很小，和你测到的一样。计算量多十倍，只换来损失温和而稳定的下降，绝不会是断崖式的。靠规模取得进步是真的，也是昂贵的。
    - 对大模型，最佳的分法测出来大约是**每个旋钮 20 个词元的文本**。在这个数被测出来之前，流行的是更大的模型读得太少；后来发现，训练得当的较小模型反而更好。

    你自己的最佳分法大约是每个旋钮 2 个词元，不是 20。这不是错误，而且是这一章里最重要的提醒：**缩放定律里的数，属于测量它们时的那套设置。**你的模型很小，旋钮大多在词元向量表里，故事也很简单。换一种模型、换一种文本或者换一种训练方式，常数就会变。方法可以沿用，数却必须重新测量。

    而且，一条定律只在测量过的范围附近才可信。你的这条只在两倍的文本量上检验过。没有人应该相信它对一千倍文本量的猜测：你没能确定的那个下限，会决定答案。

    ## 接下来

    文本越多，损失越低，但这里所有的文本都是好文本。在真实的互联网上，大部分文本是重复的、残缺的或者是垃圾。**第 25 章**讲的是怎样把一堆原始文本变成值得训练的数据。

    ## 你现在会做的事

    - 认出幂律，并说出每翻一倍能换来什么
    - 用对数把幂律变成一条直线，并从中读出 $k$ 和 $a$
    - 不用顺着山坡走，就拟合出穿过一组点的最佳直线
    - 用你自己的训练结果测出一条缩放定律
    - 算出一笔计算预算能买到什么，并选出最能用好它的模型大小
    - 说明为什么一条定律里的数不能照搬到另一套设置，也不能远远超出测量范围去用

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 第一项大规模研究是 Kaplan 等，*Scaling Laws for Neural Language Models*（2020）。三部分的定律和每个旋钮 20 个词元，出自 Hoffmann 等，*Training Compute-Optimal Large Language Models*（2022），也就是人们所说的 *Chinchilla* 论文。NumPy 的 `np.polyfit(xs, ys, 1)` 做的就是你的 `fit_line` 做的事。
    """)
    return


if __name__ == "__main__":
    app.run()
