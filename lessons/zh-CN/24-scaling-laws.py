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

    假设你拿到一笔预算，用来跑一次大规模训练：固定数量的 GPU，固定几周的时间。动手前必须先回答两个问题：模型该做多大？应该给它读多少文本？一旦猜错，整笔预算就浪费在了一个本可以更好的模型上。而且你没法靠反复试错来摸索——每试一次，花掉的就是一整次大训练的钱。

    破解这个难题的方法，是这个领域里最管用的习惯之一：先跑一批**小规模**实验——便宜到可以反复试——总结出结果背后的规律，再在真正花大钱之前，用这条规律去预测大训练的表现。这样的规律叫做**缩放定律**（scaling law）。这一章里，你会在自己的电脑上亲手测出一条缩放定律，并用它来规划一次训练。

    ## 幂律

    几乎所有缩放定律都长成同一种样子，叫做**幂律**（power law）：

    > $$y = a \times x^k$$

    这里 $x$ 是你投入的资源（训练文本量或模型旋钮数），$y$ 是你关心的训练效果（测试损失），$a$ 和 $k$ 是两个固定的常数。当指数 $k$ 是负数时，$x$ 越大，$y$ 就越小：负号表示取倒数（$x^{-1} = \frac{1}{x}$），$0.5$ 次方表示开平方根，所以 $x^{-0.5} = \frac{1}{\sqrt{x}}$。

    幂律有一条最核心的性质：**无论起点是多少**，只要把 $x$ 翻倍，$y$ 就会乘以 $2^k$：$a \times (2x)^k = 2^k \times (a \times x^k)$。也就是说，从 1 翻到 2 带来的**比例**改善，和从 100 万翻到 200 万完全一样——但每次翻倍的花费都是上一次的两倍，换来的改善比例却丝毫没有变多。

    ## 动手算：每次减半的定律

    来看 $y = 64 \times x^{-0.5}$，也就是 $\frac{64}{\sqrt{x}}$。当 $x = 1$ 时它等于 64。那么当 $x$ 分别是 4、16 和 64 时，它各等于多少？
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
    chart((0, 68), (0, 70), "x", "y", curve=[(x, 64 / np.sqrt(x)) for x in np.linspace(0.85, 64, 200)], dots=[(1, 64), (4, 32), (16, 16), (64, 8)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这张图很难看出门道：前三个点挤在最左边，肉眼根本分不清它究竟是幂律，还是随便哪条“先降后平”的曲线。

    ## 取对数，曲线变直线

    仔细看这四个点：相邻两点之间不是隔着相同的加减步长，而是隔着相同的**倍数**——$x$ 每次乘 4，$y$ 每次除以 2。既然是按倍数变化，不如直接数“乘了多少次”。这正是对数最擅长的事：把乘法变成加法，即 $\log(a \times b) = \log a + \log b$。把它用在 $x^k$（$x$ 自乘 $k$ 次）上，就得到 $\log(x^k) = k \times \log x$（这对任意实数 $k$ 都成立，不只是整数）。现在对幂律两边同时取对数：

    > $$\log y = \log a + k \times \log x$$

    把它当成一条直线方程来看：横轴画 $\log x$，纵轴画 $\log y$，**任何幂律都会变成一条直线，斜率正好是 $k$，截距是 $\log a$**。用对数作横纵坐标画出来的图叫**双对数图**（log-log plot）。如果实验数据在双对数图上排成一条直线，就说明它们遵循幂律，而最关键的指数 $k$ 就是这条直线的斜率。

    前面章节里我们用的都是自然对数（$\ln$），它数的是乘以 $e$ 的次数。手算时，数乘以 2 的次数（也就是翻倍的次数）更顺手。我们的四个点换算过来就是：

    | $x$ | $x$ 翻倍的次数 | $y$ | $y$ 翻倍的次数 |
    |---:|---:|---:|---:|
    | 1 | 0 | 64 | 6 |
    | 4 | 2 | 32 | 5 |
    | 16 | 4 | 16 | 4 |
    | 64 | 6 | 8 | 3 |

    比如 $64 = 2 \times 2 \times 2 \times 2 \times 2 \times 2$，就是从 1 开始翻倍 6 次。把两列“翻倍的次数”画成散点图：
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-0.5, 6.5), (2, 7), "x 翻倍的次数", "y 翻倍的次数", dots=[(0, 6), (2, 5), (4, 4), (6, 3)], lines=[(-0.5, 6)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一条笔直的线，四个点均匀排开。每向右 2 格就向下 1 格，斜率是 $-0.5$——这就是 $k$；起点在 6，翻倍 6 次正好是 64——这就是 $a$。

    换一种对数底数，斜率会变吗？**完全不会。**在 $\log y = \log a + k \times \log x$ 里，无论两边同时取以 2 为底的对数还是自然对数 $\ln$，只是把横纵坐标按相同比例缩放，斜率 $k$ 一点也不会变。所以手算时数“翻倍次数”最省事，写代码时统一用自然对数 `np.log` 和还原函数 `np.exp` 配对即可。

    ## 一步算出最佳直线

    真实的实验数据不会严丝合缝地落在一条直线上，所以我们需要找出最贴合它们的直线。这件事其实不陌生：第 01 章我们用均方误差衡量直线偏离数据点的程度，第 02 章用梯度下降顺着误差曲面一步步走到谷底。不过对于单条直线，有一条一步到位的捷径：谷底正好是斜率和截距两个方向的导数都为 0 的地方，联立这两个方程就能直接解出答案。我们跳过推导，直接看公式。记 $\bar{x}$ 为所有 $x$ 的平均值，$\bar{y}$ 为所有 $y$ 的平均值，那么最佳直线的斜率和截距是：

    > $$\text{斜率} = \frac{(x - \bar{x}) \times (y - \bar{y}) \text{ 之和}}{(x - \bar{x})^2 \text{ 之和}} \qquad\qquad \text{截距} = \bar{y} - \text{斜率} \times \bar{x}$$

    公式很直观：分子看的是变化方向——当 $x$ 高于平均值时，$y$ 是跟着高于平均值（乘积为正）还是反而低于平均值（乘积为负）？分母则按 $x$ 自身的分散程度把结果归一化。最后算出的截距保证直线一定穿过所有点的中心 $(\bar{x}, \bar{y})$。这套直接算最佳直线的方法叫做**最小二乘拟合**（least-squares fit）。

    ## 动手算：过三个点的直线

    用表里的前三行：点 $(0, 6)$、$(2, 5)$ 和 $(4, 4)$。依次求出两个均值、分子和分母，再算出斜率和截距。
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
    和你从图上直接读出来的结果一致：斜率 $-0.5$，截距 6。

    ## 练习 1：fit_line

    编写 `fit_line(xs, ys)`。`xs` 和 `ys` 是等长的 NumPy 数组，每对元素对应一个数据点。用上面那两个公式，返回最小二乘直线的 `(slope, intercept)`。

    对 `xs = [0, 1, 2]` 和 `ys = [1, 3, 5]`，它应该返回 `(2, 1)`，因为这三个点正好落在直线 $y = 2x + 1$ 上。
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
        r"""一共 3 到 4 行。把距离 `xs - np.mean(xs)` 存进一个变量里，因为分子和分母都要用它。先算斜率，再算截距。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：整个数组的对数
    `np.log` 会对数组里的每个数取自然对数，`np.exp` 负责把它还原：`np.exp(np.log(np.array([2.0, 5.0])))` 还是 `[2, 5]`。
    ///

    ## 练习 2：fit_power_law

    编写 `fit_power_law(xs, ys)`。`xs` 和 `ys` 是由正数组成的 NumPy 数组。函数返回在双对数图上最贴合这些点的幂律 $y = a \times x^k$ 的两个参数 $(a, k)$。思路就是“取对数，曲线变直线”那一节的做法，直线拟合直接调用你刚写好的 `fit_line`。

    对前面手算过的点 `xs = [1, 4, 16]` 和 `ys = [64, 32, 16]`，它应该返回 `(64, -0.5)`。
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
        r"""先对两个数组取对数，再拟合一条直线。直线的两个参数里，哪一个是 $k$？另一个是 $\log a$，别忘了还原成 $a$。""",
        r"""一共 2 行。`fit_line` 返回一对数，可以在 `=` 左边直接拆给两个变量。直接用 `np.log` 和 `np.exp` 配对就行——刚才说过，换底数不会改变斜率 $k$。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 亲手跑一批小规模实验

    接下来动手实测。函数 `small_run(d, tokens, …)` 会完整跑完一次训练，并返回最终的测试损失：

    - 它搭建一个 2 层的 Transformer（沿用第 20 章的结构），每个词元用 `d` 个数表示；
    - 用 AdamW 优化器配合预热和余弦衰减（第 21 章的训练配方），在故事数据集上训练 `tokens` 个词元，每个词元只读一遍；
    - 最后在一小段固定的测试故事上算出每个词元的平均意外程度——这就是损失，而 $e^{\text{损失}}$ 就是困惑度。

    它的完整代码在 `learn_llm/scaling.py` 里，没有引入任何新概念。我们准备测试 4 种模型宽度（16 到 128）和 4 种训练文本量（5 万到 40 万个词元），一共 16 次实验。下面的单元格载入词元，并算出每种宽度对应的旋钮数（参数量）。
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
    参数量从大约 7.8 万个旋钮一路涨到将近 100 万个。最大的一次实验读的文本是最小那次的 8 倍，模型规模又是 12 倍，总算力开销大约相差 100 倍。点击下面的按钮会跑完全部 16 次实验，耗时约 1 到 3 分钟；在它跑完之前，后面依赖结果的单元格会显示 `Ancestor stopped`，这是正常现象。
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
    表里每一行代表一种模型大小（从小到大），每一列代表一种文本量（从少到多）。横着看一行：读的文本越多，损失越低；竖着看一列：模型旋钮越多，损失也越低。这都在意料之中，真正关键的问题是——它降得有**多快**？

    先挑第三行（每个词元 64 个数的模型），用你刚写的函数对它的四个损失拟合一条幂律：
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
    看图时不用心算 $e^{12}$：横轴自然对数 `11` 约对应 6 万个词元，`12` 约 16 万，`13` 约 44 万。

    四个数据点紧紧贴在直线上——在这个范围内，损失随文本量的变化确实遵循幂律。你应该会看到 $k$ 在 $-0.14$ 附近。按翻倍规则算，训练文本每翻一倍，损失就乘以 $2^{-0.14} \approx 0.91$：每次大约降掉 9%，但每一次翻倍的开销都是上一次的两倍。

    竖着看一列也一样。下面对最后一列（40 万个词元）按旋钮数拟合幂律：
    """)
    return


@app.cell
def _(loss_table, sizes):
    np.round(fit_power_law(sizes, loss_table[:, 3]), 3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    $k$ 在 $-0.1$ 附近：模型规模翻一倍，损失大约减少 7%。

    不过，单纯的幂律不可能永远成立。幂律会一路降向 0，但没有任何模型能把故事数据集的损失降到 0——故事下一句怎么写，本身就带有作者的自由发挥。损失下面一定有一道不可逾越的**下限**（floor），也就是哪怕完美模型也消除不掉的意外程度。稍后我们会把这道下限加进公式里。

    ## 一笔算力预算能买到什么

    第 22 章给过一条训练开销的估算公式：一个有 $N$ 个旋钮的模型读完 $D$ 个词元，大约需要 $6 \times N \times D$ 次浮点运算（FLOP）。一笔**计算预算**（compute budget）就是你总共能花掉的 FLOP 数量，所以它只锁定了乘积 $N \times D$，至于怎么在模型大小和文本量之间分配，全看你的选择：模型大一倍，能读的文本就只能减半。

    ## 动手算：分配一笔预算

    假设你的预算是 $6 \times 10^{12}$ 个 FLOP。一个有 1,000,000 个旋钮的模型能读多少词元？换成一个 4,000,000 旋钮的模型呢？
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
        '4,000,000 个旋钮能读的词元数': r"""旋钮是四倍，词元就只有四分之一""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 3：tokens_for

    编写 `tokens_for(budget, knobs)`：计算一个有 `knobs` 个旋钮的模型，花掉 `budget` 个 FLOP 能读多少词元（不必取整）。

    `tokens_for(6e12, 1e6)` 应该返回 1,000,000，和你手算的结果一致。
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
    ## 把模型大小和文本量写进同一条定律

    要想权衡怎么分配预算，我们需要一条同时考虑模型旋钮数和文本量的定律。对语言模型最管用的形式，是把三部分加在一起：

    > $$\text{损失}(N, D) = E + \frac{a}{N^{\alpha}} + \frac{b}{D^{\beta}}$$

    - $E$ 是不可逾越的下限：哪怕一个完美的模型读了无穷多文本，仍然会剩下的损失。
    - $\frac{a}{N^{\alpha}}$ 是因为模型**太小**而多出来的损失。它是 $N$ 的幂律（除以 $N^{\alpha}$ 等于乘以 $N^{-\alpha}$），随着模型变大逐渐减小。
    - $\frac{b}{D^{\beta}}$ 是因为读的文本**太少**而多出来的损失，随着文本量增加逐渐减小。

    $\alpha$ 和 $\beta$ 分别是希腊字母 alpha 和 beta。整条定律一共 5 个待测参数：$E$、$a$、$\alpha$、$b$ 和 $\beta$。

    这样一来，取舍就一目了然了。在预算固定的前提下，模型越大，留给训练文本的算力就越少，于是中间项（模型容量不足的误差）变小，最后一项（读得太少的误差）反而变大；模型太小则正好相反。两者之间一定存在一个最划算的平衡点。

    ## 练习 4：joint_law

    编写 `joint_law(knobs, tokens, floor, a, alpha, b, beta)`，实现上面的公式。它必须既能处理普通的单个数字，也能直接处理由旋钮数和词元数组成的数组或张量——只要只使用 `+`、`*` 或 `/` 以及 `**`，就能自动支持。

    当旋钮数 100、词元数 10,000、下限 2、$a = 10$、$b = 300$、两个指数都是 0.5 时，它应该返回 $2 + \frac{10}{10} + \frac{300}{100} = 6$。
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
        r"""三部分加起来。Python 里乘方用 `**`，例子里的 $100^{0.5} = 10$ 和 $10000^{0.5} = 100$ 可以帮你分别核对每一项。""",
        r"""只要一行。可以除以正次幂，也可以乘以负次幂：`x ** -k` 就是 $x^{-k}$。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：最佳分配方案

    来看一条数字简单的玩具定律：$\text{损失} = 1 + \frac{100}{N} + \frac{400}{D}$。总预算是 60,000 个 FLOP，也就是 $N \times D = 10{,}000$。先算出下面四种模型大小各自能读多少词元 $D$，再算出它们对应的损失。
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
    最小的模型旋钮太少，最大的模型文本太少。四个方案里表现最好的是 $N = 50$，此时两项误差刚好势均力敌。按这种方式选出的模型，在给定预算下称为**计算最优**（compute-optimal）。

    /// admonition | 🧰 工具箱：最小的数在哪儿？
    `np.argmin` 是 `np.argmax` 的孪生兄弟，返回最小元素所在的下标。比如 `np.argmin([7, 3, 9])` 返回 1。
    ///

    ## 练习 5：best_size

    编写 `best_size(budget, sizes, law)`。`sizes` 是可供选择的旋钮数列表，`law` 是一个函数，调用 `law(knobs, tokens)` 会返回预测损失（第 02 章讲过，函数本身也可以像数字一样当作参数传进来）。先用你的 `tokens_for` 算出每种大小能负担的词元数，再返回预测损失最低的那个模型大小。

    拿刚才手算的例子测试：预算 60,000、大小列表 `[10, 50, 100, 200]`、配合那条玩具定律，函数应该返回 50。
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
        r"""一共 2 行。第一行用列表推导式（方括号里写 `for n in sizes`），对每个 `n` 和它能负担的词元数调用 `law`；第二行返回 `sizes` 里对应的那一项。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 用实验数据拟合联合定律

    联合定律在任何坐标系下都不是直线，所以 `fit_line` 没法直接算出它的 5 个参数。但我们早就掌握了万能的拟合办法：梯度下降。下面的 `fit_joint` 就是第 10 章写过的 PyTorch 训练循环——把 `joint_law` 当作“模型”，让 Adam 优化器去调它的 5 个参数。

    代码里藏着三个很实用的小技巧：

    - **用 `torch.exp` 保证 $a$ 和 $b$ 永远为正**：系数 `a` 和 `b` 不能变成负数，所以优化器直接调整的是 `p[1]` 和 `p[3]`，传入 `joint_law` 时再取 `torch.exp(p[1])` 和 `torch.exp(p[3])`。无论优化器把 `p[1]` 调成什么负数（比如 $-10$），取 `exp` 后结果 $e^{-10} \approx 0.000045$ 依然大于 0，从而强制系数始终为正。
    - **在对数空间衡量相对误差**：损失函数算的是 `torch.log(predicted)` 与 `torch.log(measured)` 之差的均方误差。因为 $\log(1.01 y) - \log y = \log 1.01 \approx 0.01$，对数差直接对应百分比误差——不管表里某个格子的损失是 5.0 还是 3.5，1% 的相对误差权重完全一样。
    - **用广播一次算出整张 $4 \times 4$ 预测表**：`[:, None]` 把 4 种模型大小变成 $4 \times 1$ 的一列，`[None, :]` 把 4 种词元数变成 $1 \times 4$ 的一行，PyTorch 广播就能在一条式子里算出全部 16 个格子的预测值。
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
    你应该会看到指数 $\alpha$ 和 $\beta$ 在 0.25 到 0.3 附近，下限 $E$ 在 0.3 附近。不过先别太相信这个下限：你实测到的最低损失大约是 3.5，比这个下限高出十倍，所以仅凭这 16 次小实验，根本看不出这条下降曲线最终会在哪里触底。任何拟合出的规律，都只在测量覆盖的范围内才靠得住。

    ## 先预测，再验证

    现在来模拟真实场景：在真正启动训练之前，先用拟合好的定律做决策。假设总预算是 $1.86 \times 10^{12}$ 个 FLOP，比刚才表里最大那次实验的花费略少一点。四种模型大小里，该把预算交给哪一种？

    （下面代码里的 `*law_numbers` 是 Python 的拆包写法：把 `law_numbers` 里的 5 个数依次拆开，当作 `floor`、`a`、`alpha`、`b`、`beta` 这 5 个参数传给 `joint_law`。）
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
    横轴上的自然对数刻度：`11` 约对应 6 万个旋钮，`12` 约 16 万，`13` 约 44 万，`14` 约 120 万。

    输出的第一个数就是定律选出的赢家：第三种规模，387,392 个旋钮（每个词元 64 个数的模型）。曲线画出了这笔预算下不同规模的预测损失——正是你前面手算时看到的那个山谷，而且谷底很平。谷底平是个好消息：哪怕选的模型大小稍微偏离理论最优点，损失也几乎不受影响。

    按这笔预算，定律选中的模型会读 80 万个词元，是前面实验表里最大文本量的两倍；比它小一档的模型则会读 180 万个词元，超过四倍。因此这是一次真正走出测量范围的预测。点击下面的按钮，在这笔预算下实测三种较大的模型，把预测损失和实测损失并排对比。（最小的模型被跳过了：它能负担 390 万个词元，但整个故事数据集一共只有 200 万个词元——数据不够用，同样是现实中的硬限制。）耗时约 1 到 2 分钟。
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
    每一行依次是（宽度 `d`、读过的词元数、预测损失、实测损失）。在我们的机器上，实测损失大约是 3.60、3.52 和 3.64：果然是中间那种大小胜出，而且每个预测值和实测值的误差都在 0.1 以内。你的机器上数字会略有不同。

    这就是整套方法的威力：花小钱跑完 16 次小实验，就能在真正投入大预算之前，提前算出最划算的分配方案。

    ## 真实缩放定律告诉我们什么，又没告诉我们什么

    各大实验室做的正是同一套流程，只是规模放大了一百万倍。其中有两个结论特别值得记住：

    - 指数很小，和你亲手测到的一样。算力翻十倍，换来的是损失温和而稳定的下降，绝不会出现断崖式暴跌。靠堆规模换进步是真实有效的，但也非常烧钱。
    - 对于大语言模型，计算最优的配比大约是**每个旋钮配 20 个词元的训练文本**。在这条规律被测出来之前，业界普遍倾向把模型造得很大却喂得太少；后来大家才发现，用同样的算力把稍小一点的模型充分喂饱，效果反而更好。

    你自己的实验算出的最佳配比大约是每个旋钮 2 个词元，而不是 20。这并不是算错了，而是这一章最重要的提醒：**缩放定律里的具体常数，只属于测量它们时的那套实验设置。**你的模型很小，大部分旋钮集中在词元向量表里，故事文本也比较简单。只要换一种模型结构、换一份数据集或者换一套训练配方，常数就会跟着变。方法可以原封不动地沿用，但具体数字必须重新测量。

    此外，任何缩放定律都只在测量范围附近才靠得住。我们只在两倍文本量上验证过它，绝不该拿它去猜一千倍文本量时的结果——到那时，刚才根本没测准的下限 $E$ 会反过来主导整个答案。

    ## 接下来

    更多文本能压低损失，但这一章里的文本全是干净整齐的。在真实的互联网上，大多数网页充斥着重复、残缺和垃圾内容。**第 25 章**我们就来看看，怎样把一堆原始文本清洗成真正值得训练的高质量数据。

    ## 你现在会做的事

    - 认出幂律，并说出每翻一倍能换来多少比例的改善
    - 用对数把幂律拉成一条直线，并从中读出指数 $k$ 和系数 $a$
    - 不用梯度下降，直接用公式一步拟合出穿过一组点的最佳直线
    - 通过自己的小规模训练实验，测出一条联合缩放定律
    - 算出一笔计算预算能买多少词元，并选出计算最优的模型规模
    - 说明为什么缩放定律的常数不能直接照搬到别的设置，也不能远远超出测量范围去外推

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** 第一项大规模缩放定律研究是 Kaplan 等人的 *Scaling Laws for Neural Language Models*（2020）。三部分联合定律和“每个旋钮 20 个词元”的结论出自 Hoffmann 等人的 *Training Compute-Optimal Large Language Models*（2022），也就是常说的 *Chinchilla* 论文。NumPy 的 `np.polyfit(xs, ys, 1)` 做的就是你写的 `fit_line`。
    """)
    return


if __name__ == "__main__":
    app.run()
