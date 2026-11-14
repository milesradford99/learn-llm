import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="22 · 让它跑得快")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 22 · 让它跑得快

    训练一个模型意味着做海量的算术。让它变快，首先不在于巧妙的技巧，而在于**测量**，弄清时间到底花在哪里；然后是**估算**（数一数要做多少算术），弄清它最快可能有多快。这一章两件事都做，然后尝试两种提速办法，并发现其中一种可能反而变慢，在测试这一章的电脑上就是如此。

    ## 测量时间

    不管测什么，只测一次都不可靠：第一次运行往往比较慢，因为电脑还在准备，而且每次运行都会有些波动。所以我们先运行一次预热，再计时运行几次，取平均。

    /// admonition | 🧰 工具箱：时钟
    `time.perf_counter()` 给出以秒计的时间，从某个随意的起点算起。单独读一次没有意义，但两次读数的差，就是它们之间那几行花掉的时间：先写 `started = time.perf_counter()`，然后干活，最后是 `time.perf_counter() - started`。
    ///

    ## 练习 1：time_it

    写 `time_it(f, repeats)`。`f` 是一个没有输入的函数，`repeats` 是一个整数。它先调用一次 `f` 来预热，这一次不计时；然后再调用 `repeats` 次，返回这些调用里平均每次花了多少秒。

    其中一项检查会交给它这样一个函数：第一次调用要 0.1 秒，之后每次 0.01 秒。你的答案必须是 0.01。
    """)
    return


@app.function
def time_it(f, repeats):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "time_it", time_it, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""四部分：预热的那次调用，读一次时钟，一个反复调用 `f` 的循环，最后是 `return`。时钟在预热之后才读，这样很慢的第一次调用就不算在内。""",
        r"""一共 5 行。最后一行返回从 `started` 到现在过去的时间，除以 `repeats`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 数一数算术

    网络里的工作几乎全是矩阵乘法。`A @ B` 中，`A` 的形状是 `(n, k)`，`B` 的形状是 `(k, m)`，结果里的每个数都是一个长度为 $k$ 的点积：$k$ 次乘法，大约 $k$ 次加法。两个这样的数做一次乘法或一次加法，算一次**浮点运算**（floating-point operation），简称 **FLOP**。（“浮点数”就是计算机里带小数点的数，也就是我们一直在用的那种。）

    ## 动手算：一次小的乘法

    取形状为 `(2, 3)` 的 `A` 和形状为 `(3, 4)` 的 `B`。`A @ B` 里有多少个数？按 $k$ 次乘法加 $k$ 次加法来算，每个数要多少个 FLOP？整个乘法一共多少个 FLOP？
    """)
    return


@app.cell(hide_code=True)
def _():
    flop_boxes = hand.boxes(['结果里的数', '每个数的 FLOP', 'FLOP 总数'])
    return (flop_boxes,)


@app.cell(hide_code=True)
def _(flop_boxes):
    hand.marked(flop_boxes, {'结果里的数': 8, '每个数的 FLOP': 6, 'FLOP 总数': 48}, {
        '结果里的数': r"""结果的形状是 `(2, 4)`""",
        '每个数的 FLOP': r"""长度为 3 的点积：3 次乘法，3 次加法""",
        'FLOP 总数': r"""第一个答案乘以第二个答案""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一般来说，结果里有 $n \times m$ 个数，每个要 $2k$：

    > $$\text{一次矩阵乘法的 FLOP 数} \approx 2 \times n \times k \times m$$

    ## 练习 2：matmul_flops

    写 `matmul_flops(n, k, m)`，也就是这个公式：形状为 `(n, k)` 的矩阵乘以形状为 `(k, m)` 的矩阵要多少个 FLOP。你手算的答案就是检查：`matmul_flops(2, 3, 4)` 得到 `48`。

    检查之后的单元格用它测量你的电脑：给两个 $1024 \times 1024$ 矩阵的乘法计时，再用 FLOP 数除以秒数。十亿个 FLOP 叫一个 **gigaFLOP**。（这个单元格需要练习 1 和练习 2，它们通过之前会显示错误。这是正常的，后面用到你的函数的单元格也一样。）
    """)
    return


@app.function
def matmul_flops(n, k, m):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "matmul_flops", matmul_flops, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行：就是那个公式，每个 $\times$ 写成 `*`。""",
    ], LOCALE)
    return


@app.cell
def _():
    A = torch.rand(1024, 1024)
    B = torch.rand(1024, 1024)
    def product():
        return A @ B

    seconds_32 = time_it(product, 5)
    machine_speed = matmul_flops(1024, 1024, 1024) / seconds_32
    round(machine_speed / 1e9, 1)
    return A, B, machine_speed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这就是你电脑的最佳速度，单位是每秒 gigaFLOP。你的数字会和别人的不一样。（`product` 是一个没有输入的函数，所以 `time_it` 可以一遍遍调用它。）在测试这一章的机器上，它大约是 180；一块现代 GPU 大约快 1,000 倍，而最大规模的训练会同时用上成千上万块这样的 GPU。

    ## 训练的代价

    对一个有 $N$ 个旋钮的模型，下面这条规则对做计划来说足够准确。在正向计算中，每个旋钮对每个词元大约参与一次乘法和一次加法：2 个 FLOP。反向计算大约是它的两倍：一次矩阵乘法求旋钮的斜率，另一次求传回该层输入的责任（第 08 章的那两次乘法）。这又是 4 个。

    ## 动手算：一次很小的训练

    一个模型有 1,000 个旋钮，把 10 个词元读一遍。按这条规则，正向计算要多少个 FLOP？反向计算呢？两者加起来呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    cost_boxes = hand.boxes(['正向', '反向', '合计'])
    return (cost_boxes,)


@app.cell(hide_code=True)
def _(cost_boxes):
    hand.marked(cost_boxes, {'正向': 20000, '反向': 40000, '合计': 60000}, {
        '正向': r"""每个旋钮、每个词元 2 个 FLOP""",
        '反向': r"""正向的两倍""",
        '合计': r"""也就是每个旋钮、每个词元 6 个 FLOP""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    所以：

    > $$\text{训练的 FLOP 数} \approx 6 \times N \times \text{词元数}$$

    它没有算注意力分数本身，这部分只在窗口很长时才要紧。

    ## 练习 3：training_flops

    写 `training_flops(knobs, tokens)`，也就是这条规则。`tokens` 是整次训练里读过的词元数，读几遍就算几遍。`training_flops(1000, 10)` 得到 `60000`。
    """)
    return


@app.function
def training_flops(knobs, tokens):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "training_flops", training_flops, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行：6，乘以旋钮数，再乘以词元数。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 从 FLOP 到秒

    电脑不会一直以最佳速度工作。一次训练达到的速度占最佳速度的比例，叫作这次训练的**利用率**（utilisation）：0.5 就是一半。所以：

    > $$\text{秒数} = \frac{\text{FLOP 数}}{\text{速度} \times \text{利用率}}$$

    ## 动手算：要多久？

    一次训练需要 30 teraFLOP，也就是 $30 \times 10^{12}$ 个 FLOP；电脑的最佳速度是每秒 100 gigaFLOP，也就是 $100 \times 10^9$。全速运行要多少秒？利用率为 0.5 时呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    seconds_boxes = hand.boxes(['全速时的秒数', '利用率 0.5 时的秒数'])
    return (seconds_boxes,)


@app.cell(hide_code=True)
def _(seconds_boxes):
    hand.marked(seconds_boxes, {'全速时的秒数': 300, '利用率 0.5 时的秒数': 600}, {
        '全速时的秒数': r"""$30 \times 10^{12} / (100 \times 10^9)$：12 个零去掉 9 个，剩下 30,000 除以 100""",
        '利用率 0.5 时的秒数': r"""速度减半，时间加倍""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：run_seconds

    写 `run_seconds(flops, speed, utilisation)`，也就是上面的公式。`flops` 是一次训练的算术量，`speed` 是电脑的最佳速度（每秒多少个 FLOP），`utilisation` 是 0 到 1 之间的一个比例。`run_seconds(30e12, 100e9, 0.5)` 得到 `600.0`。（`30e12` 就是 $30 \times 10^{12}$：`e` 的作用是移动小数点，和第 09 章一样。）

    第 21 章的训练用大约 117 万个旋钮，把大约 200 万个词元读了两遍。检查之后的单元格算出它的算术量，单位是 teraFLOP（一个 teraFLOP 是 $10^{12}$ 个 FLOP），以及按你电脑的最佳速度要多少秒。
    """)
    return


@app.function
def run_seconds(flops, speed, utilisation):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "run_seconds", run_seconds, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行：一个除法。分母是两样东西相乘，所以要单独加括号。""",
        r"""如果 `run_seconds(30e12, 100e9, 0.5)` 得到 150 而不是 600，就是利用率跑到分子上去了：不加括号的话，Python 会先除以速度，再乘以 0.5。""",
    ], LOCALE)
    return


@app.cell
def _(machine_speed):
    flops_run = training_flops(1.17e6, 2 * 2.03e6)
    round(flops_run / 1e12, 1), round(run_seconds(flops_run, machine_speed, 1.0))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 28 teraFLOP，全速运行只要两三分钟。把第二个数和第 21 章的按钮在你电脑上实际用的时间比一比：在心里用前者除以后者，就得到你这次训练的利用率。在测试机器上，实际训练用了四五分钟，所以大约用到了电脑最佳速度的一半。剩下的时间花在小操作上，比如 softmax、归一化和搬运数字，它们算术很少，却要花时间。对一个 70 亿个旋钮、一万亿个词元的模型，同一个公式给出 $4.2 \times 10^{22}$ 个 FLOP，需要大约一千块 GPU，每块每秒约 200 万亿个 FLOP，以一半的最佳速度跑大约一周。

    ## 数一数内存

    用 Adam 训练时，每个旋钮需要四个数：它自己、它的斜率，以及 Adam 的两个平均值。用 32 位时，每个数占 4 个字节。在此之上还有**激活值**（activations）：正向计算为反向计算保留的所有值，它们随批大小和窗口长度增长。

    ## 动手算：一个模型要多少字节

    用 Adam 训练时，一个旋钮需要多少个字节？一个有 100 万个旋钮的模型，它的旋钮、斜率和平均值一共需要多少 MB？（1 MB 是一百万个字节。）
    """)
    return


@app.cell(hide_code=True)
def _():
    byte_boxes = hand.boxes(['一个旋钮的字节数', '一百万个旋钮的 MB 数'])
    return (byte_boxes,)


@app.cell(hide_code=True)
def _(byte_boxes):
    hand.marked(byte_boxes, {'一个旋钮的字节数': 16, '一百万个旋钮的 MB 数': 16}, {
        '一个旋钮的字节数': r"""4 个数，每个 4 个字节""",
        '一百万个旋钮的 MB 数': r"""一百万乘以第一个答案，再换成以百万字节为单位""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：adam_bytes

    写 `adam_bytes(knobs)`：用 Adam 训练、数字为 32 位时，一个模型的旋钮、斜率和平均值需要多少个字节。`adam_bytes(1000000)` 得到 `16000000`。
    """)
    return


@app.function
def adam_bytes(knobs):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "adam_bytes", adam_bytes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行：一个旋钮的字节数，乘以旋钮的个数。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    对第 20 章的模型（117 万个旋钮），这不到 19 MB。对一个 70 亿个旋钮的模型，是 112 GB，大多数 GPU 都装不下。第 23 章讲的就是这时该怎么办。

    ## 提速 1：更小的数

    32 位的数能保证大约 7 位正确的数字（第 10 章）。两种 16 位格式能省一半内存，在专门为它们设计的硬件上，速度能快一倍以上：

    - **float16** 大约保证 3 位数字，但装不下大于 65,504 的数；
    - **bfloat16** 保证的位数更少，大约 2 到 3 位，但能达到和 float32 一样大的数。

    /// admonition | 🧰 工具箱：查一种格式
    `torch.finfo(dtype)` 描述一种数字格式：它的 `max` 是这种格式能表示的最大数，`eps` 是 1 和它能表示的下一个数之间的间隔，用来衡量它有多精确。`eps` 约为 0.001 就是千分之一，大约 3 位正确的数字。
    ///

    ## 动手算：装得下吗？

    运行下面的单元格之前，先猜一猜。在 float16 里，大于 65,504 的数都装不下：$200 \times 200$ 装得下吗？$300 \times 300$ 呢？装得下填 1，装不下填 0。另外，一百万个 16 位格式的数占多少 MB？
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_boxes = hand.boxes(['200 × 200 装得下', '300 × 300 装得下', '一百万个数的 MB 数'])
    return (fit_boxes,)


@app.cell(hide_code=True)
def _(fit_boxes):
    hand.marked(fit_boxes, {'200 × 200 装得下': 1, '300 × 300 装得下': 0, '一百万个数的 MB 数': 2}, {
        '200 × 200 装得下': r"""$200 \times 200 = 40{,}000$。它大于 65,504 吗？""",
        '300 × 300 装得下': r"""$300 \times 300 = 90{,}000$。它大于 65,504 吗？""",
        '一百万个数的 MB 数': r"""16 位是 2 个字节""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的单元格显示每种格式的 `max` 和 `eps`，然后是 70,000 放进 float16 的结果，以及 $1 + 0.001$ 放进 bfloat16 的结果：
    """)
    return


@app.cell
def _():
    formats = []
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        formats.append((dtype, torch.finfo(dtype).max, torch.finfo(dtype).eps))
    formats, torch.tensor(70000.0, dtype=torch.float16), (torch.tensor(1.0, dtype=torch.bfloat16) + 0.001).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    70,000 在 float16 里变成 `inf`，即无穷大：太大了。而在 bfloat16 里，1 + 0.001 还是 1：变化太小，记不下来。只用 16 位训练会丢掉小的更新，所以真实的训练用**混合精度**（mixed precision）：矩阵乘法用 16 位，而旋钮、Adam 的平均值和需要精度的求和保持 32 位。

    它能让*你的*电脑变快吗？只有处理器带有 16 位运算的硬件时才行。下面的单元格用每种格式计时同一个乘法，单位是每秒 gigaFLOP，所以越高越快。可能要几秒钟。
    """)
    return


@app.cell
def _(A, B):
    speeds = []
    for dtype_try in [torch.float32, torch.bfloat16]:
        A_try, B_try = A.to(dtype_try), B.to(dtype_try)

        def product_try():
            return A_try @ B_try

        speeds.append((dtype_try, round(matmul_flops(1024, 1024, 1024) / time_it(product_try, 3) / 1e9, 1)))
    speeds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在测试机器上，bfloat16 比 float32 **慢了 100 多倍**，因为它的处理器没有 16 位运算的硬件，只能一步步模拟。你的数字可能不同：一些较新的处理器和 Apple 芯片就有这种硬件。这正是要点：一种提速办法只对特定的硬件有效，想知道有没有用，只能测量。在 GPU 上，混合精度是常规做法。

    ## 提速 2：不用的东西就别造

    注意力会造出一张分数表，每对词一个分数：窗口 1,024 个词元、4 个头、一批 8 段时，大约是 3,350 万个数，134 MB，要写进内存再读出来好几次（掩码、softmax、乘法）。这时限制速度的是内存，而不是算术：处理器算得比它从主内存取数快得多。处理器旁边那一小块快速内存，也就是**缓存**（cache），速度全靠它。

    **分块注意力**（tiled attention）从不造出整张表。对每个查询，它沿着键一个分块一个分块地走，同时保留一个累计结果。窍门在 softmax 上：它需要所有分数的最大值和总和，而这要到最后才知道。但两者都可以作为累计值保存，一路修正：

    - 保留一个累计最大值 `best`，一个 $e^{\text{score} - \text{best}}$ 的累计和 `total`，以及值的累计加权和 `acc`；
    - 新的分块带来更大的最大值时，之前累计的一切都是按旧的最大值算的，所以把 `total` 和 `acc` 乘以 $e^{\text{旧 best} - \text{新 best}}$（第 07 章：$e^{a-b}$ 能正确地重新缩放）；
    - 最后，`acc / total` 正好是 softmax 的加权平均。

    这就是 **FlashAttention** 的思路。

    先看一个小例子：每个向量 1 个数，分数是 $[1, 3, 2]$，值是 $[10, 20, 30]$，每个分块 1 个。

    | 分块 | `best` | 修正量 | `total` | `acc` |
    |---|---|---|---|---|
    | 1 | 1 | 0 | 1 | 10 |
    | 2 | 3 | $e^{1 - 3} \approx 0.135$ | $1 \times 0.135 + 1 = 1.135$ | $10 \times 0.135 + 20 = 21.35$ |
    | 3 | 3 | $e^{3 - 3} = 1$ | $1.135 + e^{2 - 3} = 1.503$ | $21.35 + 0.368 \times 30 = 32.39$ |

    而 $32.39 / 1.503 \approx 21.55$，和一次对三个分数做 softmax 的结果相同。在第二个分块，最大值变大了，所以之前的一切都被缩小。在第三个分块它没变，所以修正量是 1。

    ## 动手算：四个分数，两个分块

    一个查询，每个向量 1 个数，所以分数就是键乘以查询；我们取查询为 1，$\sqrt{d}$ 也是 1，略去不写。分数是 $[0, 1, 2, 0]$，值是 $[10, 20, 30, 40]$，每个分块 2 个：先是分数 $[0, 1]$ 和值 $[10, 20]$，再是分数 $[2, 0]$ 和值 $[30, 40]$。你会用到 $e^0 = 1$、$e^{-1} = 0.368$ 和 $e^{-2} = 0.135$。

    对每个分块：新的 `best` 是到目前为止最大的分数。每个权重是 $e^{\text{score} - \text{best}}$。之前累计的东西要乘以 $e^{\text{旧 best} - \text{新 best}}$，然后把这个分块的权重加进 `total`，把权重乘以值的结果加进 `acc`。第一个分块之前什么都没累计，所以直接加就行。
    """)
    return


@app.cell(hide_code=True)
def _():
    tile_boxes = hand.boxes(['第 1 块之后的 best', '第 1 块之后的 total', '第 1 块之后的 acc', '第 2 块之后的 best', '第 2 块之后的 total', '第 2 块之后的 acc', '答案'], step=0.001)
    return (tile_boxes,)


@app.cell(hide_code=True)
def _(tile_boxes):
    hand.marked(tile_boxes, {'第 1 块之后的 best': 1, '第 1 块之后的 total': 1.368, '第 1 块之后的 acc': 23.68, '第 2 块之后的 best': 2, '第 2 块之后的 total': 1.638, '第 2 块之后的 acc': 44.11, '答案': 26.93}, {
        '第 1 块之后的 best': r"""分数 0 和 1 里较大的那个""",
        '第 1 块之后的 total': r"""两个权重是 $e^{0 - 1}$ 和 $e^{1 - 1}$：把它们加起来""",
        '第 1 块之后的 acc': r"""$0.368 \times 10 + 1 \times 20$""",
        '第 2 块之后的 best': r"""到目前为止最大的分数现在是 2""",
        '第 2 块之后的 total': r"""原来的 total 乘以 $e^{1 - 2}$，再加上新的权重 $e^{2 - 2}$ 和 $e^{0 - 2}$""",
        '第 2 块之后的 acc': r"""原来的 `acc` 乘以 $e^{1 - 2}$，再加上 $1 \times 30 + 0.135 \times 40$""",
        '答案': r"""`acc` 除以 `total`""",
    }, LOCALE, tolerance=0.05, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一次对四个分数做 softmax，得到的也是 26.93：这两个分块从来不需要同时待在内存里。

    ## 练习 6：tiled_attention

    写 `tiled_attention(q, K, V, tile)`。输入是一个形状为 `(d,)` 的查询 `q`，形状为 `(n, d)` 的键 `K`，以及形状为 `(n, v)` 的值 `V`；不用掩码，分数和第 16 章一样按 $\sqrt{d}$ 缩放。`tile` 是每次取多少个键。它返回一个形状为 `(v,)` 的张量，和 `torch.softmax(K @ q / np.sqrt(d), dim=0) @ V` 的答案相同，但手里的分数从来不超过 `tile` 个。

    先准备三个累计值：`best` 取 `-np.inf`（负无穷大，这样任何分数都比它大；而且 $e^{-\infty}$ 是 0，所以第一个分块可以和其他分块一样处理），`total` 取 `0.0`，`acc` 取一个有 `v` 个零的张量。走过各个分块的循环直接给你，因为按步长切片是第 09 章的工具：

    ```
    for start in range(0, len(K), tile):
    ```

    在循环里面，这个分块的键是 `K[start:start + tile]`，值是 `V[start:start + tile]`。照着你手算时的做法来：

    | 步骤 | 写成数学 |
    |---|---|
    | 这个分块的分数 | $\text{键} \cdot q / \sqrt{d}$ |
    | 新的最大值 | `best` 和这个分块最大的分数里较大的那个 |
    | 给之前累计的东西用的修正量 | $e^{\text{best} - \text{新的最大值}}$ |
    | 这个分块的权重 | $e^{\text{分数} - \text{新的最大值}}$ |
    | `total` | $\text{total} \times \text{修正量} + \text{权重之和}$ |
    | `acc` | $\text{acc} \times \text{修正量} + \text{权重} \cdot \text{值}$ |
    | `best` | 变成新的最大值 |

    循环结束后，答案是 `acc` 除以 `total`。`torch.max(scores).item()` 和 `torch.sum(weights).item()` 给出普通的数；`torch.exp` 用在张量上，`np.exp` 用在普通的数上。
    """)
    return


@app.function
def tiled_attention(q, K, V, tile):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("22", "tiled_attention", tiled_attention, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""循环里面七行，表格的每一行对应一行，顺序不变。`best` 必须最后更新：修正量既要用旧的 `best`，也要用新的。一个分块的键和 `q` 的点积用 `@`，权重和这个分块的值的点积也用 `@`。""",
        r"""循环里的前两行是 `scores = K[start:start + tile] @ q / np.sqrt(len(q))` 和 `new_best = max(best, torch.max(scores).item())`。如果分数很大的那项检查没通过，说明你取 $e$ 的次方之前没有先减去最大值。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    用 Python 一次处理一个查询，这样很慢：重点在于*方法*。在真实的系统里，同样的步骤在处理器上的一个编译好的函数里运行，把分块保留在缓存中。PyTorch 就有一个（这叫“融合”：把这些步骤合成一个函数）：`F.scaled_dot_product_attention`。下面的单元格在两种窗口长度下，把它和第 16 章造整张掩码表的做法作比较。每一行依次是：窗口长度、造表做法的秒数、融合做法的秒数、两个答案的最大差别（类似 `5e-07`，也就是 0.0000005，只是舍入误差），以及这张表需要多少 MB。
    """)
    return


@app.cell
def _():
    rows = []
    for window in [256, 1024]:
        q_all = torch.rand(8, 4, window, 32)
        k_all = torch.rand(8, 4, window, 32)
        v_all = torch.rand(8, 4, window, 32)
        mask = torch.triu(torch.ones(window, window, dtype=torch.bool), diagonal=1)

        def table_way():
            scores = (q_all @ k_all.transpose(-2, -1) / np.sqrt(32)).masked_fill(mask, -torch.inf)
            return torch.softmax(scores, dim=-1) @ v_all

        def fused_way():
            return F.scaled_dot_product_attention(q_all, k_all, v_all, is_causal=True)

        difference = torch.max(torch.abs(table_way() - fused_way())).item()
        rows.append((window, round(time_it(table_way, 3), 4), round(time_it(fused_way, 3), 4), difference, 8 * 4 * window * window * 4 / 1e6))
    rows
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在测试机器上，融合版本在 256 时快了大约 7 倍，在 1,024 时快了大约 13 倍，答案在舍入误差内相同。窗口越长，提速越明显，因为这张表以及读写它的时间，都随窗口的*平方*增长，而融合做法从不造这张表。窗口加倍，表就变成四倍：这就是“随平方增长”的意思，也是长窗口昂贵的原因。

    ## 性能分析：找出慢在哪里

    在提速之前，先找出哪里慢。**性能分析器**（profiler）给一次运行里的每个操作计时。PyTorch 的叫 `torch.profiler`。下面的单元格对一个小 Transformer 块的一步训练做性能分析，它是 PyTorch 自己版本的第 17 章 `Block`，并列出最耗时的五个操作：
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    layer = torch.nn.TransformerEncoderLayer(128, 4, 512, dropout=0.0, batch_first=True)
    x_profile = torch.rand(32, 64, 128)
    with torch.profiler.profile() as profile:
        layer(x_profile).sum().backward()
    print(profile.key_averages().table(sort_by="cpu_time_total", row_limit=5))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这张表很密，只看名字和 `CPU total` 这一列就行。通常排在前面的是矩阵乘法（名字里带 `mm` 的，比如 `aten::mm` 和 `AddmmBackward0`）。如果排在前面的是别的，那就是首先要查的地方。一行的总时间包括它调用的操作，所以同一段时间可能出现在好几行里。（`print` 按原样显示文本，各列对齐，所以这里用它。）

    ## 动手算：读一张性能分析表

    你的表取决于你的电脑，所以这里给一张编出来的表用来练习。`aten::linear` 是一整个 `nn.Linear` 层，它会调用 `aten::addmm`，也就是带偏置的矩阵乘法。`aten::softmax` 这两个都不调用。

    | 名字 | CPU total |
    |---|---|
    | `aten::linear` | 9.1 ms |
    | `aten::addmm` | 8.0 ms |
    | `aten::softmax` | 1.2 ms |

    线性层在矩阵乘法之外花了多少毫秒？不重复计算的话，全部加起来花了多久？
    """)
    return


@app.cell(hide_code=True)
def _():
    profile_boxes = hand.boxes(['linear 在 addmm 之外的毫秒数', '总毫秒数'], step=0.1)
    return (profile_boxes,)


@app.cell(hide_code=True)
def _(profile_boxes):
    hand.marked(profile_boxes, {'linear 在 addmm 之外的毫秒数': 1.1, '总毫秒数': 10.3}, {
        'linear 在 addmm 之外的毫秒数': r"""`aten::linear` 的 9.1 里已经包含了 `aten::addmm` 的 8.0""",
        '总毫秒数': r"""把 `aten::linear` 和 `aten::softmax` 加起来，不要加 `aten::addmm`，它已经在第一个里面了""",
    }, LOCALE, tolerance=0.01)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    把三行都加起来会得到 18.3 ms，差不多是实际的两倍。而 10.3 ms 里有 8.0 ms 花在一个矩阵乘法上：值得去加快的正是这一行。

    ## 接下来

    一台电脑只能快到这个程度。**第 23 章**把训练分到几台设备上，这里是你电脑上的几个进程，并检查它们学到的和一个进程完全一样。

    ## 你现在会做的事

    - 公平地计时：先预热，再重复
    - 用手算和代码数出矩阵乘法和一次训练的 FLOP 数
    - 根据电脑的速度估计一次训练的时间和利用率
    - 数出用 Adam 训练需要的内存
    - 解释 float16、bfloat16 和混合精度，并测试它们在你的硬件上是否有用
    - 解释为什么注意力的分数表是瓶颈，亲手走一遍分块注意力，并把它写出来
    - 读懂性能分析结果，不把同一段时间算两次

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** FlashAttention 出自 Dao 等（2022）。`6N` 规则出自 *Scaling Laws for Neural Language Models*（Kaplan 等，2020），第 24 章会以它为基础。*算子融合*（kernel fusion）和 `torch.compile` 能把许多小操作合成一个，*激活检查点*（activation checkpointing）用重新计算换取激活值所占的内存。
    """)
    return


if __name__ == "__main__":
    app.run()
