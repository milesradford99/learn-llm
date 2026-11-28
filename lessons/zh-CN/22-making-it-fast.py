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

    训练一个模型要做海量算术。想让它跑得更快，第一步不是找巧妙的偏方，而是先**测量**——看清时间到底花在哪儿；再**估算**——算算理论上最快能跑多快。这一章先做这两件事，再试两种提速手段，你还会看到其中一种在没有配套硬件时反而会变慢（在测试这一章的电脑上就是如此）。

    ## 测量时间

    测代码耗时，只跑一次很不靠谱：第一次运行往往偏慢（电脑还在做准备），而且每次运行本来就会有点波动。所以标准的做法是先跑一次预热，再计时跑几次，取平均值。

    /// admonition | 🧰 工具箱：时钟
    `time.perf_counter()` 返回以秒为单位的时间（起点本身没有意义）。单看一次读数没用，但用结束时的读数减去开始时的读数，就是中间那段代码花掉的时间：先写 `started = time.perf_counter()`，跑完任务，再算 `time.perf_counter() - started`。
    ///

    ## 练习 1：time_it

    写 `time_it(f, repeats)`。`f` 是一个没有输入的函数，`repeats` 是一个整数。它先调用一次 `f` 来预热（这一次不计时），然后再调用 `repeats` 次，返回这 `repeats` 次调用平均每次花了多少秒。

    其中一项检查会传入这样一个函数：第一次调用要 0.1 秒，之后每次 0.01 秒。你的答案必须是 0.01。
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
        r"""分四步：先调用一次 `f()` 预热，再读一次时钟，接着循环调用 `f()`，最后 `return`。时钟一定要在预热之后再读，这样偏慢的第一次才不会被算进去。""",
        r"""一共 5 行。最后一行返回从 `started` 到现在经过的时间，再除以 `repeats`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 算一算计算量

    神经网络里几乎所有的计算都是矩阵乘法。`A @ B` 中，`A` 的形状是 `(n, k)`，`B` 的形状是 `(k, m)`，结果里的每个数都是一个长度为 $k$ 的点积：$k$ 次乘法，大约 $k$ 次加法。两个这样的数做一次乘法或一次加法，就算一次**浮点运算**（floating-point operation），简称 **FLOP**（“浮点数”就是计算机里带小数点的数，也就是我们一直在用的小数）。

    ## 动手算：一次小矩阵乘法

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
    推广到一般情况，结果矩阵里有 $n \times m$ 个数，每个数都要做 $2k$ 次运算：

    > $$\text{一次矩阵乘法的 FLOP 数} \approx 2 \times n \times k \times m$$

    ## 练习 2：matmul_flops

    写 `matmul_flops(n, k, m)`，也就是这个公式：形状为 `(n, k)` 的矩阵乘以形状为 `(k, m)` 的矩阵要多少个 FLOP。你手算的答案就是检查用例：`matmul_flops(2, 3, 4)` 得到 `48`。

    检查后面的单元格会用它来测你的电脑：给两个 $1024 \times 1024$ 矩阵的乘法计时，再用 FLOP 数除以秒数。十亿个 FLOP 叫一个 **gigaFLOP**。这一章会反复遇到百万、十亿、万亿这样的大单位，先留一张速查表在手边：

    | 数量级 | 中文 | 常用单位 |
    |---|---|---|
    | $10^6$ | 百万 | **mega**（MB = 百万字节） |
    | $10^9$ | 十亿 | **giga**（gigaFLOP、GB = 十亿字节） |
    | $10^{12}$ | 万亿 | **tera**（teraFLOP） |

    （下面的单元格依赖练习 1 和练习 2，在它们通过之前会报错，这是正常的，后面用到你写的函数的单元格也一样。）
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
    这就是你电脑能跑出的峰值速度，单位是每秒 gigaFLOP（十亿次浮点运算）。你的数字会和别人的不一样。（`product` 是一个没有输入的函数，所以 `time_it` 可以一遍遍调用它。）在测试这一章的机器上，它大约是 180；一块现代 GPU 大约快 1,000 倍，而最大规模的训练会同时用上成千上万块这样的 GPU。

    ## 训练的计算量

    对一个有 $N$ 个旋钮的模型，下面这条估算规则足够用来做计划。在正向计算中，每个旋钮对每个词元大约参与一次乘法和一次加法：2 个 FLOP。反向计算大约是它的两倍：一次算权重梯度、一次算输入梯度（也就是第 08 章那两次矩阵乘法，一次求旋钮的斜率，一次把责任传回上一层）。这又是 4 个 FLOP。

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

    这里没有算注意力分数本身，那部分只在窗口很长时才占大头。

    ## 练习 3：training_flops

    写 `training_flops(knobs, tokens)`，也就是这条规则。`tokens` 是整次训练里读过的词元总数，读几遍就算几遍。`training_flops(1000, 10)` 得到 `60000`。
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

    电脑实际训练时很难一直跑满峰值速度。实际速度占峰值速度的比例，叫作这次训练的**利用率**（utilisation）：0.5 就是跑了一半速度。所以：

    > $$\text{秒数} = \frac{\text{FLOP 数}}{\text{速度} \times \text{利用率}}$$

    /// admonition | 🧰 工具箱：大数的科学计数法简写
    在 Python 里，`30e12` 的意思是 $30 \times 10^{12}$：`e` 后面的数字表示小数点往右移几位，所以 `7e9` 就是 70 亿。带负号就是往左移，比如第 09 章的 `1e-8`。
    ///

    ## 动手算：要多久？

    一次训练需要 30 teraFLOP，也就是 $30 \times 10^{12}$ 个 FLOP（在 Python 里写作 `30e12`）；电脑的峰值速度是每秒 100 gigaFLOP，也就是 $100 \times 10^9$（写作 `100e9`）。全速运行要多少秒？利用率为 0.5 时呢？
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

    写 `run_seconds(flops, speed, utilisation)`，也就是上面的公式。`flops` 是一次训练的计算量，`speed` 是电脑的峰值速度（每秒多少个 FLOP），`utilisation` 是 0 到 1 之间的比例。`run_seconds(30e12, 100e9, 0.5)` 得到 `600.0`。

    第 21 章的训练用大约 117 万个旋钮，把大约 200 万个词元读了两遍。检查后面的单元格会算出它的计算量（单位是 teraFLOP，即 $10^{12}$ 个 FLOP），以及按你电脑的峰值速度要跑多少秒。
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
        r"""只要一行：一个除法。分母是两项相乘，记得给分母加括号。""",
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
    大约 28 teraFLOP，全速运行只要两三分钟。拿第二个数字（理论最快秒数）和第 21 章实际训练花的时间比一比：用前者除以后者，就能心算出你那次训练的利用率。在测试机器上，实际训练用了四五分钟，大约跑到了峰值速度的一半。剩下的时间花在小操作上——比如 softmax、归一化和在内存里搬运数字，它们算术量很少，却实打实占时间。对一个 70 亿旋钮、一万亿词元的模型，同一个公式给出 $4.2 \times 10^{22}$ 个 FLOP，需要大约一千块 GPU（每块每秒约 200 万亿个 FLOP），按一半的利用率跑上一周左右。

    ## 算一算内存占用

    用 Adam 训练时，每个旋钮需要存四个数：它自己、它的斜率，以及 Adam 的两个滑动平均值。用 32 位浮点数时，每个数占 4 个字节。除此之外还有**激活值**（activations）：正向计算为了给反向计算备用而留下来的所有中间结果，它们会随着批大小和窗口长度一起增长。

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
        '一百万个旋钮的 MB 数': r"""一百万乘以第一个答案，再换算成以百万字节为单位""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：adam_bytes

    写 `adam_bytes(knobs)`：用 Adam 训练、数字为 32 位时，一个模型的旋钮、斜率和平均值一共需要多少个字节。`adam_bytes(1000000)` 得到 `16000000`。
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
    对第 20 章的模型（117 万个旋钮），这不到 19 MB。但对一个 70 亿旋钮的模型，光是这四份数就要 112 GB，大多数单块 GPU 都装不下。第 23 章讲的就是这时该怎么办。

    ## 提速 1：用更省位的数字格式

    32 位浮点数大约能保留 7 位有效数字（第 10 章）。两种 16 位格式能省下一半内存，在专门支持它们的硬件上，速度还能快一倍以上：

    - **float16** 大约保留 3 位有效数字，但装不下超过 65,504 的数；
    - **bfloat16** 保留的有效数字更少，大约 2 到 3 位，但能表示的范围和 float32 一样大。

    /// admonition | 🧰 工具箱：查看数字格式的范围和精度
    `torch.finfo(dtype)` 用来查看一种浮点数格式：`max` 是它能表示的最大数，`eps` 是 1 和它能表示的下一个数之间的间隔，用来衡量精度。`eps` 约为 0.001 就是千分之一，大约 3 位有效数字。
    ///

    ## 动手算：装得下吗？

    运行下面的单元格之前，先预测一下。在 float16 里，超过 65,504 的数都装不下：$200 \times 200$ 装得下吗？$300 \times 300$ 呢？装得下填 1，装不下填 0。另外，一百万个 16 位的数占多少 MB？
    """)
    return


@app.cell(hide_code=True)
def _():
    fit_boxes = hand.boxes(['200 × 200 装得下', '300 × 300 装得下', '一百万个数的 MB 数'])
    return (fit_boxes,)


@app.cell(hide_code=True)
def _(fit_boxes):
    hand.marked(fit_boxes, {'200 × 200 装得下': 1, '300 × 300 装得下': 0, '一百万个数的 MB 数': 2}, {
        '200 × 200 装得下': r"""$200 \times 200 = 40{,}000$。它超过 65,504 吗？""",
        '300 × 300 装得下': r"""$300 \times 300 = 90{,}000$。它超过 65,504 吗？""",
        '一百万个数的 MB 数': r"""16 位是 2 个字节""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的单元格打印出每种格式的 `max` 和 `eps`，再看看 70,000 放进 float16、以及 $1 + 0.001$ 放进 bfloat16 会发生什么：
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
    70,000 在 float16 里溢出成了 `inf`（无穷大）：太大装不下了。而在 bfloat16 里，1 + 0.001 算完还是 1：变化太小，根本记不下来。全程只用 16 位训练会把细小的参数更新全丢掉，所以实际训练用的是**混合精度**（mixed precision）：大块头的矩阵乘法用 16 位算，而旋钮本身、Adam 的平均值和需要精度的求和仍保持 32 位。

    它能让*你的*电脑变快吗？只有处理器带 16 位专用计算硬件时才行。下面的单元格用两种格式各测一次同样的矩阵乘法，结果是每秒 gigaFLOP，越大越快（可能需要等几秒）。
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
    在测试机器上，bfloat16 反而比 float32 **慢了 100 多倍**，因为它的处理器没有 16 位计算硬件，只能一步步用软件模拟。你的结果可能不一样：较新的处理器和 Apple 芯片就带这种硬件。这正是这一节的要点：提速手段总是和具体硬件绑定的，有没有用，测了才知道。在 GPU 上，混合精度是绝对的标配。

    ## 提速 2：省掉庞大的中间分数表

    注意力机制会生成一张巨大的分数表，每对词之间都有一个分数：窗口长度 1,024、4 个注意力头、批大小 8 时，这张表大约有 3,350 万个数，占 134 MB，而且要反复写进主内存再读出来（做掩码、softmax、矩阵乘法）。这时拖慢速度的瓶颈不再是计算，而是内存读写：处理器算得飞快，从主内存搬运数据却慢得多。处理器旁边只有一小块高速内存，叫作**缓存**（cache）——要想真正提速，就得尽量让数据待在缓存里算完，少跑主内存。

    **分块注意力**（tiled attention）从不把整张分数表造出来。对每个查询，它沿着键一小块一小块地推进，只维护当前的累计结果。难点在 softmax 上：它需要所有分数的最大值和总和，而这要读完最后一块才知道。办法是把两者都当成累计值保存下来，遇到更大的最大值时把之前的累计值按比例重缩放：

    - 维护三个累计值：到目前为止的最大分数 `best`、各项 $e^{\text{score} - \text{best}}$ 的累计和 `total`，以及值的累计加权和 `acc`；
    - 新的分块带来更大的最大值时，之前累计的每一项都是按旧的 `best` 算的，只要把 `total` 和 `acc` 一起乘上**缩放系数** $e^{\text{旧 best} - \text{新 best}}$ 就行。为什么乘这一个系数就能把前面所有分数 $s$ 一次性换算过来？一行初中指数乘法就能看透：

    > $$e^{s - \text{旧 best}} \times e^{\text{旧 best} - \text{新 best}} = e^{s - \text{新 best}}$$

    - 全部走完后，`acc / total` 就是精确的 softmax 加权平均。

    开头把 `best` 设为 $-\infty$（负无穷大），`total` 和 `acc` 设为 0。这样第 1 块进来时，任何分数都比 $-\infty$ 大，而且缩放系数是 $e^{-\infty - \text{新 best}} = e^{-\infty} = 0$，乘在空的累计值上正好是 0——自动忽略空白起点，让第 1 块和后面的分块走完全一样的公式。

    这就是 **FlashAttention** 的核心思路。

    先看一个小例子：每个向量 1 个数，分数是 $[1, 3, 2]$，值是 $[10, 20, 30]$，每个分块 1 个。

    | 分块 | `best` | 缩放系数 | `total` | `acc` |
    |---|---|---|---|---|
    | 1 | 1 | $e^{-\infty} = 0$ | $0 \times 0 + e^{1 - 1} = 1$ | $0 \times 0 + 1 \times 10 = 10$ |
    | 2 | 3 | $e^{1 - 3} \approx 0.135$ | $1 \times 0.135 + 1 = 1.135$ | $10 \times 0.135 + 20 = 21.35$ |
    | 3 | 3 | $e^{3 - 3} = 1$ | $1.135 + e^{2 - 3} = 1.503$ | $21.35 + 0.368 \times 30 = 32.39$ |

    最后 $32.39 / 1.503 \approx 21.55$，和一次对三个分数做 softmax 的结果完全一致。第 1 块因为 `best` 从 $-\infty$ 起步，缩放系数是 0；第 2 块最大值变大了，之前累计的结果就被按比例缩小；第 3 块最大值没变，缩放系数就是 1。

    ## 动手算：四个分数，两个分块

    一个查询，每个向量 1 个数，所以分数就是键乘以查询；我们取查询为 1，$\sqrt{d}$ 也是 1，略去不写。分数是 $[0, 1, 2, 0]$，值是 $[10, 20, 30, 40]$，每个分块 2 个：先算分数 $[0, 1]$ 和值 $[10, 20]$，再算分数 $[2, 0]$ 和值 $[30, 40]$。你会用到 $e^0 = 1$、$e^{-1} = 0.368$ 和 $e^{-2} = 0.135$。

    对每个分块：新的 `best` 是到目前为止最大的分数；每个权重是 $e^{\text{score} - \text{best}}$。之前累计的 `total` 和 `acc` 都要先乘上缩放系数 $e^{\text{旧 best} - \text{新 best}}$，再把当前分块的权重加进 `total`，把权重乘以值的结果加进 `acc`。第 1 块开始时 `best` 是 $-\infty$，缩放系数为 0，直接把当前分块加进去就行。
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
        '第 2 块之后的 total': r"""原来的 `total` 乘以缩放系数 $e^{1 - 2}$，再加上新的权重 $e^{2 - 2}$ 和 $e^{0 - 2}$""",
        '第 2 块之后的 acc': r"""原来的 `acc` 乘以缩放系数 $e^{1 - 2}$，再加上 $1 \times 30 + 0.135 \times 40$""",
        '答案': r"""`acc` 除以 `total`""",
    }, LOCALE, tolerance=0.05, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一次对四个分数做 softmax，得到的也是 26.93：这两个分块从来不需要同时待在内存里。

    ## 练习 6：tiled_attention

    写 `tiled_attention(q, K, V, tile)`。输入是一个形状为 `(d,)` 的查询 `q`，形状为 `(n, d)` 的键 `K`，以及形状为 `(n, v)` 的值 `V`；不用掩码，分数和第 16 章一样除以 $\sqrt{d}$。`tile` 是每次取多少个键。函数返回一个形状为 `(v,)` 的张量，结果和 `torch.softmax(K @ q / np.sqrt(d), dim=0) @ V` 相同，但手里同时保存的分数从来不超过 `tile` 个。

    先初始化三个累计值：`best` 设为 `-np.inf`（这样任何分数都比它大，而且第 1 块的缩放系数 $e^{-\infty} = 0$ 会自动忽略空的累计值），`total` 设为 `0.0`，`acc` 设为形状为 `(v,)` 的全零张量。遍历分块的循环直接给你（按步长切片是第 09 章的工具）：

    ```
    for start in range(0, len(K), tile):
    ```

    循环里面，取出当前分块的键 `K_tile = K[start:start + tile]`（形状 `(tile, d)`）和值 `V_tile = V[start:start + tile]`（形状 `(tile, v)`，最后一块的行数可能少于 `tile`）。照着你手算的步骤往下写，顺便对照每一步的张量形状：

    | 步骤 | 数学表达 | 形状 |
    |---|---|---|
    | 当前分块的 `scores` | $\text{K\_tile} \cdot q / \sqrt{d}$ | `scores`: `(tile,)` |
    | 新的最大值 | `best` 和当前分块最大分数里较大的那个 | 普通标量 |
    | 缩放系数 | $e^{\text{best} - \text{新的最大值}}$ | 普通标量 |
    | 当前分块的 `weights` | $e^{\text{scores} - \text{新的最大值}}$ | `weights`: `(tile,)` |
    | 更新 `total` | $\text{total} \times \text{缩放系数} + \text{权重之和}$ | 普通标量 |
    | 更新 `acc` | $\text{acc} \times \text{缩放系数} + \text{weights} \cdot \text{V\_tile}$ | `weights @ V_tile`: `(v,)`，`acc`: `(v,)` |
    | 更新 `best` | 设为新的最大值 | 普通标量 |

    循环结束后，返回 `acc / total`。`torch.max(scores).item()` 和 `torch.sum(weights).item()` 会返回普通 Python 数字；对张量求指数用 `torch.exp`，对普通数字用 `np.exp`。
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
        r"""循环里面七行，正好对应表格的七行，顺序不要乱。`best` 必须最后更新，因为缩放系数同时需要旧的 `best` 和新的最大值。两次乘积都用 `@`：`K_tile @ q` 的形状是 `(tile,)`，`weights @ V_tile` 的形状是 `(v,)`。""",
        r"""循环里的前两行是 `scores = K[start:start + tile] @ q / np.sqrt(len(q))` 和 `new_best = max(best, torch.max(scores).item())`。如果分数很大的那项检查没通过，说明你算 `torch.exp` 之前没有先减去新的最大值。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在 Python 里用循环一次处理一个查询当然很慢，这里的重点是理解*算法思路*。在真实系统里，这套分块步骤会被编译成一个底层的融合函数，让每个分块始终留在缓存里算完。PyTorch 就内置了一个（把多步合成一个底层函数叫“融合”）：`F.scaled_dot_product_attention`。下面的单元格在两种窗口长度下，把它和第 16 章造整张掩码表的做法作比较。每一行依次是：窗口长度、造表做法的秒数、融合做法的秒数、两种结果的最大差别（类似 `5e-07`，也就是 0.0000005，只是浮点舍入误差），以及分数表要占多少 MB。
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
    在测试机器上，融合版本在窗口 256 时快了约 7 倍，在 1,024 时快了约 13 倍，结果在舍入误差内完全一致。窗口越长，提速越明显：因为分数表的大小和读写时间都随窗口长度的*平方*增长，而融合做法压根不造这张表。窗口加倍，表格大小就翻成四倍——这就是“随平方增长”的含义，也是长窗口如此昂贵的原因。

    ## 性能分析：找出慢在哪里

    动手优化之前，先定位瓶颈。**性能分析器**（profiler）会给一次运行里的每个操作分别计时，PyTorch 自带的工具叫 `torch.profiler`。下面的单元格对一个小 Transformer 块的一步训练做性能分析（它是 PyTorch 官方版本的第 17 章 `Block`），列出最耗时的五个操作：
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
    这张表信息很密，先只看操作名和 `CPU total` 这一列就行。排在最前面的通常是矩阵乘法（名字里带 `mm` 的，比如 `aten::mm` 和 `AddmmBackward0`）。如果排在前面的是别的操作，那就是最该优先排查的地方。注意，每一行的 `CPU total` 都**包含了它内部调用的子操作的时间**，所以同一段时间可能会同时算在好几行里。（这里用 `print` 是为了保持各列对齐、原样输出文本。）

    ## 动手算：读一张性能分析表

    每个人的实测表会因电脑不同而异，这里用一张模拟表来练习。`aten::linear` 是整个 `nn.Linear` 层，它内部会调用 `aten::addmm`（带偏置的矩阵乘法）；`aten::softmax` 和这两个都没关系。

    | 名字 | CPU total |
    |---|---|
    | `aten::linear` | 9.1 ms |
    | `aten::addmm` | 8.0 ms |
    | `aten::softmax` | 1.2 ms |

    线性层在矩阵乘法之外花了多少毫秒？不重复计算的话，全部加起来一共花了多久？
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
        '总毫秒数': r"""把 `aten::linear` 和 `aten::softmax` 加起来，不要再加 `aten::addmm`，它已经算在第一行里了""",
    }, LOCALE, tolerance=0.01)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    直接把三行加起来会得到 18.3 ms，几乎是真实耗时的两倍。而在真实的 10.3 ms 里，有 8.0 ms 都花在一个矩阵乘法上——最值得去优化的正是这一行。

    ## 接下来

    单台电脑的算力终归有上限。**第 23 章**会把训练分摊到多台设备上（在你的电脑上用多个进程来模拟），并验证它们学出的模型和单进程训练完全一致。

    ## 你现在会做的事

    - 公平地给代码计时：先预热，再多次重复取平均
    - 用手算和代码估算矩阵乘法与整次训练的 FLOP 数
    - 根据电脑峰值速度估算训练耗时和硬件利用率
    - 算出用 Adam 训练模型所需的内存占用
    - 说清 float16、bfloat16 和混合精度的区别，并实测它们在自己硬件上的效果
    - 说清为什么注意力的分数表会成为瓶颈，亲手推演并写出分块注意力
    - 读懂性能分析表，不把同一段耗时重复计算

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** FlashAttention 出自 Dao 等（2022）。`6N` 规则出自 *Scaling Laws for Neural Language Models*（Kaplan 等，2020），第 24 章会在此基础上展开。*算子融合*（kernel fusion）和 `torch.compile` 能把许多小操作合成一个底层函数，*激活检查点*（activation checkpointing）则用重新计算来换取激活值所占的内存。
    """)
    return


if __name__ == "__main__":
    app.run()
