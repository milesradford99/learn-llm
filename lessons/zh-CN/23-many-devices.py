import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="23 · 多台设备")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits
    from learn_llm.diagrams import ring
    from learn_llm.parallel import digit_model, train_in_processes, train_one_process

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 23 · 多台设备

    大模型通常要在成千上万块 GPU 上同时训练好几周才能得到。单靠一台设备根本办不到：不仅要算上几百年，模型本身也常常装不进单张卡的显存。这一章就来拆解，如何把训练任务分摊到多台设备上。

    你手头没有成千上万块 GPU，也完全不需要它们来理解背后的原理。你电脑的处理器有好几个**核心**（core），操作系统可以同时运行多个**进程**（process）——它们各自拥有独立的内存空间，互不干扰，这和彼此独立的多块 GPU 如出一辙。正因为内存互相隔离，一个进程没法直接读写另一个进程里的变量，它们必须通过通信来交换数据。学完这一章，你会在多个进程上同时训练一个网络，并亲眼验证它学到的结果和单个进程完全一致。

    要解决的问题有两个，对应的办法也不一样：

    - **太慢：** 把*数据*拆开，让每台设备各处理每一批里的一部分。
    - **太大：** 把*模型*拆开，让每台设备各存放一部分旋钮。

    ## 把数据分开

    回想一步训练做了什么：在一批数据上算出所有样本的**平均损失**，再求斜率，最后更新旋钮。现在让 4 台设备各持有一份完整的模型副本，再把这一批数据平分成四份，每台设备只负责计算*自己手里那一份*的平均损失和斜率。

    关键事实是这样的。只要四份样本数相同（比如整批 128 个样本，每份 32 个），把四份各自的均值再取平均，就正好等于整批的均值：每份均值是该份总和除以 32，而 $\frac{1}{4}\big(\frac{a}{32} + \frac{b}{32} + \frac{c}{32} + \frac{d}{32}\big) = \frac{a + b + c + d}{128}$，其中 $a, b, c, d$ 是四份各自的总和。斜率遵循损失的运算规则（第 02 章的加法和倍数规则），所以**把四台设备算出的斜率取平均，得到的正是整批数据的斜率**。随后每台设备都用这份相同的斜率去更新相同的旋钮，各份模型副本就能一步接一步地保持完全同步。

    这就是**数据并行**（data parallelism）。它唯一需要的通信，就是在每次反向传播之后，让每台设备都拿到所有斜率的平均值。

    ## 动手算：两半和整批

    一批 4 个样本，损失分别是 2、4、6 和 12。设备 A 拿前两个，设备 B 拿后两个。每台设备那一半的平均损失是多少？这两个均值的平均又是多少？整批的平均损失呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    mean_boxes = hand.boxes(['A 的均值', 'B 的均值', '两个均值的平均', '整批的均值'])
    return (mean_boxes,)


@app.cell(hide_code=True)
def _(mean_boxes):
    hand.marked(mean_boxes, {'A 的均值': 3, 'B 的均值': 9, '两个均值的平均': 6, '整批的均值': 6}, {
        'A 的均值': r"""$(2 + 4) / 2$""",
        'B 的均值': r"""$(6 + 12) / 2$""",
        '两个均值的平均': r"""把前两个答案加起来，再除以 2""",
        '整批的均值': r"""$(2 + 4 + 6 + 12) / 4$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    两个结果完全一样，因为两半样本数相同。假如 A 拿三个样本、B 只拿一个，两个均值再取平均就会变成 8 而不是 6——落单的那个样本被多算了三倍的权重。

    ## 练习 1：average_slopes

    写 `average_slopes(slopes)`。`slopes` 是一个两层列表：**外层**每项对应一台设备，**内层**是这台设备上各个旋钮的斜率张量——所以 `slopes[0][1]` 表示设备 0 上旋钮 1 的斜率。各台设备的旋钮完全一致，因此内层列表长度相同，对应位置的张量形状也相同。函数要返回一个张量列表，里面是每个旋钮在各台设备上的平均斜率。

    2 台设备、2 个旋钮时，输入和返回值的结构长这样（内层每一项都是张量）：

    ```python
    slopes = [
        [tensor([1., 2.]), tensor([[0.]])],  # 设备 0：[旋钮 0 的斜率, 旋钮 1 的斜率]
        [tensor([3., 6.]), tensor([[4.]])],  # 设备 1：[旋钮 0 的斜率, 旋钮 1 的斜率]
    ]
    # 返回：[tensor([2., 4.]), tensor([[2.]])]
    ```

    需要的工具你都学过：`torch.stack(tensors, dim=0)`（第 14 章）会**新开一个第 0 维**，把形状相同的张量摞在一起（区别于 `torch.cat` 在已有维度上直接拼接）；接着用 `torch.mean(..., dim=0)` 沿这个新维度取平均就行。
    """)
    return


@app.function
def average_slopes(slopes):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "average_slopes", average_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""循环要走的是旋钮，不是设备：`for k in range(len(slopes[0])):`。对旋钮 `k`，把每台设备的张量收进一个列表，把列表叠起来，再取平均。把结果加进一个列表，最后返回它。""",
        r"""一共 4 行。每台设备的旋钮 `k` 组成的列表是 `[device[k] for device in slopes]`。`torch.stack` 和 `torch.mean` 都要写 `dim=0`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    我们在一个进程里，用第 10 章的数字网络检查这个关键事实。一份副本算出 128 张图片一批的斜率。然后 4 台“设备”各取其中 32 张，我们把它们的斜率取平均。单元格显示两者的最大差别。（它需要练习 1，通过之前会显示错误。这是正常的，后面用到你的函数的单元格也一样。）
    """)
    return


@app.cell
def _():
    images, labels, _test_images, _test_labels = digits()
    X = torch.tensor(images[:128].reshape(128, 784) / 255, dtype=torch.float32)
    Y = torch.tensor(labels[:128], dtype=torch.long)

    def slopes_of(model, X_part, Y_part):
        model.zero_grad()
        F.cross_entropy(model(X_part), Y_part).backward()
        return [p.grad.clone() for p in model.parameters()]

    model = digit_model(0)
    whole = slopes_of(model, X, Y)
    quarters = [slopes_of(model, X[i * 32:(i + 1) * 32], Y[i * 32:(i + 1) * 32]) for i in range(4)]
    max([torch.max(torch.abs(a - b)).item() for a, b in zip(whole, average_slopes(quarters))])
    return X, Y, model, slopes_of


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    差别小于 $10^{-8}$（会显示成类似 `5.6e-09` 的数，意思是 $5.6 \times 10^{-9}$）：这只是浮点数舍入误差。把四份的结果平均起来，和整批一次算出的斜率完全一致。

    ## 梯度累积：在一台设备上用同样的技巧

    同样的性质也能帮单台显存装不下大批量的设备解围：把四份小批依次送进模型，让斜率在 `.grad` 里逐份累加。还记得第 10 章讲过的那个坑吗？如果不清空旧梯度 `.grad`，多次调用 `backward()` 算出的梯度会自动累加，而不会互相覆盖——在这里，我们正好利用这一点来做梯度累积！还差一步：直接把四个“四分之一批的均值”加在一起，总和会偏大四倍。所以每份的损失在反向传播*之前*先除以 4，这样累加起来的，就正好是四份均值的平均所对应的斜率。这就是**梯度累积**（gradient accumulation）：用多算几轮的时间，换取在任何设备上都能使用任意大批量。

    /// admonition | 🧰 工具箱：给切片起个名字
    `slice(a, b)` 可以把切片当作一个值存下来：写好 `part = slice(2, 5)` 后，`X[part]` 就等价于 `X[2:5]`。要从两个数组里截取同一段时（比如图片和对应的标签），这样写特别顺手。
    ///

    ## 练习 2：accumulated_slopes

    写 `accumulated_slopes(model, X, Y, pieces)`。`model` 是一个 PyTorch 模型，`X` 是一批输入，`Y` 是对应的标签，`pieces` 能整除批大小。函数要返回整批的 `F.cross_entropy` 损失对应的斜率（以列表形式返回每个旋钮 `.grad` 的副本），但每次只把其中一份送进模型，绝不同时处理多份：

    1. 用 `model.zero_grad()` 把模型的旧斜率清空一次。
    2. 依次处理每一份：取出 `X` 和 `Y` 里属于这一份的行，算出损失，除以 `pieces`，再反向传播。
    3. 返回 `[p.grad.clone() for p in model.parameters()]`：`.clone()` 会拷贝一份副本，避免被后续步骤改写。
    """)
    return


@app.function
def accumulated_slopes(model, X, Y, pieces):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "accumulated_slopes", accumulated_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""一份的大小是 `len(X) // pieces`。第 `i` 份从 `i * size` 开始，到 `(i + 1) * size` 之前结束：把这个切片存进一个名字，`X` 和 `Y` 都用它来取。""",
        r"""一共 6 行。在循环里，损失要先用括号括起来，再 `.backward()`：`(F.cross_entropy(model(X[part]), Y[part]) / pieces).backward()`。如果检查的结果差了 `pieces` 倍，就是漏了除法。如果 `zero_grad` 写在循环里面，就只有最后一份算数。""",
    ], LOCALE)
    return


@app.cell
def _(X, Y, model, slopes_of):
    one_go = slopes_of(model, X, Y)
    in_four = accumulated_slopes(model, X, Y, 4)
    max([torch.max(torch.abs(a - b)).item() for a, b in zip(one_go, in_four)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    同样只有舍入误差大小的差别。

    ## 通信：全归约

    在真实的设备上，“把斜率取平均”意味着设备之间要互相传递数字。完成这件事的标准操作叫**全归约**（all-reduce）：每台设备一开始手里只有自己的张量，结束时每台设备都能拿到所有张量的总和（或均值）。

    最容易想到的办法，是让所有设备把数据全发给某一台汇总，但那台设备很快就会堵成瓶颈。实际常用的做法是**环**（ring）：把设备排成一圈，每台只向下一位邻居发送数据：
    """)
    return


@app.cell(hide_code=True)
def _():
    ring(['设备 A', '设备 B', '设备 C'], '三台设备围成一圈。A 传给 B，B 传给 C，C 传给 A。')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每台设备都把自己的斜率均匀切成 $n$ 段（和设备数相同）。每次传递时，所有设备同时动手，各自把不同的一段发给下一位邻居，再把收到的那一段加到自己对应的段上。这样传 $n - 1$ 次（第一圈），每台设备手里就各自凑齐了某一段的完整总和。接下来的第二圈再传 $n - 1$ 次，把这些已经算好的段沿着环传一圈，每台设备就都拿齐全部结果了。

    下面来看 3 台设备的第一圈。假设每台设备的斜率各有 3 个数，正好每段 1 个数：A 一开始是 $[1, 2, 3]$，B 是 $[10, 20, 30]$，C 是 $[100, 200, 300]$。表中加粗的是该步刚更新的那一段：

    | 阶段 | 各设备同时发送 | A 手里的 3 段 | B 手里的 3 段 | C 手里的 3 段 |
    |---|---|---|---|---|
    | 初始 | — | $[1, 2, 3]$ | $[10, 20, 30]$ | $[100, 200, 300]$ |
    | 第 1 次传递 | A → B：第 0 段（$1$）；B → C：第 1 段（$20$）；C → A：第 2 段（$300$） | $[1, 2, \mathbf{3 + 300}]$ | $[\mathbf{10 + 1}, 20, 30]$ | $[100, \mathbf{200 + 20}, 300]$ |
    | 第 2 次传递 | A → B：刚加好的第 2 段；B → C：刚加好的第 0 段；C → A：刚加好的第 1 段 | $[1, \mathbf{2 + 200 + 20}, 3 + 300]$ | $[10 + 1, 20, \mathbf{30 + 3 + 300}]$ | $[\mathbf{100 + 10 + 1}, 200 + 20, 300]$ |

    ## 动手算：环上的一圈

    对照上面的表格算一算。第 1 次传递后，B 的第 0 段是多少？第 2 次传递后，C 的第 0 段、A 的第 1 段、B 的第 2 段各是多少？接下来的第二圈还要传 2 次，每台设备每次把一个已经算好的段往下传。四次传递加起来，每台设备一共发送了多少个数？
    """)
    return


@app.cell(hide_code=True)
def _():
    ring_boxes = hand.boxes(['第 1 次传递后 B 的第 0 段', '第 2 次传递后 C 的第 0 段', '第 2 次传递后 A 的第 1 段', '第 2 次传递后 B 的第 2 段', '每台设备发送的数的个数'])
    return (ring_boxes,)


@app.cell(hide_code=True)
def _(ring_boxes):
    hand.marked(ring_boxes, {'第 1 次传递后 B 的第 0 段': 11, '第 2 次传递后 C 的第 0 段': 111, '第 2 次传递后 A 的第 1 段': 222, '第 2 次传递后 B 的第 2 段': 333, '每台设备发送的数的个数': 4}, {
        '第 1 次传递后 B 的第 0 段': r"""B 自己的第 0 段是 10，A 传给它 1""",
        '第 2 次传递后 C 的第 0 段': r"""C 自己的第 0 段是 100，B 把刚算出的 11 传给它""",
        '第 2 次传递后 A 的第 1 段': r"""第 1 次传递后，C 的第 1 段是 $200 + 20$。第 2 次传递时 C 把它传给 A，而 A 自己的第 1 段是 2。""",
        '第 2 次传递后 B 的第 2 段': r"""第 1 次传递后，A 的第 2 段是 $3 + 300$。第 2 次传递时 A 把它传给 B，而 B 自己的第 2 段是 30。""",
        '每台设备发送的数的个数': r"""4 次传递，每次传一段，每段一个数""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第一圈结束后，完整的总和 111、222 和 333 就都算出来了，各自落在不同设备上。第二圈再把它们传给所有人。在 3 台设备之间同步 3 个数的斜率，每台设备一共只发了 4 个数。

    再算一算一般情况下的通信量。每次传递，每台设备只发一段，占全部斜率的 $1/n$。两圈一共 $2(n-1)$ 次传递，所以每台设备总共发送

    > $$\text{发送的字节数} = 2 \times \frac{n - 1}{n} \times \text{斜率的大小}$$

    代入手算的例子，正好是 $2 \times \tfrac{2}{3} \times 3 = 4$，和你数出来的一样。无论有多少台设备，$\frac{n-1}{n}$ 永远小于 1，每台设备的发送量都不会超过斜率大小的两倍。正因如此，哪怕不断增加设备数量，每台设备的通信负担也不会跟着膨胀，环形通信才能很好地扩展。

    ## 练习 3：ring_bytes

    写 `ring_bytes(size, devices)`，实现上面的公式：当斜率占 `size` 个字节时，环形全归约中每台设备需要发送多少字节。`ring_bytes(3, 3)` 返回 `4.0`，`ring_bytes(1000, 4)` 返回 `1500.0`。
    """)
    return


@app.function
def ring_bytes(size, devices):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "ring_bytes", ring_bytes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""只要一行。那个分数是设备数减一，除以设备数。""",
        r"""`devices - 1` 外面要加括号，否则 Python 只会拿 1 去除。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 真实的进程

    接下来动真格的。函数 `train_in_processes`（放在课程代码 `learn_llm/parallel.py` 里，以便新启动的进程能直接载入）会拉起多个独立进程。每个进程都用相同的随机种子搭建数字网络，因此所有副本一开始一模一样。每走一步，各进程取出这一批里属于自己的那一份，算出斜率，再调用 PyTorch 的 `dist.all_reduce` 把各进程的斜率求和、除以进程数，最后同步更新参数。这些进程完全不共享内存，只通过电脑内部的网络经由全归约交换数据——这和单机多卡之间通过高速互联通信如出一辙。

    按下按钮后，程序会先在单进程里训练 200 步（每批 128 张图片），再分别在 2 个和 4 个进程里重跑一次，并把最终的旋钮与单进程结果做对比。第一行是单进程的基准：进程数、耗时（秒）。后面每一行依次列出：进程数、所有旋钮中的最大偏差、每个进程交给全归约的斜率字节数，以及耗时（秒）。整个过程最多需要一分钟，期间页面可能暂时没有响应；结果可能会显示成一个可以点开查看的列表。
    """)
    return


@app.cell(hide_code=True)
def _():
    run_button = mo.ui.run_button(label="在多个进程上训练（最多一分钟）")
    run_button
    return (run_button,)


@app.cell
def _(run_button):
    mo.stop(not run_button.value, mo.md("*按下按钮开始运行。*"))
    started = time.perf_counter()
    alone = train_one_process(200, 128, 0.5, 0)
    runs = [(1, round(time.perf_counter() - started, 1))]
    for count in [2, 4]:
        knobs, sent, seconds = train_in_processes(count, 200, 128, 0.5, 0)
        biggest = max([torch.max(torch.abs(a - b)).item() for a, b in zip(alone, knobs)])
        runs.append((count, biggest, sent, round(seconds, 1)))
    runs
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    最大偏差只有 $10^{-7}$ 的几倍：在多个独立进程上训练 200 步后，得到的旋钮和单进程几乎完全一致，只差一点舍入误差。（加法顺序不同会影响浮点数的最后几位，所以不必指望偏差正好是 0。）数据并行改变的只是计算在*哪里*发生，不会改变模型学到*什么*。

    两种配置下，交给全归约的字节数都大约是 4,000 万：$200 \text{ 步} \times 50{,}890 \text{ 个斜率} \times 4 \text{ 字节}$。如果走环形全归约，2 个进程时每台大约发送 4,100 万字节，4 个进程时大约 6,100 万（在单台电脑上，PyTorch 也可能选用其他通信方式）。

    耗时一栏最能说明现实情况。在同一台电脑上，多个进程会争抢相同的 CPU 核心和内存，启动进程本身也要花时间，所以进程开得多未必更快；对于这么小的模型，通常反而更慢。只有当每台设备都是自带独立显存的 GPU，而且模型大到计算耗时远超通信耗时，数据并行带来的加速才会真正体现出来。

    ## 太大：把模型分开

    数据并行要求每台设备都存一份完整的副本。第 22 章算过，用 Adam 训练时每个旋钮要占 4 个数：旋钮本身、斜率、以及 Adam 的两个平均值，按 32 位算就是每个旋钮 16 字节。对于一个 70 亿旋钮的模型，每台设备光存这些就要 112 GB，远超大多数 GPU 的显存上限。

    可仔细想想每份副本是怎么处理 Adam 平均值的：每台设备每一步都在重复更新*全部*旋钮，算出来的结果一模一样，白白浪费了大量显存。**分片**（sharding）把优化器的平均值切分到各台设备上：设备 0 只保存前四分之一旋钮的平均值并只更新这部分，设备 1 负责第二份，依此类推。更新完一步后，各设备把改好的部分互相广播，让每份模型重新拼完整。同样的思路还能进一步给斜率分片，甚至给旋钮本身分片。这三个递进的级别就是 **ZeRO** 第 1、2、3 级。

    每台设备都得清楚哪些旋钮归自己管。设备从 0 开始编号，一台设备的编号叫它的 **rank**。当旋钮总数不能被设备数整除时，排在前面的设备各多领一个。

    /// admonition | 🧰 工具箱：求余数
    `a % b` 算出 `a` 除以 `b` 的**余数**（remainder），和整除 `//` 是一对：`10 // 4` 是 2，`10 % 4` 也是 2，因为 $10 = 4 \times 2 + 2$。`12 % 4` 则是 0。
    ///

    ## 动手算：10 个旋钮，4 台设备

    把编号 0 到 9 的 10 个旋钮按顺序分给 4 台设备。每台设备分到几个？再看设备 2：它拿到的第一个旋钮是几号？它的终点（也就是第一个不再归它管的旋钮编号，和 Python 切片规则一样）又是几号？
    """)
    return


@app.cell(hide_code=True)
def _():
    shard_boxes = hand.boxes(['设备 0 分到', '设备 1 分到', '设备 2 分到', '设备 3 分到', '设备 2 的起点', '设备 2 的终点'])
    return (shard_boxes,)


@app.cell(hide_code=True)
def _(shard_boxes):
    hand.marked(shard_boxes, {'设备 0 分到': 3, '设备 1 分到': 3, '设备 2 分到': 2, '设备 3 分到': 2, '设备 2 的起点': 6, '设备 2 的终点': 8}, {
        '设备 0 分到': r"""每台都分到 `10 // 4` 个，还剩下 `10 % 4` 个：前面的设备各多拿一个""",
        '设备 1 分到': r"""它是前两台设备之一，所以也多拿一个""",
        '设备 2 分到': r"""多出来的两个已经分完了，所以只有 `10 // 4` 个""",
        '设备 3 分到': r"""和设备 2 一样""",
        '设备 2 的起点': r"""排在它前面的设备 0 和设备 1 各拿了 3 个""",
        '设备 2 的终点': r"""它的起点加上它分到的个数""",
    }, LOCALE, per_row=4)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：shard

    写 `shard(count, devices, rank)`。它返回一对数：在从 0 编号的 `count` 个旋钮里，设备 `rank` 那一份的起点和终点。

    > `shard(10, 4, 2)` 返回 `(6, 8)`，`shard(10, 4, 0)` 返回 `(0, 3)`

    每台设备先分到 `count // devices` 个基本份，前 `count % devices` 台设备各多拿 1 个。回头看刚才“10 个旋钮分给 4 台设备”的例子：每台基本份是 2，余数是 2，四台设备的起点依次是 `0, 3, 6, 8`；减去各自的基本偏移 `rank * 2`（即 `0, 2, 4, 6`），差值正好是 `0, 1, 2, 2`——跟着 `rank` 递增，到了余数 2 就封顶，这正好是 `min(a, b)`（第 12 章）的用武之地。至于终点，如果当前设备也多拿了 1 个就再加 1，这里可以用 `a if test else b`（第 18 章）。
    """)
    return


@app.function
def shard(count, devices, rank):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "shard", shard, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先给两个数起名字：`size = count // devices` 和 `extra = count % devices`。排在设备 `rank` 前面的有 `rank` 台设备。其中有几台多拿了一个？`rank` 很小时是全部，但绝不会超过 `extra`：这就是 `min(rank, extra)`。""",
        r"""一共 5 行。起点是 `rank * size + min(rank, extra)`。终点是起点加上 `size`；如果这台设备属于前 `extra` 台，也就是 `rank < extra` 成立，就再加 1。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 分片能省多少？

    用 32 位数时，一个旋钮自身占 4 字节，它的斜率占 4 字节，Adam 的两个平均值占 8 字节。被分片的东西，每台设备只要付自己那一份，也就是总量除以设备数：

    | 级 | 旋钮 | 斜率 | Adam 的平均值 |
    |---|---|---|---|
    | 0，普通的数据并行 | 完整 | 完整 | 完整 |
    | 1 | 完整 | 完整 | 分片 |
    | 2 | 完整 | 分片 | 分片 |
    | 3 | 分片 | 分片 | 分片 |

    ## 动手算：四个级别

    取 1,000 个旋钮、4 台设备。完整存放时，旋钮占 4,000 字节，斜率占 4,000 字节，平均值占 8,000 字节。在每个级别，一台设备需要多少字节？
    """)
    return


@app.cell(hide_code=True)
def _():
    stage_boxes = hand.boxes(['第 0 级', '第 1 级', '第 2 级', '第 3 级'])
    return (stage_boxes,)


@app.cell(hide_code=True)
def _(stage_boxes):
    hand.marked(stage_boxes, {'第 0 级': 16000, '第 1 级': 10000, '第 2 级': 7000, '第 3 级': 4000}, {
        '第 0 级': r"""$4000 + 4000 + 8000$""",
        '第 1 级': r"""平均值缩小到 $8000 / 4$""",
        '第 2 级': r"""斜率也缩小到 $4000 / 4$""",
        '第 3 级': r"""三样都除以 4""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：bytes_per_device

    写 `bytes_per_device(knobs, devices, stage)`：一个有 `knobs` 个旋钮的模型分给 `devices` 台设备，在表里的第 0、1、2 或 3 级，一台设备需要多少字节。你手算的答案就是检查：`bytes_per_device(1000, 4, 2)` 得到 `7000.0`。

    注意，每一级都包含前一级分片的所有东西，再多分一样。`a >= b` 的意思是“a 至少是 b”。

    检查之后的表格用你的函数算 70 亿个旋钮、8 台设备的情况；70 亿写成 `7e9`，和第 22 章一样。
    """)
    return


@app.function
def bytes_per_device(knobs, devices, stage):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "bytes_per_device", bytes_per_device, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先把三样东西完整的大小存进三个名字。然后是三个各自独立的 `if`，表格的每一列对应一个，每个都把一样东西除以 `devices`。最后把三者加起来。""",
        r"""这几个 `if` 彼此对齐，不是一个套一个；它们的条件分别是 `stage >= 1`、`stage >= 2` 和 `stage >= 3`。在第 3 级，三个条件都成立，所以三样东西都被除。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    table = ["| 级 | 每台设备的 GB 数，70 亿个旋钮、8 台设备 |", "|---|---|"]
    for stage in range(4):
        table.append(f"| {stage} | {bytes_per_device(7e9, 8, stage) / 1e9:.0f} |")
    mo.md("\n".join(table))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每台设备从 112 GB 降到 14 GB（第 2 级其实是 38.5 GB，这里四舍五入了），靠的仅仅是不再把同一份数据存八遍。除此之外还要算上激活值（为反向传播保留的中间结果），它随批大小增长，而不是随旋钮数增长。

    分片也不是零代价的。在第 3 级，每一步跑到某一层之前，都得临时从其他设备把这一层的旋钮汇集过来——相当于用更多通信换取更低的显存占用。

    ## 把一层分开

    面对最大的模型，连单独一层的权重矩阵都要切开。回想一下，在 `X @ W` 里，结果的每一列都只依赖 `W` 对应的那一列。看一个极简的 $2 \times 2$ 例子：

    > $$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\ 3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

    结果的左列（$19, 43$）只用到了 `W` 的左列（$5, 7$），右列（$22, 50$）只用到了 `W` 的右列（$6, 8$）。因此可以把 `W` 按列切开：两台设备时，设备 0 拿左半边算出结果的左半边，设备 1 拿右半边算出结果的右半边，最后把两半并排拼起来。这就是**张量并行**（tensor parallelism）。

    ## 练习 6：split_product

    写 `split_product(X, W, devices)`。`X` 的形状是 `(n, k)`，`W` 的形状是 `(k, m)`。函数要返回 `X @ W`（形状为 `(n, m)`），但必须拆成 `devices` 块来算：每台设备用 `X` 乘以 `W` 里分给它的那几列，再把各块按顺序拼成完整结果。

    哪几列归哪台设备，交给刚才写好的 `shard` 决定：要切分的总列数是 `W.shape[1]`。`W[:, start:stop]` 用来取出这几列（第 14 章里 `:` 表示保留所有行），`torch.cat(pieces, dim=1)`（第 14 章）则把多块矩阵沿列方向并排拼起来。

    检查之后的单元格会在随机矩阵上，把拆给 3 台设备算出的结果和直接做矩阵乘法的结果做对比。
    """)
    return


@app.function
def split_product(X, W, devices):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("23", "split_product", split_product, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""先建一个空列表，再遍历各个 rank：`for rank in range(devices):`。每一轮用 `shard` 算出这台设备的起点和终点，做乘法，把这一块放进列表。循环结束后，把各块拼起来。""",
        r"""一共 6 行。`shard` 返回一对数，正好拆成两个名字：`start, stop = shard(W.shape[1], devices, rank)`。这一块的结果是 `X @ W[:, start:stop]`。""",
    ], LOCALE)
    return


@app.cell
def _():
    torch.manual_seed(0)
    X_demo = torch.rand(5, 8)
    W_demo = torch.rand(8, 7)
    torch.max(torch.abs(split_product(X_demo, W_demo, 3) - X_demo @ W_demo)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    偏差为 0，或者小到 $10^{-7}$ 左右：同样的数字只是相加顺序可能不同，最多在最后几位略有出入。真实的训练系统会把这些策略组合使用：多组机器之间做数据并行，组内做分片，同一台机器内互联最快的几块 GPU 之间做张量并行。如何搭配这些并行方式，正是大规模训练工程的核心课题之一。

    ## 接下来

    现在你已经知道训练一个模型要花多少资源，以及如何把任务拆分到多台设备上。**第 24 章**会把这些知识落到规划上：手头有一笔固定的算力预算，模型该设多大，又该喂多少数据？答案遵循一条简洁的规律，你会亲手从自己的小规模实验里测出它。

    ## 你现在会做的事

    - 解释为什么把等大分片的斜率取平均，得到的就是整批的斜率
    - 用梯度累积在单台设备上模拟任意大的批量
    - 讲清全归约的作用，手算走一遍环形全归约，并计算通信字节数
    - 在多个真实进程上并行训练，并验证结果与单进程一致
    - 把旋钮均匀分片到多台设备，并算出各分片级别下每台设备的显存占用
    - 把一次矩阵乘法按列拆分到多台设备上计算

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** PyTorch 的 `DistributedDataParallel` 做的正是我们这些进程所做的事，还会把通信和反向传播重叠起来以隐藏等待时间。ZeRO 出自 Rajbhandari 等（2019），*FSDP* 是 PyTorch 对第 3 级的实现。张量并行见 *Megatron-LM* 论文；而把模型的不同层分组放到不同设备上依次接力计算，则叫*流水线并行*（pipeline parallelism）。
    """)
    return


if __name__ == "__main__":
    app.run()
