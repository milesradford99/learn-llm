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

    大模型是在成千上万块 GPU 上同时训练好几周得到的。任何一台设备都做不到：那得花上几百年，而且模型往往根本装不进一台设备的内存。这一章讲怎样把工作分开。

    你没有成千上万块 GPU，也不需要它们来理解这些想法。你电脑的处理器有好几个**核心**（core），操作系统可以同时运行好几个**进程**（process），也就是不共享内存、各自独立运行的程序，这和分开的 GPU 完全一样。学完这一章，你会在几个进程上同时训练一个网络，并检查它学到的和一个进程完全一样。

    要解决的问题有两个，答案也不同：

    - **太慢：** 把*数据*分开，让每台设备做每批的一部分。
    - **太大：** 把*模型*分开，让每台设备拿一部分旋钮。

    ## 把数据分开

    回想一步训练做了什么：在一批数据上算损失，它是所有样本的**均值**；然后算斜率；然后移动旋钮。现在让 4 台设备各拿一份完整的模型副本，各拿四分之一批。每台算出*它那一份*的均值的斜率。

    关键事实是这样的。当四份大小相同时，四份各自的均值再取平均，就是整批的均值（整批 128 个样本，每份 32 个）：每份的均值是那一份的总和除以 32，而 $\frac{1}{4}\big(\frac{a}{32} + \frac{b}{32} + \frac{c}{32} + \frac{d}{32}\big) = \frac{a + b + c + d}{128}$，其中 $a, b, c, d$ 是四份各自的总和。斜率跟着损失走（第 02 章的加法和倍数规则），所以**把四台设备的斜率取平均，正好得到整批的斜率**。如果每台设备接着对同样的旋钮走同样的一步，各份副本就一步接一步地保持一致。

    这就是**数据并行**（data parallelism）。唯一需要的通信是：每次反向计算之后，每台设备都必须得到所有斜率的平均值。

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
    数字相同，因为两半一样大。假如 A 拿三个样本、B 只拿一个，两个均值的平均就是 8，而不是 6：落单的那个样本被多算了三倍的分量。

    ## 练习 1：average_slopes

    写 `average_slopes(slopes)`。`slopes` 是一个列表，每台设备一项；每一项又是一个张量列表，每个旋钮一个：`slopes[0][1]` 是设备 0 上旋钮 1 的斜率。各台设备的旋钮都一样，所以里层的列表一样长，对应的张量形状也相同。它返回一个张量列表：每个旋钮在各台设备上的均值。

    2 台设备、2 个旋钮时，把 `[[[1, 2], [[0]]], [[3, 6], [[4]]]]` 当作张量，得到 `[[2, 4], [[2]]]`。

    工具你都有：`torch.stack(tensors, dim=0)`（第 14 章）沿着一个新的第 0 维把形状相同的张量叠起来，`torch.mean(..., dim=0)` 沿这一维取平均。
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
    差别小于 $10^{-8}$（显示为类似 `5.6e-09` 的数，意思是 $5.6 \times 10^{-9}$）：只是舍入误差。四份合起来*就是*整批。

    ## 梯度累积：在一台设备上用同样的技巧

    同样的事实也能帮助一台装不下大批量的设备：把四份一份接一份地运行，让斜率在 `.grad` 里累加。PyTorch 在那里会自己累加，还记得第 10 章那个坑吗：这一次，那个坑正是我们想要的。还剩一件事：把四个“四分之一批的均值”加起来，结果大了四倍。所以每一份的损失在做反向计算*之前*先除以 4，这样累积起来的，就是四份均值的平均所对应的斜率。这叫**梯度累积**（gradient accumulation），它让任何设备都能用任何批大小，代价是时间。

    /// admonition | 🧰 工具箱：有名字的切片
    `slice(a, b)` 是一个当作值保存起来的切片：有了 `part = slice(2, 5)`，`X[part]` 就是 `X[2:5]`。要从两个数组里取同一段时（比如图片和它们的标签），它很方便。
    ///

    ## 练习 2：accumulated_slopes

    写 `accumulated_slopes(model, X, Y, pieces)`。`model` 是一个 PyTorch 模型，`X` 是一批输入，`Y` 是它们的标签，`pieces` 能整除批大小。它返回整批的 `F.cross_entropy` 损失对应的斜率，形式是一个列表，里面是每个旋钮 `.grad` 的副本；但模型每次只处理其中一份，从不处理更多：

    1. 用 `model.zero_grad()` 把模型的斜率清空一次。
    2. 依次对每一份：取出 `X` 和 `Y` 里属于这一份的行，算出它的损失，把损失除以 `pieces`，再做反向计算。
    3. 返回 `[p.grad.clone() for p in model.parameters()]`：`.clone()` 造出一份副本，之后的步骤改不到它。
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
    又是舍入误差大小的差别。

    ## 通信：全归约

    在真实的设备上，“把斜率取平均”意味着在设备之间发送数字。标准的操作叫**全归约**（all-reduce）：每台设备一开始有自己的张量，最后每台设备都得到所有张量的和（或均值）。

    最直接的办法是大家都把所有东西发给一台设备，但那台设备就成了瓶颈。常用的办法是**环**（ring）。把设备想象成围成一圈，每台只传给自己的邻居：
    """)
    return


@app.cell(hide_code=True)
def _():
    ring(['设备 A', '设备 B', '设备 C'], '三台设备围成一圈。A 传给 B，B 传给 C，C 传给 A。')
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每台设备都把自己的斜率切成和设备数一样多的段，共 $n$ 段。所有设备同时传递，每台传一段，各传各的不同的段，并把收到的加到自己那份对应的段上。这样传 $n - 1$ 次是第一圈，之后每台设备都有了某一段的完整总和。在第二圈的 $n - 1$ 次传递里，这些完成的段转一圈，让每台设备都拿到全部。

    下面是 3 台设备的第一圈。它们的斜率各有 3 个数，每段一个数。A 一开始是 $[1, 2, 3]$，B 是 $[10, 20, 30]$，C 是 $[100, 200, 300]$：

    | 第几次传递 | A 传给 B | B 传给 C | C 传给 A |
    |---|---|---|---|
    | 1 | 它的第 0 段，也就是 1 | 它的第 1 段，也就是 20 | 它的第 2 段，也就是 300 |
    | 2 | 它刚刚加好的第 2 段 | 它刚刚加好的第 0 段 | 它刚刚加好的第 1 段 |

    ## 动手算：环上的一圈

    跟着表格走。第 1 次传递之后，B 的第 0 段是多少？第 2 次传递之后，C 的第 0 段、A 的第 1 段、B 的第 2 段各是多少？接下来的第二圈还要传 2 次，每台设备每次把一个已经完成的段往下传。四次传递加起来，每台设备一共发送了多少个数？
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
    第一圈之后，完整的总和 111、222 和 333 都有了，各在一台设备上。第二圈把它们传给大家。为了在 3 台设备之间分享 3 个数的斜率，每台设备发送了 4 个数。

    再算一算一般情况下的代价。每次传递，一台设备发送一段，占斜率的 $1/n$。一共 $2(n-1)$ 次传递，所以每台设备发送

    > $$\text{发送的字节数} = 2 \times \frac{n - 1}{n} \times \text{斜率的大小}$$

    对动手算的那个例子，这是 $2 \times \tfrac{2}{3} \times 3 = 4$，和你数的一样。不管有多少台设备，它都不到斜率大小的两倍。增加设备并不会让每台的通信量增长，这就是环能很好地扩展的原因。

    ## 练习 3：ring_bytes

    写 `ring_bytes(size, devices)`，也就是这个公式：斜率占 `size` 个字节时，在环形全归约中每台设备发送多少字节。`ring_bytes(3, 3)` 得到 `4.0`，`ring_bytes(1000, 4)` 得到 `1500.0`。
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

    现在来真的。函数 `train_in_processes`（它在课程自己的代码 `learn_llm/parallel.py` 里，因为新进程必须能载入它）启动几个独立的进程。每个进程都用同样的种子建一个数字网络，所以所有副本一开始完全相同。每一步，每个进程取这一批里自己的那一份，算出斜率，然后调用 PyTorch 的 `dist.all_reduce`，把各进程的斜率加起来，再除以进程数，然后走一步。这些进程完全不共享内存：它们只通过全归约，经由电脑内部的网络交流，就像一台机器里的 GPU 通过它们之间的连接交流一样。

    按下按钮，先在一个进程里训练 200 步、每批 128 张图片，再在 2 个和 4 个进程里各训练一次，并把它们最后的旋钮和单进程的旋钮作比较。第一行是单独一个进程：进程数、秒数。之后的每一行依次是：进程数、任意旋钮的最大差别、每个进程交给全归约的斜率字节数，以及秒数。最多要一分钟，期间页面可能看起来卡住了。结果可能显示为一个可以展开的列表。
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
    差别是 $10^{-7}$ 的几倍：在独立的进程上走了 200 步之后，旋钮和单进程得到的一样，只差舍入误差。（以不同的顺序相加会改变最后几位数字，所以不要指望正好是零。）数据并行改变的是工作在*哪里*做，而不是学到*什么*。

    字节数两种情况下都是大约 4,000 万：$200 \text{ 步} \times 50{,}890 \text{ 个斜率} \times 4 \text{ 字节}$。这是每个进程交给全归约的量。如果用环，2 个进程时每个大约发送 4,100 万字节，4 个进程时大约 6,100 万（在一台电脑上，PyTorch 可能会选择别的方法）。

    耗时最能说明问题。在一台电脑上，各进程争抢同样的核心和同样的内存，启动进程也要花一点时间，所以多开进程未必更快；对这么小的模型，通常都不会更快。当每台设备是一块有自己内存的独立 GPU，而且模型大到计算远多于通信时，这个想法才划算。

    ## 太大：把模型分开

    数据并行给每台设备一份完整的副本。第 22 章算过，用 Adam 训练时这包括：旋钮、它们的斜率和 Adam 的两个平均值，每个旋钮 4 个数，所以 32 位时每个旋钮 16 字节。对一个 70 亿个旋钮的模型，每台设备上就是 112 GB，超过大多数 GPU 的容量。

    但看看每份副本是怎么处理 Adam 的平均值的：每台设备每一步都更新*全部*旋钮，而且一模一样。这浪费了内存。**分片**（sharding）把优化器的平均值分给各台设备：设备 0 只保存前四分之一旋钮的平均值并只更新它们，设备 1 负责第二个四分之一，依此类推。走完一步之后，各设备交换更新过的部分，让所有副本重新完整。同样的思路也可以给斜率分片，甚至给旋钮本身分片。这几个级别叫 **ZeRO** 第 1、2、3 级。

    每台设备需要知道哪些旋钮归自己管。设备从 0 开始编号，一台设备的编号叫它的 **rank**。旋钮不能平分时，前面的设备各多拿一个。

    /// admonition | 🧰 工具箱：剩下多少
    `a % b` 是 `a` 除以 `b` 的**余数**（remainder），和 `//` 是一对：`10 // 4` 是 2，`10 % 4` 也是 2，因为 $10 = 4 \times 2 + 2$。`12 % 4` 是 0。
    ///

    ## 动手算：10 个旋钮，4 台设备

    把编号 0 到 9 的 10 个旋钮按顺序分给 4 台设备。每台设备分到几个？再看设备 2：它的第一个旋钮是哪个？它的终点，也就是第一个不再归它管的旋钮（和切片一样），是哪个？
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

    > `shard(10, 4, 2)` 得到 `(6, 8)`，`shard(10, 4, 0)` 得到 `(0, 3)`

    每台设备分到 `count // devices` 个旋钮，前 `count % devices` 台设备各多一个。一台设备的起点，就是排在它前面的设备拿走的全部：它们各自基本的那一份，再加上其中每台多拿的那一个。`min(a, b)`（第 12 章）和 `a if test else b`（第 18 章）在这里用得上。
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
    每台设备从 112 GB 降到 14 GB（第 2 级其实是 38.5，这里四舍五入了），只是因为不再把同样的东西存八遍。激活值，也就是为反向计算保留的值，还要另外算，它们随批大小增长，而不是随旋钮数增长。

    分片不是免费的。在第 3 级，每一步在某一层运行之前，都要从其他设备收集这一层的旋钮。这是用更多的通信换取内存。

    ## 把一层分开

    对最大的模型，连一层的矩阵都要分开。回想一下，`X @ W` 结果的每一列都只由 `W` 的一列算出。所以把 `W` 按列分开：有两台设备时，设备 0 拿左半边，算出结果的左半边；设备 1 拿右半边；再把两半并排拼起来。这叫**张量并行**（tensor parallelism）。

    ## 练习 6：split_product

    写 `split_product(X, W, devices)`。`X` 的形状是 `(n, k)`，`W` 的形状是 `(k, m)`。它返回 `X @ W`，形状是 `(n, m)`，但要分成 `devices` 块来算：每台设备用 `X` 乘以 `W` 里归它管的那几列，再把各块按顺序拼起来。

    哪几列归哪台设备，由你的 `shard` 说了算：要分的一共有 `W.shape[1]` 列。`W[:, start:stop]` 取出若干列（第 14 章的 `:` 表示所有的行），`torch.cat(pieces, dim=1)`（第 14 章）把一个由表组成的列表并排拼起来。

    检查之后的单元格在随机矩阵上，把分给 3 台设备算出的结果和普通的矩阵乘法作比较。
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
        r"""先是一个空列表，然后对各个 rank 做循环：`for rank in range(devices):`。每一轮用 `shard` 问出这台设备的起点和终点，做乘法，再把这一块加进列表。循环结束后，把各块拼起来。""",
        r"""一共 6 行。`shard` 返回一对数，所以要给两半各起一个名字：`start, stop = shard(W.shape[1], devices, rank)`。这一块是 `X @ W[:, start:stop]`。""",
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
    是零，或者类似 $10^{-7}$ 的数：同样的数，相加的顺序可能不同，所以最多只有最后几位不一样。真实的系统会把这些办法组合起来：在多组机器之间用数据并行，在组内用分片，在一台机器内的各块 GPU 之间用张量并行，因为那里的连接最快。怎样搭配这些办法，是训练工程的一大块内容。

    ## 接下来

    你现在能说出训练一个模型要花多少，以及工作怎样分开。**第 24 章**把这些变成计划：如果你有一次训练的预算，模型该多大，又该读多少数据？答案遵循一条定律，你会从自己的小规模训练里测出它。

    ## 你现在会做的事

    - 解释为什么把等大分片的斜率取平均，就得到整批的斜率
    - 用梯度累积在一台设备上达到任何批大小
    - 解释全归约，亲手走一遍环，并数出它的字节数
    - 在几个真实的进程上训练，并检查结果和单进程一致
    - 把旋钮分给各台设备，并算出每个分片级别下每台设备的内存
    - 把一次矩阵乘法分到几台设备上

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** PyTorch 的 `DistributedDataParallel` 做的就是我们这些进程做的事，并让通信和反向计算同时进行。ZeRO 出自 Rajbhandari 等（2019），*FSDP* 是 PyTorch 版本的第 3 级。张量并行见 *Megatron-LM* 论文；把模型的各层分组放到不同设备上，叫*流水线并行*（pipeline parallelism）。
    """)
    return


if __name__ == "__main__":
    app.run()
