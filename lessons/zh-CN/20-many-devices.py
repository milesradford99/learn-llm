import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="20 · 多台设备")

with app.setup(hide_code=True):
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import digits
    from learn_llm.parallel import digit_model, train_in_processes, train_one_process

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 20 · 多台设备

    大模型是在成千上万块 GPU 上同时训练好几周得到的。任何一台设备都做不到：那得花上几百年，而且模型往往根本装不进一台设备的内存。这一章讲怎样把工作分开。

    你没有成千上万块 GPU，也不需要它们来理解这些想法。你电脑的处理器有好几个**核心**（core），操作系统可以同时运行好几个**进程**（process），也就是不共享内存、各自独立运行的程序，这和分开的 GPU 完全一样。学完这一章，你会在几个进程上同时训练一个网络，并检查它学到的和一个进程完全一样。

    要解决的问题有两个，答案也不同：

    - **太慢：** 把*数据*分开，让每台设备做每批的一部分。
    - **太大：** 把*模型*分开，让每台设备拿一部分旋钮。

    ## 把数据分开

    回想一步训练做了什么：在一批数据上算损失，它是所有样本的**均值**；然后算斜率；然后移动旋钮。现在让 4 台设备各拿一份完整的模型副本，各拿四分之一批。每台算出*它那一份*的均值的斜率。

    关键事实是这样的。当四份大小相同时，四份各自的均值再取平均，就是整批的均值（整批 128 个样本，每份 32 个）：每份的均值是那一份的总和除以 32，而 $\frac{1}{4}\big(\frac{a}{32} + \frac{b}{32} + \frac{c}{32} + \frac{d}{32}\big) = \frac{a + b + c + d}{128}$，其中 $a, b, c, d$ 是四份各自的总和。斜率跟着损失走（第 02 章的加法和倍数规则），所以**把四台设备的斜率取平均，正好得到整批的斜率**。如果每台设备接着对同样的旋钮走同样的一步，各份副本就一步接一步地保持一致。

    这就是**数据并行**（data parallelism）。唯一需要的通信是：每次反向计算之后，每台设备都必须得到所有斜率的平均值。

    ## 练习 1：average_slopes

    写 `average_slopes(slopes)`。`slopes` 是一个列表，每台设备一项，每一项又是一个张量列表，每个旋钮一个。返回一个张量列表：每个旋钮在各设备上的均值。`torch.stack(..., dim=0)`（第 13 章）把各设备的张量叠起来，`torch.mean(..., dim=0)` 对它们取平均。

    1. `averaged = []`
    2. `for k in range(len(slopes[0])):`，里面写 `averaged.append(torch.mean(torch.stack([device[k] for device in slopes], dim=0), dim=0))`
    3. 循环结束后，`return averaged`
    """)
    return


@app.function
def average_slopes(slopes):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "average_slopes", average_slopes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    我们在一个进程里，用第 09 章的数字网络检查这个关键事实。一份副本算出 128 张图片一批的斜率。然后 4 台“设备”各取其中 32 张，我们把它们的斜率取平均。单元格显示两者的最大差别。
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

    同样的事实也能帮助一台装不下大批量的设备：把四份一份接一份地运行，让斜率在 `.grad` 里累加（PyTorch 会累加，还记得第 09 章那个坑吗），然后在走一步之前除以 4。这叫**梯度累积**（gradient accumulation），它让任何设备都能用任何批大小，代价是时间。

    ## 练习 2：accumulated_slopes

    写 `accumulated_slopes(model, X, Y, pieces)`。把这批数据切成 `pieces` 等份，对每份的损失除以 `pieces` 之后做反向计算，让 `.grad` 把它们加起来。返回斜率的副本。

    1. 两行：`model.zero_grad()` 和 `size = len(X) // pieces`
    2. `for i in range(pieces):`，里面两行：`part = slice(i * size, (i + 1) * size)`，然后 `(F.cross_entropy(model(X[part]), Y[part]) / pieces).backward()`
    3. 循环结束后：`return [p.grad.clone() for p in model.parameters()]`

    （`slice(a, b)` 是存在一个名字里的切片：`X[slice(a, b)]` 就是 `X[a:b]`。）
    """)
    return


@app.function
def accumulated_slopes(model, X, Y, pieces):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "accumulated_slopes", accumulated_slopes, LOCALE)
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

    最直接的办法是大家都把所有东西发给一台设备，但那台设备就成了瓶颈。常用的办法是**环**（ring）。把设备想象成围成一圈，每台都把自己的斜率切成和设备数一样多的段，共 $n$ 段。在第一圈里，所有设备同时把一段传给邻居，每台传的是不同的段，并把收到的加到自己那份对应的段上。经过 $n - 1$ 次传递后，每台设备都有了某一段的完整总和。在第二圈的 $n - 1$ 次传递里，这些完成的段转一圈，让每台设备都拿到全部。

    算一算代价。每次传递，一台设备发送一段，占斜率的 $1/n$。一共 $2(n-1)$ 次传递，所以每台设备发送

    > $$\text{发送的字节数} = 2 \times \frac{n - 1}{n} \times \text{斜率的大小}$$

    不管有多少台设备，这都不到斜率大小的两倍。增加设备并不会让每台的通信量增长，这就是环能很好地扩展的原因。

    ## 练习 3：ring_bytes

    写 `ring_bytes(size, devices)`：在大小为 `size` 字节的环形全归约中，每台设备发送多少字节：

    ```
    return 2 * (devices - 1) / devices * size
    ```
    """)
    return


@app.function
def ring_bytes(size, devices):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "ring_bytes", ring_bytes, LOCALE)
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

    数据并行给每台设备一份完整的副本。第 19 章算过，用 Adam 训练时这包括：旋钮、它们的斜率和 Adam 的两个平均值，每个旋钮 4 个数，所以 32 位时每个旋钮 16 字节。对一个 70 亿个旋钮的模型，每台设备上就是 112 GB，超过大多数 GPU 的容量。

    但看看每份副本是怎么处理 Adam 的平均值的：每台设备每一步都更新*全部*旋钮，而且一模一样。这浪费了内存。**分片**（sharding）把优化器的平均值分给各台设备：设备 0 只保存前四分之一旋钮的平均值并只更新它们，设备 1 负责第二个四分之一，依此类推。走完一步之后，各设备交换更新过的部分，让所有副本重新完整。同样的思路也可以给斜率分片，甚至给旋钮本身分片。这几个级别叫 **ZeRO** 第 1、2、3 级。

    ## 练习 4：shard

    每台设备需要知道哪些旋钮归自己管。写 `shard(count, devices, rank)`，返回设备 `rank` 在 `count` 个旋钮（从 0 编号）中所分到部分的起点和终点。`stop` 是第一个不归你管的旋钮，和切片一样。当 `count` 不能整除时，前面的设备各多拿一个。`%` 给出除法的余数：`10 % 4` 是 2。

    1. 两行：`size = count // devices` 和 `extra = count % devices`
    2. `start = rank * size + min(rank, extra)`
    3. `stop = start + size + (1 if rank < extra else 0)`
    4. `return start, stop`
    """)
    return


@app.function
def shard(count, devices, rank):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "shard", shard, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：bytes_per_device

    分片能省多少？用 32 位数时，一个旋钮自身占 4 字节，它的斜率占 4 字节，Adam 的两个平均值占 8 字节。写 `bytes_per_device(knobs, devices, stage)`。第 0 级是普通的数据并行，第 1 级给 Adam 的平均值分片，第 2 级再加上斜率，第 3 级再加上旋钮。

    1. 三行：`weights = 4 * knobs`、`slopes = 4 * knobs` 和 `adam = 8 * knobs`
    2. `if stage >= 1:`，里面写 `adam = adam / devices`
    3. `if stage >= 2:`，里面写 `slopes = slopes / devices`
    4. `if stage >= 3:`，里面写 `weights = weights / devices`
    5. `return weights + slopes + adam`

    （每个 `if` 都和其他的对齐，而不是放在前一个里面。）
    """)
    return


@app.function
def bytes_per_device(knobs, devices, stage):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("20", "bytes_per_device", bytes_per_device, LOCALE)
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
    每台设备从 112 GB 降到 14 GB（第 2 级其实是 38.5，这里四舍五入了），只是因为不再把同样的东西存八遍。（这张表用的是 `bytes_per_device(7e9, 8, stage)`，其中 `7e9` 表示 $7 \times 10^9$。）激活值，也就是为反向计算保留的值，还要另外算，它们随批大小增长，而不是随旋钮数增长。

    分片不是免费的。在第 3 级，每一步在某一层运行之前，都要从其他设备收集这一层的旋钮。这是用更多的通信换取内存。

    ## 把一层分开

    对最大的模型，连一层的矩阵都要分开。回想一下，`X @ W` 结果的每一列都只由 `W` 的一列算出。所以把 `W` 按列分开：设备 0 拿左半边，算出结果的左半边；设备 1 拿右半边；再把两半拼起来。这叫**张量并行**（tensor parallelism）。下面的单元格在随机矩阵上检查这个想法：
    """)
    return


@app.cell
def _():
    torch.manual_seed(0)
    X_demo = torch.rand(5, 8)
    W_demo = torch.rand(8, 6)
    left = X_demo @ W_demo[:, :3]
    right = X_demo @ W_demo[:, 3:]
    torch.max(torch.abs(torch.cat([left, right], dim=1) - X_demo @ W_demo)).item()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    大约 $10^{-7}$：同样的数，以不同的顺序相加，所以只有最后几位不同。真实的系统会把这些办法组合起来：在多组机器之间用数据并行，在组内用分片，在一台机器内的各块 GPU 之间用张量并行，因为那里的连接最快。怎样搭配这些办法，是训练工程的一大块内容。

    ## 接下来

    你现在能说出训练一个模型要花多少，以及工作怎样分开。**第 21 章**把这些变成计划：如果你有一次训练的预算，模型该多大，又该读多少数据？答案遵循一条定律，你会从自己的小规模训练里测出它。

    ## 你现在会做的事

    - 解释为什么把等大分片的斜率取平均，就得到整批的斜率
    - 用梯度累积在一台设备上达到任意批大小
    - 解释全归约，并算出环形全归约的字节数
    - 在几个真实的进程上训练，并检查结果和单进程一致
    - 算出每一级分片下每台设备所需的内存
    - 把一次矩阵乘法分到多台设备上

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** PyTorch 的 `DistributedDataParallel` 做的就是我们这些进程做的事，并让通信和反向计算同时进行。ZeRO 出自 Rajbhandari 等（2019），*FSDP* 是 PyTorch 版的第 3 级。张量并行见 *Megatron-LM* 论文；而把模型的各层分成几组、放到不同设备上，叫*流水线并行*（pipeline parallelism）。
    """)
    return


if __name__ == "__main__":
    app.run()
