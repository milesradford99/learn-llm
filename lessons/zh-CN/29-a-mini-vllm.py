import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="29 · 迷你 vLLM")

with app.setup(hide_code=True):
    import json
    import time

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.diagrams import pipeline
    from learn_llm.qwen import END_OF_TURN, FOLDER, TYPES, QwenTokenizer, layer_knobs, missing

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 29 · 迷你 vLLM

    第 28 章让单个请求跑得很快，但真实的在线服务从来不会只面对一个用户。人们随时都会发来提问：有的只有短短一句话，有的贴了一整篇长文档；有的只需要简短回答，有的要写满一整页。
    在一台机器上同时接待所有请求的程序，叫作**推理引擎（inference engine）**，也叫服务引擎。最有名的开源推理引擎叫 *vLLM*，这一章我们就用它的三个核心设计，亲手搭一个迷你版：

    1. **批处理（batching）**：让多个请求共用模型的每一次前向计算；
    2. **分页 KV 缓存（paged KV cache）**：所有请求的缓存共用同一块内存池，按固定大小的小页分配；
    3. **分块预填充（chunked prefill）**：把长提示切成几小段来读，避免一个人把所有人卡住。

    评价一个推理引擎，主要看两个互相牵制的指标。**吞吐量（throughput）**是所有用户加在一起，系统每秒能吐出多少个词元；**延迟（latency）**是每个用户自己要等多久——也就是第 28 章量过的“首字等待时间”和“之后每个词元的间隔时间”。

    **前面章节的积木。**下面折叠起来的单元格里放着之前写好的工具：第 27 章的文件读取器和 `share_heads`、第 20 章的 `rms_norm`、第 28 章带起始位置 `start` 的 `rope` 和 `attend_new`，以及第 22 章的计时器 `time_it`。紧接着的单元格和第 28 章一样，负责加载 Qwen 模型权重和分词器。
    """)
    return


@app.function(hide_code=True)
def read_header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), 8 + n


@app.function(hide_code=True)
def read_tensor(path, entry, data_start):
    start, stop = entry["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + start)
        data = f.read(stop - start)
    return torch.frombuffer(bytearray(data), dtype=TYPES[entry["dtype"]]).reshape(entry["shape"]).float()


@app.function(hide_code=True)
def share_heads(x, times):
    return torch.repeat_interleave(x, times, dim=1)


@app.function(hide_code=True)
def rms_norm(x, gain):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    return gain * x / rms


@app.function(hide_code=True)
def rope(x, base, start):
    words, size = x.shape[-2], x.shape[-1]
    half = size // 2
    speeds = 1 / base ** (torch.arange(half) / half)
    angles = torch.arange(start, start + words)[:, None] * speeds[None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * torch.cos(angles) - b * torch.sin(angles), a * torch.sin(angles) + b * torch.cos(angles)], dim=-1)


@app.function(hide_code=True)
def attend_new(Q, K, V):
    new, total = Q.shape[-2], K.shape[-2]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones(new, total, dtype=torch.bool), diagonal=total - new + 1)
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


@app.function(hide_code=True)
def time_it(f, repeats):
    f()
    started = time.perf_counter()
    for turn in range(repeats):
        f()
    return (time.perf_counter() - started) / repeats


@app.cell
def _():
    mo.stop(len(missing()) > 0, mo.md("*这台电脑上还没有这个模型。请先用第 27 章的按钮下载它，再重新打开这一页。*"))
    weights = FOLDER / "model.safetensors"
    header, data_start = read_header(weights)
    knobs = {}
    for name in header:
        if name != "__metadata__" and name != "lm_head.weight":
            knobs[name] = read_tensor(weights, header[name], data_start)
    tokenizer = QwenTokenizer.load(FOLDER)
    len(knobs)
    return knobs, tokenizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一次运行，服务多个请求

    第 28 章发现，解码一步之所以慢，是因为哪怕只推**一个**向量过一遍模型，也得把全部 2.4 GB 的模型权重（旋钮）从内存里搬一遍。真正耗时间的是搬运数据，乘法计算反而便宜得多。

    既然搬一次权重这么贵，不如一趟多带几个向量一起算。假设有五个请求都在等下一个词元，把它们的五个向量叠成一个张量（每个请求占一行），每个权重矩阵就能一次乘完这五行——权重只搬一次，却干了五份活。这就是**批处理（batching）**。

    模型里只有一部分没法共用：注意力。每个请求的新词元只能回看**它自己**前面的词元（通过它自己的 KV 缓存），绝不能串到别人的文本里。所以权重矩阵把所有行打包一起乘，而注意力则对每个请求单独算。

    ## 动手算：共用一次运行的代价

    假设对单个请求来说，解码一步要 0.125 秒，而且时间全部花在搬运权重上。如果 8 个请求共用这一步，并且只算搬运权重的时间，这一步要多久？引擎为这 8 个请求加起来每秒能写出多少个词元？单个请求单独跑时又是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    share_boxes = hand.boxes(['8 个请求一步的秒数', '8 个一起时每秒词元数', '1 个单独时每秒词元数'], step=0.001)
    return (share_boxes,)


@app.cell(hide_code=True)
def _(share_boxes):
    hand.marked(share_boxes, {'8 个请求一步的秒数': 0.125, '8 个一起时每秒词元数': 64, '1 个单独时每秒词元数': 8}, {
        '8 个请求一步的秒数': r"""不管拼了几行，权重只搬一次""",
        '8 个一起时每秒词元数': r"""每 0.125 秒产出 8 个词元：$8 / 0.125$""",
        '1 个单独时每秒词元数': r"""$1 / 0.125$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这是理想上限。实际收益会小一些，因为每多一行确实要多做一点矩阵计算，而注意力必须逐个请求单独算，每多一个请求就多一份开销。下面几节我们先搭好引擎，再实测看看能提速多少。

    ## 两种批处理

    请求不会同时到达，也不会同时说完：有的回答只要 3 个词元，有的要写 200 个。最简单的做法叫**静态批处理（static batching）**：凑齐一批请求一起跑，必须等这一批里**所有人**都写完，再放下一批进来。
    在苦等那个长回答的时候，已经写完的请求空出来的槽位就只能干晾着。

    **连续批处理（continuous batching）**每一步都重新排班：只要有请求写完了，立刻离开这一批，下一步马上让排队的请求补上空位，让批处理槽位始终尽量满载。

    ## 动手算：空着的槽位

    一批共有 4 个请求槽位。它们的回答分别要 3、10、2、5 步。用静态批处理，这一组一共要跑多少步？定义一个**槽位步（slot-step，即 1 个请求槽占用 1 步）**：这一组总共占了多少个槽位步？其中有多少个真正用来写出了词元？利用率是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    static_boxes = hand.boxes(['这一组的步数', '槽位步', '用上的槽位步', '用上的比例'], step=0.01)
    return (static_boxes,)


@app.cell(hide_code=True)
def _(static_boxes):
    hand.marked(static_boxes, {'这一组的步数': 10, '槽位步': 40, '用上的槽位步': 20, '用上的比例': 0.5}, {
        '这一组的步数': r"""最长的回答结束，这一组才算结束""",
        '槽位步': r"""10 步里每一步都占着 4 个槽位""",
        '用上的槽位步': r"""每个请求各占 1 个槽位，占的步数等于它的回答长度：$3 + 10 + 2 + 5$""",
        '用上的比例': r"""用上的除以总数""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    整整一半的槽位都白白空着。下面的 `steps_needed` 用来统计给定的一组回答长度和槽位数（`places`）一共需要跑多少步。`running` 列表记着每个运行中请求还剩多少个词元没写：每跑一步减 1，写完的立刻离开。
    `continuous` 控制批处理模式：设为 `False` 时，只有当前批次全部清空，才允许新请求进来。
    """)
    return


@app.function
def steps_needed(lengths, places, continuous):
    waiting, running, steps = lengths, [], 0
    while len(waiting) + len(running) > 0:
        room = places - len(running)
        if continuous == False and len(running) > 0:
            room = 0
        running = running + waiting[:room]
        waiting = waiting[room:]
        running = [left - 1 for left in running if left > 1]
        steps = steps + 1
    return steps


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：在脑子里跑一遍

    在脑子里对回答长度 3、10、2、5、4、4 跑一遍 `steps_needed`，有 4 个位置。`continuous` 为 `False` 时它数出多少步？为 `True` 时呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    run_boxes = hand.boxes(['静态的步数', '连续的步数'])
    return (run_boxes,)


@app.cell(hide_code=True)
def _(run_boxes):
    hand.marked(run_boxes, {'静态的步数': 14, '连续的步数': 10}, {
        '静态的步数': r"""前四个要 10 步，和上一题一样；之后两个 4 才开始""",
        '连续的步数': r"""2 在第 2 步之后离开，所以一个 4 在第 3 到 6 步运行；3 在第 3 步之后离开，所以另一个 4 在第 4 到 7 步运行；10 最后结束""",
    }, LOCALE)
    return


@app.cell
def _():
    answer_lengths = np.random.default_rng(29).integers(5, 61, 40).tolist()
    steps_needed(answer_lengths, 8, False), steps_needed(answer_lengths, 8, True), sum(answer_lengths) / 8
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在换成 40 个请求，回答长度在 5 到 60 个词元之间随机抽取，用 8 个槽位。单元格依次打印静态批处理的步数、连续批处理的步数，以及理论最少步数（假如 8 个槽位每一步都满载）：测试机器上是 281、182 和 150。连续批处理已经非常接近理论极限，所以真实的推理引擎全都会这么做。

    ## 缓存放在哪里

    批处理带来了新的内存问题。第 28 章里，每个请求的缓存用 `torch.cat` 一步步拉长，每次都要重新分配一整块新张量。很多请求在不同时刻进进出出，内存很快就会被切得支离破碎。
    推理引擎希望在启动时**一次性**把内存预留出来，之后再按需分给各个请求。

    可每个请求该分多少呢？回答写完之前，谁也不知道它有多长。如果按最大长度给每个请求提前留好空间（比如 4,096 个词元），按第 28 章算过的每个词元约 229 KB（229,376 字节），单个请求就要占 0.94 GB。八个请求就能把一台笔记本塞爆，而且绝大部分空间根本用不到。

    vLLM 的解法借鉴了操作系统的分页内存管理：把内存切成大小完全相同的**缓存块**（也叫**内存页**，block），这里每块装 16 个词元；请求写满当前这块，就再按需领下一块。所有缓存块统一放在一个预先开好的大张量里，叫作**内存池（pool）**。
    同一个请求的缓存块在池里不必紧挨着。每个请求维护一张**块表（block table）**，按顺序列出自己占用的缓存块编号；引擎则维护一张**空闲块列表**。请求一结束，它占用的缓存块立刻归还给空闲列表。

    这就是**分页 KV 缓存（paged KV cache）**。每个请求最多只有最后一块没装满，因此最多只浪费 15 个词元的空间。

    ## 动手算：缓存块

    每个缓存块装 16 个词元。一个 300 个词元的请求需要几个缓存块？这些块里有几个词元的位置没用上？一个 32 个词元的请求需要几块？
    """)
    return


@app.cell(hide_code=True)
def _():
    block_boxes = hand.boxes(['300 个词元的块数', '没用上的位置', '32 个词元的块数'])
    return (block_boxes,)


@app.cell(hide_code=True)
def _(block_boxes):
    hand.marked(block_boxes, {'300 个词元的块数': 19, '没用上的位置': 4, '32 个词元的块数': 2}, {
        '300 个词元的块数': r"""18 块装 288 个词元，最后 12 个还要再来一块""",
        '没用上的位置': r"""$19 \times 16 - 300$""",
        '32 个词元的块数': r"""正好 2 个满块，不多不少""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：blocks_for

    写 `blocks_for(length, block_size)`。它返回装下 `length` 个词元需要多少个大小为 `block_size` 的缓存块：也就是 `length` 除以 `block_size` 后**向上**取整。0 个词元需要 0 块。

    `blocks_for(300, 16)` 必须得到 19，`blocks_for(32, 16)` 必须得到 2。

    Python 里的整除 `//`（第 11 章）会向下取整（直接丢掉余数）。向上取整的一个小技巧：先给 `length` 加上 $\text{块大小} - 1$，再做整除。
    """)
    return


@app.function
def blocks_for(length, block_size):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "blocks_for", blocks_for, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""每块 16 个词元，在 32 和 33 上试试这个窍门：$32 + 15 = 47$，$33 + 15 = 48$。其中哪个经过 `// 16` 变成 3？""",
        r"""只要一行：`(length + block_size - 1) // block_size`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 找到一个词元

    一个请求的第 $p$ 号词元在哪里？它在请求**块表**里的第 $p \,//\, 16$ 项所指的块里，在这个块里的位置是 $p \,\%\, 16$（第 23 章的 `%`，余数）。块表告诉你那是池里的哪一块。

    ## 动手算：查一查

    一个请求的块表是 `[7, 2, 9]`，每块装 16 个词元。它的第 20 号词元在池里的哪一块，在那块里的第几个位置？第 40 号词元呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    slot_boxes = hand.boxes(['第 20 号词元的块', '第 20 号词元的位置', '第 40 号词元的块', '第 40 号词元的位置'])
    return (slot_boxes,)


@app.cell(hide_code=True)
def _(slot_boxes):
    hand.marked(slot_boxes, {'第 20 号词元的块': 2, '第 20 号词元的位置': 4, '第 40 号词元的块': 9, '第 40 号词元的位置': 8}, {
        '第 20 号词元的块': r"""$20 \,//\, 16 = 1$，而块表的第 1 项是……""",
        '第 20 号词元的位置': r"""$20 \,\%\, 16$""",
        '第 40 号词元的块': r"""$40 \,//\, 16 = 2$""",
        '第 40 号词元的位置': r"""$40 - 2 \times 16$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 2：slot

    写 `slot(table, position, block_size)`。`table` 是一个请求的块表，一个块编号的列表。它返回一对数：装着第 `position` 号词元的、池里那一块的编号，以及在那块里的位置。

    `slot([7, 2, 9], 20, 16)` 必须得到 `(2, 4)`。
    """)
    return


@app.function
def slot(table, position, block_size):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "slot", slot, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""两个数，和你动手算时一样：`table` 的哪一项（用 `//`），以及位置（用 `%`）。你返回的第一个数是块表里那一项的**内容**，不是它在块表里的序号。""",
        r"""只要一行：`return table[position // block_size], position % block_size`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 读出一个请求的缓存

    *（术语约定：本章用第 27 章引入的**层（layer）**来称呼 Qwen 的 28 层网络，正好和代码里的 `layer` 一致；后文凡是说**缓存块**或**块**，一律指分页 KV 缓存里的内存页。）*

    模型每一层对应的内存池，都是一个形状为 `(缓存块数, 头数, 块大小, 每头维数)` 的四维张量：本章的 Qwen 配置是 `(160, 8, 16, 128)`。键（K）和值（V）各有自己的内存池，模型的 28 层每层都有一对 `(pool_k, pool_v)`。

    注意力计算要的键，必须和第 28 章一样是四维张量 `(1, 头数, 词元数, 每头维数)`，并且按词元先后排好。所以引擎从内存池里把这个请求的各个缓存块收集起来，沿着词元那一维拼在一起，再把末尾没装满的空位切掉。

    /// admonition | 🧰 工具箱：张量列表，以及在最前面加一维
    - `[pool[b] for b in [4, 1]]` 会取出两个张量组成列表：先是池里的第 4 号缓存块，再是第 1 号缓存块。每个的形状都是 `(头数, 块大小, 每头维数)`。
    - `torch.cat` 可以把任意长度的张量列表拼起来，和第 14 章一样：`torch.cat([a, b, c], dim=1)`。
    - `x[None]` 会在最前面加一个大小为 1 的维度，就像第 20 章的 `v[:, None]` 在第二个位置加一维：形状 `(8, 40, 128)` 会变成 `(1, 8, 40, 128)`。
    ///

    ## 练习 3：read_cache

    写 `read_cache(pool, table, length)`。`pool` 的形状是 `(缓存块数, 头数, 块大小, 每头维数)`，`table` 是一个请求的块表，`length` 是这个请求到目前为止已经拥有的词元数。
    它按顺序返回这个请求的键（或值），形状是 `(1, 头数, length, 每头维数)`。

    每块 2 个词元、块表 `[3, 0]`、`length` 为 3 时，结果先放池里第 3 号缓存块的 2 个词元，再放第 0 号缓存块的第一个词元。
    """)
    return


@app.function
def read_cache(pool, table, length):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "read_cache", read_cache, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""三步，对应工具箱的三行：把块表里的各块收集成一个列表，沿着数词元的那一维拼接（每块是（头数、块大小、数的个数）），然后留下前 `length` 个词元，再在前面加上一维。""",
        r"""一共 2 行。`joined = torch.cat([pool[b] for b in table], dim=1)`，然后返回 `joined[:, :length][None]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 让一整批请求一起过一层

    下面是把第 28 章的 `block_step` 改写成的批处理版本。不同请求这一步要喂的新词元数不一样——解码中的请求只有 1 个，正在读提示的请求可能有 20 个——所以引擎把所有请求的新词元首尾相接，拼成一条长条 `x`，形状是 `(1, total, 1024)`：

    ```text
    [请求 A (1词) | 请求 B (1词) | 请求 C (20词)]  ->  拼成一条 x，形状 (1, 22, 1024)
    ```

    `parts` 记着每个请求在这条长条里的位置：每个请求对应一个三元组 `(table, seen, new)`。读 `block_batch` 时留意三个关键细节：

    - **先整条过矩阵，再按 `at : at + new` 拆回各请求。** `q`、`k`、`v` 三个权重矩阵一口气算完所有 `total` 个词元，得到三个三维张量：`q` 形状 `(16, total, 128)`，`k` 和 `v` 形状 `(8, total, 128)`。随后用游标 `at` 沿着长条往前走，按 `at : at + new` 切出每个请求自己的那一段，从它各自的 `seen` 位置开始做 RoPE 旋转，再分别写入缓存、单独算注意力。
    - **写入四维内存池：`pool_k[b, :, place] = k_one[:, j]`。** `pool_k` 是四维的 `(缓存块数, 8, 16, 128)`，这里省略了最后一维的 `:`（等价于 `pool_k[b, :, place, :]`），所以等号两边形状都是 `(8, 128)`——刚好对应这 1 个词元槽位上全部 8 个 KV 头、每头 128 个数。（`pool_v[b, :, place] = v[:, at + j]` 同理。）
    - **为什么先 `[None]` 再 `[0]`：`attend_new(q_one[None], ...)[0]`。** `q_one` 是三维 `(16, new, 128)`，而 `read_cache` 返回四维 `(1, 8, seen + new, 128)`，第 28 章的 `attend_new` 也要求输入是四维 `(1, 头数, 词元数, 128)`。所以先用 `[None]` 临时加回第 0 维 `batch=1` 以匹配接口；算完后立刻用 `[0]` 去掉这维，变回三维 `(16, new, 128)`，方便后面用 `torch.cat(blended, dim=1)` 把所有请求重新拼成一条，一起过输出矩阵和前馈网络。

    `pools` 为模型的 28 层每层各存一对 `(pool_k, pool_v)`。
    """)
    return


@app.function
def block_batch(x, L, parts, pool_k, pool_v):
    total = x.shape[1]
    y = rms_norm(x, L["norm1"])
    q = rms_norm((y @ L["q"].T).reshape(total, 16, 128).transpose(0, 1), L["q_norm"])
    k = rms_norm((y @ L["k"].T).reshape(total, 8, 128).transpose(0, 1), L["k_norm"])
    v = (y @ L["v"].T).reshape(total, 8, 128).transpose(0, 1)
    blended, at = [], 0
    for table, seen, new in parts:
        q_one = rope(q[:, at:at + new], 1000000, seen)
        k_one = rope(k[:, at:at + new], 1000000, seen)
        for j in range(new):
            b, place = slot(table, seen + j, 16)
            pool_k[b, :, place] = k_one[:, j]
            pool_v[b, :, place] = v[:, at + j]
        K, V = read_cache(pool_k, table, seen + new), read_cache(pool_v, table, seen + new)
        blended.append(attend_new(q_one[None], share_heads(K, 2), share_heads(V, 2))[0])
        at = at + new
    x = x + torch.cat(blended, dim=1).transpose(0, 1).reshape(1, total, 2048) @ L["out"].T
    y = rms_norm(x, L["norm2"])
    return x + (F.silu(y @ L["gate"].T) * (y @ L["up"].T)) @ L["down"].T


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：跟着一批走一遍

    三个请求共用一步。两个在解码：一个的缓存里有 40 个词元，另一个有 7 个。第三个正在一次读完它 20 个词元的提示。所以 `parts` 是 `[(table_a, 40, 1), (table_b, 7, 1), (table_c, 0, 20)]`。

    `total` 是多少？转置之后 `q` 的形状中间那个数，也就是它装着几个词元，是多少？`rope` 给第二个请求的词元的位置是几？第三个请求的注意力读到的 `K` 有多少个词元长？
    """)
    return


@app.cell(hide_code=True)
def _():
    batch_boxes = hand.boxes(['total', 'q 里的词元数', '第二个请求的词元位置', '第三个请求的 K 的词元数'])
    return (batch_boxes,)


@app.cell(hide_code=True)
def _(batch_boxes):
    hand.marked(batch_boxes, {'total': 22, 'q 里的词元数': 22, '第二个请求的词元位置': 7, '第三个请求的 K 的词元数': 20}, {
        'total': r"""每个请求的新词元排在一起：$1 + 1 + 20$""",
        'q 里的词元数': r"""`q` 装着每个新词元的查询""",
        '第二个请求的词元位置': r"""它的缓存装着 0 到 6 号词元""",
        '第三个请求的 K 的词元数': r"""`seen + new`：它没有更早的词元""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `run_batch` 跑完模型的全部 28 层，为每个请求返回一行词表分数。留意最后一行的 `x[0, ends]`：`ends` 记录了每个请求**最后一个**新词元在长条里的下标。提示词中间的词元只需要把 KV 缓存填好，并不需要预测下一个词，所以 `x[0, ends]` 只抽出每个请求末尾那一个输出向量，省掉了在中间词元上白做一遍词表投影的开销。

    `new_pools` 一次性把内存池开好：为模型的 28 层各配一对 K/V 池，每个池有 160 个缓存块、每块装 16 个词元，总共 0.59 GB。
    """)
    return


@app.function
def run_batch(knobs, ids, parts, pools):
    with torch.no_grad():
        x = knobs["model.embed_tokens.weight"][torch.tensor([ids])]
        for layer in range(28):
            x = block_batch(x, layer_knobs(knobs, layer), parts, pools[layer][0], pools[layer][1])
        x = rms_norm(x, knobs["model.norm.weight"])
        ends, at = [], 0
        for table, seen, new in parts:
            at = at + new
            ends.append(at - 1)
        return x[0, ends] @ knobs["model.embed_tokens.weight"].T


@app.function
def new_pools():
    return [(torch.zeros(160, 8, 16, 128), torch.zeros(160, 8, 16, 128)) for layer in range(28)]


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 量一量共用的效果

    下面的单元格让 1、2、4、8、16 个请求各有一份同样的 30 个词元的缓存，然后为它们一起计时一步解码。每一行：请求个数、这一步的秒数、所有请求加起来的每秒词元数。大约要一分钟。
    """)
    return


@app.cell
def _(knobs, tokenizer):
    filler = tokenizer.encode("Once upon a time there was a little fox who lived at the edge of a big forest near a river. Every morning the fox ran down to the water to look at the fish")[:30]
    sharing = []
    for _count in [1, 2, 4, 8, 16]:
        _pools = new_pools()
        _tables = [[2 * i, 2 * i + 1] for i in range(_count)]
        run_batch(knobs, filler * _count, [(t, 0, 30) for t in _tables], _pools)

        def _decode():
            return run_batch(knobs, [100] * _count, [(t, 30, 1) for t in _tables], _pools)

        _seconds = time_it(_decode, 3)
        sharing.append((_count, round(_seconds, 3), round(_count / _seconds, 1)))
    sharing
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    测试机器上：1 个请求每秒 7.5 个词元，8 个请求 22 个，16 个请求 35 个。虽然没有达到理想的 16 倍，但同一台电脑不加任何硬件，总吞吐量就翻了 4 倍多。每一步耗时只缓慢增加，所以单个用户看到词元的速度稍慢了一点：高吞吐量，是用一点点延迟换来的。

    *（收益比理想上限小，部分原因是这一页的 Python 循环开销：`block_batch` 逐个请求、逐个新词元循环写入。真正的推理引擎会把这些步骤合并成一次 GPU 算子，也就是第 28 章说的**融合运算**。）*

    ## 放请求进来

    现在来看引擎的主循环。每一步，它按顺序做四件事：

    1. 把刚到达的请求放进**等待（waiting）**列表；
    2. 把等待中的请求**接纳（admit）**进**运行（running）**列表，能接纳多少就接纳多少；
    3. 决定每个运行中的请求这一步喂多少个词元；
    4. 把这些词元打包跑一遍模型，本轮把待读词元全部喂完的请求，各自生成下一个词元。

    负责决定谁能进场运行的部分叫作**调度器（scheduler）**。我们的调度器有两条硬约束：同时运行的请求不能超过 `max_running` 个，而且空闲的缓存块必须装得下新请求的提示。请求严格按先来后到排队：如果队头那个等待请求放不下，排在后面的也得跟着等。

    每个请求用一个字典表示，里面的 `"ids"` 列表装着它到目前为止的全部词元（包含提示词）。

    ## 练习 4：admit

    写 `admit(waiting, running, free_blocks, max_running, block_size)`。`waiting` 和 `running` 是请求列表，`free_blocks` 是当前空闲的缓存块个数。它返回一对列表：更新后的等待列表和运行列表。

    只要还有等待中的请求、运行列表长度小于 `max_running`、并且**队头**等待请求的提示能装进 `free_blocks`：就把这个请求移到 `running` 末尾，并从 `free_blocks` 里扣掉它占用的块数。

    比如有 3 个 16 词元的空闲块、当前没有运行中的请求、最多同时跑 4 个、等待队列里的提示长度依次为 20、10、5 个词元：前两个会被接纳（分别占 2 块和 1 块），第三个只能继续等。
    """)
    return


@app.function
def admit(waiting, running, free_blocks, max_running, block_size):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "admit", admit, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""一个 `while` 循环（第 19 章），三个条件用 `and` 连起来。循环里三行：空闲块变少，运行列表变长（`running + [waiting[0]]`），等待列表变短（`waiting[1:]`）。对 `len(waiting[0]["ids"])` 用你的 `blocks_for`。""",
        r"""一共 5 行。循环的条件是 `len(waiting) > 0 and len(running) < max_running and blocks_for(len(waiting[0]["ids"]), block_size) <= free_blocks`。循环之后，`return waiting, running`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 一个长提示卡住所有人

    还有一种情况很糟。四个人正看着回答一点点写出来，每四分之一秒一个词元。这时有人贴进一个 540 个词元的故事，并就它提问。下一步要一次读完这 540 个词元：长预填充有多贵，第 28 章量过。
    在这一步里，别人的下一个词元要等好几秒，而不是四分之一秒。

    **分块预填充**给每一步的工作量设一个上限，叫**词元预算**（token budget），比如 64 个词元。正在解码的请求每个只要一个词元，排在前面。预算剩下的部分按顺序分给还在读提示的请求，一个长提示就分几步来读，每步 61 或 62 个词元。
    谁都不用等超过一个小步。长提示自己的首个词元会晚一点出来，因为它每一步都和别人共用。

    ## 动手算：分配预算

    预算是 64。三个请求在解码，还有一个 540 个词元的提示，一个都还没读。第一步喂给提示多少个词元？如果那三个一直在解码，读完整个提示要几步？其中最后一步读了提示的多少个词元？
    """)
    return


@app.cell(hide_code=True)
def _():
    chunk_boxes = hand.boxes(['第一步的提示词元数', '读完提示的步数', '最后一步的提示词元数'])
    return (chunk_boxes,)


@app.cell(hide_code=True)
def _(chunk_boxes):
    hand.marked(chunk_boxes, {'第一步的提示词元数': 61, '读完提示的步数': 9, '最后一步的提示词元数': 52}, {
        '第一步的提示词元数': r"""$64 - 3$""",
        '读完提示的步数': r"""8 步读了 $8 \times 61 = 488$ 个词元，还剩一些""",
        '最后一步的提示词元数': r"""$540 - 488$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 5：chunk_sizes

    写 `chunk_sizes(unfed, budget)`。`unfed` 是一个列表，每个运行中的请求一个数，按服务的顺序排列：它有多少个词元还没喂给模型。正在解码的请求是 1。它返回一个同样长的列表：这一步每个请求喂多少个词元。

    按顺序处理这些请求。每个请求得到它没喂的个数和预算剩下的个数中较小的那个，得到多少就从预算里减掉多少。预算用完之后，剩下的请求都得到 0。

    `chunk_sizes([1, 1, 1, 540], 64)` 必须得到 `[1, 1, 1, 61]`，`chunk_sizes([1, 100, 30], 64)` 必须得到 `[1, 63, 0]`。
    """)
    return


@app.function
def chunk_sizes(unfed, budget):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "chunk_sizes", chunk_sizes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""从一个空列表开始。在遍历 `unfed` 的 `for` 循环里，用第 12 章的 `min` 选出大小；把它追加到列表里，再从预算里减掉它。""",
        r"""一共 6 行。循环里：`size = min(n, budget)`，然后 `sizes.append(size)`，然后 `budget = budget - size`。循环之后，`return sizes`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 引擎主循环

    下面就是完整的 36 行引擎状态机，把你写的 `admit`、`chunk_sizes`、`blocks_for`（以及 `block_batch` 里的 `slot` 和 `read_cache`）全部串了起来。`new_request` 用来创建一个请求字典：`"ids"` 装着目前的全部词元（提示词加上已生成的回答词元），`"prompt"` 记着原始提示词的长度，`"fed"` 记着已经喂给模型、写进 KV 缓存的词元数，`"table"` 是块表，`"arrives"` 是到达步号。请求列表需要按到达顺序排好。

    读这段代码时，抓住两条最核心的状态判断就能看清全局：

    - **`r["fed"] >= r["prompt"]`（预填充已完成，进入解码阶段）：**只要已喂词元数达到或超过原始提示长度，这个请求就读完了提示、进入逐词解码阶段，每步只需要喂 1 个新词元；反之 `r["fed"] < r["prompt"]` 说明还在预填充阶段。`order` 正是用这个条件把“正在解码的请求”排在前面，“还在读提示的请求”排在后面，再交给 `chunk_sizes` 分配预算。
    - **`r["fed"] == len(r["ids"])`（本轮把待喂词元全部喂完，可以采样下一个词）：**`run_batch` 算完并更新 `r["fed"]` 后，如果已喂词元数追上了 `r["ids"]` 的总长度，说明模型已经读到末尾，可以用 `argmax(scores[i])` 选出下一个词——如果遇到 `END_OF_TURN` 或达到长度上限 `limit`，请求就结束并把缓存块归还 `free`；否则把新词元追加进 `r["ids"]`，留到下一步喂入。反过来，如果一个长提示这一步只喂了一半（`r["fed"] < len(r["ids"])`），那这一步**只填缓存、先不生成新词**，剩下的提示留给下一步继续喂。

    `log` 会为每一步记下哪些请求各自喂了几个词元，以及这一步花了多少秒。

    *（真正的引擎还要处理缓存块耗尽的情况：vLLM 会暂时挂起最晚进场的请求，回收它的缓存块，等有空位再重算缓存。本章 160 个缓存块的内存池对页面里的实验来说绰绰有余。）*
    """)
    return


@app.function
def new_request(name, ids, arrives):
    return {"name": name, "ids": ids, "prompt": len(ids), "fed": 0, "table": [], "arrives": arrives}


@app.function
def engine(knobs, requests, max_running, budget, limit):
    pools, free = new_pools(), list(range(160))
    later, waiting, running, finished, log, step = requests, [], [], [], [], 0
    while len(later) + len(waiting) + len(running) > 0:
        waiting = waiting + [r for r in later if r["arrives"] <= step]
        later = [r for r in later if r["arrives"] > step]
        waiting, running = admit(waiting, running, len(free), max_running, 16)
        order = [r for r in running if r["fed"] >= r["prompt"]] + [r for r in running if r["fed"] < r["prompt"]]
        sizes = chunk_sizes([len(r["ids"]) - r["fed"] for r in order], budget)
        fed, counts, ids, parts = [], [], [], []
        for r, size in zip(order, sizes):
            if size > 0:
                need = blocks_for(r["fed"] + size, 16) - len(r["table"])
                r["table"], free = r["table"] + free[:need], free[need:]
                fed.append(r)
                counts.append(size)
                ids = ids + r["ids"][r["fed"]:r["fed"] + size]
                parts.append((r["table"], r["fed"], size))
        started = time.perf_counter()
        if len(ids) > 0:
            scores = run_batch(knobs, ids, parts, pools)
        log.append((step, [(r["name"], size) for r, size in zip(fed, counts)], round(time.perf_counter() - started, 2)))
        for i in range(len(fed)):
            r = fed[i]
            r["fed"] = r["fed"] + counts[i]
            if r["fed"] == len(r["ids"]):
                token = torch.argmax(scores[i]).item()
                if token == END_OF_TURN or len(r["ids"]) - r["prompt"] >= limit:
                    finished.append(r)
                    free = free + r["table"]
                    running = [other for other in running if other["name"] != r["name"]]
                else:
                    r["ids"] = r["ids"] + [token]
        step = step + 1
    return finished, log


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：引擎的一步

    有两个请求在运行，预算是 64，每块装 16 个词元。请求 A 已经读完提示，正在写：它的 `"fed"` 是 40，`"ids"` 装着 41 个词元，块表里有 3 块。请求 B 的提示有 100 个词元，已经喂了 40 个，块表里也有 3 块。

    下一步里，A 喂几个词元？B 呢？各自从空闲列表里拿几个新块？这一步结束时 B 能得到一个词元吗？（是填 1，否填 0。）
    """)
    return


@app.cell(hide_code=True)
def _():
    engine_boxes = hand.boxes(['喂给 A 的词元数', '喂给 B 的词元数', 'A 的新块数', 'B 的新块数', 'B 得到词元'])
    return (engine_boxes,)


@app.cell(hide_code=True)
def _(engine_boxes):
    hand.marked(engine_boxes, {'喂给 A 的词元数': 1, '喂给 B 的词元数': 60, 'A 的新块数': 0, 'B 的新块数': 4, 'B 得到词元': 1}, {
        '喂给 A 的词元数': r"""A 在解码，所以排在前面，喂它唯一没喂的那个词元""",
        '喂给 B 的词元数': r"""B 有 60 个没喂的词元，预算还剩 63""",
        'A 的新块数': r"""41 个词元装得进 3 个 16 词元的块""",
        'B 的新块数': r"""这一步之后 B 喂了 100 个词元：`blocks_for(100, 16)` 减去它已有的 3 块""",
        'B 得到词元': r"""它的 `"fed"` 到达了 `"ids"` 的长度，所以它最后一行分数会被用上""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 八个人同时提问

    八个问题一起到达。引擎先一个一个地回答，同时只允许运行一个请求，就像第 28 章那样。然后让八个请求一起共用它。单元格打印出各个回答、每个回答是否一个词元都不差，以及两种方式的秒数。大约要半分钟。
    """)
    return


@app.cell
def _(knobs, tokenizer):
    questions = [
        "What is the capital of France?", "Name three colours.", "What is 2 + 3?", "Write one sentence about the sea.",
        "Who wrote Romeo and Juliet?", "What do bees make?", "How many legs does a spider have?", "Name a planet.",
    ]
    _started = time.perf_counter()
    alone, _log = engine(knobs, [new_request(i, tokenizer.chat(questions[i]), 0) for i in range(len(questions))], 1, 64, 30)
    seconds_alone = time.perf_counter() - _started
    _started = time.perf_counter()
    together, together_log = engine(knobs, [new_request(i, tokenizer.chat(questions[i]), 0) for i in range(len(questions))], 8, 64, 30)
    seconds_together = time.perf_counter() - _started
    for _r in together:
        _same = [a["ids"] for a in alone if a["name"] == _r["name"]][0] == _r["ids"]
        print(_r["name"], _same, tokenizer.decode(_r["ids"][_r["prompt"]:]))
    print("一个一个回答的秒数：", round(seconds_alone, 1), "  一起回答：", round(seconds_together, 1))
    return questions, together_log


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在测试机器上，八个回答两种方式完全一样，一起回答用了 7.4 秒，一个一个回答用了 19 秒。回答按结束的先后列出：短回答先离开，空出它们的位置。

    日志记下了每一步发生的事，这里显示前四步和最后三步：步号、每个被喂的请求一对（请求，喂了几个词元），以及秒数。
    """)
    return


@app.cell
def _(together_log):
    for _step, _who, _seconds in together_log[:4] + together_log[-3:]:
        print(_step, _who, _seconds)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    第 0 步读了 64 个词元的提示：三个完整的提示，以及第四个的 9 个词元。到第 1 步，这三个已经在解码了，每个一个词元，同时还在读接下来的提示：解码和读提示共用这一步。从第 3 步起，八个请求一起解码，到最后只剩下两个最长的回答。

    ## 量一量长提示

    现在回到前面那种情况。四个问题在第 0 步开始。第 6 步有人带着一个 540 个词元的故事到达。引擎运行两次：一次预算大到从不切分，一次预算为 64。单元格显示两种情况下最长的一步，以及长请求的回答从哪一步开始。大约要 20 秒。
    """)
    return


@app.cell
def _(knobs, questions, tokenizer):
    story = (
        "Pip the fox lived at the edge of a big forest. Every morning Pip ran to the river to look at the fish. "
        "One day the river was gone, and only dry stones were left. Pip asked the old owl what had happened. "
        "The owl said that a beaver had built a dam far up the hill. "
    )
    long_question = "Here is a story. " + story * 8 + "Why was the river dry? Answer in one sentence."
    for _budget in [100000, 64]:
        _requests = [new_request(i, tokenizer.chat(questions[i]), 0) for i in range(4)]
        _requests = _requests + [new_request("story", tokenizer.chat(long_question), 6)]
        _done, _log = engine(knobs, _requests, 8, _budget, 30)
        _story = [r for r in _done if r["name"] == "story"][0]
        _longest = max([seconds for step, who, seconds in _log])
        _first = [step for step, who, seconds in _log if ("story", 1) in who][0]
        print("预算", _budget, "  提示", _story["prompt"], "个词元   最长的一步：", _longest, "秒   回答的第一个词元在第", _first, "步喂入")
        print("   ", tokenizer.decode(_story["ids"][_story["prompt"]:]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    测试机器上：不切分时，有一步用了 3.9 秒，这几秒里其他四个人看到的画面完全卡住不动。预算设为 64 时，最长的一步只用了 0.83 秒，长提示分 9 步读完，所以它的回答从第 15 步开始，而不是第 7 步。回答内容本身一字不差。

    这就是每个推理引擎都要权衡的旋钮：预算大，长提示能更快读完；预算小，其他人的回答就能持续流畅地输出，不会被长时间卡住。

    ## 你搭出了什么
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("到达", "随时"), ("等待", "按顺序"), ("运行", "≤ 上限"), ("一次运行", "≤ 预算"), ("词元", "每人一个")], ["排队", "接纳", "分块", "共用"], "引擎的循环：请求排队等待，被接纳，在预算内喂入，共用模型的一次运行")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这就是一个完整的推理引擎：把你写的调度器、分页 KV 缓存和词元预算，套在第 28 章带缓存的模型外面。真正的 vLLM 和同类推理引擎在工程上做了大量优化（主要是用定制 GPU 算子把每一步压到极致），但核心主循环就是你眼前这几十行。

    ## 接下来

    **第 30 章**是综合项目，你会亲手实现**推测解码（speculative decoding）**——让一个小模型先猜出几个词元，再由大模型一次前向计算批量验证——并在章末展望混合专家（MoE）模型与智能体循环（第 40–51 章）。

    ## 你现在会做的事

    - 解释批处理为什么能提高吞吐量，以及会付出怎样的延迟代价
    - 比较静态批处理和连续批处理，并算出浪费的槽位步
    - 解释分页 KV 缓存的核心组件：缓存块、内存池、块表和空闲列表
    - 在内存池里定位一个请求的任意词元，并按顺序读出它的完整缓存
    - 写一个按先来后到、槽位上限和空闲内存接纳请求的调度器
    - 在正在解码的请求和分块预填充的提示之间分配词元预算
    - 读懂引擎的运行日志，说出每一步发生了什么

    ---

    <!-- go-deeper -->
    **深入了解（选读）：**分页 KV 缓存出自 vLLM 论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention*（Kwon 等，2023）。连续批处理在 *Orca*（Yu 等，2022）中提出，分块预填充出自 *SARATHI*（Agrawal 等，2023）。
    真正的推理引擎还会让开头文本相同的请求共享缓存块，这叫作*前缀缓存（prefix caching）*。
    """)
    return


if __name__ == "__main__":
    app.run()
