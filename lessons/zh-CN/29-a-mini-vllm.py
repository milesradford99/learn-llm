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

    第 28 章让一个请求变快了。真正的服务从来不只有一个请求。人们随时发来问题：有的很短，有的贴了一长篇文档；有的只要一句话的回答，有的要一整页。
    在一台机器上回答所有这些问题的程序，叫作**推理引擎**（inference engine），也叫服务引擎。最有名的开源推理引擎叫 *vLLM*，这一章就用它的三个主要想法搭一个小的：

    1. **批处理**（batching）：几个请求共用模型的每一次运行；
    2. **分页 KV 缓存**（paged KV cache）：各个请求的缓存共用一个内存池，按小页划分；
    3. **分块预填充**（chunked prefill）：长提示分几块读，免得它把所有人都卡住。

    评价一个引擎看两样东西，它们互相拉扯。**吞吐量**（throughput）：所有人加在一起，每秒写出多少个词元。**延迟**（latency）：每个人要等多久，第 28 章把它量成了首个词元的等待时间，和之后每个词元的时间。

    **来自前面的章节。**下面折叠起来的单元格里是第 27 章的文件读取器和 `share_heads`，第 20 章的 `rms_norm`，第 28 章带 `start` 的 `rope` 和你写的 `attend_new`，以及第 22 章的 `time_it`。再下一个单元格和第 28 章一样，加载模型。
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
    ## 一次运行，多个请求

    第 28 章发现，解码一步之所以慢，是因为要把**一个**向量推过模型，每个旋钮都得从内存里取一次，一共 2.4 GB。运算倒是便宜的部分。

    那就在同一趟里推过好几个向量。五个请求都在等下一个词元，把它们的五个向量放进一个张量，每个一行，每个矩阵一次乘完这五行。旋钮取一次，用五次。这就是**批处理**。

    模型里只有一部分没法共用：注意力。每个请求的新词元只能看**它自己**前面的词元，通过它自己的缓存，绝不能看别人的文本。所以矩阵一起处理所有的行，注意力则对每个请求单独做。

    ## 动手算：共用一次运行的代价

    假设对一个请求来说，解码一步要 0.125 秒，全部花在取旋钮上。如果 8 个请求共用这一步，而且只有取旋钮这件事花时间，这一步要多久？引擎为这 8 个请求加起来每秒写出多少个词元？一个请求单独跑时又是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    share_boxes = hand.boxes(['8 个请求一步的秒数', '8 个一起时每秒词元数', '1 个单独时每秒词元数'], step=0.001)
    return (share_boxes,)


@app.cell(hide_code=True)
def _(share_boxes):
    hand.marked(share_boxes, {'8 个请求一步的秒数': 0.125, '8 个一起时每秒词元数': 64, '1 个单独时每秒词元数': 8}, {
        '8 个请求一步的秒数': r"""旋钮只取一次，不管有几行""",
        '8 个一起时每秒词元数': r"""每 0.125 秒 8 个词元：$8 / 0.125$""",
        '1 个单独时每秒词元数': r"""$1 / 0.125$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    这是理想情况。实际的收益要小一些，因为每多一行确实要多做一些运算，而注意力是每个请求单独做的，每多一个请求就多一份开销。下面几节先把引擎搭起来，再来测量它。

    ## 两种批处理

    请求不会一起到达，也不会一起结束：一个回答 3 个词元，另一个 200 个。最简单的批处理叫**静态批处理**（static batching）：拿一组请求，一直运行到它们**全部**结束，再拿下一组。
    在写那个长回答的时候，已经结束的请求的位置就空着。

    **连续批处理**（continuous batching）每一步都重新决定。一个请求结束了就立刻离开这一批，下一步就有一个等待的请求补上它的位置。这一批总是尽量满的。

    ## 动手算：空着的位置

    一批能装 4 个请求。它们的回答分别要 3、10、2、5 步。用静态批处理，这一组要跑多少步？一个**位置步**是一个位置用一步：这一组一共有多少个位置步，其中有多少个真正用来写了词元？占多大比例？
    """)
    return


@app.cell(hide_code=True)
def _():
    static_boxes = hand.boxes(['这一组的步数', '位置步', '用上的位置步', '用上的比例'], step=0.01)
    return (static_boxes,)


@app.cell(hide_code=True)
def _(static_boxes):
    hand.marked(static_boxes, {'这一组的步数': 10, '位置步': 40, '用上的位置步': 20, '用上的比例': 0.5}, {
        '这一组的步数': r"""最长的回答结束，这一组才结束""",
        '位置步': r"""这些步里每一步都有 4 个位置""",
        '用上的位置步': r"""每个请求占一个位置，占的步数就是它的回答要的步数：$3 + 10 + 2 + 5$""",
        '用上的比例': r"""用上的除以全部""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一半的位置是空的。下面的单元格数一数：40 个请求，回答长度在 5 到 60 个词元之间随机抽取，用 8 个位置，两种方式各要多少步。`continuous` 说明用哪种方式：为 `False` 时，只有这一批空了，新请求才能进来。
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


@app.cell
def _():
    answer_lengths = np.random.default_rng(29).integers(5, 61, 40).tolist()
    steps_needed(answer_lengths, 8, False), steps_needed(answer_lengths, 8, True), sum(answer_lengths) / 8
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    依次是：静态批处理的步数、连续批处理的步数，以及最少可能的步数（假如 8 个位置每一步都满着）：281、182 和 150。连续批处理离最少的步数近得多。每个真实的引擎都是这样做批处理的。

    （`running` 为每个请求记着它还要写多少个词元。每一步减一，减完了的请求就离开。）

    ## 缓存放在哪里

    批处理有代价。第 28 章里，每个请求的缓存用 `torch.cat` 变长，每次都要造一个全新的张量。很多请求在不同的时刻开始、结束，内存就被切成大大小小的碎片。
    引擎希望在开始时**一次**把内存留出来，然后按请求的需要分出去。

    一个请求该分多少？它的回答有多长，要到结束才知道。给每个请求都留出最长文本的空间，比如 4,096 个词元，按第 28 章每个词元 229,376 字节算，每个请求要 0.94 GB。八个请求就能把一台笔记本塞满，而其中大部分空间根本用不上。

    vLLM 的办法来自操作系统管理内存的方式。把内存切成大小相同的小**块**（block），这里每块 16 个词元；一个请求填满了最后一块，就再给它一块。所有的块都放在一个大张量里，叫作**池**（pool）。
    一个请求的各个块在池里不必挨在一起。每个请求有一张**块表**（block table）：按顺序列出它的各个块的编号。引擎还有一张空闲块的列表。一个请求结束时，它的块就回到这张列表上。

    这就是**分页 KV 缓存**。一个请求最多只有一个没装满的块，所以最多空着 15 个词元的位置。

    ## 动手算：块

    每块装 16 个词元。一个 300 个词元的请求需要几块？这些块里有几个词元的位置没用上？一个 32 个词元的请求呢？
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

    写 `blocks_for(length, block_size)`。它返回装下 `length` 个词元需要几个 `block_size` 个词元的块：`length` 除以 `block_size`，**向上**取整。0 个词元需要 0 块。

    `blocks_for(300, 16)` 必须得到 19，`blocks_for(32, 16)` 必须得到 2。

    第 11 章的 `//` 是向下取整。向上取整的一个办法：先给 `length` 加上 $\text{块大小} - 1$。
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

    模型的一个块对应的池，是一个形状为（块数、头数、块大小、数的个数）的张量：这一章里，Qwen 的是 (160, 8, 16, 128)。键和值各有自己的池，模型的 28 个块每个都有自己的一对。

    注意力要的键，得是第 28 章里那个样子：(1, 头数, 词元数, 数的个数)，按顺序排好。所以引擎从池里把这个请求的各块收集起来，沿着词元那一维拼接，再把末尾没用上的位置切掉。

    /// admonition | 🧰 工具箱：张量的列表，以及在最前面加一维
    - `[pool[b] for b in [4, 1]]` 是两个张量组成的列表：先是池里的第 4 块，再是第 1 块。每个的形状都是（头数、块大小、数的个数）。
    - `torch.cat` 可以拼接任意长度的列表，和第 14 章一样：`torch.cat([a, b, c], dim=1)`。
    - `x[None]` 在最前面加一个大小为 1 的维度，就像第 20 章的 `v[:, None]` 在第二个位置加一维：形状为 (8, 40, 128) 的张量变成 (1, 8, 40, 128)。
    ///

    ## 练习 3：read_cache

    写 `read_cache(pool, table, length)`。`pool` 的形状是（块数、头数、块大小、数的个数），`table` 是一个请求的块表，`length` 是这个请求到目前为止有多少个词元。
    它按顺序返回这个请求的键（或值），形状是（1、头数、length、数的个数）。

    每块 2 个词元、块表 `[3, 0]`、`length` 为 3 时，结果先是池里第 3 块的 2 个词元，再是池里第 0 块的第一个词元。
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
    ## 一整批的模型

    下面是第 28 章的 `block_step`，改写成处理一整批。所有请求的所有新词元排成一行 `x`，形状是 (1, 新词元总数, 1,024)：先是第一个请求的词元，再是第二个的，依此类推。`parts` 说明每个请求的词元在哪里：每个请求一个三元组 `(table, seen, new)`。

    矩阵一次处理所有词元。然后一个请求一个请求地：用你的 `slot` 把新的键和值写进池里，用你的 `read_cache` 读出这个请求的整个缓存，再用你的 `attend_new` 把它混合起来。
    `pools` 为模型的每个块存着一对 `(K, V)`。
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
    `run_batch` 跑完全部 28 个块，为每个请求返回一行分数：它最后一个新词元之后的分数。`new_pools` 一次性留出内存：160 个块，每块 16 个词元，模型的 28 个块，键和值各一份：0.59 GB。
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
    测试机器上：一个请求每秒 7.5 个词元，8 个请求 22 个，16 个请求 35 个。没有达到理想的 16 倍，但同一台电脑白白多出了 4 倍多。每一步的时间增长得很慢，所以每个人的词元来得稍慢一点：吞吐量是用一点延迟换来的。

    （收益比理想情况小，部分原因是这一页用的是 Python：`block_batch` 要逐个请求、逐个新词元地循环。真正的引擎用融合运算来做这些事，那是第 28 章的说法。）

    ## 放请求进来

    现在来看引擎的循环。每一步，它做四件事：

    1. 已经到达的请求加入**等待**列表；
    2. 把等待的请求**接纳**进**运行**列表，能接纳多少就接纳多少；
    3. 决定每个运行中的请求喂多少个词元；
    4. 在所有这些词元上把模型运行一次，输入用完了的请求各得到它的下一个词元。

    决定谁可以运行的那部分叫作**调度器**（scheduler）。我们的调度器有两个限制：同时运行的请求最多 `max_running` 个，而且空闲的块要装得下新请求的提示。请求按到达的顺序来：如果排在最前面的等待请求放不下，后面的也都得等。

    一个请求是一个字典：`"ids"` 装着它到目前为止的词元，包括提示。

    ## 练习 4：admit

    写 `admit(waiting, running, free_blocks, max_running, block_size)`。`waiting` 和 `running` 是请求的列表，`free_blocks` 是空闲块的个数。它返回一对东西：新的等待列表和新的运行列表。

    只要还有等待的请求、运行列表里少于 `max_running` 个、并且**第一个**等待请求的提示装得进空闲的块：就把这个请求移到运行列表的末尾，并把它的块算作不再空闲。

    3 个 16 词元的空闲块、没有请求在运行、最多 4 个、等待的提示分别是 20、10、5 个词元时：前两个被接纳（先用 2 块，再用 1 块），第三个得等。
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
    ## 引擎

    下面就是循环的四件事，里面用到你写的三个函数。`new_request` 造出一个请求。`"fed"` 数着模型已经读过的词元，`"table"` 是块表，`"arrives"` 是请求到达的那一步。
    请求必须按到达的顺序排列。

    每一步之前，运行中的请求排好顺序，正在解码的在前；一个请求需要新块时，从空闲列表的最前面拿。一个请求结束时，它的块回到列表上。`log` 为每一步记下谁喂了多少个词元，以及这一步花了多久。

    （真正的引擎还要处理块用完的情况。这时 vLLM 会暂停最后来的那个请求，释放它的块，过后再重新算出它的缓存。这个引擎的池对这一页来说足够大。）
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
    测试机器上：不切分时，有一步用了 3.9 秒，这几秒里四个人什么动静都看不到。预算为 64 时，最长的一步用了 0.83 秒，长提示分 9 步读完，所以它的回答从第 15 步开始，而不是第 7 步。回答本身是一样的。

    这就是每个引擎都要调的权衡：预算大，长提示读得快；预算小，别人的词元流得顺。

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
    这就是一个推理引擎：你的调度器、你的分页缓存、你的预算，围绕着第 28 章那个带缓存的模型。vLLM 和同类引擎加了大量工程，主要是为了让每一步在 GPU 上都快起来，但它们的循环就是这一个。

    ## 接下来

    **第 30 章**是选做的综合项目，由你选择再走一步：推测解码（speculative decoding），让一个小模型猜几个词元，大模型一次运行把它们检查完；混合专家（mixture-of-experts）模型；或者在你自己的引擎上跑一个小小的智能体循环。

    ## 你现在会做的事

    - 解释批处理为什么能提高吞吐量，又要付出多少延迟
    - 比较静态批处理和连续批处理，并数出空着的位置
    - 解释分页 KV 缓存：块、池、块表和空闲列表
    - 在池里找到一个请求的任何一个词元，并按顺序读出一个请求的缓存
    - 写一个按顺序、按名额、按空闲内存接纳请求的调度器
    - 在正在解码的请求和分块读入的提示之间分配词元预算
    - 读懂引擎的日志，说出每一步做了什么

    ---

    <!-- go-deeper -->
    **深入了解（选读）：**分页 KV 缓存出自 vLLM 的论文 *Efficient Memory Management for Large Language Model Serving with PagedAttention*（Kwon 等，2023）。连续批处理在 *Orca*（Yu 等，2022）中提出，分块预填充出自 *SARATHI*（Agrawal 等，2023）。
    引擎还会让开头文本相同的请求共用块，这叫作*前缀缓存*（prefix caching）。
    """)
    return


if __name__ == "__main__":
    app.run()
