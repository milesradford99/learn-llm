import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="28 · 让单个请求变快")

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
    # 28 · 让单个请求变快

    第 27 章里，一个真实的模型通过你自己的代码回答了你。答案是对的，做法却很浪费：每写一个词元，它都让全部 28 个块把整段文本重新算一遍。

    这一章要让一个请求变快。一个**请求**（request）就是一个人提的一个问题，从问题到达算起，到回答写完为止。它给人的快慢感觉由两个数决定：

    - **首个词元的等待时间**（time to first token）：你对着空白的回答要等多久；
    - 之后的**每秒词元数**（tokens per second）：回答增长得有多快。

    这两个数你都会改进，而回答里的词元一个也不变。然后你会把模型的矩阵缩小到原来四分之一的字节数，看看这在你的电脑上带来了什么，又没带来什么。

    **来自前面的章节。**下面折叠起来的单元格里是第 27 章的代码：你的文件读取器、`share_heads`、块、`qwen_scores` 和 `generate`，还有第 20 章的 `rms_norm`、`rope` 和 `attend`，以及第 22 章的 `time_it`。
    有一处改动：`rope` 多了第三个输入 `start`，也就是交给它的第一个词元的位置。到现在为止它一直是 0。后面有一节会用到它。

    下一个单元格加载你在第 27 章下载的模型。
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
def attend(Q, K, V):
    words = Q.shape[-2]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones(words, words, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


@app.function(hide_code=True)
def block(x, L):
    words = x.shape[1]
    y = rms_norm(x, L["norm1"])
    q = (y @ L["q"].T).reshape(1, words, 16, 128).transpose(1, 2)
    k = (y @ L["k"].T).reshape(1, words, 8, 128).transpose(1, 2)
    v = (y @ L["v"].T).reshape(1, words, 8, 128).transpose(1, 2)
    q = rope(rms_norm(q, L["q_norm"]), 1000000, 0)
    k = rope(rms_norm(k, L["k_norm"]), 1000000, 0)
    blended = attend(q, share_heads(k, 2), share_heads(v, 2))
    x = x + blended.transpose(1, 2).reshape(1, words, 2048) @ L["out"].T
    y = rms_norm(x, L["norm2"])
    return x + (F.silu(y @ L["gate"].T) * (y @ L["up"].T)) @ L["down"].T


@app.function(hide_code=True)
def qwen_scores(knobs, ids):
    with torch.no_grad():
        x = knobs["model.embed_tokens.weight"][torch.tensor([ids])]
        for layer in range(28):
            x = block(x, layer_knobs(knobs, layer))
        x = rms_norm(x, knobs["model.norm.weight"])
        return x[0, -1] @ knobs["model.embed_tokens.weight"].T


@app.function(hide_code=True)
def generate(next_scores, ids, limit, stop):
    for turn in range(limit):
        token = torch.argmax(next_scores(ids)).item()
        if token == stop:
            return ids
        ids = ids + [token]
    return ids


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
    ## 量一量浪费了多少

    先看看：对不同长度的文本，模型运行一次要多久？下面的单元格在一个故事的前 25、50、100 和 200 个词元上运行 `qwen_scores`。
    """)
    return


@app.cell
def _(knobs, tokenizer):
    story = (
        "Pip the fox lived at the edge of a big forest. Every morning Pip ran to the river to look at the fish. "
        "One day the river was gone, and only dry stones were left. Pip asked the old owl what had happened. "
        "The owl said that a beaver had built a dam far up the hill."
    )
    story_ids = tokenizer.encode(story + " " + story + " " + story + " " + story)
    qwen_scores(knobs, story_ids[:5])
    for _length in [25, 50, 100, 200]:
        _started = time.perf_counter()
        qwen_scores(knobs, story_ids[:_length])
        print(_length, "个词元：", round(time.perf_counter() - _started, 2), "秒")
    return (story,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    测试机器上是 0.28、0.36、0.66 和 1.07 秒。文本越长，运行一次就越久。两者并不完全成正比，原因后面有一节会讲，但从 100 个词元起，文本长一倍，时间也差不多长一倍。

    再回想一下 `generate` 是怎么工作的。它在整段文本上运行模型，得到一个词元，把这个词元加上去，然后又在整段文本上运行模型。

    ## 动手算：数一数趟数

    一个词元走完全部 28 个块，叫作一**趟**（pass）。在 10 个词元上运行一次，就是 10 趟。

    一段提示有 10 个词元，`generate` 在它后面写了 4 个词元。所以它运行了 4 次模型：先在 10 个词元上，然后是 11、12、13 个。第二次运行是多少趟？4 次运行一共多少趟？
    在这 4 次运行里，有多少趟是花在模型**从没**处理过的词元上的？
    """)
    return


@app.cell(hide_code=True)
def _():
    pass_boxes = hand.boxes(['第二次运行的趟数', '4 次运行的总趟数', '花在新词元上的趟数'])
    return (pass_boxes,)


@app.cell(hide_code=True)
def _(pass_boxes):
    hand.marked(pass_boxes, {'第二次运行的趟数': 11, '4 次运行的总趟数': 46, '花在新词元上的趟数': 13}, {
        '第二次运行的趟数': r"""提示的 10 个词元，加上已经写出的 1 个""",
        '4 次运行的总趟数': r"""$10 + 11 + 12 + 13$""",
        '花在新词元上的趟数': r"""第一次运行的 10 个词元都是新的；之后每次运行只有 1 个新词元，就是刚写出的那个""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    46 趟，其中只有 13 趟在处理新东西。另外 33 趟都在重复已经做过的工作。文本越长情况越糟：提示 100 个词元、回答 20 个词元时，是 2,190 趟对 119 趟。

    下面来真的：一个关于这个故事的问题，连同外面的对话模板一共 92 个词元。这个单元格要跑一会儿，而这正是要说明的问题。
    """)
    return


@app.cell
def _(knobs, story, tokenizer):
    def next_scores(ids):
        return qwen_scores(knobs, ids)

    question_ids = tokenizer.chat("Here is a story. " + story + " Why was the river dry? Answer in one sentence.")
    _started = time.perf_counter()
    slow_ids = generate(next_scores, question_ids, 40, END_OF_TURN)
    slow_seconds = time.perf_counter() - _started
    len(question_ids), tokenizer.decode(slow_ids[len(question_ids):]), len(slow_ids) - len(question_ids), round(slow_seconds, 1)
    return question_ids, slow_ids, slow_seconds


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    依次是：问题的长度、回答、回答有多少个词元、花了多少秒。测试机器上是 14 个词元用了 9.2 秒。这是 15 次运行，每个词元一次，再加上给出停止词元的那一次：一共 1,485 趟，而最后整段文本只有 106 个词元。记住这个秒数。

    ## 什么东西永远不变

    重复的那些趟能不能省掉？只有当重新算一遍的结果总和上次一样时才行。事实正是如此，原因就是第 16 章的掩码。

    掩码让一个词元只能看自己和前面的词元，绝不能看后面的。所以在第 0 个块里，第 3 号词元得到的结果只由第 0 到 3 号词元算出，别无其他。第 1 个块在这些结果上工作，用的是同一张掩码，所以它给第 3 号词元的结果仍然只由第 0 到 3 号词元算出。依此类推，一直到 28 个块全部走完。

    **模型为一个词元算出的任何东西，都不依赖它后面的词元。**在末尾加上一个词元时，前面那些词元在每个块里算出的每个数，都和原来一模一样。可 `generate` 还是把它们全部重算了一遍。

    ## 新词元需要什么

    那就只让新词元走一遍这些块。它需要从前面的词元那里拿什么？把第 27 章的 `block` 逐行看一遍，每一行都问一句：新词元的结果用到别的词元的数了吗？

    - `rms_norm`：没有，每个词元的向量各自归一化；
    - 造出它的查询、键和值的那些矩阵：没有；
    - 注意力：**用到了**。新词元的查询要和到目前为止所有词元的**键**比较，结果是把它们的**值**混合起来；
    - 注意力之后的矩阵，以及 SwiGLU：没有。

    只有一行往旁边看，它向前面每个词元要两样东西：这个词元在这个块里的键和值。它们永远不变。所以每个块把它们留下来：每个词元的键和值，在这个词元经过时存一次。这个存放处叫 **KV 缓存**（KV cache），K 指键（key），V 指值（value）。
    （“缓存”也是第 22 章里处理器旁边那块快速内存的名字。在这里它只表示一个存放结果的地方，免得再算一遍。）

    ## 只为最新的词元做注意力

    现在注意力里的查询比键少。`Q` 只装新词元的查询，`K` 和 `V` 装到目前为止每个词元的键和值，新的排在最后。分数仍然是 $Q K^\top / \sqrt{d}$，这张表每个新词元一行，到目前为止的每个词元一列。

    掩码的形状也跟着变。规则不变：一个词元可以看自己，以及它前面的一切。

    ## 动手算：谁可以看哪里

    到目前为止有 5 个词元，编号 0 到 4，其中最后 2 个是新的。掩码有 2 行 5 列。第 0 行属于几号词元？这个词元可以看 5 个键里的几个？整张掩码一共挡住几格？
    """)
    return


@app.cell(hide_code=True)
def _():
    mask_boxes = hand.boxes(['第 0 行是几号词元', '第 0 行能看几个键', '掩码挡住的格数'])
    return (mask_boxes,)


@app.cell(hide_code=True)
def _(mask_boxes):
    hand.marked(mask_boxes, {'第 0 行是几号词元': 3, '第 0 行能看几个键': 4, '掩码挡住的格数': 1}, {
        '第 0 行是几号词元': r"""新词元是最后两个：3 号和 4 号""",
        '第 0 行能看几个键': r"""它自己和它前面的一切：0、1、2、3 号词元""",
        '掩码挡住的格数': r"""第 0 行不能看 4 号词元；第 1 行是 4 号词元，也就是最后一个，什么都可以看""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：`torch.triu` 用在任意的表上
    第 16 章把 `torch.triu(table, diagonal=1)` 用在一张方形的表上。它对任意的表都能用，`diagonal` 也可以是任意整数：`diagonal=2` 留下的是列号至少比行号大 2 的那些格子。
    `torch.triu(torch.ones(2, 4), diagonal=2)` 的两行是 `[0, 0, 1, 1]` 和 `[0, 0, 0, 1]`。
    ///

    ## 练习 1：attend_new

    写 `attend_new(Q, K, V)`。`Q` 的形状是（批、头、新词元数、数的个数）：最新那些词元的查询。`K` 和 `V` 的形状是（批、头、词元总数、数的个数）：到目前为止所有词元的键和值，最新的排在最后。
    它返回每个新词元的混合结果，形状是（批、头、新词元数、数的个数）。

    它就是第 16 章的注意力，只有一处不同：掩码。掩码有 `new` 行、`total` 列。第 $i$ 行属于第 $\text{total} - \text{new} + i$ 号词元，编号比它大的键全部挡住。

    5 个词元里有 2 个是新的时，掩码就是你动手算的那一张：只挡住一格，在第 0 行的末尾。只有 1 个新词元时什么都不挡。`new` 等于 `total` 时，结果和第 16 章的一样。
    """)
    return


@app.function
def attend_new(Q, K, V):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "attend_new", attend_new, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""打开页面顶部附近折叠着的 `attend`：从它出发。有两处要改。掩码不再是方形，而是 `new` 行 `total` 列；它的 `diagonal` 也不再是 1。在你动手算的那题里，第 0 行第一个被挡住的格子在第 4 列：那里需要的 diagonal 就是这个数。""",
        r"""一共 4 行。第一行是 `new, total = Q.shape[-2], K.shape[-2]`。分数和最后一行都和 `attend` 里一样。掩码是对 `torch.ones(new, total, dtype=torch.bool)` 做 `torch.triu`，其中 `diagonal=total - new + 1`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 把键和值留下来

    每个块的缓存是一对 `(K, V)`。两个张量的形状都是 (1, 8, seen, 128)：1 段文本，8 个键头，到目前为止见过的每个词元占一项，每项 128 个数。存的是 8 个头的版本，也就是 `share_heads` 把它们复制成 16 个头之前的样子。复制品不必留着。

    还没喂进任何词元时，`seen` 是 0。PyTorch 允许这样：`torch.zeros(1, 8, 0, 128)` 是一个不装任何数、却仍然有形状的张量，`torch.cat` 可以把真正的键接在它后面。所以缓存不用为第一个词元做特殊处理。

    ## 练习 2：remember

    写 `remember(cache, k, v)`。`cache` 是一对张量 `(K, V)`，形状是（批、头、已见词元数、数的个数）。`k` 和 `v` 是最新那些词元的键和值，形状是（批、头、新词元数、数的个数）。
    它返回新的一对：`K` 沿着词元那一维在后面接上 `k`，`V` 用同样的方式接上 `v`。这时两者的形状都是（批、头、已见词元数 + 新词元数、数的个数）。

    缓存里有 2 个词元、新来 1 个时，你返回的这一对各装着 3 个词元，新的排在最后。
    """)
    return


@app.function
def remember(cache, k, v):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "remember", remember, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`cache[0]` 是 `K`，`cache[1]` 是 `V`。第 14 章的 `torch.cat` 沿着某一维把张量拼接起来。在（批、头、已见词元数、数的个数）里，数词元的是第几个位置？""",
        r"""只要一行，返回两样东西：`torch.cat([cache[0], k], dim=2)`，一个逗号，再对 `cache[1]` 和 `v` 做同样的事。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 词元得知道自己的位置

    还剩一个陷阱。RoPE 把每个查询和键转过一个角度，这个角度取决于词元在文本里的位置。第 27 章的 `rope` 每次拿到的都是整段文本，所以它从 0 开始给词元编号。
    如果只把一个新词元单独交给它，它也会把这个词元当成 0 号，模型读到的文本就好像最新的那个词站在最开头。

    所以 `rope` 现在多了 `start`：交给它的第一个词元的位置。缓存正好知道这个位置。如果缓存里有 12 个词元，编号 0 到 11，下一个就是 12 号。

    ## 一个带记忆的块

    下面还是第 27 章的块，改成了只处理新词元。`x` 装着新词元的向量，`cache` 是这个块的那一对。它返回两样东西：新词元的向量，以及装进了新词元的缓存。

    和矩阵相乘现在写成 `product(y, M)`，其中 `product` 是作为输入交进来的一个函数。眼下它总是 `plain`，也就是第 27 章的 `y @ M.T`。后面有一节会交进来另一个。
    """)
    return


@app.function
def plain(y, M):
    return y @ M.T


@app.function
def block_step(x, L, cache, product):
    new, seen = x.shape[1], cache[0].shape[2]
    y = rms_norm(x, L["norm1"])
    q = product(y, L["q"]).reshape(1, new, 16, 128).transpose(1, 2)
    k = product(y, L["k"]).reshape(1, new, 8, 128).transpose(1, 2)
    v = product(y, L["v"]).reshape(1, new, 8, 128).transpose(1, 2)
    q = rope(rms_norm(q, L["q_norm"]), 1000000, seen)
    k = rope(rms_norm(k, L["k_norm"]), 1000000, seen)
    cache = remember(cache, k, v)
    blended = attend_new(q, share_heads(cache[0], 2), share_heads(cache[1], 2))
    x = x + product(blended.transpose(1, 2).reshape(1, new, 2048), L["out"])
    y = rms_norm(x, L["norm2"])
    return x + product(F.silu(product(y, L["gate"])) * product(y, L["up"]), L["down"]), cache


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：跟着一个词元走一遍

    一段 19 个词元的提示已经喂进去了，现在再单独喂进一个词元。针对这次调用读一遍 `block_step`。`new` 和 `seen` 各是多少？`rope` 给这个词元的位置是几？
    `remember` 之后，`cache[0]` 里有几个词元，有几个头？`attend_new` 拿到的 `K` 又有几个头？
    """)
    return


@app.cell(hide_code=True)
def _():
    step_boxes = hand.boxes(['new', 'seen', 'rope 给的位置', 'remember 之后 cache[0] 里的词元数', 'cache[0] 的头数', 'attend_new 里 K 的头数'])
    return (step_boxes,)


@app.cell(hide_code=True)
def _(step_boxes):
    hand.marked(step_boxes, {'new': 1, 'seen': 19, 'rope 给的位置': 19, 'remember 之后 cache[0] 里的词元数': 20, 'cache[0] 的头数': 8, 'attend_new 里 K 的头数': 16}, {
        'new': r"""`x` 里只有喂进去的那一个词元""",
        'seen': r"""这次调用之前缓存里有的东西：那段提示""",
        'rope 给的位置': r"""提示里的词元是 0 到 18 号""",
        'remember 之后 cache[0] 里的词元数': r"""提示里的那些，加上新的这一个""",
        'cache[0] 的头数': r"""看 `k` 的 reshape""",
        'attend_new 里 K 的头数': r"""`share_heads(cache[0], 2)`""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    整个模型就是 28 个这样的块，每个块有自己的缓存。`empty_caches` 造出开始时用的 28 对空缓存。`qwen_step` 让一些新词元走过所有的块，返回它们之后那个词元的分数，以及 28 个缓存现在的样子。
    """)
    return


@app.function
def empty_caches():
    return [(torch.zeros(1, 8, 0, 128), torch.zeros(1, 8, 0, 128)) for layer in range(28)]


@app.function
def qwen_step(knobs, new_ids, caches, product):
    with torch.no_grad():
        x = knobs["model.embed_tokens.weight"][torch.tensor([new_ids])]
        grown = []
        for layer in range(28):
            x, cache = block_step(x, layer_knobs(knobs, layer), caches[layer], product)
            grown.append(cache)
        x = rms_norm(x, knobs["model.norm.weight"])
        return x[0, -1] @ knobs["model.embed_tokens.weight"].T, grown


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    记忆管用吗？做个测试：把关于故事的那个问题一次喂完，再分两段喂，先 60 个词元，然后是剩下的。如果你的 `attend_new` 和 `remember` 没错，两种喂法得到的最后一个块的键是一样的。
    单元格显示它们的形状，以及两者之间最大的差（`torch.abs` 把每个数变成正数，和第 09 章的 `np.abs` 一样）。
    """)
    return


@app.cell
def _(knobs, question_ids):
    _scores, at_once = qwen_step(knobs, question_ids, empty_caches(), plain)
    _scores, first_part = qwen_step(knobs, question_ids[:60], empty_caches(), plain)
    _scores, in_two_parts = qwen_step(knobs, question_ids[60:], first_part, plain)
    at_once[27][0].shape, torch.max(torch.abs(at_once[27][0] - in_two_parts[27][0])).item()
    return (at_once,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    92 个词元的键，差是 0，或者是某个远小于千分之一的数，那只是舍入误差。如果差很大，再检查一下你的掩码，以及 `remember` 里的先后顺序。

    ## 带着记忆写

    现在来写循环。第 27 章的 `generate` 每次都把全部内容交给模型。新的循环只交模型没见过的东西：第一次是整段提示，之后每次只是刚加上的那一个词元。

    ## 练习 3：generate_cached

    写 `generate_cached(step, caches, ids, limit, stop)`。`step` 是一个函数：`step(new_ids, caches)` 把一列新的词元编号喂给带着这些缓存的模型，返回一对东西：下一个词元的分数，以及装进了这些词元的缓存。
    `caches` 是开始时的缓存，`ids` 是开始时的词元编号列表。

    最多重复 `limit` 次：调用 `step`，取分数最高的词元；如果它是 `stop`，就立刻结束，不把它加进去；否则把它加到编号列表里。和 `generate` 一样，返回完整的列表，包括开头的部分。

    第一次调用 `step` 时，交给它整个 `ids` 和你拿到的缓存。之后每次调用，交给它一个只装着刚加上的那个词元的列表，以及上一次调用返回的缓存。

    有一项检查用的是一个会数数的假模型：它最想写的词元，是到目前为止喂给它的词元个数加 1。从 `[1, 2, 3]` 开始、`stop = 7` 时，你必须返回 `[1, 2, 3, 4, 5, 6]`，而 `step` 被喂的必须依次是 `[1, 2, 3]`、`[4]`、`[5]` 和 `[6]`。
    """)
    return


@app.function
def generate_cached(step, caches, ids, limit, stop):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "generate_cached", generate_cached, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""从第 27 章的 `generate` 出发，它折叠在页面顶部附近。你还需要一个名字，记着下一次要喂什么。循环之前，它是整个 `ids`。`scores, caches = step(..., caches)` 把两个结果都接住，并让变长了的缓存留在同一个名字下，供下一轮使用。""",
        r"""一共 8 行。循环之前：`new = ids`。循环里：`scores, caches = step(new, caches)`，然后用 `torch.argmax(scores).item()` 得到词元，接着是停止的判断和提前的 `return ids`，再是 `ids = ids + [token]`，最后是 `new = [token]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在用你的循环来回答和刚才一样的问题。单元格把 `qwen_step` 包在一个函数里，这个函数还会把每次调用花的时间记进列表 `step_seconds`。
    """)
    return


@app.cell
def _(knobs, question_ids, slow_ids, slow_seconds, tokenizer):
    step_seconds = []

    def timed_step(new_ids, caches):
        started = time.perf_counter()
        result = qwen_step(knobs, new_ids, caches, plain)
        step_seconds.append(time.perf_counter() - started)
        return result

    fast_ids = generate_cached(timed_step, empty_caches(), question_ids, 40, END_OF_TURN)
    print(tokenizer.decode(fast_ids[len(question_ids):]))
    print("词元和之前完全一样：", fast_ids == slow_ids)
    print("之前的秒数：", round(slow_seconds, 1), " 现在：", round(sum(step_seconds), 1))
    print("第一次调用：", round(step_seconds[0], 2), "秒；之后每次调用：", round(sum(step_seconds[1:]) / len(step_seconds[1:]), 3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    回答一样，一个词元都不差。测试机器上用了 2.6 秒，原来是 9.2 秒；回答每多一个词元，差距就再拉大一点。你的秒数会不同，但词元不该不同。

    最后一行说明，一个请求现在分成两个阶段，它们各有名字：

    - **预填充**（prefill）：第一次调用，读完整段提示并把缓存填好。92 个词元用了 0.57 秒。这就是首个词元的等待时间。
    - **解码**（decode）：之后的每次调用，一次一个词元。每次用了 0.14 秒，大约每秒 7 个词元；写到第 1,000 个词元时和写到第 100 个时差不多一样快，只是注意力里那一行分数变长了，会多花一点点。

    每个为语言模型提供服务的引擎都是这样工作的，也都会报告这两个数。

    ## 缓存的代价

    速度是用内存换来的：只要请求还没结束，缓存就得为每个词元、在每个块里留着一个键和一个值。

    ## 动手算：缓存有多大

    为 Qwen 算一算：28 个块，8 个键头，每个头 128 个数，键和值各一份。**一个块**为一个词元留多少个数？所有块为一个词元留多少个数？每个数 32 位，那是多少字节？
    如果 16 个头都有自己的键和值，又是多少字节？
    """)
    return


@app.cell(hide_code=True)
def _():
    size_boxes = hand.boxes(['每个词元在一个块里的数', '每个词元在所有块里的数', '每个词元的字节数', '16 个键头时每个词元的字节数'])
    return (size_boxes,)


@app.cell(hide_code=True)
def _(size_boxes):
    hand.marked(size_boxes, {'每个词元在一个块里的数': 2048, '每个词元在所有块里的数': 57344, '每个词元的字节数': 229376, '16 个键头时每个词元的字节数': 458752}, {
        '每个词元在一个块里的数': r"""一个键有 $8 \times 128$ 个数，一个值也一样大""",
        '每个词元在所有块里的数': r"""一个块的 28 倍""",
        '每个词元的字节数': r"""32 位是 4 个字节""",
        '16 个键头时每个词元的字节数': r"""头多一倍，数也多一倍""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个词元大约 0.23 MB。下面的单元格数出那个 92 个词元的问题的缓存里一共有多少个数，再除以 92。
    """)
    return


@app.cell
def _(at_once, question_ids):
    cache_numbers = sum([K.numel() + V.numel() for K, V in at_once])
    cache_numbers, cache_numbers / len(question_ids)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    一场 4,000 个词元的对话，对这个小模型来说，要留着大约 0.9 GB 的键和值。第 27 章里共用的头在这里见了成效：要是有 16 个键头，就是它的两倍。
    这也是下一章存在的原因。一个人的缓存放在哪里都行。一千个人的缓存，在不同的时刻变大、结束，就必须有人来管理。

    ## 为什么一个词元还要十分之一秒

    解码一个词元现在只是一个很小的计算：一个向量经过每个矩阵。为什么它还要 0.14 秒？

    原因是第 22 章学到的那件事：处理器做计算，比它从主内存里取数快得多。要让一个向量走完整个模型，每个旋钮都得取一次：5.96 亿个数，每个 4 字节，一共 2.4 GB，每个词元都要来一遍。
    在这些数上做的运算很便宜。花时间的是搬运它们。这也解释了第一次测量里，为什么 200 个词元的时间不是 25 个词元的 8 倍：运行一次，旋钮只取一遍，不管有多少个词元用到它们。

    来检验这个说法：同一个向量和 Qwen 的同一个矩阵相乘，一次全用 32 位，一次全用 64 位。运算是一样的。不同的只有字节数。
    """)
    return


@app.cell
def _(knobs):
    W = knobs["model.layers.0.mlp.gate_proj.weight"]
    x = torch.randn(1, 1024)
    W64, x64 = W.double(), x.double()

    def in_32_bits():
        return plain(x, W)

    def in_64_bits():
        return plain(x64, W64)

    W.shape, round(1000 * time_it(in_32_bits, 30), 2), round(1000 * time_it(in_64_bits, 30), 2)
    return W, in_32_bits, x


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    依次是：矩阵的形状，然后是用 32 位和用 64 位做一次乘法的毫秒数（`.double()` 把张量变成 64 位，就像 `.float()` 把它变成 32 位）。测试机器上是 0.26 和 0.54。字节多一倍，时间也多一倍。

    所以要让解码更快，就要**用更少的字节存每个旋钮**。第 22 章的 16 位格式是一步。部署服务时通常迈得更大：8 位的整数，甚至 4 位。

    ## 把旋钮存成小整数

    一个字节能装一个 −128 到 127 的整数。旋钮则是 0.0731 或 −0.4125 这样的数。要把它存进一个字节，先选一个很小的数，叫作**缩放系数**（scale），然后存下这个旋钮相当于多少个缩放系数，四舍五入成整数：

    > $$q = \text{round}\left(\frac{w}{\text{scale}}\right) \qquad\qquad w \approx q \times \text{scale}$$

    这就是**量化**（quantization）：$w$ 是旋钮，$q$ 是存下来的整数。因为取了整，读回来的只是旋钮的近似值。误差最多是半个缩放系数。

    缩放系数的选法，是让最大的旋钮正好落在最大的整数上。用 $b$ 位时，我们用 $-(2^{b-1} - 1)$ 到 $2^{b-1} - 1$ 之间的整数，正负两边一样远：8 位是 −127 到 127，4 位是 −7 到 7。所以

    > $$\text{scale} = \frac{\text{这些旋钮里最大的大小}}{2^{b-1} - 1}$$

    这里一个旋钮的大小，指的是去掉正负号之后的旋钮。

    ## 动手算：量化三个旋钮

    三个旋钮 1.4、−0.66 和 0.25，要用 4 位存成 −7 到 7 的整数。缩放系数是多少？−0.66 和 0.25 各存成哪个整数？把这两个再读回来，得到的旋钮各是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    quant_boxes = hand.boxes(['缩放系数', '−0.66 存成', '0.25 存成', '−0.66 读回来', '0.25 读回来'], step=0.1)
    return (quant_boxes,)


@app.cell(hide_code=True)
def _(quant_boxes):
    hand.marked(quant_boxes, {'缩放系数': 0.2, '−0.66 存成': -3, '0.25 存成': 1, '−0.66 读回来': -0.6, '0.25 读回来': 0.2}, {
        '缩放系数': r"""最大的大小是 1.4，最大的整数是 7""",
        '−0.66 存成': r"""$-0.66 / 0.2 = -3.3$，再取整""",
        '0.25 存成': r"""$0.25 / 0.2 = 1.25$，再取整""",
        '−0.66 读回来': r"""存下的整数乘以缩放系数""",
        '0.25 读回来': r"""存下的整数乘以缩放系数""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    旋钮 0.25 读回来成了 0.2。只有 15 个整数可选，台阶就很粗。

    **每一行一个缩放系数。**一个矩阵里的旋钮大小相差很远。如果整个矩阵共用一个缩放系数，它就由某一个大旋钮决定，成千上万个小旋钮全都会取整成 0 或 ±1。
    所以矩阵的每一**行**都有自己的缩放系数，由这一行里最大的旋钮决定。一行就是造出一个输出数的那组旋钮。代价是每行多存一个数，而一行有一千个以上的旋钮。

    /// admonition | 🧰 工具箱：大小、每行的最大值、取整、小整数
    - `x.abs()` 把每个数变成正数，和 `torch.abs(x)` 一样。
    - `x.amax(dim=1)` 是每一行里最大的数：对 `x = torch.tensor([[1.0, 5.0], [2.5, 0.4]])`，它是 `[5, 2.5]`。
    - `torch.round(x)` 把每个数取整到最近的整数：`torch.round(torch.tensor([2.4, -1.6]))` 是 `[2, -2]`。它们仍然是按浮点数存的。
    - `torch.int8` 是一种 dtype：−128 到 127 的整数，每个占一个字节。`x.to(torch.int8)` 把张量变成这种 dtype，就像 `.float()` 把张量变成 32 位浮点数。
    ///

    ## 练习 4：quantize

    写 `quantize(W, bits)`。`W` 是一个浮点数矩阵，形状是（行数、列数）。它返回一对东西：

    - `q`，形状和 `W` 一样，dtype 是 `torch.int8`：每个旋钮除以它所在行的缩放系数，再取整；
    - `scale`，一个向量，每行一个数：这一行里最大的大小，除以 $2^{\text{bits} - 1} - 1$。

    对两行分别是 `[1.4, -0.66, 0.25]` 和 `[-0.07, 0.03, 0.012]` 的矩阵，用 4 位时，`q` 的两行是 `[7, -3, 1]` 和 `[-7, 3, 1]`，`scale` 是 `[0.2, 0.01]`。

    要让 `W` 的每一行除以自己的缩放系数，缩放系数得竖成一列：第 20 章的 `scale[:, None]`。
    """)
    return


@app.function
def quantize(W, bits):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "quantize", quantize, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""分三步，和你动手算时一样：先由 `bits` 得到最大的整数；再用 `.abs()` 和 `.amax(dim=1)` 得到每一行的缩放系数；然后相除、取整、变成 `torch.int8`。例子的第二行里，大小最大的是一个负的旋钮：`.abs()` 就是为它准备的。""",
        r"""一共 3 行。`levels = 2 ** (bits - 1) - 1`，然后 `scale = W.abs().amax(dim=1) / levels`，最后返回 `torch.round(W / scale[:, None]).to(torch.int8)` 和 `scale`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    损失了多少？下面的单元格把计时测试里的矩阵 `W` 分别用 8、6、4 位量化，再读回来，显示误差的典型大小占旋钮典型大小的多少（第 09 章的典型大小：平方的均值再开平方根）。
    最后一列是整个矩阵共用一个缩放系数时的同一个数。
    """)
    return


@app.cell
def _(W):
    def typical(t):
        return torch.sqrt(torch.mean(t * t)).item()

    for _bits in [8, 6, 4]:
        _q, _scale = quantize(W, _bits)
        _shared = torch.max(torch.abs(W)) / (2 ** (_bits - 1) - 1)
        _row_error = typical(_q.float() * _scale[:, None] - W) / typical(W)
        _shared_error = typical(torch.round(W / _shared) * _shared - W) / typical(W)
        print(_bits, "位   每行一个缩放系数：", round(_row_error, 3), "  整个矩阵共用一个：", round(_shared_error, 3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    用 8 位、每行一个缩放系数时，旋钮的偏差大约是它们典型大小的 1%。每少一位，相邻整数之间的台阶就大一倍，误差也跟着大一倍：6 位时大约 3%，4 位时大约 15%。整个矩阵共用一个缩放系数的话，不管用几位，都要差上大约三倍。

    ## 不读回来就相乘

    存成小整数的模型，仍然要和浮点数的向量相乘。最直接的做法是把每个旋钮读回来，$q \times \text{scale}$，然后像以前一样乘。但一整行的缩放系数是同一个，所以可以留到最后：

    > $$(q_1 s) x_1 + (q_2 s) x_2 + (q_3 s) x_3 = (q_1 x_1 + q_2 x_2 + q_3 x_3)\, s$$

    先让向量和整数相乘，再把每个输出乘以它那一行的缩放系数，只乘一次。对一个 3,072 行、1,024 列的矩阵来说，这是乘 3,072 次缩放系数，而不是 300 万次。

    ## 动手算：缩放系数留到最后

    一行存成整数 3、−1、2，缩放系数是 0.5。输入是向量 1、2、4。每个整数乘以对应的输入，加起来是多少？输出又是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    product_boxes = hand.boxes(['整数乘输入的和', '输出'], step=0.1)
    return (product_boxes,)


@app.cell(hide_code=True)
def _(product_boxes):
    hand.marked(product_boxes, {'整数乘输入的和': 9, '输出': 4.5}, {
        '整数乘输入的和': r"""$3 \times 1 - 1 \times 2 + 2 \times 4$""",
        '输出': r"""这个和乘以缩放系数""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    先读回来再乘，结果一样：旋钮是 1.5、−0.5 和 1，而 $1.5 \times 1 - 0.5 \times 2 + 1 \times 4 = 4.5$。

    ## 练习 5：quantized_matmul

    写 `quantized_matmul(x, q, scale)`。`q` 和 `scale` 是 `quantize` 给出的那种矩阵：`q` 的形状是（输出数、输入数），dtype 是 `torch.int8`；`scale` 每个输出一个数。
    `x` 装着浮点数的向量，它最后一维的大小是输入数。`q` 和 `scale` 代表着某个矩阵 `W`，函数要返回 `plain(x, W)` 会给出的结果，但不把 `W` 造出来：先和整数相乘，再乘缩放系数。

    结果最后一维的大小是输出数，所以直接乘以向量 `scale`，每个输出就乘上了自己的那个数，不需要 `None`。

    PyTorch 的 `@` 不肯混用不同的 dtype（第 10 章），也没有 `int8` 和浮点数相乘的运算。所以做乘法时得先把 `q` 变成浮点数：`q.float()`。记住这一点，下一节要用。

    对你动手算的那题，`x = [1, 2, 4]`、唯一的一行 `[3, -1, 2]`、缩放系数 `[0.5]`，它必须给出 `[4.5]`。
    """)
    return


@app.function
def quantized_matmul(x, q, scale):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "quantized_matmul", quantized_matmul, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""看看 `plain`：它乘的是 `M.T`。对变成浮点数之后的 `q` 做同样的事，再把结果乘以 `scale`。""",
        r"""只要一行：`(x @ q.float().T) * scale`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8 位的真实模型

    现在轮到整个模型。`shrink` 用你的 `quantize` 把 28 个块里的每个矩阵都量化，其余的保持原样：归一化的增益（它们很小），以及词元表。
    `small_product` 是给 `block_step` 用来代替 `plain` 的乘法：`plain` 接收一个矩阵，它接收的是一对 `(q, scale)`。`product` 之所以做成输入，就是为了这个。
    """)
    return


@app.function
def shrink(knobs, bits):
    small = {}
    for name in knobs:
        if name != "model.embed_tokens.weight" and len(knobs[name].shape) == 2:
            small[name] = quantize(knobs[name], bits)
        else:
            small[name] = knobs[name]
    return small


@app.function
def small_product(y, M):
    return quantized_matmul(y, M[0], M[1])


@app.function
def answer(knobs, product, tokenizer, question):
    def step(new_ids, caches):
        return qwen_step(knobs, new_ids, caches, product)

    ids = tokenizer.chat(question)
    return tokenizer.decode(generate_cached(step, empty_caches(), ids, 30, END_OF_TURN)[len(ids):])


@app.cell
def _(knobs, tokenizer):
    small_knobs = shrink(knobs, 8)
    matrix_numbers = 0
    for _name in knobs:
        if _name != "model.embed_tokens.weight" and len(knobs[_name].shape) == 2:
            matrix_numbers = matrix_numbers + knobs[_name].numel()
    print("各个块的矩阵，32 位时的字节数：", 4 * matrix_numbers, "  8 位时：", matrix_numbers)
    print("32 位：", answer(knobs, plain, tokenizer, "Write one sentence about the sea."))
    print(" 8 位：", answer(small_knobs, small_product, tokenizer, "Write one sentence about the sea."))
    return (small_knobs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    各个块的矩阵从 1.76 GB 变成了 0.44 GB，只有四分之一。而在测试机器上，模型写出了一字不差的同一句话，尽管它那 4.4 亿个矩阵旋钮每个都偏了大约 1%。
    （如果你的结果差了一个词，那是舍入在你的处理器上落在了不同的地方。句子应该同样通顺。）

    模型经得起这样折腾，是因为它的计算方式。每个输出都是一千个以上的旋钮加起来的和，它们的舍入误差方向是随机的，所以误差大部分互相抵消了（第 09 章：随机部分的大小加起来，远比它们的个数增长得慢）。

    还能再压多少？同一个问题，分别用 8、6、4 位：
    """)
    return


@app.cell
def _(knobs, tokenizer):
    for _bits in [8, 6, 4]:
        print(_bits, "位：", answer(shrink(knobs, _bits), small_product, tokenizer, "What is the capital of France?"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    8 位和 6 位时答的是巴黎（Paris）。4 位时，在测试机器上，模型十分笃定地说出了法国的另一座城市。每个旋钮都有 15% 的误差，连着 28 个块，就消化不了了。

    这个结果说的是**这个简单的做法**，不是 4 位本身。每天都有模型在用 4 位提供服务，只是做得更细：不是每行一个缩放系数，而是每一小组旋钮一个；取整的方式也是量过它对输出的影响之后才定的。
    另外，4 位的旋钮是两个挤在一个字节里的，`int8` 做不到这一点。“深入了解”里列出了这些方法的名字。

    ## 变小了，但还没变快

    字节只剩四分之一，解码应该更快才对。用同一段简短的提示做完预填充之后，分别用两套旋钮测一次解码调用：
    """)
    return


@app.cell
def _(knobs, small_knobs, tokenizer):
    _scores, ready = qwen_step(knobs, tokenizer.chat("What is the capital of France?"), empty_caches(), plain)

    def decode_plain():
        return qwen_step(knobs, [END_OF_TURN], ready, plain)

    def decode_small():
        return qwen_step(small_knobs, [END_OF_TURN], ready, small_product)

    round(time_it(decode_plain, 10), 3), round(time_it(decode_small, 10), 3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    依次是 32 位和 8 位时写一个词元的秒数。测试机器上是 0.12 和 0.16。小模型反而**更慢**。

    原因是你自己写下的：`q.float()`。每做一次乘法，PyTorch 都先把整个小整数矩阵复制成 32 位浮点数，每个 4 字节，再用这份复制品去乘。省下的字节在每个词元上又花了回去，还要搭上复制的时间。

    ## 融合运算

    同一个乘法有三种算法，数一数每种算法里，有多少次是矩阵那么大的东西在内存里走一趟：

    - **先读回来，再乘**：把 `q` 复制成浮点数，把复制品乘以缩放系数，再和向量相乘。三次。
    - **你的 `quantized_matmul`**：把 `q` 复制成浮点数，和向量相乘。两次。缩放系数只碰那个很小的结果。
    - **32 位浮点数**，和第 27 章一样：一次，但每个旋钮 4 字节。
    """)
    return


@app.cell
def _(W, in_32_bits, x):
    q8, scale8 = quantize(W, 8)

    def read_back_first():
        return plain(x, q8.float() * scale8[:, None])

    def scale_last():
        return quantized_matmul(x, q8, scale8)

    round(1000 * time_it(read_back_first, 30), 2), round(1000 * time_it(scale_last, 30), 2), round(1000 * time_it(in_32_bits, 30), 2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    依次是三种算法的毫秒数。测试机器上是 1.24、0.63 和 0.27。时间跟着搬运的字节数走，而不是跟着运算走：三种算法的运算是一样的。

    缺的是第四种算法：取出小整数，每个旋钮一个字节，取出来就直接乘、直接加，在内存里只走**一趟**，不做任何复制。这样搬运的字节只有 32 位乘法的四分之一，而字节的代价你已经量过了。

    一个运算一口气完成好几步的工作，不把中间结果写进内存，这样的运算叫**融合运算**（fused operation）。你在第 22 章见过一个：`F.scaled_dot_product_attention` 把分数、掩码、softmax 和混合一起做完，从不造出那张表。
    量化乘法是部署服务时另一个要紧的融合运算。它没法用 Python 写：在 Python 里一个数一个数地走过矩阵，要慢上几千倍。它是用编译型语言写的，每种处理器各写一份，而对笔记本处理器上普通的 `int8` 张量，PyTorch 没有提供这样的运算。

    人们用来在自己电脑上运行模型的那些引擎，大体上就是一批这样的融合运算，外面套着你写过的那个循环。笔记本上的 4 位模型就是这样快起来的：你的 KV 缓存，你的缩放系数，再加上一个每个旋钮只读一次的乘法。

    ## 你搭出了什么
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("提示", "92 个词元"), ("28 个缓存", "键和值"), ("新词元", "1 趟"), ("回答", "文本")], ["预填充，一次", "解码", "重复"], "带 KV 缓存的一个请求：提示只读一遍，之后每个词元只花一趟")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    现在一个请求里，每个词元只花一趟，而你也知道剩下的时间受什么限制：每个词元要搬运的字节数。

    ## 接下来

    真正的服务面对的不是一个请求。它同时有几百个请求，随时会到，每个都有自己的缓存，先变大，然后结束。
    **第 29 章**为此搭一个小型服务器：模型每运行一次，由好几个请求共用；它们的缓存共用同一块内存。

    ## 你现在会做的事

    - 解释为什么每次全部重算的写法浪费了大部分工作，并数出趟数
    - 解释为什么前面词元的键和值永远不变，并把它们留在 KV 缓存里
    - 只为新词元写注意力，掩码和位置都正确
    - 说出一个请求的两个阶段，预填充和解码，并分别测量
    - 算出一个 KV 缓存占多少内存
    - 用每行一个缩放系数把矩阵量化成整数，并说出误差有多大
    - 和量化后的矩阵相乘，把缩放系数留到最后
    - 解释什么是融合运算，以及为什么量化要靠它才能变快

    ---

    <!-- go-deeper -->
    **深入了解（选读）：**在普通电脑上运行量化模型的引擎有 *llama.cpp*（使用 GGUF 文件）和苹果的 *MLX*；模型文件名里的 `Q4_K_M` 之类，说的就是它的量化方式。
    量过输出之后再决定怎么取整，是 *GPTQ*（Frantar 等，2022）和 *AWQ*（Lin 等，2023）的思路。`torch.compile` 能把一连串 PyTorch 运算融合起来。
    """)
    return


if __name__ == "__main__":
    app.run()
