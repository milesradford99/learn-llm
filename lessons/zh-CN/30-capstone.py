import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="30 · 综合项目：先猜，再检查")

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
    # 30 · 综合项目：先猜，再检查

    这是主课程的最后一章，选做。它用上第四部分的全部内容，再提速一次，而这次提速听上去不可能做到：模型运行一次就写出好几个词元，**而回答里的词元一个也不变**。

    这个想法叫**推测解码**（speculative decoding）。先由某个便宜的东西**猜**出接下来的几个词元。真正的模型再用一次运行把这些猜测全部**检查**一遍，留下它同意的，并纠正它第一个不同意的。

    第 28 章和第 29 章的两个事实让它划得来：

    - 解码一步，喂 1 个词元和喂几个词元花的时间差不多，因为大部分开销在取旋钮上；
    - 在几个新词元上运行一次，得到的是**每个**新词元之后的分数，而不只是最后一个之后的。

    **来自前面的章节。**下面折叠起来的单元格里是第 28 章的模型代码：你的文件读取器、`share_heads`、`rms_norm`、`rope`、`attend_new`、`remember`、`block_step`、`empty_caches`、`qwen_step`、`generate_cached` 和 `time_it`。再下一个单元格加载模型。
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
def remember(cache, k, v):
    return torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)


@app.function(hide_code=True)
def block_step(x, L, cache):
    new, seen = x.shape[1], cache[0].shape[2]
    y = rms_norm(x, L["norm1"])
    q = (y @ L["q"].T).reshape(1, new, 16, 128).transpose(1, 2)
    k = (y @ L["k"].T).reshape(1, new, 8, 128).transpose(1, 2)
    v = (y @ L["v"].T).reshape(1, new, 8, 128).transpose(1, 2)
    q = rope(rms_norm(q, L["q_norm"]), 1000000, seen)
    k = rope(rms_norm(k, L["k_norm"]), 1000000, seen)
    cache = remember(cache, k, v)
    blended = attend_new(q, share_heads(cache[0], 2), share_heads(cache[1], 2))
    x = x + blended.transpose(1, 2).reshape(1, new, 2048) @ L["out"].T
    y = rms_norm(x, L["norm2"])
    return x + (F.silu(y @ L["gate"].T) * (y @ L["up"].T)) @ L["down"].T, cache


@app.function(hide_code=True)
def empty_caches():
    return [(torch.zeros(1, 8, 0, 128), torch.zeros(1, 8, 0, 128)) for layer in range(28)]


@app.function(hide_code=True)
def qwen_step(knobs, new_ids, caches):
    with torch.no_grad():
        x = knobs["model.embed_tokens.weight"][torch.tensor([new_ids])]
        grown = []
        for layer in range(28):
            x, cache = block_step(x, layer_knobs(knobs, layer), caches[layer])
            grown.append(cache)
        x = rms_norm(x, knobs["model.norm.weight"])
        return x[0, -1] @ knobs["model.embed_tokens.weight"].T, grown


@app.function(hide_code=True)
def generate_cached(step, caches, ids, limit, stop):
    new = ids
    for turn in range(limit):
        scores, caches = step(new, caches)
        token = torch.argmax(scores).item()
        if token == stop:
            return ids
        ids = ids + [token]
        new = [token]
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
    ## 喂几个和喂一个差不多贵

    先验证这个说法。第 28 章的 `qwen_step` 只返回最后一个新词元之后的分数。`qwen_step_all` 是同一个函数，只改了最后一行：它在**每个**新词元之后都返回一行分数，所以喂 5 个词元就得到 5 行。

    下面的单元格先用一个问题把缓存填好，然后计时一次运行分别喂入 1、2、5、9、17 个词元。每一行：喂入的词元数和秒数。
    """)
    return


@app.function
def qwen_step_all(knobs, new_ids, caches):
    with torch.no_grad():
        x = knobs["model.embed_tokens.weight"][torch.tensor([new_ids])]
        grown = []
        for layer in range(28):
            x, cache = block_step(x, layer_knobs(knobs, layer), caches[layer])
            grown.append(cache)
        x = rms_norm(x, knobs["model.norm.weight"])
        return x[0] @ knobs["model.embed_tokens.weight"].T, grown


@app.cell
def _(knobs, tokenizer):
    _scores, filled = qwen_step_all(knobs, tokenizer.chat("What is the capital of France?"), empty_caches())
    feed_costs = []
    for _count in [1, 2, 5, 9, 17]:

        def _feed():
            return qwen_step_all(knobs, [100] * _count, filled)

        feed_costs.append((_count, round(time_it(_feed, 5), 3)))
    feed_costs
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    测试机器上：1 个词元 0.14 秒，2 个 0.14 秒，5 个 0.23 秒，9 个 0.28 秒，17 个 0.33 秒。十七个词元，价钱还不到一个词元的两倍半。在 GPU 上，运算和取数相比更便宜，这条曲线还要更平。

    ## 检查猜测

    假设到目前为止的文本以词元 `"the"` 结尾，一个猜测器给出接下来的四个词元：`" cat" " sat" " on" " a"`。把这五个，也就是 `"the"` 和四个猜测，在一次运行里喂进去。得到的五行分数说明了模型自己会写什么：

    - 第 0 行，在 `"the"` 之后：模型为第一个新词元做的选择。拿它和第一个猜测比。
    - 第 1 行，在 `"the cat"` 之后：模型为第二个新词元做的选择。只有第一个猜测对了，这一行才有意义，因为它是当作 `" cat"` 已经在文本里算出来的。
    - 依此类推，直到第 4 行，在四个猜测全部之后。

    所以按顺序走过这些猜测。只要一个猜测等于模型在那个位置的贪心选择，它就正是普通解码会写的东西：留下它。遇到第一个不同的猜测就停下：那个位置上模型自己的选择就是下一个词元，它之后的猜测全部扔掉。
    如果所有猜测都对，最后一行还能白送一个词元。

    不管哪种情况，**每次运行至少加上一个词元**，也就是普通解码本来会加的那一个，而且加上的词元和普通贪心解码写出的完全一样。

    ## 动手算：留下什么

    猜测是词元 5、9、2、7。一次运行给出模型在每个前缀之后的选择：5、9、4、1、8。留下几个猜测？这次运行一共给文本加上几个词元？最后加上的是哪个词元？
    如果选择是 5、9、2、7、8 呢？
    """)
    return


@app.cell(hide_code=True)
def _():
    accept_boxes = hand.boxes(['留下的猜测数', '加上的词元数', '最后加上的词元', '全都猜对时加上的词元数'])
    return (accept_boxes,)


@app.cell(hide_code=True)
def _(accept_boxes):
    hand.marked(accept_boxes, {'留下的猜测数': 2, '加上的词元数': 3, '最后加上的词元': 4, '全都猜对时加上的词元数': 5}, {
        '留下的猜测数': r"""5 和 9 对上了；第三个猜测是 2，模型说是 4""",
        '加上的词元数': r"""留下的猜测，加上它们停下的地方模型自己的选择""",
        '最后加上的词元': r"""第一个猜错的位置上模型的选择""",
        '全都猜对时加上的词元数': r"""四个猜测全部，加上最后一行的选择""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 1：accept

    写 `accept(draft, best)`。`draft` 是猜测的词元列表。`best` 比它长一个：模型在每个前缀之后的贪心选择，`best[0]` 是还没有任何猜测时的，`best[i]` 是前 $i$ 个猜测之后的。
    它返回要加上的词元：第一个和 `best` 在同一位置不同的猜测之前的那些猜测（不包括它），再加上 `best` 在那个位置的值。如果没有哪个猜测不同，就是全部猜测，再加上 `best` 的最后一项。

    `accept([5, 9, 2, 7], [5, 9, 4, 1, 8])` 必须得到 `[5, 9, 4]`，没有猜测时，`accept([], [6])` 必须得到 `[6]`。
    """)
    return


@app.function
def accept(draft, best):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "accept", accept, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""用一个 `while` 循环（第 19 章）从头数出有几个猜测对上了，它在 `draft` 结束或第一次对不上时停下。把这个数叫作 `m`。然后就是前 `m` 个猜测，再加上 `best` 的某一项：哪一项？""",
        r"""一共 4 行。`m = 0`，然后 `while m < len(draft) and draft[m] == best[m]:`，里面是 `m = m + 1`，最后 `return draft[:m] + [best[m]]`。条件里的先后顺序很要紧：左边为假时，`and` 不会去看右边。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 收回猜错的部分

    一次运行喂进了最后一个词元和所有猜测，所以每个块的缓存现在为它们全都存了键和值，猜错的也在内。这些必须去掉，否则下一次运行会去注意文本里并不存在的词元。

    守住这条规则：**缓存装着文本里除最后一个以外的每个词元。**最后一个词元总是在下一次运行里，和下一批猜测一起喂进去。一次运行之后，文本多了 `accept` 返回的那些词元，所以缓存必须装
    $\text{运行前的词元数} + \text{加上的词元数} - 1$ 项，其余的切掉。

    ## 动手算：留多少

    文本有 31 个词元，所以缓存装着 30 个。一次运行喂进最后一个词元和 4 个猜测。运行之后缓存装着多少个词元？`accept` 返回 3 个词元：文本现在有多长，缓存应该留下几项？
    """)
    return


@app.cell(hide_code=True)
def _():
    trim_boxes = hand.boxes(['运行之后的项数', '文本现在的词元数', '要留下的项数'])
    return (trim_boxes,)


@app.cell(hide_code=True)
def _(trim_boxes):
    hand.marked(trim_boxes, {'运行之后的项数': 35, '文本现在的词元数': 34, '要留下的项数': 33}, {
        '运行之后的项数': r"""30 个，加上刚喂进的 5 个""",
        '文本现在的词元数': r"""$31 + 3$""",
        '要留下的项数': r"""文本里除最后一个以外的每个词元""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    留下的两个猜测还在缓存里，它们的键也是对的：算它们的时候，前面的文本是对的。去掉的只有两个猜错的。

    ## 练习 2：trim

    写 `trim(caches, length)`。`caches` 是一列 `(K, V)` 对，每个块一对，每个的形状都是（批、头、词元数、数的个数），就像 `empty_caches` 和 `qwen_step` 造出来的那样。它按同样的顺序返回一列对，每一对只留下前 `length` 个词元。

    一个块装着 5 个词元时，`length` 为 3 的 `trim` 返回一对，其中的张量装着 3 个词元。
    """)
    return


@app.function
def trim(caches, length):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "trim", trim, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""对这些对做一个列表推导式，`for K, V in caches`，每一对给出一对。词元在第 2 维，所以切片是 `[:, :, :length]`。""",
        r"""只要一行：`return [(K[:, :, :length], V[:, :, :length]) for K, V in caches]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 猜测从哪里来

    通常的猜测器是一个**草稿模型**（draft model）：同一家族里一个小得多的模型，用同一个分词器，贪心地写出几个词元。Qwen3-0.6B 已经是它家族里最小的一个，所以这一页用的猜测器一点代价都不花。

    模型写的东西有很多是在重复已经有的文本：引用的句子、名字、正在修改的一行代码、正在复述的故事。所以看看文本最后几个词元，找到这几个词元在**更早**出现过的最近的位置，然后猜：当时跟在它们后面的东西，现在还会跟在后面。
    这叫**提示查找**（prompt lookup）。文本不重复自己的时候，它要么什么都找不到，要么猜错，检查会把这些猜测扔掉。

    ## 动手算：查一查

    文本是词元 4、7、1、9、4、7、2、4、7，从 0 开始编号。查它最后 2 个词元，最多猜 3 个。最后 2 个词元**最近的**更早副本从哪里开始？猜的第一个词元是什么？一共猜几个词元？
    """)
    return


@app.cell(hide_code=True)
def _():
    lookup_boxes = hand.boxes(['最近的更早副本的起点', '猜的第一个词元', '猜的词元数'])
    return (lookup_boxes,)


@app.cell(hide_code=True)
def _(lookup_boxes):
    hand.marked(lookup_boxes, {'最近的更早副本的起点': 4, '猜的第一个词元': 2, '猜的词元数': 3}, {
        '最近的更早副本的起点': r"""4、7 出现在 0、4 和末尾；末尾那个本身不算，而 4 比 0 晚""",
        '猜的第一个词元': r"""紧跟在那个副本后面的""",
        '猜的词元数': r"""2、4、7：后面的词元够多，要几个就猜几个""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：倒着数，以及比较列表
    - `range(5, -1, -1)` 依次是 5、4、3、2、1、0：第 09 章 `range` 的步长可以是负数。终点仍然不包括在内，所以用 −1 让它停在 0。
    - `[4, 7] == [4, 7]` 是 `True`：两个列表装着同样的东西、顺序也一样时，它们相等。`ids[-2:]` 是最后 2 项（第 09 章的切片可以用负数从末尾数起）。
    ///

    ## 练习 3：lookup_draft

    写 `lookup_draft(ids, n, k)`。`ids` 是到目前为止的文本，一个词元列表。取它的最后 `n` 个词元。找到同样这 `n` 个词元更早出现的最近的位置，起点至少比最后 `n` 个的起点早一个词元，然后返回跟在那个副本后面的词元，最多 `k` 个。
    如果没有更早的副本，返回空列表。

    `lookup_draft([4, 7, 1, 9, 4, 7, 2, 4, 7], 2, 3)` 必须得到 `[2, 4, 7]`。同样的文本，`k = 5` 时也必须得到 `[2, 4, 7]`：后面只有这几个。
    """)
    return


@app.function
def lookup_draft(ids, n, k):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "lookup_draft", lookup_draft, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""用一个倒着数的 `range`，从最晚的可能起点一直走到 0，把 `ids[start:start + n]` 和最后 `n` 个比较。最晚的可能起点是 $\text{len(ids)} - n - 1$。第一次对上就返回；循环之后返回 `[]`。切片超出末尾时就停在末尾，所以不必为 `k` 操心。""",
        r"""一共 5 行。`tail = ids[-n:]`，然后 `for start in range(len(ids) - n - 1, -1, -1):`，里面是 `if ids[start:start + n] == tail:` 和 `return ids[start + n:start + n + k]`。最后 `return []`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 练习 4：speculate

    现在是循环，综合项目的主体。它就是第 28 章的 `generate_cached` 加上猜测。

    写 `speculate(step, caches, ids, limit, stop, draft, k)`。`step(new_ids, caches)` 喂入词元，返回一对东西：一个张量，**每喂入一个词元一行分数**，以及变长了的缓存。按照规则，`caches` 装着 `ids` 里除最后一个以外的每个词元。`draft(ids, k)` 返回最多 `k` 个猜测组成的列表。

    只要在开头的 `ids` 之后加上的词元少于 `limit` 个，就重复：

    1. 从 `draft` 拿到猜测；
    2. 在一次 `step` 调用里，喂入 `ids` 的最后一个词元，后面跟着这些猜测；
    3. 每一行的贪心选择组成列表 `best`（见下面的工具箱），`accept` 给出要加上的词元；
    4. 用 `trim` 裁剪缓存，守住规则；
    5. 一个一个地加上这些词元：如果有一个是 `stop`，就立刻返回 `ids`，不带上它。

    循环结束时返回 `ids`，并截断，让开头之后最多只有 `limit` 个词元：一次运行可能比需要的多加几个。

    /// admonition | 🧰 工具箱：每一行一个贪心选择
    `torch.argmax(scores, dim=-1)` 在每一行里挑出最大的数的位置，第 25 章的 `.tolist()` 把它们变成普通列表：对两行 `[1.0, 5.0, 2.0]` 和 `[7.0, 0.0, 3.0]`，结果是 `[1, 0]`。
    ///

    有一项检查用的是一个会数数的假模型，和一个先猜对两次再猜错的猜测器。从 `[1, 2, 3]` 开始、`stop = 9` 时，回答必须是 `[1, 2, 3, 4, 5, 6, 7, 8]`，和普通解码给出的完全一样，而 `step` 只调用 2 次，不是 6 次。
    """)
    return


@app.function
def speculate(step, caches, ids, limit, stop, draft, k):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "speculate", speculate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""循环之前，把开头的长度存在一个名字里，供 `while` 的条件和最后的截断使用。循环里的五步是五六行。第 2 步喂入 `[ids[-1]] + guess`。第 4 步留下 $\text{len(ids)} + \text{len(new)} - 1$ 项，所以要在加长 `ids` 的那个 `for` 循环**之前**做。""",
        r"""大约 12 行。`prompt = len(ids)`，然后 `while len(ids) - prompt < limit:`。里面：`guess = draft(ids, k)`，`scores, caches = step([ids[-1]] + guess, caches)`，`new = accept(guess, torch.argmax(scores, dim=-1).tolist())`，`caches = trim(caches, len(ids) + len(new) - 1)`，然后 `for token in new:`，里面是第 27 章的停止判断和 `ids = ids + [token]`。循环之后，`return ids[:prompt + limit]`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 来真的

    三个请求，每个回答两遍：一遍用第 28 章的 `generate_cached`，一遍用你的 `speculate`，提示查找看最后 2 个词元，最多猜 8 个。`speculate` 开始之前，按照规则，缓存先装进问题里除最后一个以外的每个词元。

    每一行输出：两个回答是否一个词元都不差、写了多少个词元、`speculate` 用了几次运行、两种方式各用的秒数。然后是回答的开头。这三个请求是特意挑的，分别展示好的情况、一般的情况和坏的情况。大约要 40 秒。
    """)
    return


@app.cell
def _(knobs, tokenizer):
    story = (
        "Pip the fox lived at the edge of a big forest. Every morning Pip ran to the river to look at the fish. "
        "One day the river was gone, and only dry stones were left. Pip asked the old owl what had happened. "
        "The owl said that a beaver had built a dam far up the hill."
    )
    requests = [
        "Copy this story exactly:\n\n" + story,
        "Here is a story. " + story + " Why was the river dry? Answer in one sentence.",
        "List the numbers from 1 to 20, separated by commas.",
    ]

    def plain_step(new_ids, caches):
        return qwen_step(knobs, new_ids, caches)

    def all_step(new_ids, caches):
        return qwen_step_all(knobs, new_ids, caches)

    def guesser(ids, k):
        return lookup_draft(ids, 2, k)

    for _question in requests:
        _ids = tokenizer.chat(_question)
        _started = time.perf_counter()
        _plain = generate_cached(plain_step, empty_caches(), _ids, 120, END_OF_TURN)
        _plain_seconds = time.perf_counter() - _started
        _runs = []

        def _counted(new_ids, caches):
            _runs.append(len(new_ids))
            return all_step(new_ids, caches)

        _started = time.perf_counter()
        _scores, _filled = qwen_step_all(knobs, _ids[:-1], empty_caches())
        _fast = speculate(_counted, _filled, _ids, 120, END_OF_TURN, guesser, 8)
        _fast_seconds = time.perf_counter() - _started
        print(_fast == _plain, len(_plain) - len(_ids), "个词元，", len(_runs), "次运行，普通解码", round(_plain_seconds, 1), "秒，推测解码", round(_fast_seconds, 1), "秒")
        print("   ", tokenizer.decode(_plain[len(_ids):])[:90])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    测试机器上：

    - **照抄一个故事：**同样的 65 个词元，用了 11 次运行而不是 65 次，3.6 秒而不是 9.3 秒。几乎每个猜测都对。
    - **回答一个关于它的问题：**同样的 14 个词元，时间差不多。回答用了故事里的几个短语，但不多。
    - **从 1 数到 20：**同样的 70 个词元，反而**更慢**，12.6 秒对 9.3 秒。文本里 `", "` 一直在重复，所以查找总是找到错的副本，猜出 8 个错的词元，而每次运行都要为喂进 9 个付出代价。

    每种情况下回答都一样，正如承诺的那样。猜测改变的只有速度。

    ## 什么时候划得来

    假设每个猜测猜对的机会是 $a$，彼此独立，猜测器每次给出 $k$ 个。一次运行总会加上第一个词元；第一个猜测对了，就加上第二个；前两个都对，就加上第三个；依此类推。平均下来一次运行加上

    > $$1 + a + a^2 + \dots + a^k$$

    个词元。值不值，要看喂入 $k + 1$ 个词元的一次运行要多少钱，这在页面顶部量过了：测试机器上，$k = 1$ 时大约是一个普通步，$k = 16$ 时大约是 2.4 个普通步。

    ## 练习 5：expected_tokens

    写 `expected_tokens(a, k)`：上面那个和，$1 + a + a^2 + \dots + a^k$，一共 $k + 1$ 项。

    `expected_tokens(0.5, 2)` 必须得到 1.75，`k = 0`，也就是不猜时，不管 `a` 是多少都是 1。
    """)
    return


@app.function
def expected_tokens(a, k):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "expected_tokens", expected_tokens, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""对 0 到 $k$ 的各个幂次做一个循环，把 $a$ 的每个幂加起来：`range(k + 1)` 给出这些幂次。第 00 章的乘方是 `**`。""",
        r"""一共 4 行：`total = 0`，然后 `for i in range(k + 1):`，里面是 `total = total + a ** i`，最后 `return total`。""",
    ], LOCALE)
    return


@app.cell
def _():
    [(a, round(expected_tokens(a, 4), 2), round(expected_tokens(a, 8), 2)) for a in [0.2, 0.5, 0.8, 0.95]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每一行：机会 $a$，然后是猜 4 个和猜 8 个时一次运行平均加上的词元数。$a = 0.2$ 时，猜 8 个比猜 4 个什么都没多得，而两者都比一个普通步贵：猜测是亏的。$a = 0.95$ 时，一次 9 个词元的运行大约能加上 7 个。
    所以引擎会一边运行一边看猜测对得有多频繁，随时改变 $k$，或者干脆不猜。

    一个好的草稿模型在普通文本上大约能猜对 70% 到 90% 的词元，所以推测解码如今是部署服务的标准做法。有些模型训练时就带着几个小的额外的头，自己猜接下来的几个词元，根本不需要单独的草稿模型。

    ## 你搭出了什么
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("文本", "n 个词元"), ("猜测", "≤ k"), ("一次运行", "k + 1 行"), ("留下", "≥ 1 个词元"), ("裁剪后", "n + 留下 − 1")], ["查找", "检查", "接受", "裁剪"], "推测解码：猜几个词元，一次运行检查完，留下对上的，再把缓存裁回去")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 走过的路

    主课程到这里就结束了。看看你现在能用自己的代码做的事，全部是：

    - **第一部分：**拟合一条直线，沿着斜坡往下走，让责任沿着计算往回传，造一个替你做这件事的引擎，训练能认数字的网络。
    - **第二部分：**把词变成向量，预测下一个词，给模型记忆，然后是注意力、自注意力和 Transformer。
    - **第三部分：**对字节做分词，搭一个现代 Transformer，好好地训练它，让它变快，把它分到多个进程上，拟合缩放定律，清洗它的数据，并对齐它。
    - **第四部分：**打开一个真实模型的文件并运行它，留住它的缓存，缩小它的旋钮，同时为许多人服务，现在还能一次运行写出好几个词元。

    从这里出发有两个方向，都超出了这门课：

    - **混合专家**（mixture of experts）。如今很多大模型在每个块里有好几份 SwiGLU，叫作*专家*，再用一个小层把每个词元只送到其中 2 个左右。模型的旋钮很多，但每个词元只用到很少，这让部署服务变成了一个新的内存问题。有了第 20 章的 SwiGLU 和第 07 章用来做选择的 softmax，你就能读懂它们的代码。
    - **智能体**（agents）。一个能调用工具、读取结果并继续下去的模型，就是套在第四部分全部内容外面的一个循环。这门课选做的智能体部分，第 40 到 51 章，把两个真实的智能体系统一直读到源代码。

    ## 你现在会做的事

    - 解释为什么喂几个词元和喂一个差不多贵
    - 用一次运行检查一列猜测，留下的正好是贪心解码会写的东西
    - 扔掉猜测时让缓存保持正确
    - 通过查找文本本身来猜词元
    - 写一个推测解码的循环，并预测它什么时候划得来

    ---

    <!-- go-deeper -->
    **深入了解（选读）：**推测解码由 Leviathan 等（2023）发表，Chen 等（2023）以*推测采样*（speculative sampling）的名字发表，其中的规则在随机抽取词元时也适用。提示查找解码由 Saxena（2023）提出。
    *Medusa* 和 *EAGLE* 训练额外的头来猜；DeepSeek-V3 的*多词元预测*（multi-token prediction）在训练时就这样做。现代形式的混合专家见 *Switch Transformers*（Fedus 等，2021）和 *Mixtral of Experts*（Jiang 等，2024）。
    """)
    return


if __name__ == "__main__":
    app.run()
