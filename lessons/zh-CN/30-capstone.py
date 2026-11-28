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

    这是主课程的最后一章，选做。它基于第 27、28 章的真实模型和 KV 缓存，为单请求解码再提一次速——而且听上去几乎不可能做到：模型每运行一次就能写出好几个词元，**回答里的每一个词元却丝毫不变**。

    这个办法叫**推测解码**（speculative decoding）：先让一个代价极低的猜测器**猜**出接下来的几个词元，再让真正的模型用一次运行把这些猜测全部**检查**一遍——猜对的留下，遇到第一个猜错的就当场纠正。

    第 28 章和第 29 章的两个事实让它划得来：

    - 解码一步时，喂 1 个词元和喂几个词元耗时差不多，因为大部分时间都花在读取模型参数（旋钮）上；
    - 一次喂进几个新词元，模型会顺带算出**每个**新词元之后的预测分数，而不只是最后一个之后的。

    **沿用前章代码。**下面折叠的单元格里是第 28 章的模型代码：文件读取器、`share_heads`、`rms_norm`、`rope`、`attend_new`、`remember`、`block_step`、`empty_caches`、`qwen_step`、`generate_cached` 和 `time_it`。紧接着的单元格负责加载模型。
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
    ## 喂几个词元，耗时和喂一个差不多

    先来验证第一条。第 28 章的 `qwen_step` 只返回最后一个新词元之后的分数。`qwen_step_all` 是同一个函数，只改了最后一行：它在**每个**新词元之后都返回一行分数，所以喂 5 个词元就得到 5 行。

    下面的单元格先用一个问题把缓存填好，再分别测一次喂入 1、2、5、9、17 个词元的耗时。输出每一项为 `(词元数, 秒数)`。
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
    在测试机上：喂 1 个词元耗时 0.14 秒，2 个还是 0.14 秒，5 个 0.23 秒，9 个 0.28 秒，17 个也只要 0.33 秒——词元数翻了 17 倍，耗时还不到原来的两倍半。在 GPU 上，算力相对内存读取更充裕，这条曲线还会更平。

    ## 检查猜测

    假设到目前为止的文本以词元 `"the"` 结尾，猜测器给出了接下来的 4 个词元：`" cat" " sat" " on" " a"`。把这 5 个词元（最后一个已确认词元 `"the"` 加上 4 个猜测）在一次运行里一起喂进去。模型返回 5 行分数，每行取最高分对应的词元就得到 `best[0..4]`。把它们和待检验的 4 个猜测 `draft[0..3]` 上下对齐，就能一眼看清错位对应关系：

    | 位置 | 0 | 1 | 2 | 3 | 4 |
    | --- | --- | --- | --- | --- | --- |
    | **喂入的 5 个词元** | `"the"`（最后已确认词） | `" cat"` | `" sat"` | `" on"` | `" a"` |
    | **模型输出的 `best[0..4]`** | `best[0]` | `best[1]` | `best[2]` | `best[3]` | `best[4]` |
    | **待检验的 `draft[0..3]`** | `draft[0]`（`" cat"`） | `draft[1]`（`" sat"`） | `draft[2]`（`" on"`） | `draft[3]`（`" a"`） | 全对时白送 |

    按列从左往右看：

    - 第 0 列，在 `"the"` 之后：`best[0]` 是模型为第 1 个新词元做的选择，拿它和 `draft[0]` 比。
    - 第 1 列，在 `"the cat"` 之后：`best[1]` 是模型为第 2 个新词元做的选择。只有 `draft[0]` 猜对了，这一列才有意义，因为它是把 `" cat"` 当作已经写进文本来算的。
    - 依此类推，一直到第 4 列（4 个猜测全部之后）。

    按顺序逐个核对这些猜测：只要猜测和模型在同一列的贪心选择一致，说明它正是普通解码原本会写的词，直接留下；一旦遇到第一个对不上的猜测就立刻停下，改用模型在这一列自己的选择作为下一个词元，后面剩下的猜测全部作废。如果 4 个猜测全对，最后一列 `best[4]` 还能白送 1 个词元。

    无论哪种情况，**每次运行都至少推进 1 个词元**（也就是普通解码本来就会写出的那一个），而且最终生成的词元序列与普通贪心解码完全一致。

    ## 动手算：留下什么

    假设猜测的词元是 5、9、2、7，模型一次运行在每个前缀之后给出的选择是 5、9、4、1、8。会留下几个猜测？这次运行一共往文本里加上了几个词元？最后加上的是哪个词元？
    如果模型的选择是 5、9、2、7、8 呢？
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

    写 `accept(draft, best)`。`draft` 是猜测的词元列表；`best` 比它多 1 项，记录模型在每个前缀之后的贪心选择——`best[0]` 是加上任何猜测之前的选择，`best[i]` 是加上前 $i$ 个猜测之后的选择。

    从头逐个对比 `draft` 和 `best`：留下前面所有对上的猜测，在第一次对不上的位置停下来，再补上 `best` 在那个位置的值。如果全部猜测都对上了，就返回全部猜测，再加上 `best` 的最后一项。

    例如，`accept([5, 9, 2, 7], [5, 9, 4, 1, 8])` 必须返回 `[5, 9, 4]`；没有猜测时，`accept([], [6])` 必须返回 `[6]`。
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
        r"""用一个 `while` 循环（第 19 章）从头数出连续对上了几个猜测，在 `draft` 走完或第一次对不上时停下。把这个数记作 `m`。结果就是前 `m` 个猜测，再补上 `best` 的某一项——具体是哪一项？""",
        r"""一共 4 行。`m = 0`，然后 `while m < len(draft) and draft[m] == best[m]:`，里面是 `m = m + 1`，最后 `return draft[:m] + [best[m]]`。条件里的先后顺序很要紧：左边为假时，`and` 不会去看右边。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 收回猜错的部分

    一次运行喂进了最后一个词元和全部猜测，所以每一层的缓存现在都存下了它们的键和值——包括猜错的那些。猜错的条目必须删掉，否则下一次运行就会去注意根本不在文本里的词元。

    因此要守住一条规则：**缓存永远只存文本里除最后一个以外的所有词元。**为什么不能把最后一个词元也提前存进去？因为第 28 章每步只喂 1 个新词；而推测解码下一步要把“最后一个已确认词元 + 多个猜测”打包一起喂进模型，如果最后一个词元提前进了缓存，就会被重复存两遍！

    守住这条规则后，每次运行结束，文本新添了 `accept` 返回的那些词元，缓存需要保留的长度自然就是
    $\text{运行前的词元数} + \text{加上的词元数} - 1$，多出来的直接裁掉。

    ## 动手算：留多少

    文本原本有 31 个词元，所以缓存里存着 30 项。这一步喂进最后一个词元和 4 个猜测。运行之后缓存里一共有多少项？如果 `accept` 返回 3 个词元，文本现在有多长？缓存应该保留几项？
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
    对上的两个猜测会继续留在缓存里，而且它们的键和值完全正确——因为计算它们时，前面的文本都是对的。被裁掉的只有那两个猜错的词元。

    ## 练习 2：trim

    写 `trim(caches, length)`。`caches` 是一个由 `(K, V)` 对组成的列表，每个块一对，每个张量的形状都是（批、头、词元数、每个头的维数），与 `empty_caches` 和 `qwen_step` 生成的格式相同。函数按原顺序返回新列表，每对只保留前 `length` 个词元。

    例如，一个块原本存着 5 个词元，调用 `trim` 并传入 `length = 3` 后，返回的对里每个张量只保留 3 个词元。
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

    通常的猜测器是一个**草稿模型**（draft model）：同一家族里小得多的模型，共用同一个分词器，负责先贪心地猜出几个词元。Qwen3-0.6B 已经是它家族里最小的一个，所以这一章换用一个零成本的猜测器。

    模型生成的文本里，有很多内容其实在重复前面已经出现过的片段：引用的句子、人名、正在修改的代码行、正在复述的故事。因此，我们可以看看文本末尾的几个词元，往前找到它们**最近一次**出现的位置，然后猜：当时跟在它们后面的词，现在还会接着出现。
    这个方法叫**提示查找**（prompt lookup）。如果文本没有重复，它要么找不到匹配，要么猜错，后面的检查步骤会把错误猜测全部丢弃。

    ## 动手算：查一查

    假设文本的词元序列是 4、7、1、9、4、7、2、4、7，从 0 开始编号。用最后 2 个词元往前查，最多猜 3 个。这最后 2 个词元**最近一次**在更早位置出现时，起点下标是多少？猜出的第一个词元是什么？一共猜出几个词元？
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
    - `range(5, -1, -1)` 依次生成 5、4、3、2、1、0：第 09 章学过的 `range` 步长可以是负数。终点仍然不包含在内，所以终点写 `-1` 才能停在 0。
    - `[4, 7] == [4, 7]` 的结果是 `True`：两个列表的元素相同且顺序一致时即相等。`ids[-2:]` 取最后 2 项，就像第 17 章的 `words[-32:]` 取最后 32 项一样。
    ///

    ## 练习 3：lookup_draft

    写 `lookup_draft(ids, n, k)`。`ids` 是到目前为止的词元列表。取出它的最后 `n` 个词元，往前找到同样这 `n` 个词元最近一次出现的位置（起点至少比末尾这 `n` 个词元的起点早 1 位），然后返回紧跟在那个副本后面的词元，最多 `k` 个。
    如果前面没有出现过相同的副本，就返回空列表。

    `lookup_draft([4, 7, 1, 9, 4, 7, 2, 4, 7], 2, 3)` 必须返回 `[2, 4, 7]`。同样的文本、换成 `k = 5` 时也必须返回 `[2, 4, 7]`，因为后面一共只有这 3 个词元。
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

    现在来写主循环，也就是这个综合项目的核心——它其实就是给第 28 章的 `generate_cached` 加上了“先猜再查”。

    写 `speculate(step, caches, ids, limit, stop, draft, k)`。其中 `step(new_ids, caches)` 喂入词元并返回一对结果：一个分数张量（**每喂入 1 个词元对应一行分数**）和更新后的缓存。按照前面的规则，`caches` 里存着 `ids` 中除最后一个以外的全部词元。`draft(ids, k)` 返回一个最多包含 `k` 个猜测的列表。

    只要在初始 `ids` 之后新加的词元还少于 `limit` 个，就不断重复以下步骤：

    1. 调用 `draft` 拿到一批猜测；
    2. 在一次 `step` 调用里，把 `ids` 的最后一个词元连同这些猜测一起喂进去；
    3. 取每一行的贪心选择组成列表 `best`（见下方工具箱），再用 `accept` 算出这一步要加上的词元；
    4. 用 `trim` 把缓存裁回 `len(ids) + len(new) - 1` 项，守住“少存最后一个词元”的规则；
    5. 把这些新词元逐个加到 `ids` 末尾：一旦遇到 `stop`，立刻返回当前的 `ids`（不包含 `stop`）。

    循环结束后返回 `ids`，并截断到初始长度之后最多保留 `limit` 个词元（因为最后一次运行可能一口气多加了几个）。

    /// admonition | 🧰 工具箱：每一行取一个贪心选择
    `torch.argmax(scores, dim=-1)` 会挑出每一行里最大值的下标，再用第 25 章的 `.tolist()` 转成普通列表：例如对两行 `[1.0, 5.0, 2.0]` 和 `[7.0, 0.0, 3.0]`，结果就是 `[1, 0]`。
    ///

    其中一项检查会用一个“会数数”的假模型，搭配一个“先猜对两次再猜错”的猜测器。从 `[1, 2, 3]` 开始、`stop = 9` 时，结果必须是 `[1, 2, 3, 4, 5, 6, 7, 8]`，与普通解码完全相同，但 `step` 只调用 2 次，而不是 6 次。
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
    ## 实战测试

    下面用三个请求各跑两遍：一遍用第 28 章的 `generate_cached`，一遍用你刚写的 `speculate`（提示查找看最后 2 个词元，最多猜 8 个）。在 `speculate` 开始前，按规则先把问题中除最后一个词元之外的部分填进缓存。

    每组输出依次显示：两种方式的回答是否逐词一致、一共生成了多少词元、`speculate` 调用了几次模型、两种方式各耗时多少秒，下面附上回答的开头。这三个请求分别代表顺风、一般和逆风三种场景，全部跑完大约需要 40 秒。
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
    在测试机上：

    - **照抄故事：**同样生成 65 个词元，只运行了 11 次而不是 65 次，耗时从 9.3 秒降到 3.6 秒。几乎每次都猜对了。
    - **回答故事相关的问题：**同样 14 个词元，耗时基本持平。回答只复用了故事里的几个短语，能猜中的不多。
    - **从 1 数到 20：**同样 70 个词元，反而**变慢了**（12.6 秒对 9.3 秒）。因为文本里不断重复 `", "`，提示查找总是指向错误的旧位置、一口气猜出 8 个错词，白白让每次运行多算了 9 个词元。

    三种场景下，回答都与普通解码一字不差——正如开头所说，推测解码改变的只有速度。

    ## 什么时候划得来

    假设每个猜测猜对的概率都是 $a$，且彼此独立，猜测器每次猜 $k$ 个。

    先用具体数字算一遍：设 $k = 2$（每次猜 2 个），$a = 0.5$（每个猜对的概率都是 $0.5$）。一次运行一共有 3 种结局：

    - **第 1 个就猜错**（概率 $0.5$）：保底加上 1 个词元；
    - **第 1 个猜对、第 2 个猜错**（概率 $0.5 \times 0.5 = 0.25$）：加上 2 个词元；
    - **2 个全猜对**（概率 $0.5 \times 0.5 = 0.25$）：加上 2 个猜测再白送 1 个，共 3 个词元。

    把每种结局的“词元数 $\times$ 概率”加起来，平均每次运行能推进：

    > $$1 \times 0.5 + 2 \times 0.25 + 3 \times 0.25 = 1.75$$

    巧的是，这个数正好等于 $1 + 0.5 + 0.5^2 = 1.75$！换个角度看就一目了然：第 1 个词元无论如何都有（概率 $1$），第 1 个猜对就能拿到第 2 个词元（概率 $a$），前 2 个全对就能拿到第 3 个词元（概率 $a^2$）。

    推广到每次猜 $k$ 个，平均每次运行推进的词元数就是：

    > $$1 + a + a^2 + \dots + a^k$$

    这笔买卖划不划算，还要看喂入 $k + 1$ 个词元的一次运行耗时多少——我们刚在页面顶部测过：测试机上，$k = 1$ 时耗时约等于 1 个普通解码步，$k = 16$ 时约等于 2.4 个普通解码步。

    ## 练习 5：expected_tokens

    写 `expected_tokens(a, k)`：计算上面这个和 $1 + a + a^2 + \dots + a^k$（一共 $k + 1$ 项）。

    例如，`expected_tokens(0.5, 2)` 必须返回 `1.75`；当 `k = 0`（不猜）时，不管 `a` 是多少都必须返回 `1`。
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
    每一项依次是：猜对概率 $a$，以及猜 4 个和猜 8 个时每次运行平均推进的词元数。当 $a = 0.2$ 时，猜 8 个几乎不比猜 4 个多推进多少，而两者的单步耗时都高于普通解码——这时硬猜反而划不来。到了 $a = 0.95$，一次喂进 9 个词元就能平均推进将近 7 个词元。
    因此，推理引擎会边跑边统计命中率，动态调整 $k$，命中率太低时就干脆暂停猜测。

    在日常文本上，一个好的草稿模型命中率常常能达到 60% 到 90%，因此推测解码如今已成为线上推理服务的标配。有些大模型甚至在训练时就自带几个轻量的预测头，自己就能顺手猜出后面几个词元，连单独的草稿模型都省了。

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
    ## 一路走来的风景

    主课程到这里就圆满收官了。回头看看，你已经完全用自己的代码走通了从零到一的全程：

    - **第一部分：**拟合直线、沿坡下降、把误差责任反向传遍整张计算图、亲手搭出自动求导引擎，并训练出能识别手写数字的神经网络。
    - **第二部分：**把词变成向量、预测下一个词、给模型装上记忆，一路搭出注意力、自注意力与完整的 Transformer。
    - **第三部分：**从字节出发做分词、搭起现代 Transformer、扎实训练并给它提速、扩展到多进程并行、拟合缩放定律、清洗数据，再完成对齐。
    - **第四部分：**打开真实大模型的权重文件并跑起来、维护 KV 缓存、压缩模型参数、同时服务多个请求，直到刚刚实现一次运行写出多个词元。

    站在这里往外看，还有两个值得探索的方向：

    - **混合专家**（Mixture of Experts）。如今许多大模型在每个块里都放了好几份 SwiGLU（称为*专家*），再用一个轻量路由层把每个词元只分配给其中 2 个左右。整张网络的参数总量很大，但每个词元实际只动用很小一部分，这给推理部署带来了新的内存调度课题。掌握了第 20 章的 SwiGLU 和第 07 章用来做选择的 softmax，你已经完全能读懂它们的代码。
    - **智能体**（Agents）。所谓能调用工具、读取结果并持续推进任务的智能体，本质上就是套在第四部分推理能力外面的一层循环。本课程选修的智能体篇（第 40 至 51 章），会带你直读两个真实智能体系统的源代码。

    ## 你现在已经掌握

    - 解释为什么一次喂入几个词元的耗时与喂入一个差不多
    - 用一次运行校验一批猜测，产出与贪心解码逐词一致的结果
    - 在丢弃错误猜测的同时，让 KV 缓存始终保持正确
    - 通过查找前文内容来零成本猜测后续词元
    - 写出完整的推测解码循环，并估算它何时划得来

    ---

    <!-- go-deeper -->
    **深入了解（选读）：**推测解码由 Leviathan 等（2023）提出，Chen 等（2023）也以*推测采样*（speculative sampling）之名独立发表，并将其推广到了随机采样的场景。提示查找解码由 Saxena（2023）提出。
    *Medusa* 和 *EAGLE* 通过训练额外的预测头来自行猜测；DeepSeek-V3 的*多词元预测*（multi-token prediction）则在训练阶段就融入了这一机制。现代形式的混合专家可参考 *Switch Transformers*（Fedus 等，2021）和 *Mixtral of Experts*（Jiang 等，2024）。
    """)
    return


if __name__ == "__main__":
    app.run()
