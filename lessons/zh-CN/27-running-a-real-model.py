import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="27 · 运行真实的模型")

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
    from learn_llm.qwen import END_OF_TURN, TYPES, QwenTokenizer, config, fetch, layer_knobs, missing

    LOCALE = "zh-CN"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 27 · 运行真实的模型

    第三部分结束时，你有了一个自己搭建、自己训练的模型。它大约有一百万个旋钮，写的是幼儿故事。人们真正在用的模型也是这样造出来的，只是每样东西都多出一千到一百万倍。
    第四部分讲模型生命的另一半：**部署服务**（serving），也就是把一个训练好的模型运行起来，又快又省地服务使用它的人。

    这一章从最朴素的部署服务开始。你要拿一个真实的模型，由训练它的实验室发布的那个，用你自己写的代码把它运行起来。不借助把一切藏起来的库：你要亲手打开文件，自己把里面的数读出来，
    再让它们经过第 20 章的 `rms_norm`、`rope`、`swiglu` 和 `attend`。如果这门课做到了它该做的，它会回答你。

    这个模型是 **Qwen3-0.6B**，来自阿里巴巴的 Qwen 团队：大约 6 亿个旋钮，对已发布的模型来说算小的，但比你的模型大五百倍。

    ## 权重文件是什么

    训练的结果是一组旋钮，第 21 章用 `torch.save` 把它们存了下来。要发布一个模型，实验室把旋钮放进一个文件，任何人都可以下载。人们把训练完成的模型的旋钮叫**权重**（weights），把这个文件叫**权重文件**（weight file）。

    常用的格式叫 *safetensors*，简单到可以手工读取。文件分三个部分，依次排列：

    1. **8 个字节**，存着一个整数 $n$；
    2. **文件头**（header），共 $n$ 个字节：一段文本，列出文件里的每个张量，包括它的名字、数的种类、形状，以及它的字节在哪里；
    3. **数据**：所有张量的数，一个张量接着一个张量，中间什么都没有。

    ## 把字节变成一个数

    第 19 章说过，一个字节是 0 到 255 的整数。更大的数要用好几个字节，读法像数字的各个位，只不过每一位的值是前一位的 256 倍，而不是 10 倍。
    而且如今几乎所有文件里，**第一个**字节是最小的那一位，这种顺序叫**小端序**（little-endian）。所以字节 5, 2 表示 $5 + 2 \times 256 = 517$。

    ## 动手算：读出一个长度

    一个文件以 8 个字节 2, 1, 0, 0, 0, 0, 0, 0 开头。这是哪个数？那字节 0, 2, 0, 0, 0, 0, 0, 0 又是哪个数？
    """)
    return


@app.cell(hide_code=True)
def _():
    byte_boxes = hand.boxes(['字节 2, 1, 0, …', '字节 0, 2, 0, …'])
    return (byte_boxes,)


@app.cell(hide_code=True)
def _(byte_boxes):
    hand.marked(byte_boxes, {'字节 2, 1, 0, …': 258, '字节 0, 2, 0, …': 512}, {
        '字节 2, 1, 0, …': r"""$2 + 1 \times 256$""",
        '字节 0, 2, 0, …': r"""$0 + 2 \times 256$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：把字节变成整数
    `int.from_bytes(data, "little")` 把字节读成一个整数，第一个字节最小：`int.from_bytes(bytes([5, 2]), "little")` 是 517。用 `"big"` 的话，第一个字节就是最大的那一位。
    ///

    ## 练习 1：header_length

    写 `header_length(first_bytes)`。`first_bytes` 装着权重文件的前 8 个字节。它返回这些字节所表示的整数，按小端序读。

    `header_length(bytes([2, 1, 0, 0, 0, 0, 0, 0]))` 必须得到 258。
    """)
    return


@app.function
def header_length(first_bytes):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "header_length", header_length, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""工具箱里有这个工具，那里是用在 2 个字节上的。用在 8 个字节上，做法一样。""",
        r"""只要一行：`int.from_bytes`，传入 `first_bytes` 和 `"little"`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 文件头

    文件头是一段文本，格式叫 **JSON**，看起来像写出来的 Python 字典。对一个只有一个小张量的文件，它可以是这样：

    ```
    {"gain": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}
    ```

    每个张量一项。`dtype` 是数的种类，`shape` 是形状，`data_offsets` 说明这个张量的字节从哪里开始、到哪里结束，**是从数据部分的开头算起**，而不是从文件的开头。
    数据部分紧跟在文件头后面，从文件的第 $8 + n$ 个字节开始。

    /// admonition | 🧰 工具箱：读文件
    - `with open(path, "rb") as f:` 以读字节的方式打开文件（`"rb"`：读，二进制）。缩进的那几行可以用 `f`，它们结束时文件会被关上。
    - `f.read(4)` 给出接下来的 4 个字节，并往前走：再来一次 `f.read(4)`，给出再后面的 4 个。
    - `f.seek(100)` 跳到文件的第 100 个字节，这样下一次 `read` 就从那里开始。
    - `json.loads(data)` 把 JSON 文本，或者它的字节，变成 Python 的字典和列表：`json.loads('{"a": [1, 2]}')` 是 `{"a": [1, 2]}`。`json` 已经载入了。
    ///

    ## 练习 2：read_header

    写 `read_header(path)`。`path` 是一个权重文件。它返回一对东西：作为字典的文件头，以及数据部分在文件里开始的位置。

    对一个前 8 个字节表示 258 的文件，文件头就是后面的 258 个字节，数据从 266 开始。
    """)
    return


@app.function
def read_header(path):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "read_header", read_header, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""在一个 `with open(path, "rb") as f:` 里读两次：先读 8 个字节，交给你的 `header_length`，再读它所说的那么多个字节。第二次读到的就是 JSON 文本。""",
        r"""3 行。把长度存进名字 `n`，然后返回 `json.loads(f.read(n)), 8 + n`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 把字节变成带小数点的数

    数据部分只是字节，在你说明它们装的是哪一种数之前，字节没有任何意义。下面是同样的 4 个字节，按三种方式来读。`torch.frombuffer` 根据给定的 dtype，直接用字节造出张量
    （它要求把字节包进 `bytearray`，那是一种可以修改的字节）。
    """)
    return


@app.cell
def _():
    four = bytearray([0, 0, 128, 63])
    torch.frombuffer(four, dtype=torch.uint8), torch.frombuffer(four, dtype=torch.int16), torch.frombuffer(four, dtype=torch.float32)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    四个小整数，或者两个较大的整数（$128 + 63 \times 256 = 16256$），或者第 10 章 32 位格式里的单独一个数 1.0。用错了 dtype，权重文件就是一堆噪声。所以文件头里的每一项都写明了它的 dtype。

    权重文件对 dtype 有自己的简短名字，`TYPES` 把它们翻译成 PyTorch 的：`TYPES["F32"]` 是 `torch.float32`，`TYPES["BF16"]` 是 `torch.bfloat16`，也就是第 22 章那种 16 位、范围和 float32 一样大的格式。
    模型用 16 位发布，是为了让下载量减半。第 22 章还测量过，CPU 做 16 位运算很慢，所以我们在读取时把一切都变成 32 位。

    ## 动手算：找到一个张量

    一个文件的前 8 个字节表示 100。它的文件头里有一项 `{"dtype": "BF16", "shape": [2, 2], "data_offsets": [40, 48]}`。这个张量从文件的第几个字节开始？它一共占多少个字节？每个数占多少个字节？
    """)
    return


@app.cell(hide_code=True)
def _():
    find_boxes = hand.boxes(['在文件里的第一个字节', '一共的字节数', '每个数的字节数'])
    return (find_boxes,)


@app.cell(hide_code=True)
def _(find_boxes):
    hand.marked(find_boxes, {'在文件里的第一个字节': 148, '一共的字节数': 8, '每个数的字节数': 2}, {
        '在文件里的第一个字节': r"""8 个字节存长度，100 个字节是文件头，再往数据里走 40 个字节""",
        '一共的字节数': r"""从 40 到 48，不包括 48""",
        '每个数的字节数': r"""形状说明有 $2 \times 2 = 4$ 个数""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个数两个字节：16 位，正如 `BF16` 所承诺的。

    ## 练习 3：read_tensor

    写 `read_tensor(path, entry, data_start)`。`entry` 是文件头里某一个张量的那一项，是一个字典，有键 `"dtype"`、`"shape"` 和 `"data_offsets"`；`data_start` 是数据部分开始的位置，就是 `read_header` 给出的那个。
    它返回这个张量，形状正确，数是 32 位浮点数：

    1. 跳到这个张量的第一个字节，恰好读出它的那些字节。
    2. 用 `torch.frombuffer(bytearray(data), dtype=...)` 用它们造出张量，dtype 用 `TYPES` 来查。
    3. 用 `.reshape(...)` 给它形状，用 `.float()` 把它变成 32 位。
    """)
    return


@app.function
def read_tensor(path, entry, data_start):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "read_tensor", read_tensor, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""你的手算练习就是第 1 步：第一个字节是 `data_start` 加上第一个偏移量，字节数是第二个偏移量减去第一个。`start, stop = entry["data_offsets"]` 给这两个数起了名字。""",
        r"""大约 5 行。在 `with open(path, "rb") as f:` 里，先 `f.seek(data_start + start)`，再 `data = f.read(stop - start)`。之后一行：`torch.frombuffer(bytearray(data), dtype=TYPES[entry["dtype"]])`，后面接 `.reshape(entry["shape"])` 和 `.float()`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 获取模型

    现在来看真正的文件。这个按钮会把 Qwen3-0.6B 的三个文件从实验室的页面下载到 `my-work/.models/`：权重（1.5 GB）、分词器的合并表，和一个小小的配置文件。这一次需要联网，
    要花几分钟，或者长得多，取决于你的网速。单元格下面有一个进度条，数着已经到达的兆字节（MB）；下载期间页面做不了别的事。如果下载中断了，再按一次：它会从停下的地方继续。每个文件到达时都会检查。

    运行这个模型大约需要 4 GB 的空闲内存。如果你下载不了或者运行不了，上面你写的三个函数仍然会被检查，这一章的其余部分也可以只读不跑。
    """)
    return


@app.cell(hide_code=True)
def _():
    download_button = mo.ui.run_button(label="下载 Qwen3-0.6B（1.5 GB，只需一次）")
    download_button
    return (download_button,)


@app.cell
def _(download_button):
    mo.stop(len(missing()) > 0 and not download_button.value, mo.md("*按下按钮下载模型。*"))
    folder = fetch()
    weights = folder / "model.safetensors"
    header, data_start = read_header(weights)
    data_start, len(header), header["model.layers.0.self_attn.q_proj.weight"]
    return data_start, folder, header, weights


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    你的 `read_header` 读一个真实的模型：数据从第 35,560 个字节开始，在一个列着 312 项的文件头之后，其中一项已经显示出来了。每个名字都有规律。`model.layers.0.self_attn.q_proj.weight` 是第 0 块里、自注意力里，生成查询的那个矩阵。
    这里把块叫作**层**（layer），编号从 0 到 27。

    其余的信息在配置文件里。它也是 JSON：
    """)
    return


@app.cell
def _(folder):
    settings = config(folder)
    [(key, settings[key]) for key in ["vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size", "rope_theta", "tie_word_embeddings"]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    和第 21 章的模型比一比，那个模型有 2,258 个词元、每个词元 128 个数、3 个块和 4 个头：

    | | 你的 | Qwen3-0.6B |
    |---|---:|---:|
    | 词表里的词元数 | 2,258 | 151,936 |
    | 每个词元的数（`hidden_size`） | 128 | 1,024 |
    | 块 | 3 | 28 |
    | 头 | 4 | 16 |
    | SwiGLU 里的隐藏数 | 341 | 3,072 |

    设计就是你搭建的那一个：块的每一半之前都有 RMSNorm，查询和键上有 RoPE，用 SwiGLU，没有偏置。只有三样东西是新的，接下来三节依次来讲。

    ## 新东西 1：一张表，用两次

    `tie_word_embeddings` 是真。模型的两头各有一张表：一张在输入时把词元变成向量，另一张在输出时把最后的向量变成每个词元的分数。两张表的形状都是（词元数，每个词元的数）。
    Qwen 对两头用的是**同一张表**：一个词元的分数，就是最后的向量和这个词元自己的向量的点积。这叫**共用词向量表**（tied embeddings），在这里省下了 1.56 亿个旋钮。

    文件里仍然用两个名字列出这张表。我们用你的 `read_tensor` 来检查它们装的是不是同样的数：
    """)
    return


@app.cell
def _(data_start, header, weights):
    table_in = read_tensor(weights, header["model.embed_tokens.weight"], data_start)
    table_out = read_tensor(weights, header["lm_head.weight"], data_start)
    table_in.shape, torch.equal(table_in, table_out)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    只有每个数都相同时，`torch.equal` 才是 `True`。所以我们载入除了 `lm_head.weight` 以外的每个张量，并跳过文件头里那一个不是张量的项 `__metadata__`：
    """)
    return


@app.cell
def _(data_start, header, weights):
    knobs = {}
    for name in header:
        if name != "__metadata__" and name != "lm_head.weight":
            knobs[name] = read_tensor(weights, header[name], data_start)
    len(knobs), sum([tensor.numel() for tensor in knobs.values()])
    return (knobs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    310 个张量，596,049,920 个数，你的代码几秒钟就读完了。

    ## 新东西 2：共用的头

    看看文件头给出的形状。在你的模型里，每个头都有自己的查询、键和值。这里，查询矩阵为每个词元生成 $16 \times 128 = 2048$ 个数：16 个头，每个 128。但键和值矩阵只生成 $8 \times 128 = 1024$ 个：**8 个头**。

    16 个查询头两两共用 8 个键头和值头：查询头 0 和 1 都看键头 0，查询头 2 和 3 看键头 1，依此类推。这叫**分组查询注意力**（grouped-query attention，GQA）。
    人们发现，模型这样做损失很小。它省下的东西在这里还不算重要，到下一章才重要：那时，读过的每个词元的键和值都要留在内存里。

    ## 动手算：谁和谁共用

    有 16 个查询头、8 个键头，查询头 5 用哪个键头？查询头 15 呢？一个块的键矩阵和值矩阵加起来，比起各用 16 个完整的头，一共省下多少个数？
    """)
    return


@app.cell(hide_code=True)
def _():
    gqa_boxes = hand.boxes(['查询头 5 的键头', '查询头 15 的键头', '一个块省下的数'])
    return (gqa_boxes,)


@app.cell(hide_code=True)
def _(gqa_boxes):
    hand.marked(gqa_boxes, {'查询头 5 的键头': 2, '查询头 15 的键头': 7, '一个块省下的数': 2097152}, {
        '查询头 5 的键头': r"""两两一组：0 和 1 用键头 0，2 和 3 用键头 1，4 和 5 用……""",
        '查询头 15 的键头': r"""最后一对用最后一个键头""",
        '一个块省下的数': r"""每个矩阵是 $1024 \times 1024$，原本该是 $2048 \times 1024$，所以每个省下 $1024 \times 1024$，一共有两个""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    在代码里，最简单的共用办法就是复制：把每个键头重复一遍，用 8 个造出 16 个，这样你的 `attend` 不用改就能用。

    /// admonition | 🧰 工具箱：沿某个维度重复
    `torch.repeat_interleave(x, 2, dim=0)` 沿一个维度把每一项重复，副本挨在一起：对 `x = torch.tensor([7, 8, 9])`，它给出 `[7, 7, 8, 8, 9, 9]`。
    ///

    ## 练习 4：share_heads

    写 `share_heads(x, times)`。`x` 的形状是（批大小，头数，词数，数的个数），就像第 16 章 `split_heads` 之后的键和值那样。它返回一个头数是原来 `times` 倍的张量，`x` 的每个头连续重复 `times` 次：
    有 3 个头、`times = 2` 时，结果的头依次是 `x` 的头 0、0、1、1、2、2。
    """)
    return


@app.function
def share_heads(x, times):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "share_heads", share_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""工具箱里的那个工具，用在正确的维度上。在（批大小，头数，词数，数的个数）里，头在第几个位置？""",
        r"""只要一行：`torch.repeat_interleave(x, times, dim=1)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 新东西 3：整理查询和键

    第三处不同很小。在 RoPE 转动它们之前，Qwen 先让每个头的查询和键各经过一个自己的 RMSNorm，各有 128 个增益：就是文件头里的 `q_norm` 和 `k_norm`。
    它让查询和键的点积保持在稳定的范围内，使大模型训练起来更平稳。这个函数你已经有了。

    ## 整个模型，用你的代码

    **来自前面的章节。**下面折叠起来的单元格里是第 20 章的 `rms_norm`、`rope` 和 `attend`。`rope` 现在把它的转速的底数作为输入，因为 Qwen 用的是 1,000,000，更适合较长的文本，而我们用的是 10,000。

    接着是一个块和整个模型。`layer_knobs(knobs, i)` 从 `knobs` 里挑出第 `i` 块的十一个张量，给它们起了简短的名字。你之前的模型调用 `nn.Linear` 的地方，这段代码直接乘以矩阵本身：`y @ L["q"].T`，一个没有偏置的 `nn.Linear` 做的就是这件事（第 10 章）。
    """)
    return


@app.function(hide_code=True)
def rms_norm(x, gain):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    return gain * x / rms


@app.function(hide_code=True)
def rope(x, base):
    words, size = x.shape[-2], x.shape[-1]
    half = size // 2
    speeds = 1 / base ** (torch.arange(half) / half)
    angles = torch.arange(words)[:, None] * speeds[None, :]
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * torch.cos(angles) - b * torch.sin(angles), a * torch.sin(angles) + b * torch.cos(angles)], dim=-1)


@app.function(hide_code=True)
def attend(Q, K, V):
    words = Q.shape[-2]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(K.shape[-1])
    mask = torch.triu(torch.ones(words, words, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, -torch.inf), dim=-1) @ V


@app.function
def block(x, L):
    words = x.shape[1]
    y = rms_norm(x, L["norm1"])
    q = (y @ L["q"].T).reshape(1, words, 16, 128).transpose(1, 2)
    k = (y @ L["k"].T).reshape(1, words, 8, 128).transpose(1, 2)
    v = (y @ L["v"].T).reshape(1, words, 8, 128).transpose(1, 2)
    q = rope(rms_norm(q, L["q_norm"]), 1000000)
    k = rope(rms_norm(k, L["k_norm"]), 1000000)
    blended = attend(q, share_heads(k, 2), share_heads(v, 2))
    x = x + blended.transpose(1, 2).reshape(1, words, 2048) @ L["out"].T
    y = rms_norm(x, L["norm2"])
    return x + (F.silu(y @ L["gate"].T) * (y @ L["up"].T)) @ L["down"].T


@app.function
def qwen_scores(knobs, ids):
    with torch.no_grad():
        x = knobs["model.embed_tokens.weight"][torch.tensor([ids])]
        for layer in range(28):
            x = block(x, layer_knobs(knobs, layer))
        x = rms_norm(x, knobs["model.norm.weight"])
        return x[0, -1] @ knobs["model.embed_tokens.weight"].T


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 动手算：跟着形状走

    读 `block`，设文本有 5 个词元。`x` 进来时形状是（1, 5, 1024）。`y @ L["q"].T` 的最后一个维度有多大？经过 reshape 和 transpose 之后，`q` 的形状是（1, 16, 5, 128）：`k` 在 `share_heads` 之前有几个头，之后有几个头？
    还有，`qwen_scores` 返回最后一个词元的分数：那是多少个数？
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['q 矩阵之后每个词元的数', 'k 之前的头数', 'k 之后的头数', '返回的分数个数'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'q 矩阵之后每个词元的数': 2048, 'k 之前的头数': 8, 'k 之后的头数': 16, '返回的分数个数': 151936}, {
        'q 矩阵之后每个词元的数': r"""16 个头，每个 128""",
        'k 之前的头数': r"""看它的 reshape""",
        'k 之后的头数': r"""`share_heads(k, 2)`""",
        '返回的分数个数': r"""词表里的每个词元一个""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 最初的几个词

    模型需要词元的 id，所以它需要自己的分词器。Qwen 的分词器就是第 19 章的方法，字节级 BPE，大约有 151,000 次合并，而你训练的只有几百次。`QwenTokenizer` 从下载的文件里读出合并表。它的 `encode` 和 `decode` 做的事和你的一样。

    现在是见分晓的时刻。我们给模型一句话的开头，看它给下一个词元打出的最高的五个分数（`torch.topk` 给出最大的几个值和它们的位置）：
    """)
    return


@app.cell
def _(folder, knobs):
    tokenizer = QwenTokenizer.load(folder)
    france_ids = tokenizer.encode("The capital of France is")
    top = torch.topk(qwen_scores(knobs, france_ids), 5)
    france_ids, [tokenizer.decode([i]) for i in top.indices.tolist()]
    return (tokenizer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    输入五个词元，五个候选里的第一个是 ` Paris`。

    想想刚才发生了什么。这个计算里没有哪一步来自懂语言模型的库。你写的文件读取器找到了 5.96 亿个数。你在第 20 章写的函数让五个向量穿过了 28 个块。
    然后，一条关于法国的事实就出来了。语言模型就是这些，再没有别的了。

    ## 写出一个回答

    一个词元还不是回答。第 17 章的 `write` 靠把模型的选择加进输入、再问一次来写文本，这个循环对每个语言模型都一样。对一个用来对话的模型，有两件事是新的。

    **知道何时停下。**你的故事模型一直写到长度用完。聊天模型在回答结束时会写出一个**停止词元**（stop token），循环必须留意它。Qwen 的是词元 151,645，写作 `<|im_end|>`，存在 `END_OF_TURN` 这个名字里。

    **知道谁在说话。**训练时，这个模型见到的对话都写成一种固定的格式，叫它的**对话模板**（chat template），用特殊的词元标出每位说话者的发言：

    ```
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant
    ```

    对模型来说，以这样的方式结尾的文本，就是接下来该由助手来回答的文本。“聊天”不过如此：和以前一样接着写文本，只是开头看上去像一场对话。`tokenizer.chat(question)` 把一个问题包进这种格式。

    ## 练习 5：generate

    写 `generate(next_scores, ids, limit, stop)`。`next_scores` 是一个函数：给它一个词元 id 的列表，它返回下一个词元的分数，是一个张量，词表里的每个词元一个数。`ids` 是起始用的词元 id 列表。
    最多重复 `limit` 次：取分数最高的词元，这叫贪心选择（greedy choice）。如果它是 `stop`，立刻结束，不把它加进去。否则把它加进 id 列表。返回完整的列表，包括开头的部分。

    不要改动传给你的那个列表：`ids + [token]` 造出一个新的、更长的列表，而 `ids.append(token)` 会改动调用者的列表。

    如果 `next_scores` 总是偏向最后一个 id 加 1，那么从 `[3]` 出发、`stop = 7`，必须得到 `[3, 4, 5, 6]`。
    """)
    return


@app.function
def generate(next_scores, ids, limit, stop):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "generate", generate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""一个运行 `limit` 次的 `for` 循环。每一轮：用目前的 ids 去问 `next_scores`，用 `torch.argmax` 找到最大分数的位置，再用 `.item()` 把它变成普通的数。然后要么离开函数（第 21 章的提前 `return`），要么把列表加长。""",
        r"""6 行。循环里面：`token = torch.argmax(next_scores(ids)).item()`，然后 `if token == stop:`，下面是 `return ids`，再然后 `ids = ids + [token]`。循环之后再来一个 `return ids`，用在次数用完的时候。""",
    ], LOCALE)
    return


@app.cell
def _(knobs, tokenizer):
    def next_scores(ids):
        return qwen_scores(knobs, ids)

    question_ids = tokenizer.chat("What is the capital of France?")
    started = time.perf_counter()
    answer_ids = generate(next_scores, question_ids, 40, END_OF_TURN)
    seconds = time.perf_counter() - started
    tokenizer.decode(answer_ids[len(question_ids):]), len(answer_ids) - len(question_ids), round(seconds, 1)
    return (next_scores,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    模型的回答，它用了多少个词元，以及花了多少秒。它说出了 Paris，并且自己停了下来。你正在通过自己的代码和一个真实的模型对话。把单元格里的问题改一改，再运行一次。
    （`tokenizer.chat` 还加了一个简短的标记，告诉这个模型直接回答。没有它，Qwen3 会先写出一大段思考过程。）

    ## 不总是选最高的词元

    贪心选择每次给出同样的答案，对开放的问题，它给出的文本呆板又重复。第 13 章按模型给出的概率随机抽取下一个词。真实的模型用两个控制项来做这件事。

    **温度**（temperature）。你在第 14 章见过它：softmax 之前，把每个分数除以一个数 $T$。$T$ 小于 1 时，分数之间的差距变大，排在前面的词元得到更多的概率：文本更稳妥、更平淡。$T$ 大于 1 时，差距缩小：文本更出人意料，错误也更多。

    **Top-k。**一个有 151,936 个词元的词表有一条很长的尾巴，里面的每个词元概率都很小，但加在一起，被抽中的次数多到足以让一句话跑偏。所以只保留概率最高的 $k$ 个词元，把其他的概率设为 0，再把保留下来的按比例放大，让它们重新加起来等于 1。

    ## 动手算：保留前两个

    四个词元的概率是 0.5、0.3、0.15 和 0.05。保留前 2 个。之后第一个和第二个词元的概率各是多少？
    """)
    return


@app.cell(hide_code=True)
def _():
    topk_boxes = hand.boxes(['第一个词元', '第二个词元'], step=0.001)
    return (topk_boxes,)


@app.cell(hide_code=True)
def _(topk_boxes):
    hand.marked(topk_boxes, {'第一个词元': 0.625, '第二个词元': 0.375}, {
        '第一个词元': r"""保留下来的概率加起来是 0.8，所以要除一下：$0.5 / 0.8$""",
        '第二个词元': r"""$0.3 / 0.8$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 工具箱：最大的几个，以及按位置存入
    - `top = torch.topk(x, 2)` 找出 `x` 里最大的 2 个数：`top.values` 装着它们，`top.indices` 装着它们的位置。对 `x = torch.tensor([5.0, 9.0, 7.0])`，它们是 `[9, 7]` 和 `[1, 2]`。
    - 按一个位置列表也可以存入（第 09 章）：对 `z = torch.zeros(3)`，这一行 `z[top.indices] = top.values` 把 `z` 变成 `[0, 9, 7]`。
    - `torch.zeros_like(x)` 是一个全 0 的张量，形状和 dtype 与 `x` 相同（第 15 章）。
    ///

    ## 练习 6：top_k_chances

    写 `top_k_chances(scores, temperature, k)`。`scores` 是一个张量，每个词元一个分数。它返回一个形状相同的张量，装着用来抽取的概率：

    1. 分数除以温度之后的 softmax（`torch.softmax(..., dim=-1)`）；
    2. 除了最大的 `k` 个概率，其他都设为 0；
    3. 结果除以它自己的和。

    在你手算的例子上，分数的 softmax 是 0.5、0.3、0.15、0.05，温度为 1、`k = 2`，它必须得到 `[0.625, 0.375, 0, 0]`。
    """)
    return


@app.function
def top_k_chances(scores, temperature, k):
    # ← 把下面的 todo() 换成你的代码
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "top_k_chances", top_k_chances, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""按三个步骤的顺序来做。第 2 步别想着把小概率去掉：从一个全 0 的张量出发，把最大的 `k` 个概率按它们自己的位置复制进去。工具箱里展示了这个做法。""",
        r"""5 行：概率，对它们做 `torch.topk`，一个 `torch.zeros_like`，把 `top.values` 存到 `top.indices` 位置的那一行，以及除以 `torch.sum`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的 `sample` 就是你的 `generate` 改了一行：词元是按你算出的概率抽出来的，而不是取最高的那个。`torch.multinomial(chances, 1)` 按给定的概率抽出一个位置，就像第 13 章的 `rng.choice` 那样。
    这个单元格用的是 Qwen 的制作者推荐的设置，温度 0.6 和 top 20，用在一个有许多好答案的问题上。
    """)
    return


@app.function
def sample(next_scores, ids, limit, stop, temperature, k):
    for step in range(limit):
        token = torch.multinomial(top_k_chances(next_scores(ids), temperature, k), 1).item()
        if token == stop:
            return ids
        ids = ids + [token]
    return ids


@app.cell
def _(next_scores, tokenizer):
    torch.manual_seed(0)
    sea_ids = tokenizer.chat("Write one sentence about the sea.")
    tokenizer.decode(sample(next_scores, sea_ids, 40, END_OF_TURN, 0.6, 20)[len(sea_ids):])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    换一个种子，它会写出不同的句子。也试试温度 1.5 和 0.1。

    ## 它的代价

    回头看看你的 `generate` 用了多少秒，再看看它是怎么工作的。要写第 30 个词元，它让 28 个块读一遍问题**加上**已经写出的 29 个词元，然后除了一行分数，其余全部扔掉。写第 31 个词元时，这一切又做一遍，还要多一个。
    每个新词元的工作量随文本的长度增长，尽管每次只有一个词元是新的。

    对小模型写几十个词元，这还能忍受。对大模型的一个长回答，就是灾难，没有人这样部署服务。
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("权重文件", "1.5 GB"), ("旋钮", "5.96 亿"), ("分数", "151,936"), ("词元", "1"), ("回答", "文本")], ["你的读取器", "你的块", "贪心或 top-k", "重复"], "从磁盘上的文件到一个回答，用你写的代码")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 接下来

    **第 28 章**让单个请求变快。它的主要想法是不再重复计算：已经读过的词元的键和值永远不会变，所以可以保存起来。这就是今天的共用头派上用场的地方。

    ## 你现在会做的事

    - 说出权重文件里有什么，并亲手读出它的文件头和张量
    - 解释为什么同样的字节在不同的 dtype 下是不同的数
    - 读懂一个模型的配置和名字，并把它们对应到 transformer 的各个部分
    - 解释共用词向量表和分组查询注意力，以及各自省下了什么
    - 用你自己的代码运行一个已发布的模型
    - 写一个带停止词元的解码循环，并解释什么是对话模板
    - 用温度和 top-k 控制模型写出的内容

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** *safetensors* 格式由 Hugging Face 制定，用来安全地取代那些一打开就可能运行代码的文件。分组查询注意力出自 Ainslie 等（2023）。
    Qwen3 在 *Qwen3 Technical Report*（2025）中有介绍。另一个常见的控制项是 *top-p*，它保留概率加起来达到 $p$ 的最小词元集合。课程自己关于这个模型的声明在 `learn_llm/data/QWEN-NOTICE.md`。
    """)
    return


if __name__ == "__main__":
    app.run()
