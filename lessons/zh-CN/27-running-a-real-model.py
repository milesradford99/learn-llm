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

    第三部分结束时，你已经亲手搭好并训练出了自己的语言模型。它大约有 100 万个旋钮，能写点简单的幼儿小故事。人们日常在用的大模型也是按同样的方法造出来的，只是每样东西的规模都大了上千倍到上百万倍。
    第四部分讲模型生命的另一半：**部署服务**（serving），也就是把训练好的模型跑起来，又快又省地响应每一个请求。

    这一章从最直接的做法开始：拿一个实验室公开发布的真实模型，用你自己写的代码把它跑起来。不借助帮你包办一切的高层库——你要亲手打开权重文件，自己把里面的数字读出来，再送进你在第 20 章写过的 `rms_norm`、`rope`、`swiglu` 和 `attend`。如果前面学的全都能串起来，它会真的回答你的问题。

    我们要跑的模型是阿里巴巴 Qwen 团队发布的 **Qwen3-0.6B**：大约有 6 亿个旋钮。在公开发布的模型里它算小个头，但已经比你之前训练的模型大了 500 倍。

    ## 权重文件是什么

    训练结束时，模型留下的是一组调好的旋钮，第 21 章我们曾用 `torch.save` 把它们存下来。实验室发布模型时，也是把这些旋钮打包成一个文件供大家下载。训练好的旋钮通常叫作**权重**（weights），装权重的文件就叫**权重文件**（weight file）。

    最常见的格式叫 *safetensors*，结构简单到可以直接手工解析。整个文件按顺序分成三段：

    1. **长度区（8 个字节）**：存着一个整数 $n$，表示后面文件头有多长；
    2. **JSON 文件头（$n$ 个字节）**：一段文本，列出文件里每个张量的名字、数据类型、形状，以及它的字节区间；
    3. **张量数据区**：把所有张量的原始字节紧挨着排在一起，中间没有任何间隔。

    ```
    [0..7: 长度 n] | [8..8+n-1: JSON 文件头] | [8+n..: 张量数据区 (data_offsets 从这里重新从 0 算起)]
    ```

    ## 把字节变成一个数

    第 19 章说过，一个字节是 0 到 255 之间的整数。要表示更大的数，就要把几个字节拼在一起，道理和十进制的个位、十位、百位一样，只不过这里每高一位是前一位的 256 倍，而不是 10 倍。
    在如今绝大多数文件格式里，**第一个**字节放在最低位，这种顺序叫**小端序**（little-endian）。所以两个字节 5, 2 表示 $5 + 2 \times 256 = 517$。如果字节更多，后面的高位字节就依次乘以 $256^2, 256^3, \dots$；如果某个字节是 0，它就不贡献数值。

    ## 动手算：读出一个长度

    一个文件以 8 个字节 2, 1, 0, 0, 0, 0, 0, 0 开头。它表示哪个数？那字节 0, 2, 0, 0, 0, 0, 0, 0 又是哪个数？
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
    `int.from_bytes(data, "little")` 会按小端序（第一个字节为最低位）把一串字节读成一个整数：比如 `int.from_bytes(bytes([5, 2]), "little")` 得到 517。如果传 `"big"`，第一个字节就会被当成最高位。
    ///

    ## 练习 1：header_length

    写 `header_length(first_bytes)`。`first_bytes` 是权重文件的前 8 个字节，函数按小端序返回它们表示的整数。

    `header_length(bytes([2, 1, 0, 0, 0, 0, 0, 0]))` 必须返回 258。
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
        r"""工具箱里已经给出了现成工具，那里演示的是 2 个字节，用在 8 个字节上完全一样。""",
        r"""只要一行：调用 `int.from_bytes`，传入 `first_bytes` 和 `"little"`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 文件头

    文件头是一段 **JSON** 格式的文本，看起来很像写出来的 Python 字典。如果文件里只有一个小张量，文件头大概长这样：

    ```
    {"gain": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}
    ```

    每个张量对应一条记录：`dtype` 是数据类型，`shape` 是形状，`data_offsets` 标出了这个张量在数据区里的起止字节位置——和 Python 切片一样**左闭右开**（这里的 `[0, 6]` 表示第 0 到第 5 个字节，不含 6，共 6 个字节）。注意，`data_offsets` 是**从数据区开头重新从 0 算起**的，不是从整个文件开头算起。
    张量数据区紧跟在文件头后面，从文件的第 $8 + n$ 个字节开始。

    /// admonition | 🧰 工具箱：读文件
    - `with open(path, "rb") as f:` 以二进制只读方式打开文件（`"rb"` 表示 read binary）。缩进块里的代码可以通过 `f` 读取内容，缩进块结束后文件会自动关闭。
    - `f.read(4)` 读出接下来的 4 个字节，并把读取位置往后移 4 个字节；再调用一次 `f.read(4)`，就会接着读后面的 4 个字节。
    - `f.seek(100)` 把读取位置直接跳到文件的第 100 个字节，下一次 `read` 就从那里开始。
    - `json.loads(data)` 把 JSON 文本（或它的字节串）解析成 Python 的字典和列表，比如 `json.loads('{"a": [1, 2]}')` 会得到 `{"a": [1, 2]}`。本页开头已经导入了 `json`。
    ///

    ## 练习 2：read_header

    写 `read_header(path)`。`path` 是权重文件的路径，函数返回一对值：解析成字典的文件头，以及张量数据区在文件中的起始位置。

    例如，如果一个文件的前 8 个字节表示 258，那么后面的 258 个字节就是文件头，数据区从第 266 个字节开始。
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
        r"""在一个 `with open(path, "rb") as f:` 块里读两次：先读 8 个字节交给你的 `header_length` 算出长度，再按这个长度读出后面的 JSON 字节。""",
        r"""一共 3 行：把长度存进变量 `n`，然后返回 `json.loads(f.read(n)), 8 + n`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 把字节变成带小数点的数

    数据区里只是一串原始字节。在说明它们装的是哪一种数据类型之前，字节本身并没有固定含义。下面用三种不同类型来读同一串 4 个字节：`torch.frombuffer` 可以按指定的 `dtype` 直接把字节转成张量（它要求先用可修改的 `bytearray` 把字节包起来）。
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
    同样的 4 个字节，按 `uint8` 读是四个小整数 `[0, 0, 128, 63]`；按 `int16` 读是两个两字节整数（前两字节 `[0, 0]` 对应 `0`，后两字节 `[128, 63]` 按小端序算得 $128 + 63 \times 256 = 16256$）；按第 10 章的 32 位浮点数 `float32` 读，则是单独一个小数 1.0。如果用错了 `dtype`，权重文件读出来就是一团乱码——这就是为什么文件头里必须给每个张量标明 `dtype`。

    权重文件对数据类型用的是简写，本页提供的字典 `TYPES` 会把它们对应到 PyTorch 的类型：`TYPES["F32"]` 是 `torch.float32`，`TYPES["BF16"]` 是 `torch.bfloat16`（也就是第 22 章那种 16 位宽、数值范围和 float32 一样大的格式）。
    实验室用 16 位发布模型，是为了让下载体积减半。不过第 22 章也测过，CPU 做 16 位运算反而很慢，所以我们在读取时统一把所有张量转成 32 位浮点数。

    ## 动手算：找到一个张量

    一个文件的前 8 个字节表示 100。它的文件头里有一项 `{"dtype": "BF16", "shape": [2, 2], "data_offsets": [40, 48]}`（`data_offsets` 和 Python 切片一样左闭右开，不含 48）。这个张量从文件的第几个字节开始？一共占多少个字节？平均每个数占多少个字节？
    """)
    return


@app.cell(hide_code=True)
def _():
    find_boxes = hand.boxes(['在文件里的第一个字节', '一共的字节数', '每个数的字节数'])
    return (find_boxes,)


@app.cell(hide_code=True)
def _(find_boxes):
    hand.marked(find_boxes, {'在文件里的第一个字节': 148, '一共的字节数': 8, '每个数的字节数': 2}, {
        '在文件里的第一个字节': r"""8 个字节存长度，100 个字节是文件头，再往数据区里走 40 个字节""",
        '一共的字节数': r"""从 40 到 48，左闭右开，不含 48""",
        '每个数的字节数': r"""形状说明共有 $2 \times 2 = 4$ 个数""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    每个数刚好占 2 个字节，也就是 16 位，和 `BF16` 对得上。

    ## 练习 3：read_tensor

    写 `read_tensor(path, entry, data_start)`。`entry` 是文件头里某一个张量的记录字典，包含 `"dtype"`、`"shape"` 和 `"data_offsets"` 三个键；`data_start` 是数据区在文件中的起始位置（也就是 `read_header` 返回的第二个值）。
    函数返回这个张量，形状正确，类型为 32 位浮点数：

    1. 跳到这个张量的起始字节，恰好读出属于它的那些字节；
    2. 用 `torch.frombuffer(bytearray(data), dtype=...)` 把字节转成张量，`dtype` 通过 `TYPES` 查表得到；
    3. 用 `.reshape(...)` 恢复它的形状，再用 `.float()` 转成 32 位浮点数。
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
        r"""前面的手算题其实就是第 1 步：起始位置是 `data_start` 加上第一个偏移量，要读的字节数是第二个偏移量减去第一个。可以用 `start, stop = entry["data_offsets"]` 把它们拆出来。""",
        r"""大约 5 行：在 `with open(path, "rb") as f:` 里先 `f.seek(data_start + start)`，再 `data = f.read(stop - start)`；接着一行 `torch.frombuffer(bytearray(data), dtype=TYPES[entry["dtype"]])`，再链式接上 `.reshape(entry["shape"])` 和 `.float()`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 获取模型

    现在来读真实的文件。点击下面的按钮，会把 Qwen3-0.6B 的三个文件下载到 `my-work/.models/`：权重文件（1.5 GB）、分词器合并表，以及一个很小的配置文件。只需要联网这一次，耗时从几分钟到更久不等，取决于你的网速。单元格下方的进度条会实时显示已下载的兆字节（MB）；下载期间页面会暂停响应。如果中途断开了，再点一次按钮就会从断点继续，每个文件下载完都会自动校验。

    运行这个模型大约需要 4 GB 空闲内存。如果当前设备不方便下载或运行，前面你写的三个函数依然可以正常通过检查，后面的内容也可以先直接阅读。
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
    这就是你的 `read_header` 读取真实模型的结果：文件头一共列了 312 项，数据区从第 35,560 个字节开始，上面还打印出了其中一项。所有张量的命名都有规律：比如 `model.layers.0.self_attn.q_proj.weight`，指的是第 0 块自注意力里用来生成查询（query）的权重矩阵。
    在 Qwen 的命名里，块被叫作**层**（layer），编号从 0 到 27。

    模型的其余参数写在配置文件里，同样是 JSON 格式：
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
    拿它和你第 21 章自己训练的模型比一比（当时是 2,258 个词元、每个词元 128 维、3 个块、4 个头）：

    | | 你的模型 | Qwen3-0.6B |
    |---|---:|---:|
    | 词表里的词元数 | 2,258 | 151,936 |
    | 每个词元的维数（`hidden_size`） | 128 | 1,024 |
    | 块数（层数） | 3 | 28 |
    | 注意力头数 | 4 | 16 |
    | SwiGLU 隐藏维数 | 341 | 3,072 |

    整体架构和你之前亲手搭的一模一样：每个子块前加 RMSNorm，给查询和键做 RoPE，前馈层用 SwiGLU，全都不带偏置。多出来的只有三个小改动，下面三节逐个来看。

    ## 新改动 1：一张表，两头共用

    配置里的 `tie_word_embeddings` 为 `True`。语言模型的首尾各有一张表：输入端用一张表把词元查表变成向量，输出端用一张表把最终向量投影成词表里每个词元的得分，两张表的形状都是 `(词元数, hidden_size)`。
    Qwen 在首尾直接**共用同一张表**：每个词元的最终得分，就是输出向量和它自己的词向量的点积。这种做法叫**共享词向量表**（tied embeddings），在这里直接省下了 1.56 亿个旋钮。

    权重文件里虽然给这张表列了两个名字，我们可以用你刚写的 `read_tensor` 验证一下，里面存的其实是不是完全相同的数：
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
    只有每个位置的数完全一致，`torch.equal` 才会返回 `True`。既然两张表完全一样，我们就省去重复读取 `lm_head.weight`，同时跳过文件头里唯一不是张量的元信息项 `__metadata__`：
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
    310 个张量、596,049,920 个数字，你写的代码几秒钟就全部读进了内存。

    ## 新改动 2：多个查询头共用键和值

    仔细看文件头里给出的形状。注意：前面你自己写的模型里，头数乘以每头维数刚好等于 `hidden_size`，而 Qwen3 把 16 个查询头拉宽到了 $16 \times 128 = 2048$（比 `hidden_size = 1024` 大一倍），注意力算完后再用 `out` 矩阵压回 1,024。与此同时，它的键和值矩阵只输出 $8 \times 128 = 1024$ 个数——也就是只有 **8 个头**。

    这 16 个查询头两两分组，共用这 8 组键头和值头：查询头 0 和 1 共看键头 0，查询头 2 和 3 共看键头 1，依此类推。这叫**分组查询注意力**（grouped-query attention，简称 GQA）。
    人们发现，这样做对模型效果几乎没有损失。它省下来的开销在这一章还不算显眼，到下一章就会派上大用场——到那时，已经读过的所有词元的键和值都要留在内存里。

    ## 动手算：谁和谁共用

    有 16 个查询头和 8 个键头，查询头 5 用的是哪个键头？查询头 15 呢？一个块里的键矩阵和值矩阵，比起各自用满 16 个头，合起来一共省下了多少个数？
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
    写代码时，最省事的“共用”办法就是复制：把 8 个键头和值头各自重复一遍扩成 16 个，这样你之前写的 `attend` 一行都不用改就能直接用。

    /// admonition | 🧰 工具箱：沿某个维度重复
    `torch.repeat_interleave(x, 2, dim=0)` 会沿着指定维度把每一项连续重复，副本紧挨在一起：对 `x = torch.tensor([7, 8, 9])`，它返回 `[7, 7, 8, 8, 9, 9]`。
    ///

    ## 练习 4：share_heads

    写 `share_heads(x, times)`。`x` 的形状是 `(批大小, 头数, 词数, 每头维数)`，和第 16 章 `split_heads` 之后的键和值一样。函数返回一个头数扩大为 `times` 倍的张量，把 `x` 的每个头连续重复 `times` 次：
    比如原本有 3 个头、`times = 2` 时，结果的头依次是 `x` 的头 0、0、1、1、2、2。
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
        r"""把工具箱里的函数用在正确的维度上。在 `(批大小, 头数, 词数, 每头维数)` 里，“头数”排在第几个位置？""",
        r"""只要一行：`torch.repeat_interleave(x, times, dim=1)`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 新改动 3：给查询和键先做归一化

    第三处改动很小。在做 RoPE 旋转之前，Qwen 会先让每个头的查询和键各自经过一次独立的 RMSNorm，各有 128 个缩放系数——就是文件头里的 `q_norm` 和 `k_norm`。
    它能把查询和键的点积控制在稳定范围内，大模型训练时更不容易失控。`rms_norm` 函数你早就写过了。

    ## 用你的代码跑起整个模型

    **复用前面章节的函数。** 下面折叠起来的单元格里是第 20 章的 `rms_norm`、`rope` 和 `attend`。`rope` 现在把转速底数 `base` 作为参数传进来，因为我们之前用的是 10,000，而 Qwen 用的是 1,000,000，更适合处理较长的文本。

    接着是一个块和整个模型的前向计算。`layer_knobs(knobs, i)` 会从 `knobs` 里挑出第 `i` 块的 11 个张量并起好简短的名字。还记得第 10 章的约定吗？权重矩阵按 `(输出维, 输入维)` 存放，所以右乘输入时要写 `.T`——之前用过的不带偏置的 `nn.Linear`，做的就是 `y @ L["q"].T` 这一步。

    顺着 `block` 里的关键形状流转看一遍就清楚了：输入 `x` 的形状是 `(1, words, 1024)`；投影出查询并拆成 16 个头后变成 `(1, 16, words, 128)`（`rms_norm` 默认作用在最后一维 `128` 上，刚好对每个头各自归一化）；注意力算完再把 16 个头拼回 `(1, words, 2048)`，最后右乘 `L["out"].T` 压回 `(1, words, 1024)`，才能和残差 `x` 相加。
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

    读一遍上面的 `block`，设输入文本有 5 个词元，`x` 进来时形状是 `(1, 5, 1024)`。`y @ L["q"].T` 的最后一维是多少？经过 reshape 和 transpose 之后，`q` 的形状是 `(1, 16, 5, 128)`；请问 `k` 在 `share_heads` 之前有几个头，之后有几个头？
    还有，`qwen_scores` 返回最后一个词元的预测分数，这会是多少个数？
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

    模型只认词元 ID，所以还得用它自己的分词器把文字切开。Qwen 用的和第 19 章的方法完全一样，都是字节级 BPE，只是它做了约 151,000 次合并，而你之前只训练了几百次。`QwenTokenizer` 会从下载好的文件里加载合并表，它的 `encode` 和 `decode` 用法和你的完全一致。

    现在来亲眼验证一下。我们给模型一句没说完的话，看看它给下一个词元打出的前五个最高分（`torch.topk` 会返回最大的几个值以及它们的位置）：
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
    输入 5 个词元，排在第一的候选词就是 ` Paris`。

    停下来想想刚才发生了什么。整个计算过程里，没有一行代码来自封装好的大模型库：是你亲手写的文件读取器从磁盘里读出了 5.96 亿个数，是你在第 20 章写的几个基础函数推着 5 个向量穿过了 28 个块——然后，一条关于法国首都的事实就这么算出来了。大语言模型的底层秘密，全在这里，再没有别的黑魔法。

    ## 写出完整回答

    蹦出一个词元还算不上回答。第 17 章的 `write` 把模型选出的词元拼回输入末尾、再算下一个，这个循环对所有语言模型都一样。不过对一个专门用来对话的模型，有两件事是新的。

    **知道何时停下。** 你之前写的讲故事模型会一直写到字数上限为止。对话模型在回答结束时会输出一个专门的**停止词元**（stop token），生成循环一看到它就该立刻停下。Qwen 的停止词元是第 151,645 号，写作 `<|im_end|>`，已经存在常量 `END_OF_TURN` 里。

    **知道轮到谁说话。** 训练时，这个模型见过的对话都遵循一套固定格式，叫作**对话模板**（chat template），用特殊词元标出每一轮发言的起点和终点：

    ```
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant
    ```

    在模型眼里，以这样结尾的文本，后面紧跟着的就该是助手的回答。“聊天”不过如此：本质上还是和以前一样接着续写文本，只是开头被包装成了对话的样子。`tokenizer.chat(question)` 会把你的问题自动套进这个模板里。

    ## 练习 5：generate

    写 `generate(next_scores, ids, limit, stop)`。`next_scores` 是一个函数：传入一个词元 ID 列表，它返回一个张量，装着词表里每个词元作为下一个词的分数。`ids` 是起始的词元 ID 列表。
    最多循环 `limit` 次：每一步选出分数最高的词元（这叫**贪心选择**，greedy choice）。如果它刚好等于 `stop`，就立刻结束，不要把它加进去；否则把它拼到 ID 列表末尾。最后返回包含开头在内的完整列表。

    不要原地修改传入的列表：`ids + [token]` 会造出一个更长的新列表，而 `ids.append(token)` 会改动调用者原本的列表。

    如果 `next_scores` 总是给“最后一个 ID 加 1”打最高分，那么从 `[3]` 出发、`stop = 7`，函数必须返回 `[3, 4, 5, 6]`。
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
        r"""写一个最多运行 `limit` 次的 `for` 循环。每轮用当前的 `ids` 调用 `next_scores`，用 `torch.argmax` 找到最高分的位置，再用 `.item()` 转成普通整数。然后要么提前 `return` 离开函数（第 21 章），要么把新词元拼进列表。""",
        r"""一共 6 行：循环里先写 `token = torch.argmax(next_scores(ids)).item()`，接着 `if token == stop:` 下写 `return ids`，再写 `ids = ids + [token]`。循环结束后再写一次 `return ids`，用来处理步数用完的情况。""",
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
    上面打印出了模型的回答、生成的词元数，以及耗时（秒）。它答出了 Paris，并且自己知道什么时候停下——你正在用自己写的代码和一个真实的模型对话。不妨改一改单元格里的问题再运行一次。
    （`tokenizer.chat` 还顺手加了一个简短标记，让这个模型直接回答；如果不加，Qwen3 会先输出一大段内部思考过程。）

    ## 不必每次都选最高分

    贪心选择每次都会给出完全相同的回答，遇到开放式问题时，写出来的文字往往呆板又爱重复。第 13 章我们曾按模型给出的概率随机抽取下一个词，真实的模型通常会再配上两个旋钮来控制抽样。

    **温度（temperature）。** 你在第 14 章见过它：做 softmax 之前，先把每个分数除以一个数 $T$。当 $T < 1$ 时，分数之间的差距会被拉大，排在前面的词元占到更大比重，文字更稳妥、但也更平淡；当 $T > 1$ 时，差距会被缩小，文字更出人意料，但也更容易出错。

    **Top-k。** 151,936 个词元的词表拖着一条很长的尾巴：排在后面的成千上万个冷门词，单个概率极小，但加起来的概率仍然不小，一不小心抽中就会把整句话带偏。解决办法是只保留概率最高的 $k$ 个词元，把其他词元的概率清零，再把留下的概率按比例放大，让它们重新加起来等于 1。

    ## 动手算：保留前两个

    四个词元的概率分别是 0.5、0.3、0.15 和 0.05。只保留前 2 个后，第一个和第二个词元的概率各是多少？
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
    /// admonition | 🧰 工具箱：挑出最大的几个，以及按位置写入
    - `top = torch.topk(x, 2)` 会找出 `x` 里最大的 2 个数：`top.values` 装着它们的值，`top.indices` 装着它们的位置。对 `x = torch.tensor([5.0, 9.0, 7.0])`，结果分别是 `[9, 7]` 和 `[1, 2]`。
    - 按一个位置列表也能批量赋值（第 09 章）：对 `z = torch.zeros(3)`，执行 `z[top.indices] = top.values` 后，`z` 会变成 `[0, 9, 7]`。
    - `torch.zeros_like(x)` 会创建一个与 `x` 形状和 `dtype` 相同的全 0 张量（第 15 章）。
    ///

    ## 练习 6：top_k_chances

    写 `top_k_chances(scores, temperature, k)`。`scores` 是一个一维张量，每个词元对应一个分数。函数返回一个形状相同的张量，装着用来抽样的概率：

    1. 把分数除以温度后做 softmax（`torch.softmax(..., dim=-1)`）；
    2. 除了最大的 `k` 个概率，其余全部设为 0；
    3. 把结果除以它自身的总和。

    在前面手算的例子上，如果分数的 softmax 结果是 0.5、0.3、0.15、0.05，温度为 1、`k = 2`，函数必须返回 `[0.625, 0.375, 0, 0]`。
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
        r"""按三个步骤的顺序写即可。第 2 步不要想着去删掉小概率：先建一个全 0 张量，再把最大的 `k` 个概率按原位置填进去，工具箱里演示的就是这个写法。""",
        r"""一共 5 行：算出概率，对它做 `torch.topk`，建一个 `torch.zeros_like`，把 `top.values` 写入 `top.indices` 对应的位置，最后除以 `torch.sum`。""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下面的 `sample` 只是把你的 `generate` 改了一行：不再硬选最高分，而是按你算出的概率随机抽取。`torch.multinomial(chances, 1)` 会按给定概率抽出一个位置，作用和第 13 章的 `rng.choice` 一样。
    这个单元格采用 Qwen 官方推荐的设置——温度 0.6、top-k 取 20，拿一个有无数种写法的开放问题试一试：
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
    换个随机种子，它就会写出不一样的句子；也可以试试把温度调到 1.5 或 0.1，感受风格的变化。

    ## 这样跑的代价

    回头看看刚才 `generate` 花了几秒，再想想它到底干了多少活。为了生成第 30 个词元，它把问题**加上**已经写出的 29 个词元全部塞进 28 个块里重算了一遍，最后却只留下末尾那一行分数，其余全部扔掉；生成第 31 个词元时，又把这一切重来一遍，还多算一个词元。
    虽然每次只新增了一个词元，每一步的计算量却随着文本变长一路水涨船高。

    用小模型写几十个词元，这样还勉强扛得住；要是换到大模型写长回答，耗时就会变成灾难——现实中没人会这样部署服务。
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

    **第 28 章**会让单次请求变快。核心思路很简单——不再重复计算：已经读过的词元，它们的键和值永远不会变，完全可以存下来复用。今天学的分组查询注意力（共用头），到了那里就会真正显出威力。

    ## 你现在会做的事

    - 说清权重文件的结构，并亲手读出它的文件头和张量
    - 解释为什么同一串字节在不同的 `dtype` 下会读出不同的数
    - 读懂真实模型的配置和张量命名，并把它们对应到 Transformer 的各个部分
    - 解释共享词向量表和分组查询注意力，以及各自省下了什么
    - 用自己写的代码把公开发布的真实模型跑起来
    - 写出带停止词元的解码循环，并解释对话模板的作用
    - 用温度和 top-k 控制模型写出的内容

    ---

    <!-- go-deeper -->
    **深入了解（选读）：** *safetensors* 格式由 Hugging Face 制定，用来安全地取代那些一打开就可能执行代码的文件格式。分组查询注意力出自 Ainslie 等（2023）。
    Qwen3 在 *Qwen3 Technical Report*（2025）中有详细介绍。另一个常见的抽样控制是 *top-p*，它保留概率之和达到 $p$ 的最小词元集合。本课程关于该模型的说明见 `learn_llm/data/QWEN-NOTICE.md`。
    """)
    return


if __name__ == "__main__":
    app.run()
