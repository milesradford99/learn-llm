import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="27 · Running a real model")

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

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 27 · Running a real model

    Part 3 ended with a model you built and trained yourself. It has about a million knobs and writes toddler stories. The models people actually use were built the same way, with a thousand to a million times more of everything.
    Part 4 is about the other half of a model's life: **serving** it, which means running a finished model for the people who use it, quickly and cheaply.

    This chapter starts with the plainest form of serving. You'll take a real model, published by the laboratory that trained it, and run it with code you wrote. Not with a library that hides it: you'll open the file, read the numbers out of it yourself,
    and feed them through chapter 20's `rms_norm`, `rope`, `swiglu` and `attend`. If everything we've built holds up, it will answer you.

    The model is **Qwen3-0.6B**, from Alibaba's Qwen team: about 600 million knobs, small for a published model and five hundred times bigger than yours.

    ## What a weight file is

    Training ends with a set of knobs, and chapter 21 saved them with `torch.save`. To publish a model, a laboratory puts the knobs in a file and anyone can download it. People call the knobs of a finished model its **weights**, and the file a **weight file**.

    The usual format is called *safetensors*, and it's simple enough to read by hand. The file has three parts, laid out one after another:

    1. **8 bytes** that hold one whole number, $n$;
    2. **a header** of $n$ bytes: text that lists every tensor in the file with its name, its kind of number, its shape, and where its bytes are;
    3. **the data**: the numbers of all the tensors, one tensor after another, with nothing between them.

    ```
    [0..7: length n] | [8..8+n-1: JSON header] | [8+n..: tensor data (data_offsets start at 0 here)]
    ```

    ## Bytes into a number

    Chapter 19 said a byte is a whole number from 0 to 255. Bigger numbers take several bytes, read like the digits of a number, except that each place is worth 256 times the one before instead of 10 times.
    And in nearly all files today the **first** byte is the smallest place, an order called **little-endian**. So the two bytes 5, 2 mean $5 + 2 \times 256 = 517$. Any further bytes are multiplied by $256^2, 256^3, \dots$ in turn—and any byte that is 0 adds nothing to the total.

    ## By hand: read a length

    A file starts with the 8 bytes 2, 1, 0, 0, 0, 0, 0, 0. What number is that? And what number would the bytes 0, 2, 0, 0, 0, 0, 0, 0 be?
    """)
    return


@app.cell(hide_code=True)
def _():
    byte_boxes = hand.boxes(['bytes 2, 1, 0, …', 'bytes 0, 2, 0, …'])
    return (byte_boxes,)


@app.cell(hide_code=True)
def _(byte_boxes):
    hand.marked(byte_boxes, {'bytes 2, 1, 0, …': 258, 'bytes 0, 2, 0, …': 512}, {
        'bytes 2, 1, 0, …': r"""$2 + 1 \times 256$""",
        'bytes 0, 2, 0, …': r"""$0 + 2 \times 256$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: bytes into a whole number
    `int.from_bytes(data, "little")` reads bytes as one whole number, first byte smallest: `int.from_bytes(bytes([5, 2]), "little")` is 517. With `"big"` the first byte would be the largest place.
    ///

    ## Exercise 1: header_length

    Write `header_length(first_bytes)`. `first_bytes` holds the first 8 bytes of a weight file. It gives back the whole number they stand for, in little-endian order.

    `header_length(bytes([2, 1, 0, 0, 0, 0, 0, 0]))` must give 258.
    """)
    return


@app.function
def header_length(first_bytes):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "header_length", header_length, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The Toolbox box has the tool, shown on 2 bytes. It works the same on 8.""",
        r"""It's one line: `int.from_bytes` with `first_bytes` and `"little"`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The header

    The header is text in a format called **JSON**, which looks like a Python dictionary written out. For a file with one small tensor it could read:

    ```
    {"gain": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 6]}}
    ```

    One entry per tensor. `dtype` is the kind of number, `shape` is the shape, and `data_offsets` says where the tensor's bytes start and stop—half-open like a Python slice, so `[0, 6]` covers bytes 0 through 5 (not including 6), **counted from the beginning of the data part**, not of the file.
    The data part begins right after the header, at byte $8 + n$ of the file.

    /// admonition | 🧰 Toolbox: reading a file
    - `with open(path, "rb") as f:` opens a file for reading bytes (`"rb"`: read, binary). The indented block can use `f`, and the file closes automatically when the block ends.
    - `f.read(4)` gives the next 4 bytes and moves on: a second `f.read(4)` gives the 4 after those.
    - `f.seek(100)` jumps to byte 100 of the file, so that the next `read` starts there.
    - `json.loads(data)` turns JSON text, or its bytes, into Python dictionaries and lists: `json.loads('{"a": [1, 2]}')` is `{"a": [1, 2]}`. `json` is already loaded.
    ///

    ## Exercise 2: read_header

    Write `read_header(path)`. `path` is a weight file. It gives back a pair: the header as a dictionary, and the position in the file where the data part starts.

    For a file whose first 8 bytes say 258, the header is the next 258 bytes, and the data starts at 266.
    """)
    return


@app.function
def read_header(path):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "read_header", read_header, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Inside one `with open(path, "rb") as f:`, read twice: first 8 bytes, for your `header_length`, then as many bytes as it says. The second read is the JSON text.""",
        r"""It's 3 lines. Keep the length in a name `n`, and give back `json.loads(f.read(n)), 8 + n`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Bytes into numbers with a point

    The data part is nothing but bytes, and bytes mean nothing until you say what kind of number they hold. Here are the same 4 bytes read three ways. `torch.frombuffer` makes a tensor straight from bytes, given a dtype
    (it wants them wrapped in a `bytearray`, a kind of bytes that may be changed).
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
    Four small whole numbers, or two larger ones (the first two bytes `[0, 0]` give `0`, and `[128, 63]` gives $128 + 63 \times 256 = 16256$), or the single number 1.0 in chapter 10's 32-bit format. Read with the wrong dtype, a weight file is noise. That's why every entry of the header names its dtype.

    Weight files have their own short names for dtypes, and `TYPES` translates them to PyTorch's: `TYPES["F32"]` is `torch.float32`, and `TYPES["BF16"]` is `torch.bfloat16`, chapter 22's 16-bit format with float32's range.
    Models are published in 16 bits to halve the download. Chapter 22 also measured that a CPU computes slowly in 16 bits, so we'll turn everything into 32 bits as we read it.

    ## By hand: find a tensor

    A file's first 8 bytes say 100. Its header has the entry `{"dtype": "BF16", "shape": [2, 2], "data_offsets": [40, 48]}` (half-open like a Python slice, so it stops before byte 48). At which byte of the file does this tensor start? How many bytes does it take? And how many bytes is that for each of its numbers?
    """)
    return


@app.cell(hide_code=True)
def _():
    find_boxes = hand.boxes(['first byte in the file', 'bytes in all', 'bytes per number'])
    return (find_boxes,)


@app.cell(hide_code=True)
def _(find_boxes):
    hand.marked(find_boxes, {'first byte in the file': 148, 'bytes in all': 8, 'bytes per number': 2}, {
        'first byte in the file': r"""8 for the length, 100 for the header, then 40 into the data""",
        'bytes in all': r"""from 40 up to, not including, 48""",
        'bytes per number': r"""the shape says $2 \times 2 = 4$ numbers""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Two bytes each: 16 bits, as `BF16` promises.

    ## Exercise 3: read_tensor

    Write `read_tensor(path, entry, data_start)`. `entry` is one tensor's entry from the header, a dictionary with the keys `"dtype"`, `"shape"` and `"data_offsets"`, and `data_start` is where the data part starts, as `read_header` gives it.
    It gives back the tensor, with the right shape, as 32-bit floats:

    1. Jump to the tensor's first byte and read exactly its bytes.
    2. Make a tensor from them with `torch.frombuffer(bytearray(data), dtype=...)`, using `TYPES` for the dtype.
    3. Give it its shape with `.reshape(...)`, and turn it into 32 bits with `.float()`.
    """)
    return


@app.function
def read_tensor(path, entry, data_start):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "read_tensor", read_tensor, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Your hand exercise is step 1: the first byte is `data_start` plus the first offset, and the number of bytes is the second offset minus the first. `start, stop = entry["data_offsets"]` names the two.""",
        r"""It's about 5 lines. Inside `with open(path, "rb") as f:`, `f.seek(data_start + start)` and then `data = f.read(stop - start)`. After it, one line: `torch.frombuffer(bytearray(data), dtype=TYPES[entry["dtype"]])`, followed by `.reshape(entry["shape"])` and `.float()`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get the model

    Now the real file. The button downloads three files of Qwen3-0.6B from the laboratory's page into `my-work/.models/`: the weights (1.5 GB), the tokenizer's merges and a small file of settings. It needs the internet this once,
    and takes a few minutes or much longer, depending on your connection. A bar under the cell counts the megabytes as they arrive, and the page can't do anything else meanwhile. If the download is cut off, press again: it continues where it stopped. Each file is checked when it arrives.

    Running the model needs about 4 GB of free memory. If you can't download or run it, your three functions above are still checked, and the rest of the chapter can be read.
    """)
    return


@app.cell(hide_code=True)
def _():
    download_button = mo.ui.run_button(label="Download Qwen3-0.6B (1.5 GB, once)")
    download_button
    return (download_button,)


@app.cell
def _(download_button):
    mo.stop(len(missing()) > 0 and not download_button.value, mo.md("*Press the button to download the model.*"))
    folder = fetch()
    weights = folder / "model.safetensors"
    header, data_start = read_header(weights)
    data_start, len(header), header["model.layers.0.self_attn.q_proj.weight"]
    return data_start, folder, header, weights


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Your `read_header` on a real model: the data starts at byte 35,560, after a header listing 312 entries, and one of them is shown. Every name follows a pattern. `model.layers.0.self_attn.q_proj.weight` is, in block 0, in the self-attention, the matrix that makes the queries.
    Blocks are called **layers** here, and they're numbered from 0 to 27.

    The settings file says the rest. It's JSON too:
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
    Compare with chapter 21's model, which had 2,258 tokens, 128 numbers per token, 3 blocks and 4 heads:

    | | yours | Qwen3-0.6B |
    |---|---:|---:|
    | tokens in the vocabulary | 2,258 | 151,936 |
    | numbers per token (`hidden_size`) | 128 | 1,024 |
    | blocks | 3 | 28 |
    | heads | 4 | 16 |
    | hidden numbers in SwiGLU | 341 | 3,072 |

    The design is the one you built: RMSNorm before each half of a block, RoPE on the queries and keys, SwiGLU, no biases. Only three things are new, and the next three sections take them in turn.

    ## New thing 1: one table, used twice

    `tie_word_embeddings` is true. A model has a table at each end: one turns a token into a vector on the way in, the other turns the final vector into a score for each token on the way out. Both have the shape (tokens, numbers per token).
    Qwen uses **the same table** for both: the score of a token is the dot product of the final vector with that token's own vector. This is called **tied embeddings**, and it saves 156 million knobs here.

    The file still lists the table under two names. Let's check they hold the same numbers, with your `read_tensor`:
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
    `torch.equal` is `True` only if every number matches. So we load every tensor except `lm_head.weight`, and skip the header's one entry that isn't a tensor, `__metadata__`:
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
    310 tensors and 596,049,920 numbers, read by your code in a few seconds.

    ## New thing 2: heads that share

    Look at the shapes the header gave. Notice a difference from the model you built earlier: there, the number of heads times the size per head equaled `hidden_size`. Qwen3 instead widens its 16 query heads to $16 \times 128 = 2048$ (double `hidden_size = 1024`), and then projects back down to 1,024 with the `out` matrix at the end of attention.
    Meanwhile, the key and value matrices make only $8 \times 128 = 1024$ numbers per token: **8 heads**.

    The 16 query heads share the 8 key and value heads in pairs: query heads 0 and 1 both look at key head 0, query heads 2 and 3 at key head 1, and so on. This is **grouped-query attention**, GQA.
    It was found that models lose very little this way. What it saves matters less here than in the next chapter, where the keys and values of everything read so far are kept in memory.

    ## By hand: who shares with whom

    With 16 query heads and 8 key heads, which key head does query head 5 use? And query head 15? And how many numbers does one block save in its key and value matrices together, compared with 16 full heads of each?
    """)
    return


@app.cell(hide_code=True)
def _():
    gqa_boxes = hand.boxes(['key head for query head 5', 'key head for query head 15', 'numbers saved in one block'])
    return (gqa_boxes,)


@app.cell(hide_code=True)
def _(gqa_boxes):
    hand.marked(gqa_boxes, {'key head for query head 5': 2, 'key head for query head 15': 7, 'numbers saved in one block': 2097152}, {
        'key head for query head 5': r"""pairs: 0 and 1 use key head 0, 2 and 3 use 1, 4 and 5 use …""",
        'key head for query head 15': r"""the last pair uses the last key head""",
        'numbers saved in one block': r"""each matrix is $1024 \times 1024$ where it would be $2048 \times 1024$, so each saves $1024 \times 1024$, and there are two""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In code, the simplest way to share is to copy: make 16 key heads out of 8 by repeating each one, and then your `attend` works unchanged.

    /// admonition | 🧰 Toolbox: repeating along a dimension
    `torch.repeat_interleave(x, 2, dim=0)` repeats each item along a dimension, keeping copies next to each other: for `x = torch.tensor([7, 8, 9])` it gives `[7, 7, 8, 8, 9, 9]`.
    ///

    ## Exercise 4: share_heads

    Write `share_heads(x, times)`. `x` has the shape (batch, heads, words, numbers), as keys and values have after chapter 16's `split_heads`. It gives back a tensor with `times` times as many heads, each head of `x` repeated `times` times in a row:
    with 3 heads and `times = 2`, the heads of the result are heads 0, 0, 1, 1, 2, 2 of `x`.
    """)
    return


@app.function
def share_heads(x, times):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "share_heads", share_heads, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The Toolbox tool, on the right dimension. Which position in (batch, heads, words, numbers) are the heads?""",
        r"""It's one line: `torch.repeat_interleave(x, times, dim=1)`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## New thing 3: tidy queries and keys

    The third difference is small. Before RoPE turns them, Qwen puts each head's queries and keys through an RMSNorm of their own, with 128 gains each: the `q_norm` and `k_norm` in the header.
    It keeps the dot products of queries and keys in a steady range, which makes large models train more calmly. You have the function already.

    ## The whole model, in your code

    **From earlier chapters.** The folded cells below hold chapter 20's `rms_norm`, `rope` and `attend`. `rope` now takes the base of its speeds as an input, because Qwen uses 1,000,000, which suits longer texts, where we used 10,000.

    Then comes one block and the whole model. `layer_knobs(knobs, i)` picks block `i`'s eleven tensors out of `knobs` and gives them short names. Remember chapter 10's convention: weight matrices are stored with shape `(out_size, in_size)`, so multiplying an input on the right needs `.T`—`y @ L["q"].T` is all an `nn.Linear` without a bias does.

    Following the shapes through `block`: input `x` enters with shape `(1, words, 1024)`; projecting queries and splitting into 16 heads gives `(1, 16, words, 128)` (`rms_norm` always normalizes along the last dimension, so calling it on `q` and `k` normalizes each head's 128 numbers); after attention, the 16 heads merge into `(1, words, 2048)`, and multiplying by `L["out"].T` brings it back to `(1, words, 1024)` so it can be added back to `x`.
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
    ## By hand: follow the shapes

    Read `block` for a text of 5 tokens. `x` comes in with the shape (1, 5, 1024). What's the size of the last dimension of `y @ L["q"].T`? After the reshape and the transpose, `q` has the shape (1, 16, 5, 128): how many heads does `k` have before `share_heads`, and after?
    And `qwen_scores` gives back the last token's scores: how many numbers is that?
    """)
    return


@app.cell(hide_code=True)
def _():
    shape_boxes = hand.boxes(['numbers per token after the q matrix', 'heads of k before', 'heads of k after', 'scores given back'])
    return (shape_boxes,)


@app.cell(hide_code=True)
def _(shape_boxes):
    hand.marked(shape_boxes, {'numbers per token after the q matrix': 2048, 'heads of k before': 8, 'heads of k after': 16, 'scores given back': 151936}, {
        'numbers per token after the q matrix': r"""16 heads of 128""",
        'heads of k before': r"""look at its reshape""",
        'heads of k after': r"""`share_heads(k, 2)`""",
        'scores given back': r"""one for every token of the vocabulary""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The first words

    The model needs token ids, so it needs its tokenizer. Qwen's is chapter 19's method exactly, byte-level BPE, with about 151,000 merges where you trained a few hundred. `QwenTokenizer` reads the merges from the downloaded file. Its `encode` and `decode` do what yours did.

    Now the moment of truth. We give the model the start of a sentence and look at its five highest scores for the next token (`torch.topk` gives the largest values and their positions):
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
    Five tokens in, and the first of the five candidates is ` Paris`.

    Think about what just happened. Nothing in that calculation came from a library that knows about language models. A file reader you wrote found 596 million numbers. Functions you wrote in chapter 20 pushed five vectors through 28 blocks.
    And a fact about France came out. There is nothing more in a language model than this.

    ## Writing an answer

    One token isn't an answer. Chapter 17's `write` made text by adding the model's choice to the input and asking again, and that loop is the same for every language model. Two things are new for a model you talk to.

    **Knowing when to stop.** Your story models wrote until a length ran out. A chat model writes a **stop token** when its answer is finished, and the loop must watch for it. Qwen's is token 151,645, written `<|im_end|>`, which `END_OF_TURN` holds.

    **Knowing who is speaking.** In training, this model saw conversations written in one fixed pattern, called its **chat template**, with special tokens marking each speaker's turn:

    ```
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant
    ```

    Text that ends like this is, to the model, text where an assistant's answer comes next. That's all "chatting" is: continuing text, as always, from a start that looks like a conversation. `tokenizer.chat(question)` wraps a question in this pattern.

    ## Exercise 5: generate

    Write `generate(next_scores, ids, limit, stop)`. `next_scores` is a function: given a list of token ids, it gives back the scores for the next token, a tensor with one number per token of the vocabulary. `ids` is a list of token ids to start from.
    Repeat at most `limit` times: take the token with the highest score, which is called the greedy choice. If it's `stop`, finish at once, without adding it. Otherwise add it to the ids. Give back the full list, start included.

    Don't change the list you were given: `ids + [token]` makes a new, longer list, where `ids.append(token)` would change the caller's.

    If `next_scores` always favours the last id plus 1, then starting from `[3]` with `stop = 7` must give `[3, 4, 5, 6]`.
    """)
    return


@app.function
def generate(next_scores, ids, limit, stop):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "generate", generate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""A `for` loop that runs `limit` times. Each turn: ask `next_scores` about the ids so far, find the position of the largest score with `torch.argmax`, and make it a plain number with `.item()`. Then either leave the function (chapter 21's early `return`) or lengthen the list.""",
        r"""It's 6 lines. Inside the loop: `token = torch.argmax(next_scores(ids)).item()`, then `if token == stop:` with `return ids` under it, then `ids = ids + [token]`. After the loop, `return ids` again, for when the limit runs out.""",
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
    The model's answer, how many tokens it took, and the seconds it took. It names Paris, and stops by itself. You're talking to a real model through your own code. Change the question in the cell and run it again.
    (`tokenizer.chat` also adds a short marker that tells this model to answer directly. Without it, Qwen3 first writes out a long passage of thinking.)

    ## Not always the top token

    Greedy choice gives the same answer every time, and for open questions it gives dull, repetitive text. Chapter 13 drew the next word at random with the model's chances. Real models do that with two controls.

    **Temperature.** You met it in chapter 14: divide every score by a number $T$ before the softmax. With $T$ below 1 the gaps between scores grow, so the top tokens get even more of the chance: safer, plainer text. With $T$ above 1 the gaps shrink: more surprising text, and more mistakes.

    **Top-k.** A vocabulary of 151,936 tokens has a long tail of tokens that each have a tiny chance, and together they're drawn often enough to derail a sentence. So keep only the $k$ likeliest tokens, set the others' chances to 0, and scale the kept ones so they add up to 1 again.

    ## By hand: keep the top two

    Four tokens have the chances 0.5, 0.3, 0.15 and 0.05. Keep the top 2. What are the chances of the first and the second token afterwards?
    """)
    return


@app.cell(hide_code=True)
def _():
    topk_boxes = hand.boxes(['first token', 'second token'], step=0.001)
    return (topk_boxes,)


@app.cell(hide_code=True)
def _(topk_boxes):
    hand.marked(topk_boxes, {'first token': 0.625, 'second token': 0.375}, {
        'first token': r"""the kept chances add up to 0.8, so divide: $0.5 / 0.8$""",
        'second token': r"""$0.3 / 0.8$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: the largest few, and storing at positions
    - `top = torch.topk(x, 2)` finds the 2 largest numbers of `x`: `top.values` holds them and `top.indices` their positions. For `x = torch.tensor([5.0, 9.0, 7.0])` those are `[9, 7]` and `[1, 2]`.
    - Storing works at a list of positions too (chapter 09): with `z = torch.zeros(3)`, the line `z[top.indices] = top.values` makes `z` into `[0, 9, 7]`.
    - `torch.zeros_like(x)` is a tensor of zeros with the shape and dtype of `x` (chapter 15).
    ///

    ## Exercise 6: top_k_chances

    Write `top_k_chances(scores, temperature, k)`. `scores` is a tensor with one score per token. It gives back a tensor of the same shape holding the chances to draw from:

    1. the softmax of the scores divided by the temperature (`torch.softmax(..., dim=-1)`);
    2. everything but the `k` largest chances set to 0;
    3. the result divided by its sum.

    On your hand exercise, scores whose softmax is 0.5, 0.3, 0.15, 0.05, with temperature 1 and `k = 2`, it must give `[0.625, 0.375, 0, 0]`.
    """)
    return


@app.function
def top_k_chances(scores, temperature, k):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("27", "top_k_chances", top_k_chances, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Work in the order of the three steps. For step 2, don't try to remove the small chances: start from a tensor of zeros and copy the `k` largest chances into it, at their own positions. The Toolbox box shows that move.""",
        r"""It's 5 lines: the chances, `torch.topk` of them, a `torch.zeros_like`, the line that stores `top.values` at `top.indices`, and the division by `torch.sum`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `sample` below is your `generate` with one line changed: the token is drawn from your chances instead of taken from the top. `torch.multinomial(chances, 1)` draws one position with the given chances, as chapter 13's `rng.choice` did.
    The cell uses the settings Qwen's makers recommend, a temperature of 0.6 and the top 20, on a question with many good answers.
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
    Change the seed and it writes a different sentence. Try a temperature of 1.5 and of 0.1 as well.

    ## What it cost

    Look back at the seconds your `generate` took, and at how it works. To write token 30, it runs all 28 blocks over the question **and** the 29 tokens already written, then throws everything away except one row of scores. For token 31 it does all of that again, plus one.
    The work for each new token grows with the length of the text, though only one token is new each time.

    For a few dozen tokens on a small model, that's bearable. For a long answer from a large model, it's ruinous, and nobody serves a model this way.
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("weight file", "1.5 GB"), ("knobs", "596 million"), ("scores", "151,936"), ("token", "1"), ("answer", "text")], ["your reader", "your blocks", "greedy, top-k", "repeat"], "From a file on disk to an answer, with the code you wrote")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What's next

    **Chapter 28** makes a single request fast. Its main idea is to stop recomputing: the keys and values of the tokens already read never change, so they can be kept. That's where today's shared heads pay off.

    ## What you can do now

    - describe what's in a weight file, and read its header and its tensors yourself
    - explain why the same bytes are different numbers under different dtypes
    - read a model's settings and names, and match them to the parts of a transformer
    - explain tied embeddings and grouped-query attention, and what each saves
    - run a published model with your own code
    - write a decoding loop with a stop token, and explain what a chat template is
    - control what a model writes with temperature and top-k

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** The *safetensors* format was made by Hugging Face as a safe replacement for files that can run code when opened. Grouped-query attention is from Ainslie et al. (2023).
    Qwen3 is described in the *Qwen3 Technical Report* (2025). Another common control is *top-p*, which keeps the smallest set of tokens whose chances add up to $p$. The course's own notice about the model is in `learn_llm/data/QWEN-NOTICE.md`.
    """)
    return


if __name__ == "__main__":
    app.run()
