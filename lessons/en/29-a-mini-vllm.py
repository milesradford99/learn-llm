import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="29 · A mini vLLM")

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

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 29 · A mini vLLM

    Chapter 28 made one request fast. A real service never has just one. People send questions whenever they like: some short, some with a long document pasted in, some wanting a sentence back and some a page.
    The program that answers all of them on one machine is called an **inference engine**, or a serving engine. The best known open one is called *vLLM*, and this chapter builds a small one with its three main ideas:

    1. **batching**: several requests share each run of the model;
    2. **a paged KV cache**: the requests' caches share one pool of memory, in small pages;
    3. **chunked prefill**: a long prompt is read in pieces, so it can't freeze everyone else.

    An engine is judged by two things that pull against each other. **Throughput**: how many tokens it writes per second, for everyone together. And **latency**: how long each person waits, which chapter 28 measured as the time to the first token and the time per token after it.

    **From earlier chapters.** The folded cells below hold chapter 27's file reader and `share_heads`, chapter 20's `rms_norm`, chapter 28's `rope` with its `start` and your `attend_new`, and chapter 22's `time_it`. The cell after them loads the model, as in chapter 28.
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
    mo.stop(len(missing()) > 0, mo.md("*The model isn't on this computer yet. Download it with the button in chapter 27, then open this page again.*"))
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
    ## One run, many requests

    Chapter 28 found that a decode step is slow because every knob has to be fetched from memory, 2.4 GB of them, to push **one** vector through. The arithmetic was the cheap part.

    So push several vectors through on the same trip. Five requests, each waiting for its next token, put their five vectors in one tensor, one row each, and every matrix multiplies all five rows at once. The knobs are fetched once and used five times. This is **batching**.

    Only one part of the model can't be shared: attention. Each request's new token must look at **its own** earlier tokens, through its own cache, and never at another person's text. So the matrices work on all the rows together, and attention works on each request alone.

    ## By hand: what a shared run costs

    Say one decode step takes 0.125 seconds for one request, all of it spent fetching knobs. If 8 requests share the step and the fetching is all that counts, how long does the step take? How many tokens per second does the engine write, for all 8 together? And for one request alone?
    """)
    return


@app.cell(hide_code=True)
def _():
    share_boxes = hand.boxes(['seconds for a step of 8', 'tokens per second, 8 together', 'tokens per second, 1 alone'], step=0.001)
    return (share_boxes,)


@app.cell(hide_code=True)
def _(share_boxes):
    hand.marked(share_boxes, {'seconds for a step of 8': 0.125, 'tokens per second, 8 together': 64, 'tokens per second, 1 alone': 8}, {
        'seconds for a step of 8': r"""the knobs are fetched once, whatever the number of rows""",
        'tokens per second, 8 together': r"""8 tokens every 0.125 seconds: $8 / 0.125$""",
        'tokens per second, 1 alone': r"""$1 / 0.125$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That's the ideal. The real gain is smaller, because the rows do cost some arithmetic, and attention, done for each request alone, costs more for each request added. The next sections build the engine, and then measure it.

    ## Two ways to batch

    Requests don't arrive together, and they don't finish together: one answer takes 3 tokens, another 200. The simplest way to batch, **static batching**, takes a group of requests, runs them until **all** are finished, then takes the next group.
    While the long answer is being written, the places of the finished ones stand empty.

    **Continuous batching** decides again at every step. A request that finishes leaves the batch at once, and a waiting request takes its place at the next step. The batch is always as full as it can be.

    ## By hand: empty places

    A batch has room for 4 requests. Their answers take 3, 10, 2 and 5 steps. With static batching, how many steps does the group take? A **place-step** is one place for one step: how many place-steps does the group hold, and how many of them are used to write a token? What share is that?
    """)
    return


@app.cell(hide_code=True)
def _():
    static_boxes = hand.boxes(['steps for the group', 'place-steps', 'place-steps used', 'share used'], step=0.01)
    return (static_boxes,)


@app.cell(hide_code=True)
def _(static_boxes):
    hand.marked(static_boxes, {'steps for the group': 10, 'place-steps': 40, 'place-steps used': 20, 'share used': 0.5}, {
        'steps for the group': r"""the group ends when its longest answer does""",
        'place-steps': r"""4 places for each of those steps""",
        'place-steps used': r"""each request uses one place for as many steps as its answer takes: $3 + 10 + 2 + 5$""",
        'share used': r"""used divided by all""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Half the places stand empty. The cell below counts the steps for 40 requests with answers between 5 and 60 tokens long, drawn at random, on 8 places, both ways. `continuous` says which way: with `False`, new requests may only come in when the batch is empty.
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
    Steps with static batching, steps with continuous batching, and the fewest possible, if all 8 places were busy every step: 281, 182 and 150. Continuous batching is much closer to the fewest. Every real engine batches this way.

    (`running` holds, for each request, the tokens it still has to write. Each step takes one off, and a request leaves when it has none left.)

    ## Where to keep the caches

    Batching has a cost. In chapter 28 each request's cache grew with `torch.cat`, which builds a whole new tensor every time. With many requests that start and end at different moments, memory gets cut into pieces of every size.
    An engine wants to set its memory aside **once**, at the start, and hand it out as requests need it.

    How much should a request get? Its answer's length isn't known until it ends. Giving every request room for the longest possible text, say 4,096 tokens, costs 0.94 GB each at chapter 28's 229,376 bytes a token. Eight requests would fill a laptop, and most of that room would never be used.

    vLLM's answer comes from how operating systems manage memory. Cut the memory into small **blocks** of the same size, here 16 tokens each, and give a request one more block whenever it fills the last one. All blocks live in one big tensor, the **pool**.
    A request's blocks needn't be next to each other in the pool. Each request keeps a **block table**: the list of its blocks' numbers, in order. And the engine keeps a list of the free blocks. When a request ends, its blocks go back on that list.

    This is the **paged KV cache**. A request never holds more than one block it doesn't fill, so at most 15 unused tokens' room.

    ## By hand: blocks

    Blocks hold 16 tokens. How many blocks does a request of 300 tokens need? How many of the tokens' places in them are unused? And a request of 32 tokens?
    """)
    return


@app.cell(hide_code=True)
def _():
    block_boxes = hand.boxes(['blocks for 300 tokens', 'unused places', 'blocks for 32 tokens'])
    return (block_boxes,)


@app.cell(hide_code=True)
def _(block_boxes):
    hand.marked(block_boxes, {'blocks for 300 tokens': 19, 'unused places': 4, 'blocks for 32 tokens': 2}, {
        'blocks for 300 tokens': r"""18 blocks hold 288 tokens, so the last 12 need one more""",
        'unused places': r"""$19 \times 16 - 300$""",
        'blocks for 32 tokens': r"""exactly 2 full blocks, and no more""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: blocks_for

    Write `blocks_for(length, block_size)`. It gives back how many blocks of `block_size` tokens are needed to hold `length` tokens: `length` divided by `block_size`, rounded **up**. 0 tokens need 0 blocks.

    `blocks_for(300, 16)` must give 19, and `blocks_for(32, 16)` must give 2.

    Chapter 11's `//` rounds down. One way to round up: add $\text{block size} - 1$ to `length` first.
    """)
    return


@app.function
def blocks_for(length, block_size):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "blocks_for", blocks_for, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Try the trick on 32 and on 33 with blocks of 16: $32 + 15 = 47$ and $33 + 15 = 48$. Which of them does `// 16` turn into 3?""",
        r"""It's one line: `(length + block_size - 1) // block_size`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Finding a token

    Where is token number $p$ of a request? Its block is number $p \,//\, 16$ **in the request's block table**, and inside that block it's at place $p \,\%\, 16$ (chapter 23's `%`, the remainder). The table says which block of the pool that is.

    ## By hand: look it up

    A request's block table is `[7, 2, 9]`, and blocks hold 16 tokens. In which block of the pool is its token 20, and at which place in that block? And token 40?
    """)
    return


@app.cell(hide_code=True)
def _():
    slot_boxes = hand.boxes(['block for token 20', 'place for token 20', 'block for token 40', 'place for token 40'])
    return (slot_boxes,)


@app.cell(hide_code=True)
def _(slot_boxes):
    hand.marked(slot_boxes, {'block for token 20': 2, 'place for token 20': 4, 'block for token 40': 9, 'place for token 40': 8}, {
        'block for token 20': r"""$20 \,//\, 16 = 1$, and entry 1 of the table is …""",
        'place for token 20': r"""$20 \,\%\, 16$""",
        'block for token 40': r"""$40 \,//\, 16 = 2$""",
        'place for token 40': r"""$40 - 2 \times 16$""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: slot

    Write `slot(table, position, block_size)`. `table` is a request's block table, a list of block numbers. It gives back a pair: the number of the pool's block that holds token `position`, and the place inside that block.

    `slot([7, 2, 9], 20, 16)` must give `(2, 4)`.
    """)
    return


@app.function
def slot(table, position, block_size):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "slot", slot, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two numbers, as in your hand exercise: which entry of `table` (with `//`), and the place (with `%`). The first number you give back is the **entry** of the table, not its position in the table.""",
        r"""It's one line: `return table[position // block_size], position % block_size`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Reading a request's cache

    The pool for one block of the model is a tensor of the shape (blocks, heads, block size, numbers): for Qwen, (160, 8, 16, 128) in this chapter. Keys and values each have their own pool, and each of the 28 blocks of the model has its own pair.

    Attention wants a request's keys as chapter 28 had them: (1, heads, tokens, numbers), in order. So the engine gathers the request's blocks from the pool, joins them along the token dimension, and cuts off the unused places at the end.

    /// admonition | 🧰 Toolbox: a list of tensors, and a new first dimension
    - `[pool[b] for b in [4, 1]]` is a list of two tensors: block 4 of the pool, then block 1. Each has the shape (heads, block size, numbers).
    - `torch.cat` joins a list of any length, as in chapter 14: `torch.cat([a, b, c], dim=1)`.
    - `x[None]` puts a new dimension of size 1 in front, like chapter 20's `v[:, None]` does in the second place: a tensor of the shape (8, 40, 128) becomes (1, 8, 40, 128).
    ///

    ## Exercise 3: read_cache

    Write `read_cache(pool, table, length)`. `pool` has the shape (blocks, heads, block size, numbers), `table` is a request's block table, and `length` is how many tokens the request has so far.
    It gives back the request's keys (or values) in order, with the shape (1, heads, length, numbers).

    With blocks of 2 tokens, the table `[3, 0]` and `length` 3, the result holds the 2 tokens of pool block 3, then the first token of pool block 0.
    """)
    return


@app.function
def read_cache(pool, table, length):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "read_cache", read_cache, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three moves, one per Toolbox line: gather the table's blocks in a list, join them on the dimension that counts the tokens (each block is (heads, block size, numbers)), then keep the first `length` tokens and add the dimension in front.""",
        r"""It's 2 lines. `joined = torch.cat([pool[b] for b in table], dim=1)`, then give back `joined[:, :length][None]`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The model, for a whole batch

    Here is chapter 28's `block_step`, rewritten for a batch. All new tokens of all requests sit in one row, `x`, of the shape (1, total new tokens, 1,024): the first request's tokens, then the second's, and so on. `parts` says where each request's tokens are: one triple `(table, seen, new)` per request.

    The matrices work on all tokens at once. Then, one request at a time, the new keys and values are written into the pool with your `slot`, the request's whole cache is read with your `read_cache`, and your `attend_new` blends it.
    The `pools` hold one pair `(K, V)` for each block of the model.
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
    ## By hand: follow a batch

    Three requests share a step. Two are decoding: one has 40 tokens in its cache, the other 7. The third is reading its prompt of 20 tokens, all at once. So `parts` is `[(table_a, 40, 1), (table_b, 7, 1), (table_c, 0, 20)]`.

    What's `total`? What's the shape's middle number for `q`, after the transpose, that is, how many tokens does it hold? Which position does `rope` give the second request's token? And how many tokens long is the `K` that the third request's attention reads?
    """)
    return


@app.cell(hide_code=True)
def _():
    batch_boxes = hand.boxes(['total', 'tokens in q', 'position of the second request\'s token', 'tokens in the third request\'s K'])
    return (batch_boxes,)


@app.cell(hide_code=True)
def _(batch_boxes):
    hand.marked(batch_boxes, {'total': 22, 'tokens in q': 22, 'position of the second request\'s token': 7, 'tokens in the third request\'s K': 20}, {
        'total': r"""every request's new tokens, side by side: $1 + 1 + 20$""",
        'tokens in q': r"""`q` holds the queries of every new token""",
        'position of the second request\'s token': r"""its cache holds tokens 0 to 6""",
        'tokens in the third request\'s K': r"""`seen + new`: it has nothing older""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `run_batch` runs all 28 blocks and gives back one row of scores per request: the scores after its last new token. `new_pools` sets aside the memory once: 160 blocks of 16 tokens, for 28 blocks of the model, keys and values: 0.59 GB.
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
    ## Measure the sharing

    The cell gives 1, 2, 4, 8 and 16 requests a cache of the same 30 tokens each, then times one decode step for all of them together. Each row: the number of requests, the seconds for the step, and the tokens per second for all of them together. It takes about a minute.
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
    On the test machine: 7.5 tokens per second for one request, 22 for 8, 35 for 16. Not the ideal 16 times, but more than 4 times as much from the same computer, for free. The time per step grows slowly, so each person's tokens come a little slower: throughput bought with a little latency.

    (The gain is smaller than the ideal partly because of this page's Python: `block_batch` loops over the requests and over every new token. A real engine does that work in fused operations, chapter 28's word.)

    ## Letting requests in

    Now the engine's loop. Every step, it does four things:

    1. requests that have arrived join the **waiting** list;
    2. it **admits** waiting requests into the **running** list, as many as it may;
    3. it decides how many tokens of each running request to feed;
    4. it runs the model once on all of them, and gives each request whose input is used up its next token.

    The part that decides who may run is called the **scheduler**. Ours has two limits: at most `max_running` requests at once, and enough free blocks for a new request's prompt. Requests are taken in the order they arrived: if the first waiting request doesn't fit, the ones behind it wait too.

    A request is a dictionary: `"ids"` holds its tokens so far, prompt included.

    ## Exercise 4: admit

    Write `admit(waiting, running, free_blocks, max_running, block_size)`. `waiting` and `running` are lists of requests, and `free_blocks` is how many blocks are free. It gives back a pair: the new waiting list and the new running list.

    As long as there's a waiting request, the running list has fewer than `max_running`, and the **first** waiting request's prompt fits in the free blocks: move that request to the end of the running list, and count its blocks as no longer free.

    With 3 free blocks of 16 tokens, nobody running, room for 4, and waiting prompts of 20, 10 and 5 tokens: the first two are admitted (2 blocks, then 1), and the third must wait.
    """)
    return


@app.function
def admit(waiting, running, free_blocks, max_running, block_size):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "admit", admit, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""A `while` loop (chapter 19) with three conditions joined by `and`. Inside it, three lines: fewer free blocks, a longer running list (`running + [waiting[0]]`), a shorter waiting list (`waiting[1:]`). Use your `blocks_for` on `len(waiting[0]["ids"])`.""",
        r"""It's 5 lines. The loop's test is `len(waiting) > 0 and len(running) < max_running and blocks_for(len(waiting[0]["ids"]), block_size) <= free_blocks`. After the loop, `return waiting, running`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A long prompt freezes everyone

    One case is still bad. Four people are reading answers as they're written, a token every quarter of a second. Then someone pastes a 540-token story and asks about it. The next step reads all 540 tokens at once: chapter 28 measured what a long prefill costs.
    For that step, everyone else's next token waits seconds instead of a quarter of one.

    **Chunked prefill** caps the work of one step at a **token budget**, for example 64 tokens. Decoding requests need one token each and go first. What's left of the budget goes to prompts still being read, in order, and a long prompt is read 61 or 62 tokens at a time over several steps.
    Nobody waits more than one small step. The long prompt's first token comes a little later, because it shares every step with the others.

    ## By hand: share out a budget

    The budget is 64. Three requests are decoding and one has a 540-token prompt, all unread. How many prompt tokens are fed in the first step? How many steps does it take to read the whole prompt, if the three keep decoding? And how many tokens does the last of those steps read of the prompt?
    """)
    return


@app.cell(hide_code=True)
def _():
    chunk_boxes = hand.boxes(['prompt tokens in the first step', 'steps to read the prompt', 'prompt tokens in the last of them'])
    return (chunk_boxes,)


@app.cell(hide_code=True)
def _(chunk_boxes):
    hand.marked(chunk_boxes, {'prompt tokens in the first step': 61, 'steps to read the prompt': 9, 'prompt tokens in the last of them': 52}, {
        'prompt tokens in the first step': r"""$64 - 3$""",
        'steps to read the prompt': r"""8 steps read $8 \times 61 = 488$ tokens, and some are left""",
        'prompt tokens in the last of them': r"""$540 - 488$""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: chunk_sizes

    Write `chunk_sizes(unfed, budget)`. `unfed` is a list with one number per running request, in the order they'll be served: how many of its tokens haven't been fed to the model yet. A decoding request has 1. It gives back a list of the same length: how many tokens each request is fed in this step.

    Go through the requests in order. Each gets as many as it has unfed, or as many as are left of the budget, whichever is smaller, and what it gets is taken off the budget. When the budget is used up, the rest get 0.

    `chunk_sizes([1, 1, 1, 540], 64)` must give `[1, 1, 1, 61]`, and `chunk_sizes([1, 100, 30], 64)` must give `[1, 63, 0]`.
    """)
    return


@app.function
def chunk_sizes(unfed, budget):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("29", "chunk_sizes", chunk_sizes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Start an empty list. In a `for` loop over `unfed`, chapter 12's `min` picks the size; append it, and lower the budget by it.""",
        r"""It's 6 lines. Inside the loop: `size = min(n, budget)`, then `sizes.append(size)`, then `budget = budget - size`. After the loop, `return sizes`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The engine

    Here are the four steps of the loop, with your three functions in them. `new_request` makes a request. `"fed"` counts the tokens the model has already read, `"table"` is the block table, and `"arrives"` is the step at which the request arrives.
    Requests must be listed in the order they arrive.

    Before each step, the running requests are put in order, decoding ones first, and new blocks are taken from the front of the free list as a request needs them. When a request ends, its blocks go back on the list. The `log` notes, for every step, who was fed how many tokens and how long the step took.

    (A real engine also handles running out of blocks. vLLM then pauses the request that came last, frees its blocks and computes its cache again later. This engine's pool is large enough for this page.)
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
    ## Eight people at once

    Eight questions arrive together. First the engine answers them one at a time, with room for a single running request, as chapter 28 did. Then all eight share it. The cell prints the answers, whether each is the same token for token, and the seconds both ways. It takes about half a minute.
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
    print("seconds one at a time:", round(seconds_alone, 1), "  together:", round(seconds_together, 1))
    return questions, together_log


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On the test machine, all eight answers were the same both ways, and they took 7.4 seconds together against 19 one at a time. The answers are listed in the order they finished: short answers leave first, and their places stand free.

    The log shows what happened at each step, here its first four steps and its last three: the step's number, a pair (request, tokens fed) for each request fed, and the seconds.
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
    Step 0 reads 64 tokens of prompts: three whole prompts and 9 tokens of the fourth. At step 1, those three are already decoding, one token each, while the next prompts are read: decoding and reading prompts share the step. From step 3 all eight decode together, and at the end only the two longest answers are left.

    ## The long prompt, measured

    Now the case from before. Four questions start at step 0. At step 6 someone arrives with a story of 540 tokens. The engine runs twice: with a budget so large that nothing is ever cut, and with a budget of 64. The cell shows, both ways, the longest step and when the long request's answer started. It takes about 20 seconds.
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
        print("budget", _budget, "  prompt", _story["prompt"], "tokens   longest step:", _longest, "seconds   first answer token fed at step", _first)
        print("   ", tokenizer.decode(_story["ids"][_story["prompt"]:]))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On the test machine: with no cut, one step took 3.9 seconds, and for those seconds four people saw nothing move. With a budget of 64 the longest step took 0.83 seconds, and the long prompt was read over 9 steps, so its answer started at step 15 instead of 7. The answer itself is the same.

    That's the trade every engine tunes: a larger budget reads long prompts sooner, a smaller one keeps everyone else's tokens flowing.

    ## What you built
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("arrivals", "any time"), ("waiting", "in order"), ("running", "≤ max"), ("one run", "≤ budget"), ("tokens", "one each")], ["queue", "admit", "chunk", "share"], "The engine's loop: requests wait, are admitted, are fed within a budget, and share one run of the model")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That's an inference engine: your scheduler, your paged cache, your budget, around chapter 28's model with its cache. vLLM and the engines like it add a lot of engineering, mostly to make each of these steps fast on a GPU, but their loop is this one.

    ## What's next

    **Chapter 30**, the optional capstone, takes one more step of your choice: speculative decoding, where a small model guesses several tokens and the large one checks them in one run; mixture-of-experts models; or a small agent loop on your own engine.

    ## What you can do now

    - explain why batching raises throughput, and what it costs in latency
    - compare static and continuous batching, and count the empty places
    - explain a paged KV cache: blocks, the pool, block tables and the free list
    - find any token of a request in the pool, and read a request's cache in order
    - write a scheduler that admits requests by order, room and free memory
    - share a token budget between decoding requests and prompts in chunks
    - read an engine's log and say what each step did

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** The paged KV cache is from the vLLM paper, *Efficient Memory Management for Large Language Model Serving with PagedAttention* (Kwon et al., 2023). Continuous batching was described in *Orca* (Yu et al., 2022), and chunked prefill in *SARATHI* (Agrawal et al., 2023).
    Engines also share blocks between requests that begin with the same text, which is called *prefix caching*.
    """)
    return


if __name__ == "__main__":
    app.run()
