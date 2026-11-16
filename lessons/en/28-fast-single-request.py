import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="28 · Fast single request")

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
    # 28 · Fast single request

    In chapter 27 a real model answered you through your own code. It was correct, and it was wasteful: for every token it wrote, it ran all 28 blocks over the whole text again.

    This chapter makes one request fast. A **request** is one person's question, from the moment it arrives until the answer is complete. Two numbers describe how fast it feels:

    - the **time to the first token**: how long you stare at an empty answer;
    - the **tokens per second** after that: how fast the answer grows.

    You'll improve both without changing a single token of the answer. Then you'll shrink the model's matrices to a quarter of their bytes, and see what that does, and doesn't do, on your computer.

    **From earlier chapters.** The folded cells below hold chapter 27's code: your file reader, `share_heads`, the block, `qwen_scores` and `generate`, with chapter 20's `rms_norm`, `rope` and `attend`, and chapter 22's `time_it`.
    One thing changed: `rope` takes a third input, `start`, the position of the first token it's given. Until now that was always 0. A later section needs it.

    The next cell loads the model you downloaded in chapter 27.
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
    ## The waste, measured

    First, how long does one run of the model take, for texts of different lengths? The cell runs `qwen_scores` on the first 25, 50, 100 and 200 tokens of a story.
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
        print(_length, "tokens:", round(time.perf_counter() - _started, 2), "seconds")
    return (story,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On the test machine: 0.28, 0.36, 0.66 and 1.07 seconds. The longer the text, the longer one run takes. Not quite in proportion, and a later section says why, but from 100 tokens on, twice the text costs nearly twice the time.

    Now remember how `generate` works. It runs the model on the whole text to get one token, adds the token, and runs the model on the whole text again.

    ## By hand: count the passes

    Call one token going through all 28 blocks one **pass**. A run on 10 tokens is 10 passes.

    A prompt has 10 tokens, and `generate` writes 4 tokens after it. So it runs the model 4 times: on 10 tokens, then on 11, 12 and 13. How many passes is the second run? How many passes are all 4 runs together?
    And in all 4 runs together, how many passes were for a token the model had **not** been run on before?
    """)
    return


@app.cell(hide_code=True)
def _():
    pass_boxes = hand.boxes(['passes in the second run', 'passes in all 4 runs', 'passes for tokens not seen before'])
    return (pass_boxes,)


@app.cell(hide_code=True)
def _(pass_boxes):
    hand.marked(pass_boxes, {'passes in the second run': 11, 'passes in all 4 runs': 46, 'passes for tokens not seen before': 13}, {
        'passes in the second run': r"""the prompt's 10 tokens and the 1 token written so far""",
        'passes in all 4 runs': r"""$10 + 11 + 12 + 13$""",
        'passes for tokens not seen before': r"""the first run's 10 tokens were all new; each later run has 1 new token, the one just written""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    46 passes, and only 13 of them on anything new. The other 33 repeat work that was already done. It gets worse with length: for a prompt of 100 tokens and an answer of 20, it's 2,190 passes against 119.

    Here is such a case for real: a question about the story, 92 tokens with the chat template around it. The cell takes a while, which is the point.
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
    The length of the question, the answer, how many tokens the answer has, and the seconds it took. On the test machine: 14 tokens in 9.2 seconds. That was 15 runs, one for each token and one more that gave the stop token: 1,485 passes, for a text that ends up 106 tokens long. Keep the seconds in mind.

    ## What never changes

    Could the repeated passes be skipped? Only if running them again always gives the same result as before. It does, and the reason is chapter 16's mask.

    The mask lets a token look at itself and at earlier tokens, never at later ones. So in block 0, what comes out for token 3 is made from tokens 0 to 3 and nothing else. Block 1 works on those results, under the same mask, so its result for token 3 is again made from tokens 0 to 3 only. And so on, through all 28 blocks.

    **Nothing the model computes for a token depends on the tokens after it.** When a token is added at the end, every number computed for the earlier tokens, in every block, stays exactly what it was. `generate` computes them all again anyway.

    ## What a new token needs

    So run only the new token through the blocks. What does it need from the earlier tokens? Go through chapter 27's `block` and ask of each line whether the new token's result uses another token's numbers:

    - `rms_norm`: no, each token's vector is normed by itself;
    - the matrices that make its query, key and value: no;
    - attention: **yes**. The new token's query is compared with the **keys** of all tokens so far, and the result blends their **values**;
    - the matrix after attention, and SwiGLU: no.

    Only one line looks sideways, and it wants two things from each earlier token: its key and its value, in this block. Those never change. So each block keeps them: every token's key and value, stored once, when the token passes through. This store is called the **KV cache**, K for keys and V for values.
    ("Cache" is also chapter 22's word for the fast memory next to the processor. Here it only means a place where results are kept so they needn't be computed again.)

    ## Attention for the newest tokens

    Attention now has fewer queries than keys. `Q` holds the queries of the new tokens only, and `K` and `V` hold the keys and values of every token so far, with the new ones at the end. The scores are still $Q K^\top / \sqrt{d}$, a table with one row per new token and one column per token so far.

    The mask changes shape with it. Its rule stays the same: a token may look at itself and at everything before it.

    ## By hand: who may look where

    There are 5 tokens so far, numbered 0 to 4, and the last 2 of them are new. The mask has 2 rows and 5 columns. Which token does row 0 belong to? How many of the 5 keys may that token look at? And how many cells of the whole mask are blocked?
    """)
    return


@app.cell(hide_code=True)
def _():
    mask_boxes = hand.boxes(['token number of row 0', 'keys row 0 may look at', 'blocked cells in the mask'])
    return (mask_boxes,)


@app.cell(hide_code=True)
def _(mask_boxes):
    hand.marked(mask_boxes, {'token number of row 0': 3, 'keys row 0 may look at': 4, 'blocked cells in the mask': 1}, {
        'token number of row 0': r"""the new tokens are the last two: numbers 3 and 4""",
        'keys row 0 may look at': r"""itself and everything before it: tokens 0, 1, 2, 3""",
        'blocked cells in the mask': r"""row 0 may not look at token 4; row 1 is token 4, the last one, and may look at everything""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: `torch.triu` on any table
    Chapter 16 used `torch.triu(table, diagonal=1)` on a square table. It works on any table, and `diagonal` can be any whole number: `diagonal=2` keeps the cells that are at least 2 columns to the right of their row's number.
    For `torch.triu(torch.ones(2, 4), diagonal=2)` the rows are `[0, 0, 1, 1]` and `[0, 0, 0, 1]`.
    ///

    ## Exercise 1: attend_new

    Write `attend_new(Q, K, V)`. `Q` has the shape (batch, heads, new, numbers): the queries of the newest tokens. `K` and `V` have the shape (batch, heads, total, numbers): the keys and values of all tokens so far, the newest at the end.
    It gives back the blend for each new token, with the shape (batch, heads, new, numbers).

    It's chapter 16's attention with one difference, the mask. The mask has `new` rows and `total` columns. Row $i$ belongs to token number $\text{total} - \text{new} + i$, and it blocks every key with a higher number than that.

    With 5 tokens of which 2 are new, the mask is your hand exercise: one blocked cell, at the end of row 0. With 1 new token nothing is blocked. With `new` equal to `total`, the result is the same as chapter 16's.
    """)
    return


@app.function
def attend_new(Q, K, V):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "attend_new", attend_new, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Open the folded `attend` near the top of the page: it's your starting point. Two things change. The mask is `new` by `total` instead of square, and its `diagonal` is no longer 1. In your hand exercise the first blocked cell of row 0 was in column 4: that's the diagonal you need there.""",
        r"""It's 4 lines. The first is `new, total = Q.shape[-2], K.shape[-2]`. The scores and the last line are as in `attend`. The mask is `torch.triu` of `torch.ones(new, total, dtype=torch.bool)` with `diagonal=total - new + 1`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Keeping keys and values

    Each block's cache is a pair `(K, V)`. Both tensors have the shape (1, 8, seen, 128): 1 text, 8 key heads, one entry for each token seen so far, 128 numbers. They're stored with 8 heads, before `share_heads` copies them to 16. Copies needn't be kept.

    Before any token has been fed, `seen` is 0. PyTorch allows that: `torch.zeros(1, 8, 0, 128)` is a tensor that holds no numbers and still has a shape, and `torch.cat` can join real keys onto it. So the cache needs no special case for the first token.

    ## Exercise 2: remember

    Write `remember(cache, k, v)`. `cache` is a pair `(K, V)` of tensors with the shape (batch, heads, seen, numbers). `k` and `v` are the keys and values of the newest tokens, with the shape (batch, heads, new, numbers).
    It gives back a new pair: `K` with `k` joined on after it along the token dimension, and `V` with `v` joined on the same way. Both then have the shape (batch, heads, seen + new, numbers).

    With 2 tokens in the cache and 1 new one, the pair you give back holds 3 tokens each, the new one last.
    """)
    return


@app.function
def remember(cache, k, v):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "remember", remember, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""`cache[0]` is `K` and `cache[1]` is `V`. Chapter 14's `torch.cat` joins tensors along one dimension. Which position in (batch, heads, seen, numbers) counts the tokens?""",
        r"""It's one line that gives back two things: `torch.cat([cache[0], k], dim=2)`, a comma, and the same for `cache[1]` and `v`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A token must know its place

    One trap is left. RoPE turns each query and key by an angle that depends on the token's position in the text. Chapter 27's `rope` got the whole text every time, so it numbered the tokens from 0.
    Feed it one new token alone and it would call that token number 0 as well, and the model would read the text as if its latest word stood at the very beginning.

    So `rope` now takes `start`: the position of the first token it's given. And the cache knows that position. If it holds 12 tokens, numbered 0 to 11, the next one is number 12.

    ## One block, with a memory

    Here is chapter 27's block again, changed to work on new tokens only. `x` holds the vectors of the new tokens, and `cache` is this block's pair. It gives back two things: the new tokens' vectors, and the cache with the new tokens in it.

    The products with matrices are now written `product(y, M)`, where `product` is a function handed in as an input. For now it's always `plain`, which is chapter 27's `y @ M.T`. A later section hands in a different one.
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
    ## By hand: follow one token through

    A prompt of 19 tokens has been fed, and now one more token is fed alone. Read `block_step` for that call. What are `new` and `seen`? Which position does `rope` give the token?
    How many tokens does `cache[0]` hold after `remember`, and how many heads does it have? And how many heads does `attend_new` get as its `K`?
    """)
    return


@app.cell(hide_code=True)
def _():
    step_boxes = hand.boxes(['new', 'seen', 'position given by rope', 'tokens in cache[0] after remember', 'heads in cache[0]', 'heads of K in attend_new'])
    return (step_boxes,)


@app.cell(hide_code=True)
def _(step_boxes):
    hand.marked(step_boxes, {'new': 1, 'seen': 19, 'position given by rope': 19, 'tokens in cache[0] after remember': 20, 'heads in cache[0]': 8, 'heads of K in attend_new': 16}, {
        'new': r"""`x` holds only the token that was fed""",
        'seen': r"""what the cache held before this call: the prompt""",
        'position given by rope': r"""the prompt's tokens are numbers 0 to 18""",
        'tokens in cache[0] after remember': r"""the prompt's and the new one""",
        'heads in cache[0]': r"""look at the reshape of `k`""",
        'heads of K in attend_new': r"""`share_heads(cache[0], 2)`""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The whole model is 28 of these, each with its own cache. `empty_caches` makes the 28 empty pairs to start from. `qwen_step` feeds some new tokens through all blocks and gives back the scores for the token after them, and the 28 caches as they are now.
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
    Does the memory work? A test: feed the story question all at once, and feed it in two parts, 60 tokens and then the rest. If your `attend_new` and `remember` are right, the last block's keys come out the same both ways.
    The cell shows their shape and the largest difference between the two (`torch.abs` makes every number positive, like chapter 09's `np.abs`).
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
    92 tokens of keys, and a difference of 0, or of some number far below a thousandth, which is only rounding. If the difference is large, look at your mask again, and at the order in `remember`.

    ## Writing with a memory

    Now the loop. Chapter 27's `generate` handed the model everything each time. The new loop hands it only what it hasn't seen: the whole prompt the first time, and after that just the one token that was added.

    ## Exercise 3: generate_cached

    Write `generate_cached(step, caches, ids, limit, stop)`. `step` is a function: `step(new_ids, caches)` feeds a list of new token ids to a model that has the given caches, and gives back a pair: the scores for the next token, and the caches grown by those tokens.
    `caches` are the caches to start from, and `ids` is the list of token ids to start from.

    Repeat at most `limit` times: call `step`, take the token with the highest score, finish at once without adding it if it's `stop`, otherwise add it to the ids. Give back the full list, start included, as `generate` did.

    The first call of `step` gets all of `ids` and the caches you were handed. Every later call gets a list holding only the token just added, and the caches the call before it gave back.

    One of the checks uses a pretend model that counts: its favourite token is the number of tokens it has been fed so far, plus 1. Started on `[1, 2, 3]` with `stop = 7`, you must give back `[1, 2, 3, 4, 5, 6]`, and `step` must have been fed `[1, 2, 3]`, then `[4]`, `[5]` and `[6]`.
    """)
    return


@app.function
def generate_cached(step, caches, ids, limit, stop):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "generate_cached", generate_cached, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Start from chapter 27's `generate`, folded near the top of the page. You need one more name, for what to feed next. Before the loop it's all of `ids`. `scores, caches = step(..., caches)` catches both results and keeps the grown caches under the same name for the next turn.""",
        r"""It's 8 lines. Before the loop: `new = ids`. In the loop: `scores, caches = step(new, caches)`, then the token with `torch.argmax(scores).item()`, the stop test with its early `return ids`, `ids = ids + [token]`, and last `new = [token]`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now the same question as before, with your loop. The cell wraps `qwen_step` in a function that also notes how long each call took, in the list `step_seconds`.
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
    print("the same tokens as before:", fast_ids == slow_ids)
    print("seconds before:", round(slow_seconds, 1), " now:", round(sum(step_seconds), 1))
    print("first call:", round(step_seconds[0], 2), "seconds;  each later call:", round(sum(step_seconds[1:]) / len(step_seconds[1:]), 3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The same answer, token for token. On the test machine it took 2.6 seconds where it took 9.2, and the gap widens with every token an answer gets longer. Your seconds differ, but the tokens shouldn't.

    The last line shows that a request now has two phases, and they have names:

    - **Prefill**: the first call, which reads the whole prompt and fills the caches. It took 0.57 seconds for 92 tokens. This is the time to the first token.
    - **Decode**: every call after it, one token each. They took 0.14 seconds each, about 7 tokens per second, and they'd take the same after 1,000 tokens as after 100, give or take a little for the longer row of scores in attention.

    Every engine that serves language models works this way, and reports these two numbers.

    ## What the cache costs

    The speed is paid for with memory: the cache holds a key and a value for every token, in every block, for as long as the request lasts.

    ## By hand: the size of the cache

    Count it for Qwen: 28 blocks, 8 key heads of 128 numbers, a key and a value each. How many numbers does **one block** keep for one token? How many do all blocks keep for one token? How many bytes is that at 32 bits a number?
    And how many bytes would it be if all 16 heads had their own keys and values?
    """)
    return


@app.cell(hide_code=True)
def _():
    size_boxes = hand.boxes(['numbers per token, one block', 'numbers per token, all blocks', 'bytes per token', 'bytes per token with 16 key heads'])
    return (size_boxes,)


@app.cell(hide_code=True)
def _(size_boxes):
    hand.marked(size_boxes, {'numbers per token, one block': 2048, 'numbers per token, all blocks': 57344, 'bytes per token': 229376, 'bytes per token with 16 key heads': 458752}, {
        'numbers per token, one block': r"""a key of $8 \times 128$ numbers and a value of the same size""",
        'numbers per token, all blocks': r"""28 times one block's""",
        'bytes per token': r"""32 bits are 4 bytes""",
        'bytes per token with 16 key heads': r"""twice as many heads, twice as many numbers""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.23 MB a token. The cell below counts the numbers in the caches of the 92-token question, and divides by 92.
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
    A conversation of 4,000 tokens holds about 0.9 GB of keys and values, for this small model. This is where chapter 27's shared heads pay off: with 16 key heads it would be twice that.
    And it's the reason the next chapter exists. One person's cache fits anywhere. A thousand people's caches, growing and ending at different moments, have to be managed.

    ## Why a token still takes a tenth of a second

    Decoding one token is now a small calculation: one vector goes through each matrix. Why does it still take 0.14 seconds?

    Because of chapter 22's lesson: a processor computes far faster than it can fetch numbers from main memory. To push one vector through the model, every knob has to be fetched once: 596 million numbers of 4 bytes each, 2.4 GB, for every token.
    The arithmetic on them is cheap. Moving them is what takes the time. It's also why, in the first measurement, 200 tokens didn't cost 8 times what 25 did: the knobs are fetched once for a run, however many tokens use them.

    A test of that claim: the same product of one vector with one of Qwen's matrices, once with everything in 32 bits and once in 64. The arithmetic is the same. Only the number of bytes differs.
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
    The matrix's shape, then the milliseconds for one product in 32 bits and in 64 (`.double()` turns a tensor into 64 bits, as `.float()` turns it into 32). On the test machine: 0.26 and 0.54. Twice the bytes, twice the time.

    So to decode faster, **store each knob in fewer bytes**. Chapter 22's 16-bit formats are one step. The usual step in serving is bigger: whole numbers of 8 bits, or even 4.

    ## Knobs as small whole numbers

    One byte can hold a whole number from −128 to 127. A knob is a number like 0.0731 or −0.4125. To store it in a byte, pick a small number called the **scale**, and store how many scales the knob is worth, rounded to a whole number:

    > $$q = \text{round}\left(\frac{w}{\text{scale}}\right) \qquad\qquad w \approx q \times \text{scale}$$

    This is **quantization**: $w$ is the knob, $q$ the whole number that's stored. Reading it back gives the knob only roughly, because of the rounding. The error is at most half a scale.

    The scale is chosen so that the largest knob lands on the largest whole number. With $b$ bits we use the whole numbers from $-(2^{b-1} - 1)$ to $2^{b-1} - 1$, the same distance both ways: −127 to 127 for 8 bits, −7 to 7 for 4 bits. So

    > $$\text{scale} = \frac{\text{the largest size among the knobs}}{2^{b-1} - 1}$$

    where the size of a knob is the knob without its sign.

    ## By hand: quantize three knobs

    Three knobs, 1.4, −0.66 and 0.25, are to be stored in 4 bits, as whole numbers from −7 to 7. What's the scale? Which whole numbers are stored for −0.66 and for 0.25? And which knobs come back when those two are read again?
    """)
    return


@app.cell(hide_code=True)
def _():
    quant_boxes = hand.boxes(['scale', 'stored for −0.66', 'stored for 0.25', '−0.66 read back', '0.25 read back'], step=0.1)
    return (quant_boxes,)


@app.cell(hide_code=True)
def _(quant_boxes):
    hand.marked(quant_boxes, {'scale': 0.2, 'stored for −0.66': -3, 'stored for 0.25': 1, '−0.66 read back': -0.6, '0.25 read back': 0.2}, {
        'scale': r"""the largest size is 1.4, and the largest whole number is 7""",
        'stored for −0.66': r"""$-0.66 / 0.2 = -3.3$, rounded""",
        'stored for 0.25': r"""$0.25 / 0.2 = 1.25$, rounded""",
        '−0.66 read back': r"""the stored number times the scale""",
        '0.25 read back': r"""the stored number times the scale""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The knob 0.25 comes back as 0.2. With only 15 whole numbers to choose from, the steps are coarse.

    **One scale per row.** A matrix has knobs of very different sizes. If the whole matrix shared one scale, a single large knob would set it, and thousands of small knobs would all round to 0 or ±1.
    So each **row** of the matrix gets its own scale, from its own largest knob. A row is the set of knobs that make one output number. The price is one extra number to store per row, next to a thousand or more knobs.

    /// admonition | 🧰 Toolbox: sizes, the largest in each row, rounding, small whole numbers
    - `x.abs()` makes every number positive, like `torch.abs(x)`.
    - `x.amax(dim=1)` is the largest number in each row: for `x = torch.tensor([[1.0, 5.0], [2.5, 0.4]])` it's `[5, 2.5]`.
    - `torch.round(x)` rounds every number to the nearest whole number: `torch.round(torch.tensor([2.4, -1.6]))` is `[2, -2]`. They're still stored as floats.
    - `torch.int8` is a dtype: whole numbers from −128 to 127, one byte each. `x.to(torch.int8)` turns a tensor into it, the way `.float()` turns one into 32-bit floats.
    ///

    ## Exercise 4: quantize

    Write `quantize(W, bits)`. `W` is a matrix of floats with the shape (rows, columns). It gives back a pair:

    - `q`, with the shape of `W` and the dtype `torch.int8`: each knob divided by its row's scale, and rounded;
    - `scale`, a vector with one number per row: the largest size in that row, divided by $2^{\text{bits} - 1} - 1$.

    For the matrix with the two rows `[1.4, -0.66, 0.25]` and `[-0.07, 0.03, 0.012]` and 4 bits, `q` has the rows `[7, -3, 1]` and `[-7, 3, 1]`, and `scale` is `[0.2, 0.01]`.

    To divide every row of `W` by its own scale, the scales must stand as a column: chapter 20's `scale[:, None]`.
    """)
    return


@app.function
def quantize(W, bits):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "quantize", quantize, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three steps, as in your hand exercise: the largest whole number, from `bits`; the scale of each row, with `.abs()` and `.amax(dim=1)`; then divide, round, and turn into `torch.int8`. The second row of the example has its largest size on a negative knob: that's what `.abs()` is for.""",
        r"""It's 3 lines. `levels = 2 ** (bits - 1) - 1`, then `scale = W.abs().amax(dim=1) / levels`, then give back `torch.round(W / scale[:, None]).to(torch.int8)` and `scale`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    How much is lost? The cell quantizes the matrix `W` from the timing test with 8, 6 and 4 bits, reads it back, and shows the typical size of the error as a share of the typical size of the knobs (chapter 09's typical size: the square root of the mean square).
    The last column does the same with one scale for the whole matrix.
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
        print(_bits, "bits   a scale per row:", round(_row_error, 3), "  one scale for all:", round(_shared_error, 3))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    With 8 bits and a scale per row, the knobs are off by about 1% of their typical size. Each bit taken away doubles the step between whole numbers, and so doubles the error: about 3% at 6 bits and 15% at 4. One scale for the whole matrix is about three times worse at every width.

    ## Multiplying without reading back

    A model stored as small whole numbers still has to be multiplied by vectors of floats. The obvious way is to read every knob back, $q \times \text{scale}$, and then multiply as before. But the scale is the same for a whole row, so it can wait until the end:

    > $$(q_1 s) x_1 + (q_2 s) x_2 + (q_3 s) x_3 = (q_1 x_1 + q_2 x_2 + q_3 x_3)\, s$$

    Multiply the vector by the whole numbers first, then multiply each output by its row's scale, once. For a matrix of 3,072 rows and 1,024 columns that's 3,072 multiplications by a scale instead of 3 million.

    ## By hand: the scale comes last

    A row is stored as the whole numbers 3, −1, 2 with the scale 0.5. The input is the vector 1, 2, 4. What's the sum of each whole number times its input? And what's the output?
    """)
    return


@app.cell(hide_code=True)
def _():
    product_boxes = hand.boxes(['sum of whole numbers times inputs', 'output'], step=0.1)
    return (product_boxes,)


@app.cell(hide_code=True)
def _(product_boxes):
    hand.marked(product_boxes, {'sum of whole numbers times inputs': 9, 'output': 4.5}, {
        'sum of whole numbers times inputs': r"""$3 \times 1 - 1 \times 2 + 2 \times 4$""",
        'output': r"""the sum times the scale""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Reading back first gives the same: the knobs are 1.5, −0.5 and 1, and $1.5 \times 1 - 0.5 \times 2 + 1 \times 4 = 4.5$.

    ## Exercise 5: quantized_matmul

    Write `quantized_matmul(x, q, scale)`. `q` and `scale` are a matrix as `quantize` gives it: `q` with the shape (outputs, inputs) and the dtype `torch.int8`, `scale` with one number per output.
    `x` holds vectors of floats, and its last dimension has the size `inputs`. It gives back what `plain(x, W)` would give for the matrix `W` that `q` and `scale` stand for, but without building `W`: the product with the whole numbers first, then the scales.

    The result's last dimension has the size `outputs`, so multiplying it by the vector `scale` scales each output by its own number, with no `None` needed.

    PyTorch's `@` won't mix dtypes (chapter 10), and it has no product for `int8` with floats. So `q` has to be turned into floats for the product: `q.float()`. Keep that in mind for the next section.

    On your hand exercise, `x = [1, 2, 4]`, the single row `[3, -1, 2]` and the scale `[0.5]`, it must give `[4.5]`.
    """)
    return


@app.function
def quantized_matmul(x, q, scale):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("28", "quantized_matmul", quantized_matmul, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Look at `plain`: it multiplies by `M.T`. Do the same with `q`, once it's floats, and multiply what comes out by `scale`.""",
        r"""It's one line: `(x @ q.float().T) * scale`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The real model in 8 bits

    Now the whole model. `shrink` quantizes every matrix in the 28 blocks with your `quantize` and leaves the rest as it is: the gains of the norms, which are tiny, and the token table.
    `small_product` is a product for `block_step` to use in place of `plain`: it takes a pair `(q, scale)` where `plain` takes a matrix. That's why `product` was an input.
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
    print("bytes for the blocks' matrices in 32 bits:", 4 * matrix_numbers, "  in 8 bits:", matrix_numbers)
    print("32 bits:", answer(knobs, plain, tokenizer, "Write one sentence about the sea."))
    print(" 8 bits:", answer(small_knobs, small_product, tokenizer, "Write one sentence about the sea."))
    return (small_knobs,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The matrices of the blocks went from 1.76 GB to 0.44 GB, a quarter. And on the test machine the model wrote the same sentence, word for word, with every one of its 440 million matrix knobs off by about 1%.
    (If yours differs by a word, that's the rounding landing differently on your processor. It should be as good a sentence.)

    A model survives this because of how it computes. Each output is a sum over a thousand or more knobs, and their rounding errors point in random directions, so most of the error cancels (chapter 09: the sizes of random parts add up far more slowly than their number).

    How far can it go? The same question at 8, 6 and 4 bits:
    """)
    return


@app.cell
def _(knobs, tokenizer):
    for _bits in [8, 6, 4]:
        print(_bits, "bits:", answer(shrink(knobs, _bits), small_product, tokenizer, "What is the capital of France?"))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    At 8 and 6 bits it's Paris. At 4 bits, on the test machine, the model named another French city, with full confidence. Errors of 15% in every knob are more than 28 blocks in a row can absorb.

    That's a result about **this simple recipe**, not about 4 bits. Models are served in 4 bits every day, with more care: a scale for each small group of knobs instead of each row, and rounding that's chosen by measuring what it does to the outputs.
    And 4-bit knobs are packed two to a byte, which `int8` doesn't do. The go-deeper note names the methods.

    ## Smaller, but not faster yet

    A quarter of the bytes should mean faster decoding. Measure one decode call with each set of knobs, after a prefill of the same short prompt:
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
    Seconds for one token in 32 bits, and in 8 bits. On the test machine: 0.12 and 0.16. The small model is **slower**.

    You wrote the reason yourself: `q.float()`. For every product, PyTorch first copies the whole matrix of small numbers into 32-bit floats, 4 bytes each, and then multiplies with the copy. The bytes we saved are spent again on every token, and the copying comes on top.

    ## Fused operations

    Count how many times something the size of the matrix goes through memory, for three ways of computing the same product:

    - **read back, then multiply**: copy `q` to floats, multiply the copy by the scales, multiply by the vector. Three times.
    - **your `quantized_matmul`**: copy `q` to floats, multiply by the vector. Two times. The scales touch only the small result.
    - **32-bit floats**, as in chapter 27: once, but 4 bytes a knob.
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
    Milliseconds for each of the three. On the test machine: 1.24, 0.63 and 0.27. The time follows the bytes moved, not the arithmetic, which is the same in all three.

    What's missing is a fourth way: fetch the small whole numbers, one byte a knob, and multiply and add them straight away, **once** through memory, with no copy. That would move a quarter of the bytes of the 32-bit product, and you've measured what bytes cost.

    An operation that does the work of several steps in one go, without writing the results in between to memory, is called a **fused operation**. You met one in chapter 22: `F.scaled_dot_product_attention` does scores, mask, softmax and blend without ever building the table.
    The quantized product is the other one that matters for serving. It can't be written in Python: stepping through a matrix number by number in Python is thousands of times too slow. It's written in a compiled language, once for each kind of processor, and PyTorch doesn't have one for a plain `int8` tensor on a laptop's processor.

    The engines people use to run models on their own computers are, for the most part, collections of exactly these fused operations, wrapped around the same loop you wrote. That's where a 4-bit model on a laptop gets its speed: your KV cache, your scales, and a product that reads each knob once.

    ## What you built
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("prompt", "92 tokens"), ("28 caches", "keys, values"), ("new token", "1 pass"), ("answer", "text")], ["prefill, once", "decode", "repeat"], "A request with a KV cache: the prompt is read once, then each token costs one pass")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One request now costs one pass per token, and you know what limits the rest: bytes moved per token.

    ## What's next

    A real service doesn't have one request. It has hundreds at once, arriving at any moment, each with its own cache that grows and then ends.
    **Chapter 29** builds a small server for that: several requests share each run of the model, and their caches share the memory.

    ## What you can do now

    - explain why writing with full recomputation wastes most of its work, and count the passes
    - explain why earlier tokens' keys and values never change, and keep them in a KV cache
    - write attention for new tokens only, with the right mask and the right positions
    - name the two phases of a request, prefill and decode, and measure each
    - compute how much memory a KV cache takes
    - quantize a matrix to whole numbers with a scale per row, and say how large the error is
    - multiply by a quantized matrix with the scales applied last
    - explain what a fused operation is, and why quantization needs one to be fast

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** Engines that run quantized models on ordinary computers include *llama.cpp*, with its GGUF files, and Apple's *MLX*; names like `Q4_K_M` on a model file say how it was quantized.
    Rounding that's chosen by measuring the outputs is the idea of *GPTQ* (Frantar et al., 2022) and *AWQ* (Lin et al., 2023). `torch.compile` fuses chains of PyTorch operations.
    """)
    return


if __name__ == "__main__":
    app.run()
