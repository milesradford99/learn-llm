import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="30 · Capstone: guess, then check")

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
    # 30 · Capstone: guess, then check

    This is the last chapter of the main course, and it's optional. Building on the real model and KV cache from chapters 27 and 28, it adds one more speed-up for single-request decoding — one that sounds impossible at first: writing several tokens per run of the model **without changing a single token of the answer**.

    The idea is called **speculative decoding**. Something cheap **guesses** the next few tokens. The real model then **checks** all the guesses in one run, keeps the ones it agrees with, and corrects the first one it doesn't.

    Two facts from chapters 28 and 29 make it pay:

    - a decode step takes about the same time whether it's fed 1 token or a few, because fetching the knobs is most of the cost;
    - one run on several new tokens gives the scores after **each** of them, not only after the last.

    **From earlier chapters.** The folded cells below hold the model code of chapter 28: your file reader, `share_heads`, `rms_norm`, `rope`, `attend_new`, `remember`, `block_step`, `empty_caches`, `qwen_step`, `generate_cached` and `time_it`. The cell after them loads the model.
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
    ## Feeding a few costs about one

    First the claim. Chapter 28's `qwen_step` gives back the scores after the last new token only. `qwen_step_all` is the same function with its last line changed: it gives back a row of scores after **every** new token, so feeding 5 tokens gives 5 rows.

    The cell fills a cache with a question, then times one run fed 1, 2, 5, 9 and 17 tokens. Each row: tokens fed, and seconds.
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
    On the test machine: 0.14 seconds for 1 token, 0.14 for 2, 0.23 for 5, 0.28 for 9 and 0.33 for 17. Seventeen tokens take less than two and a half times as long as one. On a GPU, where arithmetic is even cheaper next to fetching, the curve is flatter still.

    ## Checking guesses

    Say the text so far ends with the token `"the"`, and a guesser proposes the next four tokens: `" cat" " sat" " on" " a"`. Feed all five — `"the"` plus the four guesses — in one run. The model gives back five rows of scores, and taking the top choice of each row gives five tokens `best[0..4]`. Lining them up with the four guesses `draft[0..3]` shows what each row checks:

    | Position | 0 | 1 | 2 | 3 | 4 |
    | --- | --- | --- | --- | --- | --- |
    | **5 tokens fed in** | `"the"` (last confirmed) | `" cat"` | `" sat"` | `" on"` | `" a"` |
    | **Model output `best[0..4]`** | `best[0]` | `best[1]` | `best[2]` | `best[3]` | `best[4]` |
    | **Guesses to check `draft[0..3]`** | `draft[0]` (`" cat"`) | `draft[1]` (`" sat"`) | `draft[2]` (`" on"`) | `draft[3]` (`" a"`) | bonus if all right |

    Read column by column:

    - column 0, after `"the"`: `best[0]` is the model's choice for the first new token. Compare it with `draft[0]`.
    - column 1, after `"the cat"`: `best[1]` is the model's choice for the second. It means something only if `draft[0]` was right, because it was computed assuming `" cat"` was in the text.
    - and so on, to column 4, after all four guesses.

    So walk the guesses in order. While a guess equals the model's own greedy choice at that column, it's exactly what plain decoding would have written: keep it. At the first guess that differs, stop: the model's own choice there is the next token, and every guess after it is thrown away.
    If all four guesses are right, the last column `best[4]` gives one more token for free.

    Either way, **every run adds at least one token**, the one plain decoding would have added, and the tokens are exactly the ones plain greedy decoding writes.

    ## By hand: what's kept

    The guesses are the tokens 5, 9, 2, 7. One run gives the model's choices after each prefix: 5, 9, 4, 1, 8. How many guesses are kept? How many tokens does this run add to the text in all? Which token is added last?
    And if the choices had been 5, 9, 2, 7, 8?
    """)
    return


@app.cell(hide_code=True)
def _():
    accept_boxes = hand.boxes(['guesses kept', 'tokens added', 'last token added', 'tokens added if all were right'])
    return (accept_boxes,)


@app.cell(hide_code=True)
def _(accept_boxes):
    hand.marked(accept_boxes, {'guesses kept': 2, 'tokens added': 3, 'last token added': 4, 'tokens added if all were right': 5}, {
        'guesses kept': r"""5 and 9 match; the third guess is 2, the model says 4""",
        'tokens added': r"""the kept guesses, and the model's own choice where they stopped""",
        'last token added': r"""the model's choice at the first wrong guess""",
        'tokens added if all were right': r"""all four guesses, and the last row's choice""",
    }, LOCALE, per_row=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: accept

    Write `accept(draft, best)`. `draft` is a list of guessed tokens, and `best` is one item longer: the model's greedy choice after each prefix, with `best[0]` before any guess and `best[i]` after the first $i$ guesses.

    Walk `draft` and `best` from the start: keep every matching guess up to the first mismatch, then append `best` at that position. If every guess matches, return all of `draft` followed by the last entry of `best`.

    `accept([5, 9, 2, 7], [5, 9, 4, 1, 8])` must give `[5, 9, 4]`, and with no guesses, `accept([], [6])` must give `[6]`.
    """)
    return


@app.function
def accept(draft, best):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "accept", accept, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Count how many guesses match, from the start, with a `while` loop (chapter 19) that stops at the end of `draft` or at the first mismatch. Call the count `m`. Then it's the first `m` guesses and one entry of `best`: which one?""",
        r"""It's 4 lines. `m = 0`, then `while m < len(draft) and draft[m] == best[m]:` with `m = m + 1` inside, then `return draft[:m] + [best[m]]`. The order inside the test matters: `and` doesn't look at its right side when the left is false.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Taking back the wrong guesses

    One run fed the last token and every guess, so every block's cache now holds keys and values for all of them, including the wrong guesses. Those must go, or the next run would attend to tokens that aren't in the text.

    Keep this rule: **the caches always hold every token of the text except the last.** Why leave the last token out? In chapter 28, each step fed only 1 brand-new token; here, the next run feeds the last confirmed token together with the new guesses, so if that last token were already in the cache, it would be stored twice!

    After a run, the text has grown by the tokens `accept` gave back, so the caches must keep
    $\text{tokens before the run} + \text{tokens added} - 1$ entries, and the rest are cut off.

    ## By hand: how much to keep

    The text has 31 tokens, so the caches hold 30. A run feeds the last token and 4 guesses. How many entries do the caches hold after the run? `accept` gives back 3 tokens: how long is the text now, and how many entries should the caches keep?
    """)
    return


@app.cell(hide_code=True)
def _():
    trim_boxes = hand.boxes(['entries after the run', 'tokens in the text now', 'entries to keep'])
    return (trim_boxes,)


@app.cell(hide_code=True)
def _(trim_boxes):
    hand.marked(trim_boxes, {'entries after the run': 35, 'tokens in the text now': 34, 'entries to keep': 33}, {
        'entries after the run': r"""30, and the 5 tokens just fed""",
        'tokens in the text now': r"""$31 + 3$""",
        'entries to keep': r"""every token of the text except the last""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The two kept guesses stay in the cache, and their keys are right: they were computed with the right text before them. Only the two wrong guesses go.

    ## Exercise 2: trim

    Write `trim(caches, length)`. `caches` is a list of pairs `(K, V)`, one per block, each of the shape (batch, heads, tokens, numbers), as `empty_caches` and `qwen_step` make them. It gives back a list of pairs in the same order, each keeping only its first `length` tokens.

    For one block holding 5 tokens, `trim` with `length` 3 gives back one pair whose tensors hold 3 tokens.
    """)
    return


@app.function
def trim(caches, length):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "trim", trim, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""A list comprehension over the pairs, `for K, V in caches`, giving a pair for each. The tokens are dimension 2, so the slice is `[:, :, :length]`.""",
        r"""It's one line: `return [(K[:, :, :length], V[:, :, :length]) for K, V in caches]`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Where guesses come from

    The usual guesser is a **draft model**: a much smaller model of the same family, with the same tokenizer, that writes a few tokens greedily. Qwen3-0.6B is already the smallest of its family, so this page uses a guesser that costs nothing at all.

    A lot of what a model writes repeats text that's already there: a quoted sentence, a name, a line of code being fixed, a story being retold. So look at the last few tokens of the text, find the latest **earlier** place where those same tokens appear, and guess that what followed them then follows them now.
    This is **prompt lookup**. When the text doesn't repeat itself, it finds nothing, or guesses wrong, and the check throws the guesses away.

    ## By hand: look it up

    The text is the tokens 4, 7, 1, 9, 4, 7, 2, 4, 7, numbered from 0. Look up its last 2 tokens and guess at most 3. Where does the **latest** earlier copy of the last 2 tokens start? What's the first guessed token? How many tokens are guessed?
    """)
    return


@app.cell(hide_code=True)
def _():
    lookup_boxes = hand.boxes(['start of the latest earlier copy', 'first guessed token', 'tokens guessed'])
    return (lookup_boxes,)


@app.cell(hide_code=True)
def _(lookup_boxes):
    hand.marked(lookup_boxes, {'start of the latest earlier copy': 4, 'first guessed token': 2, 'tokens guessed': 3}, {
        'start of the latest earlier copy': r"""4, 7 appears at 0, at 4 and at the end; the end itself doesn't count, and 4 is later than 0""",
        'first guessed token': r"""what came right after that copy""",
        'tokens guessed': r"""2, 4, 7: as many as asked for, since enough tokens follow""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: counting down, and comparing lists
    - `range(5, -1, -1)` counts 5, 4, 3, 2, 1, 0: chapter 09's step of `range` can be negative. The stop is still left out, so `-1` makes it end at 0.
    - `[4, 7] == [4, 7]` is `True`: two lists are equal when they hold the same items in the same order. `ids[-2:]` is the last 2 items, as chapter 17's `words[-32:]` was the last 32.
    ///

    ## Exercise 3: lookup_draft

    Write `lookup_draft(ids, n, k)`. `ids` is the text so far, a list of tokens. Take its last `n` tokens. Find the latest place where the same `n` tokens appear earlier, starting at least one token before the last `n` start, and give back the tokens that followed that copy, at most `k` of them.
    If there's no earlier copy, give back an empty list.

    `lookup_draft([4, 7, 1, 9, 4, 7, 2, 4, 7], 2, 3)` must give `[2, 4, 7]`. With the same text and `k = 5`, it must also give `[2, 4, 7]`: only those follow.
    """)
    return


@app.function
def lookup_draft(ids, n, k):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "lookup_draft", lookup_draft, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Go through the possible starts from the latest down to 0, with a `range` that counts down, and compare `ids[start:start + n]` with the last `n`. The latest possible start is $\text{len(ids)} - n - 1$. Return at the first match; after the loop, `[]`. A slice that runs past the end just stops there, so `k` needs no care.""",
        r"""It's 5 lines. `tail = ids[-n:]`, then `for start in range(len(ids) - n - 1, -1, -1):`, inside it `if ids[start:start + n] == tail:` and `return ids[start + n:start + n + k]`. Last, `return []`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: speculate

    Now the loop, the capstone's main piece. It's chapter 28's `generate_cached` with guesses.

    Write `speculate(step, caches, ids, limit, stop, draft, k)`. `step(new_ids, caches)` feeds tokens and gives back a pair: a tensor with **one row of scores per token fed**, and the grown caches. `caches` holds every token of `ids` except the last, as the rule says. `draft(ids, k)` gives back a list of at most `k` guesses.

    Repeat as long as fewer than `limit` tokens have been added after the starting `ids`:

    1. get the guesses from `draft`;
    2. feed the last token of `ids` followed by the guesses, in one call of `step`;
    3. the greedy choice of each row is the list `best` (see the Toolbox below), and `accept` gives the tokens to add;
    4. `trim` the caches to keep the rule;
    5. add the tokens one at a time: if one is `stop`, give back `ids` at once, without it.

    When the loop ends, give back `ids`, cut so that at most `limit` tokens follow the start: a run can add a few more than were needed.

    /// admonition | 🧰 Toolbox: one greedy choice per row
    `torch.argmax(scores, dim=-1)` picks the position of the largest number in every row, and chapter 25's `.tolist()` makes them a plain list: for the rows `[1.0, 5.0, 2.0]` and `[7.0, 0.0, 3.0]` it's `[1, 0]`.
    ///

    One of the checks uses a pretend model that counts, and a guesser that guesses right twice and then wrong. Starting from `[1, 2, 3]` with `stop = 9`, the answer must be `[1, 2, 3, 4, 5, 6, 7, 8]`, exactly what plain decoding gives, in 2 calls of `step` instead of 6.
    """)
    return


@app.function
def speculate(step, caches, ids, limit, stop, draft, k):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "speculate", speculate, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Keep the start's length in a name before the loop, for the `while` test and for the final cut. In the loop, the five steps are five or six lines. Step 2 feeds `[ids[-1]] + guess`. Step 4 keeps $\text{len(ids)} + \text{len(new)} - 1$ entries, so do it **before** the `for` loop that lengthens `ids`.""",
        r"""It's about 12 lines. `prompt = len(ids)`, then `while len(ids) - prompt < limit:`. Inside: `guess = draft(ids, k)`, `scores, caches = step([ids[-1]] + guess, caches)`, `new = accept(guess, torch.argmax(scores, dim=-1).tolist())`, `caches = trim(caches, len(ids) + len(new) - 1)`, then `for token in new:` with chapter 27's stop test and `ids = ids + [token]`. After the loop, `return ids[:prompt + limit]`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The real thing

    Three requests, each answered twice: by chapter 28's `generate_cached`, and by your `speculate` with prompt lookup of the last 2 tokens and up to 8 guesses. Before `speculate` starts, the caches are filled with every token of the question except the last, as the rule says.

    Each row of output: whether the two answers are the same token for token, the tokens written, the runs `speculate` needed, and the seconds both ways. Then the start of the answer. The three requests are chosen to show the good case, the middling one and the bad one. It takes about 40 seconds.
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
        print(_fast == _plain, len(_plain) - len(_ids), "tokens,", len(_runs), "runs,", round(_plain_seconds, 1), "seconds plain,", round(_fast_seconds, 1), "speculating")
        print("   ", tokenizer.decode(_plain[len(_ids):])[:90])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On the test machine:

    - **Copying a story:** the same 65 tokens in 11 runs instead of 65, 3.6 seconds instead of 9.3. Nearly every guess was right.
    - **Answering a question about it:** the same 14 tokens, in about the same time. The answer reuses a few phrases from the story, but not many.
    - **Counting to 20:** the same 70 tokens, and **slower**, 12.6 seconds against 9.3. The text repeats `", "` all the time, so the lookup keeps landing on the wrong earlier copy and proposing 8 wrong tokens, making every run pay for feeding 9.

    The answers were identical in every case, as promised. Speed is the only thing that guessing changes.

    ## When it pays

    Say each guess is right with probability $a$, independently of the others, and the guesser proposes $k$ tokens.

    Try concrete numbers first: $k = 2$ guesses, each right with probability $a = 0.5$. There are only three outcomes for a run:

    - **1st guess wrong** (chance $0.5$): adds 1 token;
    - **1st right, 2nd wrong** (chance $0.5 \times 0.5 = 0.25$): adds 2 tokens;
    - **both right** (chance $0.5 \times 0.5 = 0.25$): adds both guesses plus the bonus token, 3 in all.

    The average number of tokens added per run is

    > $$1 \times 0.5 + 2 \times 0.25 + 3 \times 0.25 = 1.75$$

    Notice that this is the same as $1 + 0.5 + 0.5^2 = 1.75$! Viewed token by token, that's no coincidence: a run always adds its 1st token (chance $1$), adds a 2nd if the first guess is right (chance $a$), and adds a 3rd if the first two are both right (chance $a^2$).

    For $k$ guesses in general, a run adds on average

    > $$1 + a + a^2 + \dots + a^k$$

    tokens. Whether that's worth it depends on how long a run fed $k + 1$ tokens takes, which you measured at the top of the page: about one plain step for $k = 1$, and about 2.4 plain steps at $k = 16$ on the test machine.

    ## Exercise 5: expected_tokens

    Write `expected_tokens(a, k)`: the sum above, $1 + a + a^2 + \dots + a^k$, which has $k + 1$ terms.

    `expected_tokens(0.5, 2)` must give 1.75, and with `k = 0` (no guesses), it's 1 whatever `a` is.
    """)
    return


@app.function
def expected_tokens(a, k):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("30", "expected_tokens", expected_tokens, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""A loop over the powers 0 to $k$, adding $a$ to each power: `range(k + 1)` gives them. Chapter 00's power is `**`.""",
        r"""It's 4 lines: `total = 0`, then `for i in range(k + 1):` with `total = total + a ** i` inside, then `return total`.""",
    ], LOCALE)
    return


@app.cell
def _():
    [(a, round(expected_tokens(a, 4), 2), round(expected_tokens(a, 8), 2)) for a in [0.2, 0.5, 0.8, 0.95]]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Each row: the chance $a$, then the tokens a run adds on average with 4 guesses and with 8. At $a = 0.2$, 8 guesses buy nothing over 4, and both cost more than a plain step: guessing loses. At $a = 0.95$, a run of 9 tokens adds about 7 of them.
    Engines therefore watch how often guesses are right and change $k$ as they go, or stop guessing.

    A good draft model's guesses are often right 60 to 90% of the time on ordinary text, which is why speculative decoding is now a standard part of serving. Some models are trained with small extra heads that guess the next few tokens themselves, so they need no separate draft at all.

    ## What you built
    """)
    return


@app.cell(hide_code=True)
def _():
    pipeline([("text", "n tokens"), ("guesses", "≤ k"), ("one run", "k + 1 rows"), ("kept", "≥ 1 token"), ("trimmed", "n + kept − 1")], ["lookup", "check", "accept", "trim"], "Speculative decoding: guess a few tokens, check them in one run, keep the agreed ones, cut the cache back")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The road behind you

    That's the end of the main course. Look at what you can now do with your own code, all of it:

    - **Part 1:** fit a line, slide down a slope, push blame backwards through a calculation, build an engine that does it for you, and train networks that read digits.
    - **Part 2:** turn words into vectors, predict the next word, give a model memory, and then attention, self-attention and the Transformer.
    - **Part 3:** tokenize bytes, build a modern transformer, train it properly, make it fast, spread it over processes, fit scaling laws, clean its data and align it.
    - **Part 4:** open a real model's file and run it, keep its cache, shrink its knobs, serve many people at once, and now write several tokens per run.

    Two directions from here, both beyond Part 4:

    - **Mixture of experts.** Many large models now have several SwiGLU halves per block, the *experts*, and a small layer that sends each token to only 2 or so of them. The model has many knobs but each token uses few, which makes serving it a new memory problem. With chapter 20's SwiGLU and chapter 07's softmax for the choosing, you can read their code.
    - **Agents.** A model that can call tools, read results and keep going is a loop around everything in Part 4. The course's optional agents track, chapters 40 to 51, reads two real agent systems to their source.

    ## What you can do now

    - explain why feeding a few tokens costs about as much as feeding one
    - check a list of guesses with one run and keep exactly what greedy decoding would write
    - keep the caches right when guesses are thrown away
    - guess tokens by looking up the text itself
    - write a speculative decoding loop, and predict when it pays

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** Speculative decoding was published by Leviathan et al. (2023) and, as *speculative sampling*, by Chen et al. (2023), with a rule that also works when tokens are drawn at random. Prompt lookup decoding was proposed by Saxena (2023).
    *Medusa* and *EAGLE* train extra heads that guess; DeepSeek-V3's *multi-token prediction* does it during training. Mixture of experts in its modern form is in *Switch Transformers* (Fedus et al., 2021) and *Mixtral of Experts* (Jiang et al., 2024).
    """)
    return


if __name__ == "__main__":
    app.run()
