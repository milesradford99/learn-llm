import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="19 · Tokenization")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import hand, todo
    from learn_llm.checks import check
    from learn_llm.data import stories

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 19 · Tokenization

    Part 3 rebuilds your tiny GPT the way real labs do, and it starts where every model starts: turning text into numbers.

    So far we've used a list of 2,000 words. That has three problems. Any other word becomes `<unk>`, and the model can't even
    copy it. A list has no idea that `dog` and `dogs` are related. And it only works for text with spaces between lowercase English words.

    Going the other way, to single letters, would solve those, but the model would have to spell out every word, one small step at a time.
    The answer in every modern model lies in between: **tokens**, pieces of text that are often a whole word and sometimes part of one, learned from the data. The method is called **byte-pair encoding** (BPE). It's simple, and you'll write all of it.

    ## From text to bytes

    Computers store everything as **bytes**, whole numbers from 0 to 255. **UTF-8** is the standard way to write text in bytes: English letters take
    1 byte each, and other scripts take more. Chinese characters take 3.

    /// admonition | 🧰 Toolbox: text and bytes
    - `text.encode("utf-8")` gives a text's bytes. Python shows bytes as text with a `b` in front, like `b'dog'`.
    - `list(...)` turns bytes into a plain list of numbers, and `bytes([100, 111, 103])` makes bytes from such a list.
    - `.decode("utf-8")` turns bytes back into text.
    ///
    """)
    return


@app.cell
def _():
    as_bytes = "dog".encode("utf-8")
    as_bytes, list(as_bytes), list("你好".encode("utf-8")), bytes([100, 111, 103]).decode("utf-8")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Starting from bytes means any text at all can be represented, in any language, with a starting vocabulary of just 256.

    ## Exercise 1: to_bytes

    Write `to_bytes(text)`. It takes a string and gives back its UTF-8 bytes as a plain list of numbers:

    > `to_bytes("dog")` gives `[100, 111, 103]`
    """)
    return


@app.function
def to_bytes(text):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "to_bytes", to_bytes, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two of the Toolbox's tools, one inside the other: encode the text, then make the result a list.""",
        r"""It's one line, and it starts `return list(`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The idea: merge the commonest pair

    BPE is a single move, repeated: find the pair of neighbouring ids that appears most often, give that pair a new id, and replace it everywhere.
    Here's `aaabdaaabac` as letters, to keep it readable:

    | step | most common pair | new id | text |
    |---|---|---|---|
    | start | | | `a a a b d a a a b a c` |
    | 1 | `a a` | `Z` | `Z a b d Z a b a c` |
    | 2 | `Z a` | `Y` | `Y b d Y b a c` |
    | 3 | `Y b` | `X` | `X d X a c` |

    Two details. In `a a a`, the pair `a a` could start at the first `a` or the second, but a letter can only be used once: we go from left to right, and after replacing a pair we carry on *after* it. So `a a a` becomes `Z a`.
    And at step 2, `Z a` and `a b` are tied at 2 each; we take whichever was seen first. In code, the new ids are 256, 257 and 258. Common letter pairs merge first, then the pieces join into larger pieces, and common words end up as a single token.

    ## By hand: two merges

    Your turn, on the 12 letters `a b c a b c a b d a b c`. Count how often the commonest pair appears. Merge it, calling it `Z`, and count the tokens left
    (a merged pair is one token). Then do it again on the new text, calling the new pair `Y`.
    """)
    return


@app.cell(hide_code=True)
def _():
    merge_boxes = hand.boxes(['count of the commonest pair', 'tokens after merge 1', 'count of the commonest pair now', 'tokens after merge 2'])
    return (merge_boxes,)


@app.cell(hide_code=True)
def _(merge_boxes):
    hand.marked(merge_boxes, {'count of the commonest pair': 4, 'tokens after merge 1': 8, 'count of the commonest pair now': 3, 'tokens after merge 2': 5}, {
        'count of the commonest pair': r"""`a b` appears more often than `b c`, because one `a b` is followed by `d`""",
        'tokens after merge 1': r"""each of the 4 merges turns two tokens into one, so 12 loses 4""",
        'count of the commonest pair now': r"""the text is now `Z c Z c Z d Z c`. How many times does `Z c` appear?""",
        'tokens after merge 2': r"""each of the 3 merges removes one more token""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After two merges the text is `Y Y Z d Y`: 5 tokens instead of 12, and `Y` stands for `a b c`.

    ## Exercise 2: count_pairs

    Write `count_pairs(ids)`. `ids` is a list of ids. It gives back a dictionary from each neighbouring pair, as a key in round brackets, to how many times it appears:

    > `count_pairs([1, 2, 1, 2, 3])` gives `{(1, 2): 2, (2, 1): 1, (2, 3): 1}`

    It's chapter 13's way of pairing a list with itself, with chapter 12's way of counting.
    """)
    return


@app.function
def count_pairs(ids):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "count_pairs", count_pairs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Chapter 13: `zip(ids, ids[1:])` gives every neighbouring pair. Chapter 12: `counts.get(key, 0) + 1` raises a count that may not exist yet. Each pair that `zip` gives can be used as a key as it is.""",
        r"""It's 4 lines, the same shape as chapter 12's `count_words`, with `for pair in zip(ids, ids[1:]):` as the loop.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A loop that moves at its own pace

    Now the replacement. A `for` loop goes one step at a time, but the merge sometimes needs to jump two: after replacing a pair, it must skip both halves.

    /// admonition | 🧰 Toolbox: while, and, else
    - `while test:` repeats its indented lines for as long as the test holds. Nothing moves on by itself: the lines inside must change something, or the test holds forever.
    - `a and b` holds only if both tests hold. Python checks `a` first, and if it fails, never looks at `b`.
    - `else:` lined up with an `if` runs its indented lines when the `if` test did *not* hold.
    ///

    The cell below uses all three. It counts down from 20 to 0: it takes away 5 while the number is above 5 and fewer than 2 steps have passed, and otherwise takes away 1. It shows how many steps that took.
    """)
    return


@app.cell
def _():
    n_demo = 20
    steps_demo = 0
    while n_demo > 0:
        if n_demo > 5 and steps_demo < 2:
            n_demo = n_demo - 5
        else:
            n_demo = n_demo - 1
        steps_demo = steps_demo + 1
    steps_demo
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That's 20, 15, 10, and then 10 single steps down to 0: twelve steps.

    **Careful:** if nothing inside a `while` loop changes what its test looks at, the test holds forever, and the cell never finishes. If that happens, press the square stop button at the corner of the cell, and check the lines that move you forward.

    ## Exercise 3: merge

    Write `merge(ids, pair, new_id)`. `ids` is a list of ids, `pair` is two ids in round brackets, and `new_id` is the id that replaces them. It gives back a new list:

    > `merge([1, 2, 3, 1, 2], (1, 2), 9)` gives `[9, 3, 9]`, and `merge([1, 1, 1], (1, 1), 9)` gives `[9, 1]`

    Walk along `ids` with a position `i` that you move yourself, starting at 0, building the new list as you go:

    - if the pair starts at position `i`, meaning `ids[i]` is the pair's first id and the next item is its second, add `new_id` to the new list and **jump two**;
    - otherwise copy `ids[i]` to the new list and move one.

    At the last position there is no next item, and asking for it would be an error. So check that there *is* a next item before looking at it.
    """)
    return


@app.function
def merge(ids, pair, new_id):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "merge", merge, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""An empty list and `i = 0`, then `while i < len(ids):`. Inside it, an `if` with three tests joined by `and`, and an `else`. Each branch appends one thing and adds to `i`. `pair[0]` and `pair[1]` are the pair's two ids.""",
        r"""It's 10 lines. The first of the three tests is `i < len(ids) - 1`, which says a next item exists. It must come first, so that Python never looks at `ids[i + 1]` when there isn't one. If the cell never finishes, a branch isn't adding to `i`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training: repeat the move

    Training repeats the move a given number of times, remembering each merge in a dictionary from pair to new id. New ids start at 256, after the bytes.

    /// admonition | 🧰 Toolbox: the key with the biggest value
    `max(d, key=d.get)` gives the key of a dictionary whose value is biggest, in the style of chapter 12's `sorted`.
    For `ages = {"ann": 30, "bo": 12, "cy": 30}` it gives `"ann"`: on a tie, the key that was stored first wins.
    ///

    ## Exercise 4: train_bpe

    Write `train_bpe(ids, count)`. `ids` is a list of ids and `count` is the number of merges to learn. It gives back a dictionary from each merged pair to its new id.
    Each of the `count` steps does this:

    1. count the pairs of the current `ids`;
    2. take the commonest pair;
    3. give it the next new id: 256 at the first step, 257 at the second, and so on;
    4. remember it in the dictionary;
    5. replace the pair in `ids`, so that the next step counts pairs of the new text.

    The table above is the worked example: `train_bpe(to_bytes("aaabdaaabac"), 3)` gives `{(97, 97): 256, (256, 97): 257, (257, 98): 258}`, since `a` is byte 97 and `b` is byte 98.

    Python's dictionaries keep the order in which keys were added. That matters here, because merges must later be replayed in the same order.
    """)
    return


@app.function
def train_bpe(ids, count):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "train_bpe", train_bpe, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""An empty dictionary, then `for step in range(count):` with five lines inside, one for each numbered step, using your `count_pairs` and `merge`. The step number `step` starts at 0, which is why the new id is 256 plus it.""",
        r"""It's 8 lines. The last line inside the loop stores the result of `merge` back into `ids`. Without that, every step finds the same pair.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Encoding and decoding

    To **encode** new text, turn it into bytes and replay every merge, in the order they were learned.

    To **decode**, we need the bytes behind each id. Build a table: ids below 256 are single bytes, and each merged id is its two parts joined. Since a pair's parts always have smaller ids, going through the merges in order always finds them ready.

    /// admonition | 🧰 Toolbox: pairs from a dictionary, and joining bytes
    - `d.items()` goes through a dictionary's pairs of key and value, so `for name, age in ages.items():` gives both at once.
    - `+` joins bytes: `bytes([104]) + bytes([105])` is `b'hi'`.
    - `b''.join(pieces)` joins a list of bytes into one, with nothing between them. It's chapter 12's `join`, for bytes.
    - `.decode("utf-8", errors="replace")` shows a broken sequence as a � mark, instead of stopping with an error.
    ///

    ## Exercise 5: encode

    Write `encode(text, merges)`. `text` is a string and `merges` is what `train_bpe` gives back. It gives back a list of ids: the text's bytes, with every merge applied in order.

    > with the three merges of the table, `encode("aaabdaaabac", merges)` gives `[258, 100, 258, 97, 99]`

    ## Exercise 6: decode

    Write `decode(ids, merges)`, the way back: it gives the text as a string.

    1. Make the table as a list: position $i$ holds the single byte $i$, for $i$ from 0 to 255.
    2. Go through the merges in order. Each adds one entry at the end of the table: the entry of the pair's first id joined to the entry of its second. (The first merge's id is 256, which is exactly the next free position, so the list stays in step with the ids.)
    3. Join the entries of all the `ids`, and decode the result as UTF-8, replacing broken sequences.

    > with the same merges, `decode([258, 100, 258, 97, 99], merges)` gives `"aaabdaaabac"`
    """)
    return


@app.function
def encode(text, merges):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "encode", encode, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Start from `to_bytes(text)`. Then one loop over `merges.items()`, which hands you each pair with its new id, and your `merge` inside it.""",
        r"""It's 4 lines. As in `train_bpe`, store the result of `merge` back into `ids` each time.""",
    ], LOCALE)
    return


@app.function
def decode(ids, merges):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("19", "decode", decode, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The table starts as `[bytes([i]) for i in range(256)]`. In the loop over the merges, append `table[pair[0]] + table[pair[1]]`. You don't need `new_id` here: the position in the list does its job.""",
        r"""After the loop, two lines: join `[table[i] for i in ids]` with `b''.join`, then give back the result's `.decode("utf-8", errors="replace")`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training on the stories

    Finish all six exercises first. The button trains 500 merges on 300 stories, in about a minute or less. Until it's done, the cells that need it say `Ancestor stopped`. That's normal. If you change an exercise later, press the button again. The cell shows how many bytes the text has, how many tokens it became, and the ratio.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the tokenizer")
    train_button
    return (train_button,)


@app.cell
def _(train_button):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    train_stories, test_stories = stories()
    text = " ".join(train_stories[:300])
    raw = to_bytes(text)
    merges = train_bpe(raw, 500)
    tokens = encode(text, merges)
    len(raw), len(tokens), round(len(raw) / len(tokens), 2)
    return merges, test_stories


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 236,000 bytes become about 78,000 tokens: 3 bytes per token on average. Here are the first 15 merges and the last 15, each decoded to show its text. Spaces are part of tokens, so a word and the space after it often become one token:
    """)
    return


@app.cell
def _(merges):
    all_tokens = [decode([new_id], merges) for new_id in merges.values()]
    all_tokens[:15], all_tokens[-15:]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Early merges are bits of letters, like `e ` and `th`. Many late ones are whole words, like `when ` and `lived `, and some even join two, like `smiled and `. Nobody told it what a word is.

    Now a test on text it never saw. The cell cuts a sentence into tokens, then checks that decoding a whole unseen story gives back exactly the same text. The last two items are a sentence in Chinese.
    """)
    return


@app.cell
def _(merges, test_stories):
    def pieces(sentence):
        return [decode([token], merges) for token in encode(sentence, merges)]

    story = test_stories[0]
    (
        pieces("The dog chased a xylophone."),
        decode(encode(story, merges), merges) == story,
        pieces("小狗跑了"),
        decode(encode("小狗跑了", merges), merges),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Common words are single tokens, with their space. `chased` is rare in these stories, so it's in pieces. `xylophone` was never seen, but nothing broke: it's cut into smaller pieces, and no `<unk>` is needed.
    - The whole story comes back exactly, including capitals and punctuation, which our word list threw away.
    - The Chinese comes back too. Our merges learned no Chinese, so it stays as 12 single bytes, and each shows as � in the list, because one byte of a character can't be shown alone. Decoded together, they're whole again. Trained on Chinese text, it would learn Chinese tokens the same way.

    ## Choosing the vocabulary size

    The number of merges is a hyperparameter (chapter 13). More merges mean fewer tokens for the same text, so a model with room for 32 tokens sees further back, but a bigger table with rarer tokens to learn. Models for English commonly use tens of thousands.

    ## Keeping the leaderboard fair

    Perplexity so far has been *per word*. With tokens, a model takes more steps per word, and each step is easier, so perplexity per token isn't comparable with the leaderboard. So when we retrain on tokens, we'll add up the total surprise over a story and divide by the number of *words*, as before, to keep the leaderboard honest.

    ## What's next

    Chapter 17's model was tiny, with some simple choices. **Chapter 20** upgrades it to a modern design, with a better normalisation, a better way of handling position and a gated thinking layer. Chapter 21 then trains it, with a schedule for the learning rate and the ability to save a run and continue it later.

    ## What you can do now

    - explain why word lists and single letters both fall short
    - turn text into UTF-8 bytes and back, in any language
    - run BPE merges by hand, and write the trainer: count pairs, merge the commonest, repeat
    - write `while` loops that move at their own pace
    - encode and decode new text, replaying merges in order
    - say what the vocabulary size trades off

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** BPE for text comes from Sennrich et al. (2015), and the GPT-2 tokenizer first splits text with a regular expression so that merges never cross certain boundaries. Real tokenizers keep *special tokens* such as `<|endoftext|>` outside the merges, and speed up training by updating pair counts instead of recounting them, as we do on every step. A dictionary finds keys instantly through *hashing*.
    """)
    return


if __name__ == "__main__":
    app.run()
