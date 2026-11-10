import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="19 · Tokenization")

with app.setup(hide_code=True):
    import marimo as mo

    from learn_llm import todo
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
    1 byte each, and other scripts take more. Chinese characters take 3. `text.encode("utf-8")` gives the bytes, `list(...)` makes them a plain list, `bytes([100, 111, 103])` makes bytes from a list of numbers, and `.decode("utf-8")` turns bytes back into text.
    """)
    return


@app.cell
def _():
    list("dog".encode("utf-8")), list("你好".encode("utf-8")), bytes([100, 111, 103]).decode("utf-8")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Starting from bytes means any text at all can be represented, in any language, with a starting vocabulary of just 256.

    ## Exercise 1: to_bytes

    Write `to_bytes(text)`:

    ```
    return list(text.encode("utf-8"))
    ```
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

    (At step 2, `Z a` and `a b` are tied at 2 each; we take whichever was seen first. In code, the new ids are 256, 257 and 258.) Common letter pairs merge first, then the pieces join into larger pieces, and common words end up as a single token.

    ## Exercise 2: count_pairs

    Write `count_pairs(ids)`, a dictionary from each neighbouring pair to its count. It's chapter 13's pairing, with chapter 12's counting:

    1. `counts = {}`
    2. `for pair in zip(ids, ids[1:]):`, and inside it `counts[pair] = counts.get(pair, 0) + 1`
    3. After the loop, `return counts`
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
    mo.md(r"""
    ## Exercise 3: merge

    Now the replacement. A `for` loop goes one step at a time, but here we sometimes need to jump two. For that, Python has `while`:
    `while test:` repeats its indented lines for as long as the test holds, and we move the position `i` ourselves.

    **Careful:** if `i` doesn't grow, the test holds forever, and the cell never finishes. If that happens, press the square stop button at the corner of the cell, and check the lines that change `i`.

    Write `merge(ids, pair, new_id)`, giving back a new list. `ids[i] == pair[0]` compares with the first of the pair. `and` requires both tests to hold, and if the first fails, Python skips the rest, which is why the `i < len(ids) - 1` test comes first: we never look past the end. `else:` runs when the `if` test didn't hold.

    1. Two lines: `out = []` and `i = 0`
    2. `while i < len(ids):`
    3. Inside it: `if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:`
    4. Inside that, two lines: `out.append(new_id)` and `i = i + 2`
    5. Lined up with that `if`: `else:`, and inside it two lines: `out.append(ids[i])` and `i = i + 1`
    6. After the loop, `return out`
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
    mo.md(r"""
    ## Exercise 4: train_bpe

    Training repeats the move `count` times, remembering each merge in a dictionary from pair to new id. New ids start at 256, after the bytes.
    `max(counts, key=counts.get)` gives the key with the biggest count, in the style of chapter 12's `sorted`.

    Write `train_bpe(ids, count)`:

    1. `merges = {}`
    2. `for step in range(count):`
    3. Inside it, five lines: `counts = count_pairs(ids)`, `pair = max(counts, key=counts.get)`, `new_id = 256 + step`, `merges[pair] = new_id`, `ids = merge(ids, pair, new_id)`
    4. After the loop, `return merges`

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
    mo.md(r"""
    ## Encoding and decoding

    To **encode** new text, turn it into bytes and replay every merge in order. `d.items()` goes through a dictionary's pairs of key and value, so `for pair, new_id in merges.items():` gives both at once.

    To **decode**, we need the text behind each id. Build a table: ids below 256 are single bytes, and each merged id is its two parts joined. Since a pair's parts always have smaller ids, going through merges in order always finds them ready.
    `bytes([i])` makes one byte, and `+` joins bytes. `errors="replace"` means a broken sequence shows as a � mark instead of stopping with an error.

    ## Exercise 5: encode

    Write `encode(text, merges)`:

    1. `ids = to_bytes(text)`
    2. `for pair, new_id in merges.items():`, and inside it `ids = merge(ids, pair, new_id)`
    3. After the loop, `return ids`

    ## Exercise 6: decode

    Write `decode(ids, merges)`. A list works as the table, since position 256 gets filled just after position 255:

    1. `table = [bytes([i]) for i in range(256)]`
    2. `for pair, new_id in merges.items():`, and inside it `table.append(table[pair[0]] + table[pair[1]])`
    3. After the loop, `joined = b''.join([table[i] for i in ids])`
    4. `return joined.decode("utf-8", errors="replace")`

    (`b''.join(...)` is chapter 12's join, for bytes with nothing between them.)
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

    Chapter 16's model was tiny, with some simple choices. **Chapter 20** upgrades it to a modern design, with a better normalisation, a better way of handling position, a schedule for the learning rate, and the ability to save a run and continue it later.

    ## What you can do now

    - explain why word lists and single letters both fall short
    - turn text into UTF-8 bytes and back, in any language
    - write a BPE trainer: count pairs, merge the commonest, repeat
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
