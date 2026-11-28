import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="25 · Data")

with app.setup(hide_code=True):
    import zlib

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.diagrams import pipeline
    from learn_llm.rawdata import common_words, pile_tokens, raw_pile, seeded_hash
    from learn_llm.scaling import small_run, story_tokens
    from learn_llm.tokenizer import to_words

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 25 · Data

    Every model in this course was trained on clean text: a few thousand stories, each one different, each one well-written English. Text gathered for a real model is nothing like that.
    It's collected by programs that crawl billions of web pages and keep whatever they find. The same article appears on a hundred sites. Every page carries navigation menus and notices. A good share is not real language at all.

    A model learns what it reads. Chapter 24 measured what more text buys. This chapter is about what the text is worth, and how to turn a raw pile into data worth training on, in three steps:

    1. remove **exact copies**;
    2. remove **near-copies**, which differ by a few words;
    3. remove **junk**.

    ## A raw pile

    We've built a small raw pile from the stories: 2,000 texts, of which only half are what we want.
    """)
    return


@app.cell
def _():
    texts, kinds = raw_pile()
    [texts[kinds.index(kind)][:75] for kind in ["copy", "near-copy", "gibberish", "menu", "spam"]]
    return kinds, texts


@app.cell(hide_code=True)
def _(kinds, texts):
    kind_of = dict(zip(texts, kinds))

    def tally(some_texts):
        counts = {}
        for text in some_texts:
            counts[kind_of[text]] = counts.get(kind_of[text], 0) + 1
        return counts
    return kind_of, tally


@app.cell
def _(tally, texts):
    tally(texts)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The first cell shows the start of one text of each unwanted kind. The second counts the pile: 1,000 stories, 250 copies of some of them with other spacing or capitals, 250 near-copies with a few words changed, and 500 pieces of junk of three kinds.

    `kinds` holds the answer for each text, and `tally` counts them. **A real pile has no such labels.** We planted the problems so that you can count what your cleaning catches. Your functions will never look at `kinds`.

    ## Exact copies, and hashing

    To find copies, the obvious way is to compare every text with every other one. For 2,000 texts that's two million comparisons. For a billion pages it's out of the question.

    The standard tool is a **hash function**: a fixed recipe that turns any text into a whole number (its **hash value**). The same text always produces the same hash value. So instead of comparing texts pair by pair, we compute each text's hash once and look for values we have already seen.
    Checking whether a value is in a set takes the same short time no matter how big the set grows (chapter 20: a set stores items by hash, so it jumps straight to the right slot instead of scanning).

    ## By hand: a toy hash

    Here's a hash function small enough to do by hand. Number the letters, $a = 1$, $b = 2$, $c = 3$, $d = 4$. Add up a word's letters and keep the last digit of the total. What's the hash of `cab`, of `bad`, and of `abc`?
    """)
    return


@app.cell(hide_code=True)
def _():
    hash_boxes = hand.boxes(['hash of cab', 'hash of bad', 'hash of abc'])
    return (hash_boxes,)


@app.cell(hide_code=True)
def _(hash_boxes):
    hand.marked(hash_boxes, {'hash of cab': 6, 'hash of bad': 7, 'hash of abc': 6}, {
        'hash of cab': r"""$3 + 1 + 2$""",
        'hash of bad': r"""$2 + 1 + 4$""",
        'hash of abc': r"""the same letters as `cab`, in another order""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `cab` and `abc` are different words with the same hash. That's a **collision**, and it's the price of squeezing any text into a finite number: there are more possible texts than numbers.
    Our toy has two faults. It has only 10 possible answers, and it ignores the order of the letters. A real hash function mixes every byte with its position so thoroughly that changing one letter changes the whole number, and it has billions of possible answers.
    Two different texts then collide about once in four billion pairs. For cleaning data, that's good enough: now and then a good text is thrown out by mistake.

    One more thing before hashing. To a hash function, `The cat` and `the  cat` are different texts. But for us they're the same text copied by another site. So we first **normalise**: put the text into one standard form, lower-case with single spaces, and hash that.

    /// admonition | 🧰 Toolbox: words, and a real hash
    - `text.split()` cuts a text at every run of spaces or line breaks and gives the pieces as a list: `"a  b\nc".split()` is `["a", "b", "c"]`.
    - `" ".join(words)` glues a list of strings back together with single spaces (chapter 12): `" ".join(["hi", "there"])` is `"hi there"`.
    - `zlib.crc32(data)` is a real hash function, the one that checks zip files. It takes bytes (turn a string into bytes with `.encode("utf-8")`, chapter 19) and gives a whole number from 0 to about 4 billion: `zlib.crc32(b"hi")` is 3633523372. `zlib` is already loaded.
    ///

    ## Exercise 1: normalise

    Write `normalise(text)`: the text in lower case, with every run of spaces and line breaks turned into one space, and none at the start or the end.

    `normalise("  The  CAT\nsat ")` must give `"the cat sat"`.
    """)
    return


@app.function
def normalise(text):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "normalise", normalise, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Cut the lower-case text into words, then glue the words back together with `" ".join(...)` (from chapter 12).""",
        r"""It's one line: `" ".join(...)` around the list that `.split()` gives.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 2: fingerprint

    Write `fingerprint(text)`: the `zlib.crc32` hash of the normalised text. Two texts that differ only in spacing or capitals must get the same fingerprint.

    `fingerprint(" THE CAT SAT ")` must give 2690722008, the same as `fingerprint("the cat sat")`.
    """)
    return


@app.function
def fingerprint(text):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "fingerprint", fingerprint, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three steps, each feeding the next: normalise, turn into bytes, hash. Chapter 19 turned text into bytes with `.encode("utf-8")`.""",
        r"""It's one line: `zlib.crc32(...)` around the encoded, normalised text.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: a set that grows
    `seen = set()` makes an empty set, and `seen.add(5)` puts 5 in it. Adding something that's already there changes nothing. `5 in seen` and `5 not in seen` ask whether it's there.
    ///

    ## Exercise 3: drop_copies

    Write `drop_copies(texts)`. `texts` is a list of strings. It gives back a new list with the first text of each fingerprint, in the same order, and without the later texts that have a fingerprint already seen.
    The texts it keeps are the originals, as they were written, not their normalised forms.

    `drop_copies(["A b", "c", "a  B", "c", "d"])` must give `["A b", "c", "d"]`.
    """)
    return


@app.function
def drop_copies(texts):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "drop_copies", drop_copies, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two collections, both empty at the start: a set for the fingerprints seen and a list for the texts kept. Go through the texts once. Each text either has a new fingerprint, and then both collections grow, or it doesn't, and then nothing happens.""",
        r"""It's about 7 lines. Inside the loop, keep the text's fingerprint in a name, then `if` it's `not in` the set: add it to the set and append the text to the list. Give back the list after the loop.""",
    ], LOCALE)
    return


@app.cell
def _(tally, texts):
    kept = drop_copies(texts)
    len(kept), tally(kept)
    return (kept,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    All 250 copies are gone, and so are a few menus that happened to be identical. One pass over the pile, and one small number remembered for each text.

    The 250 near-copies are all still there. Change one word of a story and its fingerprint is a completely different number, by design. A hash can say "the same" but never "nearly the same".

    ## Near-copies: how alike are two texts?

    We need a number for how alike two texts are. Counting shared words isn't enough, since all stories share words like `the` and `was`. What copies share is *runs* of words. So cut each text into every run of $n$ words in a row, called its **shingles**, and collect them in a set.
    With $n = 2$, `the cat sat down` has three shingles: `(the, cat)`, `(cat, sat)` and `(sat, down)`.

    The likeness of two sets $A$ and $B$ is then measured by the **Jaccard similarity**:

    > $$J(A, B) = \frac{\text{how many items are in both}}{\text{how many items are in at least one}}$$

    It's 1 for identical sets and 0 for sets with nothing in common.

    ## By hand: two sentences

    Take `the cat sat on the mat` and `the cat sat on the rug`, with shingles of 2 words. Each sentence has 5 shingles. How many are in both sentences? How many different shingles are there in all? And what's the Jaccard similarity, to two digits?
    """)
    return


@app.cell(hide_code=True)
def _():
    jaccard_boxes = hand.boxes(['in both', 'in at least one', 'Jaccard'], step=0.01)
    return (jaccard_boxes,)


@app.cell(hide_code=True)
def _(jaccard_boxes):
    hand.marked(jaccard_boxes, {'in both': 4, 'in at least one': 6, 'Jaccard': 0.67}, {
        'in both': r"""the two sentences agree up to `on the`""",
        'in at least one': r"""the shared ones, plus `(the, mat)` and `(the, rug)`""",
        'Jaccard': r"""the first answer divided by the second""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One changed word out of six, and the similarity is still 0.67. With longer texts a changed word matters even less.

    /// admonition | 🧰 Toolbox: more about sets
    - A set can be written with curly brackets: `{1, 2, 3}`. `set(a_list)` makes one from a list, dropping repeats.
    - `a & b` is the set of items in both, and `a | b` the set of items in at least one: `{1, 2, 3} & {2, 3, 4}` is `{2, 3}`, and `{1, 2, 3} | {2, 3, 4}` is `{1, 2, 3, 4}`. `len` counts a set's items.
    - A list can't be put in a set because its contents can change later — if it changed, its hash would change too, and the set could no longer find it. `tuple(a_list)` makes a fixed copy that can go in a set: `tuple(["a", "b"])` is `("a", "b")`.
    ///

    ## Exercise 4: shingles

    Write `shingles(words, n)`. `words` is a list of strings with at least `n` entries. It gives back the set of all runs of `n` words in a row, each run as a tuple.

    `shingles(["a", "b", "c", "d"], 2)` must give `{("a", "b"), ("b", "c"), ("c", "d")}`.
    """)
    return


@app.function
def shingles(words, n):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "shingles", shingles, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""A run of `n` words starting at position `i` is a slice `words[i:i + n]` (chapter 09). Where can the last run start? In the example, 4 words and runs of 2 give starts 0, 1 and 2.""",
        r"""It's 1 to 4 lines. Either start from an empty set and `add` each `tuple(words[i:i + n])` in a loop over `range(len(words) - n + 1)`, or build the list of tuples in square brackets and wrap it in `set(...)`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: jaccard

    Write `jaccard(a, b)` for two sets, neither of them empty: the formula above.

    For `{1, 2, 3, 4, 5}` and `{1, 2, 3, 4, 6}` it must give $4 / 6$, about 0.67.
    """)
    return


@app.function
def jaccard(a, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "jaccard", jaccard, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""The Toolbox box has one tool for the top of the fraction and one for the bottom.""",
        r"""It's one line: a `len` divided by a `len`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## MinHash: a set in a few numbers

    Jaccard brings back the problem we started with. It compares two texts at a time, and there are too many pairs. Worse, each comparison now handles two whole sets of shingles. We'd like what the fingerprint gave us: a little to remember for each text, and a quick comparison.

    Here's the idea, one of the neatest in computing. Hash every shingle of a set, and keep only the **smallest** hash — call it the set's minimum. Now ask: when do two sets $A$ and $B$ have the same minimum?

    Think of it as a fair lottery. Suppose the union of $A$ and $B$ has 5 shingles in all: 3 shared by both sets, and 2 that belong to only one. A good hash scatters numbers as if at random, so picking the smallest hash is just like drawing 1 winning ticket at random from those 5 shingles:

    - If the winner is one of the 3 **shared** shingles, both sets have it, so it is the smallest in each — the two minimums match.
    - If the winner is one of the 2 **unshared** shingles, only one set has it, so that set's minimum is smaller than the other's — they differ.

    So the chance that the two minimums match is $3 / 5$ — the share of all shingles that are in both sets, which is **exactly the Jaccard similarity**!

    With just one hash function, each set gets one number, which only answers "match or not" — too rough an estimate. For a finer one, run many different hash functions and keep the minimum under each. That list of minimums is the set's **signature**, and the share of positions where two signatures agree estimates their Jaccard similarity. This is **MinHash**.

    ## By hand: three hash functions

    Two sets of shingles, named by letters: $X = \{P, Q, R, S\}$ and $Y = \{Q, R, S, T\}$. They share 3 shingles out of 5, so their Jaccard similarity is 0.6. Here are three hash functions' values:

    | shingle | hash 1 | hash 2 | hash 3 |
    |---|---:|---:|---:|
    | $P$ | 7 | 1 | 6 |
    | $Q$ | 2 | 8 | 7 |
    | $R$ | 9 | 5 | 1 |
    | $S$ | 4 | 3 | 8 |
    | $T$ | 6 | 9 | 4 |

    Write down each set's signature: its smallest value under hash 1, then hash 2, then hash 3. Remember that $X$ has no $T$ and $Y$ has no $P$. Then count the positions where the two signatures agree.
    """)
    return


@app.cell(hide_code=True)
def _():
    minhash_boxes = hand.boxes(['X, hash 1', 'X, hash 2', 'X, hash 3', 'Y, hash 1', 'Y, hash 2', 'Y, hash 3', 'positions that agree'])
    return (minhash_boxes,)


@app.cell(hide_code=True)
def _(minhash_boxes):
    hand.marked(minhash_boxes, {'X, hash 1': 2, 'X, hash 2': 1, 'X, hash 3': 1, 'Y, hash 1': 2, 'Y, hash 2': 3, 'Y, hash 3': 1, 'positions that agree': 2}, {
        'X, hash 1': r"""the smallest of 7, 2, 9 and 4""",
        'X, hash 2': r"""the smallest of 1, 8, 5 and 3""",
        'X, hash 3': r"""the smallest of 6, 7, 1 and 8""",
        'Y, hash 1': r"""the smallest of 2, 9, 4 and 6""",
        'Y, hash 2': r"""the smallest of 8, 5, 3 and 9: the 1 belongs to $P$, which $Y$ doesn't have""",
        'Y, hash 3': r"""the smallest of 7, 1, 8 and 4""",
        'positions that agree': r"""compare your two signatures, position by position""",
    }, LOCALE, per_row=3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Two positions of three agree, an estimate of 0.67 for a true 0.6. Under hash 2 the smallest value of all belongs to $P$, which only $X$ has, so the minimums differ. Three hash functions give a rough estimate, and real signatures have dozens to hundreds.

    We need many hash functions, and `seeded_hash(seed, item)` provides them: each whole number `seed` gives a different one. `seeded_hash(0, s)` and `seeded_hash(1, s)` are unrelated numbers for the same shingle `s`.

    ## Exercise 6: minhash

    Write `minhash(shingle_set, seeds)`. `shingle_set` is a set that isn't empty, and `seeds` is a list of whole numbers. It gives back the signature: a list with one number per seed, the smallest `seeded_hash(seed, s)` over all the shingles `s` of the set.

    For the set `{"p"}` and the seeds `[5]` it must give `[seeded_hash(5, "p")]`: with a single shingle, the smallest hash is that shingle's.
    """)
    return


@app.function
def minhash(shingle_set, seeds):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "minhash", minhash, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two loops, one inside the other: the outer over the seeds, the inner over the shingles. For each seed, `min` (chapter 12) of a list of hashes gives one number of the signature.""",
        r"""It's 1 to 4 lines. For one seed, the number is `min([seeded_hash(seed, s) for s in shingle_set])`. Collect one of those per seed in a list.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 7: agreement

    Write `agreement(sig_a, sig_b)` for two signatures of the same length: the share of positions where they hold the same number.

    For `[1, 2, 3, 4]` and `[1, 5, 3, 7]` it must give 0.5.
    """)
    return


@app.function
def agreement(sig_a, sig_b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "agreement", agreement, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Walk through both lists together with `zip` (chapter 01) and count the pairs that are equal. Then divide by the length.""",
        r"""It's 1 to 5 lines. A counter that goes up by 1 inside an `if a == b:` works. So does `sum` over a list of comparisons, since `True` counts as 1.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now the pile. The cell below cuts each text that's left into words (`to_words` is chapter 12's lower-case words), takes its shingles of 3 words, and computes a signature of 32 numbers. It takes a few seconds:
    it's the one slow step, and it's done once for each text, not once for each pair.
    """)
    return


@app.cell
def _(kept):
    word_lists = [to_words(text) for text in kept]
    signatures = np.array([minhash(shingles(words, 3), range(32)) for words in word_lists])
    signatures.shape
    return signatures, word_lists


@app.cell(hide_code=True)
def _(kept, kind_of, signatures):
    near_i = [kind_of[text] for text in kept].index("near-copy")
    original_i = int(np.argmax(np.mean(signatures[:near_i] == signatures[near_i], axis=1)))
    return near_i, original_i


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    A table with one row of 32 numbers for each text. Let's test the estimate on a pair we know: `near_i` is the position of one near-copy, and `original_i` the position of the story it was made from.
    The cell shows their true Jaccard similarity, from the full sets, and the estimate from the two signatures (`.tolist()` turns a row of the table back into a plain list):
    """)
    return


@app.cell
def _(near_i, original_i, signatures, word_lists):
    true_likeness = jaccard(shingles(word_lists[near_i], 3), shingles(word_lists[original_i], 3))
    round(true_likeness, 3), agreement(signatures[near_i].tolist(), signatures[original_i].tolist())
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 0.75 and 0.91. The estimate is rough: 32 numbers can only agree in steps of $\frac{1}{32}$, and luck decides each one, so being off by 0.1 is ordinary. It doesn't need to be exact.
    Two unrelated stories have a similarity under 0.01, and no amount of luck confuses 0.01 with 0.75.

    The next cell walks through the texts in order and keeps a text only if its signature agrees with every earlier kept text in less than half the positions ($< 0.5$). The `np.mean` line does the same job as your `agreement`, except NumPy compares against all kept signatures at once using broadcasting:

    - Suppose we have kept $k$ texts so far: `signatures[unique]` has shape `(k, 32)` (one 32-number row per kept text), while `signatures[i]` has shape `(32,)`;
    - `signatures[unique] == signatures[i]` compares that `(32,)` row against every one of the $k$ rows element by element, giving a `(k, 32)` table of `True`/`False` values;
    - averaging along each row (`axis=1`, where `True` counts as 1 and `False` as 0) gives all $k$ agreement shares in one shot.
    """)
    return


@app.cell
def _(kept, signatures, tally):
    unique = [0]
    for i in range(1, len(kept)):
        agreements = np.mean(signatures[unique] == signatures[i], axis=1)
        if np.max(agreements) < 0.5:
            unique.append(i)
    deduped = [kept[i] for i in unique]
    len(deduped), tally(deduped)
    return (deduped,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every near-copy is gone, and every story is still there. Nearly all the menus are gone too: they're one template with a few numbers changed, near-copies of each other, and one of them is kept as the "original".
    That's typical. Text that machines produce from templates is the most duplicated text on the web.

    ## Junk: features and a classifier

    What's left is stories, gibberish and spam, and each of those is one of a kind, so no copy-finder helps. We need to judge a text on its own. Two simple measurements, called **features**, already tell a lot:

    - the share of its words that are common English words (we use the 2,000 most frequent words of the stories);
    - the share of its words that are different from each other.

    ## By hand: two features

    Take the text `buy buy cheap buy now`, and suppose the common words are `buy` and `now`. What share of its 5 words are common? And there are 3 different words in it: what share of 5 is that?
    """)
    return


@app.cell(hide_code=True)
def _():
    feature_boxes = hand.boxes(['share of common words', 'share of different words'], step=0.1)
    return (feature_boxes,)


@app.cell(hide_code=True)
def _(feature_boxes):
    hand.marked(feature_boxes, {'share of common words': 0.8, 'share of different words': 0.6}, {
        'share of common words': r"""`buy` three times and `now` once: 4 of the 5""",
        'share of different words': r"""`buy`, `cheap` and `now`: 3 of 5""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 8: known_share

    Write `known_share(words, known)`. `words` is a list of strings that isn't empty, and `known` is a set of strings. It gives back the share of the entries of `words` that are in `known`. A word that appears three times counts three times.

    For `["the", "cat", "zzz", "sat"]` and the set `{"the", "cat", "sat"}` it must give 0.75.
    """)
    return


@app.function
def known_share(words, known):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("25", "known_share", known_share, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""It's `agreement` again with another test: count the words for which `w in known` holds, and divide by the number of words.""",
        r"""It's 1 to 5 lines: a counter and a loop with an `if`, or `sum` over a list of `w in known` tests.""",
    ], LOCALE)
    return


@app.cell
def _(kinds, texts):
    known = common_words()

    def features(text):
        words = to_words(text)
        return [known_share(words, known), len(set(words)) / len(words)]

    np.round([features(texts[kinds.index(kind)]) for kind in ["story", "gibberish", "spam"]], 2)
    return (features,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    One row each for a story, a piece of gibberish and a piece of spam. The story has nearly all common words, and about half its words are different. Gibberish has almost no common words. Spam repeats a handful of words.

    We could now pick two thresholds by eye. But real piles need dozens of features, and nobody can tune dozens of thresholds by eye. Chapters 06 to 10 solved exactly this: let a **classifier** learn the rule from examples.
    Somebody reads a few hundred texts and labels each as good or junk, a classifier is trained on those, and it then judges the millions nobody will ever read. This is called a **quality classifier**.

    Below, the first 200 texts play the labelled ones, with their labels taken from our planted answers: 1 for junk, 0 for a story. The model is chapter 10's smallest: one linear layer (`nn.Linear(2, 2)`) mapping the 2 features to 2 scores.
    """)
    return


@app.cell
def _(deduped, features, kind_of):
    X = torch.tensor([features(text) for text in deduped])
    labels = torch.tensor([kind_of[text] != "story" for text in deduped]).long()
    torch.manual_seed(0)
    classifier = nn.Linear(2, 2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1)
    for step in range(300):
        optimizer.zero_grad()
        F.cross_entropy(classifier(X[:200]), labels[:200]).backward()
        optimizer.step()
    guesses = torch.argmax(classifier(X), dim=1)
    return guesses, labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    (`.long()` turns `True` and `False` into the whole numbers 1 and 0 that `F.cross_entropy` wants as labels.)

    How good is it on the texts it wasn't trained on? Chapter 18's three counts answer that: junk it caught, good stories it wrongly threw out, and junk it let through.
    """)
    return


@app.cell
def _(guesses, labels):
    new_guesses, new_labels = guesses[200:], labels[200:]
    caught = torch.sum((new_guesses == 1) & (new_labels == 1)).item()
    thrown_out = torch.sum((new_guesses == 1) & (new_labels == 0)).item()
    let_through = torch.sum((new_guesses == 0) & (new_labels == 1)).item()
    caught, thrown_out, let_through
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Every piece of junk caught, no story lost: both precision (what share of what it cut was real junk) and recall (what share of all real junk it caught) are 1. Be suspicious of a result that clean. Our junk was planted, and planted junk is easy to spot.
    Real junk shades into real text: a page that's half advertisement, clunky machine-translated text, a story that makes no sense. Real pipelines use the same method with larger classifiers and richer features.
    One feature is a language model's own perplexity on the text, since text a model finds very surprising is often not real language.
    They also settle for precision and recall below 1, and choose which mistake costs more.

    Keep the texts the classifier calls good:
    """)
    return


@app.cell
def _(deduped, guesses, tally):
    clean = [deduped[i] for i in range(len(deduped)) if guesses[i] == 0]
    len(clean), tally(clean)
    return (clean,)


@app.cell(hide_code=True)
def _():
    pipeline([("raw pile", "2,000"), ("no copies", "1,744"), ("no near-copies", "1,351"), ("clean", "1,000")], ["fingerprints", "MinHash", "classifier"], "The three cleaning steps and how many texts each one leaves")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The pile is half its size, and all of it is stories.

    ## Does it matter?

    The honest test is to train on both. The button trains chapter 24's small model twice, with 64 numbers per token and 150,000 tokens read: once on the raw pile, once on your clean one.
    Both get the same compute. Both are scored on the same test stories. It takes about 20 seconds.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train on the raw pile and on the clean one (about 20 seconds)")
    train_button
    return (train_button,)


@app.cell
def _(clean, texts, train_button):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    _train_tokens, test_tokens, vocab_size = story_tokens()
    raw_loss = small_run(64, 150000, pile_tokens(texts), test_tokens, vocab_size)
    clean_loss = small_run(64, 150000, pile_tokens(clean), test_tokens, vocab_size)
    round(raw_loss, 3), round(clean_loss, 3)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    On our machine: about 4.48 for the raw pile and 4.39 for the clean one. The clean model is 2% better for the same compute. That sounds small until you price it with chapter 24's law:
    each doubling of the text took 9% off the loss, so 2% is what about 15% more text would buy. Here it cost a few seconds of cleaning.

    On a real pile the gain is larger, and not only in the loss. A model trained on copies learns to recite them. A model trained on menus writes menus. And what a model should never say is best removed before it reads it.
    The people who build large models spend much of their effort on this step.

    ## What's next

    The model now reads good text, and what it learns from text is to continue it. That isn't the same as doing what you ask. **Chapter 26** is about the last step of training, teaching a model to respond the way we want:
    from examples, from comparisons, and from rewards.

    ## What you can do now

    - explain what a hash function is, what a collision is, and why text is normalised first
    - remove exact copies from a pile in one pass
    - measure how alike two texts are with shingles and the Jaccard similarity
    - explain why two sets' smallest hashes agree with a chance equal to their similarity, and build MinHash signatures
    - describe a text with features and train a quality classifier on a few labelled examples
    - measure what cleaning is worth by training on the raw and on the clean data

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** MinHash is from Broder (1997), who used it to find duplicate pages for the AltaVista search engine. Comparing every signature with every kept one is still too slow for billions of pages:
    *locality-sensitive hashing* cuts each signature into bands and only compares texts that share a whole band. The quality classifiers of large data pipelines are often *fastText* models, and Python's `hashlib` has stronger hash functions than `zlib.crc32`.
    """)
    return


if __name__ == "__main__":
    app.run()
