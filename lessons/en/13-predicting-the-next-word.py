import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="13 · Predicting the next word")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np

    from learn_llm import hand, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.leaderboard import record, show

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 13 · Predicting the next word

    Every chatbot you've used does one thing, over and over: given the words so far, it predicts
    the next one. A model that does this is called a **language model**. It's the idea this
    whole course has been heading for.

    In this chapter you'll build the simplest language models there are, with no network and no
    gradient at all, only counting. They're worth building for two reasons. They generate real
    (if silly) text. And they give us a **score** that every later model has to beat.

    ## The text as numbers

    As in chapter 12, we cut the stories into words and keep the 2,000 most common. This time
    two extra entries join the vocabulary:

    - `<unk>` ("unknown") stands for any word outside the 2,000;
    - `<end>` marks the end of a story, so that a model can learn when to stop.

    `to_words` and `count_words` are chapter 12's. `story_ids` is new: it turns a list of stories into one long array of ids.
    This time we use all 12,000 stories for training, and the 1,000 test stories for measuring.
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.function
def count_words(words):
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    return counts


@app.function
def story_ids(story_list, index):
    ids = []
    for story in story_list:
        for word in to_words(story):
            ids.append(index.get(word, index["<unk>"]))
        ids.append(index["<end>"])
    return np.array(ids)


@app.cell
def _():
    train_stories, test_stories = stories()
    counts = count_words(to_words(" ".join(train_stories)))
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000] + ["<unk>", "<end>"]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    train_ids = story_ids(train_stories, index)
    test_ids = story_ids(test_stories, index)
    len(vocab), len(train_ids), len(test_ids)
    return index, test_ids, train_ids, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    2,002 entries, about 1.9 million words to learn from, and about 159,000 kept aside. `index.get(word, index["<unk>"])`
    gives the word's id, or `<unk>`'s id if the word isn't known. (The cell can take a few seconds. So can the cell in the next section that fills the table.)

    ## Chances that depend on what came before

    After `once upon a`, the next word is almost surely `time`. After `the`, it could be many things.
    So we want the chance of each next word **given** what came before. That's called a
    **conditional probability**, written with a bar:

    > $P(\text{next} \mid \text{previous})$ is "the chance of `next`, given that we've just seen `previous`"

    The simplest way to estimate it is to count. Take this tiny text, where the full stop counts as a word:

    > the dog ran . the dog sat . the cat ran .

    `the` appears 3 times; twice `dog` came next, and once `cat`. So $P(\text{dog} \mid \text{the}) = 2 / 3$,
    $P(\text{cat} \mid \text{the}) = 1 / 3$, and every other word gets 0. A table of these counts, with one row for each
    word and one column for each possible next word, is called a **bigram** model ("bi" for two: it
    looks at pairs).

    ## By hand: a table of counts

    Here's the table for the tiny text, with the row for `the` filled in. A row is the word that came first, a column the word that came next.

    | first ↓ next → | the | dog | cat | ran | sat | . |
    |---|---|---|---|---|---|---|
    | **the** | 0 | 2 | 1 | 0 | 0 | 0 |
    | **dog** | 0 | 0 | 0 | A | B | 0 |
    | **ran** | 0 | 0 | 0 | 0 | 0 | C |
    | **.** | D | 0 | 0 | 0 | 0 | 0 |

    Fill in A to D by counting pairs in the text. Then two questions: how many neighbouring pairs
    does the text have in all? (It has 12 words.) And what is $P(\text{ran} \mid \text{dog})$?
    """)
    return


@app.cell(hide_code=True)
def _():
    count_boxes = hand.boxes(['A', 'B', 'C', 'D', 'pairs in all', 'P(ran | dog)'], step=0.01)
    return (count_boxes,)


@app.cell(hide_code=True)
def _(count_boxes):
    hand.marked(count_boxes, {'A': 1, 'B': 1, 'C': 2, 'D': 2, 'pairs in all': 11, 'P(ran | dog)': 0.5}, {
        'A': r"""how many times does `ran` come right after `dog`?""",
        'B': r"""how many times does `sat` come right after `dog`?""",
        'C': r"""`ran` is followed by a full stop each time it appears""",
        'D': r"""the last full stop has nothing after it, so only the first two count""",
        'pairs in all': r"""every word but the last starts one pair""",
        'P(ran | dog)': r"""A divided by the total of the row for `dog`""",
    }, LOCALE, per_row=6)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: neighbouring pairs
    - `items[1:]` is a slice with no stop: everything from position 1 on, so the list without its first item.
    - `zip` (chapter 01) pairs two lists item by item, and stops when the shorter one runs out. Pair a list with
      its own `[1:]` and you get every neighbouring pair: for `letters = ["a", "b", "c"]`,
      `zip(letters, letters[1:])` gives `("a", "b")`, then `("b", "c")`.
    - `for first, second in zip(...)` names the two halves of each pair as it goes.
    ///

    ## Exercise 1: bigram_counts

    Write `bigram_counts(ids, size)`. `ids` is a list or array of ids, each from 0 to `size - 1`. It gives back a NumPy
    table `C` of shape `(size, size)`, where `C[a, b]` is how many times `b` came right after `a`:

    > `bigram_counts([0, 1, 0, 1, 2], 3)` gives `[[0, 2, 0], [1, 0, 1], [0, 0, 0]]`

    Here 1 came after 0 twice, 0 after 1 once, and 2 after 1 once. Nothing ever came after 2, so its row stays 0.
    """)
    return


@app.function
def bigram_counts(ids, size):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "bigram_counts", bigram_counts, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Start from a table of zeros (`np.zeros`, with the shape in its own pair of brackets). Visit every neighbouring pair with the Toolbox's `zip`, and raise that pair's entry by 1. Give the table back after the loop.""",
        r"""It's 4 lines. The loop is `for a, b in zip(ids, ids[1:]):`, and the one line inside it changes `C[a, b]`.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The trouble with zero

    To turn a row of counts into chances, divide by the row's total. But any pair never seen in
    training then gets a chance of exactly 0, and chapter 07's surprise, $-\log 0$, is infinite. One unseen pair
    in the test stories would make the whole score infinite, as if the model had called it impossible.

    The usual fix is to pretend every pair was seen a small number of extra times, $k$. This is called
    **smoothing**: add $k$ to every count first, then divide by the row's new total.

    > $$P(b \mid a) = \frac{C[a, b] + k}{\text{total of row } a \text{, after adding } k \text{ to each of its entries}}$$

    ## By hand: smooth one row

    Take the row for `the` from the table above, $[0, 2, 1, 0, 0, 0]$, and smooth it with $k = 1$.
    What are the new count for `dog`, the new count for `sat`, and the row's new total?
    Then, as fractions of that total: before smoothing, `dog` had $2 / 3$ of the row. Does it have more or less now?
    Write its new chance to two decimals.
    """)
    return


@app.cell(hide_code=True)
def _():
    smooth_boxes = hand.boxes(['new count for dog', 'new count for sat', 'new total', 'new chance of dog'], step=0.01)
    return (smooth_boxes,)


@app.cell(hide_code=True)
def _(smooth_boxes):
    hand.marked(smooth_boxes, {'new count for dog': 3, 'new count for sat': 1, 'new total': 9, 'new chance of dog': 0.33}, {
        'new count for dog': r"""$2 + k$""",
        'new count for sat': r"""$0 + k$: a pair never seen now counts as seen once""",
        'new total': r"""the old total was 3, and each of the 6 entries got 1 more""",
        'new chance of dog': r"""the new count divided by the new total""",
    }, LOCALE, tolerance=0.006)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The chance of `dog` fell from 0.67 to 0.33, and `sat` rose from 0 to 0.11. With only 3 real counts in the row, $k = 1$ is
    far too much: the made-up counts outweigh the real ones. We'll use a much smaller $k$ on the real text.

    ## Exercise 2: bigram_probs

    Write `bigram_probs(C, k)`. `C` is a table of counts of shape `(size, size)` and `k` a number. It gives back a
    table of the same shape, by the formula above, so each row adds up to 1:

    > `bigram_probs(np.array([[1.0, 3.0], [0.0, 0.0]]), 0.5)` gives `[[0.3, 0.7], [0.5, 0.5]]`

    The first row becomes $[1.5, 3.5]$, with total 5. The second row had no counts at all, and smoothing gives it equal chances.
    You'll need one total per row, kept in a shape that divides row by row: chapter 07's `softmax` did the same.
    """)
    return


@app.function
def bigram_probs(C, k):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "bigram_probs", bigram_probs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two steps: add `k` to the whole table at once (a number plus an array adds it to every entry), then divide by the row totals of the new table. `np.sum` with `axis=1` gives one total per row.""",
        r"""It's 2 lines. The first is `smoothed = C + k`. In the second, `np.sum` also needs `keepdims=True`, so that the totals have shape `(size, 1)` and each row is divided by its own.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's what the model thinks can follow `the`, and `once`: the five likeliest next words,
    with their chances. The cell uses your two functions, so it shows an error until Exercises 1 and 2 pass.
    That's normal, and so it is for the later cells that need them.

    /// admonition | 🧰 Toolbox: rounding
    `round(x, 3)` keeps 3 digits after the point: `round(0.12345, 3)` is `0.123`. It's only for showing numbers.
    ///

    (`np.argsort(-row)` lists positions from largest to smallest, as in chapter 12.
    After `the`, the likeliest "word" is `<unk>`: one entry stands for thousands of rare words.)
    """)
    return


@app.cell
def _(index, train_ids, vocab):
    C = bigram_counts(train_ids, len(vocab))
    P = bigram_probs(C, 0.01)
    likely = []
    for start_word in ["the", "once"]:
        row = P[index[start_word]]
        best = np.argsort(-row)[:5]
        likely.append([(vocab[j], round(row[j].item(), 3)) for j in best])
    likely
    return C, P


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Writing text

    To generate text, start from a word, draw the next word at random using its row of chances,
    and repeat. Drawing **at random**, rather than always taking the likeliest word, matters: the likeliest word after
    `the` might always be the same, and the text would soon go round in circles. Drawing from a list of chances is called **sampling**.

    /// admonition | 🧰 Toolbox: drawing with chances
    `rng.choice(n, p=chances)` draws one whole number from $0$ to $n - 1$. `chances` holds $n$ numbers that add up to 1,
    and number $i$ is drawn with chance `chances[i]`. So `rng.choice(3, p=[0.2, 0.5, 0.3])` gives 1 about half the time.
    ///

    The cell below draws 10 times with those chances.
    """)
    return


@app.cell
def _():
    rng_demo = np.random.default_rng(1)
    [rng_demo.choice(3, p=[0.2, 0.5, 0.3]) for _draw in range(10)]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 3: sample_next

    Write `sample_next(p, rng)`. `p` is a vector of chances, one for each id, and `rng` a random generator.
    It gives back one id, drawn with those chances. With `p = [0.4, 0.6]`, about 400 of 1,000 calls should give 0.

    The cell after the check starts from `once` and writes 30 words.
    """)
    return


@app.function
def sample_next(p, rng):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "sample_next", sample_next, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""It's one line: a `return` with `rng.choice`. The Toolbox example drew among 3 numbers. Here the number of choices isn't fixed: it's however many chances `p` holds.""",
        r"""`len(p)` is the number of choices, and `p` itself goes after `p=`. Don't use `np.argmax`: that always takes the likeliest id.""",
    ], LOCALE)
    return


@app.cell
def _(P, index, vocab):
    rng = np.random.default_rng(0)
    written = [index["once"]]
    for _step in range(30):
        written.append(sample_next(P[written[-1]], rng))
    " ".join([vocab[i] for i in written])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Most neighbouring pairs of words are plausible, and the whole makes no sense. That's a model with a memory of one word.
    (`written[-1]` is the last item of the list. You may see `<end>` in the text: our little loop doesn't stop there, and just carries on into a new story.)

    ## A score: perplexity

    How good is a language model? We need one number, measured on text it has never seen. Go
    through the test stories and ask: what chance did the model give to the word that really came next?

    Multiplying all those chances gives the chance of the whole text, but with 159,000 small numbers
    that's too tiny for a computer to store.

    /// admonition | 🧰 Toolbox: logs turn multiplying into adding
    $\log(a \times b) = \log a + \log b$. Chapter 07 met this the other way round, as $e^{a + b} = e^a \times e^b$.
    So instead of multiplying many chances, we can add their logs, and no number gets too small to store.
    Dividing works the same way: $\log(a / b) = \log a - \log b$.
    ///

    Add the logs of the chances, divide by the number of words and flip the sign, and you have
    a quantity you already know:

    > mean surprise = $-$(mean of $\log p$), chapter 07's cross-entropy

    To make it easier to read, we undo the log with $e^{\square}$. That gives the **perplexity**:

    > $$\text{perplexity} = e^{\text{mean surprise}}$$

    Here's what that number means. Suppose a model knows nothing, and gives each of $N$ words the chance $1 / N$. Since $\log(1 / N) = \log 1 - \log N = -\log N$,
    every surprise is $\log N$, so the mean is $\log N$, and the perplexity is $e^{\log N} = N$. So
    **a perplexity of N means the model is as unsure as if it were choosing among N equally likely words.**
    A perfect model scores 1. Lower is better.

    ## By hand: a perplexity

    A model reads a text of three words. To the words that really came next, it gave the chances 0.5, 0.25 and 0.125.
    Here are the numbers you need:

    | $p$ | 0.5 | 0.25 | 0.125 |
    |---|---|---|---|
    | surprise, $-\log p$ | 0.69 | 1.39 | 2.08 |

    | $x$ | 0.69 | 1.39 | 2.08 | 4.16 |
    |---|---|---|---|---|
    | $e^x$ | 2 | 4 | 8 | 64 |

    Work out the total surprise, the mean surprise, and the perplexity.
    """)
    return


@app.cell(hide_code=True)
def _():
    perplexity_boxes = hand.boxes(['total surprise', 'mean surprise', 'perplexity'], step=0.01)
    return (perplexity_boxes,)


@app.cell(hide_code=True)
def _(perplexity_boxes):
    hand.marked(perplexity_boxes, {'total surprise': 4.16, 'mean surprise': 1.39, 'perplexity': 4}, {
        'total surprise': r"""add the three surprises from the first table""",
        'mean surprise': r"""the total divided by the 3 words""",
        'perplexity': r"""$e$ to the mean surprise: look the mean up in the second table""",
    }, LOCALE, tolerance=0.011)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The model was as unsure as someone choosing among 4 words: sure of one word in 2 at best, one in 8 at worst.
    Note that the answer is $e$ to the *mean*, 4, not $e$ to the total, 64.

    ## Exercise 4: perplexity

    Write `perplexity(log_probs)`. `log_probs` is a NumPy array holding $\log p$ for each real next word: the logs are
    already taken, and they're negative or 0. It gives back one number, by the two formulas above. `np.exp(x)` is $e^x$.

    > `perplexity(np.log(np.array([0.5, 0.25, 0.125])))` gives `4.0`
    """)
    return


@app.function
def perplexity(log_probs):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "perplexity", perplexity, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three tools, one inside the other: `np.mean` of the logs, a minus sign to turn that into the mean surprise, and `np.exp` around it all.""",
        r"""It's one line, and it starts `return np.exp(`. If your result is below 1, the minus sign is missing.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell scores three models on the test stories, and the bigram again on its training text.

    /// admonition | 🧰 Toolbox: filling and picking
    - `np.full(n, x)` is an array of `n` copies of `x`: `np.full(3, 0.5)` is `[0.5, 0.5, 0.5]`.
    - `items[:-1]` is everything but the last item, the mirror image of `items[1:]`.
    - A table can be indexed with two arrays of positions at once, one for the rows and one for the columns. They pick
      one entry per pair: `P[[0, 2], [1, 3]]` gives `P[0, 1]` and `P[2, 3]`.
    ///

    - **Know-nothing:** $1 / 2002$ for every word.
    - **Unigram:** just how common each word is, ignoring what came before. The total of row `a` of `C`
      counts how many times `a` was followed by something, which is how often `a` appears, so the row totals give what we need.
    - **Bigram:** `P[test_ids[:-1], test_ids[1:]]` looks up one chance for each neighbouring pair of the test text.
    """)
    return


@app.cell
def _(C, P, test_ids, train_ids, vocab):
    know_nothing = perplexity(np.full(len(test_ids), np.log(1 / len(vocab))))
    unigram_p = np.sum(C, axis=1) / np.sum(C)
    unigram = perplexity(np.log(unigram_p[test_ids]))
    bigram = perplexity(np.log(P[test_ids[:-1], test_ids[1:]]))
    bigram_on_train = perplexity(np.log(P[train_ids[:-1], train_ids[1:]]))
    [round(score.item(), 1) for score in (know_nothing, unigram, bigram, bigram_on_train)]
    return bigram, know_nothing, unigram


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The first number is 2,002, as the reasoning says. Knowing how common words are cuts it to about 282. Knowing one previous word cuts it to about 54: the
    model is as unsure as choosing among 54 words. The last number is the bigram scored on its own training text: about 42. As in chapter 09, a model
    usually looks better on what it has seen, which is why we measure on the test stories.

    **How much smoothing?** `k` is a choice we make, not a knob trained by descent. Such a choice is
    called a **hyperparameter**. Here's the bigram's test score for four values:
    """)
    return


@app.cell
def _(C, test_ids):
    smoothing_scores = []
    for k_try in [1, 0.1, 0.01, 0.001]:
        P_try = bigram_probs(C, k_try)
        smoothing_scores.append((k_try, round(perplexity(np.log(P_try[test_ids[:-1], test_ids[1:]])).item(), 1)))
    smoothing_scores
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Too much smoothing spreads chance onto nonsense pairs; too little leaves the model
    overconfident about what it saw. 0.01 is about right.

    ## Looking two words back

    One word of memory isn't much. A **trigram** model looks at the two previous words, $P(\text{next} \mid a, b)$.
    A table for that would need $2{,}002 \times 2{,}002 \times 2{,}002$ entries, about 8 billion, but almost all of them would be 0. So
    we use a dictionary instead, storing only what we actually saw: each pair `(a, b)` is a key, and its value
    is a smaller dictionary counting what came next.

    /// admonition | 🧰 Toolbox: more dictionary tools
    - A key can be a pair in round brackets: `seen[("the", "dog")] = 2`.
    - `key not in seen` is the opposite of `in`: true when the dictionary doesn't have that key.
    - `seen.values()` goes through a dictionary's values, and `sum(...)` adds up a run of numbers: for
      `ages = {"ann": 30, "bo": 12}`, `sum(ages.values())` is 42.
    - `array.tolist()` turns a NumPy array into a plain list, which is faster to loop over and to use in dictionary keys.
    ///

    In the function below, `followers` isn't a copy: it's the same smaller
    dictionary that sits inside `table`, so changing it changes the table. And `zip` can pair three lists as easily as two.
    """)
    return


@app.function
def trigram_counts(ids):
    table = {}
    for a, b, c in zip(ids, ids[1:], ids[2:]):
        if (a, b) not in table:
            table[(a, b)] = {}
        followers = table[(a, b)]
        followers[c] = followers.get(c, 0) + 1
    return table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For the ids `[0, 1, 2, 0, 1, 3]`, the pair `(0, 1)` appears twice, followed once by 2 and once by 3. The cell shows the whole table:
    """)
    return


@app.cell
def _():
    trigram_counts([0, 1, 2, 0, 1, 3])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 5: trigram_chance

    Write `trigram_chance(table, a, b, c)`. `table` is what `trigram_counts` gives back, and `a`, `b`, `c` are three ids. It gives back
    one number: of all the times the pair `(a, b)` was seen, the share that were followed by `c`.
    A pair that was never seen gives `0.0`, and so does a `c` that never followed the pair. With `table = {(0, 1): {2: 3, 3: 1}}`:

    > `trigram_chance(table, 0, 1, 2)` gives `0.75`, and `trigram_chance(table, 5, 5, 2)` gives `0.0`

    The pair `(0, 1)` was seen $3 + 1 = 4$ times, and 3 of those were followed by 2.
    """)
    return


@app.function
def trigram_chance(table, a, b, c):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("13", "trigram_chance", trigram_chance, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Deal with the unseen pair first: an `if` with `not in`, and a `return` inside it. After that, the pair is known, so take its smaller dictionary out of `table`. The answer is one count from it, divided by the sum of all its counts.""",
        r"""It's 4 lines. The count of `c` comes from `.get` with 0 as the fallback, since `c` may not be a key. The total is `sum` of the smaller dictionary's `.values()`. A `KeyError` means the first check is missing.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Trigrams have the zero problem much worse: many pairs in the test stories were never seen. So
    we **mix**: half the chance comes from the trigram, half from the smoothed bigram.
    When the trigram knows nothing, the bigram still has an answer. The loop below takes a few seconds:
    """)
    return


@app.cell
def _(P, test_ids, train_ids):
    table = trigram_counts(train_ids.tolist())
    mixed = []
    for a, b, c in zip(test_ids.tolist(), test_ids.tolist()[1:], test_ids.tolist()[2:]):
        mixed.append(0.5 * trigram_chance(table, a, b, c) + 0.5 * P[b, c])
    trigram = perplexity(np.log(np.array(mixed)))
    round(trigram.item(), 1)
    return (trigram,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 38: another clear step down.

    ## The leaderboard

    From now on, every language model in this course gets scored the same way: perplexity per
    word, on the same 1,000 test stories. Your scores are saved, so this table will grow chapter by chapter, with the best model at the top:
    """)
    return


@app.cell(hide_code=True)
def _(bigram, know_nothing, trigram, unigram):
    record("uniform", know_nothing)
    record("unigram", unigram)
    record("bigram", bigram)
    record("trigram", trigram)
    show(LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What counting can't do

    Why not just count further back, with four words, or ten? Because the table runs out of evidence: most
    ten-word sequences have never been seen even once. And there's a deeper weakness. To a counting model, `the dog ran` teaches nothing
    about `the cat ran`. Each word is only a row number. But chapter 12 gave us vectors in which `dog` and
    `cat` are close, so a model built on vectors could share what it learns between similar words.

    ## What's next

    In **Chapter 14** you'll build a network that reads text one word at a time, keeping a running
    memory of what it has read. It's called a **recurrent network**, and it will take the top of this leaderboard.

    ## What you can do now

    - explain what a language model does, and read $P(\text{next} \mid \text{previous})$
    - count bigrams and trigrams, and turn counts into chances
    - explain why zero chances are a problem, and fix them with smoothing or mixing
    - generate text by sampling
    - compute perplexity, by hand and in code, and say what it means
    - name a hyperparameter, and explain why test data shouldn't choose it
    - explain why counting stops working, and what vectors would add

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** models that look back a fixed number of words are called *n-gram* models.
    Our mixing is *linear interpolation*, and a third set of data for choosing hyperparameters is called a
    *validation set*: choosing `k` with the test stories, as we did, lets a little test knowledge leak in, though with one small number the leak is negligible. Perplexity is `e` to the *cross-entropy*, or 2 to it if you measure in bits.
    """)
    return


if __name__ == "__main__":
    app.run()
