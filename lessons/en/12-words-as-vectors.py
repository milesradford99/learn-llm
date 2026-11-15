import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="12 · Words as vectors")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn

    from learn_llm import hand, laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.plot import chart

    LOCALE = "en"
    laptop_speed()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 12 · Words as vectors

    Part 2 of the course is about language. From here to the last chapter we'll use one body of
    text, a collection of short, simple stories called **TinyStories**, so that each new model can be
    compared with the ones before. It's bundled with the course, in English for every language
    edition of this course. Here's the start of the first of 12,000 stories. (Another 1,000,
    `test_stories`, are kept aside for chapter 13.)
    """)
    return


@app.cell
def _():
    train_stories, test_stories = stories()
    train_stories[0][:300]
    return (train_stories,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Networks work on numbers, so our first problem is how to turn a word into numbers. The obvious
    way is chapter 07's one-hot rows: with 2,000 known words, each word is a row of 2,000 numbers
    with a single 1. But that tells the network nothing. The dot product of any two
    different one-hot rows is 0, so `dog` is as unlike `cat` as it is unlike `yesterday`.

    We want something better: a short **word vector**, say 32 numbers, chosen so that words with
    similar meanings get similar vectors. Nobody could write those numbers by hand. But there's a
    trick to learn them from text alone, resting on one observation: **a word is known by the
    company it keeps.** Cover a word here and you can still guess what kind of word it is:

    > The ▢ barked at the mailman and wagged its tail.

    Words that appear among the same neighbours tend to mean similar things. So we'll give a
    network a job, predicting a word's neighbours, and keep the vectors it invents along the way. This is the
    **word2vec** method.

    ## From text to words

    First, cut the text into words. This helper makes everything lower-case and keeps runs of
    letters and apostrophes, dropping the rest. (`re` is Python's text-pattern tool. You don't need to
    learn it; `[a-z']+` just means "one or more letters or apostrophes".)

    /// admonition | 🧰 Toolbox: two text tools
    - `text.lower()` gives the text with every capital letter made small: `"The Dog".lower()` is `"the dog"`.
    - `" ".join(items)` glues a list of strings into one, with a space between each:
      `" ".join(["a", "b", "c"])` is `"a b c"`.
    ///
    """)
    return


@app.function
def to_words(text):
    return re.findall(r"[a-z']+", text.lower())


@app.cell
def _(train_stories):
    words = to_words(" ".join(train_stories[:6000]))
    len(words), words[:12]
    return (words,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    That's about 965,000 words from 6,000 stories, plenty for this chapter. A training run on 12,000 would
    only take twice as long.

    ## Counting with a dictionary

    We need to know how often each word appears. A list won't do: a list is looked up by position,
    and we want to look up by word. For that, Python has a **dictionary**: a value that stores pairs,
    each **key** with its **value**, and finds a key instantly.

    /// admonition | 🧰 Toolbox: dictionaries
    - `{}` makes an empty dictionary. `pets["dog"] = 3` stores 3 under the key `"dog"`, and replaces
      whatever was stored there before.
    - `pets["dog"]` reads it back. Reading a key that was never stored stops with an error.
    - `pets.get("fish", 0)` reads too, but gives the second value, here 0, when the key isn't there.
    - `for name in pets:` visits the keys, in the order they were first stored.
    ///

    The cell below stores 3 dogs, then adds 2 cats twice. The first time there's no `"cat"` key
    yet, so `.get` gives 0 and the cats start from there.
    """)
    return


@app.cell
def _():
    pets = {}
    pets["dog"] = 3
    pets["cat"] = pets.get("cat", 0) + 2
    pets["cat"] = pets.get("cat", 0) + 2
    pets, pets["dog"], pets.get("fish", 0)
    return (pets,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: count_words

    Write `count_words(words)`. It takes a list of words and gives back a dictionary from each
    different word to how many times it appears:

    > `count_words(["the", "dog", "the"])` gives `{"the": 2, "dog": 1}`

    Start from an empty dictionary and visit the words one by one. Each word raises its own count
    by 1. Think about what the count is the first time a word shows up.
    """)
    return


@app.function
def count_words(words):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "count_words", count_words, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Three parts: make the empty dictionary, loop over `words`, give the dictionary back after the loop. The cats in the Toolbox cell show how to raise a count that may not exist yet.""",
        r"""It's 4 lines. The first is `counts = {}`. Inside the loop there's one line, which stores under the key `word` one more than `.get` finds there.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## A vocabulary

    Now we keep the 2,000 most common words as our **vocabulary**, and give each a number, its
    **id**: its position in the list. Two more tools do that.

    /// admonition | 🧰 Toolbox: ordering and filtering
    - `sorted(pets)` gives the keys as a list in alphabetical order. `sorted(pets, key=pets.get)`
      orders them by what `pets.get` gives for each, so by their stored values, smallest first.
      Adding `reverse=True` puts the biggest first.
    - `[name for name in names if name in pets]` builds a list as in chapter 05, but keeps only the
      items that pass the test after `if`. `name in pets` is true when the dictionary has that key.
    ///
    """)
    return


@app.cell
def _(pets):
    sorted(pets, key=pets.get, reverse=True), [name for name in ["dog", "owl", "cat"] if name in pets]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The next cell uses your `count_words`, so it shows an error until Exercise 1 passes. That's
    normal, and the same goes for every cell that needs it further down. It builds three things:

    - `vocab`, the 2,000 most common words, most common first;
    - `index`, a dictionary the other way round, from word to id;
    - `ids`, the whole text as ids, skipping words outside the vocabulary.

    It shows the number of different words, the first 8 words of `vocab`, and the length of `ids`.
    """)
    return


@app.cell
def _(words):
    counts = count_words(words)
    vocab = sorted(counts, key=counts.get, reverse=True)[:2000]
    index = {}
    for i in range(len(vocab)):
        index[vocab[i]] = i
    ids = [index[word] for word in words if word in index]
    len(counts), vocab[:8], len(ids)
    return counts, ids, index, vocab


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    About 6,800 different words appear, and those 2,000 cover 97% of the text. Small stories have a
    small vocabulary, which is why they suit a laptop.

    ## How alike are two vectors?

    Chapter 05's dot product multiplies position by position and adds. It's large when two vectors
    point the same way. But it also grows with their lengths, and for similarity we only care about
    direction. So we divide by both lengths.

    /// admonition | 🧰 Toolbox: the length of a vector
    Writing $\cdot$ for the dot product (`@` in code), the **length** of a vector $a$ is $\sqrt{a \cdot a}$:
    the square root of the sum of its squares. For $[3, 4]$ that's $\sqrt{9 + 16} = 5$, the length of the
    arrow from $(0, 0)$ to the point $(3, 4)$ on a chart.
    ///

    This measure is called **cosine similarity**:

    > $$\operatorname{similarity}(a, b) = \frac{a \cdot b}{\text{length of } a \times \text{length of } b}$$

    It's 1 if they point the same way, 0 if they're at right angles (nothing in common), and −1 if
    they point in opposite directions. With $a = [3, 4]$ and $b = [6, 8]$, the dot product is $3 \times 6 + 4 \times 8 = 50$.
    The lengths are $\sqrt{9 + 16} = 5$ and $\sqrt{36 + 64} = 10$. So the
    similarity is $50 / (5 \times 10) = 1$: $b$ is $a$ made twice as long, so the direction is the same.

    ## By hand: two similarities

    First $a = [3, 4]$ and $b = [4, 3]$: the same two numbers, swapped. Work out the dot product,
    the two lengths, and the similarity. Then the similarity of $c = [0, 2]$ and $d = [3, 4]$.
    """)
    return


@app.cell(hide_code=True)
def _():
    cosine_boxes = hand.boxes(['a · b', 'length of a', 'length of b', 'similarity of a and b', 'similarity of c and d'], step=0.01)
    return (cosine_boxes,)


@app.cell(hide_code=True)
def _(cosine_boxes):
    hand.marked(cosine_boxes, {'a · b': 24, 'length of a': 5, 'length of b': 5, 'similarity of a and b': 0.96, 'similarity of c and d': 0.8}, {
        'a · b': r"""$3 \times 4 + 4 \times 3$""",
        'length of a': r"""$\sqrt{3 \times 3 + 4 \times 4}$""",
        'length of b': r"""swapping the numbers doesn't change the sum of the squares""",
        'similarity of a and b': r"""the dot product, divided by the two lengths multiplied together""",
        'similarity of c and d': r"""the dot product is $0 \times 3 + 2 \times 4$, and the lengths are 2 and 5""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Swapped numbers point in a nearby direction, but not the same one: 0.96, not 1.

    ## Exercise 2: similarity

    Write `similarity(a, b)` for two NumPy vectors of the same size. It gives back one number, by the
    formula above. `np.sqrt` is the square root (chapter 09).

    > `similarity(np.array([3.0, 4.0]), np.array([4.0, 3.0]))` gives `0.96`
    """)
    return


@app.function
def similarity(a, b):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "similarity", similarity, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""You need three numbers, each from one `@`: $a \cdot b$, $a \cdot a$ and $b \cdot b$. The last two go through `np.sqrt` to become lengths.""",
        r"""It can be 3 lines: one for each length, then the `return`. The first is `length_a = np.sqrt(a @ a)`. Put brackets around the two lengths multiplied together, or Python divides by the first and then multiplies by the second.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The job: guess the neighbours

    The training examples are pairs of ids. Slide along the text; each word becomes a **centre**,
    and every word within `window` positions on either side is one of its **contexts**. For the text
    `the dog ran fast` with `window = 1`:

    | centre | contexts | pairs |
    |---|---|---|
    | `the` | `dog` | (the, dog) |
    | `dog` | `the`, `ran` | (dog, the), (dog, ran) |
    | `ran` | `dog`, `fast` | (ran, dog), (ran, fast) |
    | `fast` | `ran` | (fast, ran) |

    The words at the two ends have fewer contexts, because the window hangs over the edge of the text.
    Predicting the contexts from a centre is called the **skip-gram** task.

    ## By hand: count the pairs

    Take a text of 5 words and `window = 2`, so each centre looks up to 2 positions left and up to 2
    right. How many contexts does the first word have? The middle word? And how many pairs does the
    whole text give?
    """)
    return


@app.cell(hide_code=True)
def _():
    pair_boxes = hand.boxes(['contexts of the first word', 'contexts of the middle word', 'pairs in total'])
    return (pair_boxes,)


@app.cell(hide_code=True)
def _(pair_boxes):
    hand.marked(pair_boxes, {'contexts of the first word': 2, 'contexts of the middle word': 4, 'pairs in total': 14}, {
        'contexts of the first word': r"""nothing is to its left, so only the words to its right count""",
        'contexts of the middle word': r"""it has 2 words on each side""",
        'pairs in total': r"""the five words have 2, 3, 4, 3 and 2 contexts""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | 🧰 Toolbox: staying inside a list
    - `max(a, b)` gives the larger of two numbers and `min(a, b)` the smaller. They stop a position
      from leaving a list of 10 items: `max(0, 1 - 3)` is 0, not −2, and `min(10, 8 + 3)` is 10, not 11.
    - `range(start, stop)` counts from `start` up to, but not including, `stop`: `range(2, 5)` gives 2, 3, 4.
      To include position 5, write `range(2, 5 + 1)`.
    - `a != b` means "a is not equal to b", the opposite of `==`.
    ///

    ## Exercise 3: skipgram_pairs

    Write `skipgram_pairs(ids, window)`. `ids` is a list of ids and `window` a whole number. It gives
    back two lists of the same length, the centres and the contexts, so that position $p$ of each
    list together make one pair. The centres come in text order, and each centre's contexts from left to right.
    For the table above, with the four words as ids 0 to 3:

    > `skipgram_pairs([0, 1, 2, 3], 1)` gives `([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])`

    For the centre at position $i$, the contexts sit at the positions from $i - \text{window}$ to
    $i + \text{window}$ that are inside the list, leaving out $i$ itself.
    """)
    return


@app.function
def skipgram_pairs(ids, window):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "skipgram_pairs", skipgram_pairs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Two empty lists, then two loops, one inside the other: `i` over every position, and `j` over the positions near `i`. Inside both, an `if` leaves out `j` equal to `i`, and two `append` lines add one item to each list.""",
        r"""It's 8 lines. The inner loop is a `range` whose start is `i - window` held at 0 or above with `max`, and whose stop is `i + window + 1` held at `len(ids)` or below with `min`. The lists get `ids[i]` and `ids[j]`, the ids, not the positions.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Real or made up? A cheaper question

    Each word gets two vectors, one for when it's a centre ($u$) and one for when it's a context ($v$).
    The word's meaning will end up in $u$. We'd like $u \cdot v$ to be high when $v$ is a real neighbour.

    The straightforward model is chapter 07's: the scores of all 2,000 words, then softmax. But
    that means 2,000 dot products for every pair, millions of times. Instead, we'll ask a much
    cheaper question: *is this pair real?* For each real pair, we also draw a few **noise** words at
    random, and train the network to tell the real context from the fakes. This is **negative
    sampling**.

    ## A probability from one score

    A yes-or-no question needs a probability from one score $s$. Take softmax over two scores, $s$
    for "real" and 0 for "fake". (Why 0? Softmax only cares about differences between scores, as chapter 07's shift
    showed, so one of them may as well be 0.) Anything to the power 0 is 1, so $e^0 = 1$, and the share of "real" is

    > $$\sigma(s) = \frac{e^s}{e^s + e^0} = \frac{e^s}{e^s + 1}$$

    This is called the **sigmoid**, written with the Greek letter $\sigma$ (sigma). Here it is for scores from −6 to 6:
    """)
    return


@app.cell(hide_code=True)
def _():
    chart((-6, 6), (0, 1), "score s", "σ(s)",
          curve=[(x / 10, float(1 / (1 + np.exp(-x / 10)))) for x in range(-60, 61)])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    It's near 1 for a big $s$, exactly 0.5 for $s = 0$, and near 0 for a very negative $s$.
    The curve also has a symmetry: the share of "fake" is $1 - \sigma(s)$,
    which equals $\sigma(-s)$. On the chart, $\sigma(2)$ is as far below 1 as $\sigma(-2)$ is above 0.
    (Check it: $\sigma(-s) = e^{-s} / (e^{-s} + 1)$. Multiply top and bottom by $e^s$, remembering that
    $e^{-s} \times e^s = e^0 = 1$, and you get $1 / (1 + e^s)$, which is exactly $1 - e^s / (e^s + 1)$.)

    ## The loss for one pair

    The loss is chapter 07's surprise, $-\log$ of the share given to the right answer, added up
    for one real word and $k$ noise words. For the real word, the right answer is "real", with share
    $\sigma(u \cdot v)$. For a noise word $n$, the right answer is "fake", with share $\sigma(-u \cdot n)$:

    > $$\text{loss for one pair} = -\log \sigma(u \cdot v) - \log \sigma(-u \cdot n_1) - \ldots - \log \sigma(-u \cdot n_k)$$

    The first term pulls $u$ towards its real neighbour. The others push it away from the fakes.
    Here's the surprise $-\log \sigma(s)$ for a few scores:

    | $s$ | −2 | −1 | 0 | 1 | 2 |
    |---|---|---|---|---|---|
    | $\sigma(s)$ | 0.12 | 0.27 | 0.50 | 0.73 | 0.88 |
    | $-\log \sigma(s)$ | 2.13 | 1.31 | 0.69 | 0.31 | 0.13 |

    ## By hand: one pair's loss

    Take $u = [1, 2]$, a real context $v = [2, 0]$ and one noise word $n = [1, -1]$. Work out the two
    scores, $u \cdot v$ and $u \cdot n$, then the loss, reading the surprises from the table. Mind the
    minus sign in front of the noise score.

    Then suppose the noise word were $[2, 0]$, the same as the real context. What's the loss now?
    """)
    return


@app.cell(hide_code=True)
def _():
    loss_boxes = hand.boxes(['real score', 'noise score', 'loss', 'loss with the other noise word'], step=0.01)
    return (loss_boxes,)


@app.cell(hide_code=True)
def _(loss_boxes):
    hand.marked(loss_boxes, {'real score': 2, 'noise score': -1, 'loss': 0.44, 'loss with the other noise word': 2.26}, {
        'real score': r"""$1 \times 2 + 2 \times 0$""",
        'noise score': r"""$1 \times 1 + 2 \times (-1)$""",
        'loss': r"""the real word's surprise is the table's entry for $s = 2$. The noise word's is the entry for minus its score, so for $s = 1$. Add the two.""",
        'loss with the other noise word': r"""now the noise score is 2, so its surprise is the table's entry for $s = -2$""",
    }, LOCALE, tolerance=0.011)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The second loss is five times the first. A noise word that scores as high as a real neighbour is
    what the loss punishes, and training lowers it by pushing $u$ away from that word.

    ## A whole batch at once

    The exercise computes this loss for many pairs in one go, so the vectors arrive stacked:
    `u` and `v` have shape `(b, d)`, for `b` pairs and vectors of `d` numbers, and `n` has shape
    `(b, k, d)`, `k` noise vectors for each pair. (Here `n` means noise, not a number of examples as in earlier chapters.)

    /// admonition | 🧰 Toolbox: dot products in bulk
    - `torch.sum(x * y, dim=1)` for two tensors of shape `(b, d)` multiplies position by position, then adds
      along each row: one dot product per row, shape `(b,)`.
    - `x.unsqueeze(1)` (chapter 11) puts in a dimension of size 1: `(b, d)` becomes `(b, 1, d)`. In arithmetic, a
      dimension of size 1 is repeated to match the other side, so `(b, 1, d)` times `(b, k, d)` works, and
      gives `(b, k, d)`: row 0 of `x` meets each of the `k` vectors that belong to row 0.
    - `F.logsigmoid(s)` computes $\log \sigma(s)$ for every number of a tensor, safely for big scores, in the spirit of chapter 07's softmax shift.
    ///

    The cell below tries the first two on 2 rows of 3 numbers and a stack of shape `(2, 4, 3)`, and shows three shapes.
    """)
    return


@app.cell
def _():
    rows_demo = torch.ones(2, 3)
    stack_demo = torch.ones(2, 4, 3)
    product_demo = stack_demo * rows_demo.unsqueeze(1)
    torch.sum(rows_demo * rows_demo, dim=1).shape, product_demo.shape, torch.sum(product_demo, dim=2).shape
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 4: pair_loss

    Write `pair_loss(u, v, n)` with the shapes above. It gives back one number, a tensor with no dimensions:
    the mean, over the `b` pairs, of the loss for one pair. On the way there:

    | what | shape |
    |---|---|
    | the real scores $u \cdot v$, one per pair | `(b,)` |
    | the noise scores $u \cdot n_j$, `k` per pair | `(b, k)` |
    | the loss of each pair, by the formula | `(b,)` |
    | their mean | one number |

    Your numbers from the hand exercise are one of the checks: a batch of 1 pair with
    `u = [[1, 2]]`, `v = [[2, 0]]` and `n = [[[1, -1]]]` gives about `0.44`.
    """)
    return


@app.function
def pair_loss(u, v, n):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "pair_loss", pair_loss, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    hand.stuck([
        r"""Four lines, one for each row of the table. The Toolbox cell has the pattern for the first two rows. For the third, take `F.logsigmoid` of the real scores and of minus the noise scores, add up each pair's `k` noise terms along `dim=1`, and put a minus in front of both parts.""",
        r"""The first line is `real = torch.sum(u * v, dim=1)`. For the noise scores, `u` needs `unsqueeze(1)` before it can multiply `n`, and the sum runs along `dim=2`. The last line gives back `torch.mean` of the losses.""",
    ], LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Training

    One last PyTorch piece: `nn.Embedding(2000, 32)` is a table of 2,000 rows of 32 knobs. Given a tensor of
    ids of any shape, it looks up one row for each, so `(b,)` ids give `(b, 32)` vectors, and `(b, k)`
    ids give `(b, k, 32)`. That's all there is to it: a table lookup, whose numbers learn.

    /// admonition | 🧰 Toolbox: powers below 1
    `x ** 0.5` is the square root of `x`, and other powers below 1 behave alike: they pull big numbers
    down more than small ones. `100 ** 0.75` is about 32, and `10000 ** 0.75` is 1,000. So two counts
    that are 100 times apart end up only about 32 times apart.
    ///

    Noise words are drawn from a list of chances, `noise`: each word's count raised to the power
    0.75, then scaled to add up to 1. That way common words get a little less
    dominant, and rarer words get to be noise too. `torch.multinomial(noise, m, replacement=True)`
    draws `m` ids with those chances.

    `train_vectors` is `train` from chapter 10 with `pair_loss` in place of cross-entropy. It reads like this: two
    tables, Adam over both, and for each batch, look up `u`, `v` and `n`, then clear, backward, step.
    It gives back the centre table as a NumPy array, one row per word, and we drop the context table.
    `nn.init.normal_` gives the tables a small random start.
    """)
    return


@app.function
def train_vectors(centres, contexts, noise, size, epochs, batch_size, rate):
    centre_table = nn.Embedding(len(noise), size)
    context_table = nn.Embedding(len(noise), size)
    nn.init.normal_(centre_table.weight, 0, 0.1)
    nn.init.normal_(context_table.weight, 0, 0.1)
    optimizer = torch.optim.Adam([centre_table.weight, context_table.weight], lr=rate)
    for epoch in range(epochs):
        order = torch.randperm(len(centres))
        for start in range(0, len(centres), batch_size):
            batch = order[start:start + batch_size]
            noise_ids = torch.multinomial(noise, 5 * len(batch), replacement=True)
            n = context_table(noise_ids.reshape(len(batch), 5))
            optimizer.zero_grad()
            loss = pair_loss(centre_table(centres[batch]), context_table(contexts[batch]), n)
            loss.backward()
            optimizer.step()
    return centre_table.weight.detach().numpy()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hand: read the training loop

    Find the two lines of `train_vectors` that make `noise_ids` and `n`. We'll call it with `size = 32`
    and batches of 1,024 pairs. For one full batch: how many noise ids are drawn? After the `reshape`,
    how many belong to each pair? And how many numbers does each noise vector in `n` have?
    """)
    return


@app.cell(hide_code=True)
def _():
    loop_boxes = hand.boxes(['noise ids drawn', 'noise words per pair', 'numbers per noise vector'])
    return (loop_boxes,)


@app.cell(hide_code=True)
def _(loop_boxes):
    hand.marked(loop_boxes, {'noise ids drawn': 5120, 'noise words per pair': 5, 'numbers per noise vector': 32}, {
        'noise ids drawn': r"""the second input of `torch.multinomial` is how many to draw: 5 times the batch""",
        'noise words per pair': r"""the ids are reshaped to `(len(batch), 5)`: one row per pair""",
        'numbers per noise vector': r"""`context_table` has `size` numbers in each row""",
    }, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    So `n` has shape `(1024, 5, 32)`: the `(b, k, d)` that your `pair_loss` expects, with $k = 5$.

    The button builds the pairs (about 3.7 million, with `window = 2`), then trains for 2 epochs:
    half a minute or so on a laptop. Until it finishes, the cells that need its result say `Ancestor stopped`. That's normal.
    When it's done, you'll see `(2000, 32)`: 2,000 words, 32 numbers each.
    """)
    return


@app.cell(hide_code=True)
def _():
    train_button = mo.ui.run_button(label="Train the word vectors")
    train_button
    return (train_button,)


@app.cell
def _(counts, ids, train_button, vocab):
    mo.stop(not train_button.value, mo.md("*Press the button to train.*"))
    centres, contexts = skipgram_pairs(ids, 2)
    freq = torch.tensor([counts[word] for word in vocab], dtype=torch.float32)
    noise = freq ** 0.75 / torch.sum(freq ** 0.75)
    torch.manual_seed(0)
    E = train_vectors(torch.tensor(centres), torch.tensor(contexts), noise, 32, 2, 1024, 0.003)
    E.shape
    return (E,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What did it learn?

    Nobody told the network what any word means, and it never saw a dictionary. It only got pairs of
    neighbours. Now look for the words whose vectors are the most similar to a given word's.

    /// admonition | 🧰 Toolbox: positions in sorted order
    `np.argsort(a)` gives the positions of an array's numbers, from the smallest number to the largest:
    `np.argsort(np.array([0.5, 0.1, 0.9]))` is `[1, 0, 2]`. To get the largest first, sort the negated array.
    ///

    `nearest` scales every row to length 1. With both lengths at 1, the division in the similarity formula
    goes away, so `@` alone gives the similarity with every word at once. Then it sorts
    `-sims` to get the biggest first. The first hit is always the word itself, so we skip it.
    Change the word in the box and press Enter:
    """)
    return


@app.function
def nearest(query, E, vocab, count):
    unit = E / np.sqrt(np.sum(E * E, axis=1, keepdims=True))
    sims = unit @ (query / np.sqrt(query @ query))
    return [vocab[i] for i in np.argsort(-sims)[:count]]


@app.cell(hide_code=True)
def _():
    word_box = mo.ui.text(value="dog", label="Word:")
    word_box
    return (word_box,)


@app.cell
def _(E, index, vocab, word_box):
    asked = word_box.value.strip().lower()
    nearest(E[index[asked]], E, vocab, 6)[1:] if asked in index else "That word isn't in the vocabulary."
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now a table for a few words:
    """)
    return


@app.cell(hide_code=True)
def _(E, index, vocab):
    table = ["| word | closest words |", "|---|---|"]
    for word in ["dog", "happy", "mom", "red", "ran"]:
        table.append(f"| {word} | " + ", ".join(nearest(E[index[word]], E, vocab, 6)[1:]) + " |")
    mo.md("\n".join(table))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `dog` sits among animals, `happy` among feelings and states, `mom` among family, `red` among colours and `ran` among
    ways of moving. (Your lists may differ a little from run to run.) All of it came from nothing but who stands next to whom.

    Don't expect it to be perfect. Only these 32 numbers and a million words stand behind it, so some
    results are odd, and a re-run gives somewhat different lists. It also learns only what's in
    the text. Vectors trained on children's stories know about dogs and moms, not about chemistry.
    And any habits of the text, including its prejudices about who does what, end up in the vectors
    too.

    ## What's next

    A word vector captures what a word means, but so far nothing uses the *order* of the words. In **Chapter 13**
    you'll build your first true language model, one that predicts the next word, and generates
    text. To measure it you'll meet **perplexity**, and start the **leaderboard**: a single score on
    the same held-out stories, which every model from there to the end of the course will try to
    beat.

    ## What you can do now

    - explain why one-hot rows carry no meaning, and what a word vector adds
    - use a dictionary to count, and turn text into ids
    - measure how alike two vectors are with cosine similarity
    - build skip-gram pairs from text
    - derive the sigmoid from softmax, and work out a negative-sampling loss by hand
    - use `nn.Embedding` as a table of learned vectors
    - explore nearest words, and say what their limits are

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the method is *word2vec* (Mikolov et al., 2013). Real runs also throw away
    some of the most common words, such as `the`, and *GloVe* gets similar vectors by counting
    pairs directly. The idea that a word's meaning shows in its neighbours is called
    the *distributional hypothesis*. The stories are from *TinyStories* (Eldan and Li, 2023).

    Differences between vectors capture relationships too. If the step from `boy` to `girl` is roughly
    a "make it female" direction, the same step from `he` should land near `she`. Try it in a new cell:
    `nearest(E[index["girl"]] - E[index["boy"]] + E[index["he"]], E, vocab, 6)`. The three words you
    put in tend to come back among the answers, so look past them. With only 32 numbers and a
    million words, some of these **analogies** work and many don't.
    """)
    return


if __name__ == "__main__":
    app.run()
