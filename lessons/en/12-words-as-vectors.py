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

    from learn_llm import laptop_speed, todo
    from learn_llm.checks import check
    from learn_llm.data import stories

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
    way is chapter 06's one-hot rows: with 2,000 known words, each word is a row of 2,000 numbers
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
    (`" ".join(list)` glues a list of strings into one, with a space between each. That's about
    965,000 words from 6,000 stories, plenty for this chapter. A training run on 12,000 would
    only take twice as long.)

    ## Counting with a dictionary

    We need to know how often each word appears. For that, Python has a **dictionary**: a value
    that stores pairs, each **key** with its **value**, and finds a key instantly. A list is looked up
    by position; a dictionary is looked up by key.

    - `{}` makes an empty dictionary; `counts["dog"] = 3` stores 3 under the key `"dog"`.
    - `counts["dog"]` reads it back.
    - `counts.get("dog", 0)` reads it too, but gives 0 if the key isn't there yet, instead of
      stopping with an error.
    """)
    return


@app.cell
def _():
    pets = {}
    pets["dog"] = 3
    pets["cat"] = pets.get("cat", 0) + 1
    pets, pets["dog"], pets.get("fish", 0)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exercise 1: count_words

    Write `count_words(words)`, giving back a dictionary from each word to how many times it appears:

    1. `counts = {}`
    2. `for word in words:`, and inside it `counts[word] = counts.get(word, 0) + 1`
    3. After the loop, `return counts`
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
    mo.md(r"""
    Now we keep the 2,000 most common words as our **vocabulary**, and give each a number, its
    **id**: its position in the list.

    - `sorted(counts, key=counts.get, reverse=True)` lists the keys, ordered by their counts, biggest first.
    - `index` is a dictionary the other way round: word to id.
    - `ids` is the whole text as ids. The `if word in index` at the end of the list-building
      expression skips words outside the vocabulary.
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
    direction. So we divide by both lengths. Writing $\cdot$ for the dot product (`@` in code), the
    **length** of a vector is $\sqrt{a \cdot a}$, the square root of the sum of its squares. This measure is called **cosine similarity**:

    > $$\operatorname{similarity}(a, b) = \frac{a \cdot b}{\text{length of } a \times \text{length of } b}$$

    It's 1 if they point the same way, 0 if they're at right angles (nothing in common), and −1 if
    they point in opposite directions. With `a = [3, 4]` and `b = [6, 8]`, the dot product is $3 \times 6 + 4 \times 8 = 50$.
    The lengths are $\sqrt{9 + 16} = 5$ and $\sqrt{36 + 64} = 10$. So the
    similarity is $50 / (5 \times 10) = 1$: $b$ is $a$ made twice as long, so the direction is the same.

    ## Exercise 2: similarity

    Write `similarity(a, b)` for two NumPy vectors:

    1. `length_a = np.sqrt(a @ a)`
    2. `length_b = np.sqrt(b @ b)`
    3. `return (a @ b) / (length_a * length_b)`
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

    Predicting the contexts from a centre is called the **skip-gram** task.

    ## Exercise 3: skipgram_pairs

    Write `skipgram_pairs(ids, window)`, giving back two lists of the same length, the centres and the
    contexts. For each position `i`, the neighbours run from `i - window` to `i + window`,
    but must stay inside the list, and must not be `i` itself. `max(a, b)` gives the larger number and
    `min(a, b)` the smaller, which handles the edges. `!=` means "is not equal to". And
    `range` stops just before its last number, so the `+ 1` lets it reach `i + window`.

    1. Two lines: `centres = []` and `contexts = []`
    2. `for i in range(len(ids)):`
    3. Inside it: `for j in range(max(0, i - window), min(len(ids), i + window + 1)):`
    4. Inside that: `if j != i:`
    5. Inside that, two lines: `centres.append(ids[i])` and `contexts.append(ids[j])`
    6. After the loops, `return centres, contexts`
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
    mo.md(r"""
    ## Real or made up? A cheaper question

    Each word gets two vectors, one for when it's a centre ($u$) and one for when it's a context ($v$).
    The word's meaning will end up in $u$. We'd like $u \cdot v$ to be high when $v$ is a real neighbour.

    The straightforward model is chapter 06's: the scores of all 2,000 words, then softmax. But
    that means 2,000 dot products for every pair, millions of times. Instead, we'll ask a much
    cheaper question: *is this pair real?* For each real pair, we also draw a few **noise** words at
    random, and train the network to tell the real context from the fakes. This is **negative
    sampling**.

    A yes-or-no question needs a probability from one score $s$. Take softmax over two scores, $s$
    for "real" and 0 for "fake". (Why 0? Softmax only cares about differences between scores, as chapter 06's shift
    showed, so one of them may as well be 0.) Anything to the power 0 is 1, so $e^0 = 1$, and the share of "real" is

    > $$\sigma(s) = \frac{e^s}{e^s + e^0} = \frac{e^s}{e^s + 1}$$

    This is called the **sigmoid**. It's near 1 for a big $s$, exactly 0.5 for $s = 0$, and
    near 0 for a very negative $s$. And there's a handy fact: the share of "fake" is $1 - \sigma(s)$,
    which equals $\sigma(-s)$. (Check it: $\sigma(-s) = e^{-s} / (e^{-s} + 1)$. Multiply top and bottom by $e^s$, remembering that
    $e^{-s} \times e^s = e^0 = 1$, and you get $1 / (1 + e^s)$, which is exactly $1 - e^s / (e^s + 1)$.)

    The loss is chapter 06's surprise, $-\log$ of the share given to the right answer, added up
    for one real word and $k$ noise words. For the real word, the right answer is "real", with share
    $\sigma(u \cdot v)$. For a noise word $n$, the right answer is "fake", with share $\sigma(-u \cdot n)$:

    > $$\text{loss for one pair} = -\log \sigma(u \cdot v) - \log \sigma(-u \cdot n_1) - \ldots - \log \sigma(-u \cdot n_k)$$

    The first term pulls $u$ towards its real neighbour. The others push it away from the fakes. In
    PyTorch, `F.logsigmoid(s)` computes $\log \sigma(s)$ safely, in the spirit of chapter 06's softmax shift.

    ## Exercise 4: pair_loss

    Write `pair_loss(u, v, n)` for a whole batch at once. `u` and `v` have shape `(b, d)`, for `b` pairs and vectors of
    `d` numbers. `n` has shape `(b, k, d)`, `k` noise vectors for each pair. (Here `n` means noise,
    not a number of examples as in earlier chapters.)

    1. `real = torch.sum(u * v, dim=1)`, one dot product per pair, so shape `(b,)`
    2. `fake = torch.sum(n * u.unsqueeze(1), dim=2)`. `u.unsqueeze(1)` has shape `(b, 1, d)`, and a dimension of size 1
       is repeated to match, so it acts like `(b, k, d)`. This gives `k` dot products per pair, shape `(b, k)`
    3. `surprise = -F.logsigmoid(real) - torch.sum(F.logsigmoid(-fake), dim=1)`
    4. `return torch.mean(surprise)`
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
    mo.md(r"""
    ## Training

    One last PyTorch piece: `nn.Embedding(2000, 32)` is a table of 2,000 rows of 32 knobs. Given a tensor of
    ids of any shape, it looks up one row for each, so `(b,)` ids give `(b, 32)` vectors, and `(b, k)`
    ids give `(b, k, 32)`. That's all there is to it: a table lookup, whose numbers learn.

    Noise words are drawn from a list of chances, `noise`: each word's count raised to the power
    0.75, then scaled to add up to 1. A power below 1 pulls big numbers down more than small ones
    (`100 ** 0.75` is about 32, `10000 ** 0.75` is 1,000), so common words get a little less
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

    `nearest` scales every row to length 1. With both lengths at 1, the division in the similarity formula
    goes away, so `@` alone gives the similarity with every word at once.
    `np.argsort` gives the positions in order from smallest to largest, so we sort
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

    ## Arithmetic with meaning

    Differences between vectors capture relationships too. If the step from `boy` to `girl` is
    roughly a "make it female" direction, then the same step from `he` should land near `she`.
    The cell below computes $b - a + c$ and asks for the nearest words that aren't one of the three.
    (`not in` is the opposite of `in`.)
    """)
    return


@app.function
def analogy(a, b, c, E, index, vocab):
    query = E[index[b]] - E[index[a]] + E[index[c]]
    return [word for word in nearest(query, E, vocab, 6) if word not in (a, b, c)][:3]


@app.cell(hide_code=True)
def _(E, index, vocab):
    rows = ["| a is to b … | … as c is to | best guesses |", "|---|---|---|"]
    for a, b, c in [("boy", "girl", "he"), ("he", "she", "his"), ("boy", "girl", "brother"), ("go", "went", "see")]:
        rows.append(f"| {a} : {b} | {c} | " + ", ".join(analogy(a, b, c, E, index, vocab)) + " |")
    mo.md("\n".join(rows))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In the first three rows the top guesses are `she`, `her` and `sister`, and in the last row the past
    tense `saw` is second. (`lily's` counts as one word, because our cutter keeps apostrophes.) The vectors picked up
    male and female, and even a little grammar, without being told what those are.

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
    - derive the sigmoid from softmax, and explain negative sampling
    - use `nn.Embedding` as a table of learned vectors
    - explore nearest words and analogies, and say what their limits are

    ---

    <!-- go-deeper -->
    **Go deeper (optional):** the method is *word2vec* (Mikolov et al., 2013). Real runs also throw away
    some of the most common words, such as `the`, and *GloVe* gets similar vectors by counting
    pairs directly. The idea that a word's meaning shows in its neighbours is called
    the *distributional hypothesis*. The stories are from *TinyStories* (Eldan and Li, 2023).
    """)
    return


if __name__ == "__main__":
    app.run()
