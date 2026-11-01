import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="12 · Predicting the next word")

with app.setup(hide_code=True):
    import re

    import marimo as mo
    import numpy as np

    from learn_llm import todo
    from learn_llm.checks import check
    from learn_llm.data import stories
    from learn_llm.leaderboard import record, show

    LOCALE = "en"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 12 · Predicting the next word

    Every chatbot you've used does one thing, over and over: given the words so far, it predicts
    the next one. A model that does this is called a **language model**. It's the idea this
    whole course has been heading for.

    In this chapter you'll build the simplest language models there are, with no network and no
    gradient at all, only counting. They're worth building for two reasons. They generate real
    (if silly) text. And they give us a **score** that every later model has to beat.

    ## The text as numbers

    As in chapter 11, we cut the stories into words and keep the 2,000 most common. This time
    two extra entries join the vocabulary:

    - `<unk>` ("unknown") stands for any word outside the 2,000;
    - `<end>` marks the end of a story, so that a model can learn when to stop.

    `to_words` and `count_words` are chapter 11's. `story_ids` is new: it turns a list of stories into one long array of ids.
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

    > P(next | previous) is "the chance of `next`, given that we've just seen `previous`"

    The simplest way to estimate it is to count. Take this tiny text:

    > the dog ran . the dog sat . the cat ran .

    `the` appears 3 times; twice `dog` came next, and once `cat`. So P(dog | the) = 2 / 3,
    P(cat | the) = 1 / 3, and every other word gets 0. A table of these counts, with one row for each
    word and one column for each possible next word, is called a **bigram** model ("bi" for two: it
    looks at pairs).

    ## Exercise 1: bigram_counts

    Write `bigram_counts(ids, size)`, giving back a table `C` of shape `(size, size)` where `C[a, b]`
    is how many times `b` came right after `a`. Pairing a list with itself, moved along by one, gives
    every neighbouring pair: `ids[1:]` is `ids` without its first item.

    1. `C = np.zeros((size, size))`
    2. `for a, b in zip(ids, ids[1:]):`, and inside it `C[a, b] = C[a, b] + 1`
    3. After the loop, `return C`
    """)
    return


@app.function
def bigram_counts(ids, size):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "bigram_counts", bigram_counts, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## The trouble with zero

    To turn a row of counts into chances, divide by the row's total. But any pair never seen in
    training then gets a chance of exactly 0, and chapter 06's surprise, −log 0, is infinite. One unseen pair
    in the test stories would make the whole score infinite, as if the model had called it impossible.

    The usual fix is to pretend every pair was seen a tiny number of extra times, `k`. This is called
    **smoothing**: add `k` to every count first, then divide by the row's new total.

    ## Exercise 2: bigram_probs

    Write `bigram_probs(C, k)`. Each row of the result must add up to 1:

    1. `smoothed = C + k`
    2. `return smoothed / np.sum(smoothed, axis=1, keepdims=True)`
    """)
    return


@app.function
def bigram_probs(C, k):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "bigram_probs", bigram_probs, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Here's what the model thinks can follow `the`, and `once`: the five likeliest next words,
    with their chances. (`np.argsort(-row)` lists positions from largest to smallest, as in chapter 11, and `round(x, 3)` keeps 3 digits after the point.
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
    `the` might always be the same, and the text would soon go round in circles.

    `rng.choice(n, p=p)` draws one number from 0 to n − 1, with the chances in `p`.

    ## Exercise 3: sample_next

    Write `sample_next(p, rng)`, drawing one id from the vector of chances `p`:

    ```
    return rng.choice(len(p), p=p)
    ```

    The cell after the check starts from `once` and writes 30 words.
    """)
    return


@app.function
def sample_next(p, rng):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "sample_next", sample_next, LOCALE)
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
    that's too tiny for a computer to store. Logs fix this, because log(a × b) = log a + log b: instead of multiplying
    chances, we add their logs. Divide by the number of words and flip the sign, and you have
    a quantity you already know:

    > mean surprise = −(mean of log p), chapter 06's cross-entropy

    To make it easier to read, we undo the log with e^. That gives the **perplexity**:

    > perplexity = e^(mean surprise)

    Here's what that number means. Suppose a model knows nothing, and gives each of `N` words the chance 1 / N. Since log(1 / N) = log 1 − log N = −log N,
    every surprise is log N, so the mean is log N, and the perplexity is e^(log N) = N. So
    **a perplexity of N means the model is as unsure as if it were choosing among N equally likely words.**
    A perfect model scores 1. Lower is better.

    ## Exercise 4: perplexity

    Write `perplexity(log_probs)`, where `log_probs` is an array of the log of the chance given to each real next word:

    ```
    return np.exp(-np.mean(log_probs))
    ```

    After the check, the cell scores three models on the test stories, and the bigram again on its training text.

    - **Know-nothing:** 1 / 2002 for every word. `np.full(n, x)` is an array of `n` copies of `x`.
    - **Unigram:** just how common each word is, ignoring what came before. The total of row `a` of `C`
      counts how many times `a` was followed by something, which is how often `a` appears, so the row totals give what we need.
    - **Bigram:** `P[test_ids[:-1], test_ids[1:]]` looks up one chance for each neighbouring pair. Two arrays of positions pick
      one entry per pair, so `P[[0, 2], [1, 3]]` gives `P[0, 1]` and `P[2, 3]`. (`test_ids[:-1]` is everything but the last item.)
    """)
    return


@app.function
def perplexity(log_probs):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "perplexity", perplexity, LOCALE)
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
    model is as unsure as choosing among 54 words. The last number is the bigram scored on its own training text: about 42. As in chapter 08, a model
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

    One word of memory isn't much. A **trigram** model looks at the two previous words, P(next | a, b).
    A table for that would need 2,002 × 2,002 × 2,002 entries, about 8 billion, but almost all of them would be 0. So
    we use a dictionary instead, storing only what we actually saw: each pair `(a, b)` is a key, and its value
    is a smaller dictionary counting what came next. (A key can be a pair in round brackets.) In the function below, `followers` isn't a copy: it's the same smaller
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
    ## Exercise 5: trigram_chance

    Write `trigram_chance(table, a, b, c)`: of all the times `(a, b)` was seen, the share that were
    followed by `c`. A pair never seen gives 0. `followers.values()` goes through a dictionary's values, and `sum` adds them up.

    1. `if (a, b) not in table:`, and inside it `return 0.0`
    2. `followers = table[(a, b)]`
    3. `return followers.get(c, 0) / sum(followers.values())`
    """)
    return


@app.function
def trigram_chance(table, a, b, c):
    # ← replace todo() below with your code
    todo()


@app.cell(hide_code=True)
def _():
    check("12", "trigram_chance", trigram_chance, LOCALE)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Trigrams have the zero problem much worse: many pairs in the test stories were never seen. So
    we **mix**: half the chance comes from the trigram, half from the smoothed bigram.
    When the trigram knows nothing, the bigram still has an answer. `.tolist()` turns an array into a plain list, which is faster to loop over and to use in dictionary keys. The loop below takes a few seconds:
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
    about `the cat ran`. Each word is only a row number. But chapter 11 gave us vectors in which `dog` and
    `cat` are close, so a model built on vectors could share what it learns between similar words.

    ## What's next

    In **Chapter 13** you'll build a network that reads text one word at a time, keeping a running
    memory of what it has read. It's called a **recurrent network**, and it will take the top of this leaderboard.

    ## What you can do now

    - explain what a language model does, and read P(next | previous)
    - count bigrams and trigrams, and turn counts into chances
    - explain why zero chances are a problem, and fix them with smoothing or mixing
    - generate text by sampling
    - compute perplexity and say what it means
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
