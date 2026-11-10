# Curriculum design

Tier 2, Track A. Status: decided 2026-10-24: route R3, the recommended gradient
combination, and the same English text for every language from ch 12. Decisions live in
SCOPE.md; this file keeps the reasoning and the rejected options.

## Brief

- **Job:** When I start from almost nothing and want to really understand how LLMs work,
  I want one path where every step follows from the last, so that I never get lost, never
  have to look elsewhere, and come out able to derive, code and explain every piece myself.
- **Who:** adult self-learners with high-school arithmetic (P1–P3 in `novice-review.md`).
  **Not served:** people who want a fast survey, an API-usage course or a credential.
- **Evidence:** UNVALIDATED beyond the maintainer's own experience. Cheapest validation:
  watch 2–3 real beginners do chapters 00–02 before writing chapter 05.
- **The one thing:** *no gaps.* A reader who does each page in order never meets an idea,
  symbol or step that the course hasn't already given them.
- **Signature moment:** the learner computes a gradient three ways (by hand on tiny
  numbers, by nudging, by their own backprop code) and watches all three agree.
- **Non-goals:** surveying every architecture; matching the source courses' lecture order;
  any concept taught "just trust me".
- **Success signal:** completion rate per chapter; how many readers pass each chapter's
  checks without outside help (measured later through optional, opt-in reports or issues).
- **Acceptance:** for each chapter, zero blockers from the novice review, all checks pass,
  and the concept ledger (below) has no item used before it was taught.

## Mechanisms that enforce "no gaps"

These are the parts that make the promise checkable rather than a hope.

1. **Concept ledger (machine-checked).** Each chapter declares the ideas it *introduces* and
   the ideas it *uses* (`derivative`, `chain-rule`, `matrix-shape`, ...). CI fails if a chapter
   uses an idea no earlier chapter introduced. The same ledger is handed to the novice
   reviewer as "Already taught", so the review checks the prose against it.
2. **Novice-reader gate.** Every chapter passes the adversarial review in
   `novice-review.md` (personas P1, P2, P3) with zero blockers.
3. **Three views for every new piece of maths:** in words, in small concrete numbers,
   in code, side by side. Notation (Σ, ∂, matrices) only appears *after* the reader has
   used the idea in words and numbers, as a shorthand for something they already do.
4. **Recall, don't link.** When a chapter needs an earlier idea, it restates it in two lines
   on the page instead of sending the reader back. Cross-chapter links stay optional.
5. **Every claim is checkable on the page.** Derived results (a derivative, a gradient
   formula, a shape rule) are always verified numerically right after, by the learner.
6. **Wrong answers are designed, too.** For each exercise, the review lists the likeliest
   wrong attempts; each must produce feedback that points to the fix (see
   `reviews/ch00.md`: half of chapter 00's blockers were about feedback, not prose).

## Choice 1: the route

| | Route | Through-line | Cost |
|---|---|---|---|
| **R1** | Course order as drafted: vision (231n) → language (224N) → scale (336) → serving | None; each part has its own datasets | Most faithful to sources; least continuity between parts |
| **R2** | One problem from start to end: predict the next character, solved better and better (counting → linear → MLP → RNN → attention → transformer → serving) | One dataset and one leaderboard for the whole course | Strongest continuity; vision shrinks to one optional detour; ch 01–02 harder to picture |
| **R3** (recommended) | Numbers → digits → text. Learn the machinery on things you can see (a line, then handwritten digits, then a small CNN), then switch once to text, and from then on one text dataset and one leaderboard to the end | Two through-lines, one switch at ch 12 | Keeps 231n's core and easy visuals early; one deliberate switch, announced and motivated |

R3 adds a **leaderboard** from ch 12: every new model is scored on the same held-out text
(perplexity), so each chapter visibly beats the last. For zh-CN, the text dataset could be
Chinese, character-level (open question).

## Choice 2: how gradients and backprop are taught

Headline options; the recommendation combines several.

1. **Nudge first.** The derivative starts as an experiment: move the knob a tiny bit,
   measure how much the error moves, divide. Direction and steepness come from that ratio
   before any formula.
2. **Earn the rules by algebra of nudges.** (x + h)² − x² = 2xh + h², so the ratio is
   2x + h, which becomes 2x as h shrinks. Every rule in the table is derived this way, then
   checked numerically.
3. **Exchange rates multiply (chain rule).** If moving w by a tiny amount moves the prediction
   x times as much, and moving the prediction moves the loss 2·error times as much, then
   moving w moves the loss x · 2·error times as much. Rates along a path multiply; when a
   value feeds two places, the effects add. (Mechanism borrowed from currency conversion.)
4. **Hand-trace tables.** Before any autograd code, the learner fills in a forward column
   and a backward column for a 4-node graph on the page, cell by cell, with checks.
5. **Live graph widget.** Drag an input; every node shows its value and its sensitivity
   updating live. Makes "blame flows backwards" something you can see.
6. **One element by hand, then the shape rule.** Matrix gradients (dW = xᵀ · dout) are derived
   by computing one entry in full and checking a 2×2 case; the shape rule is a sanity
   check, never the derivation.
7. **Gradient check as a permanent tool.** From ch 02 on, every gradient the learner writes
   is compared to nudging, automatically, in the page's checks.
8. **(Extreme) No calculus notation until ch 06.** Everything is "nudge ratio" in words and
   code; d/dx and ∂ appear only when the reader already does it fluently.

**Recommended:** 1 + 2 + 3 + 4 + 6 + 7, with 5 if a spike shows marimo can do it smoothly,
and notation introduced in ch 02 (not as late as option 8) once the nudge ratio is familiar.

## Sketch: chapters 01–08 under R3 and the recommended approach

(2026-10-25: a new chapter 04, "Keeping the books", builds the automatic backprop engine, so
the chapters below from "Many numbers at once" on are one number higher than written.)

To show what "no gaps" means in practice, here is the step sequence.

- **01 Fitting a line.** Ice-cream sales vs. temperature. Prediction = w·x + b. Error, then
  *squared* error (why: negatives cancel; big misses should count more), then mean squared
  error (the mean from ch 00). Sliders for w and b; the learner hunts for the lowest error
  by hand, then sees the error plotted against w: a valley.
- **02 Going downhill.** "Which way should w move?" Nudge experiment → slope as a ratio → step
  rule w ← w − rate · slope → learning rate too big/too small, seen live. Nudging costs two
  error calculations per knob, so with a million knobs it's hopeless: motivation for exact
  slopes. Algebra of nudges derives the rules (constant, sum, scale, x², product); each
  checked numerically. Only then: "this ratio has a name, the derivative, written d/dx".
- **03 Blame flows backwards.** Error = (w·x + b − y)² as a chain of small steps. Exchange
  rates multiply along the chain; check by nudging. A value used twice: effects add; check.
  Hand-trace table. Then code: a `Value` that remembers how it was made. The learner first
  runs a backward pass in the wrong order, sees a wrong answer, and discovers why every user
  of a value must report before it passes blame on (topological order).
- **04 Many numbers at once.** Three features instead of one → dot product, in three views.
  Matrices as many dot products at once; shapes; broadcasting. NumPy enters here as "the
  same loops, written once".
- **05 Recognising digits.** Scores → make positive (exp) → share of the total (softmax).
  "Surprise" → log → cross-entropy. The gradient (probabilities − one-hot) is *derived* with
  ch 03's rules on a 3-class example by hand, then vectorised, then gradient-checked.
- **06 Neural networks.** Why a straight line can't bend; ReLU. Matrix backprop derived one
  entry at a time (option 6). The learner's ch 03 engine and the matrix version must agree.

## Step plan: chapters 06–09 (written 2026-10-25)

Data: a bundled 12,000-image MNIST subset (`learn_llm/data`), so every chapter works offline.

- **06 Recognizing digits.** An image is a 28 × 28 grid of numbers (indexing `a[0]` taught
  here), flattened to 784. One weight column per digit: `X @ W` is many dot products at once
  (matrix @ matrix, the shape rule extended). Guess = the biggest score (`np.argmax`), accuracy
  = share of correct guesses; an untrained model gets about 10%. Right/wrong has no slope, so:
  make scores positive with `e^s` (e built from nudges: the base whose growth rate equals
  itself), then shares that add to 1 (softmax; subtracting each row's largest score changes
  nothing and avoids overflow; `axis` and `keepdims` taught). Loss = surprise `−log p` of the
  right digit (log as "undo e^"). Slopes of `e^x` and `log x` derived by nudging; the slope
  of the loss for each score is `p − 1` for the right digit and `p` for the others, derived on
  a 3-class example and nudge-checked. Vectorized: `G = (P − Y) / n`, `X.T @ G` (transpose
  taught), then gradient descent: about 90% after 100 steps. The 10 weight columns shown as
  images.
- **07 Neural networks.** One weighted sum per digit can't express "either this or that". A
  bend: ReLU, `max(0, z)`; sums of ReLUs make bendy curves (|x| = relu(x) + relu(−x), then
  ch 04's heatwave hill). Hidden layer `H = relu(X @ W1 + b1)`, scores `H @ W2 + b2`. Backward
  through matrices, one entry at a time first: `dW2 = H.T @ G` (ch 06 with H as input),
  `dH = G @ W2.T` (each hidden value feeds all ten scores: blames add), ReLU passes blame
  where z > 0, `dW1 = X.T @ dZ`. Checked against nudging. 64 hidden units: about 94%.
- **08 Training that works.** Mini-batches and epochs (shuffling with a seeded random
  generator): the same accuracy about 60 times faster. Train vs test: overfitting, and weight
  decay as the fix. Initialization: all-zero weights never learn (symmetry); the spread
  (variance, taught here) of values through a deep stack shrinks or explodes unless the
  scale is about √(2 / inputs). Moving averages → momentum (ch 02's crawling `b`, fixed);
  dividing by the size of recent slopes → Adam.

## Step plan: chapters 10–12 (written 2026-10-31)

- **09 Hello PyTorch.** Not a new subject: each piece is compared with the learner's own. Tensors
  and dtypes (32 against 64 bits, why last digits differ); `requires_grad` and `backward()` as
  chapter 04's book, with Mia's 476 and 34 again; blame adds up, so it must be cleared. Chapter
  07's network computed by hand and by PyTorch, slopes equal to rounding. `nn.Linear`,
  `nn.Sequential`, then `torch.optim.Adam` against the learner's `adam_step`: the same knobs.
  Clear, forward, backward, step. GPU and device in words only: the install is CPU-only.
- **10 Seeing with convolutions.** Digits pasted at random spots on a 40 × 40 canvas: a flat
  network memorises (412,426 knobs, 100% train, 88% test). Convolution by hand, then in loops;
  feature maps; it moves with the picture; max pooling; `nn.Conv2d` checked against the learner's
  own; a CNN with 6,218 knobs gets 96%. Blame fades through depth: with 10 blocks, about 1e-12
  reaches the first layer, about 0.015 with `x + f(x)`; 10% against about 80%. Own layers as classes.
- **11 Words as vectors.** TinyStories from here on. One-hot rows carry no meaning; neighbours do.
  Dictionaries to count, ids, cosine similarity, skip-gram pairs. Softmax over 2,000 words is too
  slow, so a yes/no question: the sigmoid is softmax over `s` and 0, and negative sampling reuses
  chapter 06's surprise. `nn.Embedding`, then nearest words and analogies, with their limits. The
  leaderboard starts in chapter 13, once there is a language model to score.

## Step plan: chapters 13–15 (written 2026-11-01)

- **12 Predicting the next word.** No gradients: counting. Conditional probability from a tiny
  text, a bigram table, smoothing because −log 0 is infinite, sampling, then perplexity derived so
  that a know-nothing model over N words scores N. Scores: 2,002, unigram 282, bigram 54, trigram
  mixed with bigram 38. The leaderboard starts. Counting can't share between `dog` and `cat`.
- **13 Memory: RNNs.** tanh, a recurrent step, `nn.RNN` checked against the learner's own. Blame on
  the first of 60 words: about 4e-15. A fresh LSTM is just as bad (forget gates near 0.5); with
  them opened it's about 0.04, so the LSTM's advantage is that it has the dial. An LSTM of 128
  scores about 31. Temperature.
- **14 Attention.** Encoder and decoder on reversing text, so answers can be checked at a glance.
  Without attention: 99% at 5 words, 23% at 20. Weighted averages, then query, key and value, then
  compare, share out, blend: 95% at 20 words with the same memory. The attention picture is a
  diagonal nobody asked for. Not on the leaderboard: that's for the transformer, which next chapter
  answers the question of whether the LSTMs are needed at all.

## Step plan: chapters 16–19 (written 2026-11-01)

- **15 The Transformer.** Self-attention, the future mask, scaling by √d (variances add), multi-head
  by cutting vectors, a measured demo that attention ignores order, a position table, layer norm,
  and blocks of two `x + f(x)` steps. About 1.1 million knobs and perplexity about 23, first on the board.
- **16 Pretrain, fine-tune, evaluate.** Name the blanked animal in a story. Majority baseline 40%,
  precision and recall, a margin of error from variances adding. Pretraining reaches about 51% with
  no labels, but with labels the three columns sit inside the margin. Leaving the word in gives 81%
  on leaky test data and 45% on the real task.
- **17 Tokenization.** UTF-8 bytes, then count pairs, merge, train, encode, decode. `while` is taught
  here. 500 merges on 300 stories: 3 bytes per token, whole-story round trip, Chinese survives.
  Scores stay per word when later chapters retrain on tokens.

## Step plan: agents track, chapters 40–51 (written 2026-11-01)

The brief: finish able to say "I know Pi and DeepSeek's Super-agent, their source and their architecture", and
defend it. The promise differs from the main course in one way: the reader finished chapter 19,
so Python is assumed and only new features are taught. Everything else carries over: toolboxes
just in time, the ledger, on-page checks, no required links, en and zh-CN.

Each chapter has one rhythm: quote the real code, then rebuild the mechanism in Python, with
checks built from the behaviour of the source (`Tool X not found`, 2,000 lines and 50 KB, warn
at 3 and stop at 5). A scripted `FakeModel` plays the model, in `learn_llm/agents.py`. Excerpts
can't drift unnoticed: `learn_llm/source_check.py` compares every one with a checkout.

- **40–46, Pi:** depth on the loop. Chapter 40 gives the reader a working agent in 20 lines and
  lists what's wrong with it, and each later chapter is one of those problems: steering, streams
  and providers, tools, the session tree, compaction, then extensions and trust.
- **47–51, Super-agent:** many loops working for a team. The Router and Workers, the gate and
  Mandates, Subscriptions and Ticket stages, the store-first engine, then prompts and context.
  The prototype differs from DeepSeek's essay, so every claim says which one it rests on. It's
  taught by contrast, so each chapter names what Pi does instead.
- **41 ends with an exam:** twelve graded questions, then seven practical tasks with real models,
  because the course is honest that reading isn't experience. The tasks need keys and money, so
  they stay outside the graded path.

Rejected: a separate top-level folder with its own framework copy (it would fork the checks,
ledger and i18n); numbering from 18 (it would collide with the main course's map); chapters that
need a real API key (it breaks "no accounts" and gives different results each run).

## Step plan: chapters 20–23 (written 2026-11-09)

- **18 A modern transformer.** Fair tokens first (same text as the word models, scored per word).
  RMSNorm, angles and rotations taught from the circle, RoPE, SwiGLU, AdamW checked against
  PyTorch, warmup and cosine, checkpoints that resume exactly. Measured on the same text and time:
  chapter 17's design 23.4, modern 19.3. Spelling rare words out costs about 40% more.
- **19 Making it fast.** Timing, FLOPs of a product and of training (6N per token), the machine's
  speed and a run's utilisation, Adam's 16 bytes per knob. bfloat16 measured on the learner's own
  machine (about 100 times slower on the test CPU). Tiled attention with an online softmax, then
  the fused version (8 to 13 times faster). A profile.
- **20 Many devices.** Equal slices average to the whole batch, checked; gradient accumulation;
  ring all-reduce bytes; real training on 2 and 4 processes matching one process to about 1e-7;
  sharding stages from 112 to 14 GB per device; tensor parallel by columns.

## Decisions

- Route: **R3** (user). R1 rejected: restarts between parts. R2 rejected: vision shrinks to a
  detour and the first gradient chapters are harder to picture.
- Gradients: **the recommended combination** (user). Option 8 (no notation until ch 06)
  rejected: readers would meet standard notation too late to connect with other material.
- Text from ch 12: **the same English text for every language** (user). One dataset and one
  set of expected numbers.
- Chapter 04 is a new chapter, "Keeping the books", for the automatic engine (user, 2026-10-25).
  Folding it into "Many numbers at once" would have taught two hard new ideas (classes and
  matrices) in one chapter.
- Chapters 06–09 use a bundled MNIST subset (user, 2026-10-25). Normalization layers move to
  the Transformer chapter, where layer norm is first needed; dropout is mentioned only, since
  weight decay already shows regularization (agent-default).
