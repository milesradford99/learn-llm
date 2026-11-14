# Even depth: plan for chapters 06–20

Status: done 2026-11-14 for chapters 06–23 (new numbers). Every one of them is off the
`WAITING` list in `tests/test_depth.py`; what remains on it is outside this plan (see
"Not covered here"). Still open: the novice-reader review of the reworked chapters, in both
languages (`docs/design/novice-review.md`).

**Up to "Chapter by chapter", numbers are the old ones**, from before the renumbering, because
the measurements were taken then. The table under "Splits and numbering" maps old to new.
The chapter-by-chapter section uses the new numbers.

## The problem, measured

| Chapters | Prose words per new idea | Solution lines printed in the lesson | Learner's job |
|---|---|---|---|
| 00–05 | 160–280 (ch 00: 72) | 0–77% | works it out by hand, then codes it |
| 06–20 | 80–180 | 78–100% | copies lines the text supplies |
| 30–41 | 95–290 | 0–55% | writes 5–15 line functions from a description |

"New idea" is a concept in `lessons/ledger.toml`. "Printed" means a line of `solutions/chNN.py`
appears word for word in the lesson text. Chapters 06, 07, 08, 13, 14, 16, 18 and 20 are at 100%.

Chapters 00–05 are the model to match. This plan brings 06–20 up to them. It doesn't make
anything longer for its own sake: every addition is a worked example, a picture, or an exercise.

## Rules for every chapter (06–20)

1. **No dictated answers.** An exercise gives the function's name, inputs, output, shapes and
   one worked input with its result. It doesn't print the solution's lines. Formulas are given
   as math, since turning math into code is the skill.
2. **New syntax gets a Toolbox box**, shown on a *different* example from the exercise
   (`while`, `masked_fill`, `triu`, `torch.cat`, slices, `max(..., key=...)`).
3. **Hints are opt-in.** Each exercise has a collapsed "Stuck?" block under the check
   (`mo.accordion`): hint 1 names the tools, hint 2 gives the line count and the first line.
   The full answer stays in `solutions/`.
4. **Hand first.** Each chapter's main idea is first done by hand on tiny numbers, with
   number boxes checked on the page, as in chapter 03's blame table. Then it's coded.
5. **Given code is read, not skipped.** A given cell over 10 lines gets either one piece
   the learner writes (usually `forward`) or a "predict the shape/number" question on it.
   Code re-pasted from an earlier chapter goes in one cell labelled "From earlier chapters".
6. **Budget:** at most 12 new ideas a chapter, not counting Toolbox items. Over that, split.
7. **One picture for each structure**: network, unrolled RNN, attention, block, ring.

## Splits and numbering (needs a decision)

Three chapters are too full to fix in place: 06 (25 ideas, 3,650 words), 15 (the whole
transformer in 1,700 words) and 18 (six topics).

| Now | After | Title |
|---|---|---|
| 06 | 06 | Recognizing digits (scores, guess, accuracy) |
| 06 | 07 | A loss for choices (softmax, cross-entropy, its slopes) |
| 07–14 | 08–15 | unchanged titles |
| 15 | 16 | Self-attention |
| 15 | 17 | The Transformer (the GPT) |
| 16, 17 | 18, 19 | unchanged titles |
| 18 | 20 | A modern transformer (RMSNorm, RoPE, SwiGLU) |
| 18 | 21 | Training it properly (AdamW, schedule, checkpoints) |
| 19, 20 | 22, 23 | unchanged titles |
| 21–27 (unwritten) | 24–30 | unchanged titles |
| 30–41 (agents) | 40–51 | unchanged titles |

The agents track must move because the main course would reach 30. It's still uncommitted, so
this is the cheapest moment. The renumbering is one scripted commit before any content change:
lesson files in both locales, `learn_llm/checks/chNN.py`, `solutions/chNN.py`, ledger keys,
`check("NN", ...)` calls, every "chapter NN" in prose and check messages, `SCOPE.md`,
`docs/design/reviews/`, `AGENT-SOURCES.md`, the lock file. `make test` must pass with no
content edits.


## Chapter by chapter

Chapter numbers in this section are the **new** ones (renumbered here on 2026-11-14, when the work was finished).

Each entry lists: what to add, what to do with each exercise, and what to do with given code.

### 06 and 07 · Recognizing digits → two chapters

**06 (scores, guess, accuracy).** Sections up to "Why accuracy can't be the loss".
- Add: a hand exercise on a 3-picture, 3-class score table: pick each guess, count the share right.
- Add exercise `guess(S)` (argmax along rows); it's used but never written by the learner now.
- `accuracy`: from spec.
- Toolbox boxes: indexing and `reshape`; `axis`.

**07 (a loss for choices).** From "$e^s$" to the end.
- Add: hand table for softmax of scores `[2, 1, 0]`, with the three $e^s$ values given.
- Add: hand table for one picture's slope, $P - Y$, before "All pictures, all knobs".
- `softmax`, `cross_entropy`: from spec. `gradient`: formula in math, code by the learner.
- Keep the two derivation sections (307 and 349 words) as they are: this is the right depth.

### 08 · Neural networks
- Add: a picture of the network with the shape on every arrow.
- Add: hand exercise on a 2-2-2 network with whole numbers: forward values, then blames.
  "Slopes through two layers" is 528 words with no table or numbers to follow.
- `relu`: from spec. `forward`: from the picture's shapes.
- `backward` (8 of 8 lines printed): give the four formulas in math and the shape each result
  must have. The nudge check that follows is already the safety net.

### 09 · Training that works
Don't split; four topics, but each is small once the Python is boxed.
- Toolbox boxes: slices and `range(start, stop, step)`; shuffling with a permutation.
- Add: predict exercise for the depth table (does the spread grow, shrink or hold?).
- Add: hand table of a moving average over 4 steps, then Adam's bias correction for steps 1–3.
- `variance`, `batches`: from spec. `momentum_step`, `adam_step`: math given, code by the learner.
- "Learning or memorising?" is 61 words and has no exercise: add a read-the-two-curves question.

### 10 · Hello PyTorch
The comparison with the learner's own engine works well; keep the structure. 28 ideas, but most
are new names for known things.
- `torch_gradient`: remove the 6 printed lines; the engine-to-PyTorch table is the hint.
- `train_step`: replace the dictation with "find the bug" (a step that forgets `zero_grad`),
  then write the correct one.
- `scores`, `make_model`: from spec.
- Put dtype and "8 bits, 32 bits" in a Toolbox box.

### 11 · Seeing with convolutions
- Add: hand exercise sliding a 2×2 kernel over a 3×3 picture (four numbers).
- Add: predict the output size for two cases before the formula is stated.
- `convolve`, `max_pool`: loop header given, body by the learner. `residual_block`: from spec.
- `make_cnn` already works this way (1 of 6 lines printed): use it as the pattern.

### 12 · Words as vectors
26 ideas; about ten are Python tools.
- Toolbox boxes: dictionaries and `.get`; `sorted` with a key.
- Add: cosine similarity by hand for two pairs of 2-number vectors.
- Add: list the skip-gram pairs of a 4-word sentence by hand, then write `skipgram_pairs`.
- "Real or made up?" (397 words) carries negative sampling, sigmoid, log-sigmoid and the noise
  distribution at once. Split it in two sections, add a sigmoid chart, and a hand calculation
  of one pair's loss.
- `count_words`, `similarity`, `pair_loss`: from spec.
- Word analogies: shorten, or move to go-deeper.

### 13 · Predicting the next word
No table or picture in the chapter.
- Add: a bigram count table for a three-sentence toy text, filled in by hand, then smoothed.
- Add: perplexity by hand for three probabilities.
- All five exercises are 1–4 lines: from spec.
- Toolbox boxes: tuples as dictionary keys; `rng.choice`.

### 14 · Memory: RNNs
- Add: a picture of the RNN unrolled over three words.
- Add: one RNN step by hand with a one-number memory.
- Add: a gate table by hand (`f = 1, i = 0` leaves the cell alone; `f = 0, i = 1` replaces it).
- `lstm_step` (8 of 8 printed): the two formulas and the gate table are given; code by the learner.
- `rnn_step`, `run_rnn`, `windows`, `sample_word`: from spec; `windows` gets a worked example.
- `Reader` (given): the learner writes its `forward`.
- `train_reader` (13 lines, given): add a predict question on the shape that reaches the loss.

### 15 · Attention
- "Where to look" explains the core idea in 206 words with no numbers. Add a worked example:
  one query, three keys of two numbers each, dot products, shares, blend. Then the same by hand
  with different numbers.
- Add: a picture of the decoder state looking at the encoder states.
- `weighted_average` (`return a @ V`): keep, from spec, after the hand version.
- `attention_shares`, `attend`: from spec, with shapes.
- `shift_right`: Toolbox box for column slices and `zeros_like`; from spec with the example.
- `with_attention` (2 lines, given): the learner writes it.
- `Translator` and `translate` (26 lines, given): add a shape walk-through and a predict question.

### 16 and 17 · The Transformer → two chapters

**16 (self-attention).**
- Query, key, value get 89 words now. Rework chapter 15's three-word example with three
  separate small matrices, so the learner sees why one vector needs three roles.
- Mask: mark the blocked cells of a 4×4 table by hand. Toolbox box for `triu` and
  `masked_fill`. `future_mask` from spec.
- Scale: add a cell showing softmax's largest share at `d = 4, 32, 128`, with and without
  dividing by $\sqrt{d}$. The text only asserts it now.
- Heads: trace a `(1, 2, 4)` tensor through `reshape` and `transpose` with 2 heads, printed at
  each step. `split_heads` from spec.
- `SelfAttention`: `__init__` given, the learner writes `forward`.
- Ends with "Attention can't count".

**17 (the GPT).**
- Positions: a tiny demo that the order test now gives different answers.
- Layer norm: normalise one 4-number vector by hand. `layer_norm` from spec.
- Block: a picture with the two `x + f(x)` roads. `block_step` from spec.
- `TinyGPT`: `__init__` given, the learner writes `forward`.
- Add: count the knobs by hand for a small configuration, then compare with the cell.
- `write`: the learner adapts chapter 14's loop (the one change is `words[-32:]`).
- `train_gpt` and `score_gpt` are chapter 14's: one "From earlier chapters" cell.

### 18 · Pretrain, fine-tune, evaluate
Opens with about 100 lines of given code before the second exercise.
- Collapse the re-pasted model into one "From earlier chapters" cell.
- Precision and recall: a table of ten guesses and labels, counted by hand, then the function.
- Margin of error: add a simulation (score the same model on 200 random test draws and look at
  the spread) before the formula; the formula arrives in 133 words unsupported.
- Leakage is 52 words. Add an exercise: a split function that leaks, find it and fix it.
- Pretraining (52 words) and fine-tuning (62 words): add a picture of what's reused and frozen.
- `blank_out`, `majority_share`, `margin`, `freeze`: from spec.
- BERT (120 words, explained not trained): keep short, with one picture of "both ways".

### 19 · Tokenization
Right length; only the exercises are wrong.
- Add: run two merges by hand on a 12-letter string, entering the pair and the new string.
- Toolbox box for `while`, on a different example (the lesson teaches it inside the exercise).
- `to_bytes`, `count_pairs`, `train_bpe`, `encode`, `decode`: from spec. `count_pairs` reuses
  chapter 12's counting; say so and nothing more.
- `merge`: give the worked example and the "jump two" idea; code by the learner.

### 20 and 21 · A modern transformer → two chapters

**20 (the architecture).** Fair tokens, RMSNorm, rotations and RoPE, SwiGLU.
- Add: a circle picture for angle, cos and sin; none exists.
- Add: rotate `(1, 0)` and `(0, 1)` by a quarter turn by hand.
- `rope` (6 lines, given) is the point of the section: the learner writes it for one vector,
  the batched version stays given.
- `rms_norm`, `rotate`, `swiglu`: from spec.
- `ModernBlock`: the learner writes `forward`. Ends with a shape and knob-count check, no long run.

**21 (training it properly).** AdamW, warm-up and cosine decay, checkpoints, the leaderboard run.
- AdamW: one step by hand next to chapter 09's Adam, to see the single difference.
- `learning_rate`: the chart is shown first; the learner writes the function to match it.
- `adamw_step`, `checkpoint`: from spec.
- Add: stop a run, load the checkpoint, check the loss continues where it left off.

### 22 · Making it fast
Four of five exercises are one printed formula (`return 16 * knobs`).
- Replace each with an estimate for the learner's own model: FLOPs for one step, seconds
  predicted from their measured machine speed, bytes for Adam. Number boxes first, then the function.
- `time_it`: from spec.
- Tiled attention is the hardest algorithm in the course, explained in 217 words and then
  dictated in 12 lines. Add a hand trace: four scores in two tiles, tracking `best`, `total`
  and `acc`. Then the loop header is given and the learner writes the body.
- Mixed precision (105 words, demo only): add a predict question on `float16` overflow.
- Profiling (50 words): add a question on reading a profiler table.

### 23 · Many devices
- Add: a ring picture, and a hand trace of all-reduce on three devices.
- `shard`: 10 knobs on 4 devices by hand, then the function from spec.
- `bytes_per_device`: fill the four-stage table by hand for one size, then the function.
- `average_slopes`, `accumulated_slopes`, `ring_bytes`: from spec.
- "Splitting one layer" (69 words, no exercise): add a small exercise (split a matrix product
  by columns, check it equals the whole) or move it to go-deeper.

## Tooling, so it stays even

- `tests/test_depth.py`: fails if more than 30% of a chapter's solution lines appear verbatim in
  its lesson, or if a chapter introduces more than 12 non-Toolbox ideas. Needs a `toolbox`
  marker in the ledger for syntax-level concepts.
- `docs/design/novice-review.md`: add finding 5, "Dictation: an exercise the reader can pass
  by copying", and finding 6, "Unread code: a given cell the reader has no reason to read".
- `AGENTS.md`: one gotcha line with rules 1–3.
- A helper for hand exercises (number boxes with per-box hints), lifted from chapter 03's cell
  into `learn_llm/`, since every chapter will use it. New UI strings go in both locale files.

## Order of work

In old numbers, as planned:

1. Tooling and the helper; renumbering commit.
2. Pilot: 14 and 15 (three chapters after the split). Review gate, then adjust these rules.
3. 06, 07 (where dictation starts), then 13, 17, 18.
4. 08–12, 16, 19, 20.
5. zh-CN for each chapter after its `en` passes the gate; code is shared, so only prose and stamps.

What happened: the order followed steps 2 to 4 only loosely, `en` and zh-CN went into the
same commits, and the review gate of step 2 has not been run yet.

## Not covered here

- Chapters 00–05: chapter 00 teaches 20 ideas in 1,435 words and is worth a separate look.
- The agents track (chapters 40–51 now): once 06–23 ask for real code, the jump into chapter 40 mostly disappears.
  Whether five chapters on one prototype is the right proportion is a scope question.
