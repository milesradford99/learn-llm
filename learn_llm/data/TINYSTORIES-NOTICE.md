# TinyStories subset

`tinystories-13k.txt.xz` holds 13,000 short stories from the TinyStories dataset, separated by
`<|endoftext|>` lines: 12,000 for training and 1,000 kept aside for testing. They were chosen at
random with a fixed seed (2026) from the file `TinyStoriesV2-GPT4-valid.txt`.

- **Source:** TinyStories by Ronen Eldan and Yuanzhi Li (*TinyStories: How Small Can Language
  Models Be and Still Speak Coherent English?*, 2023), from
  `https://huggingface.co/datasets/roneneldan/TinyStories`.
- **Licence:** the Community Data License Agreement, Sharing, Version 1.0 (CDLA-Sharing-1.0,
  `https://cdla.dev/sharing-1-0/`). This subset is shared under the same licence. It is not
  covered by the repository's MIT or CC BY 4.0 licences.
- **Changes:** curly quotes, long dashes and "…" are replaced with plain ASCII ones; the file's
  first story, which is cut off, and stories with any other non-ASCII character are left out.
  The words are unchanged.
