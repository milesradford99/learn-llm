from learn_llm.checking import Case

TEXT = "aaabdaaabac"
A = list(TEXT.encode("utf-8"))


MERGES = {(97, 97): 256, (256, 97): 257, (257, 98): 258}


EXERCISES = {
    "to_bytes": [
        Case("ch17.to_bytes.dog", lambda f: f("dog"), [100, 111, 103], hint="ch17.to_bytes.hint"),
        Case("ch17.to_bytes.chinese", lambda f: len(f("你好")), 6, hint="ch17.to_bytes.hint"),
    ],
    "count_pairs": [
        Case("ch17.count_pairs.page", lambda f: f([1, 2, 1, 2, 3]), {(1, 2): 2, (2, 1): 1, (2, 3): 1}, hint="ch17.count_pairs.hint"),
        Case("ch17.count_pairs.one", lambda f: f([1]), {}, hint="ch17.count_pairs.hint"),
    ],
    "merge": [
        Case("ch17.merge.some", lambda f: f([1, 2, 3, 1, 2], (1, 2), 9), [9, 3, 9], hint="ch17.merge.hint"),
        Case("ch17.merge.overlap", lambda f: f([1, 1, 1], (1, 1), 9), [9, 1], hint="ch17.merge.hint"),
        Case("ch17.merge.end", lambda f: f([3, 1], (1, 2), 9), [3, 1], hint="ch17.merge.hint"),
    ],
    "train_bpe": [
        Case("ch17.train_bpe.page", lambda f: f(A, 3), MERGES, hint="ch17.train_bpe.hint"),
        Case("ch17.train_bpe.none", lambda f: f(A, 0), {}, hint="ch17.train_bpe.hint"),
    ],
    "encode": [
        Case("ch17.encode.page", lambda f: f(TEXT, MERGES), [258, 100, 258, 97, 99], hint="ch17.encode.hint"),
        Case("ch17.encode.none", lambda f: f("hi", {}), [104, 105], hint="ch17.encode.hint"),
    ],
    "decode": [
        Case("ch17.decode.page", lambda f: f([258, 100, 258, 97, 99], MERGES), TEXT, hint="ch17.decode.hint"),
        Case("ch17.decode.bytes", lambda f: f(list("你好".encode("utf-8")), {}), "你好", hint="ch17.decode.hint"),
        Case("ch17.decode.broken", lambda f: "�" in f([228], {}), True, hint="ch17.decode.hint"),
    ],
}
