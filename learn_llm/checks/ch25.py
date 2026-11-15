import zlib

from learn_llm.checking import Case
from learn_llm.rawdata import seeded_hash


def _same_marks(f):
    return f("The  Cat\nsat") == f("the cat sat"), f("the cat sat") == f("the cat sat down"), f(" THE CAT SAT ")


def _shingles(f, words, n):
    out = f(words, n)
    return sorted(out) if isinstance(out, set) else out


EXERCISES = {
    "normalise": [
        Case("ch25.normalise.messy", lambda f: f("  The  CAT\nsat "), "the cat sat", hint="ch25.normalise.hint"),
        Case("ch25.normalise.clean", lambda f: f("one day"), "one day", hint="ch25.normalise.hint"),
    ],
    "fingerprint": [
        Case("ch25.fingerprint.marks", _same_marks, (True, False, zlib.crc32(b"the cat sat")), hint="ch25.fingerprint.hint"),
    ],
    "drop_copies": [
        Case("ch25.drop_copies.small", lambda f: f(["A b", "c", "a  B", "c", "d"]), ["A b", "c", "d"], hint="ch25.drop_copies.hint"),
        Case("ch25.drop_copies.none", lambda f: f(["x", "y"]), ["x", "y"], hint="ch25.drop_copies.hint"),
    ],
    "shingles": [
        Case("ch25.shingles.pairs", lambda f: _shingles(f, ["a", "b", "c", "d"], 2), [("a", "b"), ("b", "c"), ("c", "d")], hint="ch25.shingles.hint"),
        Case("ch25.shingles.triples", lambda f: _shingles(f, ["a", "b", "c", "d"], 3), [("a", "b", "c"), ("b", "c", "d")], hint="ch25.shingles.hint"),
        Case("ch25.shingles.repeat", lambda f: _shingles(f, ["a", "b", "a", "b"], 2), [("a", "b"), ("b", "a")], hint="ch25.shingles.hint"),
    ],
    "jaccard": [
        Case("ch25.jaccard.hand", lambda f: f({1, 2, 3, 4, 5}, {1, 2, 3, 4, 6}), 4 / 6, tolerance=1e-9, hint="ch25.jaccard.hint"),
        Case("ch25.jaccard.same", lambda f: f({"a", "b"}, {"b", "a"}), 1.0, tolerance=1e-9, hint="ch25.jaccard.hint"),
        Case("ch25.jaccard.apart", lambda f: f({"a"}, {"b", "c"}), 0.0, tolerance=1e-9, hint="ch25.jaccard.hint"),
    ],
    "minhash": [
        Case("ch25.minhash.three", lambda f: f({"p", "q", "r"}, [0, 1, 2]), [min(seeded_hash(seed, s) for s in "pqr") for seed in (0, 1, 2)], hint="ch25.minhash.hint"),
        Case("ch25.minhash.single", lambda f: f({"p"}, [5]), [seeded_hash(5, "p")], hint="ch25.minhash.hint"),
    ],
    "agreement": [
        Case("ch25.agreement.half", lambda f: f([1, 2, 3, 4], [1, 5, 3, 7]), 0.5, tolerance=1e-9, hint="ch25.agreement.hint"),
        Case("ch25.agreement.hand", lambda f: f([2, 1, 1], [2, 3, 1]), 2 / 3, tolerance=1e-9, hint="ch25.agreement.hint"),
    ],
    "known_share": [
        Case("ch25.known_share.most", lambda f: f(["the", "cat", "zzz", "sat"], {"the", "cat", "sat"}), 0.75, tolerance=1e-9, hint="ch25.known_share.hint"),
        Case("ch25.known_share.none", lambda f: f(["qx", "zv"], {"the"}), 0.0, tolerance=1e-9, hint="ch25.known_share.hint"),
    ],
}
