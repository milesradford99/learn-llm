from learn_llm.checking import Case


def _q(f):
    queue = []
    f(queue, "reply", "T1", "hello", 10)
    f(queue, "launch", "T2", "go", 11)
    f(queue, "reply", "T1", "hello again", 20)
    return [(r["kind"], r["key"], r["payload"], r["due"]) for r in queue]


EXERCISES = {
    "enqueue": [
        Case("ch50.enqueue.new", lambda f: f([], "reply", "T1", "hi", 5), [{"kind": "reply", "key": "T1", "payload": "hi", "due": 5}], hint="ch50.enqueue.hint"),
        Case("ch50.enqueue.replace", _q, [("reply", "T1", "hello again", 20), ("launch", "T2", "go", 11)], hint="ch50.enqueue.hint"),
        Case("ch50.enqueue.kind", lambda f: len(f([{"kind": "reply", "key": "T1", "payload": "x", "due": 1}], "launch", "T1", "y", 2)), 2, hint="ch50.enqueue.hint"),
    ],
    "retry_delay": [
        Case("ch50.retry_delay.grow", lambda f: [f(0, 2000, 60000), f(1, 2000, 60000), f(2, 2000, 60000)], [2000, 4000, 8000], hint="ch50.retry_delay.hint"),
        Case("ch50.retry_delay.cap", lambda f: f(9, 2000, 60000), 60000, hint="ch50.retry_delay.hint"),
        Case("ch50.retry_delay.huge", lambda f: f(1000, 1, 10 ** 9), 65536, hint="ch50.retry_delay.hint"),
    ],
    "reserve": [
        Case("ch50.reserve.free", lambda f: f(2, 3, "launch"), True, hint="ch50.reserve.hint"),
        Case("ch50.reserve.full", lambda f: [f(3, 3, "launch"), f(3, 3, "wake")], [False, False], hint="ch50.reserve.hint"),
        Case("ch50.reserve.person", lambda f: [f(3, 3, "person"), f(4, 3, "person")], [True, False], hint="ch50.reserve.hint"),
    ],
}
