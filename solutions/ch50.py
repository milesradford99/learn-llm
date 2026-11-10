"""Reference solutions for chapter 50. Peek only after trying!"""

def enqueue(queue, kind, key, payload, now):
    for row in queue:
        if row["kind"] == kind and row["key"] == key:
            row["payload"] = payload
            row["due"] = now
            return queue
    queue.append({"kind": kind, "key": key, "payload": payload, "due": now})
    return queue


def retry_delay(attempts, first, cap):
    return min(cap, first * 2 ** min(attempts, 16))


def reserve(used, max_workers, by):
    extra = 0
    if by == "person":
        extra = 1
    return used < max_workers + extra
