"""Reference solutions for chapter 47. Peek only after trying!"""

def receive(store, trigger, listened):
    if len(trigger["threads"]) == 0:
        raise ValueError("A trigger must belong to at least one thread")
    if trigger["id"] in store:
        return "duplicate"
    store[trigger["id"]] = trigger
    if not listened:
        return "filtered"
    return "stored"


def deliver(trigger, subscriptions):
    tickets = []
    for thread in trigger["threads"]:
        for ticket in subscriptions.get(thread, []):
            if ticket not in tickets:
                tickets.append(ticket)
    if len(tickets) == 0:
        return "router"
    return tickets
