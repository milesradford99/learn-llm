from learn_llm.checking import Case


def _tk(live=True, ended_by=None):
    return {"live": live, "ended_by": ended_by}


def _stage(**changes):
    ticket = {"ended": False, "delivery_failed": False, "question": None, "report": None, "person_since": False, "assignment": "running"}
    ticket.update(changes)
    return ticket


def _pr(own, closed=None):
    return {"own": own, "closed": closed}


EXERCISES = {
    "reaches": [
        Case("ch49.reaches.live", lambda f: [f(_tk(), False, False), f(_tk(), True, True)], [True, True], hint="ch49.reaches.hint"),
        Case("ch49.reaches.resume", lambda f: f(_tk(False, "executor"), True, False), True, hint="ch49.reaches.hint"),
        Case("ch49.reaches.bot", lambda f: f(_tk(False, "executor"), False, False), False, hint="ch49.reaches.hint"),
        Case("ch49.reaches.stopped", lambda f: f(_tk(False, "human"), True, False), False, hint="ch49.reaches.hint"),
        Case("ch49.reaches.context", lambda f: f(_tk(False, "executor"), True, True), False, hint="ch49.reaches.hint"),
    ],
    "ticket_stage": [
        Case("ch49.ticket_stage.ended", lambda f: f(_stage(ended=True, delivery_failed=True)), "ended", hint="ch49.ticket_stage.hint"),
        Case("ch49.ticket_stage.order", lambda f: [f(_stage(delivery_failed=True, question="?")), f(_stage(question="?", report="ready"))], ["person/delivery", "person/answer"], hint="ch49.ticket_stage.hint"),
        Case("ch49.ticket_stage.blocked", lambda f: [f(_stage(report="blocked")), f(_stage(report="failed", person_since=True))], ["person/blocked", "working"], hint="ch49.ticket_stage.hint"),
        Case("ch49.ticket_stage.queued", lambda f: [f(_stage(assignment=None)), f(_stage(assignment="pending"))], ["working/queued", "working/queued"], hint="ch49.ticket_stage.hint"),
        Case("ch49.ticket_stage.idle", lambda f: [f(_stage(report="waiting")), f(_stage(assignment="ended")), f(_stage(report="progress"))], ["idle", "idle", "working"], hint="ch49.ticket_stage.hint"),
    ],
    "follows_done": [
        Case("ch49.follows_done.none", lambda f: f([]), None, hint="ch49.follows_done.hint"),
        Case("ch49.follows_done.open", lambda f: f([_pr(True, ("2", "merged")), _pr(True)]), None, hint="ch49.follows_done.hint"),
        Case("ch49.follows_done.own", lambda f: f([_pr(True, ("2", "merged")), _pr(False)]), ("2", "merged"), hint="ch49.follows_done.hint"),
        Case("ch49.follows_done.latest", lambda f: f([_pr(False, ("5", "closed")), _pr(False, ("3", "merged"))]), ("5", "closed"), hint="ch49.follows_done.hint"),
    ],
}
