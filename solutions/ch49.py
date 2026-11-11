"""Reference solutions for chapter 49. Peek only after trying!"""

def reaches(ticket, human, context):
    resumes = human and ticket["ended_by"] != "human" and not context
    return ticket["live"] or resumes


def ticket_stage(ticket):
    if ticket["ended"]:
        return "ended"
    if ticket["delivery_failed"]:
        return "person/delivery"
    if ticket["question"]:
        return "person/answer"
    report = ticket["report"]
    if report == "ready":
        return "person/review"
    if (report == "blocked" or report == "failed") and not ticket["person_since"]:
        return "person/" + report
    if ticket["assignment"] is None or ticket["assignment"] == "pending":
        return "working/queued"
    if report == "waiting" or ticket["assignment"] == "ended":
        return "idle"
    return "working"


def follows_done(follows):
    own = [item for item in follows if item["own"]]
    counted = follows
    if len(own) > 0:
        counted = own
    if len(counted) == 0:
        return None
    for item in counted:
        if item["closed"] is None:
            return None
    closes = sorted([item["closed"] for item in counted])
    return closes[-1]
