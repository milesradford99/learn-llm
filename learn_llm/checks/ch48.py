from learn_llm.checking import Case

EXERCISES = {
    "admission_cell": [
        Case("ch48.admission_cell.explicit", lambda f: f(True, True), None, hint="ch48.admission_cell.hint"),
        Case("ch48.admission_cell.cells", lambda f: [f(True, False), f(False, True), f(False, False)],
             ["listedUnaddressed", "unlistedAddressed", "unlistedUnaddressed"], hint="ch48.admission_cell.hint"),
    ],
    "gate": [
        Case("ch48.gate.asked", lambda f: f(True, True, "ignore", False, "off"), "explicit", hint="ch48.gate.hint"),
        Case("ch48.gate.cell", lambda f: f(False, True, "explicit", False, "off"), "explicit", hint="ch48.gate.hint"),
        Case("ch48.gate.followup", lambda f: [f(True, False, "ignore", True, "off"), f(False, False, "ignore", True, "observe")], ["explicit", "inferred"], hint="ch48.gate.hint"),
        Case("ch48.gate.off", lambda f: [f(False, False, "inferred", True, "off"), f(False, False, "inferred", False, "auto")], ["not admitted", "inferred"], hint="ch48.gate.hint"),
        Case("ch48.gate.stranger", lambda f: f(False, True, "ignore", False, "auto"), "not admitted", hint="ch48.gate.hint"),
    ],
    "outcome": [
        Case("ch48.outcome.passes", lambda f: [f("ignore", ["copy"], "explicit", "auto"), f("stop", [], "explicit", "off")], ["ignore", "stop"], hint="ch48.outcome.hint"),
        Case("ch48.outcome.norule", lambda f: f("accept", [], "explicit", "auto"), "ignore", hint="ch48.outcome.hint"),
        Case("ch48.outcome.observe", lambda f: f("accept", ["copy"], "inferred", "observe"), "would accept", hint="ch48.outcome.hint"),
        Case("ch48.outcome.accept", lambda f: [f("accept", ["copy"], "inferred", "auto"), f("accept", ["copy"], "explicit", "observe")], ["accept", "accept"], hint="ch48.outcome.hint"),
    ],
    "on_deadline": [
        Case("ch48.on_deadline.twice", lambda f: [f(0), f(1)], ["requeue", "requeue"], hint="ch48.on_deadline.hint"),
        Case("ch48.on_deadline.third", lambda f: f(2), "unavailable", hint="ch48.on_deadline.hint"),
    ],
}
