"""Answer metrics are deferred. Do not treat a diagnostic answer as a QA score."""

STATUS = "not_implemented"


def unavailable() -> None:
    raise NotImplementedError("answer EM/F1 is not part of batch 1")
