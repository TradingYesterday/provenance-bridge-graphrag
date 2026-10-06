"""Evidence metrics are deferred to the evaluation batch."""

STATUS = "not_implemented"


def unavailable() -> None:
    raise NotImplementedError("evidence metrics are not part of batch 1")
