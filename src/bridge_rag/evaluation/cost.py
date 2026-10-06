"""Cost curves require measured API calls. Batch 1 records logical budget only."""

STATUS = "not_implemented"


def unavailable() -> None:
    raise NotImplementedError("API cost curves are not part of batch 1")
