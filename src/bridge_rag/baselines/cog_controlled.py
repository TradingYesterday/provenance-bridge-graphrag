"""CoG controlled baseline. Not available in batch 1."""


class BaselineUnavailable(RuntimeError):
    pass


def run(*_args, **_kwargs):
    raise BaselineUnavailable(
        "cog_controlled is not implemented. Batch 1 only runs the oracle branch engine. "
        "This is a controlled adaptation target, not a claim that the original CoG global-KG setup is reproduced."
    )
