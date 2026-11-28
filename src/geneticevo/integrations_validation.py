"""Checks for environments with mutable evaluation statistics."""


def reject_normalization(env):
    current = env
    while current is not None:
        if type(current).__name__ in ("NormalizeObservation", "NormalizeReward"):
            raise ValueError("Normalization wrappers require a supported frozen-statistics adapter")
        current = getattr(current, "env", None)

