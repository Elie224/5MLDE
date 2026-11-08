from retail.orchestration.errors import ReplayDone, ReplayError

__all__ = ["ReplayDone", "ReplayError", "run_tick"]


def __getattr__(name: str):
    if name == "run_tick":
        from retail.orchestration.replay import run_tick

        return run_tick
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
