"""Real observation clocks that do not change AppWorld's frozen business time.

Freezegun patches time.time/monotonic/perf_counter and aliases to those functions.
Resolve OS clocks on every call. The Windows path dynamically reads freezegun's
preserved real functions when that library is loaded, without retaining a
patchable function alias. No task-facing datetime or clock policy is changed.
"""

import sys
import time as time_module

CLOCK_SOURCE = (
    "clock_gettime(CLOCK_MONOTONIC/CLOCK_REALTIME)"
    if hasattr(time_module, "clock_gettime")
    else "native clocks; dynamic freezegun real_* when loaded"
)


def perf_counter() -> float:
    """Return real elapsed seconds, including inside a frozen AppWorld task."""
    if hasattr(time_module, "clock_gettime"):
        return time_module.clock_gettime(time_module.CLOCK_MONOTONIC)
    frozen_api = sys.modules.get("freezegun.api")
    return frozen_api.real_perf_counter() if frozen_api is not None else time_module.perf_counter()


def monotonic() -> float:
    """Return the same real monotonic clock used by observation durations."""
    return perf_counter()


def time() -> float:
    """Return real UTC epoch seconds for correlating independent telemetry."""
    if hasattr(time_module, "clock_gettime"):
        return time_module.clock_gettime(time_module.CLOCK_REALTIME)
    frozen_api = sys.modules.get("freezegun.api")
    return frozen_api.real_time() if frozen_api is not None else time_module.time()
