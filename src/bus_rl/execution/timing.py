"""Optional function timers. Disabled by default; diagnosis enables them."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter


class Timers:
    def __init__(self) -> None:
        self.enabled = False
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        self.totals.clear()
        self.counts.clear()

    @contextmanager
    def span(self, name: str):
        if not self.enabled:
            yield
            return
        started = perf_counter()
        try:
            yield
        finally:
            self.totals[name] += perf_counter() - started
            self.counts[name] += 1

    def snapshot(self) -> list[dict]:
        rows = []
        for name in sorted(self.totals, key=self.totals.get, reverse=True):
            total = self.totals[name]
            count = self.counts[name]
            rows.append(
                {
                    "name": name,
                    "seconds": total,
                    "calls": count,
                    "us_per_call": (total / count) * 1e6 if count else 0.0,
                }
            )
        return rows


TIMERS = Timers()
