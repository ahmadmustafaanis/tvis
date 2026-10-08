"""Timing marks with overhead accounting.

A :class:`Mark` is a point in time on the device timeline. On CUDA it is a ``torch.cuda.Event``
recorded on the current stream (GPU-accurate; kernels are async so host clocks would be wrong);
on CPU it is ``time.perf_counter`` (CPU execution is synchronous, so host time is device time).

tvis does its own work (cloning tensors, computing stats) in the middle of the user's calls. That
work is bracketed as an *overhead region*; :meth:`Timer.duration` subtracts every overhead region
that lies inside the measured interval, so a parent call's time doesn't include our capture cost.
"""

from __future__ import annotations

import bisect
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, eq=False)
class Mark:
    seq: int
    wall: float
    event: Any = None
    _ms: float | None = field(default=None, repr=False)


class Timer:
    def __init__(self, record_cuda: bool = False):
        # Events are recorded whenever CUDA is available; whether they are *used* is decided once the
        # model's device is known (a CPU model on a GPU machine must be timed with host clocks).
        self.record_cuda = record_cuda
        self.use_cuda = record_cuda
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._regions: list[tuple[Mark, Mark]] = []  # disjoint, ordered by start seq
        self._region_starts: list[int] = []
        self._region_prefix: list[float] = [0.0]
        self._resolved_upto = 0
        self.base = self.mark()

    @property
    def kind(self) -> str:
        return "cuda_event" if self.use_cuda else "perf_counter"

    def mark(self) -> Mark:
        event = None
        if self.record_cuda:
            import torch

            event = torch.cuda.Event(enable_timing=True)
            event.record()
        with self._lock:
            seq = next(self._seq)
        return Mark(seq=seq, wall=time.perf_counter(), event=event)

    def add_overhead(self, start: Mark, end: Mark) -> None:
        with self._lock:
            self._regions.append((start, end))

    def synchronize(self) -> None:
        if self.record_cuda:
            import torch

            torch.cuda.synchronize()

    def ms(self, mark: Mark) -> float:
        """Milliseconds since the timer was created, on the device timeline."""
        if mark._ms is None:
            if self.use_cuda and mark.event is not None:
                mark._ms = float(self.base.event.elapsed_time(mark.event))
            else:
                mark._ms = (mark.wall - self.base.wall) * 1000.0
        return mark._ms

    def wall_ms(self, mark: Mark) -> float:
        return (mark.wall - self.base.wall) * 1000.0

    def duration(self, start: Mark, end: Mark) -> float:
        """Device time between two marks, minus tvis overhead inside the interval (ms, >= 0)."""
        raw = self.ms(end) - self.ms(start)
        return max(0.0, raw - self._overhead_between(start, end))

    def overhead_total(self) -> float:
        self._index_regions()
        return self._region_prefix[-1]

    def _overhead_between(self, start: Mark, end: Mark) -> float:
        self._index_regions()
        lo = bisect.bisect_right(self._region_starts, start.seq)
        hi = lo
        # regions are disjoint and ordered, so those fully inside (start, end) are contiguous
        while hi < len(self._regions) and self._regions[hi][1].seq < end.seq:
            hi += 1
        return self._region_prefix[hi] - self._region_prefix[lo]

    def _index_regions(self) -> None:
        with self._lock:
            if self._resolved_upto == len(self._regions):
                return
            self._regions.sort(key=lambda r: r[0].seq)
            self._region_starts = [r[0].seq for r in self._regions]
            prefix = [0.0]
            for s, e in self._regions:
                prefix.append(prefix[-1] + max(0.0, self.ms(e) - self.ms(s)))
            self._region_prefix = prefix
            self._resolved_upto = len(self._regions)
