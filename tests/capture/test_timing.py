from __future__ import annotations

import time

import pytest

from tvis.capture.timing import Timer


def spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def test_duration_measures_elapsed_time_between_marks():
    timer = Timer()
    start = timer.mark()
    spin(0.01)
    end = timer.mark()

    assert timer.duration(start, end) == pytest.approx(10.0, abs=5.0)


def test_overhead_inside_the_interval_is_subtracted():
    timer = Timer()
    start = timer.mark()
    spin(0.005)
    overhead_start = timer.mark()
    spin(0.02)
    timer.add_overhead(overhead_start, timer.mark())
    end = timer.mark()

    raw = timer.ms(end) - timer.ms(start)
    assert raw > 20
    assert timer.duration(start, end) == pytest.approx(raw - 20, abs=4.0)


def test_overhead_outside_the_interval_is_not_subtracted():
    timer = Timer()
    overhead_start = timer.mark()
    spin(0.01)
    timer.add_overhead(overhead_start, timer.mark())
    start = timer.mark()
    spin(0.005)
    end = timer.mark()

    assert timer.duration(start, end) == pytest.approx(timer.ms(end) - timer.ms(start))


def test_overhead_only_partially_inside_the_interval_is_not_subtracted():
    # A parent call can't own work that started before it; only fully-contained regions count.
    timer = Timer()
    overhead_start = timer.mark()
    start = timer.mark()
    spin(0.005)
    timer.add_overhead(overhead_start, timer.mark())
    end = timer.mark()

    assert timer.duration(start, end) == pytest.approx(timer.ms(end) - timer.ms(start))


def test_multiple_regions_are_all_subtracted_and_regions_added_out_of_order_are_handled():
    timer = Timer()
    start = timer.mark()
    a0 = timer.mark()
    spin(0.004)
    a1 = timer.mark()
    b0 = timer.mark()
    spin(0.004)
    b1 = timer.mark()
    end = timer.mark()
    timer.add_overhead(b0, b1)  # registered out of order (e.g. from another thread)
    timer.add_overhead(a0, a1)

    expected = (
        (timer.ms(end) - timer.ms(start)) - (timer.ms(a1) - timer.ms(a0)) - (timer.ms(b1) - timer.ms(b0))
    )
    assert timer.duration(start, end) == pytest.approx(expected)
    assert timer.overhead_total() == pytest.approx(
        (timer.ms(a1) - timer.ms(a0)) + (timer.ms(b1) - timer.ms(b0))
    )


def test_duration_is_never_negative():
    timer = Timer()
    start = timer.mark()
    end = timer.mark()
    timer.add_overhead(start, end)  # degenerate: everything is overhead

    assert timer.duration(start, end) >= 0.0


def test_cpu_timer_reports_its_clock_kind():
    assert Timer(record_cuda=False).kind == "perf_counter"


def test_corrected_positions_remove_all_earlier_overhead():
    timer = Timer()
    a = timer.mark()
    o0 = timer.mark()
    spin(0.01)
    o1 = timer.mark()
    timer.add_overhead(o0, o1)
    b = timer.mark()

    assert timer.corrected_ms(a) == pytest.approx(timer.ms(a))
    assert timer.corrected_ms(b) == pytest.approx(timer.ms(b) - (timer.ms(o1) - timer.ms(o0)))
    assert timer.corrected_ms(b) - timer.corrected_ms(a) == pytest.approx(timer.duration(a, b))
