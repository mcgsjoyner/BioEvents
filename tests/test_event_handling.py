import tempfile
from enum import IntEnum
from pathlib import Path

import numpy as np
import pytest

from bioevents import event_handling, hypnogram

from .utils import simulate_hypnogram


def test_overlap_tolerance_setattr():
    illegal_param = -1
    # test setattr via init
    with pytest.raises(AssertionError):
        event_handling.OverlapTolerances(diff_on=illegal_param)
    tol = event_handling.OverlapTolerances()
    with pytest.raises(AssertionError):
        tol.diff_on = illegal_param


def test_event_init_on_only():
    on = 20
    event = event_handling.Event(on=on)
    assert event.on == on
    assert event.duration == event.default_duration
    assert event.off == on + event.default_duration


def test_event_init():
    on, off = 20, 35
    event = event_handling.Event(on=on, off=off)
    assert event.on == on
    assert event.off == off
    assert event.duration == off - on


def test_event_init_invalid():
    off = 10
    on_after_off = 12
    on_negative = -1
    for on in [on_after_off, on_negative]:
        with pytest.raises(AssertionError):
            event_handling.Event(on=on, off=off)


def test_event_from_duration():
    on, duration = 20, 15
    event = event_handling.Event.from_duration(on=on, duration=duration)
    assert event.on == on
    assert event.off == on + duration
    assert event.duration == duration


def test_event_abuts():
    event_a = event_handling.Event(on=10, off=20)
    assert event_handling.Event(on=20, off=30).abuts(event_a)
    assert event_handling.Event(on=5, off=10).abuts(event_a)
    assert not event_handling.Event(on=5, off=10.5).abuts(event_a)
    assert not event_handling.Event(on=19, off=30).abuts(event_a)
    assert not event_handling.Event(on=12, off=18).abuts(event_a)
    assert not event_handling.Event(on=8, off=22).abuts(event_a)
    assert not event_handling.Event(on=3, off=8).abuts(event_a)
    assert not event_handling.Event(on=22, off=29).abuts(event_a)


def test_event_overlaps_default():
    event_a = event_handling.Event(on=10, off=20)

    # test clearly overlapping events overlap under default conditions
    event_b = event_handling.Event(on=15, off=30)
    assert event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)

    # test directly adjacent events overlap under default conditions
    event_b = event_handling.Event(on=20, off=30)
    assert event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)

    # test totally separate events do not overlap
    event_b = event_handling.Event(on=21, off=30)
    assert not event_a.overlaps(event_b)
    assert not event_b.overlaps(event_a)

    # test event fitting inside other event will overlap
    event_b = event_handling.Event(on=12, off=18)
    assert event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)

    # test identical events overlap
    event_b = event_a.__copy__()
    assert event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)


def test_event_overlaps_diffs():
    event_a = event_handling.Event(on=10, off=20)
    event_b = event_handling.Event(on=12, off=22)

    # test events overlap with acceptable on diff
    tol = event_handling.OverlapTolerances(diff_on=5)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events overlap with acceptable off diff
    tol = event_handling.OverlapTolerances(diff_off=5)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events overlap with maximum acceptable on diff
    tol = event_handling.OverlapTolerances(diff_on=2)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events overlap with maximum acceptable off diff
    tol = event_handling.OverlapTolerances(diff_off=2)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events do not overlap with unacceptable on diff
    tol = event_handling.OverlapTolerances(diff_on=1)
    event_a.tolerances = tol
    assert not event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)  # event_b should be unaffected by updated tolerances

    # test events do not overlap with unacceptable off diff
    tol = event_handling.OverlapTolerances(diff_off=1)
    event_a.tolerances = tol
    assert not event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)  # event_b should be unaffected by updated tolerances


def test_event_overlaps_ratios():
    event_a = event_handling.Event(on=10, off=20)
    event_b = event_handling.Event(on=12, off=22)

    # test events overlap with acceptable on ratio
    tol = event_handling.OverlapTolerances(ratio_on=0.5)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events overlap with acceptable off ratio
    tol = event_handling.OverlapTolerances(ratio_off=0.5)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events overlap with maximum acceptable on ratio
    tol = event_handling.OverlapTolerances(ratio_on=0.2)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events overlap with maximum acceptable off ratio
    tol = event_handling.OverlapTolerances(ratio_off=0.2)
    event_a.tolerances = tol
    assert event_a.overlaps(event_b)

    # test events do not overlap with unacceptable on ratio
    tol = event_handling.OverlapTolerances(ratio_on=0.1)
    event_a.tolerances = tol
    assert not event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)  # event_b should be unaffected by updated tolerances

    # test events do not overlap with unacceptable off ratio
    tol = event_handling.OverlapTolerances(ratio_off=0.1)
    event_a.tolerances = tol
    assert not event_a.overlaps(event_b)
    assert event_b.overlaps(event_a)  # event_b should be unaffected by updated tolerances


@pytest.mark.parametrize(
    "on, bools",
    [
        (2, [False, False, True, True, False]),  # event is contained within series
        (2, [False, False, True, True]),  # event is clipped by series end
        (0, [True, True, False]),  # event is clipped by series start
    ],
)
def test_event_series_from_bools_clipped(bools, on):
    dut = event_handling.EventSeries.from_bools(bools)

    # we're expecting one event
    assert len(dut) == 1

    # check the interpreted timestamps
    event = dut[0]
    assert event.on == on
    assert event.off == on + 2


@pytest.mark.parametrize(
    "on, off, duration, ref",
    [
        (2, 4, 5, [False, False, True, True, False]),  # event is contained within series
        (2, 4, 4, [False, False, True, True]),  # event is clipped by series end
        (0, 2, 3, [True, True, False]),  # event is clipped by series start
    ],
)
def test_event_series_as_bools_clipped(on, off, duration, ref):
    series = event_handling.EventSeries(events=[event_handling.Event(on, off)], duration=duration)
    dut = series.as_bools()
    np.testing.assert_equal(ref, dut)


def test_event_series_from_bools():
    """Define two events in boolean form and make sure they're ingested correctly as events"""
    bools = [False, False, True, True, False, True, False, False]
    series = event_handling.EventSeries.from_bools(bools)
    assert len(series) == 2
    assert series[0].on == 2
    assert series[0].off == 4
    assert series[1].on == 5
    assert series[1].off == 6


def test_event_series_as_bools():
    """Define two events in Event form and make sure they're exported correctly in boolean form"""
    ref = [False, False, True, True, False, True, False, False]
    events = [
        event_handling.Event(on=2, off=4),
        event_handling.Event(on=5, off=6),
    ]
    series = event_handling.EventSeries(events, duration=len(ref))
    dut = series.as_bools()
    np.testing.assert_equal(ref, dut)


@pytest.mark.parametrize("sampling_rate", [0.5, 1.0, 10.0])
def test_event_series_resample(sampling_rate):
    original_sampling_rate = 1.0
    # make a dummy event array and resample it
    ref = event_handling.EventSeries(
        [
            event_handling.Event(0, 1),
            event_handling.Event(3, 6),
            event_handling.Event(7.5, 8.5),
            event_handling.Event(9, 11),
            event_handling.Event(10, 12),
            event_handling.Event(14, 20),
        ],
        duration=30,
        sampling_rate=original_sampling_rate,
    )
    dut = ref.resample(sampling_rate=sampling_rate)

    # make sure the sample duration is adjusted correctly
    np.testing.assert_allclose(dut.duration / sampling_rate, ref.duration)

    # make sure we end up with the same number of events
    assert len(dut) == len(ref)

    # make sure each timestamp is accurately resampled
    for d, r in zip(dut, ref):
        assert d.on / sampling_rate == r.on

    # resample back to original rate
    dut = dut.resample(sampling_rate=original_sampling_rate)

    # make sure the sample duration is adjusted correctly
    np.testing.assert_allclose(dut.duration / original_sampling_rate, ref.duration)


@pytest.mark.parametrize("start", [0, 1, 1.5])
def test_event_series_trim(start):
    # make a dummy event array
    series = event_handling.EventSeries(
        [
            event_handling.Event(
                0, 1
            ),  # this event should be eliminated entirely unless start == 0
            event_handling.Event(3, 6),  # this event should remain the same
            event_handling.Event(7.5, 8.5),  # just testing float values
            event_handling.Event(9, 11),  # this event should be cut in half
            event_handling.Event(10, 12),  # this event should be eliminated entirely
            event_handling.Event(14, 20),  # this event should be eliminated entirely
        ],
        duration=30,
    )

    # make sure we raise an error with parameters out of bounds
    with pytest.raises(ValueError):
        series.trim(-1, 10)
    with pytest.raises(ValueError):
        series.trim(10, 40)

    # trim our boolean array
    expected = event_handling.EventSeries(
        [
            event_handling.Event(3 - start, 6 - start),
            event_handling.Event(7.5 - start, 8.5 - start),
            event_handling.Event(9 - start, 10 - start),
        ],
        duration=10 - start,
    )
    if start < 1:
        expected.append(event_handling.Event(0, 1))
    expected._resolve_events()
    actual = series.trim(start, 10)

    assert actual.duration == expected.duration
    assert len(actual) == len(expected)

    for act, exp in zip(actual, expected):
        assert act == exp


@pytest.mark.parametrize(
    "duration, events_original, events_expected",
    [
        (10, [], [event_handling.Event(0, 10)]),
        (
            10,
            [event_handling.Event(0.5, 2), event_handling.Event(4, 5)],
            [
                event_handling.Event(0, 0.5),
                event_handling.Event(2, 4),
                event_handling.Event(5, 10),
            ],
        ),
        (
            10,
            [event_handling.Event(0, 2), event_handling.Event(4, 5)],
            [event_handling.Event(2, 4), event_handling.Event(5, 10)],
        ),
        (
            5,
            [event_handling.Event(0.5, 2), event_handling.Event(4, 5)],
            [event_handling.Event(0, 0.5), event_handling.Event(2, 4)],
        ),
        (
            10,
            [event_handling.Event(0.5, 2)],
            [event_handling.Event(0, 0.5), event_handling.Event(2, 10)],
        ),
    ],
)
def test_event_series_invert(duration, events_original, events_expected):
    original = event_handling.EventSeries(events=events_original, duration=duration)
    expected = event_handling.EventSeries(events=events_expected, duration=duration)
    inverted = ~original
    assert inverted.duration == original.duration
    assert inverted == expected


def test_event_class_series_missing_enum():
    class MyIntEnum(IntEnum):
        A = 0
        B = 1
        C = 2

    event_stack = {
        MyIntEnum.A: event_handling.EventSeries.from_bools([False, True, False]),
        MyIntEnum.B: event_handling.EventSeries.from_bools([True, False, False]),
    }

    # make sure the missing bin is rejected if no missing enum arg is given
    with pytest.raises(AssertionError):
        event_handling.EventClassSeries(event_stack)

    # make sure the missing bin is filled with the missing enum arg given
    dut = event_handling.EventClassSeries(event_stack, missing_enum=MyIntEnum.C)
    assert dut.as_array() == [MyIntEnum.B, MyIntEnum.A, MyIntEnum.C]


def test_write_read_json():
    ref = simulate_hypnogram(series_duration=100)
    ref = ref.resample(2)
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "temp.json"
        ref.write_json(fp)
        dut = event_handling.EventClassSeries.read_json(fp, hypnogram.SleepStages)

    assert ref.classes == dut.classes
    assert ref.duration == dut.duration
    assert ref.sampling_rate == dut.sampling_rate
    assert ref.as_array() == dut.as_array()


@pytest.mark.parametrize(
    "on, duration, expected",
    [
        (4, 10, True),  # this is the only valid one
        (4, 20, False),  # no events for 10-20
        (3, 10, False),
        (3.9, 10, False),
        (4.1, 10, False),
        (5, 10, False),
    ],
)
def test_event_stack_is_continuous_and_flat(on, duration, expected):
    func = event_handling.EventClassSeries._is_continuous_and_flat

    # note that events are not in order -- this shouldn't be a problem...
    events_a = event_handling.EventSeries(
        events=[event_handling.Event(3, 4), event_handling.Event(1, 3)], duration=duration
    )
    events_b = event_handling.EventSeries(
        events=[event_handling.Event(on, 6), event_handling.Event(6, 10)], duration=duration
    )
    stack = {"a": events_a, "b": events_b}

    # this stack doesn't start at 0 -- this SHOULD be a problem
    assert not func(stack.values())

    # add an event at 0 to make it continuous
    stack["c"] = event_handling.EventSeries([event_handling.Event(0, 1)], duration=duration)
    assert func(stack.values()) == expected


def test_event_stack_from_string_keys():
    event_stack = {
        "b": event_handling.EventSeries.from_bools([False, True, True, False]),
        "a": event_handling.EventSeries.from_bools([False, True, True, False]),
    }
    dut = event_handling.EventStack.from_string_keys(event_stack)

    # make sure an IntEnum subclass was assigned
    assert all(isinstance(k, IntEnum) for k in dut)

    # make sure elements were assigned to a zero-indexed enum
    assigned_enum = type(list(dut.keys())[0])
    assert assigned_enum["b"] == 0
    assert assigned_enum.__name__ == event_handling.EventStack.KEY_ASSIGNED_ENUM


def test_event_series_append():
    # make a dummy boolean event array
    events = [
        event_handling.Event(0, 1),
        event_handling.Event(3, 6),
        event_handling.Event(7.5, 8.5),
        event_handling.Event(9, 11),
        event_handling.Event(10, 12),
        event_handling.Event(14, 20),
    ]
    duration = 30

    # test appending each individual event in turn
    ref = event_handling.EventSeries(events, duration)
    for event in events:
        dut = event_handling.EventSeries(
            events=[ev for ev in events if ev != event], duration=duration
        )
        dut.append(event)
        assert len(ref) == len(dut), "Reference and test have different number of events!"

        # note: this will also check the order of the events!
        assert ref == dut, "Reference events do not match test events!"


def test_event_series_extend():
    # make a dummy boolean event array
    events = [
        event_handling.Event(0, 1),
        event_handling.Event(7.5, 8.5),
        event_handling.Event(9, 11),
        event_handling.Event(10, 12),
    ]
    duration = 30

    extend_events = [
        event_handling.Event(3, 6),
        event_handling.Event(14, 20),
    ]

    # test appending each individual event in turn
    ref = event_handling.EventSeries(events + extend_events, duration)
    dut = event_handling.EventSeries(events=events, duration=duration)
    dut.extend(extend_events)
    assert len(ref) == len(dut), "Reference and test have different number of events!"

    # note: this will also check the order of the events!
    assert ref == dut, "Reference events do not match test events!"


def test_event_series_debounce():
    # metadata we need to preserve
    duration = 30
    sampling_rate = 2

    # debouncing parameters
    persistence_period = 1.5
    min_duration = 3.5

    # make a dummy boolean event array
    events = [
        event_handling.Event(0, 3.5),
        event_handling.Event(5.5, 6),  # too short -- should be pruned away
        event_handling.Event(8, 9),  # won't duration test, but should be merged w following
        event_handling.Event(10.25, 15),
    ]

    # make the expected debounced events based on the parameters above
    events_debounced = [
        event_handling.Event(0, 3.5),
        event_handling.Event(8, 15),
    ]

    original = event_handling.EventSeries(events, duration=duration, sampling_rate=sampling_rate)
    expected = event_handling.EventSeries(
        events_debounced, duration=duration, sampling_rate=sampling_rate
    )
    actual = original.debounce(persistence_period=persistence_period, min_duration=min_duration)
    assert actual.sampling_rate == expected.sampling_rate
    assert actual.duration == expected.duration
    assert actual == expected
