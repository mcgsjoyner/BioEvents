#!/usr/bin/env python

import tempfile
from pathlib import Path

import pytest

from bioevents import event_handling, hypnogram

from . import utils as test_utils


@pytest.fixture(scope="function")
def dummy_hypnogram():
    return test_utils.simulate_hypnogram(series_duration=100, num_cycles=5, avg_duration=5)


def test_sleep_metrics():
    stages = hypnogram.SleepStages

    # create a dummy hypnogram
    hypno = hypnogram.Hypnogram.from_array(
        [stages.W] * 90
        + [stages.N1] * 30
        + [stages.W] * 30
        + [stages.N1] * 30
        + [stages.N2] * 60
        + [stages.N3] * 90
        + [stages.REM] * 30
        + [stages.W] * 60
    )
    # ensure the total sleep time is the sum of total non-wake stages in minutes
    assert hypno.total_recording_time() == 420
    assert hypno.total_sleep_time() == 240
    assert hypno.sleep_efficiency_percent() == (240 / 420) * 100
    assert hypno.wake_after_sleep_onset() == 90

    # test functions given bedtime_series
    bedtime_series = event_handling.EventSeries(
        events=[event_handling.Event(30, 390)],
        duration=hypno.duration,
    )
    assert hypno.total_recording_time(bedtime_series=bedtime_series) == 360
    assert hypno.total_sleep_time(bedtime_series=bedtime_series) == 240
    assert hypno.sleep_efficiency_percent(bedtime_series=bedtime_series) == 240 / 360 * 100


def test_time_in_stage():
    stages = [hypnogram.SleepStages.W] * 60 * 20
    stages.extend([hypnogram.SleepStages.N1] * 60 * 10)
    stages.extend([hypnogram.SleepStages.REM] * 60 * 10)
    stages.extend([hypnogram.SleepStages.W] * 60 * 60)
    hypno = hypnogram.Hypnogram.from_array(stages)
    assert hypno.time_in_stage(hypnogram.SleepStages.N1) == 60 * 10
    assert hypno.time_in_stage(hypnogram.SleepStages.REM) == 60 * 10
    assert hypno.time_in_stage(hypnogram.SleepStages.N2) == 0
    assert hypno.time_in_stage(hypnogram.SleepStages.N3) == 0


@pytest.mark.parametrize("n1_tolerance", [0, 60])
def test_sleep_onset_latency(n1_tolerance):
    # TODO refactor this test to multiple atomic ones
    stages = hypnogram.SleepStages

    # define two N1 events that we expect to treat differently when excluding sub-threshold N1 activity
    n1_exclude, n1_include = [stages.N1] * 60, [stages.N1] * 90

    # create a dummy hypnogram
    hypno = hypnogram.Hypnogram.from_array(
        [stages.W] * 120 + n1_exclude + [stages.W] * 30 + n1_include + [stages.W] * 90
    )

    # make sure the first N1 detection is flagged when including all N1 activity by default
    # make sure the first N1 detection is ignored when excluding short N1 activity
    ref = 210 if n1_tolerance else 120
    dut = hypno.sleep_onset_latency(n1_tolerance)
    assert dut == ref

    # create a dummy hypnogram with a non-N1 sleep stage
    hypno = hypnogram.Hypnogram.from_array(
        [stages.W] * 120
        + n1_exclude
        + [stages.W] * 30
        + n1_exclude
        + [stages.N2] * 30
        + [stages.W] * 90
    )

    # make sure the first N1 detection is flagged when including all N1 activity by default
    # with tolerance, make sure the second, short N1 is actually included if followed by a non-N1 sleep stage
    ref = 210 if n1_tolerance else 120
    dut = hypno.sleep_onset_latency(n1_tolerance)
    assert dut == ref

    # make sure that, if resampled, the result is consistent with the new units
    sampling_rate = 1 / 30
    resampled = hypno.resample(sampling_rate=sampling_rate)
    dut = resampled.sleep_onset_latency(n1_tolerance=n1_tolerance * sampling_rate)
    assert dut == ref * sampling_rate

    # create a dummy hypnogram with no sleep
    hypno = hypnogram.Hypnogram.from_array([stages.W] * 120)

    # make sure we get None if there is no sleep whatsoever
    ref = None
    dut = hypno.sleep_onset_latency(n1_tolerance)
    assert dut == ref

    hypno = hypnogram.Hypnogram.from_array([stages.W] * 120 + n1_exclude + [stages.W] * 30)
    # make sure we get None if there is sub-threshold N1
    ref = None if n1_tolerance else 120
    dut = hypno.sleep_onset_latency(n1_tolerance)
    assert dut == ref


def test_hypnogram_waso():
    # a hypnogram with sleep should have float-valued WASO
    wake_epochs = [hypnogram.SleepStages.W] * 10
    sleep_epochs = [hypnogram.SleepStages.REM] * 10
    dummy_hypnogram = hypnogram.Hypnogram.from_array(wake_epochs + sleep_epochs)
    WASO = dummy_hypnogram.wake_after_sleep_onset()
    assert isinstance(WASO, float)

    # a hypnogram with no sleep should have no WASO (i.e., None)
    dummy_hypnogram = hypnogram.Hypnogram.from_array(wake_epochs)
    WASO = dummy_hypnogram.wake_after_sleep_onset()
    assert WASO is None


@pytest.mark.parametrize("with_disagreement", [False, True])
def test_hypnogram_read_write_json(dummy_hypnogram, with_disagreement):
    if with_disagreement:
        # knock out a few samples
        arr = dummy_hypnogram.as_array()
        arr[5:9] = [hypnogram.Labels.DISAGREE] * 3
        dummy_hypnogram = hypnogram.Hypnogram.from_array(arr)
    with tempfile.TemporaryDirectory() as td:
        tf = Path(td) / "hypnogram.json"
        dummy_hypnogram.write_json(tf)
        dut = hypnogram.Hypnogram.read_json(tf)
    assert dut.duration == dummy_hypnogram.duration
    assert dut.classes == dummy_hypnogram.classes
    # note: this will iteratively compare the Events, which are in a list which is in this dict
    assert dut == dummy_hypnogram


def test_latency_rem_no_sleep():
    num_seconds = 60 * 100
    stages = [hypnogram.SleepStages.W] * num_seconds
    hypno = hypnogram.Hypnogram.from_array(stages)
    assert hypno.latency_to_stage(hypnogram.SleepStages.REM) is None


def test_latency_rem_no_rem():
    # simulate 100 minutes of sleep with persistent sleep but no REM
    stages = [hypnogram.SleepStages.W] * 60 * 20
    stages.extend([hypnogram.SleepStages.N1] * 60 * 10)
    stages.extend([hypnogram.SleepStages.N2] * 60 * 10)
    stages.extend([hypnogram.SleepStages.W] * 60 * 60)
    hypno = hypnogram.Hypnogram.from_array(stages)

    assert hypno.latency_to_stage(hypnogram.SleepStages.REM) is None


def test_latency_to_stage():
    stages = [hypnogram.SleepStages.W] * 20
    stages.extend([hypnogram.SleepStages.N1] * 10)
    stages.extend([hypnogram.SleepStages.N2] * 10)
    stages.extend([hypnogram.SleepStages.REM] * 10)
    hypno = hypnogram.Hypnogram.from_array(stages)
    assert hypno.latency_to_stage(hypnogram.SleepStages.N1) == 20
    assert hypno.latency_to_stage(hypnogram.SleepStages.N2) == 30
    assert hypno.latency_to_stage(hypnogram.SleepStages.REM) == 40
    assert hypno.latency_to_stage(hypnogram.SleepStages.N3) is None


def test_hypnogram_plot_no_errors(dummy_hypnogram):
    """A rudimentary check that we hit no errors when building a Hypnogram plot"""
    dummy_hypnogram.plot()


def test_hypnogram_report_no_errors(dummy_hypnogram):
    report = dummy_hypnogram.generate_report()
