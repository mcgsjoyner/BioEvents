import abc
import datetime
from enum import IntEnum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    confloat,
    root_validator,
)

from bioevents import event_handling

WAKE_PERSISTENCE_THRESHOLD_IN_MINUTES = 2


class SleepStages(IntEnum):
    W = 5
    REM = 4
    N1 = 3
    N2 = 2
    N3 = 1


class Labels(IntEnum):
    W = SleepStages.W.value
    REM = SleepStages.REM.value
    N1 = SleepStages.N1.value
    N2 = SleepStages.N2.value
    N3 = SleepStages.N3.value
    MISSING = event_handling.MissingDataLabels.MISSING.value
    UNSCORABLE = event_handling.MissingDataLabels.UNSCORABLE.value
    ARTIFACT = event_handling.MissingDataLabels.ARTIFACT.value
    DISAGREE = event_handling.MissingDataLabels.DISAGREE.value


class HypnogramReport(BaseModel, abc.ABC):
    """Used for the EDS reports"""

    duration_minutes: NonNegativeFloat
    latency_minutes_rem: Optional[NonNegativeFloat] = None
    latency_minutes_n1: Optional[NonNegativeFloat] = None
    latency_minutes_n2: Optional[NonNegativeFloat] = None
    latency_minutes_n3: Optional[NonNegativeFloat] = None
    latency_minutes_persistent_sleep: Optional[NonNegativeFloat] = None
    latency_minutes_rem_with_persistence: Optional[NonNegativeFloat] = None
    latency_minutes_sleep_onset: Optional[NonNegativeFloat] = None
    latency_minutes_sleep_onset_unequivocal: Optional[NonNegativeFloat] = None
    total_minutes_rem: NonNegativeFloat
    total_minutes_n1: NonNegativeFloat
    total_minutes_n2: NonNegativeFloat
    total_minutes_n3: NonNegativeFloat
    total_minutes_nrem: NonNegativeFloat
    total_minutes_sleep: NonNegativeFloat
    efficiency_rem: confloat(ge=0, le=1)
    efficiency_rem_first_two_hours: confloat(ge=0, le=1)
    efficiency_sleep: confloat(ge=0, le=1)
    event_count_rem: NonNegativeInt
    event_count_wake: NonNegativeInt
    event_count_wake_persistent: NonNegativeInt
    wake_after_sleep_onset_minutes: Optional[NonNegativeFloat] = None

    def __str__(self):
        rows = [
            f"Latency to REM:                       {self.latency_minutes_rem:.1f} minutes",
            f"Latency to N1:                        {self.latency_minutes_n1:.1f} minutes",
            f"Latency to N2:                        {self.latency_minutes_n2:.1f} minutes",
            f"Latency to N3:                        {self.latency_minutes_n3:.1f} minutes",
            f"Latency to persistent sleep:          {self.latency_minutes_persistent_sleep:.1f} minutes",
            f"Latency to REM from persistent sleep: {self.latency_minutes_rem_with_persistence:.1f} minutes",
            f"Latency to sleep onset (SOL)          {self.latency_minutes_sleep_onset:.1f} minutes",
            f"Latency to sleep onset (SOL-U)        {self.latency_minutes_sleep_onset_unequivocal:.1f} minutes",
            f"Total time in REM:                    {self.total_minutes_rem:.1f} minutes",
            f"Total time in N1:                     {self.total_minutes_n1:.1f} minutes",
            f"Total time in N2:                     {self.total_minutes_n2:.1f} minutes",
            f"Total time in N3:                     {self.total_minutes_n3:.1f} minutes",
            f"Total time in non-REM sleep:          {self.total_minutes_nrem:.1f} minutes",
            f"Total time asleep:                    {self.total_minutes_sleep:.1f} minutes",
            f"Efficiency of REM:                    {100 * self.efficiency_rem:.1f} %",
            f"Efficiency of REM (first two hours):  {100 * self.efficiency_rem_first_two_hours:.1f} %",
            f"Efficiency of sleep:                  {100 * self.efficiency_sleep:.1f} %",
            f"Number of REM cycles:                 {self.event_count_rem}",
            f"Number of awakenings:                 {self.event_count_wake}",
            f"Number of awakenings >120s:           {self.event_count_wake_persistent}",
        ]
        return "\n".join(rows)

    @root_validator(allow_reuse=True)
    def check_total_times_add_up(cls, values):
        rem = values.get("total_minutes_rem")
        n1 = values.get("total_minutes_n1")
        n2 = values.get("total_minutes_n2")
        n3 = values.get("total_minutes_n3")
        nrem = values.get("total_minutes_nrem")
        sleep = values.get("total_minutes_sleep")
        if not np.isclose(n1 + n2 + n3, nrem):
            raise ValueError(
                "Total minutes in non-REM sleep doesn't match individual stage times."
            )
        if not np.isclose(nrem + rem, sleep):
            raise ValueError("Total minutes in sleep doesn't match REM + NREM times.")
        return values

    @root_validator(allow_reuse=True)
    def check_efficiencies(cls, values):
        rem = values.get("efficiency_rem")
        rem_time = values.get("total_minutes_rem")
        sleep = values.get("efficiency_sleep")
        sleep_time = values.get("total_minutes_sleep")
        dur = values.get("duration_minutes")
        if not np.isclose(rem_time / dur, rem):
            raise ValueError("REM efficiency and total time don't match.")
        if not np.isclose(sleep_time / dur, sleep):
            raise ValueError("Sleep efficiency and total time don't match.")
        return values

    @root_validator(allow_reuse=True)
    def check_awakenings(cls, values):
        total = values.get("event_count_wake")
        persistent = values.get("event_count_wake_persistent")
        if total < persistent:
            raise ValueError("Persistent event count is greater than total count!")
        return values

    @root_validator(allow_reuse=True)
    def check_no_rem(cls, values):
        value_if_no_rem = {
            "efficiency_rem": 0,
            "total_minutes_rem": 0,
            "event_count_rem": 0,
            "latency_minutes_rem": None,
        }
        no_rem_indicators = [values[k] == value_if_no_rem[k] for k in value_if_no_rem]
        if not any(no_rem_indicators):
            return values
        if all(no_rem_indicators):
            return values
        raise ValueError("Some, but not all, REM metrics indicate REM is present!")


class Hypnogram(event_handling.EventClassSeries):
    def __init__(self, event_stack):
        super().__init__(event_stack)
        # make sure that we fill in any missing sleep stages with empty event series
        self.update(
            {
                c: event_handling.EventSeries(
                    [], duration=self.duration, sampling_rate=self.sampling_rate
                )
                for c in list(SleepStages)
                if c not in self.classes
            }
        )

    @classmethod
    def read_json(cls, filepath, intenum=Labels):
        return super().read_json(filepath, intenum)

    def total_sleep_time(self, bedtime_series: event_handling.EventSeries = None) -> float:
        """Find total sleep time (in samples) from a hypnogram

        Parameters
        ----------
        bedtime_series : Optional[event_handling.EventSeries]
            An event series derived from clinical lights off/on annotations.
            In the clinical setting, this is the period of time when subjects are expected to attempt sleep.
            If given, only compute sleep time that occurs during bedtime.

        Notes
        -----
        This metric is traditionally reported in minutes. Please take note of the sample_rate to convert if needed.
        """
        if bedtime_series is not None:
            return sum([self.trim(ev.on, ev.off).total_sleep_time() for ev in bedtime_series])
        return self.duration - self.time_in_stage(SleepStages.W)

    def total_recording_time(self, bedtime_series: event_handling.EventSeries = None) -> float:
        """Find total recording time, i.e., total time in bed, from a hypnogram (in samples)

        Parameters
        ----------
        bedtime_series : Optional[event_handling.EventSeries]
            An event series derived from clinical lights off/on annotations.
            In the clinical setting, this is the period of time when subjects are expected to attempt sleep.
            If given, only compute recording time that occurs during bedtime.

        Notes
        -----
        This metric is traditionally reported in minutes. Please take note of the sample_rate to convert if needed.
        """
        if bedtime_series is None:
            return self.duration
        return sum([ev.duration for ev in bedtime_series])

    def sleep_efficiency_percent(self, bedtime_series: event_handling.EventSeries = None) -> float:
        eff = self.total_sleep_time(bedtime_series) / self.total_recording_time(bedtime_series)
        return 100 * eff

    # NOTE: the hypnogram needs to be trimmed
    def sleep_onset_latency(self, n1_tolerance: float = 0) -> Optional[float]:
        """Find sleep onset latency from a hypnogram (in samples). Assume the start of the hypnogram is the start of
        bedtime (light off).

        Parameters
        ----------
        n1_tolerance : float, default 0
            If 0 (default), return latency to sleep onset as the first non-Wake stage (used for overnight sleep)
            If positive, units are samples, and more rules are followed (see Notes)

        Returns
        -------
        sleep_onset_latency : Optional(float)
            Time (in samples) from beginning of this hypnogram until the sleep onset.
            See Notes for more information on the definition of sleep onset.

        Notes
        -----
        This metric is traditionally reported in minutes. Please take note of the sample_rate to convert if needed.
        There are multiple ways to report SOL:

        1. The start of first scored epoch of any stage of sleep.

            Typically used in overnight PSG.

        2. The start of the first epoch of "unequivocal sleep".

            Typically used in MWT for assessment of daytime
            sleepiness. "Unequivocal sleep" is "defined as 3 consecutive epochs of stage 1 sleep,
            or 1 epoch of any other stage of sleep" (see Reference #1). Epochs, in this case, are 30 seconds long.
            Thus, the first epoch of any of the following would be considered unequivocal sleep onset:

                - [N1, N1, N1, ...] three epochs of N1
                - [N1, N1, S, ...] two epochs of N1, followed immediately by any other stage of sleep
                - [N1, S, ...] one epoch of N1, followed immediately by any other stage of sleep
                - [S, ...] one epoch of any other stage of sleep

            For algorithmic purposes, this reduces to the observation that unequivocal sleep occurs at the onset of ANY
            sleep other than short, N1-only sleep episodes lasting less than 90 seconds.

        References
        ----------
        1. Clinical study using SOL as a primary endpoint: https://clinicaltrials.gov/ct2/show/NCT03522506
        """
        asleep = ~self[SleepStages.W]

        if not len(asleep):
            return None

        if n1_tolerance == 0:
            # return onset of first non-wake event
            return asleep[0].on

        # first remove n1 events shorter than the tolerance from the "asleep" time series
        n1 = self[SleepStages.N1]
        n1_exclude = event_handling.EventSeries(
            [ev for ev in n1 if ev.duration <= n1_tolerance],
            duration=self.duration,
            sampling_rate=self.sampling_rate,
        )
        asleep -= n1_exclude

        if not len(asleep):
            return None

        # IF there's ANY N1 event that preceded the first sleep event, we'll use it as the onset
        first_sleep_event = asleep[0]
        for n1_event in n1:
            if n1_event.off == first_sleep_event.on:
                first_sleep_event = n1_event
        return first_sleep_event.on

    def wake_after_sleep_onset(self) -> Optional[float]:
        """Compute WASO (wake after sleep onset) from a hypnogram. Useful for overnight sleep

        Returns
        -------
        WASO : Optional(float)
            If SOL is None, returns None.
            If SOL is not None, returns wake after sleep onset (in samples).

        Notes
        -----
        This metric is traditionally reported in minutes. Please take note of the sample_rate to convert if needed.
        """
        SOL = self.sleep_onset_latency(n1_tolerance=0)
        if SOL is None:
            return None
        TRT = self.total_recording_time()
        TST = self.total_sleep_time()
        WASO = TRT - TST - SOL
        if WASO < -0.1:
            raise RuntimeError("WASO is negative and exceeds precision error. This is a bug.")
        WASO = max(WASO, 0.0)  # avoids precision errors
        return WASO

    def time_in_stage(self, stage: SleepStages) -> float:
        """Total time spent in the given stage, in samples."""
        return sum([ev.duration for ev in self[stage]])

    def latency_to_first_persistent_sleep(self) -> Optional[float]:
        """Latency to first epoch of persistent (10 minutes) sleep.

        Returns
        -------
        latency : Optional(float)
            If no persistent sleep occurs, returns None.
            If persistent sleep occurs, returns latency (in samples).

        Notes
        -----
        This metric is traditionally reported in minutes. Please take note of the sample_rate to convert if needed.
        """
        asleep = ~self[SleepStages.W]

        persistence_threshold_seconds = datetime.timedelta(minutes=10).seconds
        persistence_threshold_samples = persistence_threshold_seconds * asleep.sampling_rate

        persistent_sleep = [ev for ev in asleep if ev.duration >= persistence_threshold_samples]
        if len(persistent_sleep):
            return persistent_sleep[0].on
        return None

    def latency_to_stage(self, stage: SleepStages) -> Optional[float]:
        """Time from lights off to first epoch of the given stage

        Parameters
        ----------
        stage : SleepStages

        Returns
        -------
        latency : Optional[float]
            If stage does not occur, returns None. Otherwise, returns latency (in samples).

        Notes
        -----
        This metric is traditionally reported in minutes. Please take note of the sample_rate to convert if needed.
        """
        events = self[stage]
        if len(events) != 0:
            return events[0].on
        return None

    def generate_report(self) -> HypnogramReport:
        self_in_minutes = self.resample(1 / 60)
        self_first_two_hours = (
            self_in_minutes if self_in_minutes.duration < 120 else self_in_minutes.trim(end=120)
        )
        stages_nrem = [SleepStages.N1, SleepStages.N2, SleepStages.N3]

        sol = self_in_minutes.sleep_onset_latency(n1_tolerance=0.0)
        waso = self_in_minutes.wake_after_sleep_onset()

        # Although it might be safe to assume we have resampled 30-seconds epochs,
        # let's tolerate n1 episodes up to 2.5 30-s epochs.
        sol_unequivocal = self_in_minutes.sleep_onset_latency(n1_tolerance=1.25)

        latency_persistent_sleep = self_in_minutes.latency_to_first_persistent_sleep()
        if latency_persistent_sleep is not None:
            self_from_persistent = self_in_minutes.trim(
                latency_persistent_sleep, self_in_minutes.duration
            )
            latency_rem_from_persistent = self_from_persistent.latency_to_stage(SleepStages.REM)
        else:
            latency_rem_from_persistent = None
        wake_time = self_in_minutes.time_in_stage(SleepStages.W)
        rem_time = self_in_minutes.time_in_stage(SleepStages.REM)
        rem_time_first_two_hours = self_first_two_hours.time_in_stage(SleepStages.REM)
        persistent_wake_events = [
            ev
            for ev in self_in_minutes[SleepStages.W]
            if ev.duration > WAKE_PERSISTENCE_THRESHOLD_IN_MINUTES
        ]
        report = HypnogramReport(
            duration_minutes=self_in_minutes.duration,
            latency_minutes_rem=self_in_minutes.latency_to_stage(SleepStages.REM),
            latency_minutes_n1=self_in_minutes.latency_to_stage(SleepStages.N1),
            latency_minutes_n2=self_in_minutes.latency_to_stage(SleepStages.N2),
            latency_minutes_n3=self_in_minutes.latency_to_stage(SleepStages.N3),
            latency_minutes_persistent_sleep=latency_persistent_sleep,
            latency_minutes_rem_with_persistence=latency_rem_from_persistent,
            latency_minutes_sleep_onset=sol,
            latency_minutes_sleep_onset_unequivocal=sol_unequivocal,
            total_minutes_rem=rem_time,
            total_minutes_n1=self_in_minutes.time_in_stage(SleepStages.N1),
            total_minutes_n2=self_in_minutes.time_in_stage(SleepStages.N2),
            total_minutes_n3=self_in_minutes.time_in_stage(SleepStages.N3),
            total_minutes_nrem=sum([self_in_minutes.time_in_stage(s) for s in stages_nrem]),
            total_minutes_sleep=self_in_minutes.duration - wake_time,
            efficiency_rem=rem_time / self_in_minutes.duration,
            efficiency_rem_first_two_hours=rem_time_first_two_hours / 120,
            efficiency_sleep=1 - wake_time / self_in_minutes.duration,
            event_count_rem=len(self_in_minutes[SleepStages.REM]),
            event_count_wake=len(self_in_minutes[SleepStages.W]),
            event_count_wake_persistent=len(persistent_wake_events),
            wake_after_sleep_onset_minutes=0.0 if waso is None else waso,
        )
        return report

    def plot(self, unit="seconds", ax=None, color=None, alpha=1, label=None, **plt_kwargs):
        """Stacks individual series onto the same plot

        Parameters
        ----------
        unit : str, Optional[default="seconds"]
            Must be one of the keyword args to datetime.timedelta
            This will affect the x-axis label's units.
        ax : plt.Axes, Optional
        color : plt.colors[Optional]
            Color to use for the hypnogram and any non-sleep labels
        alpha : float, Optional[default=1]
            Alpha to use for hypnogram. A gradient thereof will be used for any non-sleep labels.
        label : str, Optional
            Label to use for this hypnogram. Useful when plotting multiple hypnograms on the given axis "ax"
        plt_kwargs : kwargs
            Any additional keyword args meant for plt.plot AND plt.fill_betweenx

        Notes
        -----
        This function is better aligned with the format of a hypnogram plot that is expected in sleep science.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 0.5 * len(self)))

        # handle unit conversion
        try:
            unit_hz = 1 / datetime.timedelta(**{unit: 1}).total_seconds()
        except TypeError:
            raise ValueError("Unit argument must be one of those accepted by datetime.timedelta!")
        resampled = self.resample(unit_hz)
        ax.set_xlabel(f"Time ({unit})")

        events_and_labels = []
        for key, values in resampled.items():
            event_label = key if key in SleepStages else np.nan
            for event in values:
                events_and_labels.append((event, event_label))
        events_and_labels = sorted(events_and_labels, key=lambda ev: ev[0].on)
        x, event_labels = [], []
        for event, event_label in events_and_labels:
            x.append(event.on)
            event_labels.append(event_label)
            x.append(event.off)
            event_labels.append(event_label)

        # "typical" hypnogram
        p = ax.plot(x, event_labels, color=color, alpha=alpha, label=label, **plt_kwargs)
        stages = [s for s in resampled if isinstance(s, SleepStages)]
        ax.set_yticks([s.value for s in stages])
        ax.set_yticklabels([s.name for s in stages])

        # pull out the color used for the hypnogram itself in case we need to reuse it
        color = p[0].get_color() if color is None else color

        missing_data_labels = {
            event_label: events
            for event_label, events in resampled.items()
            if isinstance(event_label, event_handling.MissingDataLabels)
        }

        if len(missing_data_labels):
            top = max(event_labels)
            for k, v in missing_data_labels.items():
                for on, off in v:
                    ax.fill_betweenx(
                        y=[0, top],
                        x1=on,
                        x2=off,
                        color=color,  # reuse color from line
                        alpha=alpha / (1 - k),  # make an alpha gradient by enumeration
                        label=k if label is None else f"{label}: ({k.name})",
                        **plt_kwargs,  # any additional kwargs we'd like to reuse
                    )

        # reset the legend to remove any duplicate labels
        handles, event_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(event_labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, bbox_to_anchor=(1, 1))

        ax.set_ylim(ax.get_ylim())
        ax.set_xlabel(f"Time ({unit})")
