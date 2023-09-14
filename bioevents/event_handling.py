from __future__ import annotations

import json
from decimal import Decimal
from enum import IntEnum
from typing import Dict, Optional, ValuesView

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from bioevents.colors import Colors

DEFAULT_SAMPLING_RATE_HZ = 1.0


class OverlapTolerances:
    def __init__(self, diff_on=np.inf, diff_off=np.inf, ratio_on=np.inf, ratio_off=np.inf):
        """Useful for storing various tolerances employed by the Event.overlaps function

        Parameters
        ----------
        diff_on, diff_off, ratio_on, ratio_off : int or float
            Must be non-negative floats.
            diff_on, diff_off: maximum acceptable difference in on or off timestamps
            ratio_on, ratio_off: maximum acceptable ratio between
                difference in on or off timestamps and reference duration

        Notes
        -----
        This may be a somewhat odd architecture for a collection of configurations.
        However, note that it enables several unique handling operations via customized built-in functions.
        In addition, one may pre-configure this object and propagate it through an EventStack or EventSeries.

        All timestamp diffs are unsigned, meaning tolerances apply symmetrically about the reference timestamp.

        Note that the default configuration allows for events with ANY overlapping samples to overlap.
        This is consistent with "Seizure Detection" (Scheuer et al., 2021)

        References
        ----------
        "Spike Detection", Mark L. Scheuer et al., 2016
            "Spikes marked by different readers that occurred within 200 milliseconds of one another
            were considered to be the same event.""
        "Seizure Detection", Mark L. Scheuer et al., 2021
            "Seizure marks from a reader pair were designated a match (true positive) if they overlapped"

        See Also
        --------
        Event.overlaps
        EventSeries.set_tolerances
        EventStack.set_tolerances
        """
        self.diff_on = diff_on
        self.diff_off = diff_off
        self.ratio_on = ratio_on
        self.ratio_off = ratio_off

    def __repr__(self):
        """Allows for quick inspection"""
        return str(dict(self))

    def __iter__(self):
        """Allows for dictionary composition"""
        for k in dir(self):
            if k.startswith("_"):
                continue
            yield k, getattr(self, k)

    def __setattr__(self, k, v):
        """Ensures all tolerances are non-negative numbers"""
        assert v >= 0, ValueError("Tolerance must be a non-negative number")
        super().__setattr__(k, v)


class Event:
    default_duration = 1

    def __init__(self, on, off=None, tolerances=None):
        """
        Container for timestamped, binary events.
        Also includes functionality for determining overlap with other events.
        """
        assert on >= 0, ValueError("Onset must be non-negative")
        off = on + self.default_duration if off is None else off
        assert off > on, ValueError("Non-positive duration calculated.")
        tolerances = OverlapTolerances() if tolerances is None else tolerances
        assert isinstance(tolerances, OverlapTolerances)
        self._on = on
        self._off = off
        self._tolerances = tolerances

    @classmethod
    def from_duration(cls, on, duration, tolerances=None):
        return cls(on, on + duration, tolerances)

    def __repr__(self):
        return str(dict(self))

    def __copy__(self):
        return Event(on=self.on, off=self.off, tolerances=self.tolerances)

    def __iter__(self):
        """Utility: TBD"""
        for k, v in zip(["on", "off"], [self.on, self.off]):
            yield k, v

    def __eq__(self, item):
        """Utility: TBD"""
        return self.on == item.on and self.off == item.off

    @property
    def on(self):
        return self._on

    @property
    def off(self):
        return self._off

    @property
    def duration(self):
        return self.off - self.on

    @property
    def tolerances(self):
        return self._tolerances

    @tolerances.setter
    def tolerances(self, vals):
        assert isinstance(vals, OverlapTolerances)
        self._tolerances = vals

    def overlaps(self, other):
        """
        Determines whether this event "overlaps" another, based on various criteria.

        Notes
        -----
        It's important to note that this overlap is not necessarily a "symmetrical" operation.
        Because the duration and tolerances of THIS event are used, self.overlaps(other) may differ
        from other.overlaps(self). This is by design, given that a "reference" or "truth" event should
        govern the acceptance criteria of "predicted" event annotations, and not the other way around.

        See Also
        --------
        OverlapTolerances
        """
        assert isinstance(other, Event)

        if self == other:
            return True

        # note -- this is rather arbitrary and could be reversed (see Notes)
        ref, test = self, other
        dur, tol = ref.duration, ref.tolerances

        transitions = [ref.on, test.on, ref.off, test.off]
        if max(transitions) - min(transitions) > (ref.duration + test.duration):
            # note that directly adjacent events will pass this condition
            return False  # there is no overlap at all

        diff = np.abs(ref.on - test.on)
        if diff > tol.diff_on:
            return False
        if (diff / dur) > tol.ratio_on:
            return False

        diff = np.abs(ref.off - test.off)
        if diff > tol.diff_off:
            return False
        if (diff / dur) > tol.ratio_off:
            return False

        return True

    def abuts(self, other):
        """Determines whether the onset of one event is the offset of the other"""
        return self.on == other.off or self.off == other.on


class EventSeries(list):
    def __init__(
        self,
        events,
        duration=None,
        sampling_rate: float = DEFAULT_SAMPLING_RATE_HZ,
    ):
        """Time series container for events.
        Provides various functionality for comparison between self and others.

        Parameters
        ----------
        events : list of Event
        duration : Optional(float or int)
            Duration of the entire time series.
            If omitted, the duration is assumed to be the offset of the latest event.
        sampling_rate : float, default=DEFAULT_SAMPLING_RATE_HZ
            Sampling rate in Hertz

        Notes
        -----
        Currently assumes the time series starts at timestamp 0.
        TODO: support non-zero start.
        TODO: support configurable sample rate
        """
        assert all(isinstance(item, Event) for item in events), ValueError("Non-event given.")
        super().__init__(events)
        if duration is not None:
            assert isinstance(duration, (float, int)) and (duration > 0)
        self._duration = duration
        self._sampling_rate = float(sampling_rate)
        self._resolve_events()

    def __contains__(self, item):
        assert isinstance(item, Event)
        return any(event.overlaps(item) for event in self)

    def __eq__(self, other):
        events_equal = super(EventSeries, self).__eq__(other)
        metadata_equal = self.is_compatible(other)
        return events_equal and metadata_equal

    def resample(self, sampling_rate):
        def convert(x):
            """Avoids numerical precision issues -- try running 3 * (1 / 10)"""
            return float(Decimal(x) / Decimal(self.sampling_rate) * Decimal(sampling_rate))

        return EventSeries(
            events=[Event(convert(ev.on), convert(ev.off), ev.tolerances) for ev in self],
            duration=convert(self.duration),
            sampling_rate=sampling_rate,
        )

    def trim(self, start: Optional[int] = None, end: Optional[int] = None):
        if start is None:
            start = 0
        elif start < 0:
            raise ValueError("Cannot trim outside the bounds of this EventSeries")

        if end is None:
            end = self.duration
        elif end > self.duration:
            raise ValueError("Cannot trim outside the bounds of this EventSeries")
        elif start >= end:
            raise ValueError("Trim start must be before trim end")

        trim_window = Event(start, end)
        events = []
        for event in self:
            if trim_window.abuts(event):
                continue
            if not trim_window.overlaps(event):
                continue
            events.append(
                Event(
                    max(0, event.on - start),
                    min(end, event.off) - start,
                    tolerances=event.tolerances,
                )
            )
        return EventSeries(events, duration=trim_window.duration, sampling_rate=self.sampling_rate)

    def is_compatible(self, item):
        """See if self vs. other functions can be used with this item"""
        assert isinstance(item, EventSeries), ValueError("EventSeries not given.")
        return self.sampling_rate == item.sampling_rate and self.duration == item.duration

    def spans_event(self, item):
        """See if the given event falls within the time bounds of this series"""
        assert isinstance(item, Event), ValueError("Event not given.")
        if self._duration is not None:
            return self._duration >= item.off

    def _assert_is_compatible(self, item):
        assert self.is_compatible(item), ValueError("EventSeries is not compatible.")

    def _assert_spans_event(self, item):
        assert self.spans_event(item), ValueError("Event exceeds duration")

    def __add__(self, item):
        """Defines built-in 'add' as a boolean intersection of compatible series"""
        self._assert_is_compatible(item)
        return EventSeries.from_bools(
            self.as_bools() + item.as_bools(), sampling_rate=self.sampling_rate
        )

    def __sub__(self, item):
        """Defines built-in 'add' as a boolean difference of compatible series"""
        self._assert_is_compatible(item)
        return EventSeries.from_bools(
            self.as_bools() & ~item.as_bools(), sampling_rate=self.sampling_rate
        )

    def __or__(self, item):
        """Defines built-in 'or' operator as a "smart" Event-wise intersection of compatible series"

        Notes
        -----
        -Overlapping events will be combined via logical "or"
        -Resulting events are "resolved"
        """
        self._assert_is_compatible(item)
        return self + item

    def __and__(self, item):
        """
        Notes
        -----
        -Overlap will be determined symmetrically by agreement between both items
        -Overlapping events will be combined via logical "or"
        -Resulting events are "resolved"
        """
        self._assert_is_compatible(item)
        events = []
        for a in self:
            for b in item:
                if a.overlaps(b) and b.overlaps(a):
                    events.append(Event(min(a.on, b.on), max(a.off, b.off)))
        return EventSeries(events, duration=self.duration, sampling_rate=self.sampling_rate)

    def __invert__(self):
        if len(self):
            events = []
            if self[0].on > 0:
                events.append(Event(on=0, off=self[0].on))
            for i in range(len(self) - 1):
                events.append(Event(on=self[i].off, off=self[i + 1].on))
            if self[-1].off < self.duration:
                events.append(Event(on=self[-1].off, off=self.duration))
        else:
            events = [Event(on=0, off=self.duration)]
        return EventSeries(events, duration=self.duration, sampling_rate=self.sampling_rate)

    def append(self, item):
        """Overrides built-in default to make sure the Event is concurrent with this series"""
        self._assert_spans_event(item)
        super().append(item)
        self._resolve_events()

    def extend(self, item):
        """Overrides built-in default to make sure the Events are concurrent with this series"""
        for e in item:
            self._assert_spans_event(e)
        super().extend(item)
        self._resolve_events()

    def _resolve_events(self):
        """Sort and combine overlapping events

        Notes
        -----
        Could use as/from bools to accomplish, but this can be much more efficient
        """
        if not len(self):
            return
        self.sort(key=lambda x: x.on, reverse=False)
        after = self.pop()
        final = []
        while len(self):
            before = self.pop()
            if before.off >= after.on:
                after = Event(before.on, max(before.off, after.off))
            else:
                final.append(after)
                after = before
        final.append(after)
        self.clear()
        super().extend(list(reversed(final)))

    def debounce(self, persistence_period: float, min_duration: float) -> EventSeries:
        """Debounce the given signal to remove short events and merge close ones

        Parameters
        ----------
        persistence_period : float
            merge any two events separated by less than this number (of samples)
        min_duration : float
            after merging close events, remove all events under this duration (in samples)

        Returns
        -------
        debounced : EventSeries

        References
        ----------
        Description of debouncing applied in engineering context: https://my.eng.utah.edu/~cs5780/debouncing.pdf
        """
        events_merged = []
        if len(self):
            # merge any events that are close to each other using a persistence period
            event = self[0]
            on, off = event.on, event.off
            for event in self[1:]:
                if event.on - off < persistence_period:
                    off = event.off  # merge events
                else:
                    # stash our new timestamps as a separate event
                    events_merged.append(Event(on, off))
                    on, off = event.on, event.off
            events_merged.append(Event(on, off))

        # keep only events that are longer than the minimum duration
        events_long_enough = [ev for ev in events_merged if ev.duration >= min_duration]

        return EventSeries(
            events_long_enough, duration=self.duration, sampling_rate=self.sampling_rate
        )

    @property
    def duration(self):
        if self._duration is None:
            return max(e.off for e in self)
        return float(self._duration)

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @classmethod
    def from_bools(cls, bools, sampling_rate: float = DEFAULT_SAMPLING_RATE_HZ):
        """Converts the given boolean array into a series of discrete Events"""
        transitions = np.where(np.diff(bools))[0] + 1
        transitions = transitions.tolist()
        transitions = ([0] if bools[0] else []) + transitions + ([len(bools)] if bools[-1] else [])
        events = [Event(on, off) for on, off in zip(transitions[0::2], transitions[1::2])]
        return cls(events, duration=len(bools), sampling_rate=sampling_rate)

    def set_tolerances(self, vals):
        """Iteratively set the tolerances for all Events to the given vals"""
        for e in self:
            e.tolerances = vals

    def as_bools(self, round_method_on=np.ceil, round_method_off=np.floor):
        """Converts the Events to a boolean array, using parameterizable rounding methods"""
        bools = np.zeros(int(np.ceil(self.duration)), dtype=bool)
        for e in self:
            on, off = round_method_on(e.on).astype(int), round_method_off(e.off).astype(int)
            bools[on:off] = True
        return bools

    def as_dataframe(self):
        """Export events to a DataFrame"""
        return pd.DataFrame([(e.on, e.off) for e in self], columns=["on", "off"])

    def compute_agreement(self, other, normalize=False):
        """Computes an agreement matrix with the given EventSeries

        Parameters
        ----------
        other : EventSeries
        normalize : Optional(bool), default=False
            Normalize results rows by the number of events in the respective "reference" series

        Returns
        -------
        mat : pd.DataFrame
            2 x 2 agreement matrix
        """
        self._assert_is_compatible(other)
        mat = []
        series = [self, other]
        for ev in series:
            mat.append([sum(e in ev2 for e in ev) for ev2 in series])
        mat = pd.DataFrame(mat)
        mat.index = ["self (ref)", "other (ref)"]
        mat.columns = ["self", "other"]
        if normalize:
            mat = (mat.T / [len(ev) for ev in series]).T
        return mat

    @staticmethod
    def _format_confusion_matrix(mat):
        """Convenience function to convert to a DataFrame"""
        mat = pd.DataFrame(mat)
        mat.index = ["self (N)", "self (P)"]
        mat.columns = ["other (N)", "other (P)"]
        return mat

    def event_confusion_matrix(self, other, normalize=None, assimilate_events=True):
        """Generates a confusion matrix with the given EventSeries via Event-wise logic

        Parameters
        ----------
        other : EventSeries
        normalize : Optional(bool), default=False
        assimilate_events : Optional(bool), default=True
            If true, any overlapping Events from either series will be counted as one.

        Returns
        -------
        mat : pd.DataFrame
        """
        self._assert_is_compatible(other)
        if assimilate_events:
            truth = [e in self for e in self + other]
            pred = [e in other for e in self + other]
        else:
            truth = [e in self for e in self] + [e in self for e in other]
            pred = [e in other for e in self] + [e in other for e in other]
        mat = confusion_matrix(truth, pred, normalize)
        return self._format_confusion_matrix(mat)

    def epoch_confusion_matrix(self, other, normalize=None):
        """Generates a confusion matrix with the given EventSeries epoch-by-epoch via boolean logic

        Parameters
        ----------
        other : EventSeries
        normalize : Optional(bool), default=False

        Returns
        -------
        mat : pd.DataFrame
        """
        self._assert_is_compatible(other)
        truth, pred = self.as_bools(), other.as_bools()
        mat = confusion_matrix(truth, pred, normalize)
        return self._format_confusion_matrix(mat)

    def plot(self, bottom=0, top=1, color=None, as_seconds=True):
        """Simple plot with events filled between the given y values"""
        color = Colors.BLUE1 if color is None else color
        hz = self.sampling_rate if as_seconds else 1
        for event in self:
            plt.fill_betweenx(
                [bottom, top], event.on / hz, event.off / hz, alpha=0.3, color=color, linewidth=0
            )
        plt.xlim([0, self.duration / hz])
        plt.xlabel(f"Time ({'seconds' if as_seconds else 'samples'})")


def confusion_matrix(true, pred, normalize=None):
    """
    Notes
    -----
    This is a paraphrased implementation of scikit-learn's. We can use that method,
    but this allows our footprint and dependencies to remain minimal.
    """
    assert len(true) == len(pred)
    true, pred = np.array(true), np.array(pred)
    classes = np.unique([true, pred])
    n = len(classes)
    mat = [sum((true == a) & (pred == b)) for a in classes for b in classes]
    mat = np.reshape(mat, (n, n))

    # normalize
    if normalize is None:
        return mat
    mapping = {"all": None, "pred": 0, "true": 1}
    if normalize not in mapping:
        raise ValueError("Value of 'normalize' not understood. Try one of {true, pred, all}")
    axis = mapping[normalize]
    counts = mat.sum(axis=axis).clip(1, None)
    counts = np.ones_like(mat) * np.atleast_2d(counts)
    if axis == 1:
        counts = counts.T
    return mat / counts


class EventStack(dict):
    KEY_META = "meta"
    KEY_EVENT_STACK = "event_stack"
    KEY_DURATION = "duration"
    KEY_SAMPLING_RATE = "sampling_rate"

    KEY_ASSIGNED_ENUM = "AssignedEnum"

    def __init__(self, event_stack: Dict[IntEnum, EventSeries]):
        """Ingests multiple concurrent EventSeries objects, each with a key

        Parameters
        ----------
        event_stack : Dict[IntEnum, EventSeries]
        """
        vals = event_stack.values()
        for item in vals:
            assert isinstance(item, EventSeries)
        self._duration = self._infer_duration(vals)
        self._sampling_rate = self._infer_sampling_rate(vals)
        for k, v in event_stack.items():
            self._check_compatible(k, v)
        super().__init__(event_stack)

    @classmethod
    def from_string_keys(cls, event_stack: Dict[str, EventSeries]):
        assigned_enum = IntEnum(cls.KEY_ASSIGNED_ENUM, list(event_stack), start=0)
        return cls(event_stack={assigned_enum[k]: v for k, v in event_stack.items()})

    @staticmethod
    def _infer_duration(event_stack: ValuesView[EventSeries]) -> float:
        """Finds the common duration of each EventSeries and raises exception if there are multiple durations"""
        durations = list(set(item.duration for item in event_stack))
        if len(durations) != 1:
            raise ValueError("Events of differing durations given.")
        return durations[0]

    @staticmethod
    def _infer_sampling_rate(event_stack: ValuesView[EventSeries]) -> float:
        """Finds the common sampling rate of each EventSeries and raises exception if there are multiple values"""
        sampling_rates = list(set(item.sampling_rate for item in event_stack))
        if len(sampling_rates) != 1:
            raise ValueError("Events of differing sampling rates given.")
        return sampling_rates[0]

    @property
    def duration(self):
        return self._duration

    @property
    def sampling_rate(self):
        return self._sampling_rate

    @property
    def classes(self):
        return sorted(set(self))

    def _check_compatible(self, key, value):
        """Makes sure the given key/value pair would be a valid entry for this EventStack"""
        assert value.duration == self.duration
        assert value.sampling_rate == self.sampling_rate
        assert isinstance(key, IntEnum)

    def to_dict(self) -> dict:
        meta = {self.KEY_DURATION: self.duration, self.KEY_SAMPLING_RATE: self.sampling_rate}
        stack = {k.name: [dict(a) for a in v] for k, v in self.items()}
        return {self.KEY_META: meta, self.KEY_EVENT_STACK: stack}

    @classmethod
    def from_dict(cls, data: dict, intenum: IntEnum) -> EventStack:
        meta, event_stack = data[cls.KEY_META], data[cls.KEY_EVENT_STACK]
        return cls(
            {
                intenum[k]: EventSeries([Event(**ev) for ev in v], **meta)
                for k, v in event_stack.items()
            }
        )

    def write_json(self, filepath):
        """Serializes and exports event data to a JSON file

        Notes
        -----
        We're not exporting any OverlapTolerances here...
        seems like the kind of thing you'd only want to do explicitly during analysis anyway
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def read_json(cls, filepath, intenum: IntEnum):
        """Construct from the given filepath"""
        with open(filepath, "r") as f:
            return cls.from_dict(data=json.load(f), intenum=intenum)

    def set_tolerances(self, vals):
        """Iteratively set the tolerances of all Events within all EventSeries"""
        for e in self.values():
            e.set_tolerances(vals)

    def resample(self, sampling_rate):
        """Make a new EventStack with resampled EventSeries values"""
        cls = type(self)
        return cls({k: series.resample(sampling_rate) for k, series in self.items()})

    def trim(self, start: Optional[int] = None, end: Optional[int] = None):
        """Make a new, trimmed version of this EventStack"""
        cls = type(self)
        return cls({k: series.trim(start, end) for k, series in self.items()})

    def plot(self):
        """Stacks individual series onto the same plot"""
        n = max(self) + 1
        cmap = LinearSegmentedColormap.from_list("", ["black", Colors.BLUE1], N=n)
        fig, ax = plt.subplots(figsize=(10, 0.5 * len(self)))
        for k, v in self.items():
            v.plot(bottom=0, top=k, color=cmap(k))
        ax.set_yticks([k.value for k in self])
        ax.set_yticklabels([k.name for k in self])
        ax.set_ylim(ax.get_ylim())
        return fig


class EventClassSeries(EventStack):
    """A special type of EventStack wherein every timestamp is expected to have one and only one class

    Notes
    -----
    Useful for performing class confusion or agreement analytic, such as with hypnogram annotations
    """

    def __init__(
        self, event_stack: Dict[IntEnum, EventSeries], missing_enum: Optional[IntEnum] = None
    ):
        """Ingests multiple concurrent EventSeries objects, each with a key

        Parameters
        ----------
        event_stack : Dict[IntEnum, EventSeries]
        missing_enum : IntEnum, Optional
            value to fill where no annotation exists
        """
        vals = event_stack.values()
        for item in vals:
            assert isinstance(item, EventSeries)
        if missing_enum is not None:
            missing = np.sum([s.as_bools() for s in vals], axis=0) == 0
            event_stack[missing_enum] = EventSeries.from_bools(
                missing, sampling_rate=self._infer_sampling_rate(vals)
            )
        msg = "Input annotations must be continuous and flat!"
        assert self._is_continuous_and_flat(vals), msg
        super().__init__(event_stack)

    @classmethod
    def _is_continuous_and_flat(cls, event_stack: ValuesView[EventSeries]):
        """One and only one class active at each timestamp

        Returns
        -------
        is_continuous_and_flat : bool
        """
        events = sorted([ev for v in event_stack for ev in v], key=lambda x: x.on)
        starts_at_zero = events[0].on == 0
        ends_at_duration = events[-1].off == cls._infer_duration(event_stack)
        is_continuous_and_flat = all(
            [events[i].abuts(events[i + 1]) for i in range(len(events) - 1)]
        )
        return is_continuous_and_flat and starts_at_zero and ends_at_duration

    @classmethod
    def from_array(cls, class_array, sampling_rate: float = DEFAULT_SAMPLING_RATE_HZ):
        """Initialize from a 1-D array of classes"""
        assert isinstance(class_array, (list, np.ndarray, pd.Series))
        assert all(isinstance(c, IntEnum) for c in class_array)
        events = dict()
        for c in sorted(set(class_array)):
            bools = [a == c for a in class_array]
            events[c] = EventSeries.from_bools(bools, sampling_rate=sampling_rate)
        return cls(events)

    def as_array(self, round_method_on=np.ceil, round_method_off=np.floor):
        """Export all events as a list of classes"""
        array = [None] * int(round_method_off(self.duration))
        for k, events in self.items():
            bools = events.as_bools(round_method_on, round_method_off)
            array = [k if b else a for a, b in zip(array, bools)]
        if None in array:
            raise RuntimeError(
                "None detected in result of EventClassSeries.as_array, this is a bug"
            )
        return array

    def event_confusion_matrix(self, other, assimilate_events=True):
        """Generate a class-wise confusion matrix with 'other'

        Parameters
        ----------
        other : EventClassSeries
        assimilate_events : Optional(bool), default=True
            If true, any overlapping Events from either series will be counted as one.

        Returns
        -------
        mat : pd.DataFrame
        """
        mat = []
        for s in self.classes:
            events_self = self[s]
            for o in self.classes:
                events_other = other[o]
                if assimilate_events:
                    truth = [e in events_self for e in events_self + events_other]
                    pred = [e in events_other for e in events_self + events_other]
                else:
                    truth = [e in events_self for e in events_self] + [
                        e in events_self for e in events_other
                    ]
                    pred = [e in events_other for e in events_self] + [
                        e in events_other for e in events_other
                    ]
                mat.append(sum(np.array(truth) & np.array(pred)))
        mat = np.reshape(mat, (len(self), len(self)))
        return self._format_confusion_matrix(mat)

    def epoch_confusion_matrix(self, other, normalize=None):
        """Generates a confusion matrix with 'other', epoch-by-epoch via boolean logic

        Parameters
        ----------
        other : EventClassSeries
        normalize : Optional(bool), default=False

        Returns
        -------
        mat : pd.DataFrame
        """
        truth, pred = self.as_array(), other.as_array()
        mat = confusion_matrix(truth, pred, normalize)
        return self._format_confusion_matrix(mat, labels=set(truth) | set(pred))

    def _format_confusion_matrix(self, mat, labels=None):
        """Convenience function to convert confusion matrix to a DataFrame"""
        mat = pd.DataFrame(mat)
        labels = self.classes if labels is None else labels
        mat.index = [f"self ({c.name})" for c in labels]
        mat.columns = [f"other ({c.name})" for c in labels]
        return mat


class SeizureEvent(Event):
    default_duration = 120


class MissingDataLabels(IntEnum):
    MISSING = 0
    UNSCORABLE = -1
    ARTIFACT = -2
    DISAGREE = -3
