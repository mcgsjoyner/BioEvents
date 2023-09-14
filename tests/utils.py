import numpy as np

from bioevents import event_handling, hypnogram


def simulate_hypnogram(series_duration, num_cycles=5, avg_duration=5, seed=0):
    rng = np.random.default_rng(seed)
    classes = hypnogram.SleepStages
    stack = {k: [] for k in classes}
    c = 5
    stack_duration = 0
    cycles = (
        np.abs(np.sin(np.pi * num_cycles * np.linspace(0, 1, series_duration))) * np.ptp(classes)
        - 0.2
    )
    while stack_duration < series_duration:
        dur = int(
            np.clip(
                rng.normal(avg_duration, avg_duration / 3),
                1,
                series_duration - stack_duration,
            )
        )
        stack[c].append(event_handling.Event(stack_duration, stack_duration + dur))
        c = max(classes) - cycles[stack_duration] + rng.normal() / 3
        c = int(np.clip(c, min(classes), max(classes)))
        stack_duration += dur
    stack = hypnogram.Hypnogram(
        {
            k: event_handling.EventSeries(events, duration=series_duration)
            for k, events in stack.items()
            if events
        }
    )
    return stack
