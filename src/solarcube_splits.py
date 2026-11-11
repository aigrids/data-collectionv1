"""
Train / validation / test splits for the SolarCube sub-tasks (odd_space_*, odd_time_*, odd_spacetime_*).

The original SolarCube paper (Li et al., NeurIPS 2024) only splits by space (14 train / 5 test areas) and
explicitly does not split by time. The time-based splits below are our own definition, chosen so that every
season stays represented in training:

    train months  : Jan, Mar, Jun, Jul, Sep, Dec
    val months    : Apr, Oct
    test months   : Feb, May, Aug, Nov

A frame belongs to a month by its index: frame i is 2018-01-01 00:00 UTC + 15 min * i (35040 frames).
A window (history 96 steps + 96 target steps = frames t-96 .. t+95) is only used by a split if ALL of its
frames lie in that split's months, so no window overlaps data of another split.

    space      : train = train stations (all months) | val = val stations | test = the 5 test stations
    time       : all 19 stations in each split, by months
    spacetime  : train = train stations x train months | val = val stations x val months |
                 test = test stations x test months
"""
import numpy as np

MONTHS = {"train": (1, 3, 6, 7, 9, 12), "val": (4, 10), "test": (2, 5, 8, 11)}
FRAME_MINUTES = 15


def month_of_frame(n_frames):
    t = np.datetime64("2018-01-01T00:00") + np.arange(n_frames) * np.timedelta64(FRAME_MINUTES, "m")
    return (t.astype("datetime64[M]").astype(int) % 12) + 1


def frame_mask(months, n_frames):
    return np.isin(month_of_frame(n_frames), months)


def plan(kind, all_ids, test_ids, val_ids):
    """Returns {'train': (station_ids, months or None), 'val': ..., 'test': ...}."""
    all_ids, test_ids, val_ids = sorted(all_ids), sorted(test_ids), sorted(val_ids)
    train_space = [s for s in all_ids if s not in test_ids and s not in val_ids]
    if kind == "space":
        return {"train": (train_space, None), "val": (val_ids, None), "test": (test_ids, None)}
    if kind == "time":
        return {k: (all_ids, MONTHS[k]) for k in ("train", "val", "test")}
    if kind == "spacetime":
        return {"train": (train_space, MONTHS["train"]), "val": (val_ids, MONTHS["val"]),
                "test": (test_ids, MONTHS["test"])}
    raise ValueError(f"unknown split kind {kind!r}")


def restrict_windows(ds, stations, months, hist=96, horizon=96):
    """Keep only windows of `ds` (ds.index = [(station, t), ...]) whose whole span lies in `months`."""
    if months is None:
        return ds
    prefix = {}
    keep = []
    for sid, t in ds.index:
        if sid not in prefix:
            n = stations[sid]["meta"]["n_frames"]
            prefix[sid] = np.concatenate([[0], np.cumsum(frame_mask(months, n))])
        p = prefix[sid]
        if p[t + horizon] - p[t - hist] == hist + horizon:
            keep.append((sid, t))
    ds.index = keep
    return ds
