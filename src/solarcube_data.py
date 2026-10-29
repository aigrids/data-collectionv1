"""
Data pipeline for the SolarCube point sub-tasks (odd_space_point, ...).

Works on the per-station patch cache written by scripts/cache_solarcube_patches.py:
    station_<id>_X.npy  int16 (T, 5, P, P)  channels = CHANNELS (raw sentinels kept)
    station_<id>_y.npy  float32 (T,)         ground radiation in W/m^2, NaN = missing
    station_<id>_meta.json

Sample = (station, origin t): the previous HIST=96 steps (24 h at 15 min) of satellite
patches + ground radiation are the input, the next HORIZON=96 steps of ground radiation
are the target. Nothing is held in memory except the int16 caches (~17 MB per station).
"""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import solarcube_splits as sp

HIST = 96
HORIZON = 96
SENTINEL = -9999
CLOUD_MISSING = -1
CHANNELS = ["visual_band_47", "visual_band_86", "infrared_band_133", "cloud_mask", "solar_zenith_angle"]
SZA_IDX = CHANNELS.index("solar_zenith_angle")
Y_SCALE = 1000.0  # W/m^2 -> model units


def load_cache(cache_dir, station_ids=None):
    stations = {}
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith("_meta.json"):
            continue
        with open(os.path.join(cache_dir, fn)) as f:
            meta = json.load(f)
        sid = meta["station_id"]
        if station_ids is not None and sid not in station_ids:
            continue
        stations[sid] = {
            "meta": meta,
            "X": np.load(os.path.join(cache_dir, f"station_{sid}_X.npy")),
            "y": np.load(os.path.join(cache_dir, f"station_{sid}_y.npy")),
        }
    return stations


def space_split(stations, val_stations=(2, 15)):
    """odd_space_*: test = the 5 stations flagged test=1 in station_features.csv (verified).
    The package defines no validation stations, so two training stations are held out for
    early stopping (default: 2 = fpk (SURFRAD), 15 = ISH (BSRN))."""
    test = sorted(s for s, d in stations.items() if d["meta"]["is_test_station"])
    val = sorted(s for s in val_stations if s in stations and s not in test)
    train = sorted(s for s in stations if s not in test and s not in val)
    return train, val, test


def valid_mask(X):
    """bool array like X: True where the source value is not a missing-data sentinel."""
    v = X != SENTINEL
    v[:, CHANNELS.index("cloud_mask")] = X[:, CHANNELS.index("cloud_mask")] != CLOUD_MISSING
    return v


def compute_norm_stats(stations, ids, step=7, months=None):
    """Per-channel mean/std over valid entries of the training stations (subsampled in time)."""
    s1 = np.zeros(len(CHANNELS)); s2 = np.zeros(len(CHANNELS)); n = np.zeros(len(CHANNELS))
    for sid in ids:
        X = stations[sid]["X"]
        sel = np.arange(0, len(X), step)
        if months is not None:
            sel = sel[sp.frame_mask(months, len(X))[sel]]
        X = X[sel]
        v = valid_mask(X)
        for c in range(len(CHANNELS)):
            xs = X[:, c][v[:, c]].astype(np.float64)
            s1[c] += xs.sum(); s2[c] += (xs ** 2).sum(); n[c] += xs.size
    mean = s1 / np.maximum(n, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class SolarForecastDataset(Dataset):
    """Windows of (past 24 h inputs) -> (next 24 h ground radiation)."""

    def __init__(self, stations, ids, mean, std, stride=24, min_valid=0.8):
        self.stations, self.mean, self.std = stations, mean, std
        self.index = []  # (sid, t)
        for sid in ids:
            y = stations[sid]["y"]
            T = len(y)
            c = np.concatenate([[0], np.cumsum(np.isfinite(y))])  # prefix count of valid readings
            for t in range(HIST, T - HORIZON + 1, stride):
                hist_ok = (c[t] - c[t - HIST]) / HIST >= min_valid
                targ_ok = (c[t + HORIZON] - c[t]) / HORIZON >= min_valid
                if hist_ok and targ_ok:
                    self.index.append((sid, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sid, t = self.index[i]
        d = self.stations[sid]
        X = d["X"][t - HIST:t]
        v = valid_mask(X)
        x = (X.astype(np.float32) - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        x = np.where(v, x, 0.0).astype(np.float32)
        x = np.concatenate([x, v.astype(np.float32)], axis=1)            # (HIST, 2C, P, P)

        yh = d["y"][t - HIST:t]
        hm = np.isfinite(yh)
        hist_y = np.stack([np.where(hm, yh, 0.0) / Y_SCALE, hm.astype(np.float32)], axis=-1).astype(np.float32)

        yt = d["y"][t:t + HORIZON]
        tm = np.isfinite(yt)
        target = (np.where(tm, yt, 0.0) / Y_SCALE).astype(np.float32)

        # solar zenith angle of the *future* steps at the station pixel (astronomical, known in advance)
        oy, ox = d["meta"]["station_offset_in_patch"]
        Xf = d["X"][t:t + HORIZON, SZA_IDX, oy, ox]
        fut_sza = np.where(Xf != SENTINEL, (Xf.astype(np.float32) - self.mean[SZA_IDX]) / self.std[SZA_IDX], 0.0)

        return {
            "x": torch.from_numpy(x), "hist_y": torch.from_numpy(hist_y),
            "fut_sza": torch.from_numpy(fut_sza.astype(np.float32)),
            "target": torch.from_numpy(target), "tmask": torch.from_numpy(tm.astype(np.float32)),
            "sid": sid, "t": t,
        }
