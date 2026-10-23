"""
Cache station-centred patches of the SolarCube satellite data (point sub-tasks).

Reads each station's .h5 files in small time blocks (never the whole array),
crops a PxP patch around the station pixel and stores:

    <out_dir>/station_<id>_X.npy   int16, shape (T, 5, P, P)   raw sentinel values kept
    <out_dir>/station_<id>_y.npy   float32, shape (T,)         ground radiation, NaN = missing
    <out_dir>/station_<id>_meta.json

Channel order: visual_band_47, visual_band_86, infrared_band_133, cloud_mask, solar_zenith_angle.
Missing-value sentinels are kept as in the source files (-9999, cloud_mask -1);
normalisation / masking happens later in the Dataset.

Memory: one block of `--block` full frames (default 1024 x 120 x 120 x 2 B = ~29 MB).

Usage:
    python3 cache_solarcube_patches.py <SolarCube_dir> <out_dir> [--patch 7] [--block 1024]
                                       [--stations 1,2,3] [--max-frames N] [--force]
"""
import argparse
import csv
import json
import os
import time

import h5py
import numpy as np

CHANNELS = [
    ("visual_band_47", "visual_band_47.h5"),
    ("visual_band_86", "visual_band_86.h5"),
    ("infrared_band_133", "infrared_band_133.h5"),
    ("cloud_mask", "cloud_mask.h5"),
    ("solar_zenith_angle", "solar_zenith_angle.h5"),
]
GRID = 120


def read_stations(root):
    with open(os.path.join(root, "station_features.csv"), newline="") as f:
        return list(csv.DictReader(f))


def station_pixel(s):
    """Pixel (row, col) of the station inside its 120x120 tile, from lat/lon."""
    lat, lon = float(s["lats"]), float(s["lons"])
    lat_ul, lon_ul = float(s["lat_ulcnr"]), float(s["lon_ulcnr"])
    lat_lr, lon_lr = float(s["lat_lrcnr"]), float(s["lon_lrcnr"])
    dlat = (lat_ul - lat_lr) / GRID
    dlon = (lon_lr - lon_ul) / GRID
    row = int((lat_ul - lat) / dlat)
    col = int((lon - lon_ul) / dlon)
    # station_features.csv stores them swapped: fLine = column, fCol = row
    f_row, f_col = int(s["fCol"]), int(s["fLine"])
    if abs(row - f_row) > 1 or abs(col - f_col) > 1:
        raise ValueError(
            f"station {s['id']} ({s['name']}): pixel from lat/lon = (row {row}, col {col}) "
            f"but station_features.csv gives (row {f_row}, col {f_col}) -- geometry assumption wrong"
        )
    return row, col


def process_station(root, out_dir, s, ground, patch, block, max_frames, force):
    sid = int(s["id"])
    x_path = os.path.join(out_dir, f"station_{sid}_X.npy")
    if os.path.exists(x_path) and not force:
        print(f"[skip] station {sid} ({s['name']}) already cached", flush=True)
        return

    row, col = station_pixel(s)
    half = patch // 2
    r0 = min(max(row - half, 0), GRID - patch)
    c0 = min(max(col - half, 0), GRID - patch)

    sdir = os.path.join(root, f"station_{sid}")
    X = None
    for ci, (name, fname) in enumerate(CHANNELS):
        t_start = time.time()
        with h5py.File(os.path.join(sdir, fname), "r") as f:
            ds = f[list(f.keys())[0]]
            T = ds.shape[0] if max_frames is None else min(ds.shape[0], max_frames)
            if X is None:
                X = np.empty((T, len(CHANNELS), patch, patch), dtype=np.int16)
            elif T != X.shape[0]:
                raise ValueError(f"station {sid}: {fname} has {ds.shape[0]} frames, expected {X.shape[0]}")
            for a in range(0, T, block):
                b = min(a + block, T)
                frames = ds[a:b]  # (b-a, 120, 120), sequential read
                X[a:b, ci] = frames[:, r0:r0 + patch, c0:c0 + patch]
        print(f"  station {sid} {name}: {time.time() - t_start:.1f}s", flush=True)

    y = ground[:X.shape[0], sid - 1].astype(np.float32)
    np.save(x_path, X)
    np.save(os.path.join(out_dir, f"station_{sid}_y.npy"), y)
    meta = {
        "station_id": sid, "name": s["name"], "network": s["network"],
        "is_test_station": int(s["test"]), "pixel_row": row, "pixel_col": col,
        "patch": patch, "patch_row0": r0, "patch_col0": c0,
        "station_offset_in_patch": [row - r0, col - c0],
        "n_frames": int(X.shape[0]), "channels": [c[0] for c in CHANNELS],
        "n_valid_insitu": int(np.isfinite(y).sum()),
    }
    with open(os.path.join(out_dir, f"station_{sid}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] station {sid} ({s['name']}): X{X.shape}, valid insitu {meta['n_valid_insitu']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solarcube_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--patch", type=int, default=7)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--stations", default=None, help="comma-separated station ids (default: all)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    stations = read_stations(a.solarcube_dir)
    if a.stations:
        keep = {int(x) for x in a.stations.split(",")}
        stations = [s for s in stations if int(s["id"]) in keep]

    # one column per station (header 1..19), no timestamp column; empty field -> NaN
    ground = np.genfromtxt(os.path.join(a.solarcube_dir, "ground_radiation.csv"),
                           delimiter=",", skip_header=1, dtype=np.float64)
    print(f"ground_radiation: {ground.shape}, NaN fraction {np.isnan(ground).mean():.3f}", flush=True)

    for s in stations:
        process_station(a.solarcube_dir, a.out_dir, s, ground, a.patch, a.block, a.max_frames, a.force)


if __name__ == "__main__":
    main()
