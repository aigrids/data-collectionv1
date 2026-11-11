"""
Cache pooled SolarCube grids for the area sub-tasks (odd_*_area).

Each station's 120x120 tile is reduced by POOLxPOOL block means over the valid pixels
(default 4x4 -> 30x30, i.e. 20 km cells) and stored as one int16 array per station:

    <out_dir>/station_<id>_G.npy         int16 (T, 6, 30, 30)
    <out_dir>/station_<id>_gridmeta.json

Channel order: visual_band_47, visual_band_86, infrared_band_133, cloud_mask (x1000),
solar_zenith_angle, ssr (satellite_solar_radiation, raw units, 0.1 W/m^2 expected).
A pooled cell is -9999 when fewer than half of its pixels are valid (ssr: when no pixel is valid --
large parts of some stations' ssr tiles are invalid, e.g. over the ocean).
Each file is read once in blocks of full frames (default 512 x 120 x 120 x 2 B = ~15 MB).

Also stores a check of the ssr scale: least-squares slope (through the origin) of the ground
radiation on the station's own ssr pixel.

Usage:
    python3 scripts/cache_solarcube_grids.py <SolarCube_dir> <out_dir> [--pool 4] [--block 512]
                                             [--stations 1,2,3] [--max-frames N] [--force]
"""
import argparse
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_solarcube_patches import read_stations, station_pixel  # noqa: E402

CHANNELS = [
    ("visual_band_47", "visual_band_47.h5"),
    ("visual_band_86", "visual_band_86.h5"),
    ("infrared_band_133", "infrared_band_133.h5"),
    ("cloud_mask", "cloud_mask.h5"),
    ("solar_zenith_angle", "solar_zenith_angle.h5"),
    ("ssr", "satellite_solar_radiation.h5"),
]
GRID = 120
SENTINEL = -9999
CLOUD_MISSING = -1
CLOUD_SCALE = 1000.0


def pool_block(frames, pool, missing, scale=1.0, min_cnt=None):
    """frames: int16 (b, 120, 120) -> int16 (b, 120/pool, 120/pool); mean over valid pixels."""
    b, n = frames.shape[0], GRID // pool
    valid = frames != missing
    x = np.where(valid, frames, 0).astype(np.float32).reshape(b, n, pool, n, pool)
    s = x.sum(axis=(2, 4))
    c = valid.reshape(b, n, pool, n, pool).sum(axis=(2, 4))
    mean = s / np.maximum(c, 1) * scale
    thr = (pool * pool) // 2 if min_cnt is None else min_cnt
    return np.where(c >= thr, np.rint(mean), SENTINEL).astype(np.int16)


def process_station(root, out_dir, s, ground, pool, block, max_frames, force):
    sid = int(s["id"])
    g_path = os.path.join(out_dir, f"station_{sid}_G.npy")
    if os.path.exists(g_path) and not force:
        print(f"[skip] station {sid} ({s['name']}) already cached", flush=True)
        return
    row, col = station_pixel(s)
    n = GRID // pool
    G, centre = None, None
    for ci, (name, fname) in enumerate(CHANNELS):
        t0 = time.time()
        with h5py.File(os.path.join(root, f"station_{sid}", fname), "r") as f:
            ds = f[list(f.keys())[0]]
            T = ds.shape[0] if max_frames is None else min(ds.shape[0], max_frames)
            if G is None:
                G = np.empty((T, len(CHANNELS), n, n), dtype=np.int16)
            elif T != G.shape[0]:
                raise ValueError(f"station {sid}: {fname} has {ds.shape[0]} frames, expected {G.shape[0]}")
            if name == "ssr":
                centre = np.full(T, np.nan, dtype=np.float64)
            for a in range(0, T, block):
                b = min(a + block, T)
                fr = ds[a:b].astype(np.int16)
                is_cloud = name == "cloud_mask"
                G[a:b, ci] = pool_block(fr, pool, CLOUD_MISSING if is_cloud else SENTINEL,
                                        CLOUD_SCALE if is_cloud else 1.0,
                                        min_cnt=1 if name == "ssr" else None)
                if name == "ssr":
                    c = fr[:, row, col].astype(np.float64)
                    c[c == SENTINEL] = np.nan
                    centre[a:b] = c
        print(f"  station {sid} {name}: {time.time() - t0:.1f}s", flush=True)

    g = ground[:G.shape[0], sid - 1]
    k = min(len(g), len(centre))
    ok = np.isfinite(g[:k]) & np.isfinite(centre[:k])
    slope = float((centre[:k][ok] * g[:k][ok]).sum() / max((centre[:k][ok] ** 2).sum(), 1e-9)) if ok.any() else None
    corr = float(np.corrcoef(centre[:k][ok], g[:k][ok])[0, 1]) if ok.sum() > 10 else None
    np.save(g_path, G)
    ssr_valid = float((G[:, -1] != SENTINEL).mean())
    ssr_cells = float(np.median((G[:, -1] != SENTINEL).sum(axis=(1, 2))))
    meta = {
        "station_id": sid, "name": s["name"], "network": s["network"], "is_test_station": int(s["test"]),
        "pool": pool, "grid": n, "n_frames": int(G.shape[0]), "channels": [c[0] for c in CHANNELS],
        "cloud_scale": CLOUD_SCALE, "ssr_valid_cell_fraction": ssr_valid, "ssr_median_valid_cells": ssr_cells,
        "ssr_slope_wm2_per_raw_unit": slope, "ssr_corr_with_ground": corr,
    }
    with open(os.path.join(out_dir, f"station_{sid}_gridmeta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] station {sid} ({s['name']}): G{G.shape}, ssr valid cells {ssr_valid:.3f}, "
          f"ssr->W/m2 slope {slope}, corr {corr}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solarcube_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--stations", default=None, help="comma-separated station ids (default: all)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if GRID % a.pool:
        raise SystemExit(f"--pool must divide {GRID}")
    os.makedirs(a.out_dir, exist_ok=True)
    stations = read_stations(a.solarcube_dir)
    if a.stations:
        keep = {int(x) for x in a.stations.split(",")}
        stations = [s for s in stations if int(s["id"]) in keep]
    ground = np.genfromtxt(os.path.join(a.solarcube_dir, "ground_radiation.csv"),
                           delimiter=",", skip_header=1, dtype=np.float64)
    for s in stations:
        process_station(a.solarcube_dir, a.out_dir, s, ground, a.pool, a.block, a.max_frames, a.force)


if __name__ == "__main__":
    main()
