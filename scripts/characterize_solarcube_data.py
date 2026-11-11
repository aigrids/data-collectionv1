"""
SolarCube characterization from the ACTUAL data (second pass).

scripts/characterize_solarcube.py counts the non -1 entries of availability_IDs/*.csv. Those flags do not
mean "data present" (e.g. the ssr flag is set for ~39% of frames, while ~99.6% of the ssr values are valid),
so this script counts what is really in the data, using the compact caches written by
scripts/cache_solarcube_patches.py (point) and scripts/cache_solarcube_grids.py (area). No .h5 files are read.

Per station: valid in-situ readings (total and per month), valid satellite values at the station pixel per
channel, valid ssr cells, and the number of forecasting windows (24 h history -> 24 h target, one origin every 6 h = the training stride of the benchmark)
available in each split of the six sub-tasks, using the same window rules as the benchmark
(>= 80% valid readings; area: frame usable if >= 50% of the station's typical valid ssr cells).

Usage:
    python3 scripts/characterize_solarcube_data.py --point-cache ~/solarcube_cache \
        --grid-cache ~/solarcube_grid_cache --out results/analysis/analysis_results_SolarCube_data.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import solarcube_area as sa  # noqa: E402
import solarcube_data as sd  # noqa: E402
import solarcube_splits as sp  # noqa: E402

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def window_counts(stations, make_ds, kind, val_ids):
    """windows per split {train,val,test} and per station for one sub-task kind."""
    _, val_ids, test_ids = sd.space_split(stations, val_ids)
    pl = sp.plan(kind, list(stations), test_ids, val_ids)
    mean = np.zeros(5); std = np.ones(5)
    out = {}
    for split, (ids, months) in pl.items():
        ds = sp.restrict_windows(make_ds(stations, ids, mean, std, stride=24), stations, months)
        per = {}
        for sid, _ in ds.index:
            per[sid] = per.get(sid, 0) + 1
        out[split] = {"stations": ids, "months": list(months) if months else "all",
                      "n_windows": len(ds), "windows_per_station": {str(k): v for k, v in sorted(per.items())}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point-cache", required=True)
    ap.add_argument("--grid-cache", required=True)
    ap.add_argument("--out", default="results/analysis/analysis_results_SolarCube_data.json")
    ap.add_argument("--val-stations", default="2,15")
    ap.add_argument("--flags-json", default="results/analysis/analysis_results_SolarCube.json",
                    help="first-pass (availability flag) results; their dataset_totals are copied into the output for comparison")
    a = ap.parse_args()
    val_ids = tuple(int(s) for s in a.val_stations.split(","))

    pts = sd.load_cache(os.path.expanduser(a.point_cache))
    grids = sa.load_cache(os.path.expanduser(a.grid_cache))
    flags = {}
    if os.path.exists(a.flags_json):
        flags = json.load(open(a.flags_json)).get("dataset_totals", {})

    res = {"task": "SolarCube", "note": "counts of actual valid data (not availability flags); frame 0 = 2018-01-01 00:00 UTC, 15 min",
           "stations": {}, "totals": {}, "subtasks": {}}
    tot = {"frames": 0, "insitu_valid": 0, "ssr_frames_with_data": 0}
    for sid in sorted(pts):
        d, meta = pts[sid], pts[sid]["meta"]
        X, y = d["X"], d["y"]
        n = X.shape[0]
        month = sp.month_of_frame(n)
        ok = np.isfinite(y)
        oy, ox = meta["station_offset_in_patch"]
        centre = X[:, :, oy, ox]
        valid = {c: float((centre[:, i] != (sd.CLOUD_MISSING if c == "cloud_mask" else sd.SENTINEL)).mean())
                 for i, c in enumerate(sd.CHANNELS)}
        g = grids[sid]
        ssr_frames = int((g["frame_valid"]).sum())
        entry = {
            "name": meta.get("name"), "is_test_station": bool(meta["is_test_station"]), "n_frames": int(n),
            "insitu_valid": int(ok.sum()), "insitu_valid_fraction": float(ok.mean()),
            "insitu_valid_per_month": {MONTH_NAMES[m - 1]: int(ok[month == m].sum()) for m in range(1, 13)},
            "satellite_valid_fraction_at_station_pixel": valid,
            "ssr_typical_valid_cells_of_900": float(g["ref_cells"]),
            "ssr_usable_frames": ssr_frames, "ssr_usable_fraction": ssr_frames / n,
        }
        res["stations"][str(sid)] = entry
        tot["frames"] += n; tot["insitu_valid"] += int(ok.sum()); tot["ssr_frames_with_data"] += ssr_frames
        print(f"station {sid:2d}: insitu valid {ok.mean():.1%} | ssr usable frames {ssr_frames / n:.1%} | "
              f"ssr cells {g['ref_cells']:.0f}", flush=True)
    res["totals"] = tot
    if flags:
        res["first_pass_flag_totals"] = flags   # availability_IDs flags summed over stations, NOT data presence

    for kind in ("space", "time", "spacetime"):
        res["subtasks"][f"odd_{kind}_point"] = window_counts(pts, sd.SolarForecastDataset, kind, val_ids)
        res["subtasks"][f"odd_{kind}_area"] = window_counts(grids, sa.SolarAreaDataset, kind, val_ids)
        for t in ("point", "area"):
            s = res["subtasks"][f"odd_{kind}_{t}"]
            print(f"odd_{kind}_{t}: windows " + " / ".join(f"{k} {s[k]['n_windows']}" for k in ("train", "val", "test")),
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
