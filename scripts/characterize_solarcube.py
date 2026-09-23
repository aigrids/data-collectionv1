"""
Manual characterization for SolarCube (AI.grids v1).

Computes dataset characterization stats (n_datapoints, train/test split,
per-modality availability) directly from the lightweight per-station
timestamp CSVs, WITHOUT loading any of the large .h5 satellite files.

This bypasses the aigrids package's own analyse.py / load_task(), which
was found to OOM-kill even at ~500GB of allocated memory on the HPC
cluster -- almost certainly because it loads full .h5 arrays into memory
rather than using lazy/metadata-only access.

Usage (run directly, no SLURM needed -- this is lightweight):
    python3 characterize_solarcube.py <path_to_SolarCube_folder> <output_json_path>
"""

import sys
import os
import csv
import json
from collections import defaultdict

MODALITY_COLUMNS = ["vis047", "vis086", "ir133", "ssr", "sza", "insitu"]


def load_station_features(solarcube_root):
    path = os.path.join(solarcube_root, "station_features.csv")
    stations = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stations[row["id"]] = {
                "name": row["name"],
                "network": row["network"],
                "test": int(row["test"]),
            }
    return stations


def count_station_timestamps(solarcube_root, station_id):
    """
    Reads one station's timestamps CSV and counts, per modality,
    how many rows have a valid (non -1) value.
    Only holds one row in memory at a time -- O(1) memory regardless
    of file size.
    """
    path = os.path.join(
        solarcube_root, "availability_IDs", f"station_{station_id}_timestamps.csv"
    )
    counts = {col: 0 for col in MODALITY_COLUMNS}
    total_rows = 0
    all_modalities_available = 0

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            row_all_available = True
            for col in MODALITY_COLUMNS:
                val = row.get(col, "-1")
                try:
                    val_int = int(val)
                except ValueError:
                    val_int = -1
                if val_int != -1:
                    counts[col] += 1
                else:
                    row_all_available = False
            if row_all_available:
                all_modalities_available += 1

    return total_rows, counts, all_modalities_available


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} <SolarCube_folder> <output_json_path>")
        sys.exit(1)

    solarcube_root = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Reading station features from: {solarcube_root}")
    stations = load_station_features(solarcube_root)
    print(f"Found {len(stations)} stations.")

    results = {
        "task": "SolarCube",
        "n_stations": len(stations),
        "n_stations_train": sum(1 for s in stations.values() if s["test"] == 0),
        "n_stations_test": sum(1 for s in stations.values() if s["test"] == 1),
        "stations": {},
        "totals": {
            "n_timesteps_per_station": None,
            "n_datapoints_total_rows": 0,
            "n_datapoints_all_modalities_available": 0,
            "n_datapoints_train_stations": 0,
            "n_datapoints_test_stations": 0,
            "modality_availability_counts": {col: 0 for col in MODALITY_COLUMNS},
        },
    }

    for station_id, meta in sorted(stations.items(), key=lambda x: int(x[0])):
        print(f"  Processing station {station_id} ({meta['name']})...")
        total_rows, counts, all_avail = count_station_timestamps(
            solarcube_root, station_id
        )

        results["stations"][station_id] = {
            "name": meta["name"],
            "network": meta["network"],
            "is_test_station": bool(meta["test"]),
            "n_timesteps": total_rows,
            "n_all_modalities_available": all_avail,
            "modality_counts": counts,
        }

        results["totals"]["n_timesteps_per_station"] = total_rows  # same for all stations
        results["totals"]["n_datapoints_total_rows"] += total_rows
        results["totals"]["n_datapoints_all_modalities_available"] += all_avail

        if meta["test"] == 0:
            results["totals"]["n_datapoints_train_stations"] += total_rows
        else:
            results["totals"]["n_datapoints_test_stations"] += total_rows

        for col in MODALITY_COLUMNS:
            results["totals"]["modality_availability_counts"][col] += counts[col]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results written to: {output_path}")
    print(json.dumps(results["totals"], indent=2))


if __name__ == "__main__":
    main()
