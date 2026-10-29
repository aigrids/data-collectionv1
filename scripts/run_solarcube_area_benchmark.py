"""
Train and evaluate the baseline area forecaster on SolarCube odd_space_area (held-out stations =
test=1 in station_features.csv), then compute the AI.grids v1 Table 2 general performance metrics.

Usage:
    python3 scripts/run_solarcube_area_benchmark.py --cache ~/solarcube_grid_cache --epochs 15
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import solarcube_area as sa  # noqa: E402
import solarcube_data as sd  # noqa: E402  (only for space_split)
import solarcube_splits as sp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--subtask", default="odd_space_area",
                    choices=["odd_space_area", "odd_time_area", "odd_spacetime_area"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--stride-train", type=int, default=24)
    ap.add_argument("--stride-eval", type=int, default=48)
    ap.add_argument("--val-stations", default="2,15")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-windows", type=int, default=None, help="debug: cap windows per split")
    ap.add_argument("--skip-table2", action="store_true")
    ap.add_argument("--de-epochs", type=int, default=10, help="epochs per training-data-efficiency run")
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    stations = sa.load_cache(os.path.expanduser(a.cache))
    val_ids = tuple(int(s) for s in a.val_stations.split(","))
    _, val_ids, test_ids = sd.space_split(stations, val_ids)
    kind = a.subtask.split("_")[1]                      # space | time | spacetime
    pl = sp.plan(kind, list(stations), test_ids, val_ids)
    (train_ids, train_m), (val_ids, val_m), (test_ids, test_m) = pl["train"], pl["val"], pl["test"]
    print(f"device {device} | threads {torch.get_num_threads()} | split {kind} | train {train_ids} {train_m} | "
          f"val {val_ids} {val_m} | test {test_ids} {test_m}", flush=True)
    slopes = {sid: stations[sid]["meta"].get("ssr_slope_wm2_per_raw_unit") for sid in stations}
    print("ssr -> W/m^2 slope per station (assumed %.3f): %s" % (
        sa.SSR_RAW_TO_WM2, {k: (round(v, 4) if v else v) for k, v in slopes.items()}), flush=True)
    bad = {k: v for k, v in slopes.items() if v is not None and not 0.07 <= v <= 0.13}
    if bad:
        raise SystemExit(f"ssr->W/m^2 slope far from the assumed {sa.SSR_RAW_TO_WM2}: {bad}. "
                         "Check SSR_RAW_TO_WM2 in src/solarcube_area.py before training.")

    mean, std = sa.compute_norm_stats(stations, train_ids, months=train_m)
    train_ds = sp.restrict_windows(sa.SolarAreaDataset(stations, train_ids, mean, std, stride=a.stride_train),
                                   stations, train_m)
    val_ds = sp.restrict_windows(sa.SolarAreaDataset(stations, val_ids, mean, std, stride=a.stride_eval),
                                 stations, val_m)
    test_ds = sp.restrict_windows(sa.SolarAreaDataset(stations, test_ids, mean, std, stride=a.stride_eval),
                                  stations, test_m)
    n_by = {name: {} for name in ("train", "val", "test")}
    for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
        for sid, _ in ds.index:
            n_by[name][sid] = n_by[name].get(sid, 0) + 1
    ref_cells = {sid: stations[sid]["ref_cells"] for sid in sorted(stations)}
    print("typical number of valid ssr cells per station (of 900):", ref_cells, flush=True)
    print("windows per station:", n_by, flush=True)
    no_win = [s for s in test_ids if n_by["test"].get(s, 0) == 0]
    if no_win or len(val_ds) == 0:
        raise SystemExit(f"no valid windows for test stations {no_win} (or empty validation set) -- check the grid cache")
    if a.max_windows:
        for ds in (train_ds, val_ds, test_ds):
            ds.index = ds.index[:a.max_windows]
    print(f"windows: train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}", flush=True)

    def make_model():
        return sa.AreaForecaster(hidden=a.hidden)

    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    history = sa.fit(model, train_ds, val_ds, device, epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
                     seed=a.seed, workers=a.workers, log=lambda s: print(s, flush=True))
    train_time = time.time() - t0

    test_res = sa.evaluate(model, torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False), device)
    print(f"\nTEST (held-out {kind}):")
    print(f"  model          MAE {test_res['model']['mae']:.1f}  RMSE {test_res['model']['rmse']:.1f}  W/m^2")
    print(f"  persistence24h MAE {test_res['persistence_24h']['mae']:.1f}  "
          f"RMSE {test_res['persistence_24h']['rmse']:.1f}  W/m^2", flush=True)

    results = {
        "task_name": "SolarCube", "subtask": a.subtask,
        "config": {**vars(a), "n_params": n_params, "device": str(device), "grid": 30, "pool": 4,
                   "ssr_raw_to_wm2_assumed": sa.SSR_RAW_TO_WM2, "ssr_slopes_vs_ground": slopes},
        "split": {"kind": kind, "train_stations": train_ids, "val_stations": val_ids, "test_stations": test_ids,
                  "months": {"train": train_m, "val": val_m, "test": test_m}},
        "n_windows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "training_seconds": train_time, "history": history, "test": test_res,
    }
    results["windows_per_station"], results["ssr_valid_cells_per_station"] = n_by, ref_cells
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"benchmark_results_SolarCube_{a.subtask}.json")

    def save():
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    save()
    if not a.skip_table2:
        t2 = results["table2"] = {}
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)

        print("\nTable 2: perturbation robustness ...", flush=True)
        t2["perturbation_robustness"] = sa.perturbation_robustness(model, test_ds, device)
        print(json.dumps(t2["perturbation_robustness"], indent=2), flush=True); save()

        print("Table 2: batch scaling factor ...", flush=True)
        t2["batch_scaling_factor"] = sa.batch_scaling_factor(model, test_ds, device)
        print(json.dumps(t2["batch_scaling_factor"], indent=2), flush=True); save()

        print("Table 2: computation time (on a copy of the model) ...", flush=True)
        model_copy = copy.deepcopy(model)
        t2["computation_time"] = sa.computation_time(
            model_copy, train_loader, test_ds, torch.optim.Adam(model_copy.parameters(), lr=a.lr), device)
        print(json.dumps(t2["computation_time"], indent=2), flush=True); save()

        print("Table 2: training data efficiency ...", flush=True)
        t2["training_data_efficiency"] = sa.training_data_efficiency(
            make_model, train_ds, test_ds, device, epochs=a.de_epochs, batch_size=a.batch_size, lr=a.lr)
        print(json.dumps(t2["training_data_efficiency"], indent=2), flush=True); save()

    print(f"saved {path}")


if __name__ == "__main__":
    main()
