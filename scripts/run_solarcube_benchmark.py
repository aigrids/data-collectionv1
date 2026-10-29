"""
Train and evaluate the baseline forecaster on a SolarCube point sub-task, then compute the
AI.grids v1 Table 2 general performance metrics.

Currently supported sub-task: odd_space_point (held-out stations = test=1 in station_features.csv).

Usage:
    python3 scripts/run_solarcube_benchmark.py --cache ~/solarcube_cache --epochs 20
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

import solarcube_benchmark as sb  # noqa: E402
import solarcube_data as sd  # noqa: E402
import solarcube_splits as sp  # noqa: E402
import solarcube_table2 as t2  # noqa: E402
from solarcube_models import SolarForecaster  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--subtask", default="odd_space_point",
                    choices=["odd_space_point", "odd_time_point", "odd_spacetime_point"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--stride-train", type=int, default=24)
    ap.add_argument("--stride-eval", type=int, default=48)
    ap.add_argument("--val-stations", default="2,15")
    ap.add_argument("--use-future-sza", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-table2", action="store_true")
    ap.add_argument("--de-epochs", type=int, default=15, help="epochs per training-data-efficiency run")
    ap.add_argument("--out", default="results/benchmark")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    stations = sd.load_cache(os.path.expanduser(a.cache))
    val_ids = tuple(int(s) for s in a.val_stations.split(","))
    _, val_ids, test_ids = sd.space_split(stations, val_ids)
    kind = a.subtask.split("_")[1]                      # space | time | spacetime
    pl = sp.plan(kind, list(stations), test_ids, val_ids)
    (train_ids, train_m), (val_ids, val_m), (test_ids, test_m) = pl["train"], pl["val"], pl["test"]
    print(f"device {device} | split {kind} | train {train_ids} {train_m} | val {val_ids} {val_m} | "
          f"test {test_ids} {test_m}", flush=True)

    mean, std = sd.compute_norm_stats(stations, train_ids, months=train_m)
    train_ds = sp.restrict_windows(sd.SolarForecastDataset(stations, train_ids, mean, std, stride=a.stride_train),
                                   stations, train_m)
    val_ds = sp.restrict_windows(sd.SolarForecastDataset(stations, val_ids, mean, std, stride=a.stride_eval),
                                 stations, val_m)
    test_ds = sp.restrict_windows(sd.SolarForecastDataset(stations, test_ids, mean, std, stride=a.stride_eval),
                                  stations, test_m)
    print(f"windows: train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}", flush=True)

    def make_model():
        return SolarForecaster(in_channels=2 * len(sd.CHANNELS), use_future_sza=a.use_future_sza)

    model = make_model()
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    history = sb.fit(model, train_ds, val_ds, device, epochs=a.epochs, lr=a.lr,
                     batch_size=a.batch_size, seed=a.seed, log=lambda s: print(s, flush=True))
    train_time = time.time() - t0

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    test_res, _ = sb.evaluate(model, test_loader, device)
    print(f"\nTEST (held-out {kind}):")
    print(f"  model          MAE {test_res['model']['mae']:.1f}  RMSE {test_res['model']['rmse']:.1f}  W/m^2")
    print(f"  persistence24h MAE {test_res['persistence_24h']['mae']:.1f}  RMSE {test_res['persistence_24h']['rmse']:.1f}  W/m^2", flush=True)

    results = {
        "task_name": "SolarCube", "subtask": a.subtask,
        "config": {**vars(a), "n_params": n_params, "device": str(device)},
        "split": {"kind": kind, "train_stations": train_ids, "val_stations": val_ids, "test_stations": test_ids,
                  "months": {"train": train_m, "val": val_m, "test": test_m}},
        "n_windows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "training_seconds": train_time, "history": history, "test": test_res,
    }

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"benchmark_results_SolarCube_{a.subtask}.json")

    def save():
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

    save()
    if not a.skip_table2:
        table2 = results["table2"] = {}
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)

        print("\nTable 2: perturbation robustness ...", flush=True)
        table2["perturbation_robustness"] = t2.perturbation_robustness(model, test_ds, device)
        print(json.dumps(table2["perturbation_robustness"], indent=2), flush=True); save()

        print("Table 2: batch scaling factor ...", flush=True)
        table2["batch_scaling_factor"] = t2.batch_scaling_factor(model, test_ds, device)
        print(json.dumps(table2["batch_scaling_factor"], indent=2), flush=True); save()

        print("Table 2: computation time (on a copy of the model) ...", flush=True)
        model_copy = copy.deepcopy(model)
        table2["computation_time"] = t2.computation_time(
            model_copy, train_loader, test_ds, torch.optim.Adam(model_copy.parameters(), lr=a.lr), device)
        print(json.dumps(table2["computation_time"], indent=2), flush=True); save()

        print("Table 2: training data efficiency ...", flush=True)
        table2["training_data_efficiency"] = t2.training_data_efficiency(
            make_model, train_ds, test_ds, device, epochs=a.de_epochs, batch_size=a.batch_size, lr=a.lr)
        print(json.dumps(table2["training_data_efficiency"], indent=2), flush=True); save()

    print(f"saved {path}")


if __name__ == "__main__":
    main()
