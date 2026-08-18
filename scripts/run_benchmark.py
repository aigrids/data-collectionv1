""" Benchmark a GNN model on a PowerGraph sub-task.

Supports: cascading_failure_binary, cascading_failure_multiclass,
demand_not_served_regression only.


"""
import os
import sys
import json
from pathlib import Path

import torch
from aigrids import load
from torch_geometric.loader import DataLoader

import utils
import features
import models
import benchmark

SUBTASK = sys.argv[1] if len(sys.argv) > 1 else "cascading_failure_binary"
PATH_CONFIG = "config_arsam.yml"

if SUBTASK not in features.SUBTASK_CONFIG:
    raise ValueError(
        f"Unsupported subtask: {SUBTASK}. "
        f"Supported: {list(features.SUBTASK_CONFIG.keys())}. "
        f"(cascading_failure_sequence requires a different architecture "
        f"and is not yet implemented.)"
    )


def main():
    cfg = utils.parse_config(PATH_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task_cfg = features.SUBTASK_CONFIG[SUBTASK]
    task_type = task_cfg["task_type"]
    out_channels = task_cfg["out_channels"]

    print(f"Loading PowerGraph / {SUBTASK} (task_type={task_type}) ...")
    ds = load.load_task(
        task_name="PowerGraph",
        subtask_name=SUBTASK,
        root_path=cfg["root_path_datasets"],
    )

    print("Converting to PyG format (includes structural feature computation)...")
    train_list = features.to_pyg_dataset(ds["train_data"], task_type=task_type)
    val_list = features.to_pyg_dataset(ds["val_data"], task_type=task_type)
    test_list = features.to_pyg_dataset(ds["test_data"], task_type=task_type)

    val_loader = DataLoader(val_list, batch_size=32)
    test_loader = DataLoader(test_list, batch_size=32)
    train_loader = benchmark.make_balanced_loader(train_list, task_type=task_type)

    def model_factory():
        return models.GINE(node_in=6, edge_in=4, hidden_channels=64,
                            out_channels=out_channels, task_type=task_type)

    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    best_metric, best_state = -float("inf"), None
    for epoch in range(1, 31):
        loss = benchmark.train_epoch(model, train_loader, optimizer, device, task_type=task_type)
        val_metrics, primary_metric = benchmark.evaluate(model, val_loader, device, task_type,
                                                           n_classes=out_channels)
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val metrics: {val_metrics}")
        if primary_metric > best_metric:
            best_metric = primary_metric
            best_state = model.state_dict()

    model.load_state_dict(best_state)

    print("Running full benchmark suite...")
    test_metrics, _ = benchmark.evaluate(model, test_loader, device, task_type, n_classes=out_channels)

    results = {
        "subtask": SUBTASK,
        "task_type": task_type,
        "best_val_primary_metric": best_metric,
        "test_metrics": test_metrics,
        "computation_time": benchmark.computation_time(
            model, train_loader, test_list, optimizer, device, task_type=task_type),
        "batch_scaling_factor": benchmark.batch_scaling_factor(
            model, test_list, device),
        "perturbation_robustness": benchmark.perturbation_robustness(
    model, test_list, device, task_type=task_type),
        "training_data_efficiency": benchmark.training_data_efficiency(
            model_factory, train_list, test_list, device, task_type, n_classes=out_channels),
    }

    path_root = os.path.join(cfg["root_path_results"], "benchmark")
    Path(path_root).mkdir(parents=True, exist_ok=True)
    filename = f"benchmark_results_PowerGraph_{SUBTASK}.json"
    path_results = os.path.join(path_root, filename)
    with open(path_results, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {path_results}")


if __name__ == "__main__":
    main()
