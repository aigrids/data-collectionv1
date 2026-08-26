""" Train and evaluate the edge-ranking model on the
cascading_failure_sequence PowerGraph sub-task, including the full
AI.grids v1 Table 2 general performance metrics suite.

Only cascade-positive samples (non-empty failure sequences) are used,
since this model addresses "given a cascade occurs, what order do lines
fail in", not "does a cascade occur" (already covered by the existing
cascading_failure_binary model).

Example usage:

    $ python scripts/run_sequence_benchmark.py

"""
import os
import json
from pathlib import Path

import torch
from aigrids import load

import utils
import sequence
from edge_ranker import EdgeRanker

PATH_CONFIG = "config_arsam.yml"


def main():
    cfg = utils.parse_config(PATH_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading PowerGraph / cascading_failure_sequence ...")
    ds = load.load_task(
        task_name="PowerGraph",
        subtask_name="cascading_failure_sequence",
        root_path=cfg["root_path_datasets"],
    )

    print("Filtering to cascade-positive samples and converting to PyG format...")
    train_list = sequence.build_dataset(ds["train_data"])
    val_list = sequence.build_dataset(ds["val_data"])
    test_list = sequence.build_dataset(ds["test_data"])
    print(f"Cascade-positive samples -> train: {len(train_list)}, "
          f"val: {len(val_list)}, test: {len(test_list)}")

    def model_factory():
        return EdgeRanker(node_in=3, edge_in=4, hidden_channels=64)

    model = model_factory().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    best_tau, best_state = -2.0, None  # kendall tau ranges [-1, 1]
    for epoch in range(1, 31):
        loss = sequence.train_epoch(model, train_list, optimizer, device)
        val_metrics = sequence.evaluate(model, val_list, device)
        tau = val_metrics["mean_kendall_tau"] or -2.0
        print(f"Epoch {epoch:02d} | Loss: {loss:.4f} | Val metrics: {val_metrics}")
        if tau > best_tau:
            best_tau = tau
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Evaluating on test set...")
    test_metrics = sequence.evaluate(model, test_list, device)

    print("Running full benchmark suite (AI.grids v1 Table 2 general metrics)...")
    results = {
        "subtask": "cascading_failure_sequence",
        "note": "Edge-ranking model trained/evaluated only on cascade-positive "
                "samples (ground-truth gated). The separate question of whether "
                "a cascade occurs at all is covered by the existing "
                "cascading_failure_binary model and not retrained here.",
        "n_train_cascade_positive": len(train_list),
        "n_val_cascade_positive": len(val_list),
        "n_test_cascade_positive": len(test_list),
        "best_val_mean_kendall_tau": best_tau,
        "test_metrics": test_metrics,
        "computation_time": sequence.computation_time(
            model, train_list, test_list, optimizer, device),
        "batch_scaling_factor": sequence.batch_scaling_factor(
            model, test_list, device),
        "perturbation_robustness": sequence.perturbation_robustness(
            model, test_list, device),
        "training_data_efficiency": sequence.training_data_efficiency(
            model_factory, train_list, test_list, device),
    }

    path_root = os.path.join(cfg["root_path_results"], "benchmark")
    Path(path_root).mkdir(parents=True, exist_ok=True)
    path_results = os.path.join(path_root, "benchmark_results_PowerGraph_cascading_failure_sequence.json")
    with open(path_results, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {path_results}")


if __name__ == "__main__":
    main()
