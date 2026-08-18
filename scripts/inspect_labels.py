""" multiclass and regression.

"""
from aigrids import load
import utils

cfg = utils.parse_config("config_arsam.yml")

for subtask in ["cascading_failure_multiclass", "demand_not_served_regression"]:
    print(f"\n=== {subtask} ===")
    ds = load.load_task(
        task_name="PowerGraph",
        subtask_name=subtask,
        root_path=cfg["root_path_datasets"],
    )
    sample = ds["train_data"][0]
    label = sample["labels"]
    print(f"Label value: {label}")
    print(f"Label type: {type(label)}")

    # check range of labels across first 500 samples
    labels_seen = set()
    for s in ds["train_data"][:500]:
        label = s["labels"]
        if isinstance(label, list):
            labels_seen.add(tuple(label))
        else:
            labels_seen.add(label)

    if subtask == "cascading_failure_multiclass":
        print(f"Distinct one-hot patterns seen (first 500 samples): {sorted(labels_seen)}")
    else:
        vals = sorted(labels_seen)
        print(f"Regression label range (first 500 samples): min={min(vals):.4f}, max={max(vals):.4f}")
        print(f"Number of distinct values: {len(vals)}")
        print(f"Number of zero values: {sum(1 for v in vals if v == 0.0)}")
