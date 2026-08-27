""" Check whether cascading_failure_sequence label values (edge indices)
are 0-indexed or 1-indexed, by scanning a larger sample for the presence
of 0.0 and checking the max value against edge counts.

"""
from aigrids import load
import utils

cfg = utils.parse_config("config_arsam.yml")

ds = load.load_task(
    task_name="PowerGraph",
    subtask_name="cascading_failure_sequence",
    root_path=cfg["root_path_datasets"],
)

train_data = ds["train_data"]

zero_seen = False
min_val_seen = None
max_val_seen = None
n_checked = 0
n_out_of_range = 0

for s in train_data:
    label = s["labels"]
    if not isinstance(label, list):
        continue
    n_checked += 1
    n_edges = s["x_edge"].shape[0]

    for v in label:
        if v == 0.0:
            zero_seen = True
        if min_val_seen is None or v < min_val_seen:
            min_val_seen = v
        if max_val_seen is None or v > max_val_seen:
            max_val_seen = v
        # if 1-indexed, valid values are 1..n_edges; check if v exceeds n_edges
        if v > n_edges:
            n_out_of_range += 1

print(f"Total samples with a non-empty sequence: {n_checked}")
print(f"Did we ever see 0.0 as a sequence value? {zero_seen}")
print(f"Min value seen across all sequences: {min_val_seen}")
print(f"Max value seen across all sequences: {max_val_seen}")
print(f"Number of sequence values exceeding that sample's edge count (n_edges): {n_out_of_range}")
