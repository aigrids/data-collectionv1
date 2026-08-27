""" Quick inspection of label structure for the cascading_failure_sequence
PowerGraph sub-task

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
print(f"Number of training samples: {len(train_data)}")

# Look at first 20 samples' labels
none_count = 0
lengths = []
example_nonempty = None
import math

for i, s in enumerate(train_data[:2000]):
    label = s["labels"]
    if isinstance(label, list):
        lengths.append(len(label))
        if example_nonempty is None:
            example_nonempty = (i, label)
    else:
        none_count += 1

print(f"\nOut of first 2000 samples:")
print(f"  None (no cascade) labels: {none_count}")
print(f"  Non-None (cascade) labels: {2000 - none_count}")

if lengths:
    print(f"\nSequence length stats (non-None labels):")
    print(f"  min: {min(lengths)}, max: {max(lengths)}, mean: {sum(lengths)/len(lengths):.2f}")

if example_nonempty:
    idx, label = example_nonempty
    print(f"\nExample non-None label (sample {idx}): {label}")
    print(f"Type of label: {type(label)}")
    if len(label) > 0:
        print(f"Type of first element: {type(label[0])}")

# also check: does edge_index / num edges relate to the max value in sequences?
sample = train_data[example_nonempty[0]] if example_nonempty else train_data[0]
print(f"\nFor that sample: num edges = {sample['x_edge'].shape[0]}")
