""" Model training, evaluation, and general performance benchmarking
functions for PowerGraph.

"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader


def make_balanced_loader(data_list, batch_size=32, task_type="binary"):
    """ Create a DataLoader with weighted random sampling to balance
    the class distribution during training. For regression, falls back
    to a standard shuffled loader since there are no discrete classes
    to balance.
    """
    if task_type == "regression":
        return DataLoader(data_list, batch_size=batch_size, shuffle=True)

    if task_type == "binary":
        labels = np.array([s.y.item() for s in data_list])
    else:  # multiclass
        labels = np.array([s.y.item() for s in data_list])  # class index, already scalar

    unique, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(unique, counts))
    class_weights = {cls: 1.0 / cnt for cls, cnt in freq.items()}
    sample_weights = np.array([class_weights[l] for l in labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(data_list, batch_size=batch_size, sampler=sampler)


def compute_loss(out, y, task_type):
    """ Task-appropriate loss function. """
    if task_type == "binary":
        return F.binary_cross_entropy_with_logits(out, y)
    elif task_type == "multiclass":
        return F.cross_entropy(out, y)
    elif task_type == "regression":
        return F.mse_loss(out, y)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def train_epoch(model, loader, optimizer, device, task_type="binary"):
    """ Run one training epoch. Returns average loss. """
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = compute_loss(out, batch.y, task_type)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


def get_predictions(model, loader, device, task_type="binary"):
    """ Returns raw model outputs and true labels for a dataset.
    For binary: sigmoid scores. For multiclass: softmax probabilities.
    For regression: raw predicted values.
    """
    model.eval()
    outputs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            if task_type == "binary":
                outputs.append(torch.sigmoid(out).cpu().numpy())
            elif task_type == "multiclass":
                outputs.append(F.softmax(out, dim=-1).cpu().numpy())
            else:  # regression
                outputs.append(out.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    return np.concatenate(outputs), np.concatenate(labels)


def classification_metrics(scores, labels, threshold=0.5):
    """ Binary classification: accuracy, precision, recall, F1 at a threshold. """
    pred = (scores > threshold).astype(float)
    labels = labels.reshape(-1)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    tn = ((pred == 0) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def multiclass_metrics(probs, labels, n_classes=4):
    """ Multiclass classification: accuracy and macro-averaged precision/
    recall/F1 (averaging per-class metrics equally, which matters here
    since class frequencies are likely imbalanced, mirroring the binary
    sub-task's imbalance).
    """
    pred = np.argmax(probs, axis=1)
    labels = labels.reshape(-1).astype(int)

    acc = (pred == labels).mean()

    precisions, recalls, f1s = [], [], []
    for c in range(n_classes):
        tp = ((pred == c) & (labels == c)).sum()
        fp = ((pred == c) & (labels != c)).sum()
        fn = ((pred != c) & (labels == c)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": float(acc),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": [float(f) for f in f1s],
    }


def regression_metrics(preds, labels):
    """ Regression: MAE, RMSE, and R^2. """
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    errors = preds - labels
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}


def auroc_auprc(scores, labels):
    """ Threshold-independent classification metrics for BINARY tasks only:
    AUROC and AUPRC.

    Proposed here as task-specific metrics for classification/forecasting
    tasks, filling a gap in AI.grids v1 Section 2.3.2, which is fully
    specified for optimization tasks but left unwritten for Forecasting,
    Simulation, and Control.
    """
    labels = labels.reshape(-1)
    order = np.argsort(-scores)
    labels_sorted = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    ranks = np.argsort(np.argsort(scores)) + 1
    sum_ranks_pos = ranks[labels == 1].sum()
    auroc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    tp_cum = np.cumsum(labels_sorted)
    fp_cum = np.cumsum(1 - labels_sorted)
    precision_curve = tp_cum / (tp_cum + fp_cum)
    recall_curve = tp_cum / n_pos
    recall_curve = np.concatenate([[0], recall_curve])
    precision_curve = np.concatenate([[precision_curve[0]], precision_curve])
    try:
        auprc = np.trapezoid(precision_curve, recall_curve)
    except AttributeError:
        auprc = np.trapz(precision_curve, recall_curve)

    return {"auroc": float(auroc), "auprc": float(auprc)}


def evaluate(model, loader, device, task_type, n_classes=4):
    """ Convenience wrapper: get predictions and compute the appropriate
    metrics dict for the given task_type.
    """
    outputs, labels = get_predictions(model, loader, device, task_type)
    if task_type == "binary":
        metrics = classification_metrics(outputs, labels)
        metrics.update(auroc_auprc(outputs, labels))
        primary_metric = metrics["f1"]
    elif task_type == "multiclass":
        metrics = multiclass_metrics(outputs, labels, n_classes=n_classes)
        primary_metric = metrics["macro_f1"]
    elif task_type == "regression":
        metrics = regression_metrics(outputs, labels)
        primary_metric = -metrics["rmse"]  # lower RMSE is better; negate for "higher is better" checkpoint selection
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    return metrics, primary_metric


def computation_time(model, train_loader, test_list, optimizer, device,
                      task_type="binary", k_train_runs=3, n_infer_samples=500):
    """ AI.grids v1 Table 2: average training time (per epoch) and
    average inference time (per graph).
    """
    train_times = []
    for _ in range(k_train_runs):
        model.train()
        t0 = time.perf_counter()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = compute_loss(out, batch.y, task_type)
            loss.backward()
            optimizer.step()
        train_times.append(time.perf_counter() - t0)

    model.eval()
    infer_times = []
    with torch.no_grad():
        for i in range(min(n_infer_samples, len(test_list))):
            sample = test_list[i].to(device)
            batch_idx = torch.zeros(sample.x.shape[0], dtype=torch.long, device=device)
            t0 = time.perf_counter()
            _ = model(sample.x, sample.edge_index, sample.edge_attr, batch_idx)
            infer_times.append(time.perf_counter() - t0)

    return {
        "avg_train_time_per_epoch_s": float(np.mean(train_times)),
        "avg_infer_time_per_graph_ms": float(np.mean(infer_times) * 1000),
    }


def batch_scaling_factor(model, test_list, device,
                          batch_sizes=(4, 8, 16, 32, 64), n_graphs=256, n_repeats=3):
    """ AI.grids v1 Table 2: batch scaling factor = T_b / (b * T_1). """
    model.eval()

    def time_batch(batch_size):
        loader = DataLoader(test_list[:n_graphs], batch_size=batch_size)
        times = []
        with torch.no_grad():
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                for batch in loader:
                    batch = batch.to(device)
                    _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                times.append(time.perf_counter() - t0)
        return float(np.mean(times))

    t1_total = time_batch(1)
    t1_per_instance = t1_total / n_graphs

    results = {}
    for b in batch_sizes:
        tb = time_batch(b)
        scaling_factor = tb / (b * t1_per_instance * (n_graphs // b))
        results[f"batch_{b}"] = {"total_time_s": tb, "scaling_factor": scaling_factor}
    return results


def perturbation_robustness(model, test_list, device, task_type="binary",
                             sigmas=(0.01, 0.05, 0.1), n_eval=300):
    """ AI.grids v1 Table 2: perturbation robustness = (1/(N*sigma)) * sum ||y - y'||_2.

    Outputs are normalized per task_type before comparison, so values are on
    a comparable, bounded scale for binary (sigmoid) and multiclass (softmax),
    matching the formula already used for the merged binary results. Regression
    outputs are left raw, since they are inherently unbounded and not intended
    to be compared against the classification tasks' robustness values.
    """
    model.eval()
    results = {}
    with torch.no_grad():
        for sigma in sigmas:
            diffs = []
            for i in range(min(n_eval, len(test_list))):
                sample = test_list[i].to(device)
                batch_idx = torch.zeros(sample.x.shape[0], dtype=torch.long, device=device)

                out_clean = model(sample.x, sample.edge_index, sample.edge_attr, batch_idx)
                noise = torch.randn_like(sample.x) * sigma
                x_perturbed = sample.x + noise
                out_noisy = model(x_perturbed, sample.edge_index, sample.edge_attr, batch_idx)

                if task_type == "binary":
                    out_clean = torch.sigmoid(out_clean)
                    out_noisy = torch.sigmoid(out_noisy)
                elif task_type == "multiclass":
                    out_clean = F.softmax(out_clean, dim=-1)
                    out_noisy = F.softmax(out_noisy, dim=-1)
                # regression: leave as raw values

                diffs.append(torch.norm((out_clean - out_noisy).float(), p=2).item())
            results[f"sigma_{sigma}"] = float(np.mean(diffs) / sigma)
    return results


def training_data_efficiency(model_factory, train_list, test_list, device, task_type,
                              fractions=(0.05, 0.1, 0.25, 0.5, 1.0), epochs=15, seed=42, n_classes=4):
    """ AI.grids v1 Table 2: training data efficiency curve. """
    rng = np.random.default_rng(seed)
    results = {}
    for frac in fractions:
        n = int(len(train_list) * frac)
        idx = rng.choice(len(train_list), size=n, replace=False)
        subset = [train_list[i] for i in idx]

        torch.manual_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = make_balanced_loader(subset, task_type=task_type)

        for _ in range(epochs):
            train_epoch(model, loader, optimizer, device, task_type=task_type)

        test_loader = DataLoader(test_list, batch_size=32)
        _, primary_metric = evaluate(model, test_loader, device, task_type, n_classes=n_classes)
        results[f"frac_{frac}"] = {"n_samples": n, "primary_metric": primary_metric}
    return results
