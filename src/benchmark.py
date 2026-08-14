""" Model training, evaluation, and general performance benchmarking
functions for PowerGraph, following the AI.grids v1 metric definitions.

"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader


def make_balanced_loader(data_list, batch_size=32):
    """ Create a DataLoader with weighted random sampling to balance
    the positive/negative class distribution during training.
    """
    labels = np.array([s.y.item() for s in data_list])
    n_pos = max(labels.sum(), 1)
    n_neg = max(len(labels) - n_pos, 1)
    class_weights = {0: 1.0 / n_neg, 1: 1.0 / n_pos}
    sample_weights = np.array([class_weights[l] for l in labels])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(data_list, batch_size=batch_size, sampler=sampler)


def train_epoch(model, loader, optimizer, device, use_edge_attr=True):
    """ Run one training epoch. Returns average loss. """
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        if use_edge_attr:
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.binary_cross_entropy_with_logits(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


def get_scores_and_labels(model, loader, device, use_edge_attr=True):
    """ Returns raw sigmoid scores and true labels for a dataset. """
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if use_edge_attr:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            scores.append(torch.sigmoid(out).cpu().numpy())
            labels.append(batch.y.cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def classification_metrics(scores, labels, threshold=0.5):
    """ Accuracy, precision, recall, F1 at a given decision threshold. """
    pred = (scores > threshold).astype(float)
    tp = ((pred == 1) & (labels == 1)).sum()
    fp = ((pred == 1) & (labels == 0)).sum()
    tn = ((pred == 0) & (labels == 0)).sum()
    fn = ((pred == 0) & (labels == 1)).sum()
    acc = (tp + tn) / (tp + fp + tn + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def auroc_auprc(scores, labels):
    """ Threshold-independent classification metrics: AUROC and AUPRC.

    Proposed here as task-specific metrics for classification/forecasting
    tasks, filling a gap in AI.grids v1 Section 2.3.2, which is fully
    specified for optimization tasks but left unwritten for Forecasting,
    Simulation, and Control.
    """
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


def computation_time(model, train_loader, test_list, optimizer, device,
                      use_edge_attr=True, k_train_runs=3, n_infer_samples=500):
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
            if use_edge_attr:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            else:
                out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.binary_cross_entropy_with_logits(out, batch.y)
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
            if use_edge_attr:
                _ = model(sample.x, sample.edge_index, sample.edge_attr, batch_idx)
            else:
                _ = model(sample.x, sample.edge_index, batch_idx)
            infer_times.append(time.perf_counter() - t0)

    return {
        "avg_train_time_per_epoch_s": float(np.mean(train_times)),
        "avg_infer_time_per_graph_ms": float(np.mean(infer_times) * 1000),
    }


def batch_scaling_factor(model, test_list, device, use_edge_attr=True,
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
                    if use_edge_attr:
                        _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                    else:
                        _ = model(batch.x, batch.edge_index, batch.batch)
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


def perturbation_robustness(model, test_list, device, use_edge_attr=True,
                             sigmas=(0.01, 0.05, 0.1), n_eval=300):
    """ AI.grids v1 Table 2: perturbation robustness = (1/(N*sigma)) * sum ||y - y'||_2. """
    model.eval()
    results = {}
    with torch.no_grad():
        for sigma in sigmas:
            diffs = []
            for i in range(min(n_eval, len(test_list))):
                sample = test_list[i].to(device)
                batch_idx = torch.zeros(sample.x.shape[0], dtype=torch.long, device=device)
                if use_edge_attr:
                    out_clean = torch.sigmoid(model(sample.x, sample.edge_index, sample.edge_attr, batch_idx))
                else:
                    out_clean = torch.sigmoid(model(sample.x, sample.edge_index, batch_idx))

                noise = torch.randn_like(sample.x) * sigma
                x_perturbed = sample.x + noise
                if use_edge_attr:
                    out_noisy = torch.sigmoid(model(x_perturbed, sample.edge_index, sample.edge_attr, batch_idx))
                else:
                    out_noisy = torch.sigmoid(model(x_perturbed, sample.edge_index, batch_idx))

                diffs.append(torch.norm(out_clean - out_noisy, p=2).item())
            results[f"sigma_{sigma}"] = float(np.mean(diffs) / sigma)
    return results


def training_data_efficiency(model_factory, train_list, test_list, device,
                              fractions=(0.05, 0.1, 0.25, 0.5, 1.0), epochs=15, seed=42):
    """ AI.grids v1 Table 2: training data efficiency curve.

    model_factory: a zero-argument callable returning a fresh, untrained model.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for frac in fractions:
        n = int(len(train_list) * frac)
        idx = rng.choice(len(train_list), size=n, replace=False)
        subset = [train_list[i] for i in idx]

        torch.manual_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loader = make_balanced_loader(subset)

        for _ in range(epochs):
            train_epoch(model, loader, optimizer, device, use_edge_attr=True)

        test_loader = DataLoader(test_list, batch_size=32)
        scores, labels = get_scores_and_labels(model, test_loader, device, use_edge_attr=True)
        metrics = classification_metrics(scores, labels)
        results[f"frac_{frac}"] = {"n_samples": n, "test_f1": metrics["f1"]}
    return results
