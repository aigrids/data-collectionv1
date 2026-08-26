""" Training, evaluation, and data conversion for the cascading_failure_sequence
PowerGraph sub-task.

Approach: only the subset of samples with an actual cascade (a non-empty
failure sequence) is used to train the edge-ranking model (src/edge_ranker.py).
The complementary "does a cascade happen at all" question is already covered
by the existing cascading_failure_binary model and is not retrained here;
this module focuses specifically on: given that a cascade occurs, in what
order do lines fail?

Evaluation metrics:
- top-1 accuracy: does the model's highest-scored edge match the true
  first failure?
- top-k accuracy (k=3): is the true first failure within the model's
  top-3 scored edges?
- mean Kendall's tau: rank correlation between predicted and true order,
  computed only over the edges that appear in the true sequence.

Also includes AI.grids v1 Table 2 general performance metrics, adapted
to this sub-task's per-sample training loop and (for batch scaling)
batched evaluation via torch_geometric.data.Batch, which the EdgeRanker
model supports natively since it has no graph-level pooling step.

"""
import time
import numpy as np
import torch
from torch_geometric.data import Data, Batch


def to_pyg_data(sample):
    """ Convert one cascading_failure_sequence data record into a PyG
    Data object, keeping the failure order as a 0-indexed list attribute.
    Returns None if the sample has no cascade (empty/non-list label).
    """
    label = sample["labels"]
    if not isinstance(label, list):
        return None

    x_node = sample["x_node"]
    edge_index_raw = sample["edge_index"] - 1  # 0-indexed, matches other subtasks

    x = torch.tensor(x_node, dtype=torch.float)
    edge_index = torch.tensor(edge_index_raw, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(sample["x_edge"], dtype=torch.float)

    # sequence values are 1-indexed edge positions -> convert to 0-indexed
    seq = [int(v) - 1 for v in label]

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.seq = seq  # stored as a plain python list attribute
    data.n_edges = edge_attr.shape[0]
    return data


def build_dataset(raw_data_list):
    """ Filter to only cascade-positive samples and convert to PyG format. """
    out = []
    for s in raw_data_list:
        d = to_pyg_data(s)
        if d is not None:
            out.append(d)
    return out


def pairwise_ranking_loss(scores, seq, n_edges, margin=1.0, n_negative_samples=5):
    """ Margin ranking loss encouraging:
    (a) earlier-failing edges to score higher than later-failing edges
        within the true sequence, and
    (b) every edge in the sequence to score higher than a sample of
        edges NOT in the sequence.
    """
    device = scores.device
    losses = []

    for i in range(len(seq) - 1):
        higher = scores[seq[i]]
        lower = scores[seq[i + 1]]
        losses.append(torch.clamp(margin - (higher - lower), min=0))

    seq_set = set(seq)
    non_seq = [e for e in range(n_edges) if e not in seq_set]
    if len(non_seq) > 0:
        for e in seq:
            neg_idx = np.random.choice(non_seq, size=min(n_negative_samples, len(non_seq)), replace=False)
            for neg in neg_idx:
                losses.append(torch.clamp(margin - (scores[e] - scores[neg]), min=0))

    if len(losses) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def train_epoch(model, dataset, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for data in dataset:
        data = data.to(device)
        optimizer.zero_grad()
        scores = model(data.x, data.edge_index, data.edge_attr)
        loss = pairwise_ranking_loss(scores, data.seq, data.n_edges)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _kendall_tau(true_order, pred_scores_for_seq):
    n = len(true_order)
    if n < 2:
        return None
    pred_ranking = np.argsort(-np.array(pred_scores_for_seq))
    concordant, discordant = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            pred_pos_i = np.where(pred_ranking == i)[0][0]
            pred_pos_j = np.where(pred_ranking == j)[0][0]
            if pred_pos_i < pred_pos_j:
                concordant += 1
            else:
                discordant += 1
    total_pairs = concordant + discordant
    return (concordant - discordant) / total_pairs if total_pairs > 0 else None


def evaluate(model, dataset, device, k=3):
    model.eval()
    top1_hits, topk_hits, taus = 0, 0, []
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            scores = model(data.x, data.edge_index, data.edge_attr).cpu().numpy()
            seq = data.seq

            top1_pred = int(np.argmax(scores))
            if top1_pred == seq[0]:
                top1_hits += 1

            topk_pred = set(np.argsort(-scores)[:k].tolist())
            if seq[0] in topk_pred:
                topk_hits += 1

            scores_for_seq = [scores[e] for e in seq]
            tau = _kendall_tau(list(range(len(seq))), scores_for_seq)
            if tau is not None:
                taus.append(tau)

    n = len(dataset)
    return {
        "top1_accuracy": top1_hits / n if n > 0 else 0.0,
        f"top{k}_accuracy": topk_hits / n if n > 0 else 0.0,
        "mean_kendall_tau": float(np.mean(taus)) if taus else None,
        "n_samples_evaluated": n,
    }


# ---------------------------------------------------------------------
# AI.grids v1 Table 2: general performance metrics
# ---------------------------------------------------------------------

def computation_time(model, train_list, test_list, optimizer, device,
                      k_train_runs=3, n_infer_samples=500):
    """ Average training time (per epoch, over the cascade-positive
    training subset) and average inference time (per graph).
    """
    train_times = []
    for _ in range(k_train_runs):
        model.train()
        t0 = time.perf_counter()
        for data in train_list:
            data = data.to(device)
            optimizer.zero_grad()
            scores = model(data.x, data.edge_index, data.edge_attr)
            loss = pairwise_ranking_loss(scores, data.seq, data.n_edges)
            loss.backward()
            optimizer.step()
        train_times.append(time.perf_counter() - t0)

    model.eval()
    infer_times = []
    with torch.no_grad():
        for i in range(min(n_infer_samples, len(test_list))):
            data = test_list[i].to(device)
            t0 = time.perf_counter()
            _ = model(data.x, data.edge_index, data.edge_attr)
            infer_times.append(time.perf_counter() - t0)

    return {
        "avg_train_time_per_epoch_s": float(np.mean(train_times)),
        "avg_infer_time_per_graph_ms": float(np.mean(infer_times) * 1000),
    }


def batch_scaling_factor(model, test_list, device,
                          batch_sizes=(4, 8, 16, 32, 64), n_graphs=256, n_repeats=3):
    """ Batch scaling factor = T_b / (b * T_1). The EdgeRanker model has
    no graph-level pooling step, so it can process a batched
    torch_geometric Batch object directly (edges only ever connect
    within their own graph, so message passing remains correct after
    offsetting); this makes true batched inference straightforward.
    """
    model.eval()
    n_graphs = min(n_graphs, len(test_list))
    subset = test_list[:n_graphs]

    def time_batch(batch_size):
        times = []
        with torch.no_grad():
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                for start in range(0, n_graphs, batch_size):
                    chunk = subset[start:start + batch_size]
                    batch = Batch.from_data_list(chunk).to(device)
                    _ = model(batch.x, batch.edge_index, batch.edge_attr)
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


def perturbation_robustness(model, test_list, device,
                             sigmas=(0.01, 0.05, 0.1), n_eval=300):
    """ Perturbation robustness = (1/(N*sigma)) * sum ||scores - scores'||_2,
    computed over each graph's full edge-score vector.
    """
    model.eval()
    results = {}
    n_eval = min(n_eval, len(test_list))
    with torch.no_grad():
        for sigma in sigmas:
            diffs = []
            for i in range(n_eval):
                data = test_list[i].to(device)
                scores_clean = model(data.x, data.edge_index, data.edge_attr)

                noise = torch.randn_like(data.x) * sigma
                x_perturbed = data.x + noise
                scores_noisy = model(x_perturbed, data.edge_index, data.edge_attr)

                diffs.append(torch.norm(scores_clean - scores_noisy, p=2).item())
            results[f"sigma_{sigma}"] = float(np.mean(diffs) / sigma)
    return results


def training_data_efficiency(model_factory, train_list, test_list, device,
                              fractions=(0.05, 0.1, 0.25, 0.5, 1.0), epochs=15, seed=42):
    """ Test-set mean Kendall's tau as a function of training set size. """
    rng = np.random.default_rng(seed)
    results = {}
    for frac in fractions:
        n = max(int(len(train_list) * frac), 2)  # need at least a couple samples
        idx = rng.choice(len(train_list), size=n, replace=False)
        subset = [train_list[i] for i in idx]

        torch.manual_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for _ in range(epochs):
            train_epoch(model, subset, optimizer, device)

        metrics = evaluate(model, test_list, device)
        tau = metrics["mean_kendall_tau"]
        results[f"frac_{frac}"] = {"n_samples": n, "mean_kendall_tau": tau}
    return results
