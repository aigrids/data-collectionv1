"""
AI.grids v1 Table 2 general performance metrics for the SolarCube forecaster.

Definitions mirror src/benchmark.py (PowerGraph), with these task-specific adaptations:
  * computation_time: same protocol (3 training passes over the train loader, mean inference
    time of 500 single-instance forward passes); an instance is one 24h->24h window.
  * batch_scaling_factor: same formula T_b / (b * T_1 * (n // b)); only the model forward pass is
    timed on pre-collated windows (PowerGraph also times loader collation, which for SolarCube
    would measure NumPy preprocessing instead of the model). One untimed warm-up pass is added.
  * perturbation_robustness: same formula (1/(N*sigma)) * sum ||y - y'||_2, with Gaussian noise
    (sigma in units of the normalised inputs) added to the five normalised satellite channels at
    valid pixels only (masks and the ground-radiation history are left untouched). y is the
    96-value forecast in W/m^2, so the L2 norm is over 96 outputs; `per_step_rms` divides by
    sqrt(96) for an interpretable per-time-step figure.
  * training_data_efficiency: same protocol (random subsets, fresh model, fixed epochs, no
    early stopping, evaluated on the test set); primary_metric = -RMSE (W/m^2), as for
    PowerGraph regression.
"""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import solarcube_benchmark as sb
from solarcube_data import HORIZON, Y_SCALE

N_VALUE_CH = 5  # x[:, :, :5] are the normalised value channels, x[:, :, 5:] the validity masks


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _collate_first(ds, n, device):
    items = [ds[i] for i in range(min(n, len(ds)))]
    return {k: torch.stack([it[k] for it in items]).to(device) for k in ("x", "hist_y", "fut_sza")}


def computation_time(model, train_loader, test_ds, optimizer, device, k_train_runs=3, n_infer_samples=500):
    """Average training time per epoch and average inference time per window.
    NOTE: this trains `model` further -- pass a copy."""
    train_times = []
    for _ in range(k_train_runs):
        model.train()
        _sync(device)
        t0 = time.perf_counter()
        for batch in train_loader:
            b = sb._to_device(batch, device)
            optimizer.zero_grad()
            loss = sb.masked_mse(sb._forward(model, b), b["target"], b["tmask"])
            loss.backward()
            optimizer.step()
        _sync(device)
        train_times.append(time.perf_counter() - t0)

    model.eval()
    n = min(n_infer_samples, len(test_ds))
    data = _collate_first(test_ds, n, device)
    infer_times = []
    with torch.no_grad():
        for i in range(n):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(data["x"][i:i + 1], data["hist_y"][i:i + 1], data["fut_sza"][i:i + 1])
            _sync(device)
            infer_times.append(time.perf_counter() - t0)
    return {
        "avg_train_time_per_epoch_s": float(np.mean(train_times)),
        "avg_infer_time_per_window_ms": float(np.mean(infer_times) * 1000),
    }


def batch_scaling_factor(model, test_ds, device, batch_sizes=(4, 8, 16, 32, 64), n_windows=256, n_repeats=3):
    """Batch scaling factor = T_b / (b * T_1) per instance, as in src/benchmark.py."""
    model.eval()
    data = _collate_first(test_ds, n_windows, device)
    n = data["x"].shape[0]

    def run(bs):
        with torch.no_grad():
            for a in range(0, n, bs):
                model(data["x"][a:a + bs], data["hist_y"][a:a + bs], data["fut_sza"][a:a + bs])

    def time_batches(bs):
        run(bs)  # warm-up (untimed)
        times = []
        for _ in range(n_repeats):
            _sync(device)
            t0 = time.perf_counter()
            run(bs)
            _sync(device)
            times.append(time.perf_counter() - t0)
        return float(np.mean(times))

    t1_per_instance = time_batches(1) / n
    results = {"n_windows": n}
    for b in batch_sizes:
        tb = time_batches(b)
        results[f"batch_{b}"] = {"total_time_s": tb, "scaling_factor": tb / (b * t1_per_instance * (n // b))}
    return results


def perturbation_robustness(model, test_ds, device, sigmas=(0.01, 0.05, 0.1), n_eval=300, seed=0, chunk=100):
    model.eval()
    data = _collate_first(test_ds, n_eval, device)
    n = data["x"].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    results = {}
    with torch.no_grad():
        for sigma in sigmas:
            norms = []
            for a in range(0, n, chunk):
                x = data["x"][a:a + chunk]
                hy, fs = data["hist_y"][a:a + chunk], data["fut_sza"][a:a + chunk]
                clean = model(x, hy, fs) * Y_SCALE
                noise = (torch.randn(x[:, :, :N_VALUE_CH].shape, generator=gen) * sigma).to(device)
                xp = x.clone()
                xp[:, :, :N_VALUE_CH] += noise * x[:, :, N_VALUE_CH:]  # valid pixels only
                noisy = model(xp, hy, fs) * Y_SCALE
                norms.append(torch.norm(clean - noisy, p=2, dim=1))
            d = float(torch.cat(norms).mean())
            results[f"sigma_{sigma}"] = {"robustness_l2": d / sigma, "per_step_rms": d / sigma / np.sqrt(HORIZON)}
    return results


def training_data_efficiency(model_factory, train_ds, test_ds, device,
                             fractions=(0.05, 0.1, 0.25, 0.5, 1.0), epochs=15, seed=42,
                             batch_size=64, lr=1e-3):
    rng = np.random.default_rng(seed)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    results = {}
    for frac in fractions:
        n = int(len(train_ds) * frac)
        idx = rng.choice(len(train_ds), size=n, replace=False)
        loader = DataLoader(Subset(train_ds, idx.tolist()), batch_size=batch_size, shuffle=True)
        torch.manual_seed(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            sb.train_epoch(model, loader, optimizer, device)
        res, _ = sb.evaluate(model, test_loader, device)
        results[f"frac_{frac}"] = {
            "n_samples": n, "primary_metric": -res["model"]["rmse"],
            "rmse": res["model"]["rmse"], "mae": res["model"]["mae"],
            "persistence_rmse": res["persistence_24h"]["rmse"],
        }
    return results
