"""
SolarCube area sub-tasks (odd_space_area, ...): data, model, training, evaluation and the
AI.grids v1 Table 2 metrics, on the pooled 30x30 grids written by scripts/cache_solarcube_grids.py.

Sample = (station, origin t): the previous HIST=96 steps (24 h at 15 min) of
  * the area radiation grid `ssr` (full 15-min resolution, history = the "historic data"), and
  * visual_band_47/86, infrared_band_133, cloud_mask, solar_zenith_angle (hourly means, + validity),
are the input; the next HORIZON=96 steps of the `ssr` grid are the target.

Input tensor (C=360, H, W):  [ssr history 96 | ssr validity hourly 24 | 5 satellite channels x 24 h | their validity x 24 h]
Units: ssr raw units are 0.1 W/m^2; the model works in units of 1000 W/m^2 (raw * 1e-4).
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

HIST = 96
HORIZON = 96
N_HOURS = 24
SENTINEL = -9999
N_OTHER = 5            # visual_band_47, visual_band_86, infrared_band_133, cloud_mask, solar_zenith_angle
SSR_IDX = 5
SSR_RAW_TO_WM2 = 0.1   # to be confirmed by the slope stored in the cache meta
Y_SCALE = 1000.0       # W/m^2 -> model units
RAW_TO_MODEL = SSR_RAW_TO_WM2 / Y_SCALE
OTHER_VAL = slice(HIST + N_HOURS, HIST + N_HOURS + N_HOURS * N_OTHER)
OTHER_MASK = slice(HIST + N_HOURS + N_HOURS * N_OTHER, HIST + N_HOURS + 2 * N_HOURS * N_OTHER)
IN_CHANNELS = HIST + N_HOURS + 2 * N_HOURS * N_OTHER


# ----------------------------------------------------------------------------- data
def load_cache(cache_dir, station_ids=None):
    stations = {}
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith("_gridmeta.json"):
            continue
        with open(os.path.join(cache_dir, fn)) as f:
            meta = json.load(f)
        sid = meta["station_id"]
        if station_ids is not None and sid not in station_ids:
            continue
        G = np.load(os.path.join(cache_dir, f"station_{sid}_G.npy"), mmap_mode="r")
        cells = np.empty(G.shape[0], dtype=np.int32)      # valid ssr cells per frame
        for a in range(0, G.shape[0], 4096):
            cells[a:a + 4096] = (np.asarray(G[a:a + 4096, SSR_IDX]) != SENTINEL).sum(axis=(1, 2))
        # a frame is usable when at least half of the station's typical valid area is present
        ref = float(np.median(cells[cells > 0])) if (cells > 0).any() else 0.0
        frame_valid = (cells >= 0.5 * ref) & (cells > 0) if ref > 0 else np.zeros(G.shape[0], dtype=bool)
        stations[sid] = {"meta": meta, "G": G, "frame_valid": frame_valid, "ref_cells": ref}
    return stations


def compute_norm_stats(stations, ids, step=97):
    """Mean/std of the 5 satellite channels over valid cells of the training stations."""
    s1, s2, n = np.zeros(N_OTHER), np.zeros(N_OTHER), np.zeros(N_OTHER)
    for sid in ids:
        g = np.asarray(stations[sid]["G"][::step, :N_OTHER])
        for c in range(N_OTHER):
            xs = g[:, c][g[:, c] != SENTINEL].astype(np.float64)
            s1[c] += xs.sum(); s2[c] += (xs ** 2).sum(); n[c] += xs.size
    mean = s1 / np.maximum(n, 1)
    std = np.sqrt(np.maximum(s2 / np.maximum(n, 1) - mean ** 2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class SolarAreaDataset(Dataset):
    def __init__(self, stations, ids, mean, std, stride=24, min_valid=0.8):
        self.stations, self.mean, self.std = stations, mean, std
        self.index = []
        for sid in ids:
            fv = stations[sid]["frame_valid"]
            c = np.concatenate([[0], np.cumsum(fv)])
            for t in range(HIST, len(fv) - HORIZON + 1, stride):
                if (c[t] - c[t - HIST]) / HIST >= min_valid and (c[t + HORIZON] - c[t]) / HORIZON >= min_valid:
                    self.index.append((sid, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sid, t = self.index[i]
        G = self.stations[sid]["G"]
        h = np.asarray(G[t - HIST:t]).astype(np.float32)             # (96, 6, H, W)
        hv = h != SENTINEL
        H, W = h.shape[-2:]
        ssr = np.where(hv[:, SSR_IDX], h[:, SSR_IDX] * RAW_TO_MODEL, 0.0)
        ssr_m = hv[:, SSR_IDX].reshape(N_HOURS, 4, H, W).mean(axis=1)
        o, ov = h[:, :N_OTHER], hv[:, :N_OTHER]
        o = np.where(ov, (o - self.mean[None, :, None, None]) / self.std[None, :, None, None], 0.0)
        cnt = ov.reshape(N_HOURS, 4, N_OTHER, H, W).sum(axis=1)
        o_h = o.reshape(N_HOURS, 4, N_OTHER, H, W).sum(axis=1) / np.maximum(cnt, 1)
        x = np.concatenate([ssr, ssr_m, o_h.reshape(-1, H, W), (cnt / 4.0).reshape(-1, H, W)], axis=0)
        tg = np.asarray(G[t:t + HORIZON, SSR_IDX]).astype(np.float32)
        tm = tg != SENTINEL
        target = np.where(tm, tg * RAW_TO_MODEL, 0.0)
        return {"x": torch.from_numpy(x.astype(np.float32)), "target": torch.from_numpy(target.astype(np.float32)),
                "tmask": torch.from_numpy(tm.astype(np.float32)), "sid": sid, "t": t}


# ----------------------------------------------------------------------------- model
class AreaForecaster(nn.Module):
    """1x1 conv -> 3 dilated 3x3 residual convs -> 1x1 conv to 96 steps, added to yesterday's grid
    (the head starts at zero, so training starts from the 24h-persistence forecast)."""

    def __init__(self, in_channels=IN_CHANNELS, hidden=48):
        super().__init__()
        self.inp = nn.Conv2d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList([nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d) for d in (1, 2, 4)])
        self.head = nn.Conv2d(hidden, HORIZON, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        h = F.relu(self.inp(x))
        for conv in self.blocks:
            h = h + F.relu(conv(h))
        return F.relu(self.head(h) + x[:, :HIST])


# ----------------------------------------------------------------------------- training / evaluation
def masked_mse(pred, target, mask):
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        x, tg, m = batch["x"].to(device), batch["target"].to(device), batch["tmask"].to(device)
        optimizer.zero_grad()
        loss = masked_mse(model(x), tg, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * x.shape[0]
        n += x.shape[0]
    return total / max(n, 1)


def _metrics(a):
    n, sa, sq, se = a
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n_values": 0}
    return {"mae": float(sa / n), "rmse": float(np.sqrt(sq / n)), "bias": float(se / n), "n_values": int(n)}


@torch.no_grad()
def evaluate(model, loader, device):
    """MAE/RMSE/bias in W/m^2 over valid target cells for the model and 24h persistence (yesterday's
    grid, missing -> 0); accumulated online, per station and overall."""
    model.eval()
    acc, nwin = {}, {}
    for batch in loader:
        x, tg, m = batch["x"].to(device), batch["target"].to(device) * Y_SCALE, batch["tmask"].to(device) > 0
        sids = np.asarray(batch["sid"])
        for name, p in (("model", model(x) * Y_SCALE), ("persistence_24h", x[:, :HIST] * Y_SCALE)):
            e = ((p - tg) * m).double()
            s = torch.stack([m.sum((1, 2, 3)).double(), e.abs().sum((1, 2, 3)), (e ** 2).sum((1, 2, 3)),
                             e.sum((1, 2, 3))], dim=1).cpu().numpy()
            for j, sid in enumerate(sids):
                acc.setdefault(int(sid), {}).setdefault(name, np.zeros(4))
                acc[int(sid)][name] += s[j]
        for sid in sids:
            nwin[int(sid)] = nwin.get(int(sid), 0) + 1
    res = {"model": _metrics(sum(a["model"] for a in acc.values())),
           "persistence_24h": _metrics(sum(a["persistence_24h"] for a in acc.values())), "per_station": {}}
    for sid, a in acc.items():
        res["per_station"][sid] = {"model": _metrics(a["model"]), "persistence_24h": _metrics(a["persistence_24h"]),
                                   "n_windows": nwin[sid]}
    return res


def fit(model, train_ds, val_ds, device, epochs=15, lr=1e-3, batch_size=32, seed=0, workers=0, log=print):
    """Adam + masked MSE; keeps the weights of the epoch with the lowest validation RMSE."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=workers)
    best, best_state, history = float("inf"), None, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(model, tl, opt, device)
        v = evaluate(model, vl, device)
        history.append({"epoch": ep, "train_mse": loss, "val_mae": v["model"]["mae"], "val_rmse": v["model"]["rmse"],
                        "val_persistence_rmse": v["persistence_24h"]["rmse"]})
        log(f"epoch {ep:3d}  train_mse {loss:.5f}  val MAE {v['model']['mae']:.1f}  val RMSE {v['model']['rmse']:.1f}  "
            f"(persistence RMSE {v['persistence_24h']['rmse']:.1f})  [{time.time() - t0:.0f}s]")
        if v["model"]["rmse"] < best:
            best, best_state = v["model"]["rmse"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history


# ----------------------------------------------------------------------------- Table 2 metrics
# Definitions follow src/benchmark.py (PowerGraph) and src/solarcube_table2.py (point sub-tasks);
# an instance is one 24h->24h window of one 30x30 area.
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _collate_first(ds, n, device):
    return torch.stack([ds[i]["x"] for i in range(min(n, len(ds)))]).to(device)


def computation_time(model, train_loader, test_ds, optimizer, device, k_train_runs=3, n_infer_samples=500):
    """Average training time per epoch and average inference time per window (trains `model`: pass a copy)."""
    train_times = []
    for _ in range(k_train_runs):
        _sync(device)
        t0 = time.perf_counter()
        train_epoch(model, train_loader, optimizer, device)
        _sync(device)
        train_times.append(time.perf_counter() - t0)
    model.eval()
    data = _collate_first(test_ds, n_infer_samples, device)
    times = []
    with torch.no_grad():
        for i in range(data.shape[0]):
            _sync(device)
            t0 = time.perf_counter()
            model(data[i:i + 1])
            _sync(device)
            times.append(time.perf_counter() - t0)
    return {"avg_train_time_per_epoch_s": float(np.mean(train_times)),
            "avg_infer_time_per_window_ms": float(np.mean(times) * 1000)}


def batch_scaling_factor(model, test_ds, device, batch_sizes=(4, 8, 16, 32, 64), n_windows=256, n_repeats=3):
    """T_b / (b * T_1 * (n // b)) with forward passes only on pre-collated windows (one untimed warm-up)."""
    model.eval()
    data = _collate_first(test_ds, n_windows, device)
    n = data.shape[0]

    def time_batches(bs):
        def run():
            with torch.no_grad():
                for a in range(0, n, bs):
                    model(data[a:a + bs])
        run()
        ts = []
        for _ in range(n_repeats):
            _sync(device)
            t0 = time.perf_counter()
            run()
            _sync(device)
            ts.append(time.perf_counter() - t0)
        return float(np.mean(ts))

    t1 = time_batches(1) / n
    out = {"n_windows": n}
    for b in batch_sizes:
        tb = time_batches(b)
        out[f"batch_{b}"] = {"total_time_s": tb, "scaling_factor": tb / (b * t1 * max(n // b, 1))}
    return out


def perturbation_robustness(model, test_ds, device, sigmas=(0.01, 0.05, 0.1), n_eval=300, seed=0, chunk=50):
    """(1/(N sigma)) sum ||y - y'||_2 with Gaussian noise (normalised units) added to the satellite value
    channels at valid cells only (validity masks and the ssr history are left untouched). y is the
    96 x H x W forecast in W/m^2; `per_value_rms` divides by sqrt(number of output values)."""
    model.eval()
    data = _collate_first(test_ds, n_eval, device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    res = {}
    with torch.no_grad():
        for sigma in sigmas:
            norms = []
            for a in range(0, data.shape[0], chunk):
                x = data[a:a + chunk]
                clean = model(x) * Y_SCALE
                noise = (torch.randn(x[:, OTHER_VAL].shape, generator=gen) * sigma).to(device)
                xp = x.clone()
                xp[:, OTHER_VAL] += noise * (x[:, OTHER_MASK] > 0).float()
                noisy = model(xp) * Y_SCALE
                norms.append(torch.norm((clean - noisy).flatten(1), p=2, dim=1))
            d = float(torch.cat(norms).mean())
            n_out = clean[0].numel()
            res[f"sigma_{sigma}"] = {"robustness_l2": d / sigma, "per_value_rms": d / sigma / float(np.sqrt(n_out))}
    return res


def training_data_efficiency(model_factory, train_ds, test_ds, device, fractions=(0.05, 0.1, 0.25, 0.5, 1.0),
                             epochs=10, seed=42, batch_size=32, lr=1e-3):
    rng = np.random.default_rng(seed)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    res = {}
    for frac in fractions:
        n = int(len(train_ds) * frac)
        idx = rng.choice(len(train_ds), size=n, replace=False)
        loader = DataLoader(Subset(train_ds, idx.tolist()), batch_size=batch_size, shuffle=True)
        torch.manual_seed(seed)
        model = model_factory().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            train_epoch(model, loader, opt, device)
        r = evaluate(model, test_loader, device)
        res[f"frac_{frac}"] = {"n_samples": n, "primary_metric": -r["model"]["rmse"], "rmse": r["model"]["rmse"],
                               "mae": r["model"]["mae"], "persistence_rmse": r["persistence_24h"]["rmse"]}
    return res
