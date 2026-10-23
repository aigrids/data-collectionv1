"""Training / evaluation for the SolarCube point sub-tasks (masked losses, W/m^2 metrics)."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from solarcube_data import Y_SCALE


def masked_mse(pred, target, mask):
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _forward(model, b):
    return model(b["x"], b["hist_y"], b["fut_sza"])


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        b = _to_device(batch, device)
        optimizer.zero_grad()
        loss = masked_mse(_forward(model, b), b["target"], b["tmask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item() * b["target"].shape[0]
        n += b["target"].shape[0]
    return total / max(n, 1)


@torch.no_grad()
def predict(model, loader, device):
    """Returns dict of numpy arrays in W/m^2: pred, target, mask, persistence (yesterday), sid."""
    model.eval()
    out = {k: [] for k in ("pred", "target", "mask", "persistence", "sid")}
    for batch in loader:
        b = _to_device(batch, device)
        p = _forward(model, b).clamp(min=0.0)
        out["pred"].append((p * Y_SCALE).cpu().numpy())
        out["target"].append((b["target"] * Y_SCALE).cpu().numpy())
        out["mask"].append(b["tmask"].cpu().numpy())
        out["persistence"].append((b["hist_y"][..., 0] * Y_SCALE).cpu().numpy())  # missing -> 0
        out["sid"].append(np.asarray(batch["sid"]))
    return {k: np.concatenate(v) for k, v in out.items()}


def forecast_metrics(pred, target, mask):
    m = mask > 0
    err = (pred - target)[m]
    if err.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "bias": float("nan"), "n_values": 0}
    return {
        "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
        "bias": float(err.mean()), "n_values": int(m.sum()),
    }


def evaluate(model, loader, device):
    r = predict(model, loader, device)
    res = {
        "model": forecast_metrics(r["pred"], r["target"], r["mask"]),
        "persistence_24h": forecast_metrics(r["persistence"], r["target"], r["mask"]),
        "per_station": {},
    }
    for sid in np.unique(r["sid"]):
        s = r["sid"] == sid
        res["per_station"][int(sid)] = {
            "model": forecast_metrics(r["pred"][s], r["target"][s], r["mask"][s]),
            "persistence_24h": forecast_metrics(r["persistence"][s], r["target"][s], r["mask"][s]),
            "n_windows": int(s.sum()),
        }
    return res, r


def fit(model, train_ds, val_ds, device, epochs=15, lr=1e-3, batch_size=64, seed=0, log=print):
    """Adam + masked MSE; keeps the weights of the epoch with the lowest validation RMSE."""
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    best, best_state, history = float("inf"), None, []
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, tl, opt, device)
        vres, _ = evaluate(model, vl, device)
        v = vres["model"]
        history.append({"epoch": ep, "train_mse": loss, "val_mae": v["mae"], "val_rmse": v["rmse"],
                        "val_persistence_rmse": vres["persistence_24h"]["rmse"]})
        log(f"epoch {ep:3d}  train_mse {loss:.5f}  val MAE {v['mae']:.1f}  val RMSE {v['rmse']:.1f}  "
            f"(persistence RMSE {vres['persistence_24h']['rmse']:.1f})")
        if v["rmse"] < best:
            best, best_state = v["rmse"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return history
