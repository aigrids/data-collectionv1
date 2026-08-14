""" Reimplementation of DMST (Deep Multi-relational Spatio-Temporal
network) - the GRU encoder-decoder component of team HIK's FDSTT model,
1st place at KDD Cup 2022 (Li et al., "Complementary Fusion of Deep
Spatio-Temporal Network and Tree Model for Wind Power Forecasting").

Only DMST is implemented, not the LightGBM ST-Tree module or the
hand-tuned ensemble rule (fit to a different competition test setup than
our sub-tasks) - see bug_report/benchmark report for the scoping decision.

No public code exists for this paper (checked GitHub + the official
competition site); reimplemented from the paper's description. Adapted to
our task's own fixed data structure (144-step history, defined by
aigrids, not ours to change) rather than the paper's own T=432 choice.

Simplifications vs. the paper, documented rather than hidden:
- Teacher forcing during training (ratio 1.0), free-running/autoregressive
  at evaluation - standard practice, not explicitly specified in the paper.
- Training loss is MSE (paper doesn't name a specific decoder loss).
- Semantic-graph similarity uses 0-filled diffs where data is missing,
  a simplification given real gaps in the raw SCADA data.

Example usage:

	$ python scripts/dmst_windfarm.py odd_time_predict48h

"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from aigrids import load

import utils
from windfarm_graph import WindFarmGraph

ARG = sys.argv[1] if len(sys.argv) > 1 else "odd_time_predict48h"
PATH_CONFIG = "config.yml"
DATA_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1
N_EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SEED = 0

EMBED_DIM = 5
ENC_HIDDEN = 64
DEC_HIDDEN = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 256

OTHER_COLS = ["Etmp", "Itmp", "Wdir", "Ndir", "Pab1", "Prtv", "Patv"]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dmst_arrays(records, graph):
	""" Build (turb_idx, enc_seq, dec_future, last_known_power, y) from
	aigrids records, adding the cross-turbine neighbor aggregation the
	standardized record format doesn't carry (see windfarm_graph.py).
	"""
	if not records:
		return None

	hist_len = len(records[0]["features"]["Wspd"])
	horizon = len(records[0]["label"]["Patv"])
	n = len(records)

	turb_idx = np.zeros(n, dtype=np.int64)
	enc_seq = np.zeros((n, hist_len, 4 + len(OTHER_COLS) + 3), dtype=np.float32)
	dec_future = np.zeros((n, horizon, 3), dtype=np.float32)
	last_known_power = np.zeros((n, 1), dtype=np.float32)
	y = np.zeros((n, horizon), dtype=np.float32)

	keep_rows = []
	for i, rec in enumerate(records):
		feat = rec["features"]
		label = rec["label"]["Patv"]

		if any(v != v for v in label):  # NaN check without extra numpy call
			continue

		x_pos, y_pos = feat["x"], feat["y"]
		turb_id = graph.turbid_for_position(x_pos, y_pos)
		if turb_id is None:
			continue

		end_ts = pd.Timestamp(feat["Tmstamp"])
		own_wind = np.asarray(feat["Wspd"], dtype=np.float32)
		spatial_agg = graph.neighbor_wind_speed(turb_id, end_ts, hist_len, "spatial")
		semantic_agg = graph.neighbor_wind_speed(turb_id, end_ts, hist_len, "semantic")

		# x_hat_w = CONCAT(own, spatial_agg, own, semantic_agg), paper eq 4-6
		x_hat_w = np.stack([own_wind, spatial_agg, own_wind, semantic_agg], axis=1)

		hist_timestamps = [
			end_ts - pd.Timedelta(minutes=10 * (hist_len - 1 - t))
			for t in range(hist_len)
		]
		hour = np.array([ts.hour for ts in hist_timestamps], dtype=np.float32)

		other_seqs = [np.asarray(feat[c], dtype=np.float32) for c in OTHER_COLS]
		x_pos_seq = np.full(hist_len, x_pos, dtype=np.float32)
		y_pos_seq = np.full(hist_len, y_pos, dtype=np.float32)
		x_other = np.stack(other_seqs + [hour, x_pos_seq, y_pos_seq], axis=1)

		enc_seq[i] = np.concatenate([x_hat_w, x_other], axis=1)
		turb_idx[i] = graph.turbid_to_idx[turb_id]

		future_timestamps = [
			end_ts + pd.Timedelta(minutes=10 * (t + 1)) for t in range(horizon)
		]
		hour_future = np.array([ts.hour for ts in future_timestamps], dtype=np.float32)
		dec_future[i] = np.stack(
			[hour_future,
			 np.full(horizon, x_pos, dtype=np.float32),
			 np.full(horizon, y_pos, dtype=np.float32)],
			axis=1
		)

		last_known_power[i] = feat["Patv"][-1]
		y[i] = label
		keep_rows.append(i)

	print(f"Kept {len(keep_rows)}/{n} records (dropped: NaN label or unmatched position).")
	keep = np.array(keep_rows, dtype=np.int64)
	if len(keep) == 0:
		return None
	return turb_idx[keep], enc_seq[keep], dec_future[keep], last_known_power[keep], y[keep]


def channel_stats(arr):
	""" Per-channel (last-axis) mean/std for standardization. Guards
	against a channel being NaN *everywhere* (e.g. no valid neighbor data
	for that channel across every record) - without this, np.nanmean
	silently returns NaN for that channel, and (0 - NaN) / NaN then
	poisons every value passed through it, not just the originally
	missing ones. `std < 1e-6` alone doesn't catch this: comparisons
	against NaN are always False in numpy, so a NaN std slips through.
	"""
	flat = arr.reshape(-1, arr.shape[-1])
	mean = np.nanmean(flat, axis=0)
	std = np.nanstd(flat, axis=0)
	bad = np.isnan(mean) | np.isnan(std) | (std < 1e-6)
	mean = np.where(np.isnan(mean), 0.0, mean)
	std = np.where(bad, 1.0, std)
	return mean.astype(np.float32), std.astype(np.float32)


def standardize(arr, mean, std):
	arr = np.nan_to_num(arr, nan=0.0)  # impute after standardization would
	out = (arr - mean) / std           # need mean anyway; fill raw NaN with
	return np.nan_to_num(out, nan=0.0)  # 0 pre-standardization, then re-zero


class WindFarmDMSTDataset(Dataset):
	def __init__(self, turb_idx, enc_seq, dec_future, last_known_power, y):
		self.turb_idx = torch.from_numpy(turb_idx)
		self.enc_seq = torch.from_numpy(enc_seq)
		self.dec_future = torch.from_numpy(dec_future)
		self.last_known_power = torch.from_numpy(last_known_power)
		self.y = torch.from_numpy(y)

	def __len__(self):
		return len(self.turb_idx)

	def __getitem__(self, idx):
		return (
			self.turb_idx[idx], self.enc_seq[idx], self.dec_future[idx],
			self.last_known_power[idx], self.y[idx]
		)


class DMSTModel(nn.Module):
	def __init__(self, n_turbines, enc_input_dim, embed_dim=EMBED_DIM,
	             enc_hidden=ENC_HIDDEN, dec_hidden=DEC_HIDDEN):
		super().__init__()
		self.embedding = nn.Embedding(n_turbines, embed_dim)
		self.encoder_gru = nn.GRU(embed_dim + enc_input_dim, enc_hidden, batch_first=True)
		self.enc_to_dec = nn.Linear(enc_hidden, dec_hidden)
		self.decoder_cell = nn.GRUCell(embed_dim + 1 + 3, dec_hidden)
		self.readout = nn.Linear(dec_hidden, 1)

	def forward(self, turb_idx, enc_seq, dec_future, last_known_power, horizon, teacher_y=None):
		g = self.embedding(turb_idx)  # [B, embed_dim]

		g_rep = g.unsqueeze(1).expand(-1, enc_seq.shape[1], -1)
		enc_input = torch.cat([g_rep, enc_seq], dim=-1)
		_, h_n = self.encoder_gru(enc_input)
		h = self.enc_to_dec(h_n.squeeze(0))  # [B, dec_hidden]

		y_prev = last_known_power  # [B, 1]
		outputs = []
		for t in range(horizon):
			dec_in = torch.cat([g, y_prev, dec_future[:, t, :]], dim=-1)
			h = self.decoder_cell(dec_in, h)
			y_hat = self.readout(h)  # [B, 1]
			outputs.append(y_hat)
			y_prev = teacher_y[:, t:t + 1] if teacher_y is not None else y_hat

		return torch.cat(outputs, dim=1)  # [B, horizon]


def train_model(model, loader, horizon, n_epochs):
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
	loss_fn = nn.MSELoss()

	model.train()
	for epoch in range(n_epochs):
		total_loss, n_batches = 0.0, 0
		for turb_idx, enc_seq, dec_future, last_known_power, y in loader:
			turb_idx = turb_idx.to(DEVICE)
			enc_seq = enc_seq.to(DEVICE)
			dec_future = dec_future.to(DEVICE)
			last_known_power = last_known_power.to(DEVICE)
			y = y.to(DEVICE)

			optimizer.zero_grad()
			pred = model(turb_idx, enc_seq, dec_future, last_known_power, horizon, teacher_y=y)
			loss = loss_fn(pred, y)
			loss.backward()
			# unrolling the decoder over up to 432 sequential steps is
			# classic exploding-gradient territory for RNNs; clip before
			# stepping rather than let one bad batch produce NaN weights
			# that poison every batch after it
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			optimizer.step()

			total_loss += loss.item()
			n_batches += 1

		print(f"epoch {epoch + 1}/{n_epochs}: mean MSE (standardized) = {total_loss / n_batches:.4f}")


@torch.no_grad()
def predict(model, loader, horizon):
	model.eval()
	preds, targets = [], []
	for turb_idx, enc_seq, dec_future, last_known_power, y in loader:
		turb_idx = turb_idx.to(DEVICE)
		enc_seq = enc_seq.to(DEVICE)
		dec_future = dec_future.to(DEVICE)
		last_known_power = last_known_power.to(DEVICE)

		pred = model(turb_idx, enc_seq, dec_future, last_known_power, horizon)
		preds.append(pred.cpu().numpy())
		targets.append(y.numpy())

	return np.concatenate(preds), np.concatenate(targets)


def main():
	cfg = utils.parse_config(PATH_CONFIG)

	print(f"Loading WindFarm/{ARG} (data_frac={DATA_FRAC}) ...")
	taskdata = load.load_task(
		task_name="WindFarm",
		subtask_name=ARG,
		root_path=cfg["root_path_datasets"],
		data_frac=DATA_FRAC
	)

	print("Building turbine graphs (spatial + semantic neighbors)...")
	graph = WindFarmGraph(cfg["root_path_datasets"])

	print("Building DMST training arrays (train)...")
	train_arrays = build_dmst_arrays(taskdata["train_data"], graph)
	print("Building DMST training arrays (val)...")
	val_arrays = build_dmst_arrays(taskdata["val_data"], graph)
	print("Building DMST training arrays (test)...")
	test_arrays = build_dmst_arrays(taskdata["test_data"], graph)

	turb_idx_tr, enc_tr, dec_tr, lkp_tr, y_tr = train_arrays
	turb_idx_val, enc_val, dec_val, lkp_val, y_val = val_arrays
	turb_idx_test, enc_test, dec_test, lkp_test, y_test = test_arrays

	horizon = y_tr.shape[1]

	# standardize using train statistics only
	enc_mean, enc_std = channel_stats(enc_tr)
	dec_mean, dec_std = channel_stats(dec_tr)
	y_mean, y_std = float(np.nanmean(y_tr)), float(np.nanstd(y_tr))
	y_std = y_std if y_std > 1e-6 else 1.0

	enc_tr_s = standardize(enc_tr, enc_mean, enc_std)
	enc_val_s = standardize(enc_val, enc_mean, enc_std)
	enc_test_s = standardize(enc_test, enc_mean, enc_std)

	dec_tr_s = standardize(dec_tr, dec_mean, dec_std)
	dec_val_s = standardize(dec_val, dec_mean, dec_std)
	dec_test_s = standardize(dec_test, dec_mean, dec_std)

	# last_known_power can be NaN (Patv has real missing readings, ~4% per
	# our own dataset stats). Since the decoder is autoregressive, one NaN
	# here poisons that entire record's predicted sequence - and since
	# np.mean() propagates a single NaN into the whole result, one such
	# record is enough to turn the *entire* MAE/RMSE into NaN. Impute with
	# the mean (a neutral starting point, ~0 once standardized) rather
	# than leaving it to silently corrupt everything downstream.
	lkp_tr = np.where(np.isnan(lkp_tr), y_mean, lkp_tr)
	lkp_val = np.where(np.isnan(lkp_val), y_mean, lkp_val)
	lkp_test = np.where(np.isnan(lkp_test), y_mean, lkp_test)

	lkp_tr_s = (lkp_tr - y_mean) / y_std
	lkp_val_s = (lkp_val - y_mean) / y_std
	lkp_test_s = (lkp_test - y_mean) / y_std

	y_tr_s = (y_tr - y_mean) / y_std

	train_ds = WindFarmDMSTDataset(turb_idx_tr, enc_tr_s, dec_tr_s, lkp_tr_s, y_tr_s)
	val_ds = WindFarmDMSTDataset(turb_idx_val, enc_val_s, dec_val_s, lkp_val_s, y_val)  # y unstandardized, only used for eval
	test_ds = WindFarmDMSTDataset(turb_idx_test, enc_test_s, dec_test_s, lkp_test_s, y_test)

	train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
	val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

	model = DMSTModel(
		n_turbines=graph.n_turbines, enc_input_dim=enc_tr.shape[-1]
	).to(DEVICE)

	print(f"Training on {DEVICE} for {N_EPOCHS} epochs...")
	t0 = time.perf_counter()
	train_model(model, train_loader, horizon, N_EPOCHS)
	train_time_sec = time.perf_counter() - t0

	t0 = time.perf_counter()
	pred_val_s, y_val_true = predict(model, val_loader, horizon)
	pred_test_s, y_test_true = predict(model, test_loader, horizon)
	infer_time_sec = time.perf_counter() - t0

	pred_val = pred_val_s * y_std + y_mean
	pred_test = pred_test_s * y_std + y_mean

	mean_pred_val = np.tile(y_tr.mean(axis=0), (len(y_val_true), 1))
	mean_pred_test = np.tile(y_tr.mean(axis=0), (len(y_test_true), 1))

	def mae(a, b): return float(np.mean(np.abs(a - b)))
	def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))

	results = {
		"subtask_name": ARG,
		"data_frac": DATA_FRAC,
		"model": "DMST (FDSTT/HIK reimplementation, tree+ensemble modules omitted)",
		"n_epochs": N_EPOCHS,
		"n_train": int(len(turb_idx_tr)),
		"n_val": int(len(turb_idx_val)),
		"n_test": int(len(turb_idx_test)),
		"train_time_sec": train_time_sec,
		"infer_time_sec": infer_time_sec,
		"val_mae": mae(y_val_true, pred_val),
		"val_rmse": rmse(y_val_true, pred_val),
		"test_mae": mae(y_test_true, pred_test),
		"test_rmse": rmse(y_test_true, pred_test),
		"mean_baseline_val_mae": mae(y_val_true, mean_pred_val),
		"mean_baseline_val_rmse": rmse(y_val_true, mean_pred_val),
		"mean_baseline_test_mae": mae(y_test_true, mean_pred_test),
		"mean_baseline_test_rmse": rmse(y_test_true, mean_pred_test),
	}

	print(json.dumps(results, indent=2))

	path_results_root = os.path.join(cfg["root_path_results"], "dmst")
	os.makedirs(path_results_root, exist_ok=True)
	path_out = os.path.join(path_results_root, f"dmst_windfarm_{ARG}.json")

	with open(path_out, "w") as filesave:
		json.dump(results, filesave, indent=2)

	print(f"Saved results to {path_out}")


if __name__ == "__main__":
	main()
