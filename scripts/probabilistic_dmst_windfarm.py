""" Probabilistic DMST: estimates aleatoric/epistemic/total uncertainty
for the benchmarked model.

- Same architecture as dmst_windfarm.py, but outputs (mu, log_var) and
  trains with Gaussian NLL instead of MSE
- Ensemble: N_ENSEMBLE members, different seed each, full training set
  (not bootstrap)

Example usage:

	$ python scripts/probabilistic_dmst_windfarm.py odd_time_predict48h
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aigrids import load

import utils
from windfarm_graph import WindFarmGraph
from dmst_windfarm import (
	build_dmst_arrays, channel_stats, standardize, WindFarmDMSTDataset,
	EMBED_DIM, ENC_HIDDEN, DEC_HIDDEN, LEARNING_RATE, BATCH_SIZE,
)

ARG = sys.argv[1] if len(sys.argv) > 1 else "odd_time_predict48h"
PATH_CONFIG = "config.yml"
DATA_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1
N_EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
N_ENSEMBLE = int(sys.argv[4]) if len(sys.argv) > 4 else 5
BASE_SEED = 0

WEIGHT_DECAY = 1e-2  # deep-learning analogue of the L2=0.01 fix that
# stabilized the earlier linear heteroscedastic model near the wind-speed
# cutout region (see bug_report/benchmark report) - same rationale
# (penalize large weights so no single input can push log_var to an
# extreme via a steep learned slope), different model class.
LOGVAR_CLIP = 10.0  # keeps exp(log_variance) in a safe numerical range,
# same physical-scale justification as the linear model's clip - not
# picked to silently hide the cutout instability, see report caveat.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProbabilisticDMSTModel(nn.Module):
	""" Same architecture as DMSTModel (dmst_windfarm.py) - identical
	embedding, encoder GRU, enc-to-dec projection, decoder GRUCell -
	except the single readout head is replaced with two heads producing
	(mu, log_var) per decoder step, trained with Gaussian NLL instead of
	MSE. See module docstring for what changed and why.
	"""

	def __init__(self, n_turbines, enc_input_dim, embed_dim=EMBED_DIM,
	             enc_hidden=ENC_HIDDEN, dec_hidden=DEC_HIDDEN):
		super().__init__()
		self.embedding = nn.Embedding(n_turbines, embed_dim)
		self.encoder_gru = nn.GRU(embed_dim + enc_input_dim, enc_hidden, batch_first=True)
		self.enc_to_dec = nn.Linear(enc_hidden, dec_hidden)
		self.decoder_cell = nn.GRUCell(embed_dim + 1 + 3, dec_hidden)
		self.readout_mu = nn.Linear(dec_hidden, 1)
		self.readout_logvar = nn.Linear(dec_hidden, 1)
		# stable starting point: predict ~0 log-variance in standardized
		# space (y is standardized to unit variance) before any training
		# signal, rather than an arbitrary random init - same idea as
		# HeteroscedasticGaussianRegressor.fit's variance init.
		nn.init.zeros_(self.readout_logvar.weight)
		nn.init.zeros_(self.readout_logvar.bias)

	def forward(self, turb_idx, enc_seq, dec_future, last_known_power, horizon, teacher_y=None):
		g = self.embedding(turb_idx)  # [B, embed_dim]

		g_rep = g.unsqueeze(1).expand(-1, enc_seq.shape[1], -1)
		enc_input = torch.cat([g_rep, enc_seq], dim=-1)
		_, h_n = self.encoder_gru(enc_input)
		h = self.enc_to_dec(h_n.squeeze(0))  # [B, dec_hidden]

		y_prev = last_known_power  # [B, 1]
		mu_outputs, logvar_outputs = [], []
		for t in range(horizon):
			dec_in = torch.cat([g, y_prev, dec_future[:, t, :]], dim=-1)
			h = self.decoder_cell(dec_in, h)
			mu_t = self.readout_mu(h)  # [B, 1]
			logvar_t = torch.clamp(self.readout_logvar(h), -LOGVAR_CLIP, LOGVAR_CLIP)
			mu_outputs.append(mu_t)
			logvar_outputs.append(logvar_t)
			# feed back the predicted mean, not a sample - keeps decoding
			# deterministic and matches how teacher forcing already uses
			# ground-truth y (not a sampled y) during training
			y_prev = teacher_y[:, t:t + 1] if teacher_y is not None else mu_t

		return torch.cat(mu_outputs, dim=1), torch.cat(logvar_outputs, dim=1)


def train_probabilistic_model(model, loader, horizon, n_epochs):
	optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	loss_fn = nn.GaussianNLLLoss(full=False, reduction="mean")

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
			mu, logvar = model(turb_idx, enc_seq, dec_future, last_known_power, horizon, teacher_y=y)
			var = torch.exp(logvar)
			loss = loss_fn(mu, y, var)
			loss.backward()
			# same rationale as dmst_windfarm.py's train_model: unrolling
			# the decoder over up to hundreds of sequential steps is
			# classic exploding-gradient territory.
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			optimizer.step()

			total_loss += loss.item()
			n_batches += 1

		print(f"    epoch {epoch + 1}/{n_epochs}: mean Gaussian NLL (standardized) = {total_loss / n_batches:.4f}")


@torch.no_grad()
def predict_probabilistic(model, loader, horizon):
	model.eval()
	mus, logvars, targets = [], [], []
	for turb_idx, enc_seq, dec_future, last_known_power, y in loader:
		turb_idx = turb_idx.to(DEVICE)
		enc_seq = enc_seq.to(DEVICE)
		dec_future = dec_future.to(DEVICE)
		last_known_power = last_known_power.to(DEVICE)

		mu, logvar = model(turb_idx, enc_seq, dec_future, last_known_power, horizon)
		mus.append(mu.cpu().numpy())
		logvars.append(logvar.cpu().numpy())
		targets.append(y.numpy())

	return np.concatenate(mus), np.concatenate(logvars), np.concatenate(targets)


def main():
	cfg = utils.parse_config(PATH_CONFIG)

	print(f"Loading WindFarm/{ARG} (data_frac={DATA_FRAC}) ...")
	taskdata = load.load_task(
		task_name="WindFarm", subtask_name=ARG,
		root_path=cfg["root_path_datasets"], data_frac=DATA_FRAC
	)

	print("Building turbine graphs (spatial + semantic neighbors)...")
	graph = WindFarmGraph(cfg["root_path_datasets"])

	print("Building DMST training arrays (train)...")
	train_arrays = build_dmst_arrays(taskdata["train_data"], graph)
	print("Building DMST training arrays (test)...")
	test_arrays = build_dmst_arrays(taskdata["test_data"], graph)
	# validation split isn't used here - this script only characterizes
	# uncertainty on the held-out test set, doesn't need a tuning split.

	turb_idx_tr, enc_tr, dec_tr, lkp_tr, y_tr = train_arrays
	turb_idx_test, enc_test, dec_test, lkp_test, y_test = test_arrays

	horizon = y_tr.shape[1]

	enc_mean, enc_std = channel_stats(enc_tr)
	dec_mean, dec_std = channel_stats(dec_tr)
	y_mean, y_std = float(np.nanmean(y_tr)), float(np.nanstd(y_tr))
	y_std = y_std if y_std > 1e-6 else 1.0

	enc_tr_s = standardize(enc_tr, enc_mean, enc_std)
	enc_test_s = standardize(enc_test, enc_mean, enc_std)
	dec_tr_s = standardize(dec_tr, dec_mean, dec_std)
	dec_test_s = standardize(dec_test, dec_mean, dec_std)

	lkp_tr = np.where(np.isnan(lkp_tr), y_mean, lkp_tr)
	lkp_test = np.where(np.isnan(lkp_test), y_mean, lkp_test)
	lkp_tr_s = (lkp_tr - y_mean) / y_std
	lkp_test_s = (lkp_test - y_mean) / y_std

	y_tr_s = (y_tr - y_mean) / y_std

	train_ds = WindFarmDMSTDataset(turb_idx_tr, enc_tr_s, dec_tr_s, lkp_tr_s, y_tr_s)
	test_ds = WindFarmDMSTDataset(turb_idx_test, enc_test_s, dec_test_s, lkp_test_s, y_test)

	test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

	mu_members, var_members = [], []  # each becomes [n_test, horizon], raw kW / kW^2

	checkpoint_dir = os.path.join(cfg["root_path_results"], "advanced_metrics", "checkpoints", ARG)
	os.makedirs(checkpoint_dir, exist_ok=True)
	# shared preprocessing stats - needed to reload any member's raw-unit
	# predictions later without redoing the train/test split or graph build
	torch.save({
		"enc_mean": enc_mean, "enc_std": enc_std,
		"dec_mean": dec_mean, "dec_std": dec_std,
		"y_mean": y_mean, "y_std": y_std,
		"n_turbines": graph.n_turbines, "enc_input_dim": enc_tr.shape[-1],
		"horizon": horizon,
	}, os.path.join(checkpoint_dir, "preprocessing.pt"))

	print(f"Training {N_ENSEMBLE} probabilistic DMST members "
	      f"(different seeds, full training data each)...")
	t0 = time.perf_counter()
	for m in range(N_ENSEMBLE):
		seed = BASE_SEED + m
		torch.manual_seed(seed)
		np.random.seed(seed)
		gen = torch.Generator().manual_seed(seed)

		train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=gen)

		model = ProbabilisticDMSTModel(
			n_turbines=graph.n_turbines, enc_input_dim=enc_tr.shape[-1]
		).to(DEVICE)

		print(f"  member {m + 1}/{N_ENSEMBLE} (seed={seed}):")
		train_probabilistic_model(model, train_loader, horizon, N_EPOCHS)

		mu_s, logvar_s, _ = predict_probabilistic(model, test_loader, horizon)
		mu_members.append(mu_s * y_std + y_mean)                 # kW
		var_members.append(np.exp(logvar_s) * (y_std ** 2))      # kW^2

		checkpoint_path = os.path.join(checkpoint_dir, f"member{m}_seed{seed}.pt")
		torch.save(model.state_dict(), checkpoint_path)
		print(f"    saved checkpoint to {checkpoint_path}")

	train_time_sec = time.perf_counter() - t0

	mu_members = np.stack(mu_members)    # [M, n_test, horizon]
	var_members = np.stack(var_members)  # [M, n_test, horizon]

	mu_bar = mu_members.mean(axis=0)                              # [n_test, horizon]
	aleatoric_x = var_members.mean(axis=0)                        # (1/M) sum_m sigma_m^2(x)
	epistemic_x = ((mu_members - mu_bar) ** 2).mean(axis=0)       # (1/M) sum_m (mu_m(x)-mu_bar(x))^2
	total_x = aleatoric_x + epistemic_x                           # U_aleatoric(x) + U_epistemic(x)

	# Aggregation: mean over test instances, then mean over horizon steps.
	# Unweighted - chosen for consistency with how DMST's own MAE/RMSE are
	# aggregated flat over (record, horizon) pairs elsewhere in this
	# benchmark (dmst_windfarm.py). A documented choice, not derived from
	# H&W (they specify the ensemble combination, not this aggregation).
	aleatoric_per_h = aleatoric_x.mean(axis=0)   # [horizon]
	epistemic_per_h = epistemic_x.mean(axis=0)
	total_per_h = total_x.mean(axis=0)

	aleatoric_agg = float(aleatoric_per_h.mean())
	epistemic_agg = float(epistemic_per_h.mean())
	total_agg = float(total_per_h.mean())

	results = {
		"subtask_name": ARG,
		"data_frac": DATA_FRAC,
		"model": "Probabilistic DMST ensemble (mu, log_var heads, Gaussian NLL)",
		"n_ensemble": N_ENSEMBLE,
		"n_epochs": N_EPOCHS,
		"weight_decay": WEIGHT_DECAY,
		"logvar_clip": LOGVAR_CLIP,
		"ensemble_diversity": "different random seeds, full training data (not bootstrap)",
		"checkpoint_dir": checkpoint_dir,
		"n_train": int(len(turb_idx_tr)),
		"n_test": int(len(turb_idx_test)),
		"horizon_len": int(horizon),
		"train_time_sec": train_time_sec,
		"aleatoric_uncertainty_learned": aleatoric_agg,
		"epistemic_uncertainty_learned": epistemic_agg,
		"total_uncertainty_learned": total_agg,
		"aggregation": "mean over test instances, then mean over horizon steps (unweighted)",
		"aleatoric_uncertainty_learned_per_horizon": aleatoric_per_h.tolist(),
		"epistemic_uncertainty_learned_per_horizon": epistemic_per_h.tolist(),
		"total_uncertainty_learned_per_horizon": total_per_h.tolist(),
	}

	print(json.dumps({k: v for k, v in results.items() if not k.endswith("_per_horizon")}, indent=2))

	path_results_root = os.path.join(cfg["root_path_results"], "advanced_metrics")
	os.makedirs(path_results_root, exist_ok=True)
	path_out = os.path.join(path_results_root, f"probabilistic_dmst_windfarm_{ARG}.json")

	with open(path_out, "w") as filesave:
		json.dump(results, filesave, indent=2)

	print(f"Saved results to {path_out}")


if __name__ == "__main__":
	main()
