""" Turbine spatial/semantic neighbor graphs and raw cross-turbine
lookups needed by DMST (FDSTT paper, Li et al. 2022) - not available in
aigrids' per-turbine windowed records.

- Both graphs built once from the complete raw dataset (all turbines,
  all timestamps)
- Turbine positions and their wind-pattern relationships are treated as
  a static structural property of the wind farm, same as how turbine
  position is already given per-record regardless of train/test split

"""
import os

import numpy as np
import pandas as pd

TOP_K = 5


def load_raw_scada_and_location(root_path_datasets):
	""" Load the raw WindFarm CSVs directly (not through aigrids'
	windowing), needed to build graphs and look up neighbor data.
	"""
	root = os.path.join(root_path_datasets, "WindFarm")
	scada = pd.read_csv(os.path.join(root, "sdwpf_2001_2112_full.csv"))
	loc = pd.read_csv(os.path.join(root, "sdwpf_turb_location_elevation.csv"))
	scada["Tmstamp"] = pd.to_datetime(scada["Tmstamp"])
	return scada, loc


def build_position_lookup(loc):
	""" Maps (x, y) -> TurbID. aigrids records carry each turbine's
	position but not its ID directly, so this is how we recover turbine
	identity from a record.
	"""
	lookup = {}
	for _, row in loc.iterrows():
		key = (round(float(row["x"]), 3), round(float(row["y"]), 3))
		lookup[key] = int(row["TurbID"])
	return lookup


def build_spatial_graph(loc, k=TOP_K):
	""" Top-k nearest neighbors by Euclidean distance (paper eq. 2). """
	turb_ids = loc["TurbID"].to_numpy()
	coords = loc[["x", "y"]].to_numpy()

	dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
	np.fill_diagonal(dist, np.inf)

	neighbors = {}
	for i, tid in enumerate(turb_ids):
		nearest_idx = np.argsort(dist[i])[:k]
		neighbors[int(tid)] = turb_ids[nearest_idx].astype(int).tolist()
	return neighbors


def build_semantic_graph(wind_speed_wide, k=TOP_K):
	""" Top-k most similar neighbors by "differential similarity" of wind
	speed - correlation between consecutive-step changes (paper eq. 3),
	not the raw wind speed values themselves.

	wind_speed_wide: DataFrame, index=Tmstamp, columns=TurbID, values=Wspd.
	Missing diffs are treated as 0 (no contribution to similarity) -
	a simplification given data gaps, not the exact NaN handling a fully
	rigorous implementation might use.
	"""
	turb_ids = wind_speed_wide.columns.to_numpy()
	diffs = wind_speed_wide.diff().fillna(0.0).to_numpy()  # [T, N]

	sim_matrix = diffs.T @ diffs  # [N, N], paper eq. 3 summed over T
	np.fill_diagonal(sim_matrix, -np.inf)

	neighbors = {}
	for i, tid in enumerate(turb_ids):
		nearest_idx = np.argsort(-sim_matrix[i])[:k]
		neighbors[int(tid)] = turb_ids[nearest_idx].astype(int).tolist()
	return neighbors


class WindFarmGraph:
	""" Bundles the position lookup, both neighbor graphs, and a wide
	wind-speed table for fast per-record neighbor lookups.
	"""

	def __init__(self, root_path_datasets):
		scada, loc = load_raw_scada_and_location(root_path_datasets)

		self.position_to_turbid = build_position_lookup(loc)
		wind_speed_wide = scada.pivot_table(
			index="Tmstamp", columns="TurbID", values="Wspd"
		)
		self.spatial_neighbors = build_spatial_graph(loc)
		self.semantic_neighbors = build_semantic_graph(wind_speed_wide)

		self.turb_ids = sorted(loc["TurbID"].astype(int).tolist())
		self.turbid_to_idx = {tid: i for i, tid in enumerate(self.turb_ids)}
		self.n_turbines = len(self.turb_ids)

		# precomputed for fast numpy-based lookups instead of repeated
		# pandas .loc/.iloc calls, since this runs tens of thousands of
		# times when building training data
		self._wind_speed_matrix = wind_speed_wide.to_numpy(dtype=np.float32)
		self._tmstamp_to_row = {
			ts: i for i, ts in enumerate(wind_speed_wide.index)
		}
		self._col_to_idx = {
			col: i for i, col in enumerate(wind_speed_wide.columns)
		}

	def turbid_for_position(self, x, y):
		key = (round(float(x), 3), round(float(y), 3))
		return self.position_to_turbid.get(key)

	def neighbor_wind_speed(self, turb_id, end_tmstamp, window_len, kind):
		""" Aggregated (mean) wind speed across a turbine's neighbors
		(kind: "spatial" or "semantic"), for the window_len steps ending
		at end_tmstamp (inclusive). Returns a [window_len] array, NaN
		where data is missing.
		"""
		neighbors = (
			self.spatial_neighbors if kind == "spatial" else self.semantic_neighbors
		)[turb_id]

		end_pos = self._tmstamp_to_row.get(end_tmstamp)
		if end_pos is None:
			return np.full(window_len, np.nan, dtype=np.float32)

		start_pos = end_pos - window_len + 1
		if start_pos < 0:
			return np.full(window_len, np.nan, dtype=np.float32)

		col_idx = [
			self._col_to_idx[c] for c in neighbors if c in self._col_to_idx
		]
		block = self._wind_speed_matrix[start_pos:end_pos + 1][:, col_idx]
		with np.errstate(invalid="ignore"):
			return np.nanmean(block, axis=1)
