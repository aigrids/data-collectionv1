"""Baseline forecaster for the SolarCube point sub-tasks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from solarcube_data import HIST, HORIZON


class SolarForecaster(nn.Module):
    """Per-frame CNN over the station-centred satellite patch -> GRU over the 24 h history
    (together with the past ground radiation) -> 96 future ground-radiation values.

    Output is in model units (W/m^2 / 1000) and non-negative (softplus).
    If use_future_sza is set, the astronomical solar zenith angle of the 96 future steps is
    appended to the GRU state before the output layer (off by default, to stay close to the
    inputs named in the AI.grids description: "the preceding 24 hours").
    """

    def __init__(self, in_channels=10, hidden=128, use_future_sza=False):
        super().__init__()
        self.use_future_sza = use_future_sza
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.proj = nn.Linear(32 + 2, 64)
        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.head = nn.Linear(hidden + (HORIZON if use_future_sza else 0), HORIZON)

    def forward(self, x, hist_y, fut_sza=None):
        B, T, C, H, W = x.shape
        f = self.enc(x.reshape(B * T, C, H, W)).reshape(B, T, -1)
        h = F.relu(self.proj(torch.cat([f, hist_y], dim=-1)))
        _, hn = self.gru(h)
        z = hn[-1]
        if self.use_future_sza:
            z = torch.cat([z, fut_sza], dim=-1)
        return F.softplus(self.head(z))
