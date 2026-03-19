"""MLP for PLV connectivity features: input (batch, n_pairs=171)."""

import torch
import torch.nn as nn


class ConnectivityMLP(nn.Module):
    def __init__(self, n_features: int = 171, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)