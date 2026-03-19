"""1D CNN for raw EEG windows: input (batch, n_channels, n_timepoints)."""

import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    def __init__(self, n_channels: int = 19, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=25, padding=12)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding=12)
        self.pool = nn.AdaptiveAvgPool1d(10)
        self.fc = nn.Linear(64 * 10, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)