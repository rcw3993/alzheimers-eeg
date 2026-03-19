"""2D CNN for STFT spectrograms: input (batch, n_channels, n_freq, n_time)."""

import torch
import torch.nn as nn


class Simple2DCNN(nn.Module):
    def __init__(self, n_channels: int = 19, n_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc = nn.Linear(64 * 4 * 4, n_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)