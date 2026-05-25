# -*- coding: utf-8 -*-


import torch.nn as nn
from .config import ENCODED_DIM


class SparseAutoencoder(nn.Module):

    def __init__(self, encoded_dim: int = ENCODED_DIM):
        super().__init__()
        self._flat_dim = 64 * 8 * 8  # channels * spatial after 3 stride-2 convs on 64x64

        self.encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # -> (B, 16, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> (B, 32, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> (B, 64, 8, 8)
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(self._flat_dim, encoded_dim),
        )
        self.decoder_net = nn.Sequential(
            nn.Linear(encoded_dim, self._flat_dim),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # -> (B, 32, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # -> (B, 16, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(16,  1, 3, stride=2, padding=1, output_padding=1),  # -> (B, 1, 64, 64)
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder_net(x)
        return encoded, self.decoder_net(encoded)

    def encode(self, x):
        return self.encoder_net(x)



