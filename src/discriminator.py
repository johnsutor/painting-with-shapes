#!/usr/bin/env python
# https://github.com/w86763777/pytorch-gan-collections/blob/master/source/models/sngan.py

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn.utils.spectral_norm import spectral_norm


class Discriminator(nn.Module):
    """Discriminator using spectral normalization. This is a simple discriminator that takes an image
    and returns a single value.
    """

    def __init__(
        self, embedding_size: int, embedding_dim: int = 508, image_size: int = 128
    ):
        """Args:
        image_size (int, optional): The size of the image. Defaults to 128."""
        super().__init__()
        self.image_size = image_size
        self.embedding = nn.Embedding(embedding_size, embedding_dim)

        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(image_size // 8 * image_size // 8 * 512 + 512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, std=0.02)
                init.zeros_(m.bias)
                spectral_norm(m)

    def forward(
        self,
        image: Tensor,
        image_idx: Tensor,
        scale: Tensor,
        rotation: Tensor,
        location_x: Tensor,
        location_y: Tensor,
    ) -> Tensor:
        embeddings = self.embedding(image_idx).squeeze(1)
        x = self.net(image)
        parameters = torch.cat(
            [
                x,
                embeddings,
                scale,
                rotation,
                location_x,
                location_y,
            ],
            dim=1,
        )

        return self.fc(parameters)
