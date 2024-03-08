#!/usr/bin/env python

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2


class NeuralRenderer(nn.Module):
    """Neural renderer for 2D images. This is a simple renderer that draws images on a canvas and
    returns the rendered image"""

    def __init__(
        self,
        embedding_size: int,
        embedding_dim: int = 508,
        image_size: int = 128,
    ):
        """Args:

        image_size (int, optional): The size of the image. Defaults to 128.
        """
        super().__init__()

        self.image_size = image_size
        self.embedding = nn.Embedding(embedding_size, embedding_dim)

        # Follow https://github.com/megvii-research/ICCV2019-LearningToPaint/blob/master/baseline/Renderer/model.py closely
        # Calculate first channel size
        self.conv_ff = nn.Sequential(
            nn.Linear(embedding_dim + 4, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(2048, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, (image_size // 8 * image_size // 8 * 16), bias=False),
            nn.BatchNorm1d((image_size // 8 * image_size // 8 * 16)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Unflatten(1, (16, image_size // 8, image_size // 8)),
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(8, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )

    def forward(
        self,
        image_idx: Tensor,
        scale: Tensor,
        rotation: Tensor,
        location_x: Tensor,
        location_y: Tensor,
    ) -> Tensor:
        """Renders the images on a canvas. Expected shape of the images is (B, C, H, W) if not using
        mappings, and shape (B, 1) if using mappings. The expected shape of each of the other parameters
        is (B, 1).

        Args:
            images (Tensor): The images to render. Either the indices of the images or the images themselves.
            scale (Tensor): The scale factor of the images.
            rotation (Tensor): The rotation of the images.
            location_x (Tensor): The x-axis location of the images.
            location_y (Tensor): The y-axis location of the images.
            device (torch.device, optional): The device to use. Defaults to torch.device("cpu").

        Returns:
            Tensor: The rendered images.
        """

        embeddings = self.embedding(image_idx).squeeze(1)
        parameters = torch.cat(
            [
                embeddings,
                scale,
                rotation,
                location_x,
                location_y,
            ],
            dim=1,
        )
        return self.conv_ff(parameters)
