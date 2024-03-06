# !/usr/bin/env python3

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.models import resnet18
from torchvision.transforms import v2


class NeuralRenderer(nn.Module):
    """Neural renderer for 2D images. This is a simple renderer that draws images on a canvas and
    returns the rendered image"""

    def __init__(
        self,
        image_size: int = 128,
        mapping: Dict[int, Tensor] = None,
        path: os.PathLike = None,
    ):
        """Args:
        image_size (int, optional): The size of the image. Defaults to 128.
        mapping (Dict[int, Tensor], optional): A mapping of the image index to the encoded image tensor (useful for pre-computed encodings). If not provided, the images are encoded using a ResNet18 model. Defaults to None.
        """
        super().__init__()
        valid_image_extensions = tuple(Image.registered_extensions().keys())
        self.paths = sorted(
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(valid_image_extensions)
            ]
        )

        self.image_size = image_size
        self.mapping = mapping
        self.path = path

        if not mapping:
            m = resnet18(pretrained=True)
            m = nn.Sequential(*list(m.children())[:-2], nn.Flatten())
            m[0] = nn.Conv2d(
                4,
                m[0].out_channels,
                kernel_size=m[0].kernel_size,
                stride=m[0].stride,
                padding=m[0].padding,
                bias=False,
            )
            self.image_encoder = m  # Include all but the Linear and avgpool layers

        # Follow https://github.com/megvii-research/ICCV2019-LearningToPaint/blob/master/baseline/Renderer/model.py closely
        self.parameters_ff = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
        )

        self.conv_ff = nn.Sequential(
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 4096),
            nn.ReLU(),
            nn.Unflatten(1, (16, 16, 16)),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(4, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((self.image_size // 2, self.image_size // 2)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
            ]
        )

    def _load_image(self, idx: int) -> Tensor:
        """Loads an image from the given path and returns it as a tensor. The image is resized to the
        specified image size.

        Args:
            path (os.PathLike): The path to the image.

        Returns:
            Tensor: The image as a tensor.
        """
        image = Image.open(self.paths[idx]).convert("RGBA")
        return self.transform(image)

    def get_encoded_images(self, images: Tensor, idx: Tensor) -> Dict[int, Tensor]:
        """Encodes the images using a pre-trained ResNet18 model. The images are expected to be of shape (B, C, H, W).

        Args:
            images (Tensor): The images to encode.
            idx (Tensor): The index of the images.

        Returns:
            Dict[int, Tensor]: The encoded images.
        """
        encoded_images = self.image_encoder(images)
        return {i: encoded_images[j] for j, i in enumerate(idx)}

    def forward(
        self,
        image_idx: Tensor,
        scale: Tensor,
        rotation: Tensor,
        location_x: Tensor,
        location_y: Tensor,
        device: torch.device = torch.device("cpu"),
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
        if self.mapping:
            images = torch.stack([self.mapping[i] for i in images], dim=0).to(device)
        else:
            images = torch.stack([self._load_image(i) for i in image_idx], dim=0).to(
                device
            )
            images = self.image_encoder(images)

        parameters = torch.cat([scale, rotation, location_x, location_y], dim=1).to(
            device
        )
        parameters = self.parameters_ff(parameters)
        x = torch.cat([images, parameters], dim=1)

        return self.conv_ff(x)
