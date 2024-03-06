# !/usr/bin/env python3

import os
from typing import Any, List, Tuple

import torch
from joblib import Parallel, delayed
from PIL import Image
from torch import Tensor
from torchvision.transforms import v2


class ImageRenderer:
    """Renders images from a directory of images. The images must be loadable by PIL.
    The images are rendered on a canvas of a fixed size. The canvas is then normalized
    to the range [-1, 1] (for consistency) and returned as a tensor.
    """

    canvases: List[Image.Image]

    def __init__(
        self,
        path: os.PathLike,
        canvas_size: int = 128,
        rotate_range: Tuple[float, float] = (-90.0, 90.0),
        scale_range: Tuple[float, float] = (0.5, 2),
        batch_size: int = 1,
        num_workers: int = 4,
    ):
        """Args:
        path (os.PathLike): The path to the directory containing the images.
        canvas_size (int, optional): The size of the canvas. Defaults to 128.
        rotate_range (Tuple[float, float], optional): The range of rotation in degrees. Defaults to (-90., 90.).
        scale_range (Tuple[float, float], optional): The range of scale. Defaults to (0.5, 2).
        batch_size (int, optional): The batch size to use when rendering. Defaults to 1.
        num_workers (int, optional): The number of workers to use when rendering. Defaults to 4.
        """
        valid_image_extensions = tuple(Image.registered_extensions().keys())
        self.paths = sorted(
            [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(valid_image_extensions)
            ]
        )

        self.canvas_size = canvas_size

        self.min_angle, self.max_angle = rotate_range
        self.min_scale, self.max_scale = scale_range
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]),
            ]
        )

        self.clear()

    def __len__(self) -> int:
        return len(self.paths)

    def _unnormalize(self, value: float, min_val: float, max_val: float) -> float:
        return (value / 2 + 0.5) * (max_val - min_val) + min_val

    def _draw(
        self,
        canvas_idx: int,
        image_idx: int,
        scale: float,
        rotation: float,
        location_x: float,
        location_y: float,
    ) -> Tensor:
        """Draws to the existing canvas. The default size of the object is half the canvas size."""
        img = Image.open(self.paths[image_idx]).convert("RGBA")
        img = img.rotate(rotation, expand=1)
        scale = self._unnormalize(scale, self.min_scale, self.max_scale)

        width, height = img.size
        width, height = (
            int((self.canvas_size // 2) * scale),
            int((self.canvas_size // 2) * scale),
        )
        img = img.resize((width, height))

        paste_x = int(
            self._unnormalize(location_x, -width // 2, self.canvas_size - (width // 2))
        )
        paste_y = int(
            self._unnormalize(
                location_y, -height // 2, self.canvas_size - (height // 2)
            )
        )

        self.canvases[canvas_idx].paste(img, (paste_x, paste_y), img)
        return self.transform(self.canvases[canvas_idx])

    def clear(self):
        self.canvases = [
            Image.new(
                "RGBA", (self.canvas_size, self.canvas_size), (255, 255, 255, 255)
            )
            for _ in range(self.batch_size)
        ]

    def get_canvas(self, idx: int):
        return self.canvas[idx]

    def draw(
        self,
        image_idx: Tensor,
        scale: Tensor,
        rotation: Tensor,
        location_x: Tensor,
        location_y: Tensor,
    ) -> Tensor:
        """Draw a batch of images to their respective canvases, using JobLib to parallelize the drawing.

        Args:
        image_idx (Tensor): The indices of the images to draw.
        scale (Tensor): The scale factor of the images.
        rotation (Tensor): The rotation of the images.
        location_x (Tensor): The x-axis location of the images.
        location_y (Tensor): The y-axis location of the images.

        Returns:
        Tensor: A tensor of the rendered images.
        """
        batch = Parallel(n_jobs=self.num_workers)(
            delayed(self._draw)(
                i, image_idx[i], scale[i], rotation[i], location_x[i], location_y[i]
            )
            for i in range(self.batch_size)
        )
        return torch.stack(batch, dim=0)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.draw(*args, **kwargs)
