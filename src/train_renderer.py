# !/usr/bin/env python3

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from aim import Image
from omegaconf import OmegaConf
from piq import DISTS
from torchvision.utils import make_grid
from tqdm import tqdm

from discriminator import Discriminator
from neural_renderer import NeuralRenderer
from renderer import ImageRenderer
from scheduler import LinearWarmupCosineAnnealingLR


def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()


@hydra.main(config_path="../configs", config_name="train_renderer")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    accelerator = Accelerator(
        cpu=cfg.cpu,
        log_with="aim",
        project_dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        mixed_precision=cfg.mixed_precision,
    )

    accelerator.init_trackers(
        project_name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, enum_to_str=True),
        init_kwargs={},
    )

    image_renderer = ImageRenderer(
        path=cfg.data_path,
        canvas_size=cfg.canvas_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    neural_renderer = NeuralRenderer(
        image_size=cfg.canvas_size, embedding_size=len(image_renderer)
    )

    discriminator = Discriminator(
        image_size=cfg.canvas_size, embedding_size=len(image_renderer)
    )

    criterion = nn.MSELoss()

    renderer_optimizer = torch.optim.Adam(
        neural_renderer.parameters(),
        lr=cfg.learning_rate,
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=cfg.learning_rate,
    )

    renderer_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=renderer_optimizer,
        warmup_epochs=int(cfg.warmup_percentage * cfg.steps),
        max_epochs=cfg.steps,
    )

    discriminator_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=discriminator_optimizer,
        warmup_epochs=int(cfg.warmup_percentage * cfg.steps),
        max_epochs=cfg.steps,
    )

    perceptual_loss = DISTS()

    (
        neural_renderer,
        discriminator,
        criterion,
        renderer_optimizer,
        discriminator_optimizer,
        renderer_scheduler,
        discriminator_scheduler,
        perceptual_loss,
    ) = accelerator.prepare(
        neural_renderer,
        discriminator,
        criterion,
        renderer_optimizer,
        discriminator_optimizer,
        renderer_scheduler,
        discriminator_scheduler,
        perceptual_loss,
    )

    for step in range(cfg.steps):
        # Train discriminator
        discriminator_optimizer.zero_grad()
        neural_renderer.eval()
        discriminator.train()

        image_idx = torch.randint(0, len(image_renderer), (cfg.batch_size, 1)).to(
            accelerator.device
        )
        rotation = (
            torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        )
        scale = torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        location_x = (
            torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        )
        location_y = (
            torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        )

        rendered_images = image_renderer(
            image_idx, scale, rotation, location_x, location_y
        ).to(accelerator.device)

        neural_images = neural_renderer(
            image_idx,
            scale,
            rotation,
            location_x,
            location_y,
        )

        disc_real = discriminator(
            neural_images, image_idx, scale, rotation, location_x, location_y
        )
        disc_fake = discriminator(
            rendered_images, image_idx, scale, rotation, location_x, location_y
        )

        loss = hinge_loss(disc_real, disc_fake) + criterion(
            rendered_images, neural_images
        )

        accelerator.backward(loss)
        discriminator_optimizer.step()
        discriminator_scheduler.step()

        # Train neural renderer
        discriminator.eval()
        neural_renderer.train()

        image_idx = torch.randint(0, len(image_renderer), (cfg.batch_size, 1)).to(
            accelerator.device
        )
        rotation = (
            torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        )
        scale = torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        location_x = (
            torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        )
        location_y = (
            torch.ones((cfg.batch_size, 1)).uniform_(-1, 1).to(accelerator.device)
        )

        rendered_images = image_renderer(
            image_idx, scale, rotation, location_x, location_y
        ).to(accelerator.device)

        neural_images = neural_renderer(
            image_idx,
            scale,
            rotation,
            location_x,
            location_y,
        )

        loss = torch.mean(
            discriminator(
                neural_images, image_idx, scale, rotation, location_x, location_y
            )
        )

        accelerator.backward(loss)
        renderer_optimizer.step()
        renderer_scheduler.step()

        if step % cfg.log_interval == 0:
            perceptual_loss.eval()
            with torch.no_grad():
                # Only calculate the loss on the RGB channels
                loss_p = perceptual_loss(
                    neural_images[:, :3, ...], rendered_images[:, :3, ...]
                ).item()

            accelerator.log({"perceptual_loss": loss_p, "step": step})
            accelerator.get_tracker("aim").tracker.track(
                Image(make_grid(rendered_images, normalize=True)),
                name="Rendered Images",
                step=step,
            )
            accelerator.get_tracker("aim").tracker.track(
                Image(make_grid(neural_images, normalize=True)),
                name="Neural Images",
                step=step,
            )
            accelerator.print(f"Step: {step}, Perceptual Loss: {loss_p}")

    return loss_p


if __name__ == "__main__":
    main()
