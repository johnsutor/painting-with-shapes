# !/usr/bin/env python3

import hydra
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from aim import Image
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from tqdm import tqdm

from neural_renderer import NeuralRenderer
from renderer import ImageRenderer
from scheduler import LinearWarmupCosineAnnealingLR


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
        image_size=cfg.canvas_size,
        path=cfg.data_path,
    )

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        neural_renderer.parameters(),
        lr=cfg.learning_rate,
    )

    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=int(cfg.warmup_percentage * cfg.steps),
        max_epochs=cfg.steps,
    )

    neural_renderer, criterion, scheduler, optimizer = accelerator.prepare(
        neural_renderer, criterion, scheduler, optimizer
    )

    ema_loss = 0

    for step in range(cfg.steps):
        image_renderer.clear()
        optimizer.zero_grad(set_to_none=True)
        neural_renderer.train()

        # Generate random parameters on the fly during training
        image_idx = torch.randint(0, len(image_renderer), (cfg.batch_size, 1))
        rotation = torch.ones((cfg.batch_size, 1)).uniform_(-1, 1)
        scale = torch.ones((cfg.batch_size, 1)).uniform_(-1, 1)
        location_x = torch.ones((cfg.batch_size, 1)).uniform_(-1, 1)
        location_y = torch.ones((cfg.batch_size, 1)).uniform_(-1, 1)

        rendered_images = image_renderer(
            image_idx, scale, rotation, location_x, location_y
        ).to(accelerator.device)

        neural_images = neural_renderer(
            image_idx,
            scale,
            rotation,
            location_x,
            location_y,
            device=accelerator.device,
        )

        loss = criterion(neural_images, rendered_images)

        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        ema_loss = cfg.loss_ema_decay * 0.99 + loss.item() * (1 - cfg.loss_ema_decay)

        if step % cfg.log_interval == 0:
            accelerator.log({"loss": ema_loss, "step": step})
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
            accelerator.print(f"Step: {step}, Loss: {ema_loss}")

    return ema_loss


if __name__ == "__main__":
    main()
