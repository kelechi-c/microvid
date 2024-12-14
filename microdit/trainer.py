import os, click, wandb, time, gc, random
import math, torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Any
from torch.utils.data import DataLoader, DistributedSampler
import matplotlib.pyplot as plt
from PIL import Image as pillow
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from streaming.base.format.mds.encodings import Encoding, _encodings
from streaming import StreamingDataset
from accelerate import Accelerator
from datasets import load_dataset
from tqdm import tqdm
import torchvision
import warnings
warnings.filterwarnings("ignore")

from .microdit import MicroDiT, RectFlowWrapper, random_mask


class config:
    batch_size = 128
    img_size = 32
    latent_channels = 4
    seed = 222
    patch_size = (2, 2)
    lr = 2e-4
    mask_ratio = 0.75
    epochs = 30
    data_split = 50_000
    cfg_scale = 2.0
    vaescale_factor = 0.13025
    data_id = "cloneofsimo/imagenet.int8"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8
remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./imagenet"

def sample_image_batch(model, step):
    with torch.no_grad():
        cond = torch.arange(0, 16).cuda() % 10
        uncond = torch.ones_like(cond) * 10

        init_noise = torch.randn(16, 4, 32, 32).cuda()
        images = model.sample(init_noise, cond, uncond)
        # image sequences to gif
        gif = []
        for image in images:
            # unnormalize
            image = image * 0.5 + 0.5
            image = image.clamp(0, 1)
            x_as_image = torchvision.utils.make_grid(image.float(), nrow=4)
            img = x_as_image.permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            gif.append(pillow.fromarray(img))

        gif[0].save(
            f"contents/sample_{step}.gif",
            save_all=True,
            append_images=gif[1:],
            duration=100,
            loop=0,
        )
        imgfile = f"sample_{step}.png"
        last_img = gif[-1]
        last_img.save(imgfile)

        return imgfile


def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )


def train_step(rf_engine, optimizer, data_batch):
    img_latents, label = data_batch['vae_output'], data_batch['label']
    img_latents = img_latents.reshape(-1, 4, 32, 32) * config.vaescale_factor
    bs, height, width, channels = img_latents.shape

    mask = random_mask(
        bs,
        height,
        width,
        patch_size=config.patch_size,
        mask_ratio=config.mask_ratio,
    )

    # loss = model(img_latents, label, mask)
    optimizer.zero_grad()
    
    loss = rf_engine.forward(img_latents, label, mask)
    loss.backward()
    optimizer.step()

    return loss


def trainer(epochs, model, optimizer, train_loader):
    pass


def overfit(epochs, model, optimizer, train_loader):
    train_loss = 0.0
    model.train()

    wandb_logger(
        key="", project_name="microdit_overfit"
    )

    stime = time.time()

    batch = next(iter(train_loader))
    print("start overfitting.../")
    
    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model, optimizer, batch)
        print(f"epoch {epoch+1}/{epochs}, train loss => {train_loss.item():.4f}")
        
        wandb.log(
            {"loss": train_loss.item(), "log_loss": math.log10(train_loss.item())}
        )

        if epoch % 25 == 0:
            gridfile = sample_image_batch(epoch, model)
            image_log = wandb.Image(gridfile)
            wandb.log({"image_sample": image_log})

        torch.cuda.empty_cache()
        gc.collect()

    etime = time.time() - stime
    print(f"overfit time for {epochs} epochs -> {etime/60:.4f} mins / {etime/60/60:.4f} hrs")

    epoch_file = sample_image_batch("overfit", model)
    epoch_image_log = wandb.Image(epoch_file)
    wandb.log({"overfit_sample": epoch_image_log})

    return model, train_loss


@click.command()
@click.option('-r', '--run_type', default='overfit')
@click.option('-bs', '--batch_size', default=config.batch_size)
def main(run_type, batch_size):

    dataset = StreamingDataset(
        local=local_train_dir,
        remote=remote_train_dir,
        split=None,
        shuffle=True,
        shuffle_algo="naive",
        batch_size=batch_size,
    )

    # dataset_sampler = DistributedSampler(dataset, shuffle=True, drop_last=True, num_replicas=num_processes, rank=rank, seed=config.seed)

    print(f"datasample {dataset[0]}")

    train_loader = DataLoader(
        dataset[:config.data_split],
        batch_size=batch_size,
        num_workers=0,
        drop_last=True
    )

    sp = next(iter(train_loader))
    
    print(f"loaded dataset, sample shape {sp['vae_output'].shape} /  {sp['label'].shape}, type = {type(sp['vae_output'])}")

    dit_model = MicroDiT(
        in_channels=4,
        patch_size=config.patch_size,
        embed_dim=1024,
        num_layers=24,
        num_heads=16,
        dropout=0.0,
        mlp_dim=1024
    )

    rf_engine = RectFlowWrapper(dit_model)
    optimizer = torch.optim.AdamW(dit_model.parameters(), lr=config.lr)
    accelerator = Accelerator(mixed_precision='bfloat16', gradient_accumulation_steps=2)

    dit_model, optimizer, train_loader = accelerator.prepare(dit_model, optimizer, train_loader)
