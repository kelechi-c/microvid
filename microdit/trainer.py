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

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to('cuda')
print("loaded vae")

def vae_decode(latent, vae=vae):
    # print(f'decoding... (latent shape = {latent.shape}) ')
    latent = torch.from_numpy(np.array(latent))
    x = vae.decode(latent).sample

    img = VaeImageProcessor().postprocess(
        image=x.detach(), do_denormalize=[True, True]
    )[0]

    return img


def process_img(img):
    img = vae_decode(img[None])
    return img


def image_grid(pil_images, file, grid_size=(3, 3), figsize=(10, 10)):
    rows, cols = grid_size
    assert len(pil_images) <= rows * cols, "Grid size must accommodate all images."

    # Create a matplotlib figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # Flatten for easy indexing

    for i, ax in enumerate(axes):
        if i < len(pil_images):
            # Convert PIL image to NumPy array and plot
            ax.imshow(np.array(pil_images[i]))
            ax.axis("off")  # Turn off axis labels
        else:
            ax.axis("off")  # Hide empty subplots for unused grid spaces

    plt.tight_layout()
    plt.savefig(file, bbox_inches="tight")
    plt.show()

    return file


def sample_image_batch(model, step):
    pred_model = model.eval()
    with torch.no_grad():
        cond = torch.arange(0, 8).cuda() % 10
        uncond = torch.ones_like(cond) * 10

        init_noise = torch.randn(len(cond), 4, 32, 32).cuda()
        image_batch = pred_model.sample(init_noise, cond, uncond)
        
        imgfile = f"samples/sample_{step}.png"
        batch = [process_img(x) for x in enumerate(image_batch)]
        gridfile = image_grid(batch, imgfile)
        
        del pred_model
        
        return imgfile



def wandb_logger(key: str, project_name, run_name=None):  # wandb logger
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )


def train_step(model, optimizer, data_batch):
    img_latents, label = data_batch['vae_output'].to(torch.bfloat16), data_batch['label']
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
    
    loss = model(img_latents, label, mask)
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
@click.option("-r", "--run_type", default="overfit")
@click.option("-bs", "--batch_size", default=config.batch_size)
@click.option("-e", "--epochs", default=10)
def main(run_type, batch_size, epochs):

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

    model_size = sum(p.numel() for p in rf_engine.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    optimizer = torch.optim.AdamW(rf_engine.parameters(), lr=config.lr)
    # accelerator = Accelerator(mixed_precision='bfloat16', gradient_accumulation_steps=2)

    # rf_engine, optimizer, train_loader, vae = accelerator.prepare(rf_engine, optimizer, train_loader, vae)

    if run_type == 'overfit':
        model, loss = overfit(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print(f"microdit overfitting ended at loss {loss:.4f}")

    elif run_type == "train":
        trainer(epochs, rf_engine, optimizer, train_loader)
        wandb.finish()
        print("microdit (test) training (on imagenet-1k) in JAX..done")

main()
