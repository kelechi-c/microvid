from torch.utils.data import IterableDataset, DataLoader
"""
single file [model/data/trainer] for  
microvid - video generation with modified microdiffusion architecture

This is totally experimental please.
"""

import torch, click
import torch.nn as nn
import numpy as np
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os, wandb, gc
import math
from diffusers import AutoencoderKLHunyuanVideo

from utils import apply_mask_to_tensor, random_mask
from data import Text2VideoDataset
from model import MicroViDiT

vae = (
    AutoencoderKLHunyuanVideo.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    .to("cuda")
    .eval()
)
vae.enable_slicing()
print("loaded video VAE")


data_id = "tensorkelechi/tiny_webvid_latents"
MASK_RATIO = 0.75
SEED = 333
LR = 1e-4
scale_factor = 0.18215
VAE_CHANNELS = 16


class config:
    patch_size = (2, 2)
    l_frames = 5
    l_channels = 16
    l_height = 28
    l_width = 28
    lr = 1e-4


def seed_all(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    
seed_all()


def batch_trainer(epochs, model, optimizer, train_loader, accelerator):
    batch = next(iter(train_loader))
    patch_size = (2, 2, 2)
    device = accelerator.device
    losses = []

    # wandb_logger(key='', project_name='microvid')

    print("start single batch training..../ \n")

    for epoch in tqdm(range(epochs), desc="Training..."):
        # progress_bar = tqdm(dataset, desc=f"Epoch {epoch}", leave=False)
        # for batch_idx, batch in enumerate(dataset):
        optimizer.zero_grad()

        latents = batch["video_latents"].to(device)
        caption_embeddings = batch["text_encoded"].to(device)
        bs, c, d, h, w = latents.shape
        latents = latents * scale_factor

        mask = random_mask(
            bs, d, h, w, patch_size=patch_size, mask_ratio=MASK_RATIO
        ).to(device)
        # print(f'{mask.shape = }')

        nt = torch.randn((bs,)).to(device)
        t = torch.sigmoid(nt)

        texp = t.view([bs, *([1] * len(latents.shape[1:]))]).to(device)
        z1 = torch.randn_like(latents, device=device)

        zt = (1 - texp) * latents + texp * z1

        vtheta = model(zt, t, caption_embeddings, mask)

        latents = apply_mask_to_tensor(latents, mask, patch_size)
        vtheta = apply_mask_to_tensor(vtheta, mask, patch_size)
        z1 = apply_mask_to_tensor(z1, mask, patch_size)

        batchwise_mse = ((z1 - latents - vtheta) ** 2).mean(
            dim=list(range(1, len(latents.shape)))
        )
        loss = batchwise_mse.mean()
        loss = loss * 1 / (1 - MASK_RATIO)
        print(f"epoch {epoch}, loss => {loss.item():.4f}")

        # wandb.log({'loss/train': loss.item(), "log_loss/train": math.log10(loss.item())})

        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_local_main_process:
            losses.append(loss.item())

        # if epoch % 10 == 0:
        #     vidfile = sample_video(epoch, model, batch["text_encoded"][0].cuda())
        #     vidlog = wandb.Video(vidfile, fps=4)
        #     #wandb.log({'vidsample': vidlog})

    return losses[-1]


def collate(batch):
    latents = torch.stack([item[0] for item in batch], dim=0)
    text = [item[1][0] for item in batch]
    labels = torch.stack(text, dim=0)

    return {
        "video_latents": latents,
        "text_encoded": labels,
    }


@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=10)
@click.option("-bs", "--batch_size", default=32)
def main(run, epochs, batch_size):
    embed_dim = 768
    depth = 12

    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # DiT-B config
    microdit = MicroViDiT(
        in_channels=16,
        patch_size=(2, 2),
        embed_dim=embed_dim,
        num_layers=depth,
        num_heads=8,
        mlp_dim=embed_dim,
        caption_embed_dim=768,
        num_experts=4,
        active_experts=2,
        dropout=0.0,
        patch_mixer_layers=2,
    ).to(device)

    n_params = sum(p.numel() for p in microdit.parameters())
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    dataset = Text2VideoDataset()
    t2v_train_loader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(microdit.parameters(), lr=1e-3)

    sp = next(iter(t2v_train_loader))
    print(
        f"loaded data \n data sample: latents - {sp['video_latents'].shape}, text cond - {sp['text_encoded'].shape}"
    )

    microdit, optimizer, t2v_train_loader = accelerator.prepare(
        microdit, optimizer, t2v_train_loader
    )

    if run == "single_batch":
        loss = batch_trainer(
            epochs=epochs,
            model=microdit,
            optimizer=optimizer,
            train_loader=t2v_train_loader,
            accelerator=accelerator,
        )  # type: ignore

        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f"you missed your full train loop impl boy")


main()
