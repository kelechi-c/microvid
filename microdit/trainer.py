import os, pickle, click, wandb, time
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

from .microdit import MicroDiT, RectFlowWrapper


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


class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x = np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0


_encodings["uint8"] = uint8
remote_train_dir = "./vae_mds"  # this is the path you installed this dataset.
local_train_dir = "./imagenet"

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
    print(
        f"loaded dataset, sample shape {sp['vae_output'].shape} /  {sp['label'].shape}, type = {type(sp['vae_output'])}"
    )


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
    
    