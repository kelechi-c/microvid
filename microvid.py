import torch
import torch.nn as nn
import numpy as np
import math, click
from einops import rearrange
from accelerate import Accelerator
from collections import defaultdict
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import torchvision
import os, pickle
import torch.nn.functional as F
import collections.abc
import math
from itertools import repeat
from typing import Callable, Optional


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

data_id = "tensorkelechi/tiny_webvid_latents"
VAE_SCALING_FACTOR = 0.13025
BS = 128
EPOCHS = 30
MASK_RATIO = 0.75
SEED = 333
LR = 1e-4

VAE_HF_NAME = "madebyollin/sdxl-vae-fp16-fix"
VAE_CHANNELS = 4
SIGLIP_HF_NAME = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
SIGLIP_EMBED_DIM = 1152


class Text2VideoDataset(IterableDataset):
    def __init__(self, split=512):
        super().__init__()
        self.split = split
        self.dataset = load_dataset(
            "tensorkelechi/tiny_webvid_latents",
            streaming=True,
            split="train",
            trust_remote_code=True,
        ).take(self.split)

    def __len__(self):
        return self.split

    def __iter__(self):
        for sample in self.dataset:
            latents = sample["video_latents"]  # type: ignore

            caption = sample["text_encoded"]  # type: ignore

            yield latents, caption


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

def apply_mask_to_tensor(x, mask, patch_size):
    bs, c, d, h, w = x.shape
    num_patches_d = d // patch_size[0]
    num_patches_h = h // patch_size[1]
    num_patches_w = w // patch_size[2]

    # Ensure that height and width are divisible by patch_size
    assert (
        d % patch_size[0] == 0 and h % patch_size[1] == 0 and w % patch_size[2] == 0
    ), "Height and width must be divisible by patch_size. Height: {}, Width: {}, Patch size: {}".format(
        h, w, patch_size
    )

    # Reshape mask to (bs, num_patches_d, num_patches_h, num_patches_w)
    mask = mask.view(bs, num_patches_d, num_patches_h, num_patches_w)

    # Expand the mask to cover each patch
    # (bs, num_patches_d, num_patches_h, num_patches_w) -> (bs, 1, d, h, w)
    mask = mask.unsqueeze(1)  # Add channel dimension
    mask = mask.repeat(
        1, 1, patch_size[0], patch_size[1], patch_size[2]
    )  # Repeat for patch_size
    mask = mask.view(bs, 1, d, h, w)  # Reshape to (bs, 1, d, h, w)

    # Apply the mask to the input tensor
    x = x * mask

    return x


def unpatchify(x, patch_size, depth, height, width):

    bs, num_patches, patch_dim = x.shape
    D, H, W = patch_size
    in_channels = patch_dim // (D * H * W)

    # Calculate the number of patches along each dimension
    num_patches_d = depth // D
    num_patches_h = height // H
    num_patches_w = width // W

    # Ensure num_patches equals num_patches_d * num_patches_h * num_patches_w
    assert (
        num_patches == num_patches_d * num_patches_h * num_patches_w
    ), "Mismatch in number of patches."

    # Reshape x to (bs, num_patches_d, num_patches_h, num_patches_w, D, H, W, in_channels)
    x = x.view(bs, num_patches_d, num_patches_h, num_patches_w, D, H, W, in_channels)

    # Permute x to (bs, num_patches_d, D, num_patches_h, H, num_patches_w, W, in_channels)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()

    # Reshape x to (bs, depth, height, width, in_channels)
    reconstructed = x.view(bs, depth, height, width, in_channels)

    # Permute back to (bs, in_channels, depth, height, width)
    reconstructed = reconstructed.permute(0, 4, 1, 2, 3).contiguous()

    return reconstructed


def strings_to_tensor(string_list):
    
    # Convert each string to a list using eval
    list_of_lists = [eval(s) for s in string_list]

    # Convert the list of lists to a PyTorch tensor
    tensor = torch.tensor(list_of_lists, dtype=torch.float32)

    return tensor


def random_mask(
    bs: int, depth: int, height: int, width: int, patch_size: tuple, mask_ratio: float
) -> torch.Tensor:
    
    D, H, W = patch_size
    num_patches_d = depth // D
    num_patches_h = height // H
    num_patches_w = width // W
    num_patches = num_patches_d * num_patches_h * num_patches_w
    num_patches_to_mask = int(num_patches * mask_ratio)

    # Create a tensor of random values
    rand_tensor = torch.rand(bs, num_patches)

    # Sort the random tensor and get the indices
    _, indices = torch.sort(rand_tensor, dim=1)

    # Create a mask tensor initialized with ones
    mask = torch.ones(bs, num_patches, device=rand_tensor.device)

    # Set the first num_patches_to_mask indices to 0 for each batch
    mask[torch.arange(bs).unsqueeze(1), indices[:, :num_patches_to_mask]] = 0

    return mask


def remove_masked_patches(patches, mask):
    # Ensure mask is a boolean tensor
    mask = mask.bool()
    # Get batch size and embed dimension
    bs, num_patches, embed_dim = patches.shape
    # Expand mask to match the shape of patches for correct indexing
    mask = mask.unsqueeze(-1).expand(-1, -1, embed_dim)
    # Use masked_select and reshape to maintain batch size
    unmasked_patches = torch.masked_select(patches, mask).view(bs, -1, embed_dim)

    return unmasked_patches


def add_masked_patches(patches, mask):
    # Ensure mask is a boolean tensor
    mask = mask.bool()
    # Get the total number of patches and embed dimension
    bs, num_patches = mask.shape
    embed_dim = patches.shape[-1]

    # Create a tensor of zeros with the same shape and dtype as the patches tensor
    full_patches = torch.zeros(
        bs, num_patches, embed_dim, device=patches.device, dtype=patches.dtype
    )

    # Create a mask for where patches should be placed
    mask_indices = mask.nonzero(as_tuple=True)

    # Assign the unmasked patches back to their original positions
    full_patches[mask_indices[0], mask_indices[1]] = patches

    return full_patches


class Patchify(nn.Module):
    def __init__( 
        self,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer = None,
        flatten: bool = True,
        bias: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.flatten = flatten
        self.dynamic_img_pad = dynamic_img_pad

        self.conv_proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor):
        B, _C, T, H, W = x.shape

        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))

        x = rearrange(x, "B C T H W -> (B T) C H W", B=B, T=T)
        x = self.conv_proj(x)

        # Flatten temporal and spatial dimensions.
        x = rearrange(x, "(B T) C H W -> B (T H W) C", B=B, T=T)

        x = self.norm(x)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        *,
        bias: bool = True,
        timestep_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.timestep_scale = timestep_scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
        freqs.mul_(-math.log(max_period) / half).exp_()
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        if self.timestep_scale is not None:
            t = t * self.timestep_scale
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class PooledCaptionEmbedder(nn.Module):
    def __init__(
        self,
        caption_feature_dim: int,
        hidden_size: int,
        *,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.caption_feature_dim = caption_feature_dim
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(caption_feature_dim, hidden_size, bias=bias, device=device),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias, device=device),
        )

    def forward(self, x):
        return self.mlp(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        # keep parameter count and computation constant compared to standard FFN
        hidden_size = int(2 * hidden_size / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_size = int(ffn_dim_multiplier * hidden_size)
        hidden_size = multiple_of * ((hidden_size + multiple_of - 1) // multiple_of)

        self.hidden_dim = hidden_size
        self.w1 = nn.Linear(in_features, 2 * hidden_size, bias=False, device=device)
        self.w2 = nn.Linear(hidden_size, in_features, bias=False, device=device)

    def forward(self, x: torch.Tensor):
        # torch.chun
        x, gate = self.w1(x).chunk(2, dim=-1)
        x = self.w2(F.silu(x) * gate)
        return x


@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=10)
@click.option("-bs", "--batch_size", default=32)
def main(run, epochs, batch_size):
    # DiT-B config
    dit_model = DiT(dim=1024, depth=24, attn_heads=16)

    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        drop_last=True,
        # collate_fn=jax_collate,
    )

    sp = next(iter(train_loader))
    print(f"loaded data \n data sample: {sp['vae_output'].shape}")

    if run == "single_batch":
        model, loss = batch_trainer(
            epochs, model=dit_model, optimizer=optimizer, train_loader=train_loader
        )
        wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f"you missed your train looop impl boy")


main()
