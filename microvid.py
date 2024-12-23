from torch import nn


import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from accelerate import Accelerator
from collections import defaultdict
from tqdm import tqdm
import torchvision
import os, pickle
import torch.nn.functional as F
import collections.abc
import math
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

USERNAME = "SwayStar123"
DATASET_NAME = "preprocessed_commoncatalog-cc-by"
DS_DIR_BASE = "../../datasets"
MODELS_DIR_BASE = "../../models"
VAE_SCALING_FACTOR = 0.13025

BS = 256
EPOCHS = 30
MASK_RATIO = 0.75
SEED = 42

LR = 1e-4

VAE_HF_NAME = "madebyollin/sdxl-vae-fp16-fix"
VAE_CHANNELS = 4
SIGLIP_HF_NAME = "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
SIGLIP_EMBED_DIM = 1152


# selected_ids_list = []


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_mask_to_tensor(x, mask, patch_size):
    """
    Applies a mask to a tensor. Turns the masked values to 0s.

    Args:
        x (torch.Tensor): Tensor of shape (bs, c, d, h, w)
        mask (torch.Tensor): Tensor of shape (bs, num_patches)
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: Tensor of shape (bs, c, h, w) with the masked values turned to 0s.
    """
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
    """
    Reconstructs videos from patches without using F.fold.

    Args:
        x (torch.Tensor): Tensor of shape (bs, num_patches, D * H * W * in_channels)
        patch_size (tuple of int): Size of each patch as (D, H, W).
        depth (int): Original video depth (number of frames).
        height (int): Original video height.
        width (int): Original video width.

    Returns:
        torch.Tensor: Reconstructed video of shape (bs, in_channels, depth, height, width)
    """
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
    """
    Converts a list of strings, each representing a list (e.g., "[1, 2, 3]"),
    into a PyTorch tensor.

    Args:
        string_list (list of str): A list of strings, where each string is a list in string form.

    Returns:
        torch.Tensor: A PyTorch tensor containing the data from the lists.
    """
    # Convert each string to a list using eval
    list_of_lists = [eval(s) for s in string_list]

    # Convert the list of lists to a PyTorch tensor
    tensor = torch.tensor(list_of_lists, dtype=torch.float32)

    return tensor


def random_mask(
    bs: int, depth: int, height: int, width: int, patch_size: tuple, mask_ratio: float
) -> torch.Tensor:
    """
    Generates a random mask for patched videos. Randomly selects patches across depth, height, and width to mask.

    Args:
        bs (int): Batch size.
        depth (int): Depth of the video (number of frames).
        height (int): Height of the video.
        width (int): Width of the video.
        patch_size (tuple of int): Size of the patches as (D, H, W).
        mask_ratio (float): Ratio of patches to mask. Ranges from 0 to 1.

    Returns:
        mask (torch.Tensor): A tensor of shape (bs, num_patches) with values in {0, 1}.
    """
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
        patch_size: int = 16,
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

    def forward(self, x):
        B, _C, T, H, W = x.shape

        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))

        x = rearrange(x, "B C T H W -> (B T) C H W", B=B, T=T)
        x = self.conv_proj(x)

        # Flatten temporal and spatial dimensions.
        if not self.flatten:
            raise NotImplementedError("Must flatten output.")
        x = rearrange(x, "(B T) C H W -> B (T H W) C", B=B, T=T)

        x = self.norm(x)
        return x
