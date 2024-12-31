import torch
import numpy as np
import wandb
from tqdm import tqdm
from einops import rearrange
from moviepy.video.io import ImageSequenceClip
from diffusers.utils.export_utils import export_to_video


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

    # Create a tensor of zeros with the same shape and dtype as the intended output
    full_patches = torch.zeros(
        bs, num_patches, embed_dim, device=patches.device, dtype=patches.dtype
    )

    # Identify the indices where the mask is True (unmasked patches)
    mask_indices = mask.nonzero(as_tuple=True)

    # Check if the number of unmasked patches matches the patches tensor size
    num_unmasked = patches.shape[1]

    # Assign the processed patches back to their unmasked positions
    full_patches[mask_indices[0], mask_indices[1]] = patches.reshape(-1, embed_dim)

    return full_patches


def wandb_logger(key: str, project_name, run_name=None):
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )


def save_video(final_frames, output_path="sample.mp4", fps=4):
    assert (
        final_frames.ndim == 4 and final_frames.shape[3] == 3
    ), f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)

    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path)

    return output_path


def process_video_latents(latents, vae):
    # latents = vae.decode(latents.to(torch.bfloat16))[0].sample()[0] / scale_factor
    latents = vae.tiled_decode(latents.to(torch.bfloat16))[0] / scale_factor

    print(f"tildecode {latents.shape}")
    latents = rearrange(latents, "b c t h w -> b t c h w")
    print(f"decoded {latents.shape}")
    return latents


def sample_video(step, model, captions, vae):
    pred_model = model.eval()
    noise = torch.randn(1, 16, 5, 28, 28).cuda().to(torch.bfloat16)
    vidlatent = pred_model.sample(noise, captions[None])
    vidfile = f"vidsamples/vid_{step}_microvid.mp4"

    sample = process_video_latents(vidlatent, vae)
    vidfile = save_video(sample[0])
    print(f"sample vid saved @ {vidfile}")
    del pred_model

    return vidfile
