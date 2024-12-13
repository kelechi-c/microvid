import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import torch.nn.functional as F


def apply_mask_to_tensor(x, mask, patch_size):
    """
    Applies a mask to a tensor. Turns the masked values to 0s.

    Args:
        x (torch.Tensor): Tensor of shape (bs, c, h, w)
        mask (torch.Tensor): Tensor of shape (bs, num_patches)
        patch_size (int): Size of each patch.

    Returns:
        torch.Tensor: Tensor of shape (bs, c, h, w) with the masked values turned to 0s.
    """
    bs, c, h, w = x.shape
    num_patches_h = h // patch_size[0]
    num_patches_w = w // patch_size[1]

    # Ensure that height and width are divisible by patch_size
    assert (
        h % patch_size[0] == 0 and w % patch_size[1] == 0
    ), "Height and width must be divisible by patch_size. Height: {}, Width: {}, Patch size: {}".format(
        h, w, patch_size
    )

    # Reshape mask to (bs, num_patches_h, num_patches_w)
    mask = mask.view(bs, num_patches_h, num_patches_w)

    # Expand the mask to cover each patch
    # (bs, num_patches_h, num_patches_w) -> (bs, 1, h, w)
    mask = mask.unsqueeze(1)  # Add channel dimension
    mask = mask.repeat(1, 1, patch_size[0], patch_size[1])  # Repeat for patch_size
    mask = mask.view(bs, 1, h, w)  # Reshape to (bs, 1, h, w)

    # Apply the mask to the input tensor
    x = x * mask

    return x


def unpatchify(x, patch_size, height, width):
    """
    Reconstructs images from patches.

    Args:
        x (torch.Tensor): Tensor of shape (bs, num_patches, patch_size * patch_size * in_channels)
        patch_size (int): Size of each patch.
        height (int): Original image height.
        width (int): Original image width.

    Returns:
        torch.Tensor: Reconstructed image of shape (bs, in_channels, height, width)
    """
    bs, num_patches, patch_dim = x.shape
    H, W = patch_size
    in_channels = patch_dim // (H * W)

    # Calculate the number of patches along each dimension
    num_patches_h = height // H
    num_patches_w = width // W

    # Ensure num_patches equals num_patches_h * num_patches_w
    assert (
        num_patches == num_patches_h * num_patches_w
    ), "Mismatch in number of patches."

    # Reshape x to (bs, num_patches_h, num_patches_w, H, W, in_channels)
    x = x.view(bs, num_patches_h, num_patches_w, H, W, in_channels)

    # Permute x to (bs, num_patches_h, H, num_patches_w, W, in_channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Reshape x to (bs, height, width, in_channels)
    reconstructed = x.view(bs, height, width, in_channels)

    # Permute back to (bs, in_channels, height, width)
    reconstructed = reconstructed.permute(0, 3, 1, 2).contiguous()

    return reconstructed


def random_mask(
    bs: int, height: int, width: int, patch_size: tuple[int, int], mask_ratio: float
) -> torch.Tensor:
    """
    Generates a random mask for patched images. Randomly selects patches to mask.

    Args:
        bs (int): Batch size.
        height (int): Height of the image.
        width (int): Width of the image.
        patch_size (tuple of int): Size of the patches.
        mask_ratio (float): Ratio of patches to mask. Ranges from 0 to 1. mask_ratio * 100 = percentage of 1s in the mask

    Returns:
        mask (torch.Tensor): A tensor of shape (bs, num_patches) with values in {0, 1}.
    """
    num_patches = (height // patch_size[0]) * (width // patch_size[1])
    num_patches_to_mask = int(num_patches * mask_ratio)

    # Create a tensor of random values
    rand_tensor = torch.rand(bs, num_patches)

    # Sort the random tensor and get the indices
    _, indices = torch.sort(rand_tensor, dim=1)

    # Create a mask tensor initialized with ones
    mask = torch.ones(bs, num_patches)

    # Set the first num_patches_to_mask indices to 0 for each batch
    mask[torch.arange(bs).unsqueeze(1), indices[:, :num_patches_to_mask]] = 0

    # Ensure the final shape is (bs, num_patches)
    mask = mask.view(bs, num_patches)

    return mask


def remove_masked_patches(patches, mask):
    """
    Removes the masked patches from the patches tensor while preserving batch dimensions.
    Returned tensor will have shape (bs, number_of_unmasked_patches, embed_dim).
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()
    mask = mask.logical_not()

    # Get batch size and embed dimension
    bs, num_patches, embed_dim = patches.shape

    # Expand mask to match the shape of patches for correct indexing
    mask = mask.unsqueeze(-1).expand(-1, -1, embed_dim)

    # Use masked_select and reshape to maintain batch size
    unmasked_patches = torch.masked_select(patches, ~mask).view(bs, -1, embed_dim)

    return unmasked_patches


def add_masked_patches(patches, mask):
    """
    Adds the masked patches to the patches tensor.
    Returned tensor will have shape (bs, num_patches, embed_dim).
    The missing patches will be filled with 0s.
    """
    # Ensure mask is a boolean tensor
    mask = mask.bool()

    # Get the total number of patches and embed dimension
    bs, num_patches, embed_dim = mask.shape[0], mask.shape[1], patches.shape[-1]

    # Create a tensor of zeros with the same shape and dtype as the patches tensor
    full_patches = torch.zeros(
        bs, num_patches, embed_dim, device=patches.device, dtype=patches.dtype
    )

    # Iterate over each batch and place unmasked patches back in their original positions
    for i in range(bs):
        # Use the mask to place unmasked patches back in the correct positions
        full_patches[i, mask[i]] = patches[i].to(full_patches.dtype)

    return full_patches


def get_2d_sincos_pos_embed(embed_dim, h, w):
    """
    :param embed_dim: dimension of the embedding
    :param h: height of the grid
    :param w: width of the grid
    :return: [h*w, embed_dim] or [1+h*w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0)

    grid = grid.reshape([2, 1, h, w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_timestep_embedding(timesteps, embedding_dim):

    assert len(timesteps.shape) == 1, "Timesteps should be a 1-D tensor"

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000, dtype=torch.float32)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Patchifier(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_size = patch_size

    def forward(self, x):
        x = self.proj(x)  # (B, C, H, W) -> (B, E, H', W')
        return x.flatten(2).transpose(1, 2)  # (B, E, H', W') -> (B, H'*W', E)

try:
    import flash_attn

    if hasattr(flash_attn, "__version__") and int(flash_attn.__version__[0]) == 2:
        from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention
    else:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention
except Exception as e:
    print(f"flash_attn import failed: {e}")


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        # if inference with fp16, embedding.half()
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)  # .half())
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class MoEGate(nn.Module):
    def __init__(
        self, embed_dim, num_experts=16, num_experts_per_tok=2, aux_loss_alpha=0.01
    ):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = num_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # print(bsz, seq_len, h)
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss, 
    which includes the gradient of the aux loss during backpropagation.
    """
    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss

