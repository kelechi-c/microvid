import torch
import torch.nn as nn
import numpy as np
import math, click
from einops import rearrange
from accelerate import Accelerator
from collections import defaultdict
from timm.models.vision_transformer import Attention, Mlp
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from moviepy.video.io import ImageSequenceClip
import torchvision, imageio
import os, wandb, gc
import torch.nn.functional as F
import math
from typing import Optional

from diffusers import AutoencoderKLHunyuanVideo

vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="vae", torch_dtype=torch.bfloat16).to('cuda').eval()
vae.enable_slicing()
print('loaded video VAE')


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
    
seed_all()


class Text2VideoDataset(IterableDataset):
    def __init__(self, split=512):
        super().__init__()
        self.split = split
        self.dataset = load_dataset(
            data_id,
            streaming=True,
            split="train",
            trust_remote_code=True,
        ).take(self.split)

    def __len__(self):
        return self.split

    def __iter__(self):
        for sample in self.dataset:
            latents = torch.tensor(sample["hyv_latents"], dtype=torch.bfloat16)  # type: ignore

            caption = torch.tensor(sample["text_encoded"], dtype=torch.bfloat16)  # type: ignore
            caption = caption[:, :10, :]
            # print(f'data shape = {latents.shape} / {caption.shape}')

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


def get_3d_sincos_pos_embed(embed_dim, t, h, w):
    """
    Generates 3D sinusoidal positional embeddings.

    Args:
        embed_dim (int): Embedding dimension.
        t (int): Number of time steps (frames).
        h (int): Height of the spatial grid.
        w (int): Width of the spatial grid.

    Returns:
        torch.Tensor: Positional embeddings of shape (T*H*W, embed_dim).
    """
    grid_t = torch.arange(t, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_t, grid_h, grid_w = torch.meshgrid(grid_t, grid_h, grid_w, indexing="ij")
    grid = torch.stack([grid_t, grid_h, grid_w], dim=0)  # Shape: (3, T, H, W)

    grid = grid.reshape([3, t, h, w])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generates 3D sinusoidal positional embeddings from a grid.

    Args:
        embed_dim (int): Embedding dimension.
        grid (torch.Tensor): Grid coordinates of shape (3, T, H, W).

    Returns:
        torch.Tensor: Positional embeddings of shape (T*H*W, embed_dim).
    """
    assert embed_dim % 2 == 0
    emb_dim_per_axis = embed_dim // 3  # Divide embedding dimension across 3 axes
    assert emb_dim_per_axis % 2 == 0

    emb_t = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[2])  # (T*H*W, D/3)

    emb = torch.cat([emb_t, emb_h, emb_w], dim=1)  # (T*H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generates 1D sinusoidal positional embeddings from grid positions.

    Args:
        embed_dim (int): Embedding dimension.
        pos (torch.Tensor): 1D position coordinates.

    Returns:
        torch.Tensor: Positional embeddings of shape (M, embed_dim).
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


class Patchify(nn.Module):
    def __init__( 
        self,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer = None,
        flatten: bool = True,
        bias: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
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


class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=2):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.pretraining_tp = pretraining_tp

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            # print(self.up_proj.weight.size(), self.down_proj.weight.size())
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )
            up_proj = torch.cat(
                [F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(
                slice, dim=-1
            )
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class SparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
        self,
        embed_dim,
        mlp_ratio=4,
        num_experts=16,
        num_experts_per_tok=2,
        pretraining_tp=2,
    ):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                MoeMLP(
                    hidden_size=embed_dim,
                    intermediate_size=mlp_ratio * embed_dim,
                    pretraining_tp=pretraining_tp,
                )
                for i in range(num_experts)
            ]
        )
        self.gate = MoEGate(
            embed_dim=embed_dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )
        self.n_shared_experts = 2

        if self.n_shared_experts is not None:
            intermediate_size = embed_dim * self.n_shared_experts
            self.shared_experts = MoeMLP(
                hidden_size=embed_dim,
                intermediate_size=intermediate_size,
                pretraining_tp=pretraining_tp,
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        # print(topk_idx.tolist(), print(len(topk_idx.tolist())))
        # global selected_ids_list
        # selected_ids_list.append(topk_idx.tolist())

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(
                self.num_experts_per_tok, dim=0
            )
            y = torch.empty_like(hidden_states, dtype=hidden_states.dtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(
                    hidden_states[flat_topk_idx == i]
                ).float()
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(
                hidden_states, flat_topk_idx, topk_weight.view(-1, 1)
            ).view(*orig_shape)
        if self.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(
                0,
                exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )
        return expert_cache


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

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4,
        num_experts=8,
        num_experts_per_tok=2,
        pretraining_tp=2,
        use_flash_attn=False,
        **block_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = Attention(
        #     hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        # )
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        ) #nn.MultiheadAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        # self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.moe = SparseMoeBlock(
            hidden_size, mlp_ratio, num_experts, num_experts_per_tok, pretraining_tp
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_basic_init)

        # Zero-out the last layer of adaLN modulation if it exists
        if hasattr(self, "adaLN_modulation"):
            if self.adaLN_modulation[-1].weight is not None:
                nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            if self.adaLN_modulation[-1].bias is not None:
                nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_msa * self.attn(
            t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        )
        print(x.shape)
        x = x + gate_mlp * self.moe(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        print(f'block shape {x.shape}')
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    

def nearest_divisor(scaled_num_heads, embed_dim):
    # Find all divisors of embed_dim
    divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
    # Find the nearest divisor
    nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))
    
    return nearest

class TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        class_embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        num_experts: int = 4,
        active_experts: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.class_embedding = nn.Linear(class_embed_dim, embed_dim)

        # Define scaling ranges for m_f and m_a
        mf_min, mf_max = 0.5, 4.0
        ma_min, ma_max = 0.5, 1.0

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Calculate scaling factors for the i-th layer using linear interpolation
            mf = mf_min + (mf_max - mf_min) * i / (num_layers - 1)
            ma = ma_min + (ma_max - ma_min) * i / (num_layers - 1)

            # Scale the dimensions according to the scaling factors
            scaled_mlp_dim = int(mlp_dim * mf)
            scaled_num_heads = max(1, int(num_heads * ma))
            scaled_num_heads = nearest_divisor(scaled_num_heads, embed_dim)
            mlp_ratio = int(scaled_mlp_dim / embed_dim)

            # Choose layer type based on the layer index (even/odd)
            if i % 2 == 0:  # Even layers use regular DiT
                self.layers.append(
                    DiTBlock(
                        embed_dim, scaled_num_heads, mlp_ratio, 1, 1, attn_drop=dropout
                    )
                )
            else:  # Odd layers use MoE DiT
                self.layers.append(
                    DiTBlock(
                        embed_dim,
                        scaled_num_heads,
                        mlp_ratio,
                        num_experts,
                        active_experts,
                        attn_drop=dropout,
                    )
                )

        self.output_layer = nn.Linear(embed_dim, input_dim)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Initialize TransformerEncoderLayer modules
                # Initialize the self-attention layers
                nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
                if module.self_attn.in_proj_bias is not None:
                    nn.init.constant_(module.self_attn.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
                if module.self_attn.out_proj.bias is not None:
                    nn.init.constant_(module.self_attn.out_proj.bias, 0)
                # Initialize the linear layers in the feedforward network
                for lin in [module.linear1, module.linear2]:
                    nn.init.xavier_uniform_(lin.weight)
                    if lin.bias is not None:
                        nn.init.constant_(lin.bias, 0)
                # Initialize the LayerNorm layers
                for ln in [module.norm1, module.norm2]:
                    if ln.weight is not None:
                        nn.init.constant_(ln.weight, 1.0)
                    if ln.bias is not None:
                        nn.init.constant_(ln.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_basic_init)

        # Initialize input and class embeddings
        nn.init.xavier_uniform_(self.input_embedding.weight)
        if self.input_embedding.bias is not None:
            nn.init.constant_(self.input_embedding.bias, 0)
        nn.init.xavier_uniform_(self.class_embedding.weight)
        if self.class_embedding.bias is not None:
            nn.init.constant_(self.class_embedding.bias, 0)

        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)

        # Initialize DiTBlocks if any
        for layer in self.layers:
            if isinstance(layer, DiTBlock):
                layer.initialize_weights()

    def forward(self, x, c_emb):
        x = self.input_embedding(x)
        class_emb = self.class_embedding(c_emb)

        for layer in self.layers:
            x = layer(x, class_emb)

        x = self.output_layer(x)
        return x


class PatchMixer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

    def initialize_weights(self):
        def _init_transformer_layer(module):
            # Initialize the self-attention layers
            nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
            if module.self_attn.in_proj_bias is not None:
                nn.init.constant_(module.self_attn.in_proj_bias, 0)
            nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
            if module.self_attn.out_proj.bias is not None:
                nn.init.constant_(module.self_attn.out_proj.bias, 0)
            # Initialize the linear layers in the feedforward network
            for lin in [module.linear1, module.linear2]:
                nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, 0)
            # Initialize the LayerNorm layers
            for ln in [module.norm1, module.norm2]:
                if ln.weight is not None:
                    nn.init.constant_(ln.weight, 1.0)
                if ln.bias is not None:
                    nn.init.constant_(ln.bias, 0)

        # Initialize each TransformerEncoderLayer
        for layer in self.layers:
            _init_transformer_layer(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MicroViDiT(nn.Module):

    def __init__(
        self,
        in_channels, patch_size: tuple,
        embed_dim, num_layers,
        num_heads, mlp_dim,
        caption_embed_dim,
        num_experts=4, active_experts=2,
        dropout=0.0, patch_mixer_layers=2
    ):

        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = in_channels

        # Image processing
        self.patch_embed = Patchify(
            patch_size= patch_size[0],
            in_chans = in_channels,
            embed_dim = embed_dim
          )

        # Timestep embedding
        self.time_embed = TimestepEmbedder(self.embed_dim)

        # Caption embedding
        self.caption_embed = nn.Sequential(
            nn.Linear(caption_embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # MHA for timestep and caption
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)

        # MLP for timestep and caption
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Pool + MLP for (MHA + MLP)
        self.pool_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # Linear layer after MHA+MLP
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)

        # Patch-mixer
        self.patch_mixer = PatchMixer(self.embed_dim, num_heads, patch_mixer_layers)

        # Backbone transformer model
        self.backbone = TransformerBackbone(self.embed_dim, self.embed_dim, self.embed_dim, num_layers, num_heads, mlp_dim,
                                        num_experts, active_experts, dropout)

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, patch_size[0] * patch_size[1] * in_channels)
        )

        # Fixed temporal embedding layer
        self.temporal_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        nn.init.normal_(self.temporal_embed)  # Initialize temporal embeddings

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize all linear layers and biases
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                # Initialize convolutional layers like linear layers
                nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize MultiheadAttention layers
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                # Initialize LayerNorm layers
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                # Initialize TransformerEncoderLayer modules
                # Initialize the self-attention layers
                nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
                if module.self_attn.in_proj_bias is not None:
                    nn.init.constant_(module.self_attn.in_proj_bias, 0)
                nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
                if module.self_attn.out_proj.bias is not None:
                    nn.init.constant_(module.self_attn.out_proj.bias, 0)
                # Initialize the linear layers in the feedforward network
                for lin in [module.linear1, module.linear2]:
                    nn.init.xavier_uniform_(lin.weight)
                    if lin.bias is not None:
                        nn.init.constant_(lin.bias, 0)
                # Initialize the LayerNorm layers
                for ln in [module.norm1, module.norm2]:
                    if ln.weight is not None:
                        nn.init.constant_(ln.weight, 1.0)
                    if ln.bias is not None:
                        nn.init.constant_(ln.bias, 0)

        # Apply basic initialization to all modules
        self.apply(_basic_init)

        # [Rest of the initialization code remains the same...]

        # Initialize the patch embedding projection
        w = self.patch_embed.conv_proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.conv_proj.bias is not None:
            nn.init.constant_(self.patch_embed.conv_proj.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        if self.time_embed.mlp[0].bias is not None:
            nn.init.constant_(self.time_embed.mlp[0].bias, 0)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        if self.time_embed.mlp[2].bias is not None:
            nn.init.constant_(self.time_embed.mlp[2].bias, 0)

        # Initialize caption embedding layers
        for layer in self.caption_embed:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize MLP layers in self.mlp
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize MLP layers in self.pool_mlp
        for layer in self.pool_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize the linear layer in self.linear
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

        # Zero-out the last linear layer in the output to ensure initial predictions are zero
        nn.init.constant_(self.output[-1].weight, 0)
        if self.output[-1].bias is not None:
            nn.init.constant_(self.output[-1].bias, 0)

        # Initialize the backbone
        self.backbone.initialize_weights()

        # Initialize the PatchMixer
        self.patch_mixer.initialize_weights()

    def forward(self, x, t, caption_embeddings, mask=None):
        # x: (batch_size, in_channels, height, width)
        # t: (batch_size, 1)
        # caption_embeddings: (batch_size, caption_embed_dim)
        # mask: (batch_size, num_patches)

        batch_size, channels, frames, height, width = x.shape
        psize_h, psize_w = self.patch_size
        seq_len = frames * height // psize_h * width // psize_w

        # Image processing
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)
        print(f'patched {x.shape}')
        B, N, D = x.shape

        # Generate positional embeddings
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, frames, height // psize_h, width // psize_w)

        pos_embed = pos_embed.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)
        print(f'{x.shape = } ,{pos_embed.shape = }')

        x = x + pos_embed

        # Timestep embedding
        t_emb = self.time_embed(t)  # (batch_size, embed_dim)

        # Caption embedding
        c_emb = self.caption_embed(caption_embeddings)  # (batch_size, embed_dim)
        print(f'c_emb {c_emb.shape}, t_emb {t_emb.shape}')

        mha_out = self.mha(t_emb.unsqueeze(1), c_emb, c_emb)[0].squeeze(1)
        mlp_out = self.mlp(mha_out)
        # Pool + MLP
        print(f'mlp_out {mlp_out.shape}')

        pool_out = self.pool_mlp(mlp_out.unsqueeze(2))

        # Pool + MLP + t_emb
        pool_out = (pool_out + t_emb).unsqueeze(1)
        print(f'pool out {pool_out.shape}')

        # Apply linear layer
        cond_signal = self.linear(mlp_out).unsqueeze(1)  # (batch_size, 1, embed_dim)
        cond = (cond_signal + pool_out).expand(-1, x.shape[1], -1)

        print(f'cond {cond.shape} / cond_signal {cond_signal.shape}')

        # Add conditioning signal to all patches
        # (batch_size, num_patches, embed_dim)
        x = x + cond

        # Patch-mixer
        x = self.patch_mixer(x)

        # Remove masked patches
        if mask is not None:
            x = remove_masked_patches(x, mask)
        print(f'x masked {x.shape}')

        # MHA + MLP + Pool + MLP + t_emb
        print(f'mlp out {mlp_out.unsqueeze(1).shape} / pool {pool_out.shape}')

        cond = (mlp_out.unsqueeze(1) + pool_out).expand(-1, x.shape[1], -1)
        print(f'{cond.shape = }')

        x = x + cond

        # Backbone transformer model
        x = self.backbone(x, cond)
        print(f'backbone out {x.shape}')
        # Final output layer
        # (bs, unmasked_num_patches, embed_dim) -> (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels)
        x = self.output(x)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels) -> (bs, num_patches, patch_size_h * patch_size_w * in_channels)
            x = add_masked_patches(x, mask)

        print(f'unmasked {x.shape}')

        x = rearrange(
            x,
            "B (T hp wp) (p1 p2 c) -> B c T (hp p1) (wp p2)",
            T=frames,
            hp=height // self.patch_size[0],
            wp=width // self.patch_size[1],
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            c=self.out_channels
        )

        print(f'vtheta => {x.shape}')

        return x

    @torch.no_grad()
    def sample(self, z, cond, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        latents = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device).to(torch.float16)

            vc = self(z, t, cond, None)
            null_cond = torch.zeros_like(cond)
            vu = self(z, t, null_cond)
            vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            latents.append(z)

        return latents[-1]



def batch_trainer(
    epochs,
    model,
    optimizer,
    train_loader,
    accelerator
):
    batch = next(iter(train_loader))
    patch_size = config.patch_size
    device = accelerator.device
    losses = []
    
    # wandb_logger(key='', project_name='microvid')
    
    print('start single batch training..../ \n')

    for epoch in tqdm(range(epochs), desc='Training...'):
        # progress_bar = tqdm(dataset, desc=f"Epoch {epoch}", leave=False)
    # for batch_idx, batch in enumerate(dataset):
        optimizer.zero_grad()
        
        latents = batch["vae_latent"][0].to(device)
        caption_embeddings = batch["text_embedding"][0].to(device)
        # print(f'{latents.shape = } / {caption_embeddings.shape = }')
        bs, c, d, h, w = latents.shape
        # latents = latents * VAE_SCALING_FACTOR

        mask = random_mask(bs, d, h, w, patch_size=patch_size, mask_ratio=MASK_RATIO).to(device)

        nt = torch.randn((bs,)).to(device)
        t = torch.sigmoid(nt)
        
        texp = t.view([bs, *([1] * len(latents.shape[1:]))]).to(device)
        z1 = torch.randn_like(latents, device=device)
        zt = (1 - texp) * latents + texp * z1
        
        vtheta = model(zt, t, caption_embeddings, mask)

        latents = apply_mask_to_tensor(latents, mask, patch_size)
        vtheta = apply_mask_to_tensor(vtheta, mask, patch_size)
        z1 = apply_mask_to_tensor(z1, mask, patch_size)

        batchwise_mse = ((z1 - latents - vtheta) ** 2).mean(dim=list(range(1, len(latents.shape))))
        loss = batchwise_mse.mean()
        loss = loss * 1 / (1 - MASK_RATIO)
        print(f'epoch {epoch}, loss => {loss.item():.4f}')
        wandb.log({'loss/train': loss.item(), "log_loss/train": math.log10(loss.item())})
        
        accelerator.backward(loss)
        optimizer.step()

        if accelerator.is_local_main_process:
            losses.append(loss.item())
            
    return losses[-1]

def decode_video_tensor_to_gif(video_tensor, output_path, fps=30, grid_size=None):
    """
    Decodes a video tensor batch back into a video grid and saves it as a GIF.

    Args:
        video_tensor (torch.Tensor): A tensor of shape (B, C, T, H, W) representing a batch of videos.
            - B: Batch size (number of videos)
            - C: Number of channels (e.g., 3 for RGB)
            - T: Number of frames
            - H: Height of the frame
            - W: Width of the frame
        output_path (str): Path to the output GIF file.
        fps (int, optional): Frames per second for the output GIF. Defaults to 30.
        grid_size (tuple[int, int], optional): Tuple of (rows, cols) for creating a grid of videos, default is None,
                                                which will use a horizontal grid of all videos in batch.
    """

    if not isinstance(video_tensor, torch.Tensor):
        raise TypeError("Input video_tensor must be a torch.Tensor")

    if video_tensor.ndim != 5:
        raise ValueError(
            f"Video tensor must have 5 dimensions (B, C, T, H, W), got {video_tensor.ndim}"
        )

    batch_size, channels, num_frames, height, width = video_tensor.shape

    if grid_size is None:
        grid_rows = 1
        grid_cols = batch_size
    else:
        grid_rows, grid_cols = grid_size
        if grid_rows * grid_cols != batch_size:
            raise ValueError(
                f"grid_size rows * cols {grid_rows * grid_cols} must match batch size {batch_size}"
            )

    frames_batch_list = []

    for frame_idx in range(num_frames):
        # create a grid of frames from each video
        frames_batch = []
        for row_idx in range(grid_rows):
            frame_row = []
            for col_idx in range(grid_cols):
                video_idx = row_idx * grid_cols + col_idx
                frame = video_tensor[video_idx, :, frame_idx, :, :]
                frame = frame.permute(1, 2, 0).cpu().numpy()
                frame = (frame * 255).astype(
                    np.uint8
                )  # Ensure frame is in [0, 255] range
                if frame.shape[2] == 1:  # convert graycale to RGB
                    frame = np.concatenate(
                        [frame, frame, frame], axis=2
                    )  # convert to RGB
                frame_row.append(frame)
            frame_row = np.concatenate(frame_row, axis=1)
            frames_batch.append(frame_row)
        frames_batch = np.concatenate(frames_batch, axis=0)

        frames_batch_list.append(frames_batch)
    # Save as GIF
    imageio.mimsave(output_path, frames_batch_list, fps=fps)
    
    return output_path


def wandb_logger(key: str, project_name, run_name=None):
    # initilaize wandb
    wandb.login(key=key)
    wandb.init(
        project=project_name,
        name=run_name or None,
        settings=wandb.Settings(init_timeout=120),
    )

def save_video(final_frames, output_path='sample.mp4', fps=4):
    assert final_frames.ndim == 4 and final_frames.shape[3] == 3, f"invalid shape: {final_frames} (need t h w c)"
    if final_frames.dtype != np.uint8:
        final_frames = (final_frames * 255).astype(np.uint8)
        
    ImageSequenceClip(list(final_frames), fps=fps).write_videofile(output_path)
    
    return output_path

def process_video_latents(latents, vae=ltx_vae):
    latents = vae.tiled_decode(latents) / scale_factor
    latents = rearrange(latents, 'b c t h w -> b t c h w')

    return latents

def sample_video(step, model, captions):
    pred_model = model.eval()
    image_batch = pred_model.sample(captions)
    file = f"fmsamples/{step}_microvid.gif"
    batch = [process_video_latents(x) for x in image_batch]

    gridfile = decode_video_tensor_to_gif(batch, file)
    print(f"sample saved @ {gridfile}")
    del pred_model

    return gridfile


def collate(batch):
    # print(batch[1][1].shape)
    latents = torch.stack([item[0] for item in batch], dim=0)
    # print(f'{latents.shape = }')
    text = [item[1][0] for item in batch]
    labels = torch.stack(text, dim=0)

    return {
        "video_latents": latents,
        "text_encoded": labels
    }

@click.command()
@click.option("-r", "--run", default="single_batch")
@click.option("-e", "--epochs", default=10)
@click.option("-bs", "--batch_size", default=32)
def main(run, epochs, batch_size):
    embed_dim = 384
    depth = 2

    accelerator = Accelerator(mixed_precision='bf16')
    device = accelerator.device

    # DiT-B config
    microdit = MicroViDiT(
        in_channels=256,
        patch_size = (1, 1),
        embed_dim=embed_dim,
        num_layers=depth,
        num_heads=8,
        mlp_dim=embed_dim,
        caption_embed_dim=768,
        num_experts=8, active_experts=2,
        dropout=0.0, patch_mixer_layers=2
    ).to(device)

    n_params = sum(p.numel() for p in microdit.parameters())
    print(f"model parameters count: {n_params/1e6:.2f}M, ")

    dataset = Text2VideoDataset()
    t2v_train_loader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=collate
    )

    optimizer = torch.optim.AdamW(microdit.parameters(), lr=1e-3)

    sp = next(iter(t2v_train_loader))
    print(f"loaded data \n data sample: latents - {sp['video_latents'].shape}, text cond - {sp['text_encoded'].shape}")

    microdit, optimizer, t2v_train_loader = accelerator.prepare(microdit, optimizer, t2v_train_loader)

    if run == "single_batch":
        loss = batch_trainer(
            epochs=epochs, model=microdit,
            optimizer=optimizer, train_loader=t2v_train_loader,
            accelerator=accelerator
        ) # type: ignore

        # wandb.finish()
        print(f"single batch training ended at loss: {loss:.4f}")

    elif run == "train":
        print(f"you missed your train looop impl boy")


main()
