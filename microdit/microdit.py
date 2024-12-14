import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

import torch.nn.functional as F

from microdit.trainer import config


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


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


#################################################################################
#                              Flash attention Layer.                           #
#################################################################################


class FlashSelfMHAModified(nn.Module):
    """
    self-attention with flashattention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        device=None,
        dtype=None,
        norm_layer=nn.LayerNorm,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8, got {}".format(
            self.head_dim
        )

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.inner_attn = FlashSelfAttention(attention_dropout=attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x,
    ):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        """
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        q, k, v = qkv.unbind(dim=2)  # [b, s, h, d]
        q = self.q_norm(q).half()  # [b, s, h, d]
        k = self.k_norm(k).half()

        qkv = torch.stack([q, k, v], dim=2)  # [b, s, 3, h, d]
        context = self.inner_attn(qkv)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        return out


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
        if use_flash_attn:
            self.attn = FlashSelfMHAModified(
                hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True
            )
        else:
            self.attn = Attention(
                hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
            )
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
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.moe(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
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
        dropout: float = 0.1,
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

class MicroDiT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_layers, num_heads, mlp_dim,
                 num_experts=4, active_experts=2, dropout=0.1, patch_mixer_layers=2, embed_cat=False):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Image processing
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        # Timestep embedding
        self.time_embed = TimestepEmbedder(self.embed_dim)

        # Caption embedding
        # self.caption_embed = nn.Sequential(
        #     nn.Linear(caption_embed_dim, self.embed_dim),
        #     nn.GELU(),
        #     nn.Linear(self.embed_dim, self.embed_dim)
        # )
        self.label_embed = LabelEmbedder(num_classes=1000, hidden_size=embed_dim, dropout_prob=0.0)

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
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.patch_embed.proj.bias is not None:
            nn.init.constant_(self.patch_embed.proj.bias, 0)

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

        batch_size, channels, height, width = x.shape

        patch_size_h, patch_size_w = self.patch_size

        # Image processing
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        # Generate positional embeddings
        # (height // patch_size_h, width // patch_size_w, embed_dim)
        pos_embed = get_2d_sincos_pos_embed(self.embed_dim, height // patch_size_h, width // patch_size_w)
        pos_embed = pos_embed.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)

        x = x + pos_embed

        # Timestep embedding
        t_emb = self.time_embed(t)  # (batch_size, embed_dim)

        # Caption embedding
        c_emb = self.label_embed(caption_embeddings)  # (batch_size, embed_dim)

        mha_out = self.mha(t_emb.unsqueeze(1), c_emb.unsqueeze(1), c_emb.unsqueeze(1))[0].squeeze(1)
        mlp_out = self.mlp(mha_out)

        # Pool + MLP
        pool_out = self.pool_mlp(mlp_out.unsqueeze(2))

        # Pool + MLP + t_emb
        pool_out = (pool_out + t_emb).unsqueeze(1)

        # Apply linear layer
        cond_signal = self.linear(mlp_out).unsqueeze(1)  # (batch_size, 1, embed_dim)
        cond = (cond_signal + pool_out).expand(-1, x.shape[1], -1)

        # Add conditioning signal to all patches
        # (batch_size, num_patches, embed_dim)
        x = x + cond

        # Patch-mixer
        x = self.patch_mixer(x)

        # Remove masked patches
        if mask is not None:
            x = remove_masked_patches(x, mask)

        # MHA + MLP + Pool + MLP + t_emb
        cond = (mlp_out.unsqueeze(1) + pool_out).expand(-1, x.shape[1], -1)

        x = x + cond

        # Backbone transformer model
        x = self.backbone(x, c_emb)

        # Final output layer
        # (bs, unmasked_num_patches, embed_dim) -> (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels)
        x = self.output(x)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels) -> (bs, num_patches, patch_size_h * patch_size_w * in_channels)
            x = add_masked_patches(x, mask)

        x = unpatchify(x, self.patch_size, height, width)

        return x

    @torch.no_grad()
    def sample(self, z, cond, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device).to(torch.float16)

            vc = self(z, t, cond, None)
            null_cond = torch.zeros_like(cond)
            vu = self(z, t, null_cond)
            vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)

        return (images[-1] / config.vaescale_factor)


class RectFlowWrapper:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)

        return images