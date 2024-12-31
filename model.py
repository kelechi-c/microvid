import torch
import torch.nn as nn
import math
from einops import rearrange
from timm.models.vision_transformer import Attention
import torch.nn.functional as F
from typing import Optional
from utils import remove_masked_patches, add_masked_patches

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# the text sequence from t5 is 2D already, so no need to add a 2nd dimension
def t2i_modulate(x, shift, scale): 
    return x * (1 + scale) + shift

## Note: for 3D Pos-embed, embed_dim/hidden size should be divisible by 6 (384, 768, 1152)
# Generates 3D sinusoidal positional embeddings.
def get_3d_sincos_pos_embed(embed_dim, t, h, w):
    grid_t = torch.arange(t, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_t, grid_h, grid_w = torch.meshgrid(grid_t, grid_h, grid_w, indexing="ij")
    grid = torch.stack([grid_t, grid_h, grid_w], dim=0)  # Shape: (3, T, H, W)

    grid = grid.reshape([3, t, h, w])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    
    assert embed_dim % 2 == 0
    emb_dim_per_axis = embed_dim // 3  # Divide embedding dimension across 3 axes
    assert emb_dim_per_axis % 2 == 0

    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
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
    
    emb_t = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(emb_dim_per_axis, grid[2])  # (T*H*W, D/3)

    emb = torch.cat([emb_t, emb_h, emb_w], dim=1)  # (T*H*W, D)
    return emb


# uses conv2d/spatiotemporal flattening, Strategy usd in Mochi. Will change if it doesn't work properly
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


class MoEGate(nn.Module):
    def __init__(
        self, embed_dim, num_experts=8, num_experts_per_tok=2, aux_loss_alpha=0.01
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
        # print(f'{hidden_states.shape = }')
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
    def __init__(self, hidden_size, intermediate_size, pretraining_tp=1):
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
            # print(len(gate_proj_slices))

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
        num_experts=8,
        num_experts_per_tok=2,
        pretraining_tp=1,
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
        x = x + gate_mlp * self.moe(
            t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


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
            scaled_num_heads = self.nearest_divisor(scaled_num_heads, embed_dim)
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

    def nearest_divisor(self, scaled_num_heads, embed_dim):
        # Find all divisors of embed_dim
        divisors = [i for i in range(1, embed_dim + 1) if embed_dim % i == 0]
        # Find the nearest divisor
        nearest = min(divisors, key=lambda x: abs(x - scaled_num_heads))
        
        return nearest
    
    
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
        in_channels,
        patch_size: tuple,
        embed_dim,
        num_layers,
        num_heads,
        mlp_dim,
        caption_embed_dim,
        num_experts=4,
        active_experts=2,
        dropout=0.0,
        patch_mixer_layers=4,
    ):

        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = in_channels

        # Image processing
        self.patch_embed = Patchify(
            patch_size=patch_size[0], in_chans=in_channels, embed_dim=embed_dim
        )

        # Timestep embedding
        self.time_embed = TimestepEmbedder(self.embed_dim)

        # Caption embedding
        self.caption_embed = nn.Sequential(
            nn.Linear(caption_embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # MHA for timestep and caption
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)

        # MLP for timestep and caption
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Pool + MLP for (MHA + MLP)
        self.pool_mlp = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Linear layer after MHA+MLP
        self.linear = nn.Linear(self.embed_dim, self.embed_dim)

        # Patch-mixer
        self.patch_mixer = PatchMixer(self.embed_dim, num_heads, patch_mixer_layers)

        # Backbone transformer model
        self.backbone = TransformerBackbone(
            self.embed_dim,
            self.embed_dim,
            self.embed_dim,
            num_layers,
            num_heads,
            mlp_dim,
            num_experts,
            active_experts,
            dropout,
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, patch_size[0] * patch_size[1] * in_channels),
        )

        # Fixed temporal embedding layer
        # self.temporal_embed = nn.Parameter(torch.zeros(1, 3, embed_dim))
        # nn.init.normal_(self.temporal_embed)  # Initialize temporal embeddings

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
        # caption_embeddings: (batch_size, seq_len, caption_embed_dim)
        # mask: (batch_size, num_patches)

        batch_size, channels, frames, height, width = x.shape
        psize_h, psize_w = self.patch_size
        seq_len = frames * height // psize_h * width // psize_w

        # Image processing
        x = self.patch_embed(x)  # (batch_size, num_patches, embed_dim)

        # Generate positional embeddings
        pos_embed = get_3d_sincos_pos_embed(self.embed_dim, frames, height // psize_h, width // psize_w)
        pos_embed = pos_embed.to(x.device).unsqueeze(0).expand(batch_size, -1, -1)

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

        # Apply linear layer
        cond_signal = self.linear(mlp_out).unsqueeze(1)  # (batch_size, 1, embed_dim)
        cond = (cond_signal + pool_out).expand(-1, x.shape[1], -1)

        # Add conditioning signal to all patches
        # (batch_size, num_patches, embed_dim)
        x = x + cond

        # Patch-mixer step
        x = self.patch_mixer(x)

        # Remove masked patches
        if mask is not None:
            x = remove_masked_patches(x, mask)

        # MHA + MLP + Pool + MLP + t_emb
        cond = (mlp_out.unsqueeze(1) + pool_out).expand(-1, x.shape[1], -1)
        print(f'{cond.shape = }')

        x = x + cond

        # Backbone transformer model
        x = self.backbone(x, cond)
        # Final output layer
        # (bs, unmasked_num_patches, embed_dim) -> (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels)
        x = self.output(x)

        # Add masked patches
        if mask is not None:
            # (bs, unmasked_num_patches, patch_size_h * patch_size_w * in_channels) -> (bs, num_patches, patch_size_h * patch_size_w * in_channels)
            x = add_masked_patches(x, mask)

        # unpatchify
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

        return x

    @torch.no_grad()
    def sample(self, z, cond, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        latents = [z]

        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device).to(torch.bfloat16)

            vc = self(z, t, cond, None)
            null_cond = torch.zeros_like(cond)
            vu = self(z, t, null_cond)
            vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            latents.append(z)

        return latents[-1]
