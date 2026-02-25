
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import xformers.ops as xops

    _XFORMERS_AVAILABLE = True
except Exception:
    xops = None
    _XFORMERS_AVAILABLE = False


@dataclass
class DiT3DConfig:
    input_size: Tuple[int, int, int] = (200, 200, 200)
    patch_size: Tuple[int, int, int] = (20, 20, 20)
    patch_stride: Tuple[int, int, int] = (20, 20, 20)
    in_channels: int = 2
    hidden_size: int = 768  # DiT-B equivalent
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    learn_sigma: bool = False  # If True, output has 2*in_channels
    dropout: float = 0.0
    overlap_norm: bool = True
    attn_backend: str = "auto"


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps/noise-levels into vector representations.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        t: (N,) tensor of noise levels or timesteps
        dim: embedding dimension
        max_period: max period for sinusoidal functions
        """
        # (N,) -> (N, dim)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:  # handle odd dim by padding with zero
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is 1D tensor of log-snr or noise level
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, attn_backend: str = "auto"):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_backend = attn_backend

    def _select_backend(self, x: torch.Tensor) -> str:
        backend = self.attn_backend
        if backend in ("xformers", "flash", "math"):
            if backend == "xformers" and (not _XFORMERS_AVAILABLE or not x.is_cuda):
                return "flash"
            if backend in ("flash", "math") and not x.is_cuda:
                return "math"
            return backend
        # auto
        if _XFORMERS_AVAILABLE and x.is_cuda:
            return "xformers"
        return "flash"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, N, H, Hd)

        backend = self._select_backend(x)
        if backend == "xformers":
            attn = xops.memory_efficient_attention(q, k, v)
            out = attn.reshape(b, n, d)
            return self.proj(out)

        q = q.transpose(1, 2)  # (B, H, N, Hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if backend == "flash" and x.is_cuda and hasattr(torch.backends, "cuda"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(q, k, v)
        elif backend == "math" and x.is_cuda and hasattr(torch.backends, "cuda"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class PatchEmbed3D(nn.Module):
    """3D volumetric data to Patch Embedding"""

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (200, 200, 200),
        patch_size: Tuple[int, int, int] = (20, 20, 20),
        patch_stride: Tuple[int, int, int] = (20, 20, 20),
        in_channels: int = 2,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        # Check valid stride sizing
        for d, p, s in zip(input_size, patch_size, patch_stride):
            if d < p:
                raise ValueError(f"Input dimension {d} smaller than patch size {p}")
            if (d - p) % s != 0:
                raise ValueError(
                    f"Input dimension {d} not compatible with patch size {p} and stride {s}"
                )

        self.grid_size = tuple((d - p) // s + 1 for d, p, s in zip(input_size, patch_size, patch_stride))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_stride
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, d', h', w')
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with AdaLN-Zero conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_backend: str = "auto",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, attn_backend=attn_backend)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        
        # AdaLN-Zero modulation parameters
        # 6 * hidden_size for:
        # (gamma1, beta1, alpha1) -> for attention
        # (gamma2, beta2, alpha2) -> for MLP
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), c: (B, D)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        
        # Attention block
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP block
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: Tuple[int, int, int],
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        # Predict linear projection (gamma, beta) for final norm
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
        # Project back to patch pixels
        # Output dim = patch_volume * out_channels
        self.patch_vol = patch_size[0] * patch_size[1] * patch_size[2]
        self.out_channels = out_channels
        self.linear = nn.Linear(
            hidden_size, self.patch_vol * out_channels, bias=True
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, grid_size: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        x: (B, N, hidden_size)
        c: (B, hidden_size)
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)  # (B, N, patch_vol * out_channels)
        return x


class DiT3D(nn.Module):
    """
    3D Diffusion Transformer working on raw volumetric data.
    """

    def __init__(self, config: DiT3DConfig):
        super().__init__()
        self.config = config
        self.overlap_norm = config.overlap_norm
        self._oa_chunk_size = 512

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed3D(
            input_size=config.input_size,
            patch_size=config.patch_size,
            patch_stride=config.patch_stride,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
        )
        
        # 2. Learnable Positional Embedding
        # Intentionally simple / not factorized as requested
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size)
        )
        
        # 3. Time embedding
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_backend=config.attn_backend,
                )
                for _ in range(config.depth)
            ]
        )
        
        # 5. Final Layer
        out_channels = config.in_channels * 2 if config.learn_sigma else config.in_channels
        self.final_layer = FinalLayer(
            hidden_size=config.hidden_size,
            patch_size=config.patch_size,
            out_channels=out_channels,
        )
        
        self.register_buffer("_oa_base_idx", None, persistent=False)
        self.register_buffer("_oa_offsets", None, persistent=False)
        self.register_buffer("_oa_counts", None, persistent=False)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of conv)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

        # Initialize position embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, patch_vol * out_channels)
        return: (B, out_channels, D, H, W)
        """
        if self.config.patch_stride == self.config.patch_size:
            return self._unpatchify_non_overlap(x)
        return self._unpatchify_overlap_add(x)

    def _unpatchify_non_overlap(self, x: torch.Tensor) -> torch.Tensor:
        c = self.final_layer.out_channels
        p_d, p_h, p_w = self.config.patch_size
        g_d, g_h, g_w = self.patch_embed.grid_size
        b = x.shape[0]

        # Reshape to (B, Gd, Gh, Gw, Pd, Ph, Pw, C)
        x = x.view(b, g_d, g_h, g_w, p_d, p_h, p_w, c)
        
        # Permute to (B, C, Gd, Pd, Gh, Ph, Gw, Pw)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        
        # Reshape to final image
        x = x.reshape(
            b, 
            c, 
            g_d * p_d, 
            g_h * p_h, 
            g_w * p_w
        )
        return x

    def _build_overlap_buffers(self, device: torch.device) -> None:
        if self._oa_base_idx is not None and self._oa_base_idx.device == device:
            return

        d, h, w = self.config.input_size
        p_d, p_h, p_w = self.config.patch_size
        s_d, s_h, s_w = self.config.patch_stride
        g_d, g_h, g_w = self.patch_embed.grid_size

        d0 = torch.arange(g_d, device=device, dtype=torch.int64) * s_d
        h0 = torch.arange(g_h, device=device, dtype=torch.int64) * s_h
        w0 = torch.arange(g_w, device=device, dtype=torch.int64) * s_w
        base = (
            d0[:, None, None] * (h * w)
            + h0[None, :, None] * w
            + w0[None, None, :]
        ).reshape(-1)

        od = torch.arange(p_d, device=device, dtype=torch.int64)
        oh = torch.arange(p_h, device=device, dtype=torch.int64)
        ow = torch.arange(p_w, device=device, dtype=torch.int64)
        offsets = (
            od[:, None, None] * (h * w) + oh[None, :, None] * w + ow[None, None, :]
        ).reshape(-1)

        self._oa_base_idx = base
        self._oa_offsets = offsets

        if self.overlap_norm:
            counts = torch.zeros(d * h * w, device=device, dtype=torch.float32)
            chunk = self._oa_chunk_size
            for start in range(0, offsets.numel(), chunk):
                end = min(start + chunk, offsets.numel())
                off = offsets[start:end]
                idx = base[:, None] + off[None, :]
                counts.scatter_add_(
                    0,
                    idx.reshape(-1),
                    torch.ones(idx.numel(), device=device, dtype=counts.dtype),
                )
            self._oa_counts = counts.view(1, 1, d, h, w)
        else:
            self._oa_counts = None

    def _unpatchify_overlap_add(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        c = self.final_layer.out_channels
        p_d, p_h, p_w = self.config.patch_size
        d, h, w = self.config.input_size
        patch_vol = p_d * p_h * p_w

        x = x.view(b, n, c, patch_vol)
        self._build_overlap_buffers(x.device)
        base = self._oa_base_idx
        offsets = self._oa_offsets

        out = x.new_zeros((b, c, d * h * w))
        chunk = self._oa_chunk_size
        for bi in range(b):
            out_b = out[bi]
            vals = x[bi]  # (N, C, patch_vol)
            for start in range(0, patch_vol, chunk):
                end = min(start + chunk, patch_vol)
                off = offsets[start:end]
                idx = base[:, None] + off[None, :]
                idx = idx.reshape(1, -1).expand(c, -1)
                vals_chunk = vals[:, :, start:end].permute(1, 0, 2).reshape(c, -1)
                out_b.scatter_add_(1, idx, vals_chunk)

        out = out.view(b, c, d, h, w)
        if self.overlap_norm:
            counts = self._oa_counts
            if counts is None or counts.device != out.device:
                self._build_overlap_buffers(out.device)
                counts = self._oa_counts
            out = out / counts.to(dtype=out.dtype, device=out.device)
        return out

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W) input data
        t: (B,) log-snr or noise level
        """
        # 1. Embed inputs
        x = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed
        
        # 2. Embed timestep
        c = self.t_embedder(t)  # (B, D)
        
        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)
            
        # 4. Final Layer
        x = self.final_layer(x, c, self.patch_embed.grid_size)
        
        # 5. Unpatchify
        x = self.unpatchify(x)
        
        return x
