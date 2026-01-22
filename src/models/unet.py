import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utilities
# -------------------------


def _largest_divisor_leq(n: int, max_divisor: int) -> int:
    """Largest d <= max_divisor such that n % d == 0. Falls back to 1."""
    for d in range(min(max_divisor, n), 0, -1):
        if n % d == 0:
            return d
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for diffusion timesteps (DDPM formula).
    Input: t of shape (B,) (int or float)
    Output: (B, dim)
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Time embedding dim must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Scale t from [0, 1] to [0, 1000] for proper frequency resolution
        # This is important for rectified flow where t is in [0, 1]
        t = t.float() * 1000.0
        half = self.dim // 2
        # Standard DDPM frequency formula
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class TimeMLP(nn.Module):
    def __init__(self, time_emb_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)


class LearnablePositionalEncoding3D(nn.Module):
    """
    Learnable positional encoding for 3D attention.
    Adds decomposed position embeddings (D + H + W) to tokens.
    """

    def __init__(
        self, channels: int, max_d: int = 16, max_h: int = 16, max_w: int = 16
    ):
        super().__init__()
        self.d_embed = nn.Parameter(torch.randn(1, channels, max_d, 1, 1) * 0.02)
        self.h_embed = nn.Parameter(torch.randn(1, channels, 1, max_h, 1) * 0.02)
        self.w_embed = nn.Parameter(torch.randn(1, channels, 1, 1, max_w) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, D, H, W)"""
        _, _, d, h, w = x.shape
        pos = (
            self.d_embed[:, :, :d]
            + self.h_embed[:, :, :, :h]
            + self.w_embed[:, :, :, :, :w]
        )
        return x + pos


# -------------------------
# Core blocks
# -------------------------


class ResBlock3D(nn.Module):
    """
    ResNet block with timestep conditioning via FiLM (scale/shift).
    x: (B, C, D, H, W)
    t_emb: (B, time_dim)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        g1 = _largest_divisor_leq(in_channels, 32)
        g2 = _largest_divisor_leq(out_channels, 32)

        self.norm1 = nn.GroupNorm(g1, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_dim, 2 * out_channels)  # scale + shift

        self.norm2 = nn.GroupNorm(g2, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # FiLM conditioning
        scale_shift = self.time_proj(t_emb)  # (B, 2*out_channels)
        scale, shift = scale_shift.chunk(2, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]

        h = self.norm2(h)
        h = h * (1 + scale) + shift
        h = F.silu(h)

        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


class SelfAttention3D(nn.Module):
    """
    Self-attention with AdaLN-Zero conditioning on timestep.
    Uses Flash Attention (scaled_dot_product_attention) for memory efficiency.
    """

    def __init__(
        self,
        channels: int,
        time_dim: int,
        num_heads: int = 4,
        max_d: int = 16,
        max_h: int = 16,
        max_w: int = 16,
    ):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        # Positional encoding
        self.pos_enc = LearnablePositionalEncoding3D(channels, max_d, max_h, max_w)

        # AdaLN-Zero: produces scale, shift for pre-norm, and gate for output
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, 3 * channels),  # scale, shift, gate
        )

        g = _largest_divisor_leq(channels, 32)
        self.norm = nn.GroupNorm(g, channels, affine=False)  # No learnable params
        self.qkv = nn.Conv3d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

        # Zero-init the projection for stable training
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        t_emb: (B, time_dim)
        """
        b, c, d, h, w = x.shape
        n = d * h * w

        # AdaLN modulation
        modulation = self.adaLN_modulation(t_emb)  # (B, 3C)
        scale, shift, gate = modulation.chunk(3, dim=1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        gate = gate[:, :, None, None, None]

        # Normalize and modulate
        h_norm = self.norm(x)
        h_norm = h_norm * (1 + scale) + shift

        # Add positional encoding
        h_norm = self.pos_enc(h_norm)

        # QKV projection
        qkv = self.qkv(h_norm)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for attention: (B, heads, N, head_dim)
        q = q.view(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        k = k.view(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        v = v.view(b, self.num_heads, self.head_dim, n).transpose(2, 3)

        # Flash Attention (PyTorch 2.0+)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)

        # Reshape back
        out = out.transpose(2, 3).reshape(b, c, d, h, w)
        out = self.proj(out)

        # Gated residual (AdaLN-Zero)
        return x + gate * out


class Downsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


# -------------------------
# Block wrappers
# -------------------------


class DownBlock(nn.Module):
    """A ResBlock optionally followed by Attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        dropout: float,
        use_attn: bool,
        num_heads: int,
        max_spatial: int,
    ):
        super().__init__()
        self.res = ResBlock3D(in_ch, out_ch, time_dim, dropout)
        self.attn = (
            SelfAttention3D(
                out_ch, time_dim, num_heads, max_spatial, max_spatial, max_spatial
            )
            if use_attn
            else None
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, t_emb)
        return x


class UpBlock(nn.Module):
    """A ResBlock (with skip concat handled externally) optionally followed by Attention."""

    def __init__(
        self,
        in_ch: int,  # includes skip channels
        out_ch: int,
        time_dim: int,
        dropout: float,
        use_attn: bool,
        num_heads: int,
        max_spatial: int,
    ):
        super().__init__()
        self.res = ResBlock3D(in_ch, out_ch, time_dim, dropout)
        self.attn = (
            SelfAttention3D(
                out_ch, time_dim, num_heads, max_spatial, max_spatial, max_spatial
            )
            if use_attn
            else None
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.res(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, t_emb)
        return x


# -------------------------
# UNet
# -------------------------


@dataclass
class UNet3DConfig:
    in_channels: int = 4
    out_channels: int = 4  # predict noise in same channel count
    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4, 8)  # 32 -> 16 -> 8 -> 4
    num_res_blocks: int = 2
    attn_levels: Tuple[int, ...] = (2, 3)  # apply attention at levels 8^3 and 4^3
    num_heads: int = 4
    dropout: float = 0.0
    time_emb_dim: int = 256  # sinusoidal dim before MLP
    time_hidden_dim: int = 512  # MLP output dim (used by ResBlocks)
    input_spatial: int = 32  # assumed input spatial size (D=H=W)


class UNet3DDiffusion(nn.Module):
    """
    3D diffusion UNet with timestep conditioning + self-attention at chosen levels.

    Key features:
    - FiLM conditioning in ResBlocks
    - AdaLN-Zero conditioning in Attention blocks
    - Flash Attention for memory efficiency
    - Learnable 3D positional encoding in attention
    - Proper skip connections (one per ResBlock)

    Input: x (B, C, D, H, W), t (B,) timesteps/noise level
    Output: (B, out_channels, D, H, W)
    """

    def __init__(self, cfg: UNet3DConfig):
        super().__init__()
        self.cfg = cfg

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(cfg.time_emb_dim)
        self.time_mlp = TimeMLP(cfg.time_emb_dim, cfg.time_hidden_dim)

        # Stem
        self.in_conv = nn.Conv3d(
            cfg.in_channels, cfg.base_channels, kernel_size=3, padding=1
        )

        # Down path
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = cfg.base_channels
        self.skip_channels = []  # Track channel count for each skip
        spatial = cfg.input_spatial

        for level, mult in enumerate(cfg.channel_mults):
            out_ch = cfg.base_channels * mult
            use_attn = level in cfg.attn_levels

            for _ in range(cfg.num_res_blocks):
                self.down_blocks.append(
                    DownBlock(
                        ch,
                        out_ch,
                        cfg.time_hidden_dim,
                        cfg.dropout,
                        use_attn,
                        cfg.num_heads,
                        spatial,
                    )
                )
                ch = out_ch
                self.skip_channels.append(ch)  # One skip per block

            if level != len(cfg.channel_mults) - 1:
                self.downsamples.append(Downsample3D(ch))
                spatial //= 2
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        mid_spatial = spatial
        self.mid1 = ResBlock3D(ch, ch, cfg.time_hidden_dim, cfg.dropout)
        self.mid_attn = SelfAttention3D(
            ch,
            cfg.time_hidden_dim,
            cfg.num_heads,
            mid_spatial,
            mid_spatial,
            mid_spatial,
        )
        self.mid2 = ResBlock3D(ch, ch, cfg.time_hidden_dim, cfg.dropout)

        # Up path (reverse order)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # We need to pop skip_channels in reverse order, so make a copy for iteration
        skip_channels_rev = list(reversed(self.skip_channels))
        skip_idx = 0

        for level in reversed(range(len(cfg.channel_mults))):
            out_ch = cfg.base_channels * cfg.channel_mults[level]
            use_attn = level in cfg.attn_levels

            if level != len(cfg.channel_mults) - 1:
                self.upsamples.append(Upsample3D(ch))
                spatial *= 2
            else:
                self.upsamples.append(nn.Identity())

            for _ in range(cfg.num_res_blocks):
                skip_ch = skip_channels_rev[skip_idx]
                skip_idx += 1
                self.up_blocks.append(
                    UpBlock(
                        ch + skip_ch,
                        out_ch,
                        cfg.time_hidden_dim,
                        cfg.dropout,
                        use_attn,
                        cfg.num_heads,
                        spatial,
                    )
                )
                ch = out_ch

        # Head
        g = _largest_divisor_leq(ch, 32)
        self.out_norm = nn.GroupNorm(g, ch)
        self.out_conv = nn.Conv3d(ch, cfg.out_channels, kernel_size=3, padding=1)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Proper initialization for diffusion models."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init final layer for stable training (predicts zero noise initially)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        t: (B,) (can be int timesteps or float noise levels)
        """
        t_emb = self.time_mlp(self.time_emb(t))  # (B, time_hidden_dim)

        h = self.in_conv(x)

        # Down: store skip after each block
        skips = []
        block_idx = 0
        for level in range(len(self.cfg.channel_mults)):
            for _ in range(self.cfg.num_res_blocks):
                h = self.down_blocks[block_idx](h, t_emb)
                skips.append(h)
                block_idx += 1
            h = self.downsamples[level](h)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h, t_emb)
        h = self.mid2(h, t_emb)

        # Up: pop skip before each block
        block_idx = 0
        for level in reversed(range(len(self.cfg.channel_mults))):
            h = self.upsamples[len(self.cfg.channel_mults) - 1 - level](h)
            for _ in range(self.cfg.num_res_blocks):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.up_blocks[block_idx](h, t_emb)
                block_idx += 1

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


# -------------------------
# Quick sanity check
# -------------------------
if __name__ == "__main__":
    cfg = UNet3DConfig(
        in_channels=4,
        out_channels=4,
        base_channels=96,
        channel_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_levels=(2, 3),  # 8^3 and 4^3
        num_heads=4,
        dropout=0.0,
        input_spatial=32,
    )
    model = UNet3DDiffusion(cfg)
    x = torch.randn(2, 4, 32, 32, 32)
    t = torch.randint(0, 1000, (2,))
    y = model(x, t)
    print(f"Output shape: {y.shape}")  # Expected: (2, 4, 32, 32, 32)

    # Gradient flow check
    loss = y.sum()
    loss.backward()
    assert model.in_conv.weight.grad is not None, "No gradient on in_conv"
    print("✓ Gradient flow check passed")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params:,}")
