import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Mask utility                                                                 #
# --------------------------------------------------------------------------- #


def _ensure_mask_shape(mask, target_shape, device):
    """Reshape mask to match target tensor dimensions."""
    batch_size = target_shape[0]

    if len(target_shape) == 3:  # [B, seq_len, D]
        seq_len = target_shape[1]
        if mask is None:
            return torch.ones(batch_size, seq_len, 1, device=device)
        if mask.dim() == 1:
            return mask.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
        elif mask.dim() == 2:
            if mask.shape[1] == 1:
                return mask.unsqueeze(2).expand(batch_size, seq_len, 1)
            else:
                return mask.unsqueeze(2)
        elif mask.dim() == 3:
            return mask

    elif len(target_shape) == 2:  # [B, D]
        if mask is None:
            return torch.ones(batch_size, 1, device=device)
        if mask.dim() == 1:
            return mask.view(batch_size, 1)
        elif mask.dim() == 2:
            return mask
        elif mask.dim() == 3:
            return mask.mean(dim=1)

    return mask


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask=None):
        if mask is not None:
            return self.fn(x, mask) * self.g
        else:
            return self.fn(x) * self.g


class Block1d(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()

        # Find largest divisor of dim_out that is <= groups
        if dim_out % groups != 0:
            for i in range(groups, 0, -1):
                if dim_out % i == 0:
                    groups = i
                    break
        groups = max(1, min(groups, dim_out))

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, kernel_size=3, padding=1),
            nn.GroupNorm(groups, dim_out),
            Mish(),
        )

    def forward(self, x, mask):
        B, C, L = x.shape

        if mask.dim() == 2:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 3 and mask.shape[2] != L:
            mask = F.interpolate(mask, size=L, mode="nearest")

        output = self.block(x * mask)

        if mask.shape[2] != output.shape[2]:
            mask = F.interpolate(mask, size=output.shape[2], mode="nearest")

        return output * mask


class ResnetBlock1d(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1d(dim, dim_out, groups=groups)
        self.block2 = Block1d(dim_out, dim_out, groups=groups)
        self.res_conv = (
            nn.Conv1d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        )

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        time_emb = self.mlp(time_emb).unsqueeze(-1)  # [B, C] -> [B, C, 1]
        h = self.block2(h + time_emb, mask)
        return h + self.res_conv(x * mask)


class LinearAttention1d(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, kernel_size=1)

    def forward(self, x, mask=None):
        b, c, l = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.heads, -1, l)
        k = k.view(b, self.heads, -1, l)
        v = v.view(b, self.heads, -1, l)

        if mask is not None:
            if mask.dim() == 3 and mask.shape[1] != 1:
                mask = mask.mean(dim=1, keepdim=True)
            if mask.shape[2] == l:
                k = k * mask.unsqueeze(1)

        k = k.softmax(dim=-1)
        context = torch.matmul(k, v.transpose(-2, -1))
        attended = torch.matmul(context, q)
        out = attended.reshape(b, -1, l)

        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


# --------------------------------------------------------------------------- #
# BridgeUNet                                                                   #
# --------------------------------------------------------------------------- #


class BridgeUNet(nn.Module):
    """
    1D U-Net score estimator for the Schrödinger Bridge denoising step.

    Input channels = pred_dim (x_t) + context_dim (encoded context) + pred_dim (x_1).
    Outputs a pred_dim prediction of x_0.

    Original name: GradLogPEstimator1d
    """

    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        context_dim=None,
        pred_dim=None,
        pe_scale=1000,
    ):
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.pe_scale = pe_scale

        self.context_dim = context_dim or 1
        self.pred_dim = pred_dim or 1

        # Time embedding
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        # Input channels: x_t + context + x_1
        input_channels = max(3, self.pred_dim + self.context_dim + self.pred_dim)

        dims = [input_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Down path
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock1d(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock1d(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention1d(dim_out))),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention1d(mid_dim)))
        self.mid_block2 = ResnetBlock1d(mid_dim, mid_dim, time_emb_dim=dim)

        # Up path
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock1d(
                            dim_out * 2, dim_in, time_emb_dim=dim
                        ),  # *2 for skip
                        ResnetBlock1d(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention1d(dim_in))),
                        Upsample1d(dim_in),
                    ]
                )
            )

        # Output
        self.final_block = Block1d(dim * dim_mults[0], dim * dim_mults[0])
        self.final_conv = nn.Conv1d(dim * dim_mults[0], self.pred_dim, kernel_size=1)

    def forward(self, x_t, mask, x_1, t, context=None):
        """
        Args:
            x_t:     Noisy sample [B, seq_len, pred_dim]
            mask:    Binary mask (various shapes)
            x_1:     Foundation model prediction [B, seq_len, pred_dim]
            t:       Timestep [B]
            context: Encoded context [B, seq_len, context_dim]

        Returns:
            x_0_pred: Predicted x_0 [B, seq_len, pred_dim]
        """
        t_emb = self.mlp(self.time_pos_emb(t, scale=self.pe_scale))

        is_2d_input = x_t.dim() == 2
        if is_2d_input:
            x_t = x_t.unsqueeze(1)
            x_1 = x_1.unsqueeze(1)

        B, L, D = x_t.shape
        mask = _ensure_mask_shape(mask, x_t.shape, x_t.device)

        if context is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1).expand(-1, L, -1)
            elif context.shape[1] != L:
                context = F.interpolate(
                    context.transpose(1, 2), size=L, mode="linear", align_corners=False
                ).transpose(1, 2)
            x = torch.cat([x_t, context, x_1], dim=2)
        else:
            x = torch.cat([x_t, x_1], dim=2)

        # [B, L, C] -> [B, C, L] for Conv1d
        x = x.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)  # [B, 1, L]

        original_seq_len = x.shape[2]

        # Down path
        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            if mask_down.shape[2] != x.shape[2]:
                mask_down = F.interpolate(mask_down, size=x.shape[2], mode="nearest")
            x = resnet1(x, mask_down, t_emb)
            x = resnet2(x, mask_down, t_emb)
            x = attn(x, mask_down)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(F.interpolate(mask_down, size=x.shape[2], mode="nearest"))

        # Bottleneck
        mask_mid = masks[-1]
        if mask_mid.shape[2] != x.shape[2]:
            mask_mid = F.interpolate(mask_mid, size=x.shape[2], mode="nearest")
        x = self.mid_block1(x, mask_mid, t_emb)
        x = self.mid_attn(x, mask_mid)
        x = self.mid_block2(x, mask_mid, t_emb)

        # Up path
        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            skip = hiddens.pop()
            if x.shape[2] != skip.shape[2]:
                x = F.interpolate(x, size=skip.shape[2], mode="nearest")
            x = torch.cat((x, skip), dim=1)
            if mask_up.shape[2] != x.shape[2]:
                mask_up = F.interpolate(mask_up, size=x.shape[2], mode="nearest")
            x = resnet1(x, mask_up, t_emb)
            x = resnet2(x, mask_up, t_emb)
            x = attn(x, mask_up)
            x = upsample(x * mask_up)

        # Output
        final_mask = masks[0]
        if final_mask.shape[2] != x.shape[2]:
            final_mask = F.interpolate(final_mask, size=x.shape[2], mode="nearest")
        x = self.final_block(x, final_mask)
        x = self.final_conv(x * final_mask)

        # Restore original sequence length if convolutions changed it
        if x.shape[2] != original_seq_len:
            x = F.interpolate(
                x, size=original_seq_len, mode="linear", align_corners=False
            )

        # [B, pred_dim, L] -> [B, L, pred_dim]
        x_0_pred = x.permute(0, 2, 1)

        if is_2d_input:
            x_0_pred = x_0_pred.squeeze(1)

        return x_0_pred
