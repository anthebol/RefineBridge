import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# DLinear decomposition primitives (from dlinear.py)                          #
# --------------------------------------------------------------------------- #


class moving_avg(nn.Module):
    """Moving average block to highlight the trend of time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad both ends to preserve sequence length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)


class series_decomp(nn.Module):
    """Series decomposition block: splits input into seasonal and trend."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean


# --------------------------------------------------------------------------- #
# ContextEncoder                                                               #
# --------------------------------------------------------------------------- #


class ContextEncoder(nn.Module):
    """
    Encodes a historical context window into a fixed-length feature sequence
    aligned with the prediction horizon.

    Architecture:
        1. Decompose input into seasonal + trend (DLinear)
        2. Project temporal dimension: context_seq_len → pred_seq_len
        3. Recombine and project features: context_dim → output_dim
        4. Apply residual mixing layer

    Original name: DLinearContextEncoder
    """

    def __init__(
        self,
        context_dim=1,
        context_seq_len=252,
        pred_seq_len=21,
        hidden_dim=96,
        output_dim=None,
        kernel_size=51,
        individual=False,
        dropout=0.0,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.context_seq_len = context_seq_len
        self.pred_seq_len = pred_seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else hidden_dim
        self.individual = individual

        self.decomposition = series_decomp(kernel_size)

        # Temporal projection: context_seq_len -> pred_seq_len
        if self.individual and context_dim > 1:
            self.Linear_Seasonal = nn.ModuleList(
                [nn.Linear(context_seq_len, pred_seq_len) for _ in range(context_dim)]
            )
            self.Linear_Trend = nn.ModuleList(
                [nn.Linear(context_seq_len, pred_seq_len) for _ in range(context_dim)]
            )
        else:
            self.Linear_Seasonal = nn.Linear(context_seq_len, pred_seq_len)
            self.Linear_Trend = nn.Linear(context_seq_len, pred_seq_len)

        # Feature projection: context_dim -> output_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )

        # Residual mixing layer
        self.mixing_layer = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, self.output_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, context_seq_len, context_dim]

        Returns:
            output: [B, pred_seq_len, output_dim]
        """
        # 1. Decompose
        seasonal_init, trend_init = self.decomposition(x)

        # 2. Permute for linear layers: [B, seq, dim] -> [B, dim, seq]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # 3. Temporal projection: seq_len -> pred_len
        if self.individual and self.context_dim > 1:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_seq_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_seq_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            for i in range(self.context_dim):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # 4. Combine trend and seasonal
        combined = seasonal_output + trend_output

        # 5. Permute back: [B, dim, pred_len] -> [B, pred_len, dim]
        combined = combined.permute(0, 2, 1)

        # 6. Project features to output_dim
        output = self.feature_projection(combined)

        # 7. Residual mixing
        output = output + self.mixing_layer(output)

        return output
