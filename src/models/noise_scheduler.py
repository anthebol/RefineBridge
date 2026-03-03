import torch


class NoiseScheduler:
    """
    Precomputes all noise schedule coefficients at construction time.

    All time-dependent tensors (alpha_t, sigma2_t, etc.) are stored on CPU
    and moved to the target device on-demand inside each method via .to(device).

    Note: self.predictor is set externally by BridgeSDE after construction.
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_min=1e-5,
        beta_max=0.01,
        schedule_type="gmax",
        temperature=2.0,
    ):
        """
        Args:
            num_timesteps: Number of discrete diffusion steps
            beta_min:      Starting noise level (β₀)
            beta_max:      Ending noise level (β₁)
            schedule_type: One of "gmax", "vp", "sb", "constant"
            temperature:   Sampling temperature (stored for use by BridgeSDE)
        """
        self.num_timesteps = num_timesteps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.schedule_type = schedule_type
        self.sampling_temp = temperature

        # Time grid from 0 to 1
        self.time_grid = torch.linspace(0, 1, num_timesteps)

        # ------------------------------------------------------------------ #
        # Build schedule-specific f(t) and g²(t), then integrate to get      #
        # α_t and σ²_t following the Bridge-TTS paper (Table 1).             #
        # ------------------------------------------------------------------ #

        if schedule_type == "gmax":
            # f(t) = 0  →  α_t = 1
            # g²(t) = β₀ + t(β₁ - β₀)  (linear)
            # σ²_t  = ∫₀ᵗ g²(τ) dτ  (Euler, since α_t = 1)
            self.f_t = torch.zeros_like(self.time_grid)
            self.g2_t = beta_min + self.time_grid * (beta_max - beta_min)
            self.alpha_t = torch.ones_like(self.time_grid)
            self.sigma2_t = torch.zeros_like(self.time_grid)
            for i in range(1, num_timesteps):
                delta_t = self.time_grid[i] - self.time_grid[i - 1]
                self.sigma2_t[i] = self.sigma2_t[i - 1] + self.g2_t[i - 1] * delta_t

        elif schedule_type == "vp":
            # f(t) = -0.5 * g²(t)
            # g²(t) = β₀ + t(β₁ - β₀)
            # α_t   = exp(∫₀ᵗ f(τ) dτ)
            # σ²_t  = ∫₀ᵗ g²(τ)/α²_τ dτ
            self.f_t = -0.5 * (beta_min + self.time_grid * (beta_max - beta_min))
            self.g2_t = beta_min + self.time_grid * (beta_max - beta_min)

            cumulative_drift = torch.zeros_like(self.time_grid)
            for i in range(1, num_timesteps):
                delta_t = self.time_grid[i] - self.time_grid[i - 1]
                cumulative_drift[i] = (
                    cumulative_drift[i - 1] + self.f_t[i - 1] * delta_t
                )
            self.alpha_t = torch.exp(cumulative_drift)

            self.sigma2_t = torch.zeros_like(self.time_grid)
            for i in range(1, num_timesteps):
                delta_t = self.time_grid[i] - self.time_grid[i - 1]
                self.sigma2_t[i] = (
                    self.sigma2_t[i - 1]
                    + (self.g2_t[i - 1] / (self.alpha_t[i - 1] ** 2)) * delta_t
                )

        elif schedule_type == "sb":
            # f(t) = 0  →  α_t = 1
            # g²(t) ramps linearly up to g2_max_t=0.5 then ramps back down
            # σ²_t computed analytically per segment
            self.g2_min = beta_min
            self.g2_max = beta_max
            self.g2_max_t = 0.5

            self.f_t = torch.zeros_like(self.time_grid)
            self.g2_t = torch.zeros_like(self.time_grid)

            k1 = (self.g2_max - self.g2_min) / self.g2_max_t
            k2 = (self.g2_max - self.g2_min) / (1 - self.g2_max_t)

            for i, t in enumerate(self.time_grid):
                if t <= self.g2_max_t:
                    self.g2_t[i] = self.g2_min + t * k1
                else:
                    self.g2_t[i] = self.g2_max - (t - self.g2_max_t) * k2

            self.alpha_t = torch.ones_like(self.time_grid)
            self.sigma2_t = torch.zeros_like(self.time_grid)
            for i in range(1, num_timesteps):
                t = self.time_grid[i].item()
                if t <= self.g2_max_t:
                    self.sigma2_t[i] = self.g2_min * t + k1 * t**2 / 2
                else:
                    integral_up_to_max_t = (
                        self.g2_min * self.g2_max_t + k1 * self.g2_max_t**2 / 2
                    )
                    integral_max_t_to_t = (
                        self.g2_max * (t - self.g2_max_t)
                        - k2 * ((t - self.g2_max_t) ** 2) / 2
                    )
                    self.sigma2_t[i] = integral_up_to_max_t + integral_max_t_to_t

        elif schedule_type == "constant":
            # f(t) = 0  →  α_t = 1
            # g²(t) = β₁  (constant)
            # σ²_t  = β₁ * t
            self.f_t = torch.zeros_like(self.time_grid)
            self.g2_t = torch.ones_like(self.time_grid) * beta_max
            self.alpha_t = torch.ones_like(self.time_grid)
            self.sigma2_t = beta_max * self.time_grid

        else:
            raise ValueError(
                f"Unknown schedule_type '{schedule_type}'. "
                "Choose from: 'gmax', 'vp', 'sb', 'constant'."
            )

        # ------------------------------------------------------------------ #
        # Derived quantities shared across all schedule types                 #
        # ------------------------------------------------------------------ #

        # Total variance at t=1
        self.sigma2_1 = self.sigma2_t[-1]

        # Complementary variance: σ̄²_t = σ²_1 - σ²_t
        self.sigma2_bar_t = self.sigma2_1 - self.sigma2_t

        # Reversed drift: ᾱ_t = exp(∫ₜ¹ f(τ) dτ)
        # Computed by integrating f in reverse then flipping back
        reversed_time = torch.flip(self.time_grid, [0])
        reversed_f = torch.flip(self.f_t, [0])
        cumulative_reversed_drift = torch.zeros_like(self.time_grid)
        for i in range(1, num_timesteps):
            delta_t = reversed_time[i] - reversed_time[i - 1]
            cumulative_reversed_drift[i] = (
                cumulative_reversed_drift[i - 1] + reversed_f[i - 1] * delta_t
            )
        self.alpha_bar_t = torch.exp(cumulative_reversed_drift)
        self.alpha_bar_t = torch.flip(self.alpha_bar_t, [0])

        # Standard deviations (square roots)
        self.sigma_t = torch.sqrt(self.sigma2_t)
        self.sigma_bar_t = torch.sqrt(self.sigma2_bar_t)

        self.epsilon = 1e-8

    # ---------------------------------------------------------------------- #
    # Helper methods                                                          #
    # ---------------------------------------------------------------------- #

    def compute_gaussian_product_coef(self, sigma1, sigma2):
        """
        Coefficients for the product of two Gaussians.

        Given p₁ = N(x_t | x_0, σ₁²) and p₂ = N(x_t | x_1, σ₂²),
        their product is N(x_t | coef1·x₀ + coef2·x₁, var).
        """
        denom = sigma1**2 + sigma2**2
        coef1 = sigma2**2 / denom
        coef2 = sigma1**2 / denom
        var = (sigma1**2 * sigma2**2) / denom
        return coef1, coef2, var

    def get_index_from_time(self, t):
        """Convert continuous time t ∈ [0,1] to a discrete grid index."""
        return torch.clamp(
            (t * (self.num_timesteps - 1)).long(), 0, self.num_timesteps - 1
        )

    def get_bridge_mean(self, x_0, x_1, t):
        """
        Mean of the bridge distribution q(x_t | x_0, x_1).

        Returns:
            mean: [B, seq_len, D]
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        t_idx = self.get_index_from_time(t)

        alpha_t = self.alpha_t.to(device)[t_idx].view(batch_size, 1, 1)
        alpha_bar_t = self.alpha_bar_t.to(device)[t_idx].view(batch_size, 1, 1)
        sigma2_t = self.sigma2_t.to(device)[t_idx].view(batch_size, 1, 1)
        sigma2_bar_t = self.sigma2_bar_t.to(device)[t_idx].view(batch_size, 1, 1)

        numerator = alpha_t * sigma2_bar_t * x_0 + alpha_bar_t * sigma2_t * x_1
        return numerator / self.sigma2_1.to(device)

    def get_bridge_variance(self, t):
        """
        Variance of the bridge distribution q(x_t | x_0, x_1).

        Returns:
            variance: [B, 1, 1]
        """
        batch_size = t.shape[0]
        device = t.device
        t_idx = self.get_index_from_time(t)

        alpha_t = self.alpha_t.to(device)[t_idx].view(batch_size, 1, 1)
        sigma2_t = self.sigma2_t.to(device)[t_idx].view(batch_size, 1, 1)
        sigma2_bar_t = self.sigma2_bar_t.to(device)[t_idx].view(batch_size, 1, 1)

        return (alpha_t**2) * sigma2_t * sigma2_bar_t / self.sigma2_1.to(device)

    # ---------------------------------------------------------------------- #
    # Core diffusion methods (called by BridgeSDE)                           #
    # ---------------------------------------------------------------------- #

    def forward_diffusion(self, x_0, x_1, t, noise=None):
        """
        Sample x_t from the bridge distribution at time t.
        Used during training to corrupt the data for loss computation.

        Args:
            x_0:   Ground truth [B, seq_len, D]
            x_1:   Foundation model prediction [B, seq_len, D]
            t:     Timestep tensor [B], values in [0, 1]
            noise: Optional pre-sampled noise [B, seq_len, D]

        Returns:
            x_t:    Noisy sample at time t [B, seq_len, D]
            target: x_0 (the denoising target)
            weight: All-ones loss weight [B, seq_len, D]
        """
        batch_size, seq_len, feature_dim = x_0.shape
        device = x_0.device

        if noise is None:
            noise = torch.randn_like(x_0)

        t_idx = self.get_index_from_time(t)
        alpha_t = self.alpha_t.to(device)[t_idx].view(batch_size, 1, 1)
        alpha_bar_t = self.alpha_bar_t.to(device)[t_idx].view(batch_size, 1, 1)
        sigma2_t = self.sigma2_t.to(device)[t_idx].view(batch_size, 1, 1)
        sigma2_bar_t = self.sigma2_bar_t.to(device)[t_idx].view(batch_size, 1, 1)

        mean = (
            alpha_t * sigma2_bar_t * x_0 + alpha_bar_t * sigma2_t * x_1
        ) / self.sigma2_1.to(device)
        variance = (alpha_t**2) * sigma2_t * sigma2_bar_t / self.sigma2_1.to(device)
        std = torch.sqrt(variance + 1e-8)

        x_t = mean + std * noise
        target = x_0
        weight = torch.ones_like(x_0)

        return x_t, target, weight

    def score_estimation(self, x_t, mask, x_1, t, x_0_pred):
        """
        Estimate the score ∇log p(x_t) from the model's x_0 prediction.
        Used during inference inside the SDE/ODE update steps.

        self.predictor is set externally by BridgeSDE. Supported values:
            "x0"        — direct x_0 prediction (default)
            "hpsi"      — ε-prediction style
            "noise_psb" — combined score
            "velocity"  — velocity parameterisation

        Args:
            x_t:     Noisy sample [B, seq_len, D]
            mask:    Binary mask [B, ...]
            x_1:     Foundation model prediction [B, seq_len, D]
            t:       Timestep tensor [B]
            x_0_pred: Model's prediction of x_0 [B, seq_len, D]

        Returns:
            score: Estimated score [B, seq_len, D], masked
        """
        t_idx = self.get_index_from_time(t)
        device = x_t.device

        alpha_t = self.alpha_t.to(device)[t_idx].view(-1, 1)
        sigma_t = self.sigma_t.to(device)[t_idx].view(-1, 1)
        sigma_bar_t = self.sigma_bar_t.to(device)[t_idx].view(-1, 1)

        if self.predictor == "x0":
            score = -(x_t - alpha_t * x_0_pred) / (alpha_t * sigma_t**2 + 1e-8)

        elif self.predictor == "hpsi":
            eps_t = (x_t - alpha_t * x_0_pred) / (alpha_t * sigma_t + 1e-8)
            score = -eps_t / (sigma_t + 1e-8)

        elif self.predictor == "noise_psb":
            score_t_psb = -(x_t - alpha_t * x_0_pred) / (alpha_t * sigma_t**2 + 1e-8)
            score_t_Psi = -(x_t - x_1) / (sigma_bar_t**2 + 1e-8)
            score = score_t_psb - score_t_Psi

        elif self.predictor == "velocity":
            beta_t = self.g2_t.to(device)[t_idx].view(-1, 1)
            v_t = (x_t - alpha_t * x_0_pred) / (alpha_t * sigma_t + 1e-8)
            score = -(2 * v_t) / (beta_t + 1e-8) - (x_t - x_1) / (sigma_bar_t**2 + 1e-8)

        else:
            score = -(x_t - alpha_t * x_0_pred) / (alpha_t * sigma_t**2 + 1e-8)

        return score * mask
