import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from .context_encoder import ContextEncoder
from .noise_scheduler import NoiseScheduler
from .unet import BridgeUNet

# --------------------------------------------------------------------------- #
# BridgeSDE                                                                    #
# --------------------------------------------------------------------------- #


class BridgeSDE(nn.Module):
    """
    Schrödinger Bridge SDE/ODE model for time series refinement.

    Owns the NoiseScheduler (bridge math) and BridgeUNet (score estimator).
    Called internally by RefineBridge — not intended for direct use.

    Original name: TimeSeriesBridge
    """

    def __init__(
        self,
        pred_dim=1,
        context_dim=None,
        seq_len=21,
        hidden_dim=63,
        dim_mults=(1, 2),
        beta_min=0.1,
        beta_max=20.0,
        pe_scale=1000,
        schedule_type="linear",
        predictor="x0",
        offset=1e-5,
        sampling_temp=2.0,
        **kwargs,
    ):
        super().__init__()

        self.pred_dim = pred_dim
        self.context_dim = context_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pe_scale = pe_scale
        self.predictor = predictor
        self.offset = offset
        self.sampling_temp = sampling_temp

        self.noise_scheduler = NoiseScheduler(
            num_timesteps=1000,
            beta_min=beta_min,
            beta_max=beta_max,
            schedule_type=schedule_type,
            temperature=sampling_temp,
        )
        self.noise_scheduler.predictor = predictor

        self.estimator = BridgeUNet(
            dim=hidden_dim,
            dim_mults=dim_mults,
            context_dim=context_dim,
            pred_dim=pred_dim,
            pe_scale=pe_scale,
        )

    # ---------------------------------------------------------------------- #
    # Training                                                                #
    # ---------------------------------------------------------------------- #

    def forward_diffusion(self, x_0, x_1, t, noise=None):
        """Delegates to NoiseScheduler.forward_diffusion."""
        return self.noise_scheduler.forward_diffusion(x_0, x_1, t, noise=noise)

    def loss_t(self, x_0, mask, x_1, t, context=None):
        """Compute denoising loss at a specific timestep t."""
        x_t, target, weight = self.forward_diffusion(x_0, x_1, t)
        pred = self.estimator(x_t, mask, x_1, t, context)

        if weight is not None and not torch.all(weight == 1.0):
            loss = torch.mean((pred - target) ** 2 * weight)
        else:
            loss = F.mse_loss(pred, target)

        return loss, x_t

    def compute_loss(self, x_0, mask, x_1, context=None):
        """
        Compute loss with uniformly sampled random timesteps.

        Args:
            x_0:     Ground truth [B, seq_len, pred_dim]
            mask:    Mask tensor [B, seq_len, 1]
            x_1:     Foundation model prediction [B, seq_len, pred_dim]
            context: Encoded context [B, seq_len, context_dim]

        Returns:
            loss: Scalar loss value
            x_t:  Noisy sample used in this step
        """
        t = torch.rand(x_0.shape[0], device=x_0.device)
        t = torch.clamp(t, self.offset, 1.0 - self.offset)
        return self.loss_t(x_0, mask, x_1, t, context)

    # ---------------------------------------------------------------------- #
    # Inference helpers                                                        #
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def data_estimation(self, x_t, mask, x_1, t, context=None):
        """Predict x_0 from x_t using the U-Net estimator."""
        return self.estimator(x_t, mask, x_1, t, context)

    @torch.no_grad()
    def bridge_sde_update(
        self, x_s, mask, x_1, context, s_idx, t_idx, temperature=None
    ):
        """
        First-order bridge SDE update (equation 19 from the paper).

        Args:
            x_s:     Current state [B, seq_len, pred_dim]
            mask:    Mask tensor
            x_1:     Foundation model prediction [B, seq_len, pred_dim]
            context: Encoded context
            s_idx:   Current timestep index
            t_idx:   Next timestep index (lower, moving toward 0)

        Returns:
            x_t: Updated state [B, seq_len, pred_dim]
        """
        temp = temperature if temperature is not None else self.sampling_temp
        device = x_s.device
        batch_size = x_s.shape[0]

        t_s = torch.tensor(
            [s_idx / (self.noise_scheduler.num_timesteps - 1)], device=device
        ).expand(batch_size)

        ns = self.noise_scheduler
        alpha_s = ns.alpha_t.to(device)[s_idx].view(-1, 1)
        sigma2_s = ns.sigma2_t.to(device)[s_idx].view(-1, 1)
        alpha_t = ns.alpha_t.to(device)[t_idx].view(-1, 1)
        sigma2_t = ns.sigma2_t.to(device)[t_idx].view(-1, 1)

        x_0_pred = self.data_estimation(x_s, mask, x_1, t_s, context)

        coeff_x = alpha_t * sigma2_t / (alpha_s * sigma2_s)
        coeff_x0 = alpha_t * (1 - sigma2_t / sigma2_s)

        x_t_det = coeff_x * x_s + coeff_x0 * x_0_pred

        if t_idx > 0:
            noise = torch.randn_like(x_s) / temp
            stoch_coeff = (
                alpha_t * torch.sqrt(sigma2_t) * torch.sqrt(1 - sigma2_t / sigma2_s)
            )
            x_t = x_t_det + stoch_coeff * noise
        else:
            x_t = x_t_det

        return x_t * mask

    @torch.no_grad()
    def bridge_ode_update(
        self, x_s, mask, x_1, context, s_idx, t_idx, temperature=None
    ):
        """
        First-order bridge ODE update (equation 20 from the paper).

        Args:
            x_s:     Current state [B, seq_len, pred_dim]
            mask:    Mask tensor
            x_1:     Foundation model prediction [B, seq_len, pred_dim]
            context: Encoded context
            s_idx:   Current timestep index
            t_idx:   Next timestep index (lower, moving toward 0)

        Returns:
            x_t: Updated state [B, seq_len, pred_dim]
        """
        device = x_s.device
        batch_size = x_s.shape[0]
        eps = 1e-8

        t_s = torch.tensor(
            [s_idx / (self.noise_scheduler.num_timesteps - 1)], device=device
        ).expand(batch_size)

        ns = self.noise_scheduler
        alpha_s = ns.alpha_t.to(device)[s_idx].view(-1, 1)
        sigma_s = ns.sigma_t.to(device)[s_idx].view(-1, 1) + eps
        sigma_bar_s = ns.sigma_bar_t.to(device)[s_idx].view(-1, 1) + eps
        alpha_t = ns.alpha_t.to(device)[t_idx].view(-1, 1)
        sigma_t = ns.sigma_t.to(device)[t_idx].view(-1, 1)
        sigma_bar_t = ns.sigma_bar_t.to(device)[t_idx].view(-1, 1)

        if s_idx == 0:
            return x_s * mask

        if t_idx == 0:
            x_0_pred = self.data_estimation(x_s, mask, x_1, t_s, context)
            alpha_s = ns.alpha_t.to(device)[s_idx].view(-1, 1)
            alpha_0 = ns.alpha_t.to(device)[0].view(-1, 1)
            return (alpha_0 / alpha_s) * x_0_pred * mask

        x_0_pred = self.data_estimation(x_s, mask, x_1, t_s, context)

        # First term coefficient
        denom1 = alpha_s * sigma_s * sigma_bar_s
        coeff_x = (
            (alpha_t * sigma_t * sigma_bar_t) / denom1
            if torch.abs(denom1).min() >= eps
            else torch.ones_like(denom1)
        )

        # Second term coefficient
        if self.noise_scheduler.sigma2_1 < eps:
            coeff_pred = torch.zeros_like(alpha_t)
        else:
            term2_inner = sigma_bar_t**2 - (sigma_bar_s * sigma_t * sigma_bar_t) / (
                sigma_s + eps
            )
            coeff_pred = (
                alpha_t / self.noise_scheduler.sigma2_1.to(device)
            ) * term2_inner

        # Third term coefficient
        alpha_1 = ns.alpha_t[-1].to(device).item() + eps
        term3_inner = sigma_t**2 - (sigma_s * sigma_t * sigma_bar_t) / (
            sigma_bar_s + eps
        )
        coeff_x1 = term3_inner / alpha_1

        x_t = coeff_x * x_s + coeff_pred * x_0_pred + coeff_x1 * x_1

        # Clamp for numerical stability
        input_scale = torch.max(x_s.abs().max(), x_1.abs().max())
        clamp_range = max(50.0, input_scale.item() * 5.0)
        x_t = torch.clamp(x_t, -clamp_range, clamp_range)

        return x_t * mask

    @torch.no_grad()
    def predictor_corrector_update(
        self, x_s, mask, x_1, context, s_idx, t_idx, use_sde=True, temperature=None
    ):
        """
        Predictor-Corrector method for second-order accuracy.

        Args:
            x_s:       Current state [B, seq_len, pred_dim]
            mask:      Mask tensor
            x_1:       Foundation model prediction [B, seq_len, pred_dim]
            context:   Encoded context
            s_idx:     Current timestep index
            t_idx:     Next timestep index
            use_sde:   If True use SDE predictor, else ODE predictor

        Returns:
            x_t: Updated state [B, seq_len, pred_dim]
        """
        # Predictor step
        if use_sde:
            x_t_pred = self.bridge_sde_update(
                x_s, mask, x_1, context, s_idx, t_idx, temperature
            )
        else:
            x_t_pred = self.bridge_ode_update(
                x_s, mask, x_1, context, s_idx, t_idx, temperature
            )

        device = x_s.device
        batch_size = x_s.shape[0]

        t_t = torch.tensor(
            [t_idx / (self.noise_scheduler.num_timesteps - 1)], device=device
        ).expand(batch_size)
        t_s = torch.tensor(
            [s_idx / (self.noise_scheduler.num_timesteps - 1)], device=device
        ).expand(batch_size)

        # Average x_0 predictions from both ends (midpoint correction)
        x_0_t = self.data_estimation(x_t_pred, mask, x_1, t_t, context)
        x_0_s = self.data_estimation(x_s, mask, x_1, t_s, context)
        x_0_avg = (x_0_s + x_0_t) / 2

        ns = self.noise_scheduler

        if use_sde:
            alpha_s = ns.alpha_t.to(device)[s_idx].view(-1, 1)
            sigma2_s = ns.sigma2_t.to(device)[s_idx].view(-1, 1)
            alpha_t = ns.alpha_t.to(device)[t_idx].view(-1, 1)
            sigma2_t = ns.sigma2_t.to(device)[t_idx].view(-1, 1)

            coeff_x = alpha_t * sigma2_t / (alpha_s * sigma2_s)
            coeff_x0 = alpha_t * (1 - sigma2_t / sigma2_s)
            x_t_det = coeff_x * x_s + coeff_x0 * x_0_avg

            if t_idx > 0:
                noise = torch.randn_like(x_s) / self.sampling_temp
                stoch_coeff = (
                    alpha_t * torch.sqrt(sigma2_t) * torch.sqrt(1 - sigma2_t / sigma2_s)
                )
                x_t = x_t_det + stoch_coeff * noise
            else:
                x_t = x_t_det

            x_t = x_t * mask

        else:
            alpha_s = ns.alpha_t.to(device)[s_idx].view(-1, 1)
            alpha_t = ns.alpha_t.to(device)[t_idx].view(-1, 1)
            sigma_s = ns.sigma_t.to(device)[s_idx].view(-1, 1)
            sigma_bar_s = ns.sigma_bar_t.to(device)[s_idx].view(-1, 1)
            sigma_t = ns.sigma_t.to(device)[t_idx].view(-1, 1)
            sigma_bar_t = ns.sigma_bar_t.to(device)[t_idx].view(-1, 1)

            coeff_x = (alpha_t * sigma_t * sigma_bar_t) / (
                alpha_s * sigma_s * sigma_bar_s
            )
            coeff_pred = (alpha_t / self.noise_scheduler.sigma2_1) * (
                sigma_bar_t**2 - (sigma_bar_s * sigma_t * sigma_bar_t) / sigma_s
            )
            coeff_x1 = (
                sigma_t**2 - (sigma_s * sigma_t * sigma_bar_t) / sigma_bar_s
            ) * (x_1 / self.noise_scheduler.alpha_t[-1])

            x_t = coeff_x * x_s + coeff_pred * x_0_avg + coeff_x1
            x_t = x_t * mask

        return x_t

    @torch.no_grad()
    def reverse_diffusion(
        self,
        x_1,
        mask,
        n_timesteps,
        sampler="ode",
        context=None,
        verbose=False,
        temperature=None,
    ):
        """
        Run the full reverse diffusion: x_1 → x_0.

        Args:
            x_1:         Foundation model prediction [B, seq_len, pred_dim]
            mask:        Mask tensor
            n_timesteps: Number of sampling steps
            sampler:     "sde", "ode", "pc_sde", "pc_ode"
            context:     Encoded context
            verbose:     Show tqdm progress bar
            temperature: Sampling temperature

        Returns:
            xt_traj: Full trajectory [B, n_timesteps+1, seq_len, pred_dim]
        """
        temp = temperature if temperature is not None else self.sampling_temp
        device = x_1.device

        if mask is not None:
            mask = mask.to(device)

        xt = x_1 * mask
        orig_shape = xt.shape
        is_3d = len(orig_shape) == 3
        xt_traj = [xt.detach().clone()]

        step_indices = (
            torch.linspace(self.noise_scheduler.num_timesteps - 1, 0, n_timesteps + 1)
            .round()
            .long()
        )
        step_pairs = list(zip(step_indices[:-1], step_indices[1:]))

        if verbose:
            step_pairs = tqdm(step_pairs, desc="Sampling")

        for s_idx, t_idx in step_pairs:
            s, t = s_idx.item(), t_idx.item()

            if sampler == "sde":
                xt = self.bridge_sde_update(xt, mask, x_1, context, s, t, temp)
            elif sampler == "ode":
                xt = self.bridge_ode_update(xt, mask, x_1, context, s, t, temp)
            elif sampler == "pc_sde":
                xt = self.predictor_corrector_update(
                    xt, mask, x_1, context, s, t, use_sde=True, temperature=temp
                )
            elif sampler == "pc_ode":
                xt = self.predictor_corrector_update(
                    xt, mask, x_1, context, s, t, use_sde=False, temperature=temperature
                )
            else:
                xt = self.bridge_ode_update(xt, mask, x_1, context, s, t, temperature)

            # Normalise shape relative to the first trajectory element
            if len(xt.shape) != len(orig_shape):
                if len(xt.shape) == 3 and len(orig_shape) == 2:
                    xt = xt[:, -1]
                elif len(xt.shape) == 2 and len(orig_shape) == 3:
                    xt = xt.unsqueeze(1)

            xt_traj.append(xt.detach().clone())

        # Normalise trajectory shapes before stacking
        first_shape = xt_traj[0].shape
        last_shape = xt_traj[-1].shape
        if len(first_shape) != len(last_shape):
            for i in range(len(xt_traj)):
                if len(xt_traj[i].shape) == 2 and len(last_shape) == 3:
                    xt_traj[i] = xt_traj[i].unsqueeze(1).expand(-1, last_shape[1], -1)
                elif len(xt_traj[i].shape) == 3 and len(last_shape) == 2:
                    xt_traj[i] = xt_traj[i][:, -1]

        return torch.stack(xt_traj, dim=1)

    @torch.no_grad()
    def forward(
        self,
        x_1,
        mask,
        n_timesteps,
        sampler="ode",
        context=None,
        verbose=False,
        temperature=None,
    ):
        """
        Inference wrapper. Maps public sampler names to internal ones
        then delegates to reverse_diffusion.
        """
        sampler_map = {
            "sde": "sde",
            "ode": "ode",
            "pf_ode_euler": "ode",
            "sde_euler": "sde",
            "pc_sde": "pc_sde",
            "pc_ode": "pc_ode",
        }
        return self.reverse_diffusion(
            x_1,
            mask,
            n_timesteps,
            sampler_map.get(sampler, "ode"),
            context,
            verbose,
            temperature,
        )


# --------------------------------------------------------------------------- #
# RefineBridge                                                                 #
# --------------------------------------------------------------------------- #


class RefineBridge(nn.Module):
    """
    Top-level RefineBridge model.

    Combines ContextEncoder and BridgeSDE. This is the only class
    that external code (training, evaluation, inference) should interact with.

    Original name: TimeSeriesTTS
    """

    def __init__(
        self,
        context_dim=1,
        pred_dim=1,
        context_seq_len=252,
        pred_seq_len=21,
        hidden_dim=96,
        dim_mults=(1, 2, 4),
        beta_min=0.1,
        beta_max=20.0,
        pe_scale=1000,
        schedule_type="linear",
        predictor="x0",
        **kwargs,
    ):
        super().__init__()

        self.context_dim = context_dim
        self.pred_dim = pred_dim
        self.context_seq_len = context_seq_len
        self.pred_seq_len = pred_seq_len
        self.hidden_dim = hidden_dim

        encoder_output_dim = hidden_dim

        self.context_encoder = ContextEncoder(
            context_dim=context_dim,
            context_seq_len=context_seq_len,
            pred_seq_len=pred_seq_len,
            hidden_dim=hidden_dim,
            output_dim=encoder_output_dim,
            kernel_size=25,
            individual=False,
            dropout=0.0,
        )

        self.decoder = BridgeSDE(
            context_dim=encoder_output_dim,
            pred_dim=pred_dim,
            seq_len=pred_seq_len,
            hidden_dim=hidden_dim,
            dim_mults=dim_mults,
            beta_min=beta_min,
            beta_max=beta_max,
            pe_scale=pe_scale,
            schedule_type=schedule_type,
            predictor=predictor,
        )

    @property
    def nparams(self):
        """Total number of trainable parameters."""
        return sum(
            np.prod(p.detach().cpu().numpy().shape)
            for p in self.parameters()
            if p.requires_grad
        )

    def compute_loss(self, context, foundation_pred, ground_truth, mask=None):
        """
        Compute training loss.

        Args:
            context:         Historical context [B, context_seq_len, context_dim]
            foundation_pred: Foundation model forecast [B, pred_seq_len, pred_dim]
            ground_truth:    True future values [B, pred_seq_len, pred_dim]
            mask:            Optional mask [B, 1] or [B, pred_seq_len, 1]

        Returns:
            encoder_loss: 0.0 (kept for API compatibility)
            bridge_loss:  Scalar denoising MSE loss
        """
        device = next(self.parameters()).device
        context = context.to(device)
        foundation_pred = foundation_pred.to(device)
        ground_truth = ground_truth.to(device)

        if mask is None:
            mask = torch.ones(context.shape[0], 1, device=device)
        else:
            mask = mask.to(device)

        encoded_context = self.context_encoder(context)

        if mask.dim() == 2:
            mask = mask.unsqueeze(2).expand(-1, ground_truth.shape[1], -1)

        bridge_loss, _ = self.decoder.compute_loss(
            x_0=ground_truth,
            mask=mask,
            x_1=foundation_pred,
            context=encoded_context,
        )

        return 0.0, bridge_loss

    @torch.no_grad()
    def forward(
        self,
        context,
        foundation_pred,
        mask=None,
        n_timesteps=50,
        sampler="ode",
        temperature=1.0,
    ):
        """
        Run inference: refine foundation_pred using the learned bridge.

        Args:
            context:         Historical context [B, context_seq_len, context_dim]
            foundation_pred: Foundation model forecast [B, pred_seq_len, pred_dim]
            mask:            Optional mask [B, 1] or [B, pred_seq_len, 1]
            n_timesteps:     Number of reverse diffusion steps
            sampler:         "ode", "sde", "pc_ode", "pc_sde"
            temperature:     Sampling temperature

        Returns:
            foundation_pred: Original foundation prediction (unchanged)
            refined_pred:    Refined trajectory [B, n_timesteps+1, seq_len, pred_dim]
            loss_metrics:    Empty dict (no loss computed at inference)
        """
        device = next(self.parameters()).device
        context = context.to(device)
        foundation_pred = foundation_pred.to(device)

        if mask is None:
            mask = torch.ones(foundation_pred.shape[0], 1, device=device)
        else:
            mask = mask.to(device)

        encoded_context = self.context_encoder(context)

        if mask.dim() == 2:
            mask = mask.unsqueeze(2).expand(-1, foundation_pred.shape[1], -1)

        refined_pred = self.decoder.forward(
            x_1=foundation_pred,
            mask=mask,
            context=encoded_context,
            n_timesteps=n_timesteps,
            sampler=sampler,
            temperature=temperature,
        )

        return foundation_pred, refined_pred, {}
