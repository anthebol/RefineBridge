"""
tsfm.py — Time Series Foundation Model inference wrappers for RefineBridge.

Three forecasters are provided out of the box:
    ChronosForecaster    amazon/chronos-t5-large  (and other Chronos variants)
    MoiraiForecaster     Salesforce/moirai-1.0-R-large  (and other Moirai variants)
    TimeMoEForecaster    Maple728/TimeMoE-50M  (and other TimeMoE variants)

All three implement the same BaseForecaster interface:
    predict_with_quantiles(context, horizon) -> (median, quantiles)

To integrate a different TSFM:
    1. Subclass BaseForecaster
    2. Implement predict_with_quantiles(context, horizon) -> (median, quantiles)
    3. Pass your forecaster instance to generate_dataset.py
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Base class                                                                   #
# --------------------------------------------------------------------------- #


class BaseForecaster:
    """
    Base class for foundation model forecasters.

    Subclass this to plug any TSFM into the RefineBridge data generation
    pipeline. The only method you need to implement is predict_with_quantiles.
    """

    def predict_with_quantiles(self, context: np.ndarray, horizon: int):
        """
        Run the foundation model on a context window and return forecasts.

        Args:
            context:  Historical values in raw (price) scale, shape (context_len,)
            horizon:  Number of future steps to forecast

        Returns:
            median:    Point forecast — median prediction, shape (horizon,)
            quantiles: Dict mapping quantile string to array,
                       e.g. {"0.1": array([...]), "0.9": array([...])}
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Chronos                                                                      #
# --------------------------------------------------------------------------- #


class ChronosForecaster(BaseForecaster):
    """
    Wrapper around Amazon Chronos for RefineBridge dataset generation.

    Chronos returns a set of sample trajectories. We compute quantiles across
    those samples and use the median (0.5) as the point prediction x_1 that
    RefineBridge learns to refine toward the ground truth x_0.

    Per-window StandardScaler normalisation is applied before passing context
    to Chronos, then inverse-transformed back to the original scale. This
    matches the data generation procedure used in the ICASSP 2026 paper.

    Install:
        pip install git+https://github.com/amazon-science/chronos-forecasting.git
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-large",
        device: str = "cuda",
        torch_dtype=torch.float32,
        num_samples: int = 100,
    ):
        """
        Args:
            model_name:   HuggingFace model identifier for Chronos
            device:       "cuda", "cpu", or "mps"
            torch_dtype:  Tensor dtype passed to ChronosPipeline
            num_samples:  Number of Monte-Carlo samples for quantile estimation
        """
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos-forecasting is not installed.\n"
                "Run: pip install git+https://github.com/amazon-science/"
                "chronos-forecasting.git"
            )

        print(f"Loading Chronos [{model_name}] on {device} ...")
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch_dtype,
        )
        self.device = device
        self.num_samples = num_samples

    def _run_inference(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Scales context, runs Chronos, inverse-scales samples.
        Returns:
            samples: shape (num_samples, horizon) in original scale
        """
        scaler = StandardScaler()
        context_scaled = scaler.fit_transform(context.reshape(-1, 1)).flatten()
        context_tensor = torch.tensor(context_scaled, dtype=torch.float32)

        with torch.no_grad():
            forecast = self.pipeline.predict(
                context_tensor.unsqueeze(0),
                horizon,
                num_samples=self.num_samples,
            )

        samples = forecast[0].cpu().numpy()  # (num_samples, horizon)
        samples_unscaled = np.array(
            [scaler.inverse_transform(s.reshape(-1, 1)).flatten() for s in samples]
        )
        return samples_unscaled

    def predict_with_quantiles(self, context: np.ndarray, horizon: int):
        samples = self._run_inference(context, horizon)
        median = np.median(samples, axis=0)
        quantiles = {
            str(q): np.quantile(samples, q, axis=0) for q in self.QUANTILE_LEVELS
        }
        return median, quantiles


# --------------------------------------------------------------------------- #
# Moirai                                                                       #
# --------------------------------------------------------------------------- #


class MoiraiForecaster(BaseForecaster):
    """
    Wrapper around Salesforce Moirai for RefineBridge dataset generation.

    Moirai operates in raw (unnormalised) price space — the context is
    passed directly without any manual scaling. Internally Moirai applies
    its own normalisation. The median across num_samples trajectories is
    used as the point prediction x_1.

    Inference is routed through GluonTS PandasDataset, which is the
    standard interface Moirai exposes.

    Install:
        pip install uni2ts
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        model_name: str = "Salesforce/moirai-1.0-R-large",
        device: str = "cuda",
        num_samples: int = 100,
        patch_size: str = "auto",
    ):
        """
        Args:
            model_name:   HuggingFace identifier for a Moirai model
            device:       "cuda", "cpu", or "mps"
            num_samples:  Number of Monte-Carlo trajectories for quantiles
            patch_size:   Moirai patch size — "auto" works for most cases
        """
        try:
            from uni2ts.model.moirai import MoiraiModule
        except ImportError:
            raise ImportError("uni2ts is not installed.\n" "Run: pip install uni2ts")

        print(f"Loading Moirai [{model_name}] on {device} ...")
        self.module = MoiraiModule.from_pretrained(model_name).to(device)
        self.module.eval()
        self.device = device
        self.num_samples = num_samples
        self.patch_size = patch_size

    def _build_predictor(self, context_len: int, horizon: int):
        """Build a fresh MoiraiForecast predictor for the given lengths."""
        from uni2ts.model.moirai import MoiraiForecast

        model = MoiraiForecast(
            module=self.module,
            prediction_length=horizon,
            context_length=context_len,
            patch_size=self.patch_size,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        return model.create_predictor(batch_size=1)

    def _run_inference(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Runs Moirai on a raw price context.
        Returns:
            samples: shape (num_samples, horizon) in original scale
        """
        from gluonts.dataset.pandas import PandasDataset

        # Build a daily DatetimeIndex for the context window
        dates = pd.date_range(end="2024-01-01", periods=len(context), freq="D")
        context_series = pd.Series(context.astype(float), index=dates)
        ds = PandasDataset({"target": context_series})

        predictor = self._build_predictor(len(context), horizon)

        with torch.no_grad():
            forecasts = list(predictor.predict(ds))

        if len(forecasts) == 0:
            # Fallback: repeat last value
            return np.tile(context[-1], (self.num_samples, horizon))

        samples = forecasts[0].samples  # (num_samples, horizon)
        return np.array(samples)

    def predict_with_quantiles(self, context: np.ndarray, horizon: int):
        samples = self._run_inference(context, horizon)
        median = np.median(samples, axis=0)
        quantiles = {
            str(q): np.quantile(samples, q, axis=0) for q in self.QUANTILE_LEVELS
        }
        return median, quantiles


# --------------------------------------------------------------------------- #
# TimeMoE                                                                      #
# --------------------------------------------------------------------------- #


class TimeMoEForecaster(BaseForecaster):
    """
    Wrapper around TimeMoE for RefineBridge dataset generation.

    TimeMoE is a causal language model for time series. Inference follows
    the standard HuggingFace generate() pattern:
        1. Normalise context with StandardScaler
        2. Feed as a 1D token sequence to model.generate()
        3. Extract the last `horizon` tokens as the forecast
        4. Inverse-scale back to original price space

    Because TimeMoE is autoregressive and deterministic by default, a
    small Gaussian perturbation is added to produce num_samples trajectories
    for quantile estimation. Set num_samples=1 for pure point forecasts.

    Install:
        pip install git+https://github.com/Time-MoE/Time-MoE.git
        # or:
        pip install transformers  (trust_remote_code=True is required)
    """

    QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        model_name: str = "Maple728/TimeMoE-50M",
        device: str = "cuda",
        torch_dtype=torch.float32,
        num_samples: int = 100,
        noise_std: float = 0.01,
    ):
        """
        Args:
            model_name:   HuggingFace identifier for a TimeMoE model
                          Options: "Maple728/TimeMoE-50M"
                                   "Maple728/TimeMoE-200M-Dense"
                                   "Maple728/TimeMoE-200M"  (MoE variant)
            device:       "cuda", "cpu", or "mps"
            torch_dtype:  Tensor dtype for model weights
            num_samples:  Number of perturbed trajectories for quantile estimation
            noise_std:    Std of Gaussian noise added in normalised space for sampling
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "transformers is not installed.\n" "Run: pip install transformers"
            )

        print(f"Loading TimeMoE [{model_name}] on {device} ...")
        from transformers import AutoModelForCausalLM

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        self.model.eval()

        self.device = device
        self.torch_dtype = torch_dtype
        self.num_samples = num_samples
        self.noise_std = noise_std

    def _run_single(
        self,
        context_scaled: np.ndarray,
        horizon: int,
        scaler: StandardScaler,
        noise: float = 0.0,
    ) -> np.ndarray:
        """
        Run one forward pass, optionally with Gaussian input perturbation.
        Returns a forecast in original scale, shape (horizon,).
        """
        ctx = context_scaled.copy()
        if noise > 0:
            ctx = ctx + np.random.randn(*ctx.shape) * noise

        ctx_tensor = (
            torch.tensor(ctx, dtype=self.torch_dtype).unsqueeze(0).to(self.device)
        )  # (1, context_len)

        with torch.no_grad():
            output = self.model.generate(ctx_tensor, max_new_tokens=horizon)

        # output shape: (1, context_len + horizon) — take the generated suffix
        pred_scaled = output[0, -horizon:].cpu().float().numpy()  # (horizon,)
        pred_raw = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        return pred_raw

    def _run_inference(self, context: np.ndarray, horizon: int) -> np.ndarray:
        """
        Runs TimeMoE num_samples times (with noise) on a raw price context.
        Returns:
            samples: shape (num_samples, horizon) in original scale
        """
        scaler = StandardScaler()
        context_scaled = scaler.fit_transform(context.reshape(-1, 1)).flatten()

        samples = np.zeros((self.num_samples, horizon))
        for s in range(self.num_samples):
            noise = self.noise_std if s > 0 else 0.0  # first sample is clean
            samples[s] = self._run_single(context_scaled, horizon, scaler, noise)

        return samples

    def predict_with_quantiles(self, context: np.ndarray, horizon: int):
        samples = self._run_inference(context, horizon)
        median = np.median(samples, axis=0)
        quantiles = {
            str(q): np.quantile(samples, q, axis=0) for q in self.QUANTILE_LEVELS
        }
        return median, quantiles
