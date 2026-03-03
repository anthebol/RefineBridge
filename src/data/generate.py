"""
generate.py — Core dataset generation logic for RefineBridge.

This module handles:
  - Chronological train/val/test splitting (prevents data leakage)
  - Sliding window triplet generation: (context_window, prediction, ground_truth)
  - Saving splits as .npy files

The output triplets are the training signal for RefineBridge:
  - context_window:  Historical values fed to the foundation model
  - prediction:      Foundation model's median forecast — this is x_1 in the paper,
                     the imperfect "prior" that RefineBridge refines
  - ground_truth:    Actual future values — this is x_0, the refinement target
"""

import os
import time

import numpy as np
from tqdm import tqdm


def split_series(values, timestamps=None, train_ratio=0.8, val_ratio=0.1):
    """
    Splits a single time series chronologically into train, val, and test sets.

    Splitting is done by time position, not randomly, to prevent data leakage.
    The test set is always the most recent portion of the series.

    Args:
        values:      Time series values, shape (n,)
        timestamps:  Corresponding timestamps, shape (n,) — optional
        train_ratio: Fraction of data for training (default 0.8)
        val_ratio:   Fraction of data for validation (default 0.1)
                     Test gets the remaining 1 - train_ratio - val_ratio

    Returns:
        train, val, test: each a dict with keys "values" and "timestamps"
    """
    n = len(values)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    def make_split(start, end):
        return {
            "values": values[start:end],
            "timestamps": timestamps[start:end] if timestamps is not None else None,
        }

    return (
        make_split(0, train_end),
        make_split(train_end, val_end),
        make_split(val_end, n),
    )


def generate_triplets(
    values, forecaster, context_len, pred_len, step_size, timestamps=None, series_id=""
):
    """
    Generates (context_window, prediction, ground_truth) triplets from one time series split.

    A sliding window moves along the series with the given step_size.
    For each window position:
      - context_window: the previous context_len values — fed to the foundation model
      - prediction:     the foundation model's median forecast for the next pred_len steps
      - ground_truth:   the actual next pred_len values

    The prediction is the "prior" x_1 in the Schrödinger Bridge framework.
    RefineBridge learns to transport x_1 → x_0 (ground_truth).

    Args:
        values:      Time series values for this split, shape (n,)
        forecaster:  A BaseForecaster instance (e.g. ChronosForecaster)
        context_len: Number of historical steps fed to the foundation model
        pred_len:    Forecast horizon H (number of steps to predict)
        step_size:   Stride of the sliding window
        timestamps:  Optional timestamps, shape (n,)
        series_id:   Optional identifier string for logging

    Returns:
        List of sample dicts. Each sample contains:
            context_window:       list of floats, length context_len
            ground_truth:         list of floats, length pred_len
            prediction:           list of floats, length pred_len (median forecast)
            prediction_quantiles: dict mapping quantile str → list of floats
            context_start_time:   timestamp string (if timestamps provided)
            context_end_time:     timestamp string (if timestamps provided)
            forecast_start_time:  timestamp string (if timestamps provided)
            forecast_end_time:    timestamp string (if timestamps provided)
    """
    samples = []
    n = len(values)
    min_required = context_len + pred_len

    if n < min_required:
        print(
            f"  Series '{series_id}' too short ({n} steps, need {min_required}) — skipping."
        )
        return samples

    num_windows = (n - min_required) // step_size + 1

    for i in range(num_windows):
        ctx_start = i * step_size
        ctx_end = ctx_start + context_len
        pred_end = ctx_end + pred_len

        context = values[ctx_start:ctx_end]
        ground_truth = values[ctx_end:pred_end]

        if np.isnan(context).any() or np.isnan(ground_truth).any():
            continue

        try:
            median_pred, quantiles = forecaster.predict_with_quantiles(
                context=context,
                horizon=pred_len,
            )

            sample = {
                "context_window": context.tolist(),
                "ground_truth": ground_truth.tolist(),
                "prediction": median_pred.tolist(),
                "prediction_quantiles": quantiles,
            }

            if timestamps is not None:
                sample["context_start_time"] = str(timestamps[ctx_start])
                sample["context_end_time"] = str(timestamps[ctx_end - 1])
                sample["forecast_start_time"] = str(timestamps[ctx_end])
                sample["forecast_end_time"] = str(timestamps[pred_end - 1])

            samples.append(sample)

        except Exception as e:
            print(f"  Warning: window {i} of '{series_id}' failed — {e}")
            continue

    return samples


def create_dataset(
    series_dict,
    output_dir,
    forecaster,
    context_len,
    pred_len,
    step_size,
    save_interval=10,
):
    """
    Runs the full dataset creation pipeline across train, val, and test splits.

    Each split can contain one or multiple time series. For a single asset
    (e.g. S&P 500), each split will be a list with one series. For multi-asset
    or panel data, pass multiple series per split.

    Intermediate .npy files are saved every save_interval series so long GPU
    jobs can be resumed if they crash.

    Args:
        series_dict:   Dict with keys "train", "val", "test". Each value is a list of
                       dicts with keys:
                         "values":     np.ndarray, shape (n,)
                         "timestamps": np.ndarray or None
                         "id":         str
        output_dir:    Directory where train.npy, val.npy, test.npy are saved
        forecaster:    A BaseForecaster instance
        context_len:   Historical context length
        pred_len:      Forecast horizon H
        step_size:     Sliding window stride
        save_interval: Save interim results every N series (crash recovery)
    """
    os.makedirs(output_dir, exist_ok=True)

    for split_name in ["train", "val", "test"]:
        print(f"\n{'='*52}")
        print(f"  Generating {split_name} split")
        print(f"{'='*52}")

        split_series_list = series_dict[split_name]
        all_samples = []
        start_time = time.time()

        for i, series in enumerate(tqdm(split_series_list, desc=split_name)):
            series_samples = generate_triplets(
                values=np.array(series["values"]),
                forecaster=forecaster,
                context_len=context_len,
                pred_len=pred_len,
                step_size=step_size,
                timestamps=series.get("timestamps"),
                series_id=series.get("id", str(i)),
            )
            all_samples.extend(series_samples)

            # Save intermediate results for crash recovery on long GPU jobs
            if (i + 1) % save_interval == 0 and all_samples:
                interim_path = os.path.join(
                    output_dir, f"{split_name}_interim_{i+1}.npy"
                )
                np.save(interim_path, all_samples)
                elapsed = time.time() - start_time
                print(
                    f"  [{i+1}/{len(split_series_list)}] {len(all_samples)} samples | {elapsed:.0f}s"
                )

        output_path = os.path.join(output_dir, f"{split_name}.npy")
        np.save(output_path, all_samples)
        print(f"  Saved {len(all_samples)} samples → {output_path}")

    print(f"\nDataset complete. Files saved to: {output_dir}/")
