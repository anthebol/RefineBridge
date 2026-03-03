"""
generate_dataset.py — CLI entry point for RefineBridge dataset generation.

Generates train/val/test triplet datasets from a time series CSV using a
time series foundation model (TSFM). Output is saved as .npy files ready
for training RefineBridge.

Usage:
    python scripts/generate_dataset.py \
        --input data/raw/snp500.csv \
        --value-col Close \
        --date-col Date \
        --asset snp500 \
        --context-len 252 \
        --pred-len 21 \
        --step-size 21

Output:
    data/snp500_21/train.npy
    data/snp500_21/val.npy
    data/snp500_21/test.npy
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Make src/ importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.generate import create_dataset, split_series


def load_csv(path, value_col, date_col=None):
    """
    Loads a CSV and returns (values, timestamps).

    Expects a CSV where rows are time steps, ordered chronologically.
    The value_col should contain the numeric series values.
    """
    df = pd.read_csv(path)

    if value_col not in df.columns:
        raise ValueError(
            f"Column '{value_col}' not found in {path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    values = df[value_col].values.astype(float)
    timestamps = df[date_col].values if date_col and date_col in df.columns else None

    return values, timestamps


def main():
    parser = argparse.ArgumentParser(
        description="Generate RefineBridge triplet datasets from a time series CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--value-col", required=True, help="Column name containing time series values"
    )
    parser.add_argument(
        "--date-col", default=None, help="Column name for timestamps (optional)"
    )
    parser.add_argument(
        "--asset",
        required=True,
        help="Asset name used in output folder, e.g. snp500, wti, eurusd",
    )

    # Dataset settings
    parser.add_argument(
        "--context-len",
        type=int,
        default=252,
        help="Context window length fed to the foundation model",
    )
    parser.add_argument(
        "--pred-len",
        type=int,
        default=21,
        help="Forecast horizon H (steps ahead to predict)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=21,
        help="Sliding window stride (set equal to pred-len to avoid overlap)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Fraction of data for training"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Fraction of data for validation"
    )

    parser.add_argument(
        "--tsfm",
        default="chronos",
        choices=["chronos", "moirai", "timemoe"],
        help="Which foundation model family to use",
    )

    # Foundation model
    parser.add_argument(
        "--model",
        default="amazon/chronos-t5-large",
        help="HuggingFace model name for Chronos",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to run the foundation model on",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Root output directory; dataset saved to output-dir/asset_predlen/",
    )

    args = parser.parse_args()

    # Construct output path: data/{asset}_{pred_len}/
    output_dir = os.path.join(args.output_dir, f"{args.asset}_{args.pred_len}")

    print("=" * 52)
    print("  RefineBridge Dataset Generation")
    print("=" * 52)
    print(f"  Asset:        {args.asset}")
    print(f"  Horizon:      H = {args.pred_len} steps")
    print(f"  Context:      {args.context_len} steps")
    print(f"  Step size:    {args.step_size}")
    print(
        f"  Split:        {args.train_ratio:.0%} / {args.val_ratio:.0%} / {1-args.train_ratio-args.val_ratio:.0%}"
    )
    print(f"  Model:        {args.model}")
    print(f"  Device:       {args.device}")
    print(f"  Output:       {output_dir}/")
    print()

    # Load raw data
    print(f"Loading {args.input}...")
    values, timestamps = load_csv(args.input, args.value_col, args.date_col)
    print(f"  {len(values)} time steps loaded")

    # Chronological split — no randomness, no data leakage
    train_split, val_split, test_split = split_series(
        values, timestamps, args.train_ratio, args.val_ratio
    )
    print(f"  Train: {len(train_split['values'])} steps")
    print(f"  Val:   {len(val_split['values'])} steps")
    print(f"  Test:  {len(test_split['values'])} steps")

    # Pack into series_dict format expected by create_dataset
    # Each split is a list of series dicts. Single asset = list of one.
    series_dict = {
        "train": [
            {
                "values": train_split["values"],
                "timestamps": train_split["timestamps"],
                "id": args.asset,
            }
        ],
        "val": [
            {
                "values": val_split["values"],
                "timestamps": val_split["timestamps"],
                "id": args.asset,
            }
        ],
        "test": [
            {
                "values": test_split["values"],
                "timestamps": test_split["timestamps"],
                "id": args.asset,
            }
        ],
    }

    # Load foundation model
    if args.tsfm == "chronos":
        from src.data.tsfm import ChronosForecaster

        forecaster = ChronosForecaster(
            model_name=args.model or "amazon/chronos-t5-large",
            device=args.device,
        )
    elif args.tsfm == "moirai":
        from src.data.tsfm import MoiraiForecaster

        forecaster = MoiraiForecaster(
            model_name=args.model or "Salesforce/moirai-1.0-R-large",
            device=args.device,
        )
    elif args.tsfm == "timemoe":
        from src.data.tsfm import TimeMoEForecaster

        forecaster = TimeMoEForecaster(
            model_name=args.model or "Maple728/TimeMoE-50M",
            device=args.device,
        )
    else:
        raise ValueError(
            f"Unknown --tsfm '{args.tsfm}'. Choose from: chronos, moirai, timemoe"
        )

    # Generate and save
    create_dataset(
        series_dict=series_dict,
        output_dir=output_dir,
        forecaster=forecaster,
        context_len=args.context_len,
        pred_len=args.pred_len,
        step_size=args.step_size,
    )


if __name__ == "__main__":
    main()
