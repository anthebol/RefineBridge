"""
scripts/evaluate.py — Evaluation entry point for RefineBridge.

Loads a trained checkpoint, runs evaluate_model_with_storage on the test set
(all metrics + per-sample stored tensors + inference timing), prints results,
plots top MSE and MAE performers, and saves random sample prediction figures.

Edit the CONFIG block to match your experiment, then run:

    python scripts/evaluate.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from src.dataset import RefineBridgeDataset
from src.evaluate import (
    evaluate_model_with_storage,
    plot_sample_predictions,
    plot_top_performing_samples,
    print_evaluation_results,
)
from src.models.refinebridge import RefineBridge

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Data paths
TRAIN_DATA_PATH = "data/SNP500/SNP500_price_21/train.npy"
TEST_DATA_PATH = "data/SNP500/SNP500_price_21/test.npy"

# Checkpoint to evaluate
CHECKPOINT_PATH = "checkpoints/SNP500_price_21/sb_model_best_V2.pt"

# Where evaluation figures are saved
OUTPUT_DIR = "results/SNP500_price_21"

# Model architecture — must match training config
CONTEXT_SEQ_LEN = 252
PRED_SEQ_LEN = 21
HIDDEN_DIM = 32
DIM_MULTS = (1, 2, 4)
BETA_MIN = 0.0001
BETA_MAX = 0.02
SCHEDULE_TYPE = "gmax"
PREDICTOR = "x0"

# Evaluation settings
SAMPLER = "sde"
N_TIMESTEPS = 100
TEMPERATURE = 0.01
BATCH_SIZE = 16

# Visualisation settings
NUM_RANDOM_SAMPLES = 10  # random samples for plot_sample_predictions
PLOT_CONTEXT_POINTS = 63  # historical context points shown in each plot

# ─────────────────────────────────────────────────────────────────────────────


def main():
    # ── device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device : {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets ...")
    train_dataset = RefineBridgeDataset(TRAIN_DATA_PATH, compute_stats=True)
    test_dataset = RefineBridgeDataset(
        TEST_DATA_PATH,
        global_stats=train_dataset.global_stats,
    )
    print(f"  Test samples : {len(test_dataset)}")

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building model ...")
    model = RefineBridge(
        context_dim=1,
        pred_dim=1,
        context_seq_len=CONTEXT_SEQ_LEN,
        pred_seq_len=PRED_SEQ_LEN,
        hidden_dim=HIDDEN_DIM,
        dim_mults=DIM_MULTS,
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        schedule_type=SCHEDULE_TYPE,
        predictor=PREDICTOR,
    ).to(device)

    # ── load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(
        f"  Loaded epoch {checkpoint['epoch']}"
        f"  |  val loss {checkpoint.get('val_loss', 'N/A')}"
    )

    # ── full evaluation with stored predictions + timing ─────────────────────
    print(
        f"\nRunning evaluation"
        f"  [{SAMPLER}  steps={N_TIMESTEPS}  temp={TEMPERATURE}] ..."
    )
    results = evaluate_model_with_storage(
        model=model,
        dataset=test_dataset,
        device=device,
        n_timesteps=N_TIMESTEPS,
        sampler=SAMPLER,
        temperature=TEMPERATURE,
        context_seq_len=CONTEXT_SEQ_LEN,
        batch_size=BATCH_SIZE,
    )

    # ── print structured metrics ──────────────────────────────────────────────
    print_evaluation_results(results)

    # ── top MSE performers ────────────────────────────────────────────────────
    top_mse = results.get("top_mse_performers", [])
    if top_mse:
        print("\nPlotting top MSE performers ...")
        plot_top_performing_samples(
            model=model,
            dataset=test_dataset,
            top_performers=top_mse,
            output_dir=os.path.join(OUTPUT_DIR, "top_mse"),
            n_timesteps=N_TIMESTEPS,
            sampler=SAMPLER,
            temperature=TEMPERATURE,
            context_seq_len=CONTEXT_SEQ_LEN,
            plot_type="mse",
            plot_context_points=PLOT_CONTEXT_POINTS,
        )

    # ── top MAE performers ────────────────────────────────────────────────────
    top_mae = results.get("top_mae_performers", [])
    if top_mae:
        print("\nPlotting top MAE performers ...")
        plot_top_performing_samples(
            model=model,
            dataset=test_dataset,
            top_performers=top_mae,
            output_dir=os.path.join(OUTPUT_DIR, "top_mae"),
            n_timesteps=N_TIMESTEPS,
            sampler=SAMPLER,
            temperature=TEMPERATURE,
            context_seq_len=CONTEXT_SEQ_LEN,
            plot_type="mae",
            plot_context_points=PLOT_CONTEXT_POINTS,
        )

    # ── random qualitative samples ────────────────────────────────────────────
    print(f"\nPlotting {NUM_RANDOM_SAMPLES} random sample predictions ...")
    plot_sample_predictions(
        model=model,
        dataset=test_dataset,
        output_dir=OUTPUT_DIR,
        n_timesteps=N_TIMESTEPS,
        num_samples=NUM_RANDOM_SAMPLES,
        sampler=SAMPLER,
        temperature=TEMPERATURE,
        context_seq_len=CONTEXT_SEQ_LEN,
        plot_context_points=PLOT_CONTEXT_POINTS,
    )

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
