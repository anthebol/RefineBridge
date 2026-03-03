"""
scripts/train.py — Training entry point for RefineBridge.

Wires together dataset loading, model construction, optimizer, scheduler,
and the training loop. Edit the CONFIG block at the top to match your
asset / horizon / foundation model combination, then run:

    python scripts/train.py

Checkpoints and loss curves are saved to OUTPUT_DIR.
"""

import os
import sys

# Make the src/ package importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader

from src.dataset import RefineBridgeDataset
from src.models.refinebridge import RefineBridge
from src.training import train_model

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these to match your experiment
# ─────────────────────────────────────────────────────────────────────────────

# Data paths (produced by scripts/generate_dataset.py)
TRAIN_DATA_PATH = "data/CHRONOS_bridge_dataset/EURUSD/EURUSD_price_10/train.npy"
VAL_DATA_PATH = "data/CHRONOS_bridge_dataset/EURUSD/EURUSD_price_10/val.npy"

# Where checkpoints and training curves are written
OUTPUT_DIR = "checkpoints/smoke_test"

# Model architecture  (must match the dataset's sequence lengths)
CONTEXT_SEQ_LEN = 252  # sliding window fed to the context encoder
PRED_SEQ_LEN = 10  # prediction horizon (H)
HIDDEN_DIM = 32
DIM_MULTS = (1, 2, 4)

# Noise schedule  — paper uses (0.01, 50) for H<=10, (0.0001, 0.02) for H>=21
BETA_MIN = 0.01
BETA_MAX = 50
SCHEDULE_TYPE = "gmax"  # "gmax" | "vp" | "sb" | "constant"
PREDICTOR = "x0"

# Training hyperparameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
BATCH_SIZE = 4
ACCUMULATION_STEPS = 1
USE_EARLY_STOPPING = False  # set False to run the full NUM_EPOCHS

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
    val_dataset = RefineBridgeDataset(
        VAL_DATA_PATH,
        global_stats=train_dataset.global_stats,
    )
    print(f"  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

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

    print(f"  Parameters    : {model.nparams:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\nStarting training for up to {NUM_EPOCHS} epochs ...")
    print(f"Checkpoints -> {OUTPUT_DIR}\n")

    model, train_losses, val_losses = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        accumulation_steps=ACCUMULATION_STEPS,
        output_dir=OUTPUT_DIR,
        use_early_stopping=USE_EARLY_STOPPING,
        context_seq_len=CONTEXT_SEQ_LEN,
    )

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
