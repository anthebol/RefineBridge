"""
scripts/hyperparam_search.py — Grid search over temperature x n_timesteps.

For a trained RefineBridge checkpoint, sweeps every (temperature, n_timesteps)
combination on the test set and saves:

  all_results.txt              — full per-config metric printout
  summary_results.csv          — one row per config, all key metrics
  mse_improvement_heatmap.csv  — pivot table: n_timesteps x temperature

Best configurations are printed to stdout for each metric at the end.

Edit the CONFIG block, then run:

    python scripts/hyperparam_search.py
"""

import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import torch

from src.dataset import RefineBridgeDataset
from src.evaluate import evaluate_model
from src.models.refinebridge import RefineBridge

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Data paths
TRAIN_DATA_PATH = "data/SNP500/SNP500_price_21/train.npy"
TEST_DATA_PATH = "data/SNP500/SNP500_price_21/test.npy"

# Checkpoint to sweep
CHECKPOINT_PATH = "checkpoints/SNP500_price_21/sb_model_best_V2.pt"

# Where search outputs are written
OUTPUT_DIR = "results/hyperparam_search/SNP500_price_21"

# Model architecture — must match training config
CONTEXT_SEQ_LEN = 252
PRED_SEQ_LEN = 21
HIDDEN_DIM = 32
DIM_MULTS = (1, 2, 4)
BETA_MIN = 0.0001
BETA_MAX = 0.02
SCHEDULE_TYPE = "gmax"
PREDICTOR = "x0"

# Evaluation DataLoader batch size
BATCH_SIZE = 16

# Sampler to use for the search
SAMPLER = "sde"

# Grid to sweep
TEMPERATURES = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 10000]
N_TIMESTEPS = [1, 10, 50, 100, 1000]

# ─────────────────────────────────────────────────────────────────────────────


def _log(lines, filepath):
    """Print lines to stdout and append to filepath."""
    text = "\n".join(lines)
    print(text)
    with open(filepath, "a") as f:
        f.write(text + "\n")


def format_results(results):
    """Return a list of formatted strings for one (temp, steps) result."""
    m = results.get("SNP500", {})
    lines = []

    if m.get("sample_count", 0) == 0:
        lines.append("  No valid samples.")
        return lines

    lines += [
        "",
        "  Error Metrics (Raw)",
        f"  {'MSE':<22}  orig={m['MSE_original']:.6f}  ref={m['MSE_refined']:.6f}"
        f"  ({m['MSE_improvement_percent']:+.2f}%)",
        f"  {'MAE':<22}  orig={m['MAE_original']:.6f}  ref={m['MAE_refined']:.6f}"
        f"  ({m['MAE_improvement_percent']:+.2f}%)",
        "",
        "  Error Metrics (Standard Z-Normalised)",
        f"  {'MSE':<22}  orig={m['MSE_original_znorm']:.6f}  ref={m['MSE_refined_znorm']:.6f}"
        f"  ({m['MSE_improvement_percent_znorm']:+.2f}%)",
        f"  {'MAE':<22}  orig={m['MAE_original_znorm']:.6f}  ref={m['MAE_refined_znorm']:.6f}"
        f"  ({m['MAE_improvement_percent_znorm']:+.2f}%)",
        "",
        "  Error Metrics (Robust Median-MAD)",
        f"  Median={m['robust_median']:.6f}  MAD={m['robust_mad']:.6f}",
        f"  {'MSE':<22}  orig={m['MSE_original_robust']:.6f}  ref={m['MSE_refined_robust']:.6f}"
        f"  ({m['MSE_improvement_percent_robust']:+.2f}%)",
        f"  {'MAE':<22}  orig={m['MAE_original_robust']:.6f}  ref={m['MAE_refined_robust']:.6f}"
        f"  ({m['MAE_improvement_percent_robust']:+.2f}%)",
        "",
        "  Ranking Metrics",
        f"  {'IC':<22}  orig={m['IC_original']:.4f}  ref={m['IC_refined']:.4f}"
        f"  (delta={m['IC_refined']-m['IC_original']:+.4f})",
        f"  {'ICIR':<22}  orig={m['ICIR_original']:.4f}  ref={m['ICIR_refined']:.4f}"
        f"  (delta={m['ICIR_refined']-m['ICIR_original']:+.4f})",
        f"  {'Rank IC':<22}  orig={m['RankIC_original']:.4f}  ref={m['RankIC_refined']:.4f}"
        f"  (delta={m['RankIC_refined']-m['RankIC_original']:+.4f})",
        f"  {'Rank ICIR':<22}  orig={m['RankICIR_original']:.4f}  ref={m['RankICIR_refined']:.4f}"
        f"  (delta={m['RankICIR_refined']-m['RankICIR_original']:+.4f})",
        "",
        "  Directional Accuracy",
        f"  {'Foundation':<22}  {m['DirectionalAccuracy_original']:.2f}%",
        f"  {'RefineBridge':<22}  {m['DirectionalAccuracy_refined']:.2f}%"
        f"  (delta={m['DirectionalAccuracy_refined']-m['DirectionalAccuracy_original']:+.2f}%)",
        "",
        f"  Sample count : {m['sample_count']}",
    ]
    return lines


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
    results_file = os.path.join(OUTPUT_DIR, "all_results.txt")
    summary_file = os.path.join(OUTPUT_DIR, "summary_results.csv")

    # Start fresh
    open(results_file, "w").close()

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

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded epoch {checkpoint['epoch']}")

    # ── grid search ───────────────────────────────────────────────────────────
    combos = list(product(TEMPERATURES, N_TIMESTEPS))
    n_combos = len(combos)
    all_rows = []

    print(f"\nHyperparameter search — {n_combos} combinations")
    print(f"  Temperatures : {TEMPERATURES}")
    print(f"  n_timesteps  : {N_TIMESTEPS}")
    print(f"  Sampler      : {SAMPLER}")
    print("-" * 60)

    for i, (temp, n_steps) in enumerate(combos, 1):
        header = (
            f"\n{'#'*60}\n"
            f"  Experiment {i}/{n_combos}  |  temp={temp}  steps={n_steps}\n"
            f"{'#'*60}"
        )
        _log([header], results_file)

        try:
            results = evaluate_model(
                model=model,
                dataset=test_dataset,
                device=device,
                n_timesteps=n_steps,
                sampler=SAMPLER,
                temperature=temp,
                context_seq_len=CONTEXT_SEQ_LEN,
                batch_size=BATCH_SIZE,
            )

            _log(format_results(results), results_file)

            m = results.get("SNP500", {})
            if m.get("sample_count", 0) > 0:
                all_rows.append(
                    {
                        "temperature": temp,
                        "n_timesteps": n_steps,
                        "mse_refined": m["MSE_refined"],
                        "mse_improvement_percent": m["MSE_improvement_percent"],
                        "mae_refined": m["MAE_refined"],
                        "mae_improvement_percent": m["MAE_improvement_percent"],
                        "mse_refined_znorm": m["MSE_refined_znorm"],
                        "mse_improvement_znorm": m["MSE_improvement_percent_znorm"],
                        "mse_refined_robust": m["MSE_refined_robust"],
                        "mse_improvement_robust": m["MSE_improvement_percent_robust"],
                        "ic_refined": m["IC_refined"],
                        "icir_refined": m["ICIR_refined"],
                        "rank_ic_refined": m["RankIC_refined"],
                        "rank_icir_refined": m["RankICIR_refined"],
                        "directional_accuracy_refined": m[
                            "DirectionalAccuracy_refined"
                        ],
                        "sample_count": m["sample_count"],
                    }
                )

        except Exception as e:
            _log([f"  [error] temp={temp} steps={n_steps}: {e}"], results_file)
            continue

    # ── summary ───────────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo successful evaluations — cannot produce summary.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(summary_file, index=False)

    W = 60

    def _best_row(col, higher=False):
        idx = df[col].idxmax() if higher else df[col].idxmin()
        return df.loc[idx]

    configs = [
        ("Best raw MSE", "mse_refined", False),
        ("Best MSE improvement %", "mse_improvement_percent", True),
        ("Best robust MSE improv %", "mse_improvement_robust", True),
        ("Best IC", "ic_refined", True),
        ("Best ICIR", "icir_refined", True),
        ("Best Rank IC", "rank_ic_refined", True),
        ("Best directional acc.", "directional_accuracy_refined", True),
    ]

    summary_lines = [f"\n{'='*W}", "  HYPERPARAMETER SEARCH — SUMMARY", f"{'='*W}"]
    for label, col, higher in configs:
        row = _best_row(col, higher)
        summary_lines.append(
            f"  {label:<28}  temp={row['temperature']:<8}"
            f"  steps={int(row['n_timesteps']):<6}  {col}={row[col]:.4f}"
        )

    # Top 5 by MSE improvement
    summary_lines += ["", "  Top 5 by MSE improvement %:"]
    summary_lines.append(
        f"  {'temp':<10}{'steps':<8}{'mse_improv':>12}{'ic':>10}{'dir_acc':>10}"
    )
    summary_lines.append(f"  {'-'*52}")
    top5 = df.nlargest(5, "mse_improvement_percent")
    for _, row in top5.iterrows():
        summary_lines.append(
            f"  {row['temperature']:<10}{int(row['n_timesteps']):<8}"
            f"{row['mse_improvement_percent']:>11.2f}%"
            f"{row['ic_refined']:>10.4f}"
            f"{row['directional_accuracy_refined']:>9.2f}%"
        )

    # MSE heatmap
    pivot = df.pivot_table(
        values="mse_improvement_percent",
        index="n_timesteps",
        columns="temperature",
    )
    heatmap_file = os.path.join(OUTPUT_DIR, "mse_improvement_heatmap.csv")
    pivot.to_csv(heatmap_file)

    summary_lines += [
        "",
        "  MSE Improvement % heatmap (rows=steps, cols=temp):",
        pivot.to_string(),
        "",
        f"{'='*W}",
        f"  Outputs -> {OUTPUT_DIR}",
        f"  - {os.path.basename(results_file)}",
        f"  - {os.path.basename(summary_file)}",
        f"  - {os.path.basename(heatmap_file)}",
        f"{'='*W}",
    ]

    _log(summary_lines, results_file)


if __name__ == "__main__":
    main()
