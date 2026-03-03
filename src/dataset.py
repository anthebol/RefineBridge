"""
dataset.py — RefineBridgeDataset

PyTorch Dataset that loads triplet .npy files produced by the data generation
pipeline. Applies global Median-MAD (robust) or Mean-Std normalization computed
from the training set, then reuses those statistics on val/test sets to prevent
data leakage.

Usage:
    # Training set — compute global stats
    train_dataset = RefineBridgeDataset("data/snp500_21/train.npy", compute_stats=True)

    # Val/Test — reuse training stats
    val_dataset  = RefineBridgeDataset("data/snp500_21/val.npy",  global_stats=train_dataset.global_stats)
    test_dataset = RefineBridgeDataset("data/snp500_21/test.npy", global_stats=train_dataset.global_stats)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RefineBridgeDataset(Dataset):
    """
    Dataset for RefineBridge training triplets.

    Each sample contains:
        context_window:  [context_seq_len, 1]  — historical context
        ground_truth:    [pred_seq_len, 1]     — actual future values (x_0)
        prediction:      [pred_seq_len, 1]     — foundation model forecast (x_1)

    Args:
        data_path:       Path to .npy file containing a list of sample dicts
        global_stats:    Pre-computed normalization statistics (for val/test splits)
        compute_stats:   If True, compute global stats from this dataset (use for training)
        use_robust_norm: If True, use Median-MAD normalization (default, matches paper)
        asset_name:      Stored in sample["entity_id"] (default "asset")
        variable_name:   Stored in sample["variable"] (default "value")
    """

    def __init__(
        self,
        data_path,
        global_stats=None,
        compute_stats=False,
        use_robust_norm=True,
        asset_name="asset",
        variable_name="value",
    ):
        self.use_robust_norm = use_robust_norm
        self.asset_name = asset_name
        self.variable_name = variable_name

        print(f"Loading data from {data_path}")
        self.data = np.load(data_path, allow_pickle=True)
        print(f"Loaded dataset with {len(self.data)} samples")
        print(f"Using {'ROBUST' if use_robust_norm else 'STANDARD'} normalization")

        if compute_stats:
            print("Computing global statistics from training data...")
            self.global_stats = self._compute_global_stats()
            print("Global statistics computed.")
        else:
            if global_stats is None:
                raise ValueError(
                    "Either pass global_stats or set compute_stats=True. "
                    "Call with compute_stats=True on the training set, "
                    "then pass train_dataset.global_stats to val/test."
                )
            self.global_stats = global_stats
            print("Using provided global statistics.")

    def _compute_global_stats(self):
        """Compute normalization statistics across all samples in this split."""
        all_contexts, all_gts, all_preds = [], [], []

        print("Collecting all data for global statistics...")
        for i, sample in enumerate(self.data):
            if i % 1000 == 0:
                print(f"  Processing sample {i}/{len(self.data)}")
            all_contexts.extend(sample["context_window"])
            all_gts.extend(sample["ground_truth"])
            all_preds.extend(sample["prediction"])

        all_contexts = np.array(all_contexts)
        all_gts = np.array(all_gts)
        all_preds = np.array(all_preds)

        if self.use_robust_norm:

            def robust_stats(data):
                median = np.median(data)
                mad_scaled = np.median(np.abs(data - median)) * 1.4826
                if mad_scaled < 1e-8:
                    mad_scaled = 1.0
                return float(median), float(mad_scaled)

            context_median, context_mad = robust_stats(all_contexts)
            gt_median, gt_mad = robust_stats(all_gts)
            pred_median, pred_mad = robust_stats(all_preds)

            print(f"\nRobust statistics (Median-MAD):")
            print(f"  Context:      median={context_median:.6f}, MAD={context_mad:.6f}")
            print(f"  Ground Truth: median={gt_median:.6f},      MAD={gt_mad:.6f}")
            print(f"  Prediction:   median={pred_median:.6f},    MAD={pred_mad:.6f}")

            return {
                # Robust stats (primary)
                "context_median": context_median,
                "context_mad": context_mad,
                "gt_median": gt_median,
                "gt_mad": gt_mad,
                "pred_median": pred_median,
                "pred_mad": pred_mad,
                # Standard stats stored for compatibility
                "context_mean": float(np.mean(all_contexts)),
                "context_std": float(np.std(all_contexts) + 1e-6),
                "gt_mean": float(np.mean(all_gts)),
                "gt_std": float(np.std(all_gts) + 1e-6),
                "pred_mean": float(np.mean(all_preds)),
                "pred_std": float(np.std(all_preds) + 1e-6),
            }
        else:
            stats = {
                "context_mean": float(np.mean(all_contexts)),
                "context_std": float(np.std(all_contexts) + 1e-6),
                "gt_mean": float(np.mean(all_gts)),
                "gt_std": float(np.std(all_gts) + 1e-6),
                "pred_mean": float(np.mean(all_preds)),
                "pred_std": float(np.std(all_preds) + 1e-6),
            }

            print(f"\nStandard statistics:")
            print(
                f"  Context:      mean={stats['context_mean']:.6f}, std={stats['context_std']:.6f}"
            )
            print(
                f"  Ground Truth: mean={stats['gt_mean']:.6f},      std={stats['gt_std']:.6f}"
            )
            print(
                f"  Prediction:   mean={stats['pred_mean']:.6f},    std={stats['pred_std']:.6f}"
            )

            return stats

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        context = torch.tensor(sample["context_window"], dtype=torch.float32)
        ground_truth = torch.tensor(sample["ground_truth"], dtype=torch.float32)
        prediction = torch.tensor(sample["prediction"], dtype=torch.float32)

        # Ensure all tensors are [seq_len, 1]
        if context.dim() == 1:
            context = context.unsqueeze(1)
        if ground_truth.dim() == 1:
            ground_truth = ground_truth.unsqueeze(1)
        if prediction.dim() == 1:
            prediction = prediction.unsqueeze(1)

        # Apply normalization
        if self.use_robust_norm:
            context = (
                context - self.global_stats["context_median"]
            ) / self.global_stats["context_mad"]
            ground_truth = (
                ground_truth - self.global_stats["gt_median"]
            ) / self.global_stats["gt_mad"]
            prediction = (
                prediction - self.global_stats["pred_median"]
            ) / self.global_stats["pred_mad"]

            # Store stats for denormalization — use shared key names so
            # evaluation code works regardless of norm mode
            stats = {
                "context_mean": self.global_stats["context_median"],
                "context_std": self.global_stats["context_mad"],
                "gt_mean": self.global_stats["gt_median"],
                "gt_std": self.global_stats["gt_mad"],
                "pred_mean": self.global_stats["pred_median"],
                "pred_std": self.global_stats["pred_mad"],
            }
        else:
            context = (context - self.global_stats["context_mean"]) / self.global_stats[
                "context_std"
            ]
            ground_truth = (
                ground_truth - self.global_stats["gt_mean"]
            ) / self.global_stats["gt_std"]
            prediction = (
                prediction - self.global_stats["pred_mean"]
            ) / self.global_stats["pred_std"]

            stats = {
                "context_mean": self.global_stats["context_mean"],
                "context_std": self.global_stats["context_std"],
                "gt_mean": self.global_stats["gt_mean"],
                "gt_std": self.global_stats["gt_std"],
                "pred_mean": self.global_stats["pred_mean"],
                "pred_std": self.global_stats["pred_std"],
            }

        # Guard against NaN/Inf from normalization edge cases
        context = torch.nan_to_num(context, nan=0.0)
        ground_truth = torch.nan_to_num(ground_truth, nan=0.0)
        prediction = torch.nan_to_num(prediction, nan=0.0)

        return {
            "context_window": context,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "id": f"{self.asset_name}_sample_{idx}",
            "entity_id": self.asset_name,
            "variable": self.variable_name,
            "stats": stats,
        }
