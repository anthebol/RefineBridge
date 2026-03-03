"""
evaluate.py — Evaluation utilities for RefineBridge.

    calculate_returns              — Returns from a price sequence.
    evaluate_model                 — Full dataset evaluation, no stored tensors.
    evaluate_model_with_storage    — Same + stores raw predictions + inference timing.
    print_evaluation_results       — Formatted console summary.
    plot_sample_predictions        — Qualitative plot of random samples.
    plot_top_performing_samples    — Plots for top-N performers.
    run_evaluation_with_plots      — End-to-end convenience wrapper.

Original locations (timesbridge_snp500_price.py):
    calculate_returns              lines 2882-2890
    evaluate_model                 lines 2893-3386
    print_evaluation_results       lines 3665-3828
    plot_sample_predictions        lines 2586-2741
    plot_top_performing_samples    lines 3831-4137
    run_evaluation_with_plots      lines 4141-4196
    evaluate_model_with_storage    lines 10310-10898
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .training import generate_prediction

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def calculate_returns(prices):
    """
    Compute period-over-period returns from a price sequence.
    Args:
        prices: Tensor [seq_len, 1]
    Returns:
        returns: Tensor [seq_len, 1], first element zero (no prior price).
    """
    if len(prices) < 2:
        return torch.zeros_like(prices)
    returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)
    return torch.cat([torch.zeros(1, prices.shape[1]), returns], dim=0)


def _compute_robust_params(data):
    """Median and MAD-scaled std. Scale factor 1.4826 = Gaussian consistency."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    mad_scaled = mad * 1.4826
    if mad_scaled < 1e-8:
        mad_scaled = 1.0
    return median, mad_scaled


def _aggregate_metrics(results):
    """
    Compute all aggregate metrics from accumulated per-sample results.
    Shared by evaluate_model and evaluate_model_with_storage.
    Returns a full metrics dict, or NaN-filled dict if no samples processed.
    """
    _nan = float("nan")

    if len(results["mse_original"]) == 0:
        return {
            "MSE_original": _nan,
            "MSE_refined": _nan,
            "MAE_original": _nan,
            "MAE_refined": _nan,
            "MSE_improvement_percent": _nan,
            "MAE_improvement_percent": _nan,
            "MSE_original_znorm": _nan,
            "MSE_refined_znorm": _nan,
            "MAE_original_znorm": _nan,
            "MAE_refined_znorm": _nan,
            "MSE_improvement_percent_znorm": _nan,
            "MAE_improvement_percent_znorm": _nan,
            "MSE_original_robust": _nan,
            "MSE_refined_robust": _nan,
            "MAE_original_robust": _nan,
            "MAE_refined_robust": _nan,
            "MSE_improvement_percent_robust": _nan,
            "MAE_improvement_percent_robust": _nan,
            "robust_median": _nan,
            "robust_mad": _nan,
            "IC_original": _nan,
            "IC_refined": _nan,
            "ICIR_original": _nan,
            "ICIR_refined": _nan,
            "RankIC_original": _nan,
            "RankIC_refined": _nan,
            "RankICIR_original": _nan,
            "RankICIR_refined": _nan,
            "DirectionalAccuracy_original": _nan,
            "DirectionalAccuracy_refined": _nan,
            "sample_count": 0,
        }

    # ---- Standard z-norm metrics ----
    avg_mse_orig = np.mean(results["mse_original"])
    avg_mse_ref = np.mean(results["mse_refined"])
    avg_mae_orig = np.mean(results["mae_original"])
    avg_mae_ref = np.mean(results["mae_refined"])
    avg_mse_orig_z = np.mean(results["mse_original_znorm"])
    avg_mse_ref_z = np.mean(results["mse_refined_znorm"])
    avg_mae_orig_z = np.mean(results["mae_original_znorm"])
    avg_mae_ref_z = np.mean(results["mae_refined_znorm"])

    def _pct_imp(orig, ref):
        return (orig - ref) / orig * 100 if orig > 0 else 0

    # ---- Robust normalisation metrics ----
    all_gt_raw = np.array(results["all_gt_raw"])
    all_pred_raw = np.array(results["all_pred_raw"])
    all_refined_raw = np.array(results["all_refined_raw"])

    gt_median, gt_mad = _compute_robust_params(all_gt_raw)
    gt_rob = (all_gt_raw - gt_median) / gt_mad
    pred_rob = (all_pred_raw - gt_median) / gt_mad
    ref_rob = (all_refined_raw - gt_median) / gt_mad

    mse_orig_rob = np.mean((pred_rob - gt_rob) ** 2)
    mse_ref_rob = np.mean((ref_rob - gt_rob) ** 2)
    mae_orig_rob = np.mean(np.abs(pred_rob - gt_rob))
    mae_ref_rob = np.mean(np.abs(ref_rob - gt_rob))

    # ---- Ranking metrics ----
    gt_rets = np.array(results["all_gt_returns"])
    pred_rets = np.array(results["all_pred_returns"])
    ref_rets = np.array(results["all_refined_returns"])

    ic_orig, _ = pearsonr(pred_rets, gt_rets)
    ic_ref, _ = pearsonr(ref_rets, gt_rets)
    rank_ic_orig, _ = spearmanr(pred_rets, gt_rets)
    rank_ic_ref, _ = spearmanr(ref_rets, gt_rets)

    # ICIR = IC / 0.1  (simplified, as in original)
    icir_orig = ic_orig / 0.1 if not np.isnan(ic_orig) else 0
    icir_ref = ic_ref / 0.1 if not np.isnan(ic_ref) else 0
    rank_icir_orig = rank_ic_orig / 0.1 if not np.isnan(rank_ic_orig) else 0
    rank_icir_ref = rank_ic_ref / 0.1 if not np.isnan(rank_ic_ref) else 0

    # ---- Directional accuracy ----
    dir_acc_orig = (
        results["directional_correct_original"]
        / results["total_directional_samples"]
        * 100
    )
    dir_acc_ref = (
        results["directional_correct_refined"]
        / results["total_directional_samples"]
        * 100
    )

    return {
        "MSE_original": avg_mse_orig,
        "MSE_refined": avg_mse_ref,
        "MAE_original": avg_mae_orig,
        "MAE_refined": avg_mae_ref,
        "MSE_improvement_percent": _pct_imp(avg_mse_orig, avg_mse_ref),
        "MAE_improvement_percent": _pct_imp(avg_mae_orig, avg_mae_ref),
        "MSE_original_znorm": avg_mse_orig_z,
        "MSE_refined_znorm": avg_mse_ref_z,
        "MAE_original_znorm": avg_mae_orig_z,
        "MAE_refined_znorm": avg_mae_ref_z,
        "MSE_improvement_percent_znorm": _pct_imp(avg_mse_orig_z, avg_mse_ref_z),
        "MAE_improvement_percent_znorm": _pct_imp(avg_mae_orig_z, avg_mae_ref_z),
        "MSE_original_robust": mse_orig_rob,
        "MSE_refined_robust": mse_ref_rob,
        "MAE_original_robust": mae_orig_rob,
        "MAE_refined_robust": mae_ref_rob,
        "MSE_improvement_percent_robust": _pct_imp(mse_orig_rob, mse_ref_rob),
        "MAE_improvement_percent_robust": _pct_imp(mae_orig_rob, mae_ref_rob),
        "robust_median": gt_median,
        "robust_mad": gt_mad,
        "IC_original": ic_orig,
        "IC_refined": ic_ref,
        "ICIR_original": icir_orig,
        "ICIR_refined": icir_ref,
        "RankIC_original": rank_ic_orig,
        "RankIC_refined": rank_ic_ref,
        "RankICIR_original": rank_icir_orig,
        "RankICIR_refined": rank_icir_ref,
        "DirectionalAccuracy_original": dir_acc_orig,
        "DirectionalAccuracy_refined": dir_acc_ref,
        "sample_count": len(results["mse_original"]),
    }


# --------------------------------------------------------------------------- #
# evaluate_model                                                               #
# --------------------------------------------------------------------------- #


def evaluate_model(
    model,
    dataset,
    device,
    n_timesteps=1000,
    sampler="ode",
    temperature=0.5,
    context_seq_len=252,
    batch_size=16,
):
    model.eval()

    sampler_map = {
        "sde": "sde",
        "ode": "ode",
        "pc_sde": "pc_sde",
        "pc_ode": "pc_ode",
        "pf_ode_euler": "ode",
        "sde_euler": "sde",
    }
    internal_sampler = sampler_map.get(sampler, "ode")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    results = {
        "mse_original": [],
        "mse_refined": [],
        "mae_original": [],
        "mae_refined": [],
        "mse_original_znorm": [],
        "mse_refined_znorm": [],
        "mae_original_znorm": [],
        "mae_refined_znorm": [],
        "all_gt_raw": [],
        "all_pred_raw": [],
        "all_refined_raw": [],
        "all_gt_returns": [],
        "all_pred_returns": [],
        "all_refined_returns": [],
        "directional_correct_original": 0,
        "directional_correct_refined": 0,
        "total_directional_samples": 0,
        "sample_details": [],
    }

    sample_idx_global = 0

    for batch in tqdm(dataloader, desc=f"Evaluating [{sampler}]"):
        try:
            context = batch["context_window"].to(device, non_blocking=True)
            ground_truth = batch["ground_truth"].to(device, non_blocking=True)
            pred = batch["prediction"].to(device, non_blocking=True)

            entity_ids = batch["entity_id"]
            variables = batch["variable"]
            sample_ids = batch["id"]

            if context.shape[1] > context_seq_len:
                context = context[:, -context_seq_len:]

            if context.dim() == 2:
                context = context.unsqueeze(-1)
            if ground_truth.dim() == 2:
                ground_truth = ground_truth.unsqueeze(-1)
            if pred.dim() == 2:
                pred = pred.unsqueeze(-1)

            batch_size_actual = pred.shape[0]
            mask = torch.ones(batch_size_actual, 1, device=device)

            with torch.no_grad():
                _, refined, _ = model(
                    context=context,
                    foundation_pred=pred,
                    mask=mask,
                    n_timesteps=n_timesteps,
                    sampler=internal_sampler,
                    temperature=temperature,
                )

            if refined.dim() == 4:
                refined = refined[:, -1]

            if refined.shape != ground_truth.shape:
                print(
                    f"  [warn] Shape mismatch refined {refined.shape} vs gt {ground_truth.shape}, skipping"
                )
                continue

            stats = batch["stats"]

            for i in range(batch_size_actual):
                try:
                    gt_i = ground_truth[i]
                    pred_i = pred[i]
                    refined_i = refined[i]

                    def _scalar(v):
                        return v[i].item() if torch.is_tensor(v) else float(v)

                    gt_mean = _scalar(stats["gt_mean"])
                    gt_std = _scalar(stats["gt_std"])
                    pred_mean = _scalar(stats["pred_mean"])
                    pred_std = _scalar(stats["pred_std"])

                    mse_orig_z = F.mse_loss(pred_i, gt_i).item()
                    mae_orig_z = F.l1_loss(pred_i, gt_i).item()
                    mse_ref_z = F.mse_loss(refined_i, gt_i).item()
                    mae_ref_z = F.l1_loss(refined_i, gt_i).item()

                    gt_denorm = gt_i * gt_std + gt_mean
                    pred_denorm = pred_i * pred_std + pred_mean
                    refined_denorm = refined_i * pred_std + pred_mean

                    mse_orig = F.mse_loss(pred_denorm, gt_denorm).item()
                    mae_orig = F.l1_loss(pred_denorm, gt_denorm).item()
                    mse_ref = F.mse_loss(refined_denorm, gt_denorm).item()
                    mae_ref = F.l1_loss(refined_denorm, gt_denorm).item()

                    results["all_gt_raw"].extend(gt_denorm.cpu().numpy().flatten())
                    results["all_pred_raw"].extend(pred_denorm.cpu().numpy().flatten())
                    results["all_refined_raw"].extend(
                        refined_denorm.cpu().numpy().flatten()
                    )

                    gt_rets = calculate_returns(gt_denorm.cpu())
                    pred_rets = calculate_returns(pred_denorm.cpu())
                    ref_rets = calculate_returns(refined_denorm.cpu())

                    if gt_rets.shape[0] > 1:
                        results["all_gt_returns"].extend(gt_rets[1:].flatten().numpy())
                        results["all_pred_returns"].extend(
                            pred_rets[1:].flatten().numpy()
                        )
                        results["all_refined_returns"].extend(
                            ref_rets[1:].flatten().numpy()
                        )

                        gt_dir = (gt_rets[1:] > 0).float()
                        pred_dir = (pred_rets[1:] > 0).float()
                        ref_dir = (ref_rets[1:] > 0).float()

                        results["directional_correct_original"] += (
                            (gt_dir == pred_dir).sum().item()
                        )
                        results["directional_correct_refined"] += (
                            (gt_dir == ref_dir).sum().item()
                        )
                        results["total_directional_samples"] += gt_dir.numel()

                    def _pct(orig, ref):
                        return (orig - ref) / orig * 100 if orig > 0 else 0

                    results["mse_original"].append(mse_orig)
                    results["mae_original"].append(mae_orig)
                    results["mse_refined"].append(mse_ref)
                    results["mae_refined"].append(mae_ref)
                    results["mse_original_znorm"].append(mse_orig_z)
                    results["mae_original_znorm"].append(mae_orig_z)
                    results["mse_refined_znorm"].append(mse_ref_z)
                    results["mae_refined_znorm"].append(mae_ref_z)

                    results["sample_details"].append(
                        {
                            "sample_id": sample_ids[i],
                            "global_idx": sample_idx_global,
                            "variable": variables[i],
                            "entity_id": entity_ids[i],
                            "mse_original": mse_orig,
                            "mse_refined": mse_ref,
                            "mse_improvement": _pct(mse_orig, mse_ref),
                            "mae_original": mae_orig,
                            "mae_refined": mae_ref,
                            "mae_improvement": _pct(mae_orig, mae_ref),
                            "mse_original_znorm": mse_orig_z,
                            "mse_refined_znorm": mse_ref_z,
                            "mse_improvement_znorm": _pct(mse_orig_z, mse_ref_z),
                            "mae_original_znorm": mae_orig_z,
                            "mae_refined_znorm": mae_ref_z,
                            "mae_improvement_znorm": _pct(mae_orig_z, mae_ref_z),
                        }
                    )

                    sample_idx_global += 1

                except Exception as e:
                    print(f"  [warn] Error processing sample {i}: {e}")
                    continue

        except Exception as e:
            print(f"  [warn] Error in evaluation batch: {e}")
            continue

    metrics = _aggregate_metrics(results)
    metrics["sample_details"] = results["sample_details"]

    all_samples = results["sample_details"]
    top_mse = sorted(all_samples, key=lambda x: x["mse_improvement"], reverse=True)[:50]
    top_mae = sorted(all_samples, key=lambda x: x["mae_improvement"], reverse=True)[:50]

    return {
        "sampler": sampler,
        "n_timesteps": n_timesteps,
        "temperature": temperature,
        "context_seq_len": context_seq_len,
        "SNP500": metrics,
        "top_mse_performers": top_mse,
        "top_mae_performers": top_mae,
    }


# --------------------------------------------------------------------------- #
# evaluate_model_with_storage                                                  #
# --------------------------------------------------------------------------- #


def evaluate_model_with_storage(
    model,
    dataset,
    device,
    n_timesteps=1000,
    sampler="ode",
    temperature=1.5,
    context_seq_len=252,
    batch_size=16,
):
    model.eval()

    sampler_map = {
        "sde": "sde",
        "ode": "ode",
        "pc_sde": "pc_sde",
        "pc_ode": "pc_ode",
        "pf_ode_euler": "ode",
        "sde_euler": "sde",
    }
    internal_sampler = sampler_map.get(sampler, "ode")

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    results = {
        "mse_original": [],
        "mse_refined": [],
        "mae_original": [],
        "mae_refined": [],
        "mse_original_znorm": [],
        "mse_refined_znorm": [],
        "mae_original_znorm": [],
        "mae_refined_znorm": [],
        "all_gt_raw": [],
        "all_pred_raw": [],
        "all_refined_raw": [],
        "all_gt_returns": [],
        "all_pred_returns": [],
        "all_refined_returns": [],
        "directional_correct_original": 0,
        "directional_correct_refined": 0,
        "total_directional_samples": 0,
        "sample_details": [],
        "sample_predictions": {},
    }

    sample_idx_global = 0
    total_refinement_time = 0.0
    total_samples_refined = 0
    total_batches_refined = 0
    batch_times = []

    def _sync():
        dev = str(device)
        if "cuda" in dev:
            torch.cuda.synchronize()
        elif "mps" in dev:
            torch.mps.synchronize()

    for batch in tqdm(dataloader, desc=f"Evaluating [{sampler}]"):
        try:
            context = batch["context_window"].to(device, non_blocking=True)
            ground_truth = batch["ground_truth"].to(device, non_blocking=True)
            pred = batch["prediction"].to(device, non_blocking=True)

            entity_ids = batch["entity_id"]
            variables = batch["variable"]
            sample_ids = batch["id"]

            context_original = context.clone()

            if context.shape[1] > context_seq_len:
                context = context[:, -context_seq_len:]

            if context.dim() == 2:
                context = context.unsqueeze(-1)
            if ground_truth.dim() == 2:
                ground_truth = ground_truth.unsqueeze(-1)
            if pred.dim() == 2:
                pred = pred.unsqueeze(-1)
            if context_original.dim() == 2:
                context_original = context_original.unsqueeze(-1)

            batch_size_actual = pred.shape[0]
            mask = torch.ones(batch_size_actual, 1, device=device)

            _sync()
            t0 = time.perf_counter()

            with torch.no_grad():
                _, refined, _ = model(
                    context=context,
                    foundation_pred=pred,
                    mask=mask,
                    n_timesteps=n_timesteps,
                    sampler=internal_sampler,
                    temperature=temperature,
                )

            _sync()
            elapsed = time.perf_counter() - t0

            total_refinement_time += elapsed
            total_samples_refined += batch_size_actual
            total_batches_refined += 1
            batch_times.append(elapsed)

            if refined.dim() == 4:
                refined = refined[:, -1]

            if refined.shape != ground_truth.shape:
                print(
                    f"  [warn] Shape mismatch refined {refined.shape} vs gt {ground_truth.shape}, skipping"
                )
                continue

            stats = batch["stats"]

            for i in range(batch_size_actual):
                try:
                    gt_i = ground_truth[i]
                    pred_i = pred[i]
                    refined_i = refined[i]

                    def _scalar(v):
                        return v[i].item() if torch.is_tensor(v) else float(v)

                    gt_mean = _scalar(stats["gt_mean"])
                    gt_std = _scalar(stats["gt_std"])
                    pred_mean = _scalar(stats["pred_mean"])
                    pred_std = _scalar(stats["pred_std"])
                    context_mean = _scalar(stats["context_mean"])
                    context_std = _scalar(stats["context_std"])

                    results["sample_predictions"][sample_idx_global] = {
                        "context": context[i].cpu(),
                        "context_original": context_original[i].cpu(),
                        "ground_truth": gt_i.cpu(),
                        "prediction": pred_i.cpu(),
                        "refined": refined_i.cpu(),
                        "stats": {
                            "context_mean": context_mean,
                            "context_std": context_std,
                            "gt_mean": gt_mean,
                            "gt_std": gt_std,
                            "pred_mean": pred_mean,
                            "pred_std": pred_std,
                        },
                        "entity_id": entity_ids[i],
                        "variable": variables[i],
                        "sample_id": sample_ids[i],
                    }

                    mse_orig_z = F.mse_loss(pred_i, gt_i).item()
                    mae_orig_z = F.l1_loss(pred_i, gt_i).item()
                    mse_ref_z = F.mse_loss(refined_i, gt_i).item()
                    mae_ref_z = F.l1_loss(refined_i, gt_i).item()

                    gt_denorm = gt_i * gt_std + gt_mean
                    pred_denorm = pred_i * pred_std + pred_mean
                    refined_denorm = refined_i * pred_std + pred_mean

                    mse_orig = F.mse_loss(pred_denorm, gt_denorm).item()
                    mae_orig = F.l1_loss(pred_denorm, gt_denorm).item()
                    mse_ref = F.mse_loss(refined_denorm, gt_denorm).item()
                    mae_ref = F.l1_loss(refined_denorm, gt_denorm).item()

                    results["all_gt_raw"].extend(gt_denorm.cpu().numpy().flatten())
                    results["all_pred_raw"].extend(pred_denorm.cpu().numpy().flatten())
                    results["all_refined_raw"].extend(
                        refined_denorm.cpu().numpy().flatten()
                    )

                    gt_rets = calculate_returns(gt_denorm.cpu())
                    pred_rets = calculate_returns(pred_denorm.cpu())
                    ref_rets = calculate_returns(refined_denorm.cpu())

                    if gt_rets.shape[0] > 1:
                        results["all_gt_returns"].extend(gt_rets[1:].flatten().numpy())
                        results["all_pred_returns"].extend(
                            pred_rets[1:].flatten().numpy()
                        )
                        results["all_refined_returns"].extend(
                            ref_rets[1:].flatten().numpy()
                        )

                        gt_dir = (gt_rets[1:] > 0).float()
                        pred_dir = (pred_rets[1:] > 0).float()
                        ref_dir = (ref_rets[1:] > 0).float()

                        results["directional_correct_original"] += (
                            (gt_dir == pred_dir).sum().item()
                        )
                        results["directional_correct_refined"] += (
                            (gt_dir == ref_dir).sum().item()
                        )
                        results["total_directional_samples"] += gt_dir.numel()

                    def _pct(orig, ref):
                        return (orig - ref) / orig * 100 if orig > 0 else 0

                    results["mse_original"].append(mse_orig)
                    results["mae_original"].append(mae_orig)
                    results["mse_refined"].append(mse_ref)
                    results["mae_refined"].append(mae_ref)
                    results["mse_original_znorm"].append(mse_orig_z)
                    results["mae_original_znorm"].append(mae_orig_z)
                    results["mse_refined_znorm"].append(mse_ref_z)
                    results["mae_refined_znorm"].append(mae_ref_z)

                    results["sample_details"].append(
                        {
                            "sample_id": sample_ids[i],
                            "global_idx": sample_idx_global,
                            "variable": variables[i],
                            "entity_id": entity_ids[i],
                            "mse_original": mse_orig,
                            "mse_refined": mse_ref,
                            "mse_improvement": _pct(mse_orig, mse_ref),
                            "mae_original": mae_orig,
                            "mae_refined": mae_ref,
                            "mae_improvement": _pct(mae_orig, mae_ref),
                            "mse_original_znorm": mse_orig_z,
                            "mse_refined_znorm": mse_ref_z,
                            "mse_improvement_znorm": _pct(mse_orig_z, mse_ref_z),
                            "mae_original_znorm": mae_orig_z,
                            "mae_refined_znorm": mae_ref_z,
                            "mae_improvement_znorm": _pct(mae_orig_z, mae_ref_z),
                        }
                    )

                    sample_idx_global += 1

                except Exception as e:
                    print(f"  [warn] Error processing sample {i}: {e}")
                    continue

        except Exception as e:
            print(f"  [warn] Error in evaluation batch: {e}")
            continue

    # ── timing summary ───────────────────────────────────────────────────────
    W = 60
    print()
    print("=" * W)
    print(" RefineBridge — Inference Timing")
    print("=" * W)
    print(f"  Sampler         : {sampler}")
    print(f"  Steps           : {n_timesteps}")
    print(f"  Temperature     : {temperature}")
    print(f"  Device          : {device}")
    print(f"  Samples         : {total_samples_refined}")
    print(f"  Batches         : {total_batches_refined}")
    print("-" * W)
    if total_samples_refined > 0:
        per_sample_ms = total_refinement_time / total_samples_refined * 1000
        per_batch_ms = total_refinement_time / total_batches_refined * 1000
        throughput = total_samples_refined / total_refinement_time
        print(f"  Total time      : {total_refinement_time:.3f} s")
        print(f"  Per sample      : {per_sample_ms:.3f} ms")
        print(f"  Per batch       : {per_batch_ms:.3f} ms")
        print(f"  Throughput      : {throughput:.1f} samples/s")
        if len(batch_times) > 1:
            print(f"  Batch time min  : {min(batch_times) * 1000:.3f} ms")
            print(f"  Batch time max  : {max(batch_times) * 1000:.3f} ms")
            print(f"  Batch time std  : {np.std(batch_times) * 1000:.3f} ms")
    print("=" * W)
    print()

    timing = {
        "total_refinement_time_seconds": total_refinement_time,
        "total_samples": total_samples_refined,
        "total_batches": total_batches_refined,
        "avg_time_per_sample_ms": (
            total_refinement_time / total_samples_refined * 1000
            if total_samples_refined > 0
            else 0
        ),
        "avg_time_per_batch_ms": (
            total_refinement_time / total_batches_refined * 1000
            if total_batches_refined > 0
            else 0
        ),
        "throughput_samples_per_second": (
            total_samples_refined / total_refinement_time
            if total_refinement_time > 0
            else 0
        ),
        "batch_times": batch_times,
    }

    metrics = _aggregate_metrics(results)
    metrics["sample_details"] = results["sample_details"]
    metrics["sample_predictions"] = results["sample_predictions"]

    all_samples = results["sample_details"]
    top_mse = sorted(all_samples, key=lambda x: x["mse_improvement"], reverse=True)[:50]
    top_mae = sorted(all_samples, key=lambda x: x["mae_improvement"], reverse=True)[:50]

    return {
        "sampler": sampler,
        "n_timesteps": n_timesteps,
        "temperature": temperature,
        "context_seq_len": context_seq_len,
        "timing": timing,
        "SNP500": metrics,
        "top_mse_performers": top_mse,
        "top_mae_performers": top_mae,
    }


# --------------------------------------------------------------------------- #
# print_evaluation_results                                                     #
# --------------------------------------------------------------------------- #


def print_evaluation_results(results):
    W = 60
    r = results.get("SNP500", {})
    n = r.get("sample_count", 0)

    print()
    print("=" * W)
    print(f"  Evaluation Results — {results.get('sampler', '?').upper()} sampler")
    print("=" * W)
    print(f"  Steps       : {results.get('n_timesteps')}")
    print(f"  Temperature : {results.get('temperature')}")
    print(f"  Samples     : {n}")

    if n == 0:
        print("  No valid samples — cannot compute metrics.")
        print("=" * W)
        return

    def _row(label, orig, ref):
        imp = (orig - ref) / orig * 100 if orig > 0 else float("nan")
        arrow = "down" if imp > 0 else "up"
        return f"  {label:<22}  orig={orig:.6f}  ref={ref:.6f}  ({imp:+.2f}% {arrow})"

    print()
    print("  -- Raw metrics --")
    print(_row("MSE", r["MSE_original"], r["MSE_refined"]))
    print(_row("MAE", r["MAE_original"], r["MAE_refined"]))

    print()
    print("  -- Z-norm metrics --")
    print(_row("MSE (z-norm)", r["MSE_original_znorm"], r["MSE_refined_znorm"]))
    print(_row("MAE (z-norm)", r["MAE_original_znorm"], r["MAE_refined_znorm"]))

    print()
    print("  -- Robust (Median-MAD) metrics --")
    print(f"  {'Robust median':<22}  {r['robust_median']:.6f}")
    print(f"  {'Robust MAD':<22}  {r['robust_mad']:.6f}")
    print(_row("MSE (robust)", r["MSE_original_robust"], r["MSE_refined_robust"]))
    print(_row("MAE (robust)", r["MAE_original_robust"], r["MAE_refined_robust"]))

    print()
    print("  -- Ranking metrics --")
    for label, key_o, key_r in [
        ("IC", "IC_original", "IC_refined"),
        ("ICIR", "ICIR_original", "ICIR_refined"),
        ("Rank IC", "RankIC_original", "RankIC_refined"),
        ("Rank ICIR", "RankICIR_original", "RankICIR_refined"),
    ]:
        o, ref = r[key_o], r[key_r]
        print(f"  {label:<22}  orig={o:.4f}  ref={ref:.4f}  (delta={ref-o:+.4f})")

    print()
    print("  -- Directional accuracy --")
    print(f"  {'Foundation model':<22}  {r['DirectionalAccuracy_original']:.2f}%")
    print(f"  {'RefineBridge':<22}  {r['DirectionalAccuracy_refined']:.2f}%")

    for metric, key in [("MSE", "top_mse_performers"), ("MAE", "top_mae_performers")]:
        performers = results.get(key, [])
        if not performers:
            continue
        print()
        print(f"  -- Top performers by {metric} improvement --")
        for rank, s in enumerate(performers[:20], 1):
            print(
                f"  {rank:>2}. [{s['entity_id']}] sample {s['global_idx']:>5}"
                f"  MSE {s['mse_improvement']:+.1f}%  MAE {s['mae_improvement']:+.1f}%"
            )

    print()
    print("=" * W)
    print()


# --------------------------------------------------------------------------- #
# plot_sample_predictions                                                      #
# --------------------------------------------------------------------------- #


def plot_sample_predictions(
    model,
    dataset,
    output_dir,
    n_timesteps=50,
    num_samples=5,
    sampler="sde",
    temperature=2.0,
    context_seq_len=252,
    plot_context_points=72,
):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-white")

    model.eval()
    device = next(model.parameters()).device

    rng = np.random.default_rng()
    idxs = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)

    sampler_str = sampler.replace("_", "-")
    fig, axes = plt.subplots(num_samples, 1, figsize=(20, 6 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for plot_i, (ax, idx) in enumerate(zip(axes, idxs)):
        sample = dataset[idx]

        c = sample["context_window"].unsqueeze(0).to(device)
        x0 = sample["ground_truth"].unsqueeze(0).to(device)
        p = sample["prediction"].unsqueeze(0).to(device)

        stats = sample["stats"]
        context_mean = stats["context_mean"]
        context_std = stats["context_std"]
        gt_mean = stats["gt_mean"]
        gt_std = stats["gt_std"]
        pred_mean = stats["pred_mean"]
        pred_std = stats["pred_std"]

        try:
            refined = generate_prediction(
                model, c, p, n_timesteps, sampler, temperature, context_seq_len
            )
            if refined.dim() == 4:
                refined = refined[:, -1]
            refined = refined.cpu().numpy().squeeze()
        except Exception as e:
            print(f"  [warn] Error generating prediction for sample {idx}: {e}")
            continue

        c_np = c.cpu().numpy().squeeze() * context_std + context_mean
        gt_np = x0.cpu().numpy().squeeze() * gt_std + gt_mean
        p_np = p.cpu().numpy().squeeze() * pred_std + pred_mean
        refined = refined * pred_std + pred_mean

        c_plot = (
            c_np[-plot_context_points:] if len(c_np) > plot_context_points else c_np
        )

        ax.plot(
            range(-len(c_plot), 0),
            c_plot,
            "k-",
            label=f"Context (last {len(c_plot)} pts)",
            linewidth=3,
        )
        ax.plot(range(len(gt_np)), gt_np, "k-", label="Ground Truth", linewidth=3)
        ax.plot(
            range(len(p_np)), p_np, "r--", label="Foundation Prediction", linewidth=3
        )
        ax.plot(
            range(len(refined)),
            refined,
            "g-",
            label=f"Refined ({sampler_str})",
            linewidth=3,
        )
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=2)

        ax.set_title(
            f"Entity: {sample['entity_id']}, Variable: {sample['variable']}, Sample {idx}",
            fontsize=25,
            fontweight="bold",
            pad=10,
        )
        ax.legend(fontsize=16, frameon=True)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(2)
        ax.tick_params(axis="x", labelsize=16, pad=8)
        ax.tick_params(axis="y", labelsize=16, pad=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value", fontsize=20, fontweight="bold", labelpad=10)
        if plot_i == num_samples - 1:
            ax.set_xlabel("Time Steps", fontsize=20, fontweight="bold", labelpad=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"sample_predictions_{sampler_str}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


# --------------------------------------------------------------------------- #
# plot_top_performing_samples                                                  #
# --------------------------------------------------------------------------- #


def plot_top_performing_samples(
    model,
    dataset,
    top_performers,
    output_dir,
    n_timesteps=1,
    sampler="sde",
    temperature=0.1,
    context_seq_len=252,
    plot_type="mse",
    plot_context_points=63,
):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-white")

    model.eval()
    device = next(model.parameters()).device

    sampler_str = sampler.replace("_", "-")
    top_performers = top_performers[:30]
    num_samples = len(top_performers)

    fig_all = plt.figure(figsize=(16, 6 * num_samples))

    for plot_idx, sample_info in enumerate(top_performers):
        try:
            idx = sample_info["global_idx"]
            sample = dataset[idx]

            if sample["id"] != sample_info["sample_id"]:
                print(f"  [warn] ID mismatch at index {idx}, skipping")
                continue

            c = sample["context_window"].unsqueeze(0).to(device)
            x0 = sample["ground_truth"].unsqueeze(0).to(device)
            p = sample["prediction"].unsqueeze(0).to(device)

            stats = sample["stats"]
            context_mean = stats["context_mean"]
            context_std = stats["context_std"]
            gt_mean = stats["gt_mean"]
            gt_std = stats["gt_std"]
            pred_mean = stats["pred_mean"]
            pred_std = stats["pred_std"]

            refined = generate_prediction(
                model, c, p, n_timesteps, sampler, temperature, context_seq_len
            )
            if refined.dim() == 4:
                refined = refined[:, -1]
            refined = refined.cpu().numpy().squeeze()

            c_np = c.cpu().numpy().squeeze() * context_std + context_mean
            gt_np = x0.cpu().numpy().squeeze() * gt_std + gt_mean
            p_np = p.cpu().numpy().squeeze() * pred_std + pred_mean
            refined = refined * pred_std + pred_mean

            c_plot = (
                c_np[-plot_context_points:] if len(c_np) > plot_context_points else c_np
            )
            mse_imp = sample_info["mse_improvement"]
            mae_imp = sample_info["mae_improvement"]

            # combined figure
            plt.figure(fig_all.number)
            ax = plt.subplot(num_samples, 1, plot_idx + 1)
            ax.plot(
                range(-len(c_plot), 1), list(c_plot) + [gt_np[0]], "k-", linewidth=3.5
            )
            ax.plot(range(len(gt_np)), gt_np, "k-", label="Ground Truth", linewidth=3.5)
            ax.plot(
                range(len(p_np)),
                p_np,
                "r--",
                label="Foundation Prediction",
                linewidth=3.5,
            )
            ax.plot(
                range(len(refined)),
                refined,
                "#4daf4a",
                label="Refined Prediction",
                linewidth=3.5,
            )
            ax.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=2)
            ax.set_title(
                f"#{plot_idx+1}  Sample {idx}  |  MSE {mse_imp:.1f}% down  MAE {mae_imp:.1f}% down",
                fontsize=22,
                fontweight="bold",
                pad=10,
            )
            ax.legend(loc="best", fontsize=16, frameon=True)
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(2)
            ax.tick_params(axis="x", labelsize=16, pad=8)
            ax.tick_params(axis="y", labelsize=16, pad=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Price", fontsize=20, fontweight="bold", labelpad=10)
            if plot_idx == num_samples - 1:
                ax.set_xlabel("Time Steps", fontsize=20, fontweight="bold", labelpad=10)

            # individual figure
            fig_ind, ax_ind = plt.subplots(figsize=(16, 8))
            ax_ind.plot(
                range(-len(c_plot), 1), list(c_plot) + [gt_np[0]], "k-", linewidth=3.5
            )
            ax_ind.plot(
                range(len(gt_np)), gt_np, "k-", label="Ground Truth", linewidth=3.5
            )
            ax_ind.plot(
                range(len(p_np)),
                p_np,
                "r--",
                label="Foundation Prediction",
                linewidth=3.5,
            )
            ax_ind.plot(
                range(len(refined)),
                refined,
                "#4daf4a",
                label="Refined Prediction",
                linewidth=3.5,
            )
            ax_ind.axvline(0, color="gray", linestyle="--", alpha=0.5, linewidth=2)
            ax_ind.set_title(
                f"Sample {idx}\nMSE Improvement: {mse_imp:.2f}%  |  MAE Improvement: {mae_imp:.2f}%",
                fontsize=25,
                fontweight="bold",
                pad=15,
            )
            ax_ind.legend(loc="best", fontsize=20, frameon=True)
            for spine in ax_ind.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(2)
            ax_ind.tick_params(axis="x", labelsize=20, pad=10)
            ax_ind.tick_params(axis="y", labelsize=20, pad=10)
            ax_ind.grid(True, alpha=0.3)
            ax_ind.set_xlabel("Time Steps", fontsize=20, fontweight="bold", labelpad=10)
            ax_ind.set_ylabel("Returns", fontsize=20, fontweight="bold", labelpad=10)
            plt.tight_layout()

            ind_path = os.path.join(
                output_dir, f"top_{plot_type}_{plot_idx+1}_SNP500_{sampler_str}.png"
            )
            fig_ind.savefig(ind_path, dpi=150, bbox_inches="tight")
            plt.close(fig_ind)

            print(
                f"  [{plot_idx+1}/{num_samples}] Sample {idx}  MSE {mse_imp:.1f}%  MAE {mae_imp:.1f}%"
            )

        except Exception as e:
            import traceback

            print(f"  [warn] Error plotting sample {plot_idx+1}: {e}")
            traceback.print_exc()
            continue

    # save combined figure
    plt.figure(fig_all.number)
    plt.tight_layout()
    combined_path = os.path.join(
        output_dir, f"top20_{plot_type}_performers_SNP500_{sampler_str}.png"
    )
    fig_all.savefig(combined_path, dpi=150, bbox_inches="tight")
    print(f"  Saved combined : {combined_path}")
    plt.close(fig_all)

    # improvement summary bar chart
    mse_imps = [s["mse_improvement"] for s in top_performers]
    mae_imps = [s["mae_improvement"] for s in top_performers]
    ranks = range(1, len(top_performers) + 1)

    fig_sum, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.bar(ranks, mse_imps, color="green", alpha=0.7)
    ax1.set_xlabel("Sample Rank", fontsize=25, fontweight="bold", labelpad=15)
    ax1.set_ylabel("MSE Improvement (%)", fontsize=25, fontweight="bold", labelpad=15)
    ax1.set_title(
        f"Top {len(top_performers)} MSE Improvements — S&P 500",
        fontsize=30,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(True, alpha=0.3)
    for spine in ax1.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
    ax1.tick_params(axis="x", labelsize=20, pad=10)
    ax1.tick_params(axis="y", labelsize=20, pad=10)

    ax2.bar(ranks, mae_imps, color="blue", alpha=0.7)
    ax2.set_xlabel("Sample Rank", fontsize=25, fontweight="bold", labelpad=15)
    ax2.set_ylabel("MAE Improvement (%)", fontsize=25, fontweight="bold", labelpad=15)
    ax2.set_title(
        f"Top {len(top_performers)} MAE Improvements — S&P 500",
        fontsize=30,
        fontweight="bold",
        pad=15,
    )
    ax2.grid(True, alpha=0.3)
    for spine in ax2.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(2)
    ax2.tick_params(axis="x", labelsize=20, pad=10)
    ax2.tick_params(axis="y", labelsize=20, pad=10)

    plt.tight_layout()
    summary_path = os.path.join(
        output_dir, f"top20_{plot_type}_improvement_summary_SNP500_{sampler_str}.png"
    )
    fig_sum.savefig(summary_path, dpi=150, bbox_inches="tight")
    print(f"  Saved summary  : {summary_path}")
    plt.close(fig_sum)


# --------------------------------------------------------------------------- #
# run_evaluation_with_plots                                                    #
# --------------------------------------------------------------------------- #


def run_evaluation_with_plots(
    model,
    test_dataset,
    device,
    output_dir,
    n_timesteps=1,
    sampler="sde",
    temperature=100000,
    context_seq_len=63,
):
    print(f"\nEvaluating with {sampler.upper()} sampler on test set ...")

    eval_results = evaluate_model(
        model,
        test_dataset,
        device,
        n_timesteps=n_timesteps,
        sampler=sampler,
        temperature=temperature,
        context_seq_len=context_seq_len,
    )

    print_evaluation_results(eval_results)

    for metric, key in [("mse", "top_mse_performers"), ("mae", "top_mae_performers")]:
        performers = eval_results.get(key, [])
        if performers:
            print(f"\nPlotting top {metric.upper()} performers ...")
            plot_top_performing_samples(
                model=model,
                dataset=test_dataset,
                top_performers=performers,
                output_dir=output_dir,
                n_timesteps=n_timesteps,
                sampler=sampler,
                temperature=temperature,
                context_seq_len=context_seq_len,
                plot_type=metric,
            )

    return eval_results
