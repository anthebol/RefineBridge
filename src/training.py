import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# EnhancedCheckpointTracker                                                   #
# --------------------------------------------------------------------------- #


class EnhancedCheckpointTracker:
    """
    Manages model checkpointing and early stopping during training.

    Saves three types of checkpoints:
      - Best model (overwritten each time val loss improves)
      - Best-epoch stamped  (epoch + loss in filename, keeps top max_best_models)
      - Periodic (every checkpoint_interval epochs)
      - Early-stop (when patience is exceeded and min_epochs is reached)
    """

    def __init__(
        self,
        save_dir="./output",
        model_name="sb_model",
        patience=15,
        min_delta=0.001,
        checkpoint_interval=10,
        max_best_models=5,
        min_epochs=10000,
    ):
        self.save_dir = save_dir
        self.model_name = model_name
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_interval = checkpoint_interval
        self.max_best_models = max_best_models
        self.min_epochs = min_epochs

        os.makedirs(save_dir, exist_ok=True)

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop_triggered = False
        self.best_models = []  # [(loss, epoch), ...]
        self.early_stop_epoch = None
        self.early_stop_saved = False

    def __call__(
        self,
        val_loss,
        model,
        optimizer,
        train_loss,
        epoch,
        train_losses=None,
        val_losses=None,
    ):
        """
        Called once per epoch. Handles checkpointing and early stopping.

        Returns:
            improved (bool):     True if val_loss is a new best
            stop_training (bool): True if early stopping should fire
        """
        improved = False
        stop_training = False

        # Periodic checkpoint
        if (epoch + 1) % self.checkpoint_interval == 0:
            periodic_path = os.path.join(
                self.save_dir, f"{self.model_name}_epoch_{epoch+1}_V2.pt"
            )
            self._save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                periodic_path,
                train_losses,
                val_losses,
            )
            print(f"Periodic checkpoint saved at epoch {epoch+1}")

        if val_loss < self.best_loss - self.min_delta:
            # New best model
            self.best_loss = val_loss
            self.counter = 0
            self.early_stop_triggered = False

            self.best_models.append((val_loss, epoch))
            self.best_models.sort(key=lambda x: x[0])
            self.best_models = self.best_models[: self.max_best_models]

            # Overwrite rolling best
            best_path = os.path.join(self.save_dir, f"{self.model_name}_best_V2.pt")
            self._save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                best_path,
                train_losses,
                val_losses,
            )

            # Stamped copy for tracking
            best_epoch_path = os.path.join(
                self.save_dir,
                f"{self.model_name}_best_E{epoch+1}_L{val_loss:.6f}_V2.pt",
            )
            self._save_checkpoint(
                model,
                optimizer,
                epoch,
                train_loss,
                val_loss,
                best_epoch_path,
                train_losses,
                val_losses,
            )

            print(f"Saved best model with validation loss {val_loss:.6f}")
            improved = True

        else:
            self.counter += 1
            if self.counter >= self.patience and epoch >= self.min_epochs - 1:
                if not self.early_stop_triggered:
                    self.early_stop_triggered = True
                    self.early_stop_epoch = epoch + 1

                    es_path = os.path.join(
                        self.save_dir,
                        f"{self.model_name}_early_stop_E{epoch+1}_L{val_loss:.6f}_V2.pt",
                    )
                    self._save_checkpoint(
                        model,
                        optimizer,
                        epoch,
                        train_loss,
                        val_loss,
                        es_path,
                        train_losses,
                        val_losses,
                    )

                    stop_training = True
                    print(f"Early stopping triggered at epoch {epoch+1}!")

        return improved, stop_training

    def save_final_model(
        self,
        model,
        optimizer,
        epoch,
        train_loss,
        val_loss,
        train_losses=None,
        val_losses=None,
    ):
        """Save the final model at the end of training."""
        final_path = os.path.join(
            self.save_dir, f"{self.model_name}_final_E{epoch+1}_V2.pt"
        )
        self._save_checkpoint(
            model,
            optimizer,
            epoch,
            train_loss,
            val_loss,
            final_path,
            train_losses,
            val_losses,
        )
        print(f"Final model saved after {epoch+1} epochs")

    def _save_checkpoint(
        self,
        model,
        optimizer,
        epoch,
        train_loss,
        val_loss,
        filepath,
        train_losses=None,
        val_losses=None,
    ):
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if train_losses is not None:
            checkpoint["train_losses"] = train_losses
        if val_losses is not None:
            checkpoint["val_losses"] = val_losses
        torch.save(checkpoint, filepath)

    def get_summary(self):
        """Return a summary of saved checkpoints."""
        return {
            "best_loss": self.best_loss,
            "best_models": self.best_models,
            "early_stop_triggered": self.early_stop_triggered,
            "early_stop_epoch": self.early_stop_epoch,
        }


# --------------------------------------------------------------------------- #
# train_model                                                                  #
# --------------------------------------------------------------------------- #


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    device,
    num_epochs=100,
    accumulation_steps=1,
    output_dir="./output",
    use_early_stopping=True,
    context_seq_len=252,
):
    """
    Train RefineBridge with validation, checkpointing, and loss plotting.

    Features:
      - 5-epoch linear LR warmup from lr/10 → lr
      - Gradient clipping (max_norm=5.0)
      - NaN/Inf guards on inputs and gradients
      - Sliding window context truncation to context_seq_len
      - Gradient accumulation
      - Early stopping via EnhancedCheckpointTracker

    Args:
        model:             RefineBridge instance
        train_dataloader:  DataLoader for training set
        val_dataloader:    DataLoader for validation set
        optimizer:         PyTorch optimizer
        scheduler:         LR scheduler (called once per epoch)
        device:            Device string
        num_epochs:        Maximum training epochs
        accumulation_steps: Gradient accumulation steps
        output_dir:        Directory for checkpoints and plots
        use_early_stopping: Whether to halt when early stopping fires
        context_seq_len:   Sliding window length for context

    Returns:
        model:        Trained model
        train_losses: List of per-epoch loss dicts
        val_losses:   List of per-epoch loss dicts
    """
    encoder_pretrained = getattr(model, "pre_trained_enc", False)

    train_losses = []
    val_losses = []

    initial_lr = optimizer.param_groups[0]["lr"] / 10
    target_lr = optimizer.param_groups[0]["lr"]
    warmup_epochs = 5

    checkpoint_tracker = EnhancedCheckpointTracker(
        save_dir=output_dir,
        model_name="sb_model",
        patience=10,
        min_delta=0.001,
        checkpoint_interval=500,
        max_best_models=5,
        min_epochs=10000,
    )

    for epoch in range(num_epochs):

        # ------------------------------------------------------------------ #
        # LR warmup                                                           #
        # ------------------------------------------------------------------ #
        if epoch < warmup_epochs:
            warmup_lr = initial_lr + (target_lr - initial_lr) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        # ------------------------------------------------------------------ #
        # Training phase                                                      #
        # ------------------------------------------------------------------ #
        model.train()
        epoch_train_losses = {"enc_loss": 0.0, "dec_loss": 0.0, "total_loss": 0.0}
        batch_count = 0

        train_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs} (Train) LR:{scheduler.get_last_lr()[0]:.2e}",
            leave=False,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(train_bar):
            try:
                context = batch["context_window"].to(device, non_blocking=True)
                ground_truth = batch["ground_truth"].to(device, non_blocking=True)
                pred = batch["prediction"].to(device, non_blocking=True)

                # Sliding window: keep last context_seq_len steps
                if context.shape[1] > context_seq_len:
                    context = context[:, -context_seq_len:]

                # Ensure 3D inputs
                if context.dim() == 2:
                    context = context.unsqueeze(-1)
                if ground_truth.dim() == 2:
                    ground_truth = ground_truth.unsqueeze(-1)
                if pred.dim() == 2:
                    pred = pred.unsqueeze(-1)

                # Skip batches with corrupt inputs
                if (
                    torch.isnan(context).any()
                    or torch.isinf(context).any()
                    or torch.isnan(ground_truth).any()
                    or torch.isinf(ground_truth).any()
                    or torch.isnan(pred).any()
                    or torch.isinf(pred).any()
                ):
                    print("Warning: NaN or Inf in input data, skipping batch")
                    continue

                mask = torch.ones(ground_truth.shape[0], 1, device=device)

                try:
                    enc_loss, dec_loss = model.compute_loss(
                        context, pred, ground_truth, mask
                    )
                    total_loss = dec_loss / accumulation_steps
                    total_loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                    # Skip NaN/Inf gradients
                    has_nan_grad = any(
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                    if has_nan_grad:
                        print("Warning: NaN or Inf gradient, skipping update")
                        continue

                    if (batch_idx + 1) % accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(train_dataloader):
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    epoch_train_losses["enc_loss"] = 0
                    epoch_train_losses["dec_loss"] += dec_loss.item()
                    epoch_train_losses["total_loss"] += (
                        total_loss.item() * accumulation_steps
                    )

                    batch_count += 1
                    encoder_pretrained = True

                    train_bar.set_postfix(
                        dec=f"{dec_loss.item():.4f}",
                        total=f"{total_loss.item() * accumulation_steps:.4f}",
                    )

                except Exception as e:
                    import traceback

                    print(f"Warning: Error in training step: {e}")
                    traceback.print_exc()
                    continue

            except Exception as e:
                print(f"Warning: Error processing batch: {e}")
                continue

        # Average training losses
        if batch_count > 0:
            for k in epoch_train_losses:
                epoch_train_losses[k] /= batch_count
            train_losses.append(epoch_train_losses)

        scheduler.step()

        # ------------------------------------------------------------------ #
        # Validation phase                                                    #
        # ------------------------------------------------------------------ #
        model.eval()
        epoch_val_losses = {"enc_loss": 0.0, "dec_loss": 0.0, "total_loss": 0.0}
        val_batch_count = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_dataloader,
                desc=f"Epoch {epoch+1}/{num_epochs} (Val)",
                leave=False,
                dynamic_ncols=True,
            )

            for batch_idx, batch in enumerate(val_bar):
                try:
                    context = batch["context_window"].to(device, non_blocking=True)
                    ground_truth = batch["ground_truth"].to(device, non_blocking=True)
                    pred = batch["prediction"].to(device, non_blocking=True)

                    if context.shape[1] > context_seq_len:
                        context = context[:, -context_seq_len:]

                    if context.dim() == 2:
                        context = context.unsqueeze(-1)
                    if ground_truth.dim() == 2:
                        ground_truth = ground_truth.unsqueeze(-1)
                    if pred.dim() == 2:
                        pred = pred.unsqueeze(-1)

                    if (
                        torch.isnan(context).any()
                        or torch.isinf(context).any()
                        or torch.isnan(ground_truth).any()
                        or torch.isinf(ground_truth).any()
                        or torch.isnan(pred).any()
                        or torch.isinf(pred).any()
                    ):
                        continue

                    mask = torch.ones(ground_truth.shape[0], 1, device=device)

                    enc_loss, dec_loss = model.compute_loss(
                        context, pred, ground_truth, mask
                    )

                    total_loss = dec_loss if encoder_pretrained else enc_loss + dec_loss

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        tqdm.write("Warning: NaN or Inf loss, skipping batch")
                        continue

                    epoch_val_losses["enc_loss"] += (
                        enc_loss if not encoder_pretrained else 0
                    )
                    epoch_val_losses["dec_loss"] += dec_loss.item()
                    epoch_val_losses["total_loss"] += total_loss.item()

                    val_batch_count += 1
                    encoder_pretrained = True

                    val_bar.set_postfix(
                        val_dec=f"{dec_loss.item():.4f}",
                        val_total=f"{total_loss.item():.4f}",
                    )

                except Exception as e:
                    print(f"Warning: Error in validation step: {e}")
                    continue

        # ------------------------------------------------------------------ #
        # Post-epoch bookkeeping                                              #
        # ------------------------------------------------------------------ #
        if val_batch_count > 0:
            for k in epoch_val_losses:
                epoch_val_losses[k] /= val_batch_count
            val_losses.append(epoch_val_losses)

            print(
                f"Epoch {epoch+1}/{num_epochs} — "
                f"Train: {epoch_train_losses['total_loss']:.6f}, "
                f"Val: {epoch_val_losses['total_loss']:.6f}"
            )

            improved, stop_training = checkpoint_tracker(
                epoch_val_losses["total_loss"],
                model,
                optimizer,
                epoch_train_losses["total_loss"],
                epoch,
                train_losses,
                val_losses,
            )

            if stop_training and use_early_stopping:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                print(f"Best validation loss: {checkpoint_tracker.best_loss:.6f}")
                break
            elif stop_training and not use_early_stopping:
                print(
                    f"Early stopping condition met at epoch {epoch+1}, "
                    "but training continues as early stopping is disabled."
                )

            plot_training_curves(train_losses, val_losses, output_dir)

        else:
            print(f"Warning: No valid batches in validation for epoch {epoch+1}")

    # Save final model
    if val_batch_count > 0:
        checkpoint_tracker.save_final_model(
            model,
            optimizer,
            epoch,
            epoch_train_losses["total_loss"],
            epoch_val_losses["total_loss"],
            train_losses,
            val_losses,
        )

    summary = checkpoint_tracker.get_summary()
    print("\nTraining Summary:")
    print(f"Best validation loss: {summary['best_loss']:.6f}")
    print("\nTop best models (loss, epoch):")
    for loss, epoch_num in summary["best_models"]:
        print(f"  Epoch {epoch_num+1}: {loss:.6f}")
    if summary["early_stop_triggered"]:
        print(f"\nEarly stopping triggered at epoch {summary['early_stop_epoch']}")

    return model, train_losses, val_losses


# --------------------------------------------------------------------------- #
# generate_prediction                                                          #
# --------------------------------------------------------------------------- #


def generate_prediction(
    model,
    sample_context,
    sample_prediction,
    n_timesteps=50,
    sampler="ode",
    temperature=1.0,
    context_seq_len=252,
):
    """
    Generate a refined prediction for a single sample.

    Used during evaluation and visualisation — not called during training.

    Args:
        model:             Trained RefineBridge model
        sample_context:    Context tensor [B, seq_len, features]
        sample_prediction: Foundation model prediction [B, seq_len, features]
        n_timesteps:       Number of reverse diffusion steps
        sampler:           "ode", "sde", "pc_ode", "pc_sde"
        temperature:       Sampling temperature (higher = more diversity)
        context_seq_len:   Max context window; longer inputs are truncated

    Returns:
        refined: Full reverse diffusion trajectory [B, n_timesteps+1, seq_len, features]
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        context = sample_context.to(device)
        prediction = sample_prediction.to(device)

        if context.shape[1] > context_seq_len:
            context = context[:, -context_seq_len:]

        if context.dim() == 2:
            context = context.unsqueeze(-1)
        if prediction.dim() == 2:
            prediction = prediction.unsqueeze(-1)

        batch_size = prediction.shape[0]
        mask = torch.ones(batch_size, 1, device=device)

        sampler_map = {
            "sde": "sde",
            "ode": "ode",
            "pc_sde": "pc_sde",
            "pc_ode": "pc_ode",
            "pf_ode_euler": "ode",
            "sde_euler": "sde",
        }
        internal_sampler = sampler_map.get(sampler, "ode")

        _, refined, _ = model(
            context=context,
            foundation_pred=prediction,
            mask=mask,
            n_timesteps=n_timesteps,
            sampler=internal_sampler,
            temperature=temperature,
        )

    return refined


# --------------------------------------------------------------------------- #
# plot_training_curves                                                         #
# --------------------------------------------------------------------------- #


def plot_training_curves(train_losses, val_losses, output_dir):
    """
    Save training/validation loss curves to output_dir.

    Produces two files:
      training_validation_losses_V2.png      — linear scale
      training_validation_losses_log_V2.png  — log scale
    """

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def smooth(y, box_pts=3):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode="same")
        y_smooth[: box_pts // 2] = y[: box_pts // 2]
        y_smooth[-box_pts // 2 :] = y[-box_pts // 2 :]
        return y_smooth

    epochs = np.arange(1, len(train_losses) + 1)

    train_total = np.array([to_numpy(l["total_loss"]) for l in train_losses])
    train_dec = np.array([to_numpy(l["dec_loss"]) for l in train_losses])
    train_enc = np.array([to_numpy(l["enc_loss"]) for l in train_losses])

    val_total = np.array([to_numpy(l["total_loss"]) for l in val_losses])
    val_dec = np.array([to_numpy(l["dec_loss"]) for l in val_losses])
    val_enc = np.array([to_numpy(l["enc_loss"]) for l in val_losses])

    plot_encoder = not np.allclose(train_enc, 0) and not np.allclose(val_enc, 0)

    # ---- Linear scale ----
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_total, "b-", label="Train Total Loss")
    plt.plot(epochs, val_total, "r-", label="Val Total Loss")
    if len(epochs) > 5:
        plt.plot(
            epochs,
            smooth(train_total),
            "b--",
            linewidth=2,
            label="Train Total (Smoothed)",
        )
        plt.plot(
            epochs, smooth(val_total), "r--", linewidth=2, label="Val Total (Smoothed)"
        )
    plt.title("Total Losses")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    if plot_encoder:
        plt.plot(epochs, train_enc, "b-", alpha=0.7, label="Train Encoder")
        plt.plot(epochs, val_enc, "b--", alpha=0.7, label="Val Encoder")
    plt.plot(epochs, train_dec, "g-", alpha=0.7, label="Train Decoder")
    plt.plot(epochs, val_dec, "g--", alpha=0.7, label="Val Decoder")
    plt.title("Component Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_validation_losses_V2.png"))
    plt.close()

    # ---- Log scale ----
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.semilogy(epochs, train_total, "b-", label="Train Total Loss")
    plt.semilogy(epochs, val_total, "r-", label="Val Total Loss")
    if len(epochs) > 5:
        plt.semilogy(
            epochs,
            smooth(train_total),
            "b--",
            linewidth=2,
            label="Train Total (Smoothed)",
        )
        plt.semilogy(
            epochs, smooth(val_total), "r--", linewidth=2, label="Val Total (Smoothed)"
        )
    plt.title("Total Losses (Log Scale)")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.semilogy(epochs, train_enc, "b-", alpha=0.7, label="Train Encoder")
    plt.semilogy(epochs, train_dec, "g-", alpha=0.7, label="Train Decoder")
    plt.semilogy(epochs, val_enc, "b--", alpha=0.7, label="Val Encoder")
    plt.semilogy(epochs, val_dec, "g--", alpha=0.7, label="Val Decoder")
    plt.title("Component Losses (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_validation_losses_log_V2.png"))
    plt.close()
