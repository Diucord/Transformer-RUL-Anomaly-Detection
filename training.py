"""
training.py

Clean & stable training module for AdvancedAutoInformerModel.
Supports:
- Checkpoint resume
- Timestamped final model saving (handled in main)
- CosineAnnealingWarmRestarts scheduler
- AdamW optimizer
- Full logging (JSON + CSV)
"""

import os
import csv
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ====================================================================
# Training Function (REWRITTEN & FIXED)
# ====================================================================
def train_model(
    train_loader,
    model,
    num_epochs: int,
    device,
    lr: float,
    weight_decay: float,
    save_path: str = None,
    log_dir: str = None,
    save_every: int = 5
):
    """
    Train the model using next-step prediction (reconstruction style)
    """

    # ------------------------------------------------------------
    # 1) Prepare optimizer / loss / scheduler
    # ------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=1e-6
    )

    # ------------------------------------------------------------
    # 2) Resume checkpoint if exist
    # ------------------------------------------------------------
    start_epoch = 1
    trained = False

    if save_path is not None and os.path.exists(save_path):
        print(f"[Checkpoint] Loading state from {save_path} ...")
        checkpoint = torch.load(save_path, map_location=device)

        model.load_state_dict(checkpoint["model_state"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except Exception as e:
            print(f"[Checkpoint Warning] Optimizer state load failed (ignored): {e}")

        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1

        print(f"[Checkpoint] Resuming training at epoch {start_epoch}")

    # ------------------------------------------------------------
    # 3) Training preparation
    # ------------------------------------------------------------
    model.train()
    epoch_losses = []
    lr_history = []

    print("\n[Training Start]")
    print(f" - Total epochs: {num_epochs}")
    print(f" - Start epoch : {start_epoch}")
    print(f" - Device      : {device}\n")

    # ------------------------------------------------------------
    # 4) Training Loop
    # ------------------------------------------------------------
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_loss = 0.0
        batch_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):

            seq = batch[0]
            if isinstance(seq, np.ndarray):
                seq = torch.from_numpy(seq)
            seq = seq.float().to(device)  # (B, T, C)

            # ----------------------------
            # CORRECT TARGET SHAPE (B, C)
            # ----------------------------
            target = seq[:, -1, :]        # 마지막 시점 ground truth

            pred = model(seq)             # (B, C)

            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            trained = True


        # average loss
        avg_loss = epoch_loss / max(batch_count, 1)
        epoch_losses.append(avg_loss)

        # record LR
        lr_history.append(optimizer.param_groups[0]["lr"])

        print(f"[Epoch {epoch}] Loss: {avg_loss:.5f}")

        # scheduler step
        scheduler.step(epoch)

        # --------------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------------
        if save_path is not None and epoch % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
            torch.save(ckpt, save_path)
            print(f"[Checkpoint Saved] {save_path}")

    print("\n[Training Completed]\n")

    # ------------------------------------------------------------
    # 5) Logging
    # ------------------------------------------------------------
    logs = {
        "start_epoch": start_epoch,
        "end_epoch": num_epochs,
        "epoch_losses": epoch_losses,
        "lr_history": lr_history,
        "trained": trained
    }

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)

        # JSON log
        json_path = os.path.join(log_dir, "training_log.json")
        with open(json_path, "w") as f:
            json.dump(logs, f, indent=2)

        # CSV loss log
        csv_path = os.path.join(log_dir, "epoch_losses.csv")
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch", "loss", "lr"])

            for i, loss_val in enumerate(epoch_losses, start=start_epoch):
                lr_val = lr_history[i - start_epoch]
                writer.writerow([i, loss_val, lr_val])

        print(f"[Training Logs Saved] {json_path} and {csv_path}")

    return model, trained, logs