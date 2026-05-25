"""
Training loop for BaselineLSTM and CausalLSTM.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import BaselineLSTM, CausalLSTM


def train_epoch(model, loader: DataLoader, optimizer, device: str) -> tuple[float, float]:
    model.train()
    total_mse = total_pen = 0.0
    mse_fn = nn.MSELoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if isinstance(model, CausalLSTM):
            y_hat, x_req = model(x)
            mse = mse_fn(y_hat, y)
            pen = model.causal_penalty(y_hat, x_req)
            loss = mse + pen
            total_pen += pen.item()
        else:
            y_hat = model(x)
            mse = mse_fn(y_hat, y)
            loss = mse

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_mse += mse.item()

    n = len(loader)
    return total_mse / n, total_pen / n


@torch.no_grad()
def eval_epoch(model, loader: DataLoader, device: str) -> float:
    model.eval()
    total_mse = 0.0
    mse_fn = nn.MSELoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if isinstance(model, CausalLSTM):
            y_hat, _ = model(x)
        else:
            y_hat = model(x)
        total_mse += mse_fn(y_hat, y).item()
    return total_mse / len(loader)


def fit(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    verbose: bool = True,
    patience: int = None,
) -> dict:
    """
    Train model with optional early stopping.

    Args:
        weight_decay: L2 regularization on parameters (passed to Adam).
                      Use this for the L2-LSTM baseline.

    Returns:
        dict with 'train_mse', 'val_mse', 'train_pen' history lists
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_mse": [], "val_mse": [], "train_pen": []}

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_mse, tr_pen = train_epoch(model, train_loader, optimizer, device)
        val_mse = eval_epoch(model, val_loader, device)

        history["train_mse"].append(tr_mse)
        history["val_mse"].append(val_mse)
        history["train_pen"].append(tr_pen)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            pen_str = f" | Pen {tr_pen:.2e}" if isinstance(model, CausalLSTM) else ""
            print(f"Ep {epoch:3d}/{epochs} | TrMSE {tr_mse:.4e}{pen_str} | ValMSE {val_mse:.4e}")

        if patience is not None:
            if val_mse < best_val - 1e-7:
                best_val = val_mse
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

    return history


@torch.no_grad()
def predict_one_step(model, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Collect one-step-ahead predictions and targets from a DataLoader."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        if isinstance(model, CausalLSTM):
            y_hat, _ = model(x)
        else:
            y_hat = model(x)
        preds.append(y_hat.cpu().numpy())
        targets.append(y.numpy())
    return np.vstack(preds), np.vstack(targets)
