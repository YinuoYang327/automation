from typing import Dict, Tuple

import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for drug, target, y in dataloader:
        drug = drug.to(device)
        target = target.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(drug, target)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    return running_loss / max(n_samples, 1)


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for drug, target, y in dataloader:
            drug = drug.to(device)
            target = target.to(device)
            y = y.to(device)

            preds = model(drug, target)
            y_true.append(y.cpu())
            y_pred.append(preds.cpu())

    if not y_true:
        return {"mse": float("nan"), "r2": float("nan")}

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"mse": mse, "r2": r2}
