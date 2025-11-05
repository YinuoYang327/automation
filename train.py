import argparse

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn

from data import SequencePairDataset, generate_synthetic_data
from models import SequencePairRegressor
from utils import train_one_epoch, evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--drug_vocab_size", type=int, default=30)
    parser.add_argument("--target_vocab_size", type=int, default=26)
    parser.add_argument("--drug_seq_len", type=int, default=100)
    parser.add_argument("--target_seq_len", type=int, default=250)
    parser.add_argument("--hidden_dim", type=int, nargs="+", default=[256, 128])
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare data (synthetic for demo)
    X_drug, X_target, y = generate_synthetic_data(
        n_samples=1200,
        drug_seq_len=args.drug_seq_len,
        target_seq_len=args.target_seq_len,
        drug_vocab_size=args.drug_vocab_size,
        target_vocab_size=args.target_vocab_size,
    )

    dataset = SequencePairDataset(X_drug, X_target, y)

    # 70% train, 15% val, 15% test
    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    # 2. Build model
    model = SequencePairRegressor(
        drug_vocab_size=args.drug_vocab_size,
        target_vocab_size=args.target_vocab_size,
        hidden_dims=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None

    # 3. Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_mse={val_metrics['mse']:.4f} | "
            f"val_r2={val_metrics['r2']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = model.state_dict()

    # 4. Test with best model
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test MSE={test_metrics['mse']:.4f} | "
        f"Test R^2={test_metrics['r2']:.4f}"
    )

    # 5. Save model
    torch.save(model.state_dict(), "sequence_pair_regressor.pt")
    print("Saved best model to sequence_pair_regressor.pt")


if __name__ == "__main__":
    main()
