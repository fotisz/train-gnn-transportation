from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import MetrLaDataModule
from src.model import GraphGRUForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GNN baseline on METR-LA.")
    parser.add_argument("--data", type=Path, default=Path("data/metr-la.h5"))
    parser.add_argument("--adj", type=Path, default=Path("data/sensor_graph/adj_mx.pkl"))
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--save-path", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--resume-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=Path("runs/metrics.csv"))
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = torch.nn.functional.l1_loss(pred, y, reduction="sum")
            total_loss += loss.item()
            total_count += y.numel()

    return total_loss / max(total_count, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    dm = MetrLaDataModule(
        data_path=args.data,
        adj_path=args.adj,
        input_steps=args.input_steps,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )
    dm.setup()

    model = GraphGRUForecaster(
        num_nodes=dm.num_nodes,
        input_dim=1,
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        adjacency=dm.normalized_adjacency,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_mae = float("inf")
    start_epoch = 1

    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_mae = float(checkpoint["val_mae"])
        print(
            f"resumed_from={args.resume_path} start_epoch={start_epoch} "
            f"best_val_mae={best_val_mae:.4f}",
            flush=True,
        )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_mode = "a" if args.append_metrics and args.metrics_path.exists() else "w"

    with args.metrics_path.open(metrics_mode, newline="", encoding="utf-8") as metrics_file:
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "train_mae", "val_mae", "best_val_mae"],
        )
        if metrics_mode == "w":
            metrics_writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running_loss = 0.0
            running_count = 0

            progress = tqdm(
                dm.train_loader,
                desc=f"epoch {epoch:03d}",
                leave=False,
                disable=args.quiet_progress,
            )
            for batch_index, (x, y) in enumerate(progress):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break
                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = torch.nn.functional.l1_loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                running_loss += loss.item() * y.numel()
                running_count += y.numel()
                progress.set_postfix(train_mae=running_loss / max(running_count, 1))

            train_mae = running_loss / max(running_count, 1)
            val_mae = evaluate(model, dm.val_loader, device, max_batches=args.max_eval_batches)
            best_val_mae = min(best_val_mae, val_mae)
            print(
                f"epoch={epoch:03d} train_mae={train_mae:.4f} "
                f"val_mae={val_mae:.4f} best_val_mae={best_val_mae:.4f}",
                flush=True,
            )

            metrics_writer.writerow(
                {
                    "epoch": epoch,
                    "train_mae": f"{train_mae:.6f}",
                    "val_mae": f"{val_mae:.6f}",
                    "best_val_mae": f"{best_val_mae:.6f}",
                }
            )
            metrics_file.flush()

            if val_mae <= best_val_mae:
                args.save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_mae": val_mae,
                        "args": vars(args),
                    },
                    args.save_path,
                )

    test_mae = evaluate(model, dm.test_loader, device, max_batches=args.max_eval_batches)
    print(f"test_mae={test_mae:.4f}")
    print(f"best_checkpoint={args.save_path}")


if __name__ == "__main__":
    main()
