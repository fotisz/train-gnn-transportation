from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.data import MetrLaDataModule
from src.model import GraphGRUForecaster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute explainability-oriented GNN metrics.")
    parser.add_argument("--data", type=Path, default=Path("data/metr-la.h5"))
    parser.add_argument("--adj", type=Path, default=Path("data/sensor_graph/adj_mx.pkl"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/normal_best.pt"))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--input-steps", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/explainability_smoke"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_hdf(path)
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.sort_index()
    if frame.isna().any().any():
        frame = frame.ffill().bfill()
    return frame


def load_raw_adjacency(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        payload: Any = pickle.load(file, encoding="latin1")
    adjacency = (
        payload[2]
        if isinstance(payload, (tuple, list)) and len(payload) >= 3
        else payload
    )
    return np.asarray(adjacency, dtype=np.float32)


def split_target_times(
    frame: pd.DataFrame,
    split: str,
    input_steps: int,
    horizon: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> pd.DatetimeIndex | None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        return None

    train_end = int(len(frame) * train_ratio)
    val_end = int(len(frame) * (train_ratio + val_ratio))
    start = train_end - input_steps - horizon + 1 if split == "val" else val_end - input_steps - horizon + 1
    end = val_end if split == "val" else len(frame)
    split_index = frame.index[start:end]
    target_start = input_steps
    target_stop = len(split_index) - horizon + 1 + input_steps
    return split_index[target_start:target_stop]


def inverse_transform_batch(values: torch.Tensor, scaler: StandardScaler) -> np.ndarray:
    array = values.detach().cpu().squeeze(-1).numpy()
    batch_size, horizon, num_nodes = array.shape
    flat = array.reshape(batch_size * horizon, num_nodes)
    return scaler.inverse_transform(flat).reshape(batch_size, horizon, num_nodes)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_corrcoef(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_frame(args.data)
    values = frame.to_numpy(dtype=np.float32)
    train_end = int(len(values) * 0.7)
    scaler = StandardScaler().fit(values[:train_end])

    dm = MetrLaDataModule(
        data_path=args.data,
        adj_path=args.adj,
        input_steps=args.input_steps,
        horizon=args.horizon,
        batch_size=args.batch_size,
    )
    dm.setup()

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    hidden_dim = args.hidden_dim or int(checkpoint_args.get("hidden_dim", 64))

    model = GraphGRUForecaster(
        num_nodes=dm.num_nodes,
        input_dim=1,
        hidden_dim=hidden_dim,
        horizon=args.horizon,
        adjacency=dm.normalized_adjacency,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = dm.val_loader if args.split == "val" else dm.test_loader
    target_times = split_target_times(frame, args.split, args.input_steps, args.horizon)
    raw_adjacency = load_raw_adjacency(args.adj)
    node_degree = (raw_adjacency > 0).sum(axis=1).astype(np.float32)

    all_abs_errors: list[np.ndarray] = []
    all_signed_errors: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    sample_count = 0

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            x = x.to(device)
            pred = model(x)
            pred_unscaled = inverse_transform_batch(pred, scaler)
            y_unscaled = inverse_transform_batch(y, scaler)

            signed_error = pred_unscaled - y_unscaled
            all_signed_errors.append(signed_error)
            all_abs_errors.append(np.abs(signed_error))
            all_targets.append(y_unscaled)
            sample_count += y.shape[0]

    abs_error = np.concatenate(all_abs_errors, axis=0)
    signed_error = np.concatenate(all_signed_errors, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    per_horizon_mae = abs_error.mean(axis=(0, 2))
    per_horizon_bias = signed_error.mean(axis=(0, 2))
    horizon_rows = [
        {
            "horizon_step": step + 1,
            "minutes_ahead": (step + 1) * 5,
            "mae": f"{per_horizon_mae[step]:.6f}",
            "bias": f"{per_horizon_bias[step]:.6f}",
        }
        for step in range(args.horizon)
    ]
    write_csv(
        args.output_dir / f"{args.split}_per_horizon_mae.csv",
        horizon_rows,
        ["horizon_step", "minutes_ahead", "mae", "bias"],
    )

    per_sensor_mae = abs_error.mean(axis=(0, 1))
    per_sensor_bias = signed_error.mean(axis=(0, 1))
    sensor_rows = [
        {
            "sensor_index": node,
            "degree": int(node_degree[node]),
            "mae": f"{per_sensor_mae[node]:.6f}",
            "bias": f"{per_sensor_bias[node]:.6f}",
        }
        for node in range(dm.num_nodes)
    ]
    write_csv(
        args.output_dir / f"{args.split}_per_sensor_mae.csv",
        sensor_rows,
        ["sensor_index", "degree", "mae", "bias"],
    )

    if target_times is not None:
        used_times = target_times[: sample_count]
        hour_rows = []
        for hour in range(24):
            mask = np.array(used_times.hour == hour)
            if mask.any():
                hour_rows.append(
                    {
                        "hour": hour,
                        "mae": f"{abs_error[mask].mean():.6f}",
                        "sample_count": int(mask.sum()),
                    }
                )
        write_csv(
            args.output_dir / f"{args.split}_time_of_day_mae.csv",
            hour_rows,
            ["hour", "mae", "sample_count"],
        )

    low, high = np.quantile(targets, [0.33, 0.66])
    regime_rows = []
    for name, mask in [
        ("low_speed", targets <= low),
        ("medium_speed", (targets > low) & (targets <= high)),
        ("high_speed", targets > high),
    ]:
        regime_rows.append(
            {
                "regime": name,
                "target_min_mph": f"{targets[mask].min():.6f}",
                "target_max_mph": f"{targets[mask].max():.6f}",
                "mae": f"{abs_error[mask].mean():.6f}",
                "sample_count": int(mask.sum()),
            }
        )
    write_csv(
        args.output_dir / f"{args.split}_speed_regime_mae.csv",
        regime_rows,
        ["regime", "target_min_mph", "target_max_mph", "mae", "sample_count"],
    )

    worst_sensor_indices = np.argsort(per_sensor_mae)[-10:][::-1]
    best_sensor_indices = np.argsort(per_sensor_mae)[:10]
    summary = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "sample_count": sample_count,
        "batch_count": int(np.ceil(sample_count / args.batch_size)),
        "overall_mae_mph": float(abs_error.mean()),
        "overall_bias_mph": float(signed_error.mean()),
        "degree_error_correlation": safe_corrcoef(node_degree, per_sensor_mae),
        "worst_sensors_by_mae": [
            {
                "sensor_index": int(node),
                "degree": int(node_degree[node]),
                "mae": float(per_sensor_mae[node]),
                "bias": float(per_sensor_bias[node]),
            }
            for node in worst_sensor_indices
        ],
        "best_sensors_by_mae": [
            {
                "sensor_index": int(node),
                "degree": int(node_degree[node]),
                "mae": float(per_sensor_mae[node]),
                "bias": float(per_sensor_bias[node]),
            }
            for node in best_sensor_indices
        ],
    }
    (args.output_dir / f"{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
