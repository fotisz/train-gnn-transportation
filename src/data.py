from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class TrafficWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, values: np.ndarray, input_steps: int, horizon: int) -> None:
        if values.ndim != 2:
            raise ValueError(f"Expected values with shape [time, nodes], got {values.shape}.")
        self.values = torch.from_numpy(values.astype(np.float32))
        self.input_steps = input_steps
        self.horizon = horizon
        self.length = len(values) - input_steps - horizon + 1
        if self.length <= 0:
            raise ValueError("Time series is too short for the requested windows.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.values[index : index + self.input_steps]
        y = self.values[index + self.input_steps : index + self.input_steps + self.horizon]
        return x.unsqueeze(-1), y.unsqueeze(-1)


@dataclass
class MetrLaDataModule:
    data_path: Path
    adj_path: Path
    input_steps: int = 12
    horizon: int = 12
    batch_size: int = 64
    train_ratio: float = 0.7
    val_ratio: float = 0.1

    def setup(self) -> None:
        values = self._load_h5(self.data_path)
        train_end = int(len(values) * self.train_ratio)
        val_end = int(len(values) * (self.train_ratio + self.val_ratio))

        scaler = StandardScaler()
        scaler.fit(values[:train_end])

        scaled = scaler.transform(values).astype(np.float32)
        train_values = scaled[:train_end]
        val_values = scaled[train_end - self.input_steps - self.horizon + 1 : val_end]
        test_values = scaled[val_end - self.input_steps - self.horizon + 1 :]

        self.train_dataset = TrafficWindowDataset(train_values, self.input_steps, self.horizon)
        self.val_dataset = TrafficWindowDataset(val_values, self.input_steps, self.horizon)
        self.test_dataset = TrafficWindowDataset(test_values, self.input_steps, self.horizon)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

        self.num_nodes = values.shape[1]
        self.normalized_adjacency = self._load_normalized_adjacency(self.adj_path, self.num_nodes)

    @staticmethod
    def _load_h5(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Could not find traffic data file: {path}")

        frame = pd.read_hdf(path)
        if isinstance(frame.index, pd.DatetimeIndex):
            frame = frame.sort_index()

        values = frame.to_numpy(dtype=np.float32)
        if np.isnan(values).any():
            values = pd.DataFrame(values).ffill().bfill().to_numpy(dtype=np.float32)
        return values

    @staticmethod
    def _load_normalized_adjacency(path: Path, num_nodes: int) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Could not find adjacency file: {path}")

        with path.open("rb") as file:
            payload: Any = pickle.load(file, encoding="latin1")

        adjacency = (
            payload[2]
            if isinstance(payload, (tuple, list)) and len(payload) >= 3
            else payload
        )
        adjacency = np.asarray(adjacency, dtype=np.float32)

        if adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"Expected adjacency shape {(num_nodes, num_nodes)}, got {adjacency.shape}."
            )

        adjacency = adjacency + np.eye(num_nodes, dtype=np.float32)
        degree = adjacency.sum(axis=1)
        degree_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
        np.power(degree, -0.5, out=degree_inv_sqrt, where=degree > 0)
        normalized = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
        return torch.from_numpy(normalized.astype(np.float32))
