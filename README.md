# METR-LA GNN Traffic Forecasting

Small starter project for training a graph neural network on a transportation dataset.

The recommended dataset is **METR-LA**, a traffic speed forecasting benchmark where:

- nodes are road traffic sensors,
- edges represent spatial relationships between sensors,
- the target is future traffic speed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Download the DCRNN-format METR-LA files and place them like this:

```text
data/
  metr-la.h5
  sensor_graph/
    adj_mx.pkl
```

Common source:

- https://github.com/liyaguang/DCRNN

Several mirrors also provide the same files as `metr-la.h5` and `adj_mx.pkl`.

## Train

```powershell
python train.py --data data/metr-la.h5 --adj data/sensor_graph/adj_mx.pkl
```

Useful smaller smoke test:

```powershell
python train.py --data data/metr-la.h5 --adj data/sensor_graph/adj_mx.pkl --epochs 2 --hidden-dim 32
```

## What The Baseline Does

The model uses:

- a fixed sensor graph adjacency matrix,
- normalized traffic speed time series,
- 12 previous steps as input,
- 12 future steps as target,
- a simple graph convolution plus GRU forecaster.

This is intentionally compact so it is easy to modify before moving to heavier models such as STGCN, DCRNN, or Graph WaveNet.

## Explainability Metrics

After training, run:

```powershell
python evaluate_explainability.py --data data/metr-la.h5 --adj data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/normal_best.pt
```

For a quick smoke run:

```powershell
python evaluate_explainability.py --data data/metr-la.h5 --adj data/sensor_graph/adj_mx.pkl --checkpoint checkpoints/normal_best.pt --max-batches 5 --output-dir reports/explainability_smoke
```

The evaluator writes:

- per-horizon MAE and bias,
- per-sensor MAE, bias, and graph degree,
- time-of-day MAE,
- speed-regime MAE,
- node-degree vs error correlation,
- best and worst sensors by MAE.
