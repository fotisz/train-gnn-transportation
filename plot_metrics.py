from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics as an SVG chart.")
    parser.add_argument("--metrics", type=Path, default=Path("runs/metrics_smoke.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/validation_mae.svg"))
    return parser.parse_args()


def read_points(path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train_mae: list[float] = []
    val_mae: list[float] = []

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_mae.append(float(row["train_mae"]))
            val_mae.append(float(row["val_mae"]))

    if not epochs:
        raise ValueError(f"No rows found in {path}.")

    return epochs, train_mae, val_mae


def scale_points(
    epochs: list[int],
    values: list[float],
    min_y: float,
    max_y: float,
    width: int,
    height: int,
    margin: int,
) -> list[tuple[float, float]]:
    x_span = max(max(epochs) - min(epochs), 1)
    y_span = max(max_y - min_y, 1e-9)

    points = []
    for epoch, value in zip(epochs, values):
        x = margin + ((epoch - min(epochs)) / x_span) * (width - 2 * margin)
        y = height - margin - ((value - min_y) / y_span) * (height - 2 * margin)
        points.append((x, y))
    return points


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def render_svg(epochs: list[int], train_mae: list[float], val_mae: list[float]) -> str:
    width = 900
    height = 520
    margin = 72
    all_values = train_mae + val_mae
    min_y = min(all_values) - 0.02
    max_y = max(all_values) + 0.02

    train_points = scale_points(epochs, train_mae, min_y, max_y, width, height, margin)
    val_points = scale_points(epochs, val_mae, min_y, max_y, width, height, margin)

    y_ticks = [min_y + i * (max_y - min_y) / 4 for i in range(5)]
    y_axis = []
    for tick in y_ticks:
        y = height - margin - ((tick - min_y) / (max_y - min_y)) * (height - 2 * margin)
        y_axis.append(
            f'<line x1="{margin}" y1="{y:.1f}" x2="{width - margin}" y2="{y:.1f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        y_axis.append(
            f'<text x="{margin - 12}" y="{y + 5:.1f}" text-anchor="end" '
            'font-size="13" fill="#475569">'
            f"{tick:.3f}</text>"
        )

    x_labels = []
    for epoch, (x, _) in zip(epochs, val_points):
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - margin + 32}" text-anchor="middle" '
            'font-size="13" fill="#475569">'
            f"{epoch}</text>"
        )

    val_circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2563eb" />'
        for x, y in val_points
    )
    train_circles = "\n".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#16a34a" />'
        for x, y in train_points
    )

    val_labels = "\n".join(
        f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" '
        'font-size="13" font-weight="600" fill="#1e40af">'
        f"{value:.4f}</text>"
        for (x, y), value in zip(val_points, val_mae)
    )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{margin}" y="42" font-size="24" font-weight="700" fill="#0f172a">METR-LA GNN Validation MAE</text>
  <text x="{margin}" y="66" font-size="14" fill="#64748b">Lower is better. Full training run, {len(epochs)} epochs.</text>
  {"".join(y_axis)}
  <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#94a3b8" stroke-width="1.5"/>
  <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#94a3b8" stroke-width="1.5"/>
  {"".join(x_labels)}
  <text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" fill="#334155">Epoch</text>
  <text x="22" y="{height / 2}" transform="rotate(-90 22 {height / 2})" text-anchor="middle" font-size="14" fill="#334155">MAE</text>
  <polyline points="{polyline(train_points)}" fill="none" stroke="#16a34a" stroke-width="3"/>
  <polyline points="{polyline(val_points)}" fill="none" stroke="#2563eb" stroke-width="4"/>
  {train_circles}
  {val_circles}
  {val_labels}
  <rect x="{width - 245}" y="34" width="176" height="64" rx="6" fill="#f8fafc" stroke="#cbd5e1"/>
  <line x1="{width - 224}" y1="58" x2="{width - 184}" y2="58" stroke="#2563eb" stroke-width="4"/>
  <text x="{width - 170}" y="63" font-size="13" fill="#334155">validation MAE</text>
  <line x1="{width - 224}" y1="82" x2="{width - 184}" y2="82" stroke="#16a34a" stroke-width="3"/>
  <text x="{width - 170}" y="87" font-size="13" fill="#334155">training MAE</text>
</svg>
'''


def main() -> None:
    args = parse_args()
    epochs, train_mae, val_mae = read_points(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(epochs, train_mae, val_mae), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
