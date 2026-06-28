"""Evaluate the trained YOLO model on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "combined_cancer_detection_dataset" / "data.yaml"


def find_latest_best_pt(root: Path = ROOT) -> Path:
    """Find the newest YOLO best.pt file inside runs/."""
    candidates = list((root / "runs").rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "No best.pt was found under runs/. Pass --weights with the trained model path."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def metric_value(obj: Any, name: str) -> float | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100, 2)


def write_report(save_dir: Path, summary: dict[str, Any]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_dir / "evaluation_summary.json"
    md_path = save_dir / "evaluation_summary.md"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# YOLO Test Evaluation Summary",
        "",
        f"- Weights: `{summary['weights']}`",
        f"- Data YAML: `{summary['data']}`",
        f"- Split: `{summary['split']}`",
        f"- Image size: `{summary['imgsz']}`",
        "",
        "## Metrics",
        "",
    ]

    metric_labels = {
        "precision_percent": "Precision",
        "recall_percent": "Recall",
        "map50_percent": "mAP50",
        "map50_95_percent": "mAP50-95",
        "map75_percent": "mAP75",
    }

    for key, label in metric_labels.items():
        value = summary["metrics"].get(key)
        lines.append(f"- {label}: `{value if value is not None else 'N/A'}%`")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Evaluation folder: `{save_dir}`",
            "- Use the generated plots from this folder in the project report.",
        ]
    )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate best.pt on the YOLO test split.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to trained best.pt.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to data.yaml.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for metric calculation.")
    parser.add_argument("--device", default="", help="Optional device, for example cpu or 0.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "yolo_eval"), help="Output project folder.")
    parser.add_argument("--name", default="test_eval", help="Evaluation run name.")
    parser.add_argument("--no-plots", action="store_true", help="Disable Ultralytics plots.")
    parser.add_argument("--save-json", action="store_true", help="Ask Ultralytics to save COCO-style JSON when possible.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = (args.weights or find_latest_best_pt()).resolve()
    data = args.data.resolve()

    if not weights.exists():
        raise FileNotFoundError(f"Weights file does not exist: {weights}")
    if not data.exists():
        raise FileNotFoundError(f"data.yaml does not exist: {data}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with: "
            "python -m pip install ultralytics"
        ) from exc

    model = YOLO(str(weights))
    val_args: dict[str, Any] = {
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "plots": not args.no_plots,
        "save_json": args.save_json,
    }
    if args.device:
        val_args["device"] = args.device

    metrics = model.val(**val_args)
    box_metrics = getattr(metrics, "box", metrics)

    save_dir = Path(getattr(metrics, "save_dir", Path(args.project) / args.name)).resolve()
    summary = {
        "weights": str(weights),
        "data": str(data),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "save_dir": str(save_dir),
        "metrics": {
            "precision_percent": percent(metric_value(box_metrics, "mp")),
            "recall_percent": percent(metric_value(box_metrics, "mr")),
            "map50_percent": percent(metric_value(box_metrics, "map50")),
            "map50_95_percent": percent(metric_value(box_metrics, "map")),
            "map75_percent": percent(metric_value(box_metrics, "map75")),
        },
    }

    write_report(save_dir, summary)

    print("YOLO test evaluation finished")
    print(f"Weights: {weights}")
    print(f"Data: {data}")
    print(f"Output: {save_dir}")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value if value is not None else 'N/A'}%")


if __name__ == "__main__":
    main()
