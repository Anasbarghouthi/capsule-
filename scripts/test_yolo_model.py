"""Evaluate the final trained YOLO model on the test split.

By default this script uses the dataset that includes real negative images.
That is the correct dataset for TP/FP/FN/TN, accuracy, F1, and specificity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "combined_cancer_detection_dataset_with_negatives" / "data.yaml"


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


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def parse_dataset_yaml(data_yaml: Path) -> Path:
    dataset_root: Path | None = None
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("path:"):
            continue
        value = stripped.split(":", 1)[1].strip().strip("\"'")
        dataset_root = Path(value)
        break

    if dataset_root is None:
        raise ValueError(f"Could not find dataset path in {data_yaml}")
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()
    return dataset_root


def label_has_polyp(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5 and parts[0] == "0":
            return True
    return False


def collect_split_images(data_yaml: Path, split: str) -> list[Path]:
    dataset_root = parse_dataset_yaml(data_yaml)
    image_dir = dataset_root / "images" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image split folder does not exist: {image_dir}")
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def matching_label_path(image_path: Path) -> Path:
    split = image_path.parent.name
    dataset_root = image_path.parents[2]
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def calculate_binary_metrics(
    model: Any,
    data_yaml: Path,
    split: str,
    imgsz: int,
    conf: float,
    device: str,
) -> dict[str, Any]:
    images = collect_split_images(data_yaml, split)
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    positive_images = 0
    negative_images = 0

    predict_args: dict[str, Any] = {
        "source": [str(path) for path in images],
        "imgsz": imgsz,
        "conf": conf,
        "verbose": False,
        "stream": True,
    }
    if device:
        predict_args["device"] = device

    for image_path, result in zip(images, model.predict(**predict_args), strict=True):
        has_polyp_label = label_has_polyp(matching_label_path(image_path))
        detected_polyp = getattr(result, "boxes", None) is not None and len(result.boxes) > 0

        if has_polyp_label:
            positive_images += 1
            counts["tp" if detected_polyp else "fn"] += 1
        else:
            negative_images += 1
            counts["fp" if detected_polyp else "tn"] += 1

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    accuracy = ratio(tp + tn, tp + fp + fn + tn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "threshold": conf,
        "total_images": len(images),
        "positive_images": positive_images,
        "negative_images": negative_images,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision_percent": percent(precision),
        "recall_percent": percent(recall),
        "specificity_percent": percent(specificity),
        "accuracy_percent": percent(accuracy),
        "f1_percent": percent(f1),
    }


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

    binary = summary.get("binary_image_metrics")
    if binary:
        lines.extend(
            [
                "",
                "## Image-Level Confusion Metrics",
                "",
                f"- Detection threshold: `{binary['threshold']}`",
                f"- Total images: `{binary['total_images']}`",
                f"- Positive images: `{binary['positive_images']}`",
                f"- Negative images: `{binary['negative_images']}`",
                f"- TP: `{binary['tp']}`",
                f"- FP: `{binary['fp']}`",
                f"- FN: `{binary['fn']}`",
                f"- TN: `{binary['tn']}`",
                f"- Precision: `{binary['precision_percent'] if binary['precision_percent'] is not None else 'N/A'}%`",
                f"- Recall: `{binary['recall_percent'] if binary['recall_percent'] is not None else 'N/A'}%`",
                f"- F1-score: `{binary['f1_percent'] if binary['f1_percent'] is not None else 'N/A'}%`",
                f"- Accuracy: `{binary['accuracy_percent'] if binary['accuracy_percent'] is not None else 'N/A'}%`",
                f"- Specificity: `{binary['specificity_percent'] if binary['specificity_percent'] is not None else 'N/A'}%`",
                "",
                "Note: these TP/FP/FN/TN values are image-level metrics. A positive image has at least one polyp label, and a negative image has an empty label file.",
            ]
        )

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
    parser.add_argument("--binary-conf", type=float, default=0.25, help="Confidence threshold for image-level TP/FP/FN/TN.")
    parser.add_argument("--device", default="", help="Optional device, for example cpu or 0.")
    parser.add_argument("--project", default=str(ROOT / "runs" / "yolo_eval"), help="Output project folder.")
    parser.add_argument("--name", default="test_with_negatives", help="Evaluation run name.")
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
        "binary_image_metrics": calculate_binary_metrics(
            model=model,
            data_yaml=data,
            split=args.split,
            imgsz=args.imgsz,
            conf=args.binary_conf,
            device=args.device,
        ),
    }

    write_report(save_dir, summary)

    print("YOLO test evaluation finished")
    print(f"Weights: {weights}")
    print(f"Data: {data}")
    print(f"Output: {save_dir}")
    for key, value in summary["metrics"].items():
        print(f"{key}: {value if value is not None else 'N/A'}%")
    print("Image-level confusion metrics:")
    for key, value in summary["binary_image_metrics"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
