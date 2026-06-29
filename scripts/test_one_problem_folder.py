"""Run the trained YOLO model on the one problem folder."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "one problem"
OUTPUT_CSV = INPUT_DIR / "model_results.csv"


def find_latest_best_pt(root: Path = ROOT) -> Path:
    candidates = list((root / "runs").rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("No best.pt was found under runs/.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def result_rows(results: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in results:
        image_path = Path(result.path)
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            rows.append(
                {
                    "file": image_path.name,
                    "status": "no_polyp_detected",
                    "detection_index": "",
                    "confidence": "",
                    "x": "",
                    "y": "",
                    "width": "",
                    "height": "",
                }
            )
            continue

        for index, box in enumerate(boxes, start=1):
            x_center, y_center, width, height = [float(value) for value in box.xywhn[0].tolist()]
            confidence = float(box.conf[0].item())
            rows.append(
                {
                    "file": image_path.name,
                    "status": "polyp_detected",
                    "detection_index": str(index),
                    "confidence": f"{confidence:.4f}",
                    "x": f"{max(0.0, x_center - width / 2):.6f}",
                    "y": f"{max(0.0, y_center - height / 2):.6f}",
                    "width": f"{width:.6f}",
                    "height": f"{height:.6f}",
                }
            )
    return rows


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Folder does not exist: {INPUT_DIR}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed in this Python environment.") from exc

    weights = find_latest_best_pt()
    model = YOLO(str(weights))
    files = image_files(INPUT_DIR)
    results = model.predict([str(path) for path in files], imgsz=640, conf=0.25, verbose=False)
    rows = result_rows(results)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["file", "status", "detection_index", "confidence", "x", "y", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(rows)

    detected_files = sorted({row["file"] for row in rows if row["status"] == "polyp_detected"})
    print(f"Weights: {weights}")
    print(f"Images tested: {len(files)}")
    print(f"Images with detections: {len(detected_files)}")
    print(f"Detected files: {', '.join(detected_files) if detected_files else 'none'}")
    print(f"Results CSV: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
