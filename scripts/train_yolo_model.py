r"""Train a YOLO model for capsule/endoscopy polyp detection.

The script uses the final prepared dataset:
    combined_cancer_detection_dataset/data.yaml

Examples:
    C:\Users\TUF\AppData\Local\Programs\Python\Python313\python.exe train_yolo_model.py --check-only
    C:\Users\TUF\AppData\Local\Programs\Python\Python313\python.exe train_yolo_model.py --epochs 50 --imgsz 640 --batch 8
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on the prepared polyp detection dataset."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "combined_cancer_detection_dataset" / "data.yaml",
        help="YOLO data.yaml file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Pretrained YOLO weights, for example yolov8n.pt, yolov8s.pt, or yolo11n.pt.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_ROOT / "runs" / "yolo_polyp",
        help="Folder where training runs are saved.",
    )
    parser.add_argument("--name", default="polyp_combined", help="Training run name.")
    parser.add_argument(
        "--device",
        default="",
        help="Training device. Leave empty for auto, use 0 for GPU, or cpu for CPU.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Data loader workers. 0 is safer on Windows.",
    )
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Freeze the first N layers for faster fine-tuning. 0 means do not freeze.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow writing into an existing run folder.",
    )
    parser.add_argument(
        "--test-after",
        action="store_true",
        help="Run YOLO validation on the test split after training.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate dataset files; do not import Ultralytics or train.",
    )
    return parser.parse_args()


def parse_simple_yaml(data_yaml: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"path", "train", "val", "test"}:
            config[key] = value

    return config


def resolve_dataset_path(data_yaml: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (data_yaml.parent / path).resolve()


def validate_label_file(label_path: Path) -> tuple[int, list[str]]:
    box_count = 0
    problems: list[str] = []

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{label_path}:{line_number} has {len(parts)} columns")
            continue

        if parts[0] != "0":
            problems.append(f"{label_path}:{line_number} has unexpected class id {parts[0]}")
            continue

        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            problems.append(f"{label_path}:{line_number} has a non-numeric value")
            continue

        if any(value < 0.0 or value > 1.0 for value in values):
            problems.append(f"{label_path}:{line_number} has values outside 0..1")
            continue

        box_count += 1

    return box_count, problems


def split_counts(root: Path, split_name: str, split_image_path: Path) -> tuple[int, int, int, int, list[str]]:
    image_dir = split_image_path
    label_dir = root / "labels" / split_name
    mask_dir = root / "masks" / split_name
    problems: list[str] = []

    if not image_dir.exists():
        problems.append(f"Missing image folder: {image_dir}")
        return 0, 0, 0, 0, problems

    if not label_dir.exists():
        problems.append(f"Missing label folder: {label_dir}")
        return 0, 0, 0, 0, problems

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    labels = [path for path in label_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]
    masks = [
        path
        for path in mask_dir.iterdir()
        if mask_dir.exists() and path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)
    if missing_labels:
        problems.append(f"{split_name}: {len(missing_labels)} images have no labels")
    if extra_labels:
        problems.append(f"{split_name}: {len(extra_labels)} labels have no images")

    box_count = 0
    for label_path in labels:
        label_boxes, label_problems = validate_label_file(label_path)
        box_count += label_boxes
        problems.extend(label_problems[:10])

    return len(images), len(labels), len(masks), box_count, problems


def check_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_yaml}")

    config = parse_simple_yaml(data_yaml)
    required_keys = {"path", "train", "val", "test"}
    missing_keys = required_keys - set(config)
    if missing_keys:
        raise ValueError(f"Missing keys in {data_yaml}: {', '.join(sorted(missing_keys))}")

    root = Path(config["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    print("Dataset check")
    print(f"data.yaml: {data_yaml}")
    print(f"dataset root: {root}")

    total_images = 0
    total_labels = 0
    total_masks = 0
    total_boxes = 0
    all_problems: list[str] = []

    for split_name in ("train", "val", "test"):
        split_path = resolve_dataset_path(data_yaml, config[split_name])
        images, labels, masks, boxes, problems = split_counts(root, split_name, split_path)
        total_images += images
        total_labels += labels
        total_masks += masks
        total_boxes += boxes
        all_problems.extend(problems)
        print(
            f"{split_name}: {images} images, {labels} labels, "
            f"{masks} masks, {boxes} boxes"
        )

    print(f"total: {total_images} images, {total_labels} labels, {total_masks} masks, {total_boxes} boxes")

    if all_problems:
        print("Problems found:")
        for problem in all_problems[:20]:
            print(f"- {problem}")
        raise SystemExit("Dataset check failed.")

    print("Dataset check passed.")


def train_model(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Ultralytics is not installed.\n"
            "Install it first with:\n"
            "  C:\\Users\\TUF\\AppData\\Local\\Programs\\Python\\Python313\\python.exe -m pip install ultralytics\n"
            "or:\n"
            "  C:\\Users\\TUF\\radioconda\\python.exe -m pip install ultralytics"
        ) from exc

    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(args.project),
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "exist_ok": args.exist_ok,
    }
    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze
    if args.device:
        train_kwargs["device"] = args.device

    print("Starting YOLO training")
    print(f"model: {args.model}")
    print(f"data: {args.data}")
    print(f"epochs: {args.epochs}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    if args.freeze > 0:
        print(f"freeze: first {args.freeze} layers")

    model = YOLO(args.model)
    model.train(**train_kwargs)

    best_weights = args.project / args.name / "weights" / "best.pt"
    print(f"Training finished. Best weights should be here: {best_weights}")

    if args.test_after:
        print("Running test split validation")
        model = YOLO(str(best_weights))
        val_kwargs = {
            "data": str(args.data),
            "imgsz": args.imgsz,
            "split": "test",
            "project": str(args.project),
            "name": f"{args.name}_test",
            "exist_ok": True,
        }
        if args.device:
            val_kwargs["device"] = args.device
        model.val(**val_kwargs)


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()

    check_dataset(args.data)
    if args.check_only:
        return

    train_model(args)


if __name__ == "__main__":
    main()
