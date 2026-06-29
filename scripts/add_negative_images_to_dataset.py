"""Create a YOLO dataset copy with real negative images added.

Negative images are images that do not contain the target object. In YOLO
detection, each negative image still needs a matching empty .txt label file.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
DEFAULT_NEGATIVE_CLASSES = ("normal-cecum", "normal-pylorus", "normal-z-line")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the final YOLO dataset and add real negative images with empty labels."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "combined_cancer_detection_dataset",
        help="Existing YOLO dataset folder.",
    )
    parser.add_argument(
        "--negative-dir",
        type=Path,
        required=True,
        help="Folder that contains real no-polyp / normal images, or a dataset root with normal class folders.",
    )
    parser.add_argument(
        "--negative-classes",
        nargs="*",
        default=list(DEFAULT_NEGATIVE_CLASSES),
        help="Class folder names to use as negatives when --negative-dir is a dataset root.",
    )
    parser.add_argument(
        "--include-all-images",
        action="store_true",
        help="Use every image under --negative-dir. Only use this if the folder contains negatives only.",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=0,
        help="Optional maximum number of negative images to add. 0 means use all selected negatives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "combined_cancer_detection_dataset_with_negatives",
        help="New dataset folder to create.",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Train ratio for negative images.")
    parser.add_argument("--val", type=float, default=0.20, help="Validation ratio for negative images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--force", action="store_true", help="Recreate output if it already exists.")
    return parser.parse_args()


def collect_images(folder: Path, class_names: list[str], include_all: bool) -> list[Path]:
    if include_all:
        search_roots = [folder]
    else:
        search_roots = [folder / class_name for class_name in class_names]
        missing_roots = [path for path in search_roots if not path.exists()]
        if missing_roots:
            missing = "\n".join(f"- {path}" for path in missing_roots)
            raise FileNotFoundError(
                "Some requested negative class folders were not found:\n"
                f"{missing}\n"
                "Use --include-all-images only if the folder contains negative images only."
            )

    images = sorted(
        path
        for search_root in search_roots
        for path in search_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No negative images found in: {folder}")
    return images


def prepare_output(source: Path, output: Path, force: bool) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if output.exists():
        if not force:
            raise FileExistsError(f"{output} already exists. Use --force to recreate it.")
        shutil.rmtree(output)

    shutil.copytree(source, output)
    for cache_file in output.rglob("*.cache"):
        cache_file.unlink()


def split_negative_images(images: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def limit_images(images: list[Path], max_count: int, seed: int) -> list[Path]:
    if max_count <= 0 or len(images) <= max_count:
        return images
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:max_count])


def unique_output_stem(output_image_dir: Path, original_stem: str, index: int) -> str:
    candidate = f"negative_{index:05d}_{original_stem}"
    if not (output_image_dir / f"{candidate}.jpg").exists():
        return candidate
    return f"negative_{index:05d}_{original_stem}_{random.randint(1000, 9999)}"


def copy_negative_images(output: Path, split_map: dict[str, list[Path]]) -> list[dict[str, str]]:
    manifest_rows: list[dict[str, str]] = []
    global_index = 1

    for split_name in SPLITS:
        image_dir = output / "images" / split_name
        label_dir = output / "labels" / split_name
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        for source_image in split_map[split_name]:
            stem = unique_output_stem(image_dir, source_image.stem, global_index)
            output_image = image_dir / f"{stem}{source_image.suffix.lower()}"
            output_label = label_dir / f"{stem}.txt"

            shutil.copy2(source_image, output_image)
            output_label.write_text("", encoding="utf-8")

            manifest_rows.append(
                {
                    "split": split_name,
                    "image": str(output_image.relative_to(output)),
                    "label": str(output_label.relative_to(output)),
                    "source": str(source_image),
                    "class": source_image.parent.name,
                    "boxes": "0",
                }
            )
            global_index += 1

    return manifest_rows


def rewrite_data_yaml(output: Path) -> None:
    yaml_text = (
        f'path: "{output.resolve().as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: polyp\n"
    )
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_manifest(output: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = output / "negative_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["split", "image", "label", "source", "class", "boxes"])
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    output: Path,
    negative_count: int,
    split_map: dict[str, list[Path]],
    negative_classes: list[str],
    include_all: bool,
) -> None:
    class_text = "all folders under negative-dir" if include_all else ", ".join(negative_classes)
    report = f"""# Dataset With Negative Images

This dataset was created by copying the original YOLO polyp dataset and adding real negative images.

## Why this is needed

The first dataset mainly contained positive polyp images. That can make the detector over-predict polyps because it does not see enough examples of images where no polyp exists.

## YOLO negative-image rule

A negative image must have a matching `.txt` label file, but that label file must be empty. This tells YOLO that the image contains no target object.

## Added Negative Images

- Negative source classes: {class_text}
- Total negative images added: {negative_count}
- Train negatives: {len(split_map["train"])}
- Validation negatives: {len(split_map["val"])}
- Test negatives: {len(split_map["test"])}

## Training File

Use:

```text
{output / "data.yaml"}
```
"""
    (output / "NEGATIVE_DATASET_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    test_ratio = 1.0 - args.train - args.val
    if args.train <= 0 or args.val < 0 or test_ratio < 0:
        raise ValueError("Invalid split ratios. train + val must be <= 1.0")

    negative_images = collect_images(args.negative_dir, args.negative_classes, args.include_all_images)
    negative_images = limit_images(negative_images, args.max_negatives, args.seed)
    split_map = split_negative_images(negative_images, args.train, args.val, args.seed)

    prepare_output(args.source, args.output, args.force)
    rows = copy_negative_images(args.output, split_map)
    rewrite_data_yaml(args.output)
    write_manifest(args.output, rows)
    write_report(args.output, len(negative_images), split_map, args.negative_classes, args.include_all_images)

    print("Dataset with negatives created")
    print(f"Source dataset: {args.source}")
    print(f"Negative image folder: {args.negative_dir}")
    print(
        "Negative classes: "
        f"{'all images' if args.include_all_images else ', '.join(args.negative_classes)}"
    )
    print(f"Output dataset: {args.output}")
    print(f"Negative images added: {len(negative_images)}")
    for split_name in SPLITS:
        print(f"{split_name}: {len(split_map[split_name])} negative images")


if __name__ == "__main__":
    main()
