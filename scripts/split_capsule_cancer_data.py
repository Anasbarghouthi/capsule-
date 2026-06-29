r"""Split the capsule endoscopy cancer-detection data into train/val/test sets.

The default source is ``segmented-images`` because it contains polyp images,
matching masks, and bounding boxes. Polyps are used here as the cancer-related
finding because this dataset does not contain a direct "cancer" class, while
polyps are the main annotated lesion used for early colorectal cancer detection.

Example:
    C:\Users\TUF\radioconda\python.exe split_capsule_cancer_data.py
    C:\Users\TUF\radioconda\python.exe split_capsule_cancer_data.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_ID = 0
CLASS_NAME = "polyp"


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    mask_path: Path | None
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]]
    source_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a train/validation/test split for polyp cancer-detection data."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "segmented-images",
        help="Source folder that contains images, masks, and bounding-box JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "cancer_detection_dataset",
        help="Output dataset folder.",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--val", type=float, default=0.20, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Testing split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for a repeatable split.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned split without copying files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output folder first if it already exists.",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    ratios = {"train": train, "val": val, "test": test}
    for split_name, ratio in ratios.items():
        if ratio <= 0:
            raise ValueError(f"{split_name} ratio must be greater than 0.")

    if not math.isclose(train + val + test, 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must add up to 1.0.")


def index_files(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        return {}

    files: dict[str, Path] = {}
    for file_path in sorted(folder.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            files[file_path.stem] = file_path
    return files


def load_bounding_boxes(source_dir: Path) -> dict[str, tuple[dict, Path]]:
    bbox_files = sorted(source_dir.glob("bounding-boxes*.json"))
    if not bbox_files:
        raise FileNotFoundError(f"No bounding-box JSON files found in {source_dir}")

    records: dict[str, tuple[dict, Path]] = {}
    for json_path in bbox_files:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for stem, record in data.items():
            if stem not in records:
                records[stem] = (record, json_path)
    return records


def clean_boxes(record: dict) -> list[tuple[float, float, float, float]]:
    width = int(record["width"])
    height = int(record["height"])
    cleaned: list[tuple[float, float, float, float]] = []

    for box in record.get("bbox", []):
        if box.get("label") != CLASS_NAME:
            continue

        xmin = max(0.0, min(float(box["xmin"]), float(width)))
        ymin = max(0.0, min(float(box["ymin"]), float(height)))
        xmax = max(0.0, min(float(box["xmax"]), float(width)))
        ymax = max(0.0, min(float(box["ymax"]), float(height)))

        if xmax <= xmin or ymax <= ymin:
            continue

        cleaned.append((xmin, ymin, xmax, ymax))

    return cleaned


def collect_samples(source_dir: Path) -> tuple[list[Sample], dict[str, int]]:
    image_paths = index_files(source_dir / "images")
    mask_paths = index_files(source_dir / "masks")
    bbox_records = load_bounding_boxes(source_dir)

    samples: list[Sample] = []
    stats = {
        "bbox_records": len(bbox_records),
        "image_files": len(image_paths),
        "mask_files": len(mask_paths),
        "skipped_missing_image": 0,
        "skipped_no_polyp_box": 0,
    }

    for stem, (record, json_path) in sorted(bbox_records.items()):
        image_path = image_paths.get(stem)
        if image_path is None:
            stats["skipped_missing_image"] += 1
            continue

        boxes = clean_boxes(record)
        if not boxes:
            stats["skipped_no_polyp_box"] += 1
            continue

        samples.append(
            Sample(
                stem=stem,
                image_path=image_path,
                mask_path=mask_paths.get(stem),
                width=int(record["width"]),
                height=int(record["height"]),
                boxes=boxes,
                source_json=json_path,
            )
        )

    return samples, stats


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def yolo_line(box: tuple[float, float, float, float], width: int, height: int) -> str:
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def prepare_output_dir(output_dir: Path, source_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"{output_dir} already exists. Use --force to recreate it."
            )

        output_resolved = output_dir.resolve()
        source_resolved = source_dir.resolve()
        cwd_resolved = Path.cwd().resolve()
        if output_resolved in {source_resolved, cwd_resolved}:
            raise ValueError("Refusing to delete the source folder or project folder.")
        shutil.rmtree(output_dir)

    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "masks" / split_name).mkdir(parents=True, exist_ok=True)


def write_split_files(output_dir: Path, split_name: str, sample: Sample) -> dict[str, str]:
    image_out = output_dir / "images" / split_name / sample.image_path.name
    label_out = output_dir / "labels" / split_name / f"{sample.stem}.txt"
    mask_out = (
        output_dir / "masks" / split_name / sample.mask_path.name
        if sample.mask_path is not None
        else None
    )

    shutil.copy2(sample.image_path, image_out)
    if sample.mask_path is not None and mask_out is not None:
        shutil.copy2(sample.mask_path, mask_out)

    label_lines = [
        yolo_line(box, sample.width, sample.height)
        for box in sample.boxes
    ]
    label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    return {
        "image": str(image_out.relative_to(output_dir)),
        "label": str(label_out.relative_to(output_dir)),
        "mask": str(mask_out.relative_to(output_dir)) if mask_out else "",
    }


def write_manifest(
    output_dir: Path,
    split_map: dict[str, list[Sample]],
    copied_paths: dict[tuple[str, str], dict[str, str]],
) -> None:
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "stem",
                "image",
                "label",
                "mask",
                "width",
                "height",
                "box_count",
                "source_json",
            ],
        )
        writer.writeheader()
        for split_name, samples in split_map.items():
            for sample in samples:
                paths = copied_paths[(split_name, sample.stem)]
                writer.writerow(
                    {
                        "split": split_name,
                        "stem": sample.stem,
                        "image": paths["image"],
                        "label": paths["label"],
                        "mask": paths["mask"],
                        "width": sample.width,
                        "height": sample.height,
                        "box_count": len(sample.boxes),
                        "source_json": sample.source_json.name,
                    }
                )


def write_data_yaml(output_dir: Path) -> None:
    dataset_path = output_dir.resolve().as_posix()
    yaml_text = (
        f'path: "{dataset_path}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_report(
    output_dir: Path,
    split_map: dict[str, list[Sample]],
    stats: dict[str, int],
    seed: int,
    ratios: tuple[float, float, float],
) -> None:
    train_ratio, val_ratio, test_ratio = ratios
    total = sum(len(samples) for samples in split_map.values())
    total_boxes = sum(len(sample.boxes) for samples in split_map.values() for sample in samples)

    report = f"""# Dataset Split Report

## Goal
The dataset was prepared for the **capsule endoscopy for cancer detection** project using the polyp images available in `segmented-images`.

## Image Selection Criteria
- Images with bounding boxes labeled as `polyp` were selected.
- `unlabeled-images` was excluded because it does not contain training labels.
- Landmark images such as cecum and pylorus were excluded because they are anatomical landmarks, not the detection target.
- Indirect categories such as visibility quality or therapeutic interventions were excluded from the base split.
- Medical reason: the polyp class is the most relevant visual indicator in this dataset for early colon cancer screening.

## Split Method
- Ratio: train={train_ratio:.0%}, validation={val_ratio:.0%}, test={test_ratio:.0%}
- Random seed: `{seed}` so the split can be reproduced.
- Each image was copied with its matching mask and YOLO-format label.

## Result
- Selected images: {total}
- Total bounding boxes: {total_boxes}
- Training: {len(split_map["train"])} images
- Validation: {len(split_map["val"])} images
- Testing: {len(split_map["test"])} images

## Important Files
- `images/train`, `images/val`, `images/test`: image files.
- `labels/train`, `labels/val`, `labels/test`: YOLO-format labels.
- `masks/train`, `masks/val`, `masks/test`: original masks.
- `data.yaml`: YOLO training configuration file.
- `split_manifest.csv`: table that records where each image was written.

## Validation Notes
- Original bounding box records: {stats["bbox_records"]}
- Images inside source/images: {stats["image_files"]}
- Masks inside source/masks: {stats["mask_files"]}
- Skipped images because the image file was missing: {stats["skipped_missing_image"]}
- Skipped images because no valid polyp box was found: {stats["skipped_no_polyp_box"]}
"""
    (output_dir / "split_report.md").write_text(report, encoding="utf-8-sig")


def create_dataset(
    output_dir: Path,
    split_map: dict[str, list[Sample]],
    stats: dict[str, int],
    seed: int,
    ratios: tuple[float, float, float],
    source_dir: Path,
    force: bool,
) -> None:
    prepare_output_dir(output_dir, source_dir, force)
    copied_paths: dict[tuple[str, str], dict[str, str]] = {}

    for split_name, samples in split_map.items():
        for sample in samples:
            copied_paths[(split_name, sample.stem)] = write_split_files(
                output_dir,
                split_name,
                sample,
            )

    write_manifest(output_dir, split_map, copied_paths)
    write_data_yaml(output_dir)
    write_report(output_dir, split_map, stats, seed, ratios)


def print_summary(split_map: dict[str, list[Sample]], dry_run: bool, output_dir: Path) -> None:
    total = sum(len(samples) for samples in split_map.values())
    total_boxes = sum(len(sample.boxes) for samples in split_map.values() for sample in samples)

    print("Capsule endoscopy cancer-detection split")
    print(f"Selected class: {CLASS_NAME}")
    print(f"Total images: {total}")
    print(f"Total boxes: {total_boxes}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {len(split_map[split_name])} images")

    if dry_run:
        print("Dry run only: no files were copied.")
    else:
        print(f"Output written to: {output_dir}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)

    source_dir = args.source
    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    samples, stats = collect_samples(source_dir)
    if not samples:
        raise RuntimeError("No polyp samples were found.")

    split_map = split_samples(samples, args.train, args.val, args.seed)

    if not args.dry_run:
        create_dataset(
            args.output,
            split_map,
            stats,
            args.seed,
            (args.train, args.val, args.test),
            source_dir,
            args.force,
        )

    print_summary(split_map, args.dry_run, args.output)


if __name__ == "__main__":
    main()
