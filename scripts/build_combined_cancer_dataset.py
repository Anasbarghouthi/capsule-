r"""Build a larger YOLO dataset by combining the local data with PolypDB.

Inputs:
    - cancer_detection_dataset: the current YOLO-ready dataset.
    - external_datasets/PolypDB/PolypDB_center_wise: downloaded PolypDB data.

Output:
    - combined_cancer_detection_dataset: shuffled train/val/test YOLO dataset.

Example:
    C:\Users\TUF\radioconda\python.exe build_combined_cancer_dataset.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_ID = 0
CLASS_NAME = "polyp"


@dataclass(frozen=True)
class CombinedSample:
    source: str
    source_group: str
    original_stem: str
    output_stem: str
    image_path: Path
    label_text: str
    mask_path: Path | None
    width: int
    height: int
    box_count: int
    image_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the current YOLO dataset with external PolypDB samples."
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=PROJECT_ROOT / "cancer_detection_dataset",
        help="Existing YOLO dataset folder.",
    )
    parser.add_argument(
        "--polypdb",
        type=Path,
        default=PROJECT_ROOT / "external_datasets" / "PolypDB" / "PolypDB_center_wise",
        help="PolypDB center-wise folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "combined_cancer_detection_dataset",
        help="Output combined dataset folder.",
    )
    parser.add_argument("--train", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--val", type=float, default=0.20, help="Validation split ratio.")
    parser.add_argument("--test", type=float, default=0.10, help="Testing split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output folder first if it already exists.",
    )
    return parser.parse_args()


def validate_ratios(train: float, val: float, test: float) -> None:
    if train <= 0 or val <= 0 or test <= 0:
        raise ValueError("All split ratios must be greater than zero.")
    if not math.isclose(train + val + test, 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must add up to 1.0.")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "sample"


def unique_output_stem(source: str, group: str, stem: str, used: set[str]) -> str:
    base = safe_stem(f"{source}_{group}_{stem}")
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def parse_yolo_label(label_path: Path) -> tuple[str, int]:
    lines = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        if parts[0] != str(CLASS_ID):
            continue
        values = [float(value) for value in parts[1:]]
        if any(value < 0.0 or value > 1.0 for value in values):
            continue
        lines.append(f"{CLASS_ID} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}")
    return "\n".join(lines) + ("\n" if lines else ""), len(lines)


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def collect_base_samples(base_dataset: Path, used_stems: set[str]) -> list[CombinedSample]:
    samples: list[CombinedSample] = []
    for split_name in ("train", "val", "test"):
        image_dir = base_dataset / "images" / split_name
        label_dir = base_dataset / "labels" / split_name
        mask_dir = base_dataset / "masks" / split_name
        if not image_dir.exists():
            continue

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            label_text, box_count = parse_yolo_label(label_path)
            if box_count == 0:
                continue

            width, height = image_size(image_path)
            mask_path = next(
                (
                    path
                    for suffix in (".jpg", ".jpeg", ".png")
                    for path in [mask_dir / f"{image_path.stem}{suffix}"]
                    if path.exists()
                ),
                None,
            )
            output_stem = unique_output_stem("base", split_name, image_path.stem, used_stems)
            samples.append(
                CombinedSample(
                    source="base_segmented_images",
                    source_group=split_name,
                    original_stem=image_path.stem,
                    output_stem=output_stem,
                    image_path=image_path,
                    label_text=label_text,
                    mask_path=mask_path,
                    width=width,
                    height=height,
                    box_count=box_count,
                    image_hash=file_hash(image_path),
                )
            )
    return samples


def mask_to_yolo_label(mask_path: Path, width: int, height: int) -> tuple[str, int]:
    with Image.open(mask_path) as mask:
        bbox = mask.convert("L").point(lambda pixel: 255 if pixel > 0 else 0).getbbox()

    if bbox is None:
        return "", 0

    left, top, right, bottom = bbox
    box_width = right - left
    box_height = bottom - top
    if box_width <= 0 or box_height <= 0:
        return "", 0

    x_center = (left + box_width / 2.0) / width
    y_center = (top + box_height / 2.0) / height
    norm_width = box_width / width
    norm_height = box_height / height
    label = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
    return label, 1


def collect_polypdb_samples(polypdb_dir: Path, used_stems: set[str]) -> list[CombinedSample]:
    samples: list[CombinedSample] = []
    for image_dir in sorted(polypdb_dir.rglob("images")):
        mask_dir = image_dir.parent / "masks"
        if not mask_dir.exists():
            continue

        group = safe_stem(str(image_dir.parent.relative_to(polypdb_dir)))
        mask_index = {
            mask_path.stem: mask_path
            for mask_path in mask_dir.iterdir()
            if mask_path.is_file() and mask_path.suffix.lower() in IMAGE_EXTENSIONS
        }

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = mask_index.get(image_path.stem)
            if mask_path is None:
                continue

            width, height = image_size(image_path)
            label_text, box_count = mask_to_yolo_label(mask_path, width, height)
            if box_count == 0:
                continue

            output_stem = unique_output_stem("polypdb", group, image_path.stem, used_stems)
            samples.append(
                CombinedSample(
                    source="PolypDB",
                    source_group=group,
                    original_stem=image_path.stem,
                    output_stem=output_stem,
                    image_path=image_path,
                    label_text=label_text,
                    mask_path=mask_path,
                    width=width,
                    height=height,
                    box_count=box_count,
                    image_hash=file_hash(image_path),
                )
            )
    return samples


def remove_duplicate_images(samples: list[CombinedSample]) -> tuple[list[CombinedSample], int]:
    seen_hashes: set[str] = set()
    unique_samples: list[CombinedSample] = []
    duplicates = 0

    for sample in samples:
        if sample.image_hash in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(sample.image_hash)
        unique_samples.append(sample)

    return unique_samples, duplicates


def split_samples(
    samples: list[CombinedSample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[CombinedSample]]:
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


def prepare_output(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"{output_dir} already exists. Use --force to recreate it.")
        shutil.rmtree(output_dir)

    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "masks" / split_name).mkdir(parents=True, exist_ok=True)


def copy_sample(output_dir: Path, split_name: str, sample: CombinedSample) -> dict[str, str]:
    image_out = output_dir / "images" / split_name / f"{sample.output_stem}{sample.image_path.suffix.lower()}"
    label_out = output_dir / "labels" / split_name / f"{sample.output_stem}.txt"
    mask_out = (
        output_dir / "masks" / split_name / f"{sample.output_stem}{sample.mask_path.suffix.lower()}"
        if sample.mask_path is not None
        else None
    )

    shutil.copy2(sample.image_path, image_out)
    label_out.write_text(sample.label_text, encoding="utf-8")
    if sample.mask_path is not None and mask_out is not None:
        shutil.copy2(sample.mask_path, mask_out)

    return {
        "image": str(image_out.relative_to(output_dir)),
        "label": str(label_out.relative_to(output_dir)),
        "mask": str(mask_out.relative_to(output_dir)) if mask_out else "",
    }


def write_manifest(
    output_dir: Path,
    split_map: dict[str, list[CombinedSample]],
    copied_paths: dict[tuple[str, str], dict[str, str]],
) -> None:
    with (output_dir / "combined_manifest.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "split",
                "source",
                "source_group",
                "original_stem",
                "output_stem",
                "image",
                "label",
                "mask",
                "width",
                "height",
                "box_count",
                "image_hash",
            ],
        )
        writer.writeheader()
        for split_name, samples in split_map.items():
            for sample in samples:
                paths = copied_paths[(split_name, sample.output_stem)]
                writer.writerow(
                    {
                        "split": split_name,
                        "source": sample.source,
                        "source_group": sample.source_group,
                        "original_stem": sample.original_stem,
                        "output_stem": sample.output_stem,
                        "image": paths["image"],
                        "label": paths["label"],
                        "mask": paths["mask"],
                        "width": sample.width,
                        "height": sample.height,
                        "box_count": sample.box_count,
                        "image_hash": sample.image_hash,
                    }
                )


def write_data_yaml(output_dir: Path) -> None:
    dataset_path = output_dir.resolve().as_posix()
    text = (
        f'path: "{dataset_path}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  {CLASS_ID}: {CLASS_NAME}\n"
    )
    (output_dir / "data.yaml").write_text(text, encoding="utf-8")


def write_report(
    output_dir: Path,
    split_map: dict[str, list[CombinedSample]],
    base_count: int,
    polypdb_count: int,
    duplicate_count: int,
    ratios: tuple[float, float, float],
    seed: int,
) -> None:
    total = sum(len(samples) for samples in split_map.values())
    total_boxes = sum(sample.box_count for samples in split_map.values() for sample in samples)
    train_ratio, val_ratio, test_ratio = ratios
    source_counts = Counter(sample.source for samples in split_map.values() for sample in samples)
    source_lines = "\n".join(
        f"- {source}: {count} images"
        for source, count in sorted(source_counts.items())
    )

    report = f"""# Extended Training Dataset Report

## Goal
The original dataset was combined with PolypDB to make YOLO training stronger for the **capsule endoscopy / endoscopy cancer detection** project.

## Sources Used
- Local base dataset: `cancer_detection_dataset`.
- PolypDB: an external source that contains polyp images with masks. Each mask was converted into a YOLO-format bounding box.
- Kvasir-SEG was downloaded as a reference, but it was not added again because its 1000 images were already present in the local dataset.

## Merge Criteria
- Only polyp images were kept because they are the closest class to the early colon cancer screening task.
- A bounding box was extracted from the white mask region in PolypDB.
- Exact duplicate images were removed using a SHA-256 image hash so the same image does not appear more than once across train/val/test.
- All samples were split again after merging with train={train_ratio:.0%}, validation={val_ratio:.0%}, test={test_ratio:.0%}.
- Random seed `{seed}` was used so the split can be reproduced.

## Result
- Base dataset samples before deduplication: {base_count}
- PolypDB samples before deduplication: {polypdb_count}
- Duplicate images skipped: {duplicate_count}
- Final images: {total}
- Final bounding boxes: {total_boxes}
- Training: {len(split_map["train"])} images
- Validation: {len(split_map["val"])} images
- Testing: {len(split_map["test"])} images

## Source Distribution After Deduplication
{source_lines}

## Training Files
- `data.yaml`: use this file with YOLO.
- `images/train`, `images/val`, `images/test`: image files.
- `labels/train`, `labels/val`, `labels/test`: YOLO-format labels.
- `masks/train`, `masks/val`, `masks/test`: original masks for reference or visualization.
- `combined_manifest.csv`: table that records the source of each image.
"""
    (output_dir / "combined_report.md").write_text(report, encoding="utf-8-sig")


def build_dataset(args: argparse.Namespace) -> None:
    used_stems: set[str] = set()
    base_samples = collect_base_samples(args.base_dataset, used_stems)
    polypdb_samples = collect_polypdb_samples(args.polypdb, used_stems)
    unique_samples, duplicate_count = remove_duplicate_images(base_samples + polypdb_samples)
    split_map = split_samples(unique_samples, args.train, args.val, args.seed)

    prepare_output(args.output, args.force)
    copied_paths: dict[tuple[str, str], dict[str, str]] = {}
    for split_name, samples in split_map.items():
        for sample in samples:
            copied_paths[(split_name, sample.output_stem)] = copy_sample(args.output, split_name, sample)

    write_manifest(args.output, split_map, copied_paths)
    write_data_yaml(args.output)
    write_report(
        args.output,
        split_map,
        len(base_samples),
        len(polypdb_samples),
        duplicate_count,
        (args.train, args.val, args.test),
        args.seed,
    )

    print("Combined cancer-detection dataset created")
    print(f"Base samples: {len(base_samples)}")
    print(f"PolypDB samples: {len(polypdb_samples)}")
    print(f"Duplicate images skipped: {duplicate_count}")
    print(f"Final images: {sum(len(samples) for samples in split_map.values())}")
    for split_name in ("train", "val", "test"):
        print(f"{split_name}: {len(split_map[split_name])} images")
    print(f"Output: {args.output}")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train, args.val, args.test)
    if not args.base_dataset.exists():
        raise FileNotFoundError(f"Base dataset not found: {args.base_dataset}")
    if not args.polypdb.exists():
        raise FileNotFoundError(f"PolypDB folder not found: {args.polypdb}")
    build_dataset(args)


if __name__ == "__main__":
    main()
