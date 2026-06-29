"""Create a test folder with real normal images and two polyp images."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "one problem"
NEGATIVE_ROOT = ROOT / "kvasir-dataset"
POSITIVE_DIR = ROOT / "combined_cancer_detection_dataset_with_negatives" / "images" / "test"
NORMAL_CLASSES = ("normal-cecum", "normal-pylorus", "normal-z-line")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
NEGATIVE_COUNT = 8
POSITIVE_COUNT = 2
SEED = 42


def collect_normal_images() -> list[Path]:
    images: list[Path] = []
    for class_name in NORMAL_CLASSES:
        class_dir = NEGATIVE_ROOT / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing normal class folder: {class_dir}")
        images.extend(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    if len(images) < NEGATIVE_COUNT:
        raise RuntimeError(f"Need at least {NEGATIVE_COUNT} normal images.")
    return sorted(images)


def collect_positive_images() -> list[Path]:
    images = sorted(
        path
        for path in POSITIVE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.startswith("negative_")
    )
    if len(images) < POSITIVE_COUNT:
        raise RuntimeError(f"Need at least {POSITIVE_COUNT} positive polyp images.")
    return images


def prepare_output() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)


def copy_sample(source: Path, output_name: str, expected: str) -> dict[str, str]:
    destination = OUTPUT_DIR / output_name
    shutil.copy2(source, destination)
    return {
        "file": output_name,
        "expected": expected,
        "source_image": str(source.relative_to(ROOT)),
        "source_class": source.parent.name,
    }


def write_readme() -> None:
    text = """# One Problem Test Folder

This folder is designed for a quick model/UI test.

- 8 images are real normal images from Kvasir normal classes.
- 2 images are real positive polyp images from the labeled YOLO test set.
- Use the UI `Folder` button, choose this folder, then click `Analyze Folder`.
- The expected result for each image is listed in `ground_truth.csv`.

Expected behavior:

- The model should not draw boxes on the `normal_no_polyp` images.
- The model should detect polyps on the `problem_polyp` images.
"""
    (OUTPUT_DIR / "README_ONE_PROBLEM.md").write_text(text, encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    normal_images = collect_normal_images()
    positive_images = collect_positive_images()

    selected_normals = rng.sample(normal_images, NEGATIVE_COUNT)
    selected_positives = rng.sample(positive_images, POSITIVE_COUNT)

    prepare_output()
    rows: list[dict[str, str]] = []

    for index, source in enumerate(selected_normals, start=1):
        output_name = f"{index:02d}_normal_no_polyp{source.suffix.lower()}"
        rows.append(copy_sample(source, output_name, "normal_no_polyp"))

    for offset, source in enumerate(selected_positives, start=NEGATIVE_COUNT + 1):
        output_name = f"{offset:02d}_problem_polyp{source.suffix.lower()}"
        rows.append(copy_sample(source, output_name, "polyp_present"))

    with (OUTPUT_DIR / "ground_truth.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["file", "expected", "source_image", "source_class"])
        writer.writeheader()
        writer.writerows(rows)

    write_readme()

    print(f"Created: {OUTPUT_DIR}")
    print(f"Normal no-polyp images: {NEGATIVE_COUNT}")
    print(f"Problem polyp images: {POSITIVE_COUNT}")


if __name__ == "__main__":
    main()
