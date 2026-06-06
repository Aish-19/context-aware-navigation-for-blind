#!/usr/bin/env python3
"""Convert mixed YOLO detect/segment labels into YOLO segmentation labels.

Existing polygon rows are preserved. Detection rows with exactly five values:
  class x_center y_center width height

are converted into rectangle polygons:
  class x1 y1 x2 y1 x2 y2 x1 y2

All coordinates are expected to be normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "test")


def clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def format_float(value: float) -> str:
    return f"{value:.12g}"


def is_polygon_row(parts: list[str]) -> bool:
    return len(parts) > 5 and (len(parts) - 1) % 2 == 0


def bbox_to_rectangle_polygon(parts: list[str]) -> str:
    class_id = parts[0]
    x_center, y_center, width, height = [float(value) for value in parts[1:]]

    x1 = clip(x_center - width / 2)
    y1 = clip(y_center - height / 2)
    x2 = clip(x_center + width / 2)
    y2 = clip(y_center + height / 2)

    coords = [x1, y1, x2, y1, x2, y2, x1, y2]
    return " ".join([class_id, *[format_float(coord) for coord in coords]])


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        image = images_dir / f"{stem}{ext}"
        if image.exists():
            return image
    return None


def convert_split(source_root: Path, output_root: Path, split: str) -> list[dict[str, str | int]]:
    source_labels = source_root / split / "labels"
    source_images = source_root / split / "images"
    output_labels = output_root / split / "labels"
    output_images = output_root / split / "images"
    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, str | int]] = []

    for label_file in sorted(source_labels.glob("*.txt")):
        converted_lines: list[str] = []
        polygon_rows_preserved = 0
        bbox_rows_converted = 0
        invalid_rows_skipped = 0

        for raw_line in label_file.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 5:
                converted_lines.append(bbox_to_rectangle_polygon(parts))
                bbox_rows_converted += 1
            elif is_polygon_row(parts):
                converted_lines.append(line)
                polygon_rows_preserved += 1
            else:
                invalid_rows_skipped += 1

        image = find_image(source_images, label_file.stem)
        status = "kept"

        if image is None:
            status = "missing_image"
        else:
            label_text = "\n".join(converted_lines)
            if label_text:
                label_text += "\n"
            else:
                status = "kept_empty_label"
            (output_labels / label_file.name).write_text(label_text)
            shutil.copy2(image, output_images / image.name)

        report_rows.append(
            {
                "split": split,
                "label_file": str(label_file),
                "image_file": str(image) if image else "",
                "polygon_rows_preserved": polygon_rows_preserved,
                "bbox_rows_converted": bbox_rows_converted,
                "invalid_rows_skipped": invalid_rows_skipped,
                "status": status,
            }
        )

    return report_rows


def write_data_yaml(source_root: Path, output_root: Path) -> None:
    source_yaml = source_root / "data.yaml"
    if source_yaml.exists():
        content = source_yaml.read_text()
        content = content.replace("train: ../train/images", "train: train/images")
        content = content.replace("val: ../valid/images", "val: valid/images")
        content = content.replace("test: ../test/images", "test: test/images")
    else:
        content = "train: train/images\nval: valid/images\ntest: test/images\n"
    (output_root / "data.yaml").write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("segmentation-with-bbox-rectangles"))
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists.",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists; pass --overwrite to replace it.")
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, str | int]] = []
    for split in SPLITS:
        report_rows.extend(convert_split(source_root, output_root, split))

    write_data_yaml(source_root, output_root)

    report_file = output_root / "bbox_to_polygon_report.csv"
    with report_file.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "label_file",
                "image_file",
                "polygon_rows_preserved",
                "bbox_rows_converted",
                "invalid_rows_skipped",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    copied_images = sum(1 for row in report_rows if str(row["status"]).startswith("kept"))
    preserved_polygons = sum(int(row["polygon_rows_preserved"]) for row in report_rows)
    converted_boxes = sum(int(row["bbox_rows_converted"]) for row in report_rows)
    skipped_invalid = sum(int(row["invalid_rows_skipped"]) for row in report_rows)
    print(f"Converted dataset: {output_root}")
    print(f"Images copied: {copied_images}")
    print(f"Existing polygon annotations preserved: {preserved_polygons}")
    print(f"Bbox annotations converted to rectangle polygons: {converted_boxes}")
    print(f"Invalid rows skipped: {skipped_invalid}")
    print(f"Report: {report_file}")


if __name__ == "__main__":
    main()
