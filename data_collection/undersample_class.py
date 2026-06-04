"""
undersample_class.py
--------------------
Removes images dominated by a majority class to balance the dataset.
Only removes images where the target class is the ONLY or DOMINANT class,
preserving images where multiple classes appear together.

Usage:
    python3 undersample_class.py --class_name vegetation --keep 500
    python3 undersample_class.py --class_name vegetation --remove 300
    python3 undersample_class.py --class_name vegetation --keep 500 --split train
    python3 undersample_class.py --dry_run --class_name vegetation --keep 500

Arguments:
    --class_name   : unified class name to undersample (e.g. vegetation)
    --keep         : keep this many images for that class (removes the rest)
    --remove       : alternatively, remove exactly this many images
    --split        : which split to undersample (train/val/test/all). Default: train
    --dominant_only: only remove images where target class appears in >50% of annotations
    --dry_run      : preview what would be deleted without actually deleting
"""

import os
import yaml
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

MERGED_DIR = "merged"

def load_class_map():
    yaml_path = os.path.join(MERGED_DIR, "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("merged/data.yaml not found. Run merge_datasets.py first.")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {name: idx for idx, name in enumerate(names)}
    return {name: int(idx) for idx, name in names.items()}

def get_class_counts(split, target_id):
    """
    Returns:
        dominant_files : images where target class is the only/dominant class
        shared_files   : images where target class appears alongside others
    """
    lbl_dir = os.path.join(MERGED_DIR, "labels", split)
    img_dir = os.path.join(MERGED_DIR, "images", split)

    if not os.path.exists(lbl_dir):
        print(f"  Split '{split}' not found, skipping.")
        return [], []

    dominant_files = []
    shared_files   = []

    for lbl_file in os.listdir(lbl_dir):
        if not lbl_file.endswith(".txt"):
            continue

        lbl_path = os.path.join(lbl_dir, lbl_file)
        with open(lbl_path) as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            continue

        class_ids = [int(l.split()[0]) for l in lines]
        unique_classes = set(class_ids)
        target_count = class_ids.count(target_id)
        total_count  = len(class_ids)

        # derive image filename
        ext = None
        for e in [".png", ".jpg", ".jpeg"]:
            if os.path.exists(os.path.join(img_dir, lbl_file.replace(".txt", e))):
                ext = e
                break
        if ext is None:
            continue

        img_file = lbl_file.replace(".txt", ext)
        entry = (img_file, lbl_file)

        if target_id not in unique_classes:
            continue

        # dominant = target class makes up >50% of annotations OR is the only class
        dominance_ratio = target_count / total_count
        if len(unique_classes) == 1 or dominance_ratio > 0.5:
            dominant_files.append(entry)
        else:
            shared_files.append(entry)

    return dominant_files, shared_files

def delete_files(split, files_to_remove, dry_run):
    img_dir = os.path.join(MERGED_DIR, "images", split)
    lbl_dir = os.path.join(MERGED_DIR, "labels", split)
    removed = 0
    for img_file, lbl_file in tqdm(files_to_remove, desc=f"  Removing from {split}"):
        img_path = os.path.join(img_dir, img_file)
        lbl_path = os.path.join(lbl_dir, lbl_file)
        if dry_run:
            print(f"    [DRY RUN] would delete: {img_file}")
        else:
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
        removed += 1
    return removed

def main():
    parser = argparse.ArgumentParser(description="Undersample a majority class in merged dataset.")
    parser.add_argument("--class_name",     required=True, help="Class name to undersample (e.g. vegetation)")
    parser.add_argument("--keep",           type=int, default=None, help="Number of images to KEEP for this class")
    parser.add_argument("--remove",         type=int, default=None, help="Number of images to REMOVE for this class")
    parser.add_argument("--split",          default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--dominant_only",  action="store_true", help="Only remove images where target class is dominant (>50% of annotations)")
    parser.add_argument("--dry_run",        action="store_true", help="Preview without deleting")
    parser.add_argument("--seed",           type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.keep is None and args.remove is None:
        print("ERROR: specify either --keep or --remove")
        return

    random.seed(args.seed)

    # Load class map
    name_to_id = load_class_map()
    if args.class_name not in name_to_id:
        print(f"ERROR: '{args.class_name}' not found in unified class map.")
        print(f"Available classes: {list(name_to_id.keys())}")
        return

    target_id = name_to_id[args.class_name]
    print(f"\nTarget class: '{args.class_name}' (ID {target_id})")
    if args.dry_run:
        print("DRY RUN MODE — nothing will be deleted\n")

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        print(f"\n--- Split: {split} ---")
        dominant_files, shared_files = get_class_counts(split, target_id)

        print(f"  Images where '{args.class_name}' is dominant : {len(dominant_files)}")
        print(f"  Images where '{args.class_name}' is shared   : {len(shared_files)}")
        total = len(dominant_files) + len(shared_files)
        print(f"  Total images containing '{args.class_name}'  : {total}")

        # Only remove from dominant files to protect multi-class images
        candidates = dominant_files if args.dominant_only else dominant_files
        # Always protect shared files — removing them would hurt other classes
        print(f"  Shared images will NOT be removed (protects other classes)")

        if not candidates:
            print(f"  No dominant images to remove.")
            continue

        # Determine how many to remove
        if args.keep is not None:
            current = len(candidates)
            n_remove = max(0, current - args.keep)
        else:
            n_remove = min(args.remove, len(candidates))

        if n_remove == 0:
            print(f"  Nothing to remove — already at or below target.")
            continue

        print(f"  Will remove: {n_remove} images")
        print(f"  Will keep  : {len(candidates) - n_remove} dominant + {len(shared_files)} shared")

        # Randomly select files to remove
        random.shuffle(candidates)
        files_to_remove = candidates[:n_remove]

        removed = delete_files(split, files_to_remove, args.dry_run)
        print(f"  {'[DRY RUN] Would remove' if args.dry_run else 'Removed'}: {removed} images")

    if not args.dry_run:
        print("\nUpload updated dataset to S3:")
        print("  aws s3 sync merged/ s3://object-detection-data-s3/merged-dataset/ --delete")
    else:
        print("\nRe-run without --dry_run to apply changes.")

if __name__ == "__main__":
    main()
