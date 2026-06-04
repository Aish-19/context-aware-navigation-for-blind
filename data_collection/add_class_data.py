"""
add_class_data.py
-----------------
Use this AFTER training when a specific class underperforms.
Adds more data for a target class without touching existing merged dataset.

Usage:
    python3 add_class_data.py \
        --api_key YOUR_KEY \
        --workspace YOUR_WORKSPACE \
        --project YOUR_PROJECT \
        --version 1 \
        --target_class stairs

Example:
    python3 add_class_data.py \
        --api_key PMLIf0n6W9pGHgrQErf4 \
        --workspace tom-lai-8bp7n \
        --project stairs-i2yia \
        --version 4 \
        --target_class stairs
"""

import os
import shutil
import yaml
import argparse
from tqdm import tqdm
from roboflow import Roboflow

# ─────────────────────────────────────────────
# KNOWN CLASS NAME ALIASES (same as merge_datasets.py)
# ─────────────────────────────────────────────
CLASS_NAME_ALIASES = {
    "grass":       "vegetation",
    "lawn":        "vegetation",
    "tree":        "vegetation",
    "bush":        "vegetation",
    "railing":     "fence",
    "guardrail":   "guard_rail",
    "guard rail":  "guard_rail",
    "walk":        "pedestrian_walk",
    "Walk":        "pedestrian_walk",
    "dont walk":   "pedestrian_stop",
    "don't walk":  "pedestrian_stop",
    "stop hand":   "pedestrian_stop",
    "no walk":     "pedestrian_stop",
    "scooter":     "electric_scooter",
    "e-scooter":   "electric_scooter",
    "escooter":    "electric_scooter",
}

def load_merged_class_map():
    """Read current unified class map from merged/data.yaml."""
    yaml_path = "merged/data.yaml"
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("merged/data.yaml not found. Run merge_datasets.py first.")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        name_to_id = {name: idx for idx, name in enumerate(names)}
    else:
        name_to_id = {name: int(idx) for idx, name in names.items()}
    print(f"\nExisting unified classes ({len(name_to_id)}):")
    for name, idx in name_to_id.items():
        print(f"  {idx}: {name}")
    return name_to_id, data

def remap_labels(src_label, dst_label, old_id_to_new_id):
    """Remap class IDs in a label file."""
    if not os.path.exists(src_label):
        return False
    lines = []
    with open(src_label) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(parts[0])
            if old_id not in old_id_to_new_id:
                continue
            new_id = old_id_to_new_id[old_id]
            lines.append(f"{new_id} " + " ".join(parts[1:]))
    if not lines:
        return False
    with open(dst_label, "w") as f:
        f.write("\n".join(lines))
    return True

def update_yaml(name_to_id, yaml_data):
    """Save updated data.yaml with any new classes added."""
    yaml_data["nc"] = len(name_to_id)
    yaml_data["names"] = {idx: name for name, idx in sorted(name_to_id.items(), key=lambda x: x[1])}
    with open("merged/data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    print("\nUpdated merged/data.yaml")

def main():
    parser = argparse.ArgumentParser(description="Add more data for a specific class to merged dataset.")
    parser.add_argument("--api_key",       required=True, help="Roboflow API key")
    parser.add_argument("--workspace",     required=True, help="Roboflow workspace")
    parser.add_argument("--project",       required=True, help="Roboflow project name")
    parser.add_argument("--version",       required=True, type=int, help="Roboflow version number")
    parser.add_argument("--target_class",  required=True, help="Unified class name to add data for (e.g. stairs)")
    args = parser.parse_args()

    location = f"rf_extra_{args.project}_v{args.version}"

    # 1. Load existing class map
    name_to_id, yaml_data = load_merged_class_map()

    # 2. Download new dataset
    print(f"\nDownloading {args.project} v{args.version}...")
    rf = Roboflow(api_key=args.api_key)
    rf.workspace(args.workspace).project(args.project).version(args.version).download(
        "yolo26", location=location
    )

    # 3. Read new dataset class names
    with open(os.path.join(location, "data.yaml")) as f:
        new_data = yaml.safe_load(f)
    new_names = new_data.get("names", [])
    if isinstance(new_names, dict):
        new_names = list(new_names.values())
    print(f"\nNew dataset classes: {new_names}")

    # 4. Build ID remap — only keep target class and overlaps
    old_to_new = {}
    for old_id, name in enumerate(new_names):
        normalized = CLASS_NAME_ALIASES.get(name, name.lower().replace(" ", "_"))

        # Add new class to unified map if it doesn't exist
        if normalized not in name_to_id:
            new_id = len(name_to_id)
            name_to_id[normalized] = new_id
            print(f"  New class added: '{normalized}' → ID {new_id}")

        old_to_new[old_id] = name_to_id[normalized]

    # 5. Copy only files that contain the target class
    target_id_in_new = None
    for old_id, name in enumerate(new_names):
        normalized = CLASS_NAME_ALIASES.get(name, name.lower().replace(" ", "_"))
        if normalized == args.target_class:
            target_id_in_new = old_id
            break

    if target_id_in_new is None:
        print(f"\nERROR: target_class '{args.target_class}' not found in new dataset.")
        print(f"Available classes: {new_names}")
        return

    split_map = {"train": "train", "valid": "val", "test": "test"}
    prefix = f"extra_{args.project}_v{args.version}"
    total_added = 0

    for rf_split, unified_split in split_map.items():
        img_dir = os.path.join(location, rf_split, "images")
        lbl_dir = os.path.join(location, rf_split, "labels")
        if not os.path.exists(img_dir):
            continue

        count = 0
        for img_file in tqdm(os.listdir(img_dir), desc=f"  {rf_split}"):
            ext = os.path.splitext(img_file)[1]
            lbl_file = img_file.replace(ext, ".txt")
            src_lbl = os.path.join(lbl_dir, lbl_file)

            # Only add image if it contains the target class
            if not os.path.exists(src_lbl):
                continue
            with open(src_lbl) as f:
                class_ids = [int(l.split()[0]) for l in f if l.strip()]
            if target_id_in_new not in class_ids:
                continue

            src_img = os.path.join(img_dir, img_file)
            dst_img = f"merged/images/{unified_split}/{prefix}_{img_file}"
            dst_lbl = f"merged/labels/{unified_split}/{prefix}_{lbl_file}"

            if os.path.exists(dst_img):
                continue  # skip duplicates

            shutil.copy(src_img, dst_img)
            if remap_labels(src_lbl, dst_lbl, old_to_new):
                count += 1

        print(f"  {rf_split}: added {count} images for '{args.target_class}'")
        total_added += count

    # 6. Update data.yaml with any new classes
    update_yaml(name_to_id, yaml_data)

    print(f"\nTotal added: {total_added} images for class '{args.target_class}'")
    print("\nUpload updated dataset to S3:")
    print(f"  aws s3 cp -r merged/ s3://object-detection-data-s3/merged-dataset/")

if __name__ == "__main__":
    main()
