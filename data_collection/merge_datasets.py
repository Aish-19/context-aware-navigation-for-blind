"""
merge_datasets.py
-----------------
Run on EC2. Downloads Roboflow datasets, merges with existing
Cityscapes data, remaps class IDs, outputs unified dataset.

Usage:
    pip install roboflow pyyaml tqdm
    python3 merge_datasets.py

Flags:
    INCREMENTAL = True  → skip files already in merged/, safe to re-run
    INCREMENTAL = False → full re-merge from scratch
"""

import os
import shutil
import yaml
from tqdm import tqdm
from roboflow import Roboflow

# ─────────────────────────────────────────────
# INCREMENTAL MODE
# Set True to skip already processed files (safe to re-run)
# Set False to re-merge everything from scratch
# ─────────────────────────────────────────────
INCREMENTAL = True

# ─────────────────────────────────────────────
# BASE PATHS
# ─────────────────────────────────────────────
BASE_DIR      = "/mnt/my-data"
CITYSCAPES_DIR = os.path.join(BASE_DIR, "dataset")
MERGED_DIR    = os.path.join(BASE_DIR, "merged")
RF_DIR        = os.path.join(BASE_DIR, "roboflow")

# ─────────────────────────────────────────────
# 1. EXISTING CITYSCAPES CLASSES (already extracted)
# ─────────────────────────────────────────────
CITYSCAPES_CLASSES = {
    0:  "person",
    1:  "rider",
    2:  "car",
    3:  "truck",
    4:  "bus",
    5:  "on_rails",
    6:  "motorcycle",
    7:  "bicycle",
    8:  "caravan",
    9:  "trailer",
    10: "wall",
    11: "fence",
    12: "guard_rail",
    13: "pole",
    14: "polegroup",
    15: "vegetation",
    16: "terrain",
}

# ─────────────────────────────────────────────
# 2. ROBOFLOW DATASETS TO DOWNLOAD
# ─────────────────────────────────────────────
ROBOFLOW_DATASETS = [
    {
        "api_key":   "5Na9Ub6r40FOrdUJVbps",
        "workspace": "shruthikas-workspace",
        "project":   "annotated-walk-assist",
        "version":   1,
        "location":  "rf_walkassist",
    },
    {
        "api_key":   "PMLIf0n6W9pGHgrQErf4",
        "workspace": "tom-lai-8bp7n",
        "project":   "stairs-i2yia",
        "version":   3,
        "location":  "rf_stairs",
    },
    {
        "api_key":   "5Na9Ub6r40FOrdUJVbps",
        "workspace": "shruthikas-workspace",
        "project":   "annotate-electric-scooters",
        "version":   1,
        "location":  "rf_escooter",
    },
]

# ─────────────────────────────────────────────
# 3. KNOWN CLASS NAME MAPPINGS
#    Maps Roboflow class names → unified name
#    Handles overlaps and naming inconsistencies
# ─────────────────────────────────────────────
CLASS_NAME_ALIASES = {
    # vegetation aliases
    "grass":       "vegetation",
    "lawn":        "vegetation",
    "tree":        "vegetation",
    "bush":        "vegetation",
    # fence aliases
    "railing":     "fence",
    "guardrail":   "guard_rail",
    "guard rail":  "guard_rail",
    # pedestrian signal aliases
    "walk":        "pedestrian_walk",
    "Walk":        "pedestrian_walk",
    "dont walk":   "pedestrian_stop",
    "don't walk":  "pedestrian_stop",
    "stop hand":   "pedestrian_stop",
    "no walk":     "pedestrian_stop",
    # scooter aliases
    "scooter":     "electric_scooter",
    "e-scooter":   "electric_scooter",
    "escooter":    "electric_scooter",
}

# ─────────────────────────────────────────────
# 4. DOWNLOAD ROBOFLOW DATASETS
# ─────────────────────────────────────────────
def download_datasets():
    print("\n=== Downloading Roboflow Datasets ===")
    os.makedirs(RF_DIR, exist_ok=True)
    for ds in ROBOFLOW_DATASETS:
        location = os.path.join(RF_DIR, ds["location"])
        if os.path.exists(location):
            print(f"  {ds['location']} already exists, skipping download.")
            continue
        print(f"  Downloading {ds['project']}...")
        rf = Roboflow(api_key=ds["api_key"])
        rf.workspace(ds["workspace"]).project(ds["project"]).version(ds["version"]).download(
            "yolo26", location=location
        )
        print(f"  Done: {location}")

# ─────────────────────────────────────────────
# 5. BUILD UNIFIED CLASS MAP
# ─────────────────────────────────────────────
def build_unified_class_map():
    print("\n=== Building Unified Class Map ===")

    # Start with cityscapes classes
    unified_classes = list(CITYSCAPES_CLASSES.values())

    # Collect all Roboflow class names
    roboflow_class_info = []
    for ds in ROBOFLOW_DATASETS:
        yaml_path = os.path.join(RF_DIR, ds["location"], "data.yaml")
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        names = data.get("names", [])
        # Handle both dict and list formats
        if isinstance(names, dict):
            names = list(names.values())
        print(f"  {ds['location']} classes: {names}")
        roboflow_class_info.append((ds["location"], names))

    # Add unique classes from Roboflow
    for location, names in roboflow_class_info:
        for name in names:
            normalized = CLASS_NAME_ALIASES.get(name, name.lower().replace(" ", "_"))
            if normalized not in unified_classes:
                unified_classes.append(normalized)
                print(f"  Added new class: '{name}' → '{normalized}' (ID {len(unified_classes)-1})")
            else:
                print(f"  Merged: '{name}' → '{normalized}' (ID {unified_classes.index(normalized)})")

    # Final map: name → ID
    name_to_id = {name: idx for idx, name in enumerate(unified_classes)}
    print(f"\n  Total unified classes: {len(unified_classes)}")
    for idx, name in enumerate(unified_classes):
        print(f"    {idx}: {name}")

    return unified_classes, name_to_id, roboflow_class_info

# ─────────────────────────────────────────────
# 6. COPY & REMAP LABELS
# ─────────────────────────────────────────────
def remap_labels(src_label, dst_label, old_id_to_new_id):
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
                continue  # skip classes not in unified map
            new_id = old_id_to_new_id[old_id]
            lines.append(f"{new_id} " + " ".join(parts[1:]))
    if not lines:
        return False
    with open(dst_label, "w") as f:
        f.write("\n".join(lines))
    return True

# ─────────────────────────────────────────────
# 7. MERGE ALL DATASETS
# ─────────────────────────────────────────────
def merge_all(unified_classes, name_to_id, roboflow_class_info):
    print("\n=== Merging Datasets ===")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(MERGED_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(MERGED_DIR, "labels", split), exist_ok=True)

    # --- Cityscapes (already correct IDs, just copy) ---
    print("\n  Copying Cityscapes data...")
    cityscapes_id_map = {old_id: name_to_id[name] for old_id, name in CITYSCAPES_CLASSES.items()}

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(CITYSCAPES_DIR, "images", split)
        lbl_dir = os.path.join(CITYSCAPES_DIR, "labels", split)
        if not os.path.exists(img_dir):
            print(f"    Cityscapes {split} not found, skipping.")
            continue
        count = 0
        for img_file in tqdm(os.listdir(img_dir), desc=f"    cityscapes/{split}"):
            if not img_file.endswith(".png"):
                continue
            lbl_file = img_file.replace(".png", ".txt")
            src_img = os.path.join(img_dir, img_file)
            src_lbl = os.path.join(lbl_dir, lbl_file)
            dst_img = os.path.join(MERGED_DIR, "images", split, f"cityscapes_{img_file}")
            dst_lbl = os.path.join(MERGED_DIR, "labels", split, f"cityscapes_{lbl_file}")
            if INCREMENTAL and os.path.exists(dst_img):
                continue
            shutil.copy(src_img, dst_img)
            if remap_labels(src_lbl, dst_lbl, cityscapes_id_map):
                count += 1
        print(f"    Cityscapes {split}: {count} images")

    # --- Roboflow Datasets ---
    split_map = {"train": "train", "valid": "val", "test": "test"}

    for ds, (location, rf_names) in zip(ROBOFLOW_DATASETS, roboflow_class_info):
        print(f"\n  Processing {location}...")
        full_location = os.path.join(RF_DIR, location)

        if isinstance(rf_names, dict):
            rf_names = list(rf_names.values())

        old_to_new = {}
        for old_id, name in enumerate(rf_names):
            normalized = CLASS_NAME_ALIASES.get(name, name.lower().replace(" ", "_"))
            if normalized in name_to_id:
                old_to_new[old_id] = name_to_id[normalized]

        for rf_split, unified_split in split_map.items():
            img_dir = os.path.join(full_location, rf_split, "images")
            lbl_dir = os.path.join(full_location, rf_split, "labels")
            if not os.path.exists(img_dir):
                continue
            count = 0
            prefix = location.replace("rf_", "")
            for img_file in tqdm(os.listdir(img_dir), desc=f"    {location}/{rf_split}"):
                ext = os.path.splitext(img_file)[1]
                lbl_file = img_file.replace(ext, ".txt")
                src_img = os.path.join(img_dir, img_file)
                src_lbl = os.path.join(lbl_dir, lbl_file)
                dst_img = os.path.join(MERGED_DIR, "images", unified_split, f"{prefix}_{img_file}")
                dst_lbl = os.path.join(MERGED_DIR, "labels", unified_split, f"{prefix}_{lbl_file}")
                if INCREMENTAL and os.path.exists(dst_img):
                    continue
                shutil.copy(src_img, dst_img)
                if remap_labels(src_lbl, dst_lbl, old_to_new):
                    count += 1
            print(f"    {location}/{rf_split}: {count} images")

# ─────────────────────────────────────────────
# 8. WRITE UNIFIED data.yaml
# ─────────────────────────────────────────────
def write_yaml(unified_classes):
    print("\n=== Writing data.yaml ===")
    yaml_content = {
        "path": MERGED_DIR,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(unified_classes),
        "names": {i: name for i, name in enumerate(unified_classes)},
    }
    yaml_path = os.path.join(MERGED_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {yaml_path}")

# ─────────────────────────────────────────────
# 9. PRINT FINAL SUMMARY
# ─────────────────────────────────────────────
def print_summary():
    print("\n=== Final Dataset Summary ===")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(MERGED_DIR, "images", split)
        if os.path.exists(img_dir):
            count = len(os.listdir(img_dir))
            print(f"  {split}: {count} images")
    print("\n  Upload to S3:")
    print(f"  aws s3 cp -r {MERGED_DIR}/ s3://object-detection-data-s3/merged-dataset/")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    download_datasets()
    unified_classes, name_to_id, roboflow_class_info = build_unified_class_map()
    merge_all(unified_classes, name_to_id, roboflow_class_info)
    write_yaml(unified_classes)
    print_summary()

    print("\n=== Upload to S3 ===")
    os.system(f"aws s3 cp -r {MERGED_DIR}/ s3://object-detection-data-s3/merged-dataset/")
    print("Done!")
