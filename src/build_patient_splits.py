import argparse
import csv
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def load_patient_map_csv(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in reader.fieldnames or []}
        if "image_path" not in cols or "patient_id" not in cols:
            raise ValueError("CSV must contain columns: image_path, patient_id")
        for row in reader:
            mapping[row[cols["image_path"]]] = row[cols["patient_id"]]
    return mapping


def extract_patient_id(p: Path, regex: str | None) -> str:
    stem = p.stem
    if regex:
        m = re.search(regex, stem)
        if m:
            # Use the first capturing group if present, else the whole match
            return m.group(1) if m.groups() else m.group(0)
    # Fallback: use full stem (treat each image as its own patient to avoid leakage)
    return stem


def split_groups(groups: Dict[str, List[Path]], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_ratio))
    val_keys = set(keys[:n_val])
    train_keys = set(keys[n_val:])
    train_files = [p for k in train_keys for p in groups[k]]
    val_files = [p for k in val_keys for p in groups[k]]
    return train_files, val_files


def make_rel(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir)).replace("\\", "/")
    except ValueError:
        # Not under base_dir; write absolute
        return str(path)


def main():
    ap = argparse.ArgumentParser(description="Build patient-wise train/val/test splits")
    ap.add_argument("--base_dir", default="data/dataset", help="Dataset root containing Training/ and Testing/")
    ap.add_argument("--out_dir", default="data/splits", help="Output folder for train/val/test txt files")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio from Training groups")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for shuffling groups")
    ap.add_argument("--regex", default=None, help="Optional regex to extract patient ID from filename stem (use a capturing group)")
    ap.add_argument("--csv_map", default=None, help="Optional CSV mapping with columns image_path,patient_id (paths relative to base_dir)")
    ap.add_argument("--include_testing_in_split", action="store_true", help="If set, include Testing/ into the random split instead of using it as test")
    args = ap.parse_args()

    base = Path(args.base_dir)
    train_root = base / "Training"
    test_root = base / "Testing"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_map: Dict[str, str] | None = None
    if args.csv_map:
        csv_map = load_patient_map_csv(Path(args.csv_map))

    # Collect files
    train_files = list_images(train_root)
    test_files = list_images(test_root)

    # If including testing in split, merge and clear test_files
    pool_files = train_files + (test_files if args.include_testing_in_split else [])

    # Group by patient id
    groups: Dict[str, List[Path]] = {}
    for p in pool_files:
        rel = make_rel(p, base)
        if csv_map and rel in csv_map:
            pid = csv_map[rel]
        else:
            pid = extract_patient_id(p, args.regex)
        groups.setdefault(pid, []).append(p)

    # Split
    tr_files, val_files = split_groups(groups, args.val_ratio, args.seed)
    if args.include_testing_in_split:
        test_out = []
    else:
        test_out = test_files

    # Write lists relative to base_dir where possible
    def write_list(paths: List[Path], name: str):
        with (out_dir / f"{name}.txt").open("w", encoding="utf-8") as f:
            for p in sorted(paths):
                f.write(make_rel(p, base) + "\n")

    write_list(tr_files, "train")
    write_list(val_files, "val")
    write_list(test_out, "test")

    print(f"Wrote splits to {out_dir}")
    print(f"Train images: {len(tr_files)} | Val images: {len(val_files)} | Test images: {len(test_out)}")


if __name__ == "__main__":
    main()
