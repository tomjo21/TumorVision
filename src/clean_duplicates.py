import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image


def ahash(img: Image.Image, hash_size: int = 8) -> int:
    img = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for i, p in enumerate(pixels):
        if p > avg:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find_duplicates(root: Path, threshold: int = 5) -> List[Tuple[Path, Path, int]]:
    images: List[Path] = [p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    hashes: Dict[int, List[Path]] = {}
    results: List[Tuple[Path, Path, int]] = []
    # Bucket by coarse hash (e.g., top 24 bits) to reduce comparisons
    coarse_map: Dict[int, List[Tuple[Path, int]]] = {}
    for p in images:
        try:
            with Image.open(p) as im:
                h = ahash(im)
        except Exception:
            continue
        coarse = h >> 40  # top 24 bits of the 64-bit hash
        coarse_map.setdefault(coarse, []).append((p, h))

    for _, bucket in coarse_map.items():
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                p1, h1 = bucket[i]
                p2, h2 = bucket[j]
                dist = hamming(h1, h2)
                if dist <= threshold:
                    results.append((p1, p2, dist))
    return results


def main():
    ap = argparse.ArgumentParser(description="Find near-duplicate images using perceptual hash")
    ap.add_argument("root", nargs="?", default="data/dataset", help="Root folder to scan (default: data/dataset)")
    ap.add_argument("--threshold", type=int, default=5, help="Hamming distance threshold for duplicates")
    ap.add_argument("--quarantine", default=None, help="Optional folder to move secondaries to")
    args = ap.parse_args()

    dupes = find_duplicates(Path(args.root), threshold=args.threshold)
    print(f"Found {len(dupes)} near-duplicate pairs (threshold={args.threshold})")
    for p1, p2, d in dupes[:50]:
        print(f"{p1}  <->  {p2}   dist={d}")

    if args.quarantine and dupes:
        qdir = Path(args.quarantine)
        qdir.mkdir(parents=True, exist_ok=True)
        for _, p2, _ in dupes:
            target = qdir / p2.name
            try:
                p2.replace(target)
            except Exception:
                pass
        print(f"Moved {len(dupes)} files to {qdir}")


if __name__ == "__main__":
    main()
