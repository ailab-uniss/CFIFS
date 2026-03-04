#!/usr/bin/env python3
"""
Build a single dataset root containing ~30 datasets by symlinking from the
available sources.

This is used to run a wider CV10 panorama without mixing multiple `--data-root`
inputs in downstream scripts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PANORAMA30 = [
    # Paper-protocol suite (including both Birds/birds if present).
    "Arts",
    "Business",
    "Education",
    "Entertain",
    "Health",
    "Recreation",
    "Reference",
    "Science",
    "Social",
    "CAL500",
    "corel5k",
    "scene",
    "tmc2007",
    "mediamill",
    "genbase",
    "medical",
    "yeast",
    "bibtex",
    "emotions",
    "enron",
    "Birds",
    "birds",
    # Additional datasets from other suites.
    "Computers",
    "Flags",
    "Yelp",
    "Rcv1sub1",
    "Rcv1sub2",
    "Slashdot",
    "Corel16k1",
    "Corel16k2",
]


def _has_dataset(dirpath: Path) -> bool:
    return (
        (dirpath / "dataset.npz").exists()
        or ((dirpath / "train.npz").exists() and (dirpath / "test.npz").exists())
    )


def _pick_source(name: str, roots: list[Path]) -> Path | None:
    for r in roots:
        p = r / name
        if p.is_dir() and _has_dataset(p):
            return p
    return None


def _symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        # Fallback: copy files only (shallow copy).
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                (dst / f.name).write_bytes(f.read_bytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=Path("data/panorama30_protocol"))
    ap.add_argument(
        "--source-roots",
        nargs="*",
        type=Path,
        default=[Path("data/paper_protocol"), Path("data/raw"), Path("data/asc2025/raw")],
        help="Priority-ordered dataset roots to link from.",
    )
    ap.add_argument("--datasets", nargs="*", default=None, help="Override dataset list (default: built-in PANORAMA30).")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_root = (repo_root / args.out_root).resolve() if not args.out_root.is_absolute() else args.out_root
    roots = [(repo_root / r).resolve() if not r.is_absolute() else r for r in args.source_roots]

    datasets = list(args.datasets) if args.datasets else list(PANORAMA30)
    missing = []
    for ds in datasets:
        src = _pick_source(ds, roots)
        if src is None:
            missing.append(ds)
            continue
        _symlink_or_copy(src, out_root / ds)

    if missing:
        raise SystemExit(f"Missing {len(missing)} datasets: {', '.join(missing)}")

    print(f"✓ Built dataset root: {out_root}")
    print(f"  datasets: {len(datasets)}")


if __name__ == "__main__":
    main()

