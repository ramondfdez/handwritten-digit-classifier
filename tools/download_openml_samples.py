"""Download sample handwritten digit images from OpenML MNIST.

This script fetches public OpenML data and exports a small set of PNG files
that can be used with this repository's predictor CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sample handwritten digit PNGs from OpenML MNIST."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_images/openml_mnist"),
        help="Folder where PNG samples are saved.",
    )
    parser.add_argument(
        "--per-digit",
        type=int,
        default=2,
        help="How many images to export per digit (0-9).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_digit < 1:
        raise ValueError("--per-digit must be at least 1")

    print("Fetching OpenML MNIST (public dataset)...")
    x, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False,
        parser="liac-arff",
    )

    x = x.astype(np.uint8)
    y = y.astype(int)
    rng = np.random.default_rng(args.random_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    for digit in range(10):
        idx = np.flatnonzero(y == digit)
        chosen = rng.choice(idx, size=min(args.per_digit, len(idx)), replace=False)

        for i, sample_idx in enumerate(chosen, start=1):
            image = x[sample_idx].reshape(28, 28)
            out_path = args.output_dir / f"digit_{digit}_{i}.png"
            Image.fromarray(image, mode="L").save(out_path)
            exported += 1

    print(f"Exported {exported} images to: {args.output_dir}")


if __name__ == "__main__":
    main()
