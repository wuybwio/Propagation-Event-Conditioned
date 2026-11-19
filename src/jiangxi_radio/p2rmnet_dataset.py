"""Utilities for converting multi-height NPZ maps into spatial training patches."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def spatial_patch_split(shape: tuple[int, int], *, patch_size: int, stride: int,
                        val_fraction: float = 0.2, gap_pixels: int = 0) -> dict:
    """Split the raster first, discarding patches crossing the row boundary."""
    rows, cols = shape
    if not 0 < val_fraction < 1 or gap_pixels < 0 or patch_size <= 0 or stride <= 0:
        raise ValueError("Invalid spatial split parameters")
    last_start = ((rows - patch_size) // stride) * stride
    boundary = min(int(rows * (1.0 - val_fraction)), last_start - gap_pixels)
    boundary = (boundary // stride) * stride
    train, val, discarded = [], [], []
    positions = [(y, x) for y in range(0, rows - patch_size + 1, stride)
                 for x in range(0, cols - patch_size + 1, stride)]
    for index, (y, _) in enumerate(positions):
        if y + patch_size <= boundary:
            train.append(index)
        elif y >= boundary + gap_pixels:
            val.append(index)
        else:
            discarded.append(index)
    if not train or not val:
        raise ValueError("Raster is too small for disjoint train/val patches; reduce patch size or gap")
    return {"method": "disjoint_rows_v1", "shape": list(shape), "patch_size": patch_size,
            "stride": stride, "val_fraction": val_fraction, "gap_pixels": gap_pixels,
            "boundary_row": boundary, "train_indices": train, "val_indices": val,
            "discarded_indices": discarded}


def evaluation_patch_indices(checkpoint: dict, shape: tuple[int, int], *, patch_size: int,
                             stride: int, split: str, allow_legacy: bool = False) -> list[int]:
    count = len(range(0, shape[0] - patch_size + 1, stride)) * len(range(0, shape[1] - patch_size + 1, stride))
    spec = checkpoint.get("spatial_split")
    if spec is not None:
        if spec["shape"] != list(shape) or spec["patch_size"] != patch_size or spec["stride"] != stride:
            raise ValueError("Evaluation raster/patch geometry differs from the saved training split")
        if split == "all":
            return list(range(count))
        return list(spec[f"{split}_indices"])
    if not allow_legacy:
        raise ValueError("Checkpoint lacks a reproducible spatial split")
    boundary = min(max(1, int(count * 0.8)), count - 1)
    return list(range(count)) if split == "all" else list(range(boundary)) if split == "train" else list(range(boundary, count))


def extract_spatial_patches(
    arrays: Mapping[str, np.ndarray],
    *,
    patch_size: int = 32,
    stride: int = 16,
) -> list[dict[str, np.ndarray]]:
    """Extract aligned full patches, retaining every height as one sample.

    Arrays must have shape ``[height, channels, rows, columns]``.  Border
    regions smaller than a full patch are intentionally omitted; this keeps
    train/validation tensors shape-identical and avoids zero-padding artifacts.
    """
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if not arrays:
        return []
    first = next(iter(arrays.values()))
    values = np.asarray(first)
    if values.ndim != 4:
        raise ValueError("arrays must have shape [height, channels, rows, columns]")
    height_count, _, rows, columns = values.shape
    if rows < patch_size or columns < patch_size:
        raise ValueError("spatial dimensions must be at least patch_size")
    for name, array in arrays.items():
        value = np.asarray(array)
        if value.ndim != 4 or value.shape[0] != height_count or value.shape[-2:] != (rows, columns):
            raise ValueError(f"array {name!r} is not aligned with the first array")
    patches: list[dict[str, np.ndarray]] = []
    row_starts = range(0, rows - patch_size + 1, stride)
    col_starts = range(0, columns - patch_size + 1, stride)
    for row_start in row_starts:
        for col_start in col_starts:
            row_slice = slice(row_start, row_start + patch_size)
            col_slice = slice(col_start, col_start + patch_size)
            patches.append({
                name: np.asarray(array[..., row_slice, col_slice]).copy()
                for name, array in arrays.items()
            })
    return patches
