"""Minimal dataset reader for SAGA-Net.

Expected layout (user-provided; nothing here refers to the authors' storage)::

    data_root/
        labels.csv
        cubes/kernel_0001_1x.npy
        cubes/kernel_0001_2x.npy
        ...

``labels.csv`` must have the columns ``path,label,split`` and may contain any
additional columns (e.g. ``kernel_id``, ``speed``), which are passed through
to the prediction output::

    path,label,split,kernel_id,speed
    cubes/kernel_0001_1x.npy,0,train,kernel_0001,1x
    cubes/kernel_0001_2x.npy,0,train,kernel_0001,2x

* ``path``: ``.npy`` file relative to the CSV directory (or absolute).
* ``label``: 0 = sound, 1 = slight, 2 = severe (may be empty for unlabelled
  inference; it is then stored as -1).
* ``split``: ``train``, ``val`` or ``test``.

Each ``.npy`` file holds one kernel record as a float array of shape
``462 x H x 128`` (bands x along-track rows x cross-track columns) with
radiometrically corrected reflectance and voxels outside the kernel mask set
to zero. Arrays stored as ``H x 128 x 462`` are transposed automatically.
No normalization is applied (none was used in our experiments).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_BANDS = 462


def load_cube(path: str | Path, num_bands: int = NUM_BANDS) -> torch.Tensor:
    array = np.load(path, mmap_mode="r")
    if array.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D cube, got shape {array.shape}")
    if array.shape[0] != num_bands and array.shape[-1] == num_bands:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] != num_bands:
        raise ValueError(f"{path}: expected {num_bands} bands, got shape {array.shape}")
    return torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True))


class HyperspectralKernelDataset(Dataset):
    """Kernel records listed in a CSV file.

    Args:
        csv_path: path of ``labels.csv``.
        splits: keep rows whose ``split`` is in this collection (None: all rows).
        transform: callable applied to each C x H x W cube.
        filters: optional ``{column: allowed values}`` (e.g. ``{"speed": ["1x"]}``).
        expected_height: if given, raise when a record has a different height
            (training batches must share one height).
    """

    def __init__(
        self,
        csv_path: str | Path,
        splits: Optional[Iterable[str]] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        filters: Optional[dict[str, Iterable[str]]] = None,
        expected_height: Optional[int] = None,
        num_bands: int = NUM_BANDS,
    ):
        self.csv_path = Path(csv_path)
        self.root = self.csv_path.parent
        self.transform = transform
        self.expected_height = expected_height
        self.num_bands = num_bands
        allowed = None if splits is None else {str(s) for s in splits}
        filters = {k: {str(v) for v in vs} for k, vs in (filters or {}).items()}
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = {"path", "label", "split"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
            rows = [dict(row) for row in reader]
        self.records = []
        for row in rows:
            if allowed is not None and row["split"].strip() not in allowed:
                continue
            if any(row.get(column, "").strip() not in values for column, values in filters.items()):
                continue
            label = row["label"].strip()
            row["label"] = int(label) if label != "" else -1
            if row["label"] not in (-1, 0, 1, 2):
                raise ValueError(f"label must be 0, 1 or 2, got {row['label']} ({row['path']})")
            self.records.append(row)
        if not self.records:
            raise ValueError(f"no records selected from {csv_path} (splits={allowed}, filters={filters})")

    def __len__(self) -> int:
        return len(self.records)

    def resolve(self, path: str) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def __getitem__(self, index: int):
        record = self.records[index]
        cube = load_cube(self.resolve(record["path"]), self.num_bands)
        if self.expected_height is not None and cube.shape[1] != self.expected_height:
            raise ValueError(
                f"{record['path']}: height {cube.shape[1]} != expected {self.expected_height}"
            )
        if self.transform is not None:
            cube = self.transform(cube)
        return cube, torch.tensor(record["label"], dtype=torch.long), index
