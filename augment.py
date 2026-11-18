"""Data augmentation and row-subsampling utilities for SAGA-Net.

* Fixed-size experiment (Section 2.7): random flipping, rotation, local
  erasing ("cut") and random resized cropping, applied per sample to
  C x 128 x 128 cubes. Ported from the archived training code, including its
  probabilities and random-number sources (NumPy for flip/rotate/cut, PyTorch
  for the resized crop).
* Multi-speed experiment: axis-preserving random flips (p = 0.5 per axis) and
  the optional uniform row-drop augmentation (Section 2.7, "+ row-drop").
* ``nominal_subsample`` builds the deterministic row-subsampled validation
  views used by the multi-speed checkpoint-selection loss (Section 2.6.2).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Fixed-size experiment augmentations (per sample, tensor C x H x W)
# --------------------------------------------------------------------------


class RandomFlip:
    """With probability p, apply one of {identity, flip H, flip W, flip H and W}."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand(1) < self.p:
            d = np.random.randint(0, 4)
            if d == 1:
                return x.flip(1)
            if d == 2:
                return x.flip(2)
            if d == 3:
                return x.flip([1, 2])
        return x


class RandomRotate:
    """With probability p, rotate by an integer angle in [-90, 89] degrees.

    Nearest-neighbour sampling about the image centre; coordinates falling
    outside the image are clamped to the border (as in the archived code).
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    @staticmethod
    def _rotate(x: torch.Tensor, degrees: int) -> torch.Tensor:
        theta = math.pi / 180 * degrees
        a = torch.tensor([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]], dtype=torch.float32, device=x.device)
        x = x.contiguous()
        channels, height, width = x.shape
        coords = torch.from_numpy(np.indices((height, width)).reshape(2, -1).T).float().to(x.device)
        coords[:, 0] -= height / 2.0 - 0.5
        coords[:, 1] -= width / 2.0 - 0.5
        new = coords @ a.T
        new[:, 0] += height / 2.0 - 0.5
        new[:, 1] += width / 2.0 - 0.5
        rows = torch.clamp(new[:, 0], 0, height - 1).round().long()
        cols = torch.clamp(new[:, 1], 0, width - 1).round().long()
        index = (rows * x.stride(1) + cols * x.stride(2)).unsqueeze(0).expand(channels, -1)
        return x.view(channels, -1).gather(1, index).view_as(x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand(1) < self.p:
            x = self._rotate(x, int(np.random.randint(-90, 90)))
        return x


class RandomCut:
    """With probability p, zero ``num_cuts`` random rectangles (local erasing)."""

    def __init__(self, p: float = 0.5, max_ratio: float = 0.5, num_cuts: int = 2):
        self.p = p
        self.max_ratio = max_ratio
        self.num_cuts = num_cuts

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand(1) >= self.p:
            return x
        h, w = x.shape[1:]
        x = x.clone()
        for _ in range(self.num_cuts):
            ratio = np.random.rand(1) * (self.max_ratio / self.num_cuts)
            cut_w = int(w * ratio)
            cut_h = int(h * ratio)
            pos_x = np.random.randint(0 - cut_w, w)
            pos_y = np.random.randint(0 - cut_h, h)
            # Index order kept from the archived code (identical for square inputs).
            x[:, max(0, pos_x):min(w, pos_x + cut_w), max(0, pos_y):min(h, pos_y + cut_h)] = 0.0
        return x


class RandomCrop:
    """With probability p, torchvision RandomResizedCrop back to the input size."""

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.rand(1) < self.p:
            from torchvision.transforms import RandomResizedCrop

            x = RandomResizedCrop(x.shape[-1], antialias=False)(x)
        return x


class Compose:
    def __init__(self, transforms: Sequence):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


def fixed_size_train_transform() -> Compose:
    """Augmentation pipeline of the fixed-size experiment (order and p as used)."""
    return Compose([RandomFlip(p=0.5), RandomRotate(p=0.5), RandomCut(p=0.5), RandomCrop(p=0.1)])


# --------------------------------------------------------------------------
# Multi-speed experiment
# --------------------------------------------------------------------------


class AxisPreservingFlip:
    """Flip columns and rows independently, each with probability p.

    Never exchanges the along-track and cross-track axes. Uses Python's
    ``random`` module, as in the archived multi-speed code.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return x
        if random.random() < self.p:
            x = x.flip(-1)
        if random.random() < self.p:
            x = x.flip(-2)
        return x


ROW_DROP_FACTORS = tuple(range(2, 11))


def row_drop(cube: torch.Tensor, factor: int, phases: Sequence[int]) -> torch.Tensor:
    """Keep every ``factor``-th row of each sample, starting at its own phase.

    ``cube`` is B x C x H x W. The output height is ceil(H / factor); samples
    whose phase leaves one row fewer are bottom-padded with a zero
    (background) row, exactly as in the archived implementation.
    """
    if len(phases) != cube.shape[0] or any(not 0 <= s < factor for s in phases):
        raise ValueError("invalid row-drop factor or per-sample phase")
    height = (cube.shape[-2] + factor - 1) // factor
    starts = torch.tensor(list(phases), dtype=torch.long, device=cube.device).view(-1, 1)
    indices = starts + factor * torch.arange(height, dtype=torch.long, device=cube.device).view(1, -1)
    padded = F.pad(cube, (0, 0, 0, factor), mode="constant", value=0)
    index = indices[:, None, :, None].expand(-1, cube.shape[1], -1, cube.shape[-1])
    return padded.gather(2, index).contiguous()


class RowDrop:
    """Uniform row-drop augmentation for 1x training mini-batches.

    Per mini-batch: keep the full height with probability ``keep_probability``;
    otherwise draw one stride r uniformly from {2, ..., 10} for the batch and
    an independent random start in {0, ..., r - 1} for every kernel. Uses its
    own ``random.Random(seed + 20261112)`` generator so that enabling it does
    not change the other random streams. Applied after flipping/collation,
    before transfer to the GPU.
    """

    RNG_OFFSET = 20261112

    def __init__(self, seed: int, keep_probability: float = 0.5,
                 factors: Sequence[int] = ROW_DROP_FACTORS):
        self.rng = random.Random(int(seed) + self.RNG_OFFSET)
        self.keep_probability = float(keep_probability)
        self.factors = tuple(int(f) for f in factors)

    def __call__(self, cube: torch.Tensor) -> torch.Tensor:
        if self.rng.random() < self.keep_probability:
            return cube
        factor = self.rng.choice(self.factors)
        phases = [self.rng.randrange(factor) for _ in range(cube.shape[0])]
        return row_drop(cube, factor, phases)


def nominal_subsample(cube: torch.Tensor, factor: int, phase_fraction: float = 0.5) -> torch.Tensor:
    """Deterministic row-subsampled view at a nominal speed factor.

    Selects rows round((i + phase_fraction) * factor - 0.5), i = 0..ceil(H/factor)-1
    (clamped to the valid range; torch.round rounds half to even), i.e. rows
    2i for factor 2, 5i+2 for factor 5 and 10i+4 for factor 10 when H = 128.
    Pixels whose spectrum is entirely zero stay zero. This is the "point"
    operator used for the multi-speed validation selection loss.
    """
    length = int(cube.shape[-2])
    output_length = int(math.ceil(length / factor))
    positions = (torch.arange(output_length, device=cube.device, dtype=torch.float32)
                 + phase_fraction) * factor - 0.5
    indices = positions.round().long().clamp_(0, length - 1)
    rows = cube.index_select(-2, indices)
    valid = (cube.abs().sum(dim=1, keepdim=True) > 0).to(cube.dtype).index_select(-2, indices)
    return rows * (valid >= 0.5).to(rows.dtype)
