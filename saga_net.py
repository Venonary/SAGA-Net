"""SAGA-Net: spectral amplitude-gradient aggregation network.

Reference implementation of the network described in Section 2.5 of

    "Amplitude-gradient spatial-spectral network with cross-speed global
    aggregation for mold-severity grading of maize kernels in pushbroom
    hyperspectral imaging" (Frontiers in Plant Science).

The code reproduces, layer by layer and in the same construction order (so a
given random seed yields the same initial weights), the archived model that
produced the reported results. Only module names were changed for
readability; ``remap_archived_state_dict`` converts checkpoints that use the
archived names (see ``ARCHIVED_KEY_PREFIXES``).

Input:  float tensor ``B x 462 x H x 128`` (reflectance, zero background).
Output: ``B x 3`` logits (0 = sound, 1 = slight, 2 = severe).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "OrderedGaussianAggregation",
    "CrossStreamGating",
    "SAGANet",
    "build_saga_net",
    "count_parameters",
    "remap_archived_state_dict",
]


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _spectral_kernel_sizes(num_virtual_bands: int) -> tuple[int, int]:
    """Spectral kernel depths of the two Conv3d layers (24 and 16 for K=40)."""
    if num_virtual_bands >= 40:
        return 24, 16
    if num_virtual_bands < 5:
        raise ValueError(f"at least five virtual bands are required, got {num_virtual_bands}")
    kernel_1 = max(3, round(num_virtual_bands * 24 / 40))
    kernel_2 = max(3, round(num_virtual_bands * 16 / 40))
    if kernel_1 + kernel_2 > num_virtual_bands + 1:
        kernel_2 = num_virtual_bands + 1 - kernel_1
    return kernel_1, kernel_2


class OrderedGaussianAggregation(nn.Module):
    """Fixed ordered Gaussian aggregation of C bands into K virtual bands.

    a_{k,c} = exp(-(u_c - mu_k)^2 / (2 sigma_k^2)) / (sum_c' exp(...) + 1e-8)
    V_k     = g_k * sum_c a_{k,c} X_c

    with u_c = (c - 1/2) / C, mu_k = (k - 1/2) / K, sigma_k = 0.5 / K + 1e-4 and
    g_k = 1. As in the archived implementation, mu_k and sigma_k are stored
    through the parameterization widths = softmax(delta), centers =
    cumsum(widths) - widths / 2, sigma = softplus(rho) * widths + 1e-4 with
    delta = 0 and softplus(rho) = 0.5. ``delta``, ``rho`` and ``gain`` (3K
    values) are registered as non-trainable parameters (requires_grad=False),
    so they appear in the total parameter count (2 streams x 3 x 40 = 240) but
    are never updated.
    """

    def __init__(self, in_bands: int, out_bands: int):
        super().__init__()
        self.in_bands = int(in_bands)
        self.out_bands = int(out_bands)
        self.delta = nn.Parameter(torch.zeros(out_bands))
        self.rho = nn.Parameter(torch.full((out_bands,), _inverse_softplus(0.5)))
        self.gain = nn.Parameter(torch.ones(out_bands))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        positions = (torch.arange(in_bands, dtype=torch.float32) + 0.5) / in_bands
        self.register_buffer("positions", positions)

    def band_weights(self) -> torch.Tensor:
        """Return the K x C aggregation matrix a_{k,c}."""
        widths = torch.softmax(self.delta, dim=0)
        edges = torch.cumsum(widths, dim=0)
        centers = edges - widths / 2
        sigma = F.softplus(self.rho) * widths + 1e-4
        distance = self.positions.unsqueeze(0) - centers.unsqueeze(1)
        weights = torch.exp(-0.5 * (distance / sigma.unsqueeze(1)) ** 2)
        return weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        weights = self.band_weights().to(dtype=cube.dtype)
        output = torch.einsum("kc,bchw->bkhw", weights, cube)
        return output * self.gain.view(1, -1, 1, 1)


class CrossStreamGating(nn.Module):
    """A' = A * (1 + tanh(phi_G(G))),  G' = G * (1 + tanh(phi_A(A))).

    phi_G and phi_A are 1x1 convolutions whose weights and biases are
    initialized to zero, so both gates start at exactly 1.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.phi_g = nn.Conv2d(channels, channels, kernel_size=1)  # gates A from G
        self.phi_a = nn.Conv2d(channels, channels, kernel_size=1)  # gates G from A
        for layer in (self.phi_g, self.phi_a):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, amplitude: torch.Tensor, gradient: torch.Tensor):
        amplitude_out = amplitude * (1.0 + torch.tanh(self.phi_g(gradient)))
        gradient_out = gradient * (1.0 + torch.tanh(self.phi_a(amplitude)))
        return amplitude_out, gradient_out


class SAGANet(nn.Module):
    """Spectral amplitude-gradient aggregation network (SAGA-Net).

    Args:
        num_bands: number of measured bands C (462 in the paper).
        num_classes: number of grades (3).
        num_virtual_bands: K, virtual bands per stream (40).
        width: required cross-track width W (128). The along-track height H
            may vary (H >= 10); records of 128, 64, 26 and 13 rows are accepted.
        use_gradient_stream: False gives the single-stream (amplitude only)
            ablation of Section 2.6.1, which also removes cross-stream gating.
        use_cross_gate: enable cross-stream gating (paper model: True).
    """

    def __init__(
        self,
        num_bands: int = 462,
        num_classes: int = 3,
        num_virtual_bands: int = 40,
        width: int = 128,
        use_gradient_stream: bool = True,
        use_cross_gate: bool = True,
    ):
        super().__init__()
        if num_virtual_bands < 5:
            raise ValueError("num_virtual_bands must be at least five")
        if num_virtual_bands > num_bands - 1:
            raise ValueError("num_virtual_bands cannot exceed the number of differenced bands")
        self.num_bands = int(num_bands)
        self.num_classes = int(num_classes)
        self.num_virtual_bands = int(num_virtual_bands)
        self.width = int(width)
        self.use_gradient_stream = bool(use_gradient_stream)
        # Two valid 5x5 convolutions followed by a 2x2 max pooling.
        self.minimum_height = 2 * (5 - 1) + 2

        k = self.num_virtual_bands
        # NOTE: module construction order matches the archived implementation
        # so that torch.manual_seed(s) gives identical initial weights.
        self.amp_aggregation = OrderedGaussianAggregation(num_bands, k)
        self.amp_proj = nn.Sequential(
            nn.Conv2d(k, k, kernel_size=1, bias=False),
            nn.BatchNorm2d(k),
            nn.ReLU(inplace=True),
        )
        if self.use_gradient_stream:
            self.grad_aggregation = OrderedGaussianAggregation(num_bands - 1, k)
            self.grad_proj = nn.Sequential(
                nn.Conv2d(k, k, kernel_size=1, bias=False),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True),
            )
            self.cross_gate = CrossStreamGating(k) if use_cross_gate else None
        else:
            self.grad_aggregation = None
            self.grad_proj = None
            self.cross_gate = None

        in_streams = 2 if self.use_gradient_stream else 1
        kernel_1, kernel_2 = _spectral_kernel_sizes(k)
        self.spectral_out = k - kernel_1 - kernel_2 + 2  # 40 -> 17 -> 2
        self.encoder = nn.Sequential(
            nn.Conv3d(in_streams, 32, kernel_size=(kernel_1, 5, 5)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(kernel_2, 5, 5)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2)),
        )
        self.embedding_dim = 64 * self.spectral_out  # 128
        self.classifier = nn.Linear(self.embedding_dim, num_classes)  # 387 parameters

    def _check_input(self, cube: torch.Tensor) -> None:
        if cube.ndim != 4:
            raise ValueError(f"expected B x C x H x W input, got {tuple(cube.shape)}")
        if cube.shape[1] != self.num_bands:
            raise ValueError(f"expected {self.num_bands} bands, got {cube.shape[1]}")
        height, width = int(cube.shape[-2]), int(cube.shape[-1])
        if width != self.width:
            raise ValueError(f"SAGA-Net requires cross-track width {self.width}, got {width}")
        if height < self.minimum_height:
            raise ValueError(
                f"along-track height {height} is too small; minimum is {self.minimum_height}"
            )

    def dual_stream(self, cube: torch.Tensor) -> torch.Tensor:
        """Return the stacked B x S x K x H x W volume (S = 2 streams)."""
        amplitude = self.amp_proj(self.amp_aggregation(cube))
        if not self.use_gradient_stream:
            return amplitude.unsqueeze(1)
        gradient = self.grad_aggregation(cube[:, 1:] - cube[:, :-1])  # D_c = X_{c+1} - X_c
        gradient = self.grad_proj(gradient)
        if self.cross_gate is not None:
            amplitude, gradient = self.cross_gate(amplitude, gradient)
        return torch.stack([amplitude, gradient], dim=1)

    def forward(self, cube: torch.Tensor, return_embedding: bool = False):
        self._check_input(cube)
        features = self.encoder(self.dual_stream(cube))  # B x 64 x 2 x H' x W'
        embedding = features.mean(dim=(-2, -1)).flatten(1)  # spatial global average -> B x 128
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


def build_saga_net(**overrides) -> SAGANet:
    """Build the paper configuration (C=462, K=40, W=128, 3 classes)."""
    config = dict(num_bands=462, num_classes=3, num_virtual_bands=40, width=128,
                  use_gradient_stream=True, use_cross_gate=True)
    config.update(overrides)
    return SAGANet(**config)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "non_trainable": total - trainable}


# Archived (internal) module prefix -> public prefix used in this release.
ARCHIVED_KEY_PREFIXES = (
    ("avp_amplitude.", "amp_aggregation."),
    ("avp_gradient.", "grad_aggregation."),
    ("refine_amplitude.", "amp_proj."),
    ("refine_gradient.", "grad_proj."),
    ("cross_gate.amplitude_from_gradient.", "cross_gate.phi_g."),
    ("cross_gate.gradient_from_amplitude.", "cross_gate.phi_a."),
    ("features.", "encoder."),
    ("classifier.", "classifier."),
)


def remap_archived_state_dict(state_dict: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Rename keys of a checkpoint saved with the archived module names."""
    remapped: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        for old, new in ARCHIVED_KEY_PREFIXES:
            if key.startswith(old):
                key = new + key[len(old):]
                break
        remapped[key] = value
    return remapped
