"""Smoke test for SAGA-Net (CPU, random inputs, no data needed).

    python tests/smoke_test.py        (or: python -m pytest tests)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from augment import RowDrop, nominal_subsample, row_drop  # noqa: E402
from saga_net import build_saga_net, count_parameters  # noqa: E402

EXPECTED_TOTAL = 865_155        # reported as 865.2K
EXPECTED_NON_TRAINABLE = 240    # fixed aggregation parameters (2 streams x 3 x 40)
EXPECTED_CLASSIFIER = 387       # Linear 128 -> 3
HEIGHTS = {128: 60, 64: 28, 26: 9, 13: 2}  # input rows H -> pooled height H'


def test_parameter_count():
    model = build_saga_net()
    counts = count_parameters(model)
    print("parameters:", counts)
    assert counts["total"] == EXPECTED_TOTAL, counts
    assert counts["non_trainable"] == EXPECTED_NON_TRAINABLE, counts
    assert round(counts["total"] / 1000, 1) == 865.2
    assert sum(p.numel() for p in model.classifier.parameters()) == EXPECTED_CLASSIFIER


def test_aggregation_weights():
    model = build_saga_net()
    for module, bands in ((model.amp_aggregation, 462), (model.grad_aggregation, 461)):
        k = torch.arange(1, 41, dtype=torch.float32)
        c = torch.arange(1, bands + 1, dtype=torch.float32)
        mu, sigma, u = (k - 0.5) / 40, 0.5 / 40 + 1e-4, (c - 0.5) / bands
        g = torch.exp(-((u[None, :] - mu[:, None]) ** 2) / (2 * sigma**2))
        expected = g / (g.sum(1, keepdim=True) + 1e-8)
        assert torch.allclose(module.band_weights(), expected, atol=1e-6)
        assert torch.allclose(module.gain, torch.ones(40))


def test_forward_variable_heights():
    torch.manual_seed(0)
    model = build_saga_net().eval()
    with torch.no_grad():
        for height, pooled in HEIGHTS.items():
            x = torch.rand(2, 462, height, 128)
            x[:, :, :2] = 0.0  # zero background rows
            volume = model.dual_stream(x)
            assert volume.shape == (2, 2, 40, height, 128)
            features = model.encoder(volume)
            assert features.shape == (2, 64, 2, pooled, 60), features.shape
            logits, embedding = model(x, return_embedding=True)
            assert logits.shape == (2, 3) and embedding.shape == (2, 128)
            assert torch.isfinite(logits).all()
            print(f"H={height:3d}: features {tuple(features.shape)} -> logits {tuple(logits.shape)}")


def test_gating_starts_at_identity():
    model = build_saga_net().eval()
    a, g = torch.rand(1, 40, 13, 128), torch.rand(1, 40, 13, 128)
    a2, g2 = model.cross_gate(a, g)
    assert torch.equal(a, a2) and torch.equal(g, g2)


def test_single_stream_ablation():
    model = build_saga_net(use_gradient_stream=False).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 462, 26, 128)).shape == (1, 3)


def test_row_operations():
    x = torch.arange(128, dtype=torch.float32).view(1, 1, 128, 1).expand(2, 3, 128, 4).clone() + 1
    for factor in range(2, 11):
        out = row_drop(x, factor, [0, factor - 1])
        assert out.shape[-2] == math.ceil(128 / factor)
        assert out[0, 0, :, 0].tolist()[:2] == [1.0, 1.0 + factor]
    for factor, first in ((2, [0, 2, 4]), (5, [2, 7, 12]), (10, [4, 14, 24])):
        view = nominal_subsample(x, factor)
        assert view.shape[-2] == math.ceil(128 / factor)
        assert (view[0, 0, :3, 0] - 1).tolist() == first
    augment = RowDrop(seed=0)
    heights = {augment(x).shape[-2] for _ in range(200)}
    assert 128 in heights and heights <= {128} | {math.ceil(128 / r) for r in range(2, 11)}


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
    print("All smoke tests passed.")
