"""Run a trained SAGA-Net on kernel records of any along-track height.

    python predict.py --checkpoint runs/multi_speed/seed_314159/best.pt \
        --csv /path/to/labels.csv --split test --group-by speed \
        --output predictions_test.csv

Records are processed one at a time (batch size 1) in FP32, without speed
information, so 462 x H x 128 records with H = 128, 64, 26 or 13 (any H >= 10)
can be mixed. If labels are present, overall accuracy (OA), macro-F1,
quadratic weighted kappa (QWK) and the confusion matrix are printed, for all
records and per value of ``--group-by``.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dataset import HyperspectralKernelDataset
from saga_net import build_saga_net, remap_archived_state_dict

CLASS_NAMES = ("sound", "slight", "severe")
MODEL_KEYS = ("num_bands", "num_classes", "num_virtual_bands", "width", "use_gradient_stream", "use_cross_gate")


def load_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model_state_dict", payload))
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    model_config = model_config if isinstance(model_config, dict) else {}
    # Only keys understood by build_saga_net are used (checkpoints written by
    # other code fall back to the default configuration).
    model_config = {k: v for k, v in model_config.items() if k in MODEL_KEYS}
    model = build_saga_net(**model_config)
    model.load_state_dict(remap_archived_state_dict(state), strict=True)
    return model.to(device).eval()


def metrics(targets, predictions, num_classes: int = 3) -> dict:
    targets, predictions = np.asarray(targets), np.asarray(predictions)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, predictions):
        confusion[t, p] += 1
    n = confusion.sum()
    f1 = []
    for c in range(num_classes):
        tp = confusion[c, c]
        denominator = 2 * tp + confusion[:, c].sum() - tp + confusion[c, :].sum() - tp
        f1.append(2 * tp / denominator if denominator else 0.0)
    i, j = np.indices((num_classes, num_classes))
    weights = (i - j) ** 2 / (num_classes - 1) ** 2
    observed = confusion / n
    expected = np.outer(confusion.sum(1), confusion.sum(0)) / n**2
    denominator = (weights * expected).sum()
    qwk = 1.0 - (weights * observed).sum() / denominator if denominator else 0.0
    return {"n": int(n), "OA": float(np.trace(confusion) / n), "macro_F1": float(np.mean(f1)),
            "QWK": float(qwk), "confusion_matrix": confusion.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--split", default="test", help="split to evaluate ('all' for every row)")
    parser.add_argument("--group-by", default=None, help="CSV column for per-group metrics, e.g. speed")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = HyperspectralKernelDataset(args.csv, None if args.split == "all" else [args.split])
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            cube, label, _ = dataset[index]
            logits = model(cube.unsqueeze(0).to(device)).float()
            probabilities = logits.softmax(1)[0].cpu().tolist()
            record = dict(dataset.records[index])
            record.update(height=int(cube.shape[1]), prediction=int(np.argmax(probabilities)),
                          **{f"p_{name}": p for name, p in zip(CLASS_NAMES, probabilities)})
            rows.append(record)
    with open(args.output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} predictions to {args.output}")

    labelled = [r for r in rows if int(r["label"]) >= 0]
    if labelled:
        report = {"all": metrics([int(r["label"]) for r in labelled], [r["prediction"] for r in labelled])}
        if args.group_by:
            groups = defaultdict(list)
            for r in labelled:
                groups[str(r.get(args.group_by, ""))].append(r)
            for key, members in sorted(groups.items()):
                report[f"{args.group_by}={key}"] = metrics([int(r["label"]) for r in members],
                                                           [r["prediction"] for r in members])
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
