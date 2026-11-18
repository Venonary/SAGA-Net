"""Train SAGA-Net with the fixed-size or the multi-speed training protocol.

    python train.py --config configs/fixed_size.yaml  --csv /path/to/labels.csv
    python train.py --config configs/multi_speed.yaml --csv /path/to/labels.csv [--row-drop]

The two protocols reproduce the archived training loops:

fixed_size
    Adam (lr 3e-3, weight decay 1e-6), batch 8, StepLR x0.1 every 10 epochs,
    50 epochs, AMP; flip/rotate/erase/crop augmentation; the checkpoint with
    the minimum validation cross-entropy is kept (validation computed under
    AMP). The archived loop also contained an early-stopping rule (stop after
    10 consecutive epochs whose validation loss does not decrease relative to
    the previous epoch); it never triggered in the reported runs (all ran 50
    epochs) and is kept for fidelity.

multi_speed
    Training only on the 1x (H = 128) records of the training kernels. AdamW
    (lr 3e-4, weight decay 1e-4), batch 2 with 4-step gradient accumulation,
    cosine annealing over 80 epochs, AMP, axis-preserving flips (p = 0.5),
    optional uniform row-drop. The selection loss is the mean of the FP32
    cross-entropy on the full-height 1x validation records and on
    row-subsampled views at nominal factors 2, 5 and 10; early stopping after
    15 epochs without improvement of the best selection loss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from augment import AxisPreservingFlip, RowDrop, fixed_size_train_transform, nominal_subsample
from dataset import HyperspectralKernelDataset
from saga_net import build_saga_net, count_parameters


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------


def set_global_seed(seed: int, deterministic: bool) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate(batch):
    cubes, labels, indices = zip(*batch)
    return torch.stack(cubes), torch.stack(labels), list(indices)


def make_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool, workers: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = dict(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                  num_workers=workers, collate_fn=collate, worker_init_fn=seed_worker,
                  generator=generator, pin_memory=torch.cuda.is_available())
    if workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_model(config: dict) -> nn.Module:
    return build_saga_net(**config.get("model", {}))


def save_checkpoint(path: Path, model: nn.Module, config: dict, **extra) -> None:
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"model_state": state, "config": config, **extra}, path)


def write_history(handle, row: dict) -> None:
    handle.write(json.dumps(row) + "\n")
    handle.flush()


# --------------------------------------------------------------------------
# Fixed-size protocol
# --------------------------------------------------------------------------


def train_fixed_size(config: dict, device: torch.device, output: Path) -> Path:
    data, train_cfg = config["data"], config["train"]
    seed = int(train_cfg["seed"])
    set_global_seed(seed, deterministic=bool(train_cfg.get("deterministic", False)))
    transform = fixed_size_train_transform() if train_cfg.get("augmentation", True) else None
    height = data.get("height")
    train_set = HyperspectralKernelDataset(data["csv"], [data.get("train_split", "train")],
                                           transform=transform, filters=data.get("train_filters"),
                                           expected_height=height)
    val_set = HyperspectralKernelDataset(data["csv"], [data.get("val_split", "val")],
                                         filters=data.get("val_filters"), expected_height=height)
    train_loader = make_loader(train_set, int(train_cfg["batch_size"]), True,
                               bool(train_cfg.get("drop_last", True)),
                               int(train_cfg.get("num_workers", 2)), seed)
    val_loader = make_loader(val_set, int(train_cfg.get("val_batch_size", train_cfg["batch_size"])),
                             False, False, 0, seed)
    model = build_model(config).to(device)
    print(f"[fixed_size] train={len(train_set)} val={len(val_set)} params={count_parameters(model)}",
          flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]),
                                 weight_decay=float(train_cfg["weight_decay"]))
    sched = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(sched.get("step_size", 10)),
                                                gamma=float(sched.get("gamma", 0.1)))
    criterion = nn.CrossEntropyLoss()
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    validation_amp = amp and bool(train_cfg.get("validation_amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    epochs = int(train_cfg["epochs"])
    patience = train_cfg.get("early_stopping_patience")  # consecutive-epoch rule; None disables
    best_loss, best_epoch, previous_loss, no_improvement = math.inf, 0, math.inf, 0
    checkpoint = output / "best.pt"
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            model.train()
            train_sum, train_n = 0.0, 0
            for x, y, _ in train_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, amp):
                    loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_sum += float(loss.detach()) * len(y)
                train_n += len(y)
            model.eval()
            val_sum, val_n, val_correct = 0.0, 0, 0
            with torch.inference_mode():
                for x, y, _ in val_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with autocast(device, validation_amp):
                        logits = model(x)
                        loss = criterion(logits, y)
                    val_sum += float(loss) * len(y)
                    val_n += len(y)
                    val_correct += int(logits.argmax(1).eq(y).sum())
            val_loss = val_sum / val_n
            row = {"epoch": epoch, "train_loss": train_sum / max(train_n, 1), "val_loss": val_loss,
                   "val_accuracy": val_correct / val_n, "lr": optimizer.param_groups[0]["lr"],
                   "seconds": time.perf_counter() - started}
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch
                save_checkpoint(checkpoint, model, config, epoch=epoch, selection_loss=val_loss, seed=seed)
            no_improvement = 0 if val_loss < previous_loss else no_improvement + 1
            previous_loss = val_loss
            row.update(best_epoch=best_epoch, best_val_loss=best_loss)
            write_history(history, row)
            print(json.dumps(row), flush=True)
            if patience is not None and no_improvement >= int(patience):
                print(f"[fixed_size] early stop: {patience} consecutive non-decreasing epochs", flush=True)
                break  # (the archived loop did not step the scheduler on this epoch)
            scheduler.step()
    print(f"[fixed_size] best epoch {best_epoch}, val loss {best_loss:.6f} -> {checkpoint}", flush=True)
    return checkpoint


# --------------------------------------------------------------------------
# Multi-speed protocol
# --------------------------------------------------------------------------


@torch.no_grad()
def selection_losses(model, loader, device, factors, phase_fraction):
    """FP32 cross-entropy on full-height and row-subsampled validation views."""
    model.eval()
    totals = {"1": 0.0, **{str(f): 0.0 for f in factors}}
    correct, count = 0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        totals["1"] += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int(logits.argmax(1).eq(y).sum())
        for factor in factors:
            view = nominal_subsample(x, int(factor), phase_fraction)
            totals[str(factor)] += float(F.cross_entropy(model(view), y, reduction="sum"))
        count += len(y)
    losses = {k: v / max(count, 1) for k, v in totals.items()}
    return losses, statistics.mean(losses.values()), correct / max(count, 1)


def train_multi_speed(config: dict, device: torch.device, output: Path) -> Path:
    data, train_cfg = config["data"], config["train"]
    seed = int(train_cfg["seed"])
    set_global_seed(seed, deterministic=bool(train_cfg.get("deterministic", True)))
    height = int(data.get("height", 128))
    flip = AxisPreservingFlip(float(train_cfg.get("flip_probability", 0.5)))
    train_set = HyperspectralKernelDataset(data["csv"], [data.get("train_split", "train")],
                                           transform=flip, filters=data.get("train_filters"),
                                           expected_height=height)
    val_set = HyperspectralKernelDataset(data["csv"], [data.get("val_split", "val")],
                                         filters=data.get("val_filters"), expected_height=height)
    batch_size = int(train_cfg["batch_size"])
    workers = int(train_cfg.get("num_workers", 0))
    train_loader = make_loader(train_set, batch_size, True, True, workers, seed)
    val_loader = make_loader(val_set, int(train_cfg.get("val_batch_size", batch_size)), False, False,
                             workers, seed)
    row_drop = RowDrop(seed, float(train_cfg.get("row_drop_keep_probability", 0.5))) \
        if train_cfg.get("row_drop", False) else None
    model = build_model(config).to(device)
    print(f"[multi_speed] train={len(train_set)} val={len(val_set)} row_drop={row_drop is not None} "
          f"params={count_parameters(model)}", flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=float(train_cfg["lr"]), weight_decay=float(train_cfg["weight_decay"]))
    epochs = int(train_cfg["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg.get("scheduler_t_max", epochs)))
    criterion = nn.CrossEntropyLoss()
    amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    accumulation = max(1, int(train_cfg.get("grad_accumulation", 4)))
    patience = int(train_cfg.get("patience", 15))
    factors = [int(f) for f in train_cfg.get("selection_factors", [2, 5, 10])]
    phase_fraction = float(train_cfg.get("selection_phase_fraction", 0.5))
    best_loss, best_epoch, stale = math.inf, -1, 0
    checkpoint = output / "best.pt"
    with (output / "history.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(epochs):
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_sum, train_n = 0.0, 0
            for batch_index, (x, y, _) in enumerate(train_loader):
                if row_drop is not None:
                    x = row_drop(x)  # CPU, after flipping/collation
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast(device, amp):
                    batch_loss = criterion(model(x), y)
                    loss = batch_loss / accumulation
                scaler.scale(loss).backward()
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                train_sum += float(batch_loss.detach()) * len(y)
                train_n += len(y)
            scheduler.step()
            losses, selection, val_accuracy = selection_losses(model, val_loader, device, factors,
                                                               phase_fraction)
            improved = selection < best_loss
            if improved:
                best_loss, best_epoch, stale = selection, epoch, 0
                save_checkpoint(checkpoint, model, config, epoch=epoch, selection_loss=selection, seed=seed)
            else:
                stale += 1
            row = {"epoch": epoch, "train_loss": train_sum / max(train_n, 1),
                   "val_loss_1x": losses["1"], "val_subsampled_losses": {f"{k}x": v for k, v in losses.items() if k != "1"},
                   "val_selection_loss": selection, "val_accuracy_1x": val_accuracy,
                   "lr": optimizer.param_groups[0]["lr"], "improved": improved, "stale_epochs": stale,
                   "seconds": time.perf_counter() - started}
            write_history(history, row)
            print(json.dumps(row), flush=True)
            if stale >= patience:
                print(f"[multi_speed] early stop after {patience} epochs without improvement", flush=True)
                break
    print(f"[multi_speed] best epoch {best_epoch}, selection loss {best_loss:.6f} -> {checkpoint}", flush=True)
    return checkpoint


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML file, e.g. configs/fixed_size.yaml")
    parser.add_argument("--csv", help="override data.csv")
    parser.add_argument("--output", help="override output_dir")
    parser.add_argument("--seed", type=int, help="override train.seed")
    parser.add_argument("--row-drop", action="store_true", help="multi_speed: enable row-drop augmentation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.csv:
        config["data"]["csv"] = args.csv
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.row_drop:
        config["train"]["row_drop"] = True
    if not config["data"].get("csv"):
        parser.error("data.csv is not set; pass --csv /path/to/labels.csv")
    output = Path(args.output or config.get("output_dir", "runs")) / f"seed_{config['train']['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = torch.device(args.device)
    protocol = config.get("protocol")
    if protocol == "fixed_size":
        train_fixed_size(config, device, output)
    elif protocol == "multi_speed":
        train_multi_speed(config, device, output)
    else:
        raise ValueError(f"unknown protocol {protocol!r}; expected fixed_size or multi_speed")


if __name__ == "__main__":
    main()
