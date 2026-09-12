from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import classification_metrics, parse_curve, true_class_margin


ORDER_CONDITIONS = ["natural", "reverse", "weak_to_strong", "strong_to_weak"]


def cache_arrays(folder: Path):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    n, d = metadata["clips"], metadata["feature_dimension"]
    features = np.load(folder / "features.npy", mmap_mode="r").reshape(n, 16, d)
    return features, np.load(folder / "clip_ids.npy"), np.load(folder / "labels.npy"), metadata


def curves_by_id(path: str | Path, split: str | None = None) -> dict[int, np.ndarray]:
    frame = pd.read_csv(path)
    if split and "split" in frame:
        frame = frame[frame.split == split]
    return {int(row.clip_id): parse_curve(row) for _, row in frame.iterrows()}


def order_indices(curve: np.ndarray | None, condition: str, permutation: np.ndarray | None = None) -> np.ndarray:
    if condition == "natural": return np.arange(16)
    if condition == "reverse": return np.arange(15, -1, -1)
    if condition == "shuffle": return permutation
    if curve is None: raise ValueError(f"Condition {condition} requires a human intensity curve")
    ascending = np.argsort(curve, kind="stable")
    return ascending if condition == "weak_to_strong" else np.argsort(-curve, kind="stable")


class SequenceDataset(Dataset):
    def __init__(self, features, clip_ids, labels, indices, condition, curves=None):
        self.features = features; self.clip_ids = clip_ids; self.labels = labels
        self.indices = np.asarray(indices); self.condition = condition; self.curves = curves or {}
    def __len__(self): return len(self.indices)
    def __getitem__(self, position):
        index = int(self.indices[position]); clip_id = int(self.clip_ids[index])
        order = order_indices(self.curves.get(clip_id), self.condition)
        values = np.asarray(self.features[index][order], dtype=np.float32)
        return torch.from_numpy(values), int(self.labels[index]), clip_id


class AveragePoolHead(nn.Module):
    def __init__(self, input_dim, classes=7): super().__init__(); self.classifier = nn.Linear(input_dim, classes)
    def forward(self, x): return self.classifier(x.mean(dim=1))


class LastFrameHead(nn.Module):
    """Endpoint-only control for ordered heads that read out at the final step."""
    def __init__(self, input_dim, classes=7): super().__init__(); self.classifier = nn.Linear(input_dim, classes)
    def forward(self, x): return self.classifier(x[:, -1])


class GRUHead(nn.Module):
    def __init__(self, input_dim, hidden=128, bidirectional=False, classes=7):
        super().__init__(); self.input = nn.Linear(input_dim, 256); self.gru = nn.GRU(256, hidden, batch_first=True, bidirectional=bidirectional); self.classifier = nn.Linear(hidden * (2 if bidirectional else 1), classes)
    def forward(self, x):
        _, hidden = self.gru(torch.relu(self.input(x)))
        summary = torch.cat([hidden[-2], hidden[-1]], dim=1) if self.gru.bidirectional else hidden[-1]
        return self.classifier(summary)


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__(); self.pad = 2 * dilation; self.conv = nn.Conv1d(channels, channels, 3, dilation=dilation); self.norm = nn.BatchNorm1d(channels)
    def forward(self, x):
        value = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return torch.relu(self.norm(value) + x)


class CausalTCNHead(nn.Module):
    def __init__(self, input_dim, hidden=128, classes=7):
        super().__init__(); self.input = nn.Conv1d(input_dim, hidden, 1); self.blocks = nn.Sequential(*(CausalBlock(hidden, d) for d in [1, 2, 4])); self.classifier = nn.Linear(hidden, classes)
    def forward(self, x):
        value = self.blocks(torch.relu(self.input(x.transpose(1, 2))))
        return self.classifier(value[:, :, -1])


def make_head(name: str, dimension: int) -> nn.Module:
    if name == "average": return AveragePoolHead(dimension)
    if name == "last_frame": return LastFrameHead(dimension)
    if name == "unidirectional_gru": return GRUHead(dimension, bidirectional=False)
    if name == "bidirectional_gru": return GRUHead(dimension, bidirectional=True)
    if name == "causal_tcn": return CausalTCNHead(dimension)
    raise ValueError(name)


def evaluate_loader(model, loader):
    model.eval(); logits, labels, ids = [], [], []
    with torch.inference_mode():
        for x, y, clip_id in loader:
            logits.append(model(x).cpu().numpy()); labels.append(y.numpy()); ids.append(clip_id.numpy())
    return np.concatenate(ids), np.concatenate(labels), np.concatenate(logits)


def train_one(model, train_loader, val_loader, class_weights, epochs, destination):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best, stale, history = -np.inf, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); losses = []
        for x, y, _ in train_loader:
            optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step(); losses.append(float(loss.detach()))
        _, labels, logits = evaluate_loader(model, val_loader)
        score = f1_score(labels, logits.argmax(axis=1), labels=np.arange(7), average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "validation_macro_f1": score})
        if score > best + 1e-5:
            best, stale = score, 0; torch.save(model.state_dict(), destination)
        else:
            stale += 1
            if stale >= 7: break
    model.load_state_dict(torch.load(destination, map_location="cpu", weights_only=True))
    return pd.DataFrame(history)


def training(args):
    torch.manual_seed(42); np.random.seed(42); torch.use_deterministic_algorithms(True)
    if args.reference_full and args.train_order != "natural":
        raise ValueError("The all-training-data reference uses natural order")
    if args.head == "average" and args.train_order != "natural":
        raise ValueError("Average pooling is order-invariant and is trained once in natural order")
    folder = Path(args.feature_root) / f"{args.backbone}_train"
    features, clip_ids, labels, metadata = cache_arrays(folder)
    selection = pd.read_csv(args.selection)
    selected_ids = set(selection.clip_id.astype(int))
    indices = np.asarray([i for i, clip_id in enumerate(clip_ids) if int(clip_id) in selected_ids])
    missing_features = selected_ids - set(clip_ids.astype(int))
    if missing_features:
        raise ValueError(f"Training feature cache is missing {len(missing_features)} selected clips")
    if len(indices) != len(selected_ids):
        raise ValueError("Training selection or feature cache contains duplicate clip IDs")
    if args.reference_full:
        indices = np.arange(len(clip_ids)); condition = "natural"; curves = {}
    else:
        condition = args.train_order; curves = curves_by_id(args.trajectories, "train_alignment")
        missing = selected_ids - set(curves)
        if missing: raise ValueError(f"Missing training trajectories for {len(missing)} selected clips")
    split = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=20260912)
    train_local, val_local = next(split.split(indices, labels[indices]))
    train_indices, val_indices = indices[train_local], indices[val_local]
    train_set = SequenceDataset(features, clip_ids, labels, train_indices, condition, curves)
    val_set = SequenceDataset(features, clip_ids, labels, val_indices, condition, curves)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    counts = np.bincount(labels[train_indices], minlength=7); weights = len(train_indices) / (7 * np.maximum(counts, 1))
    model = make_head(args.head, metadata["feature_dimension"])
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    tag = f"{args.backbone}_{args.head}_{'full_' if args.reference_full else ''}{condition}"
    checkpoint = output / f"{tag}.pth"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(f"{checkpoint} exists; pass --overwrite to retrain explicitly")
    history = train_one(model, train_loader, val_loader, torch.tensor(weights, dtype=torch.float32), args.epochs, checkpoint)
    history.to_csv(output / f"{tag}_history.csv", index=False)
    (output / f"{tag}.json").write_text(json.dumps({
        "backbone": args.backbone, "head": args.head, "train_order": condition,
        "reference_full": args.reference_full, "train_clip_ids": clip_ids[train_indices].astype(int).tolist(),
        "validation_clip_ids": clip_ids[val_indices].astype(int).tolist(), "feature_dimension": metadata["feature_dimension"],
    }, indent=2), encoding="utf-8")
    print(f"Saved {checkpoint}")


def evaluation(args):
    folder = Path(args.feature_root) / f"{args.backbone}_test"
    features, clip_ids, labels, metadata = cache_arrays(folder)
    curves = curves_by_id(args.trajectories, "test")
    missing_curves = set(clip_ids.astype(int)) - set(curves)
    if missing_curves:
        raise ValueError(f"Test trajectories are missing for {len(missing_curves)} clips")
    model = make_head(args.head, metadata["feature_dimension"])
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)); model.eval()
    training_metadata_path = Path(args.checkpoint).with_suffix(".json")
    training_metadata = json.loads(training_metadata_path.read_text(encoding="utf-8")) if training_metadata_path.exists() else {}
    permutations = [np.random.default_rng(7000 + index).permutation(16) for index in range(20)]
    conditions = [(name, None) for name in ORDER_CONDITIONS] + [(f"shuffle_{index + 1:02d}", permutation) for index, permutation in enumerate(permutations)]
    rows, metrics = [], []
    for name, permutation in conditions:
        indices = np.arange(len(clip_ids))
        if permutation is None:
            dataset = SequenceDataset(features, clip_ids, labels, indices, name, curves)
        else:
            class FixedShuffleDataset(SequenceDataset):
                def __getitem__(self, position):
                    index = int(self.indices[position]); values = np.asarray(self.features[index][permutation], dtype=np.float32)
                    return torch.from_numpy(values), int(self.labels[index]), int(self.clip_ids[index])
            dataset = FixedShuffleDataset(features, clip_ids, labels, indices, "natural", curves)
        ids, y, logits = evaluate_loader(model, DataLoader(dataset, batch_size=args.batch_size, shuffle=False))
        prediction = logits.argmax(axis=1); margin = true_class_margin(logits, y)
        shared = {
            "backbone": args.backbone, "head": args.head,
            "train_order": training_metadata.get("train_order", "unknown"),
            "reference_full": training_metadata.get("reference_full", False),
        }
        metrics.append({**shared, "test_order": name, **classification_metrics(y, prediction)})
        for i, clip_id in enumerate(ids):
            rows.append({
                **shared, "clip_id": int(clip_id), "label": int(y[i]), "test_order": name,
                "prediction": int(prediction[i]), "true_class_logit_margin": float(margin[i]),
                **{f"logit_{j}": float(logits[i, j]) for j in range(7)},
            })
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    tag = Path(args.checkpoint).stem
    pd.DataFrame(rows).to_csv(output / f"{tag}_test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(metrics).to_csv(output / f"{tag}_test_metrics.csv", index=False)
    pd.DataFrame({
        "shuffle": [f"shuffle_{index + 1:02d}" for index in range(20)],
        "positions_zero_based_json": [json.dumps(value.tolist()) for value in permutations],
    }).to_csv(output / "fixed_test_permutations.csv", index=False)
    print(f"Evaluation complete for {tag}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--backbone", choices=["alexnet", "resnet18"], required=True)
    shared.add_argument("--head", choices=["average", "last_frame", "unidirectional_gru", "bidirectional_gru", "causal_tcn"], required=True)
    shared.add_argument("--feature-root", default="local_data/features")
    shared.add_argument("--trajectories", required=True)
    shared.add_argument("--batch-size", type=int, default=64)
    shared.add_argument("--output-dir", default="artifacts/temporal_models")
    train = sub.add_parser("train", parents=[shared]); train.add_argument("--selection", required=True); train.add_argument("--train-order", choices=["natural","weak_to_strong","strong_to_weak"], default="natural"); train.add_argument("--epochs", type=int, default=50); train.add_argument("--reference-full", action="store_true"); train.add_argument("--overwrite", action="store_true"); train.set_defaults(func=training)
    evaluate = sub.add_parser("evaluate", parents=[shared]); evaluate.add_argument("--checkpoint", required=True); evaluate.set_defaults(func=evaluation)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
