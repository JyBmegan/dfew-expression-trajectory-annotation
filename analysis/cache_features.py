from __future__ import annotations

import argparse
import hashlib
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T

from app.frame_source import FrameSource
from .common import read_config, resolve_path


ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "training"))
from models.multi_channel_alexnet import MultiChannelAlexNet  # noqa: E402
from models.multi_channel_resnet import MultiChannelResNet  # noqa: E402


FEATURE_SOURCE = "model(tensors, return_features=True)"


class Frames(Dataset):
    def __init__(self, source: FrameSource, rows: pd.DataFrame):
        self.source = source
        self.items = [(int(row.video_name), int(row.label) - 1, frame) for row in rows.itertuples() for frame in range(1, 17)]
        self.transform = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        clip_id, label, frame = self.items[index]
        with Image.open(BytesIO(self.source.read(clip_id, frame))) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, clip_id, label, frame


def build_model(name: str, checkpoint: dict) -> torch.nn.Module:
    config = checkpoint.get("config", {})
    if name == "alexnet":
        model = MultiChannelAlexNet(
            in_channels=3, num_outputs=7,
            se_position=config.get("se_position"), reduction=config.get("reduction", 4),
        )
    else:
        model = MultiChannelResNet(
            in_channels=3, num_outputs=7, model_name="resnet18",
            se_position=config.get("se_position"), reduction=config.get("reduction", 4), pretrained=False,
        )
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state)
    return model.eval()


def extract_batch(name: str, model: torch.nn.Module, tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the feature tensor defined by each frozen model's public interface.

    In particular, MultiChannelAlexNet returns the output of its second 4096-unit
    linear layer before the following ReLU.  Calling ``classifier[:-1]`` here
    would instead cache the post-ReLU tensor and silently depart from the
    features used by the original temporal-model and representation scripts.
    """
    del name  # Both frozen backbones expose the same return contract.
    return model(tensors, return_features=True)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["alexnet", "resnet18"], required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output-dir", default="local_data/features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--checkpoint-batches", type=int, default=50,
        help="Flush memmaps and update resumable progress after this many batches",
    )
    parser.add_argument("--limit-clips", type=int, help="Development-only clip limit")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config); paths = config["paths"]
    rows = pd.read_csv(resolve_path(config, paths[f"{args.split}_csv"]))
    rows = rows[rows.label.between(1, 7)].drop_duplicates("video_name").sort_values("video_name").reset_index(drop=True)
    if args.limit_clips:
        rows = rows.iloc[:args.limit_clips].copy()
    source_value = paths.get("frames_16_archive", paths["frames"])
    source = FrameSource(resolve_path(config, source_value), paths.get("frames_archive_password"))
    if source.kind == "archive" and args.workers:
        raise ValueError("Archive-backed feature caching uses --workers 0 so one encrypted archive handle is shared safely")
    dataset = Frames(source, rows)
    checkpoint = torch.load(resolve_path(config, paths[f"{args.model}_checkpoint"]), map_location="cpu", weights_only=False)
    model = build_model(args.model, checkpoint)
    destination = Path(args.output_dir) / f"{args.model}_{args.split}"
    destination.mkdir(parents=True, exist_ok=True)
    feature_path = destination / "features.npy"
    logits_path = destination / "logits.npy"
    progress_path = destination / "progress.json"
    row_signature = hashlib.sha256(rows[["video_name", "label"]].to_csv(index=False).encode()).hexdigest()
    progress = None
    if args.overwrite:
        for path in [feature_path, logits_path, progress_path, destination / "metadata.json", destination / "frame_logits.csv.gz"]:
            path.unlink(missing_ok=True)
    elif progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {
            "model": args.model, "split": args.split, "items": len(dataset),
            "row_signature": row_signature, "feature_source": FEATURE_SOURCE,
        }
        if any(progress.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing feature-cache progress belongs to a different model, split, or clip list")
        if progress.get("status") == "complete":
            print(f"Feature cache is already complete: {destination}")
            source.close()
            return
    elif feature_path.exists() or logits_path.exists():
        raise ValueError(f"Incomplete cache lacks {progress_path}; inspect it or pass --overwrite explicitly")

    seen = int(progress.get("seen", 0)) if progress else 0
    if seen:
        feature_map = np.load(feature_path, mmap_mode="r+")
        logits_map = np.load(logits_path, mmap_mode="r+")
    else:
        feature_map = logits_map = None
    remaining = Subset(dataset, range(seen, len(dataset)))
    loader = DataLoader(remaining, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    batches_since_checkpoint = 0
    with torch.inference_mode():
        for tensors, _clip, _label, _frame in loader:
            logits, features = extract_batch(args.model, model, tensors)
            features = features.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            if feature_map is None:
                feature_map = np.lib.format.open_memmap(feature_path, mode="w+", dtype=np.float16, shape=(len(dataset), features.shape[1]))
                logits_map = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(len(dataset), 7))
            end = seen + len(features)
            feature_map[seen:end] = features.astype(np.float16)
            logits_map[seen:end] = logits.astype(np.float32)
            seen = end
            batches_since_checkpoint += 1
            if batches_since_checkpoint >= args.checkpoint_batches or seen == len(dataset):
                feature_map.flush(); logits_map.flush()
                write_json_atomic(progress_path, {
                    "model": args.model, "split": args.split, "items": len(dataset),
                    "row_signature": row_signature, "feature_source": FEATURE_SOURCE,
                    "seen": seen, "status": "running",
                })
                print(f"Cached {seen}/{len(dataset)} frames")
                batches_since_checkpoint = 0
    if feature_map is None or logits_map is None:
        raise RuntimeError("No frames were cached")
    feature_map.flush(); logits_map.flush()
    np.save(destination / "clip_ids.npy", rows.video_name.to_numpy(dtype=np.int32))
    np.save(destination / "labels.npy", rows.label.to_numpy(dtype=np.int16) - 1)
    long = pd.DataFrame({
        "model": args.model,
        "clip_id": np.repeat(rows.video_name.to_numpy(dtype=np.int32), 16),
        "frame": np.tile(np.arange(1, 17, dtype=np.int16), len(rows)),
        "label": np.repeat(rows.label.to_numpy(dtype=np.int16) - 1, 16),
    })
    for index in range(7):
        long[f"logit_{index}"] = np.asarray(logits_map[:, index])
    long.to_csv(destination / "frame_logits.csv.gz", index=False, compression="gzip")
    metadata = {
        "model": args.model, "split": args.split, "clips": len(rows), "frames_per_clip": 16,
        "feature_dimension": int(feature_map.shape[1]), "feature_dtype": "float16",
        "feature_source": FEATURE_SOURCE,
        "checkpoint": str(resolve_path(config, paths[f"{args.model}_checkpoint"])),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_json_atomic(progress_path, {
        "model": args.model, "split": args.split, "items": len(dataset),
        "row_signature": row_signature, "feature_source": FEATURE_SOURCE,
        "seen": seen, "status": "complete",
    })
    source.close()
    print(f"Feature cache complete: {destination}")


if __name__ == "__main__":
    main()
