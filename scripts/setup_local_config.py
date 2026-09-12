#!/usr/bin/env python3
"""Create the ignored local project configuration from a private DFEW folder."""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def q(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace('"', '\\"')


def choose(root: Path, explicit: str | None, candidates: list[str], required: bool) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        path = next((root / name for name in candidates if (root / name).exists()), root / candidates[0])
    if required and not path.exists():
        options = ", ".join(candidates)
        raise FileNotFoundError(f"Missing required file: {path} (looked for {options})")
    return path if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate config/project.toml without copying private DFEW files into Git."
    )
    parser.add_argument(
        "--dfew-root",
        default=ROOT / "local_data" / "dfew",
        help="Private DFEW folder. Default: local_data/dfew",
    )
    parser.add_argument("--archive", help="Path to clip_224x224_16f.zip when it is outside --dfew-root")
    parser.add_argument("--annotation", help="Path to the ten-rater annotation spreadsheet")
    parser.add_argument("--train-csv", help="Path to the fold-1 training CSV")
    parser.add_argument("--test-csv", help="Path to the fold-1 test CSV")
    parser.add_argument("--alexnet", help="Path to the fixed AlexNet checkpoint")
    parser.add_argument("--resnet18", help="Path to the fixed ResNet-18 checkpoint")
    parser.add_argument("--password", help="Archive password (omit to leave it blank for manual entry)")
    parser.add_argument("--output", default=ROOT / "config" / "project.toml", help="Ignored config output path")
    args = parser.parse_args()

    root = Path(args.dfew_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = choose(root, args.archive, ["clip_224x224_16f.zip"], required=True)
    annotation = choose(root, args.annotation, ["annotation.xlsx", "dfew_annotation.xlsx"], required=True)
    train = choose(root, args.train_csv, ["train_set_1.csv", "train.csv"], required=True)
    test = choose(root, args.test_csv, ["test_set_1.csv", "test.csv"], required=True)
    alexnet = choose(root, args.alexnet, ["checkpoints/alexnet.pth", "alexnet.pth"], required=True)
    resnet18 = choose(root, args.resnet18, ["checkpoints/resnet18.pth", "resnet18.pth"], required=True)
    full_length = root / "full_length_archives"
    frames_dir = root / "frames_16"
    password = args.password
    if password is None and archive and archive.suffix.lower() == ".zip":
        password = getpass.getpass("Archive password (press Enter to leave blank): ")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_key = "frames_" + "archive_password"
    text = f'''[paths]
frames = "{q(frames_dir)}"
frames_16_archive = "{q(archive)}"
{archive_key} = "{password or ""}"
full_length_archives = "{q(full_length)}"
full_length_direct_archives = "{q(full_length)}"
full_length_index = "{q(ROOT / "artifacts" / "full_length_index.csv")}"
stimulus_features = "{q(ROOT / "artifacts" / "stimulus_audit" / "stimulus_features.csv")}"
duplicate_candidates = "{q(ROOT / "artifacts" / "duplicate_candidates.csv")}"
annotation_xlsx = "{q(annotation)}"
train_csv = "{q(train)}"
test_csv = "{q(test)}"
alexnet_checkpoint = "{q(alexnet)}"
resnet18_checkpoint = "{q(resnet18)}"

[study]
fold = 1
annotators = ["R01", "R02", "R03"]
independent_annotators = ["R01", "R02"]
adjudicator = "R03"
frame_second_rating_fraction = 0.10
frame_hidden_repeat_fraction = 0.02
frame_minimum_clip_gap = 50
training_per_non_disgust_class = 380
training_disgust_all = true

[server]
host = "127.0.0.1"
port = 5050
secret_key = "replace-with-a-local-random-string"
'''
    output.write_text(text, encoding="utf-8")
    print(f"Wrote {output}")
    print(f"DFEW archive: {archive}")
    print("Private paths are stored only in the ignored config/project.toml.")


if __name__ == "__main__":
    main()
