from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BACKBONES = ("alexnet", "resnet18")
HEADS = ("average", "last_frame", "unidirectional_gru", "bidirectional_gru", "causal_tcn")
ALIGNED_ORDERS = ("natural", "weak_to_strong", "strong_to_weak")


def run(command: list[str], dry_run: bool) -> None:
    print(" ".join(shlex.quote(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def paths_for(output: Path, backbone: str, head: str, order: str, reference: bool = False) -> tuple[Path, Path]:
    tag = f"{backbone}_{head}_{'full_' if reference else ''}{order}"
    return output / f"{tag}.pth", output / f"{tag}_test_predictions.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the prespecified temporal-head training and matched-frame test matrix"
    )
    parser.add_argument("--feature-root", default="local_data/features")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output-dir", default="artifacts/temporal_models")
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reference-head", choices=HEADS, default="unidirectional_gru")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    trajectories = Path(args.trajectories)
    selection = Path(args.selection)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        for path in (trajectories, selection):
            if not path.is_file():
                raise FileNotFoundError(path)
        for backbone in args.backbones:
            for split in ("train", "test"):
                if not (feature_root / f"{backbone}_{split}" / "metadata.json").is_file():
                    raise FileNotFoundError(f"Incomplete feature cache: {backbone}_{split}")

    common = [
        "--feature-root", str(feature_root), "--trajectories", str(trajectories),
        "--batch-size", str(args.batch_size), "--output-dir", str(output),
    ]
    for backbone in args.backbones:
        for head in HEADS:
            train_orders = ("natural",) if head == "average" else ALIGNED_ORDERS
            for order in train_orders:
                checkpoint, predictions = paths_for(output, backbone, head, order)
                if args.overwrite or not checkpoint.exists():
                    command = [
                        sys.executable, "-m", "analysis.temporal_models", "train",
                        "--backbone", backbone, "--head", head, *common,
                        "--selection", str(selection), "--train-order", order,
                        "--epochs", str(args.epochs),
                    ]
                    if args.overwrite:
                        command.append("--overwrite")
                    run(command, args.dry_run)
                if args.overwrite or not predictions.exists():
                    run([
                        sys.executable, "-m", "analysis.temporal_models", "evaluate",
                        "--backbone", backbone, "--head", head, *common,
                        "--checkpoint", str(checkpoint),
                    ], args.dry_run)

        if not args.skip_reference:
            checkpoint, predictions = paths_for(
                output, backbone, args.reference_head, "natural", reference=True
            )
            if args.overwrite or not checkpoint.exists():
                command = [
                    sys.executable, "-m", "analysis.temporal_models", "train",
                    "--backbone", backbone, "--head", args.reference_head, *common,
                    "--selection", str(selection), "--train-order", "natural",
                    "--epochs", str(args.epochs), "--reference-full",
                ]
                if args.overwrite:
                    command.append("--overwrite")
                run(command, args.dry_run)
            if args.overwrite or not predictions.exists():
                run([
                    sys.executable, "-m", "analysis.temporal_models", "evaluate",
                    "--backbone", backbone, "--head", args.reference_head, *common,
                    "--checkpoint", str(checkpoint),
                ], args.dry_run)


if __name__ == "__main__":
    main()
