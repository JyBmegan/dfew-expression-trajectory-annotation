from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", default="artifacts/trajectory_consensus.csv")
    parser.add_argument("--subset-results", default="artifacts/frame_subsets/subset_results.csv.gz")
    parser.add_argument("--temporal-dir", default="artifacts/temporal_models")
    parser.add_argument("--feature-root", default="local_data/features")
    parser.add_argument("--expected-test-clips", type=int, default=2341)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    errors, pending = [], []
    trajectory = Path(args.trajectory)
    if trajectory.exists():
        frame = pd.read_csv(trajectory)
        intensity = [f"intensity_{index}" for index in range(1,17)]
        missing = set(intensity) - set(frame.columns)
        if missing: errors.append(f"Trajectory output is missing {sorted(missing)}")
        if frame.clip_id.duplicated().any(): errors.append("Trajectory output contains duplicate clip IDs")
        if "dominant_category" in frame and "apex_position" in frame:
            neutral = frame.dominant_category == "Neutral"
            if frame.loc[neutral, "apex_position"].notna().any():
                errors.append("Neutral trajectories contain apex values")
    else: pending.append(str(trajectory))
    subsets = Path(args.subset_results)
    if subsets.exists():
        frame = pd.read_csv(subsets)
        expected = {"-".join(map(str, values)) for values in itertools.combinations(range(1,17),4)}
        for model, group in frame.groupby("model"):
            found = set(group.subset.astype(str))
            if found != expected: errors.append(f"{model}: expected 1,820 unique subsets, found {len(found)}")
            for subset in found:
                positions = subset.split("-")
                if len(positions) != 4 or len(set(positions)) != 4:
                    errors.append(f"{model}: invalid four-frame subset {subset}")
                    break
    else: pending.append(str(subsets))
    temporal = Path(args.temporal_dir)
    predictions = list(temporal.glob("*_test_predictions.csv.gz")) if temporal.exists() else []
    if predictions:
        expected_orders = {"natural","reverse","weak_to_strong","strong_to_weak", *{f"shuffle_{i:02d}" for i in range(1,21)}}
        for path in predictions:
            frame = pd.read_csv(path, usecols=["clip_id","test_order"])
            if set(frame.test_order) != expected_orders: errors.append(f"{path.name}: incomplete test-order set")
            counts = frame.groupby("test_order").clip_id.nunique()
            if counts.nunique() != 1: errors.append(f"{path.name}: test orders use different clip sets")
            if counts.iloc[0] != args.expected_test_clips:
                errors.append(f"{path.name}: expected {args.expected_test_clips} clips per test order, found {counts.iloc[0]}")
    else: pending.append(str(temporal / "*_test_predictions.csv.gz"))
    metadata = list(temporal.glob("*.json")) if temporal.exists() else []
    split_signatures = {}
    for path in metadata:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("reference_full"):
            continue
        key = payload.get("backbone")
        signature = (tuple(payload.get("train_clip_ids", [])), tuple(payload.get("validation_clip_ids", [])))
        if key in split_signatures and split_signatures[key] != signature:
            errors.append(f"{key}: temporal heads do not share the same train/validation split")
        split_signatures[key] = signature
    feature_root = Path(args.feature_root)
    for backbone in ["alexnet", "resnet18"]:
        for split in ["train", "test"]:
            progress = feature_root / f"{backbone}_{split}" / "progress.json"
            if progress.exists():
                payload = json.loads(progress.read_text(encoding="utf-8"))
                if payload.get("status") != "complete" or payload.get("seen") != payload.get("items"):
                    errors.append(f"Incomplete feature cache: {backbone}_{split}")
                if payload.get("feature_source") != "model(tensors, return_features=True)":
                    errors.append(f"Unexpected feature definition: {backbone}_{split}")
            else:
                pending.append(str(progress))
    if errors:
        print("Output validation failed:")
        for error in errors: print(f"- {error}")
        raise SystemExit(1)
    if pending:
        print("Valid outputs found; pending human/model stages:")
        for item in pending: print(f"- {item}")
        if args.strict: raise SystemExit(2)
    else:
        print("All requested output validations passed")


if __name__ == "__main__":
    main()
