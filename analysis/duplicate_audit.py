from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.frame_source import FrameSource
from .common import EMOTIONS, read_config, resolve_path
from .stimulus_audit import difference_hash


def bits(signature: str) -> np.ndarray:
    return np.asarray([(int(char, 16) >> bit) & 1 for char in signature for bit in range(4)], dtype=np.uint8)


def compare_full_sequences(source: FrameSource, first: int, second: int) -> dict:
    hamming, pixel_difference, exact = [], [], 0
    for frame in range(1, 17):
        left = cv2.imdecode(np.frombuffer(source.read(first, frame), dtype=np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(source.read(second, frame), dtype=np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None or left.shape != right.shape:
            raise ValueError(f"Cannot compare candidate pair {first:05d}/{second:05d} at frame {frame}")
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        hamming.append(int(np.not_equal(bits(difference_hash(left_gray)), bits(difference_hash(right_gray))).sum()))
        pixel_difference.append(float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32)))))
        exact += int(np.array_equal(left, right))
    mean_hamming = float(np.mean(hamming))
    mean_pixel = float(np.mean(pixel_difference))
    if mean_hamming <= 2 and mean_pixel <= 8:
        tier = "very_high_similarity"
    elif mean_hamming <= 6 and mean_pixel <= 15:
        tier = "high_similarity"
    else:
        tier = "fingerprint_candidate"
    return {
        "mean_frame_hamming": mean_hamming,
        "maximum_frame_hamming": int(max(hamming)),
        "mean_pixel_absolute_difference": mean_pixel,
        "exact_matching_frames": exact,
        "review_tier": tier,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimulus-features", required=True)
    parser.add_argument("--output", default="artifacts/duplicate_candidates.csv")
    parser.add_argument("--maximum-hamming", type=int, default=12)
    parser.add_argument("--config", help="Authorized local config; if supplied, verify candidates across all 16 frames")
    args = parser.parse_args()
    frame = pd.read_csv(args.stimulus_features).dropna(subset=["clip_fingerprint"])
    buckets: dict[str, list[int]] = defaultdict(list)
    rows = frame.reset_index(drop=True)
    for index, signature in enumerate(rows.clip_fingerprint.astype(str)):
        for band in range(6):
            start = band * 8
            buckets[f"{band}:{signature[start:start + 8]}"].append(index)
    candidates = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add((members[i], members[j]))
    output = []
    vectors = {index: bits(str(rows.loc[index, "clip_fingerprint"])) for pair in candidates for index in pair}
    for left, right in candidates:
        if rows.loc[left, "split"] == rows.loc[right, "split"]:
            continue
        distance = int(np.not_equal(vectors[left], vectors[right]).sum())
        if distance <= args.maximum_hamming:
            output.append({
                "clip_id_1": int(rows.loc[left, "clip_id"]), "split_1": rows.loc[left, "split"],
                "clip_id_2": int(rows.loc[right, "clip_id"]), "split_2": rows.loc[right, "split"],
                "hamming_distance": distance,
            })
    result = pd.DataFrame(output).sort_values("hamming_distance") if output else pd.DataFrame(columns=["clip_id_1","split_1","clip_id_2","split_2","hamming_distance"])
    if args.config and not result.empty:
        config = read_config(args.config); paths = config["paths"]
        labels = {}
        for split in ("train", "test"):
            split_frame = pd.read_csv(resolve_path(config, paths[f"{split}_csv"]))
            for row in split_frame.itertuples():
                if 1 <= int(row.label) <= 7:
                    labels[(split, int(row.video_name))] = int(row.label) - 1
        result["label_1"] = [labels.get((row.split_1, int(row.clip_id_1)), np.nan) for row in result.itertuples()]
        result["label_2"] = [labels.get((row.split_2, int(row.clip_id_2)), np.nan) for row in result.itertuples()]
        result["emotion_1"] = result.label_1.map(
            lambda value: EMOTIONS[int(value)] if np.isfinite(value) else np.nan
        )
        result["emotion_2"] = result.label_2.map(
            lambda value: EMOTIONS[int(value)] if np.isfinite(value) else np.nan
        )
        result["same_dfew_label"] = result.label_1 == result.label_2
        source_value = paths.get("frames_16_archive", paths["frames"])
        source = FrameSource(resolve_path(config, source_value), paths.get("frames_archive_password"))
        try:
            verified = [
                compare_full_sequences(source, int(row.clip_id_1), int(row.clip_id_2))
                for row in result.itertuples()
            ]
        finally:
            source.close()
        result = pd.concat([result.reset_index(drop=True), pd.DataFrame(verified)], axis=1)
        result = result.sort_values(
            ["review_tier", "mean_frame_hamming", "mean_pixel_absolute_difference"],
            key=lambda values: values.map({
                "very_high_similarity": 0, "high_similarity": 1, "fingerprint_candidate": 2,
            }) if values.name == "review_tier" else values,
        )
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); result.to_csv(path, index=False)
    print(f"Found {len(result)} cross-split duplicate candidates")


if __name__ == "__main__":
    main()
