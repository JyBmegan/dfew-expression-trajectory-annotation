from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FRAME_MEASURES = ["brightness", "contrast", "sharpness"]
TRANSITION_MEASURES = [
    "mean_abs_change", "rigid_translation", "rigid_rotation_deg",
    "rigid_scale_change", "nonrigid_flow", "photometric_residual",
]


def paired_interval(draws: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    means = np.empty(repeats, dtype=float)
    for index in range(repeats):
        means[index] = rng.choice(draws, size=len(draws), replace=True).mean()
    return tuple(np.quantile(means, [.025, .975]))


def clip_position_features(frames: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, clip_id), clip in frames.groupby(["split", "clip_id"], sort=False):
        clip = clip.sort_values("frame")
        if len(clip) != 16:
            continue
        row = {"split": split, "clip_id": int(clip_id)}
        for measure in FRAME_MEASURES:
            if measure not in clip:
                continue
            values = clip[measure].to_numpy(dtype=float)
            row[f"late_minus_early_{measure}"] = float(np.nanmean(values[12:16]) - np.nanmean(values[0:4]))
            row[f"last_minus_first_{measure}"] = float(values[-1] - values[0])
            finite = np.isfinite(values)
            row[f"position_slope_{measure}"] = float(
                np.polyfit(np.arange(1, 17)[finite], values[finite], 1)[0]
            ) if finite.sum() >= 2 else np.nan
        rows.append(row)
    result = pd.DataFrame(rows)

    transition_rows = []
    for (split, clip_id), clip in transitions.groupby(["split", "clip_id"], sort=False):
        clip = clip.sort_values("transition")
        if len(clip) != 15:
            continue
        row = {"split": split, "clip_id": int(clip_id)}
        for measure in TRANSITION_MEASURES:
            if measure not in clip:
                continue
            values = clip[measure].to_numpy(dtype=float)
            middle = values[1:14]
            interior = float(np.nanmean(middle)) if np.isfinite(middle).any() else np.nan
            row[f"first_transition_minus_interior_{measure}"] = float(values[0] - interior) if np.isfinite(values[0]) else np.nan
            row[f"last_transition_minus_interior_{measure}"] = float(values[-1] - interior) if np.isfinite(values[-1]) else np.nan
        transition_rows.append(row)
    return result.merge(pd.DataFrame(transition_rows), on=["split", "clip_id"], how="outer")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create clip-level directional image-quality and boundary variables")
    parser.add_argument("--frame-features", required=True)
    parser.add_argument("--transition-features", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--output-dir", default="artifacts/stimulus_position")
    args = parser.parse_args()

    features = clip_position_features(
        pd.read_csv(args.frame_features), pd.read_csv(args.transition_features)
    )
    rng = np.random.default_rng(20260912)
    summary = []
    measures = [column for column in features if column not in {"clip_id", "split"}]
    for split, group in features.groupby("split", sort=True):
        for measure in measures:
            values = group[measure].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            lower, upper = paired_interval(values, args.bootstrap, rng)
            summary.append({
                "split": split, "measure": measure, "clips": len(values),
                "mean": float(values.mean()), "median": float(np.median(values)),
                "ci_lower": float(lower), "ci_upper": float(upper),
            })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "stimulus_position_features.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "stimulus_position_summary.csv", index=False)
    print(f"Directional stimulus features written for {len(features)} clips")


if __name__ == "__main__":
    main()
