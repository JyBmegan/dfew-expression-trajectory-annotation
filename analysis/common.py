from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


EMOTIONS = ["Happiness", "Sadness", "Neutral", "Anger", "Surprise", "Disgust", "Fear"]


def read_config(path: str | Path) -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.9/3.10
        import tomli as tomllib
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_base"] = str(path.parent)
    return config


def resolve_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = Path(config["_base"]) / path
    return candidate.resolve() if candidate.exists() else (Path(__file__).parents[1] / path).resolve()


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def parse_curve(row: pd.Series) -> np.ndarray:
    columns = [f"intensity_{index}" for index in range(1, 17)]
    if all(column in row.index for column in columns):
        curve = row[columns].to_numpy(dtype=float)
    elif "intensities_json" in row.index:
        curve = np.asarray(json.loads(row["intensities_json"]), dtype=float)
    else:
        raise ValueError("No 16-point intensity curve found")
    if curve.shape != (16,) or not np.isfinite(curve).all():
        raise ValueError("Intensity curve must contain 16 finite values")
    return curve


def trajectory_descriptors(curve: np.ndarray, category: str) -> dict:
    curve = np.asarray(curve, dtype=float)
    if curve.shape != (16,) or not np.isfinite(curve).all():
        raise ValueError("Trajectory must contain 16 finite values")
    reference_or_unusable = category in {"Neutral", "Face not visible", "No consensus"}
    no_visible_expression = not reference_or_unusable and float(curve.max()) < 1.0
    expression_present = curve >= 1
    high = curve >= 4
    maximum_positions = np.flatnonzero(curve == curve.max())
    apex = int(np.rint(np.median(maximum_positions)))
    near_peak = curve >= max(1.0, float(curve.max()) - 1.0)
    left = apex
    while left > 0 and near_peak[left - 1]:
        left -= 1
    right = apex
    while right < 15 and near_peak[right + 1]:
        right += 1
    padded = np.r_[-np.inf, curve, -np.inf]
    peak_positions = find_peaks(padded, prominence=1.0, plateau_size=True)[0] - 1
    peak_positions = peak_positions[(peak_positions >= 0) & (peak_positions < 16)]
    if curve.max() - curve.min() < 1:
        peak_type = "flat"
        peak_count = 0
    else:
        if not len(peak_positions):
            peak_positions = np.asarray([apex])
        peak_count = int(len(peak_positions))
        peak_type = "multiple" if peak_count > 1 else ("edge" if peak_positions[0] in [0, 15] else "single")
    if reference_or_unusable or no_visible_expression:
        onset = offset = apex_value = peak_width = np.nan
        if no_visible_expression:
            peak_type = "no_visible_expression"
            peak_count = 0
        else:
            peak_type = {
                "Neutral": "neutral_reference",
                "Face not visible": "face_not_visible",
                "No consensus": "no_consensus",
            }[category]
            peak_count = np.nan
    else:
        present_positions = np.flatnonzero(expression_present)
        onset = int(present_positions[0]) + 1 if len(present_positions) else np.nan
        offset = int(present_positions[-1]) + 1 if len(present_positions) else np.nan
        apex_value = apex + 1
        peak_width = right - left + 1
    return {
        "start_intensity": float(curve[0]), "end_intensity": float(curve[-1]),
        "start_end_difference": float(curve[-1] - curve[0]),
        "mean_intensity": float(curve.mean()),
        "onset_position": onset, "apex_position": apex_value, "offset_position": offset,
        "peak_width": peak_width, "peak_count": peak_count, "peak_type": peak_type,
        "expression_present_fraction": float(expression_present.mean()),
        "high_intensity_fraction": float(high.mean()),
        "trajectory_total_variation": float(np.abs(np.diff(curve)).sum()),
        "mean_absolute_adjacent_change": float(np.abs(np.diff(curve)).mean()),
        "adjacent_differences_json": json.dumps(np.diff(curve).round(6).tolist()),
    }


def true_class_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    true = logits[np.arange(len(logits)), labels]
    masked = logits.copy()
    masked[np.arange(len(logits)), labels] = -np.inf
    return true - masked.max(axis=1)


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    return {
        "macro_f1": float(f1_score(labels, predictions, labels=np.arange(7), average="macro", zero_division=0)),
        "uar": float(recall_score(labels, predictions, labels=np.arange(7), average="macro", zero_division=0)),
        "war": float(accuracy_score(labels, predictions)),
    }
