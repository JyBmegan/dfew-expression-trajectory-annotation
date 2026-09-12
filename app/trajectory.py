from __future__ import annotations

import json

import numpy as np
from scipy.signal import find_peaks


def trajectory_descriptors(curve: np.ndarray, category: str) -> dict:
    curve = np.asarray(curve, dtype=float)
    if curve.shape != (16,) or not np.isfinite(curve).all():
        raise ValueError("Trajectory must contain 16 finite values")
    reference = category in {"Neutral", "Face not visible", "No consensus"}
    no_expression = not reference and float(curve.max()) < 1.0
    present = curve >= 1
    high = curve >= 4
    maxima = np.flatnonzero(curve == curve.max())
    apex = int(np.rint(np.median(maxima)))
    near_peak = curve >= max(1.0, float(curve.max()) - 1.0)
    left = apex
    while left > 0 and near_peak[left - 1]: left -= 1
    right = apex
    while right < 15 and near_peak[right + 1]: right += 1
    padded = np.r_[-np.inf, curve, -np.inf]
    peaks = find_peaks(padded, prominence=1.0, plateau_size=True)[0] - 1
    peaks = peaks[(peaks >= 0) & (peaks < 16)]
    if curve.max() - curve.min() < 1:
        peak_type, peak_count = "flat", 0
    else:
        if not len(peaks): peaks = np.asarray([apex])
        peak_count = int(len(peaks))
        peak_type = "multiple" if peak_count > 1 else ("edge" if peaks[0] in [0, 15] else "single")
    if reference or no_expression:
        onset = offset = apex_value = peak_width = np.nan
        if no_expression:
            peak_type, peak_count = "no_visible_expression", 0
        else:
            peak_type = {"Neutral": "neutral_reference", "Face not visible": "face_not_visible", "No consensus": "no_consensus"}[category]
            peak_count = np.nan
    else:
        positions = np.flatnonzero(present)
        onset = int(positions[0]) + 1 if len(positions) else np.nan
        offset = int(positions[-1]) + 1 if len(positions) else np.nan
        apex_value, peak_width = apex + 1, right - left + 1
    return {
        "start_intensity": float(curve[0]), "end_intensity": float(curve[-1]),
        "start_end_difference": float(curve[-1] - curve[0]), "mean_intensity": float(curve.mean()),
        "onset_position": onset, "apex_position": apex_value, "offset_position": offset,
        "peak_width": peak_width, "peak_count": peak_count, "peak_type": peak_type,
        "expression_present_fraction": float(present.mean()), "high_intensity_fraction": float(high.mean()),
        "trajectory_total_variation": float(np.abs(np.diff(curve)).sum()),
        "mean_absolute_adjacent_change": float(np.abs(np.diff(curve)).mean()),
        "adjacent_differences_json": json.dumps(np.diff(curve).round(6).tolist()),
    }
