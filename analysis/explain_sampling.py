from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler


BLOCKS = {
    "base": ["emotion", "model", "sampling_method"],
    "trajectory": ["start_end_difference", "apex_position", "peak_width", "peak_count", "expression_present_fraction", "high_intensity_fraction", "cluster"],
    "label_clarity": ["vote_max", "vote_margin", "vote_entropy", "neutral_votes"],
    "timebase": ["original_frame_count"],
    "visual_quality": [
        "brightness_range", "minimum_sharpness", "face_detection_rate",
        "near_duplicate_transitions", "face_center_drift", "pose_yaw_range",
        "late_minus_early_brightness", "late_minus_early_contrast",
        "late_minus_early_sharpness", "last_minus_first_brightness",
        "last_minus_first_contrast", "last_minus_first_sharpness",
    ],
    "motion": ["mean_rigid_translation", "mean_rigid_rotation_deg", "mean_rigid_scale_change", "mean_nonrigid_flow", "mean_photometric_residual"],
}


def estimator(frame: pd.DataFrame, columns: list[str], binary: bool):
    categorical = [column for column in columns if frame[column].dtype == object or column in ["cluster"]]
    numeric = [column for column in columns if column not in categorical]
    transform = ColumnTransformer([
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("spline", SplineTransformer(n_knots=4, degree=2)), ("scale", StandardScaler(with_mean=False))]), numeric),
    ])
    model = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced") if binary else Ridge(alpha=10.0)
    return Pipeline([("transform", transform), ("model", model)])


def cross_validated_score(frame: pd.DataFrame, columns: list[str], outcome: str, binary: bool) -> float:
    predictions = np.zeros(len(frame), dtype=float)
    splitter = GroupKFold(n_splits=5)
    y = frame[outcome].to_numpy()
    for train, test in splitter.split(frame, y, groups=frame.clip_id):
        model = estimator(frame.iloc[train], columns, binary)
        model.fit(frame.iloc[train][columns], y[train])
        predictions[test] = model.predict_proba(frame.iloc[test][columns])[:, 1] if binary else model.predict(frame.iloc[test][columns])
    return float(log_loss(y, predictions, labels=[0, 1])) if binary else float(r2_score(y, predictions))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", required=True, help="Clip-level sampling outcomes with clip_id, model and emotion")
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--label-clarity", required=True)
    parser.add_argument("--stimulus", required=True)
    parser.add_argument("--stimulus-position")
    parser.add_argument("--output", default="artifacts/sampling_block_explanation.csv")
    args = parser.parse_args()
    frame = pd.read_csv(args.outcomes)
    for path in [args.trajectory, args.label_clarity, args.stimulus]:
        frame = frame.merge(pd.read_csv(path), on="clip_id", how="left", suffixes=("", "_extra"))
    if args.stimulus_position:
        position = pd.read_csv(args.stimulus_position).drop(columns=["split"], errors="ignore")
        frame = frame.merge(position, on="clip_id", how="left", suffixes=("", "_position"))
    available = {name: [column for column in columns if column in frame] for name, columns in BLOCKS.items()}
    available = {name: columns for name, columns in available.items() if columns}
    columns = [column for values in available.values() for column in values]
    results = []
    for outcome, binary in [("absolute_margin_difference", False), ("prediction_disagreement", True)]:
        if outcome not in frame:
            continue
        analysis = frame.dropna(subset=[outcome, "clip_id"]).reset_index(drop=True)
        full = cross_validated_score(analysis, columns, outcome, binary)
        results.append({"outcome": outcome, "removed_block": "none", "score": full, "score_type": "log_loss" if binary else "r2", "delta_from_full": 0.0})
        for name, removed in available.items():
            retained = [column for column in columns if column not in removed]
            score = cross_validated_score(analysis, retained, outcome, binary)
            delta = score - full if binary else full - score
            results.append({"outcome": outcome, "removed_block": name, "score": score, "score_type": "log_loss" if binary else "r2", "delta_from_full": delta})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(results).to_csv(output, index=False)
    print(f"Wrote blockwise out-of-fold explanation to {output}")


if __name__ == "__main__":
    main()
