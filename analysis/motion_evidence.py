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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .common import EMOTIONS, true_class_margin


MOTION_COMPONENTS = [
    "rigid_translation", "rigid_rotation_deg", "rigid_scale_change",
    "nonrigid_flow", "photometric_residual",
]


def make_pipeline(frame: pd.DataFrame, columns: list[str], binary: bool) -> Pipeline:
    categorical = [column for column in columns if column in ["emotion", "transition"]]
    numeric = [column for column in columns if column not in categorical]
    transform = ColumnTransformer([
        ("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), numeric),
    ])
    model = LogisticRegression(max_iter=2000, class_weight="balanced") if binary else Ridge(alpha=10)
    return Pipeline([("transform", transform), ("model", model)])


def grouped_score(frame: pd.DataFrame, columns: list[str], outcome: str, binary: bool) -> float:
    y = frame[outcome].to_numpy()
    predictions = np.zeros(len(frame), dtype=float)
    splitter = GroupKFold(n_splits=5)
    for train, test in splitter.split(frame, y, groups=frame.clip_id):
        pipeline = make_pipeline(frame, columns, binary)
        pipeline.fit(frame.iloc[train][columns], y[train])
        predictions[test] = pipeline.predict_proba(frame.iloc[test][columns])[:, 1] if binary else pipeline.predict(frame.iloc[test][columns])
    return float(log_loss(y, predictions, labels=[0, 1])) if binary else float(r2_score(y, predictions))


def transition_outcomes(logits: pd.DataFrame) -> pd.DataFrame:
    logit_columns = [f"logit_{index}" for index in range(7)]
    records = []
    for (model, clip_id), clip in logits.sort_values("frame").groupby(["model", "clip_id"]):
        if len(clip) != 16:
            continue
        label = int(clip.label.iloc[0])
        values = clip[logit_columns].to_numpy(dtype=np.float32)
        margin = true_class_margin(values, np.full(16, label))
        prediction = values.argmax(axis=1)
        for index in range(15):
            records.append({
                "model": model, "clip_id": int(clip_id), "label": label,
                "emotion": EMOTIONS[label], "transition": index + 1,
                "absolute_margin_change": float(abs(margin[index + 1] - margin[index])),
                "prediction_switch": int(prediction[index + 1] != prediction[index]),
            })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relate registered motion components to adjacent classification changes.")
    parser.add_argument("--frame-logits", nargs="+", required=True)
    parser.add_argument("--transition-features", required=True)
    parser.add_argument("--output-dir", default="artifacts/motion_evidence")
    args = parser.parse_args()

    logits = pd.concat([pd.read_csv(path) for path in args.frame_logits], ignore_index=True)
    outcomes = transition_outcomes(logits)
    motion = pd.read_csv(args.transition_features)
    frame = outcomes.merge(motion, on=["clip_id", "transition"], how="inner", suffixes=("", "_stimulus"))
    available = [column for column in MOTION_COMPONENTS if column in frame]
    if not available:
        raise ValueError("Transition features contain no registered motion components")
    scores = []
    for model_name, model_frame in frame.groupby("model"):
        model_frame = model_frame.reset_index(drop=True)
        baseline = ["emotion", "transition"]
        for outcome, binary in [("absolute_margin_change", False), ("prediction_switch", True)]:
            base_score = grouped_score(model_frame, baseline, outcome, binary)
            score_type = "log_loss" if binary else "r2"
            scores.append({
                "model": model_name, "outcome": outcome, "predictors": "category_and_position",
                "score_type": score_type, "score": base_score, "change_from_baseline": 0.0,
            })
            for component in available:
                score = grouped_score(model_frame, baseline + [component], outcome, binary)
                improvement = base_score - score if binary else score - base_score
                scores.append({
                    "model": model_name, "outcome": outcome, "predictors": component,
                    "score_type": score_type, "score": score, "change_from_baseline": improvement,
                })
            full = grouped_score(model_frame, baseline + available, outcome, binary)
            scores.append({
                "model": model_name, "outcome": outcome, "predictors": "all_motion_components",
                "score_type": score_type, "score": full,
                "change_from_baseline": base_score - full if binary else full - base_score,
            })
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "transition_motion_outcomes.csv.gz", index=False, compression="gzip")
    pd.DataFrame(scores).to_csv(output / "grouped_cross_validation_scores.csv", index=False)
    print(f"Motion-evidence analysis: {frame.clip_id.nunique()} clips and {len(frame)} model-transition rows")


if __name__ == "__main__":
    main()
