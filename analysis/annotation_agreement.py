from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .common import trajectory_descriptors


def safe_kappa(first: pd.Series, second: pd.Series) -> float:
    if len(first) < 2 or len(set(first.astype(str)) | set(second.astype(str))) < 2:
        return np.nan
    value = cohen_kappa_score(first.astype(str), second.astype(str))
    return float(value) if np.isfinite(value) else np.nan


def frame_agreement(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "repeat_kind", "primary_category", "repeat_category",
        "primary_intensity", "repeat_intensity",
    }
    missing = required - set(pairs.columns)
    if missing:
        raise ValueError(f"Frame reliability table is missing {sorted(missing)}")
    rows = []
    for repeat_kind, group in pairs.groupby("repeat_kind", sort=True):
        difference = group.repeat_intensity.astype(float) - group.primary_intensity.astype(float)
        rows.append({
            "rating_type": repeat_kind,
            "pairs": len(group),
            "category_agreement": float((group.primary_category == group.repeat_category).mean()),
            "category_cohen_kappa": safe_kappa(group.primary_category, group.repeat_category),
            "intensity_mean_absolute_gap": float(np.abs(difference).mean()),
            "intensity_median_absolute_gap": float(np.median(np.abs(difference))),
            "intensity_mean_signed_gap": float(difference.mean()),
        })
    confusion = pd.crosstab(
        [pairs.repeat_kind, pairs.primary_category], pairs.repeat_category,
        normalize="index", dropna=False,
    )
    return pd.DataFrame(rows), confusion


def rated_apex(category: str, curve: np.ndarray) -> float:
    return float(trajectory_descriptors(curve, category)["apex_position"])


def sequence_agreement(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"clip_id", "split", "annotator_code", "role", "dominant_category", "intensities_json"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Clip ratings table is missing {sorted(missing)}")
    base = ratings[ratings.role != "adjudication"].copy()
    pair_rows = []
    for (split, clip_id), group in base.groupby(["split", "clip_id"], sort=False):
        if len(group) != 2:
            continue
        first, second = group.sort_values("annotator_code").iloc[0], group.sort_values("annotator_code").iloc[1]
        curve_a = np.asarray(json.loads(first.intensities_json), dtype=float)
        curve_b = np.asarray(json.loads(second.intensities_json), dtype=float)
        if curve_a.shape != (16,) or curve_b.shape != (16,):
            continue
        apex_a = rated_apex(first.dominant_category, curve_a)
        apex_b = rated_apex(second.dominant_category, curve_b)
        pair_rows.append({
            "split": split, "clip_id": int(clip_id),
            "annotator_a": first.annotator_code, "annotator_b": second.annotator_code,
            "category_a": first.dominant_category, "category_b": second.dominant_category,
            "category_agreement": int(first.dominant_category == second.dominant_category),
            "curve_mean_absolute_gap": float(np.abs(curve_a - curve_b).mean()),
            "curve_maximum_absolute_gap": float(np.abs(curve_a - curve_b).max()),
            "apex_absolute_gap": float(abs(apex_a - apex_b)) if np.isfinite(apex_a) and np.isfinite(apex_b) else np.nan,
        })
    pairs = pd.DataFrame(pair_rows)
    rows = []
    for split, group in pairs.groupby("split", sort=True):
        rows.append({
            "split": split, "pairs": len(group),
            "category_agreement": float(group.category_agreement.mean()),
            "category_cohen_kappa": safe_kappa(group.category_a, group.category_b),
            "curve_mean_absolute_gap": float(group.curve_mean_absolute_gap.mean()),
            "curve_median_absolute_gap": float(group.curve_mean_absolute_gap.median()),
            "apex_median_absolute_gap": float(group.apex_absolute_gap.median()),
        })
    return pd.DataFrame(rows), pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prespecified frame repeats and paired sequence ratings")
    parser.add_argument("--frame-pairs", required=True)
    parser.add_argument("--clip-ratings", required=True)
    parser.add_argument("--output-dir", default="artifacts/annotation_agreement")
    args = parser.parse_args()

    frame_summary, frame_confusion = frame_agreement(pd.read_csv(args.frame_pairs))
    sequence_summary, sequence_pairs = sequence_agreement(pd.read_csv(args.clip_ratings))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame_summary.to_csv(output / "frame_repeat_summary.csv", index=False)
    frame_confusion.to_csv(output / "frame_repeat_category_confusion.csv")
    sequence_summary.to_csv(output / "sequence_pair_summary.csv", index=False)
    sequence_pairs.to_csv(output / "sequence_pairs.csv.gz", index=False, compression="gzip")
    print(f"Agreement summaries written to {output}")


if __name__ == "__main__":
    main()
