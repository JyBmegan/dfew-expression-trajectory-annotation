from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import EMOTIONS, require_columns


def paired_interval(values: np.ndarray, labels: np.ndarray, repeats: int = 5000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    rng = np.random.default_rng(20260912)
    draws = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        draws[repeat] = values[sample].mean()
    return tuple(np.quantile(draws, [.025, .975]))


def attach_human_variables(frame: pd.DataFrame, trajectories: str | None, stimulus: str | None) -> pd.DataFrame:
    result = frame.copy()
    if trajectories:
        human = pd.read_csv(trajectories)
        if "split" in human:
            human = human[human.split == "test"]
        columns = [column for column in [
            "clip_id", "dominant_category", "start_end_difference", "cluster",
            "apex_position", "peak_width", "high_intensity_fraction",
        ] if column in human]
        result = result.merge(human[columns].drop_duplicates("clip_id"), on="clip_id", how="left")
        if "start_end_difference" in result:
            result["human_direction"] = np.select(
                [result.start_end_difference >= 1, result.start_end_difference <= -1],
                ["weak_to_strong", "strong_to_weak"],
                default="limited_start_end_change",
            )
    if stimulus:
        clips = pd.read_csv(stimulus)
        columns = [column for column in ["clip_id", "original_frame_count"] if column in clips]
        result = result.merge(clips[columns].drop_duplicates("clip_id"), on="clip_id", how="left")
        if "original_frame_count" in result and result.original_frame_count.notna().sum() >= 4:
            result["original_length_quartile"] = pd.qcut(
                result.original_frame_count.rank(method="first"), 4,
                labels=["Q1", "Q2", "Q3", "Q4"],
            )
    return result


def summarize(frame: pd.DataFrame, contrast_column: str, repeats: int) -> pd.DataFrame:
    rows = []
    identity = [column for column in ["backbone", "head", "train_order", "reference_full", "contrast"] if column in frame]
    groupings = [("all", None)]
    groupings.extend((column, column) for column in [
        "emotion", "dominant_category", "human_direction", "cluster", "original_length_quartile",
    ] if column in frame)
    for identity_values, base in frame.groupby(identity, dropna=False, sort=True):
        identity_values = identity_values if isinstance(identity_values, tuple) else (identity_values,)
        shared = dict(zip(identity, identity_values))
        for grouping_name, grouping_column in groupings:
            parts = [("all", base)] if grouping_column is None else base.dropna(subset=[grouping_column]).groupby(grouping_column, sort=True)
            for group_name, group in parts:
                if group.empty:
                    continue
                lower, upper = paired_interval(
                    group[contrast_column].to_numpy(), group.label.to_numpy(), repeats,
                )
                rows.append({
                    **shared, "grouping": grouping_name, "group": group_name,
                    "clips": group.clip_id.nunique(), "mean_margin_contrast": group[contrast_column].mean(),
                    "ci_lower": lower, "ci_upper": upper,
                })
    return pd.DataFrame(rows)


def within_run_contrasts(predictions: pd.DataFrame) -> pd.DataFrame:
    key = ["backbone", "head", "train_order", "reference_full", "clip_id", "label"]
    pivot = predictions.pivot(index=key, columns="test_order", values="true_class_logit_margin")
    required = {"natural", "reverse", "weak_to_strong", "strong_to_weak"}
    missing = required - set(pivot.columns)
    if missing:
        raise ValueError(f"Temporal predictions are missing test orders {sorted(missing)}")
    shuffle_columns = sorted(column for column in pivot.columns if str(column).startswith("shuffle_"))
    if len(shuffle_columns) != 20:
        raise ValueError(f"Expected 20 fixed-shuffle conditions, found {len(shuffle_columns)}")
    outputs = []
    definitions = {
        "natural_minus_reverse": pivot["natural"] - pivot["reverse"],
        "weak_to_strong_minus_strong_to_weak": pivot["weak_to_strong"] - pivot["strong_to_weak"],
        "natural_minus_mean_shuffle": pivot["natural"] - pivot[shuffle_columns].mean(axis=1),
    }
    for name, values in definitions.items():
        current = values.rename("margin_contrast").reset_index()
        current["contrast"] = name
        outputs.append(current)
    return pd.concat(outputs, ignore_index=True)


def train_test_alignment_contrast(predictions: pd.DataFrame) -> pd.DataFrame:
    subset = predictions[
        (~predictions.reference_full.astype(bool))
        & predictions.train_order.isin(["weak_to_strong", "strong_to_weak"])
        & predictions.test_order.isin(["weak_to_strong", "strong_to_weak"])
    ]
    key = ["backbone", "head", "clip_id", "label"]
    pivot = subset.pivot(index=key, columns=["train_order", "test_order"], values="true_class_logit_margin")
    columns = {
        ("weak_to_strong", "weak_to_strong"), ("weak_to_strong", "strong_to_weak"),
        ("strong_to_weak", "weak_to_strong"), ("strong_to_weak", "strong_to_weak"),
    }
    if not columns.issubset(set(pivot.columns)):
        return pd.DataFrame()
    matched = (
        pivot[("weak_to_strong", "weak_to_strong")]
        + pivot[("strong_to_weak", "strong_to_weak")]
    ) / 2
    mismatched = (
        pivot[("weak_to_strong", "strong_to_weak")]
        + pivot[("strong_to_weak", "weak_to_strong")]
    ) / 2
    result = (matched - mismatched).rename("margin_contrast").reset_index()
    result["train_order"] = "aligned_vs_opposite"
    result["reference_full"] = False
    result["contrast"] = "matched_train_test_strength_direction"
    return result


def temporal_minus_endpoint_control(within: pd.DataFrame) -> pd.DataFrame:
    control = within[within["head"] == "last_frame"].copy()
    temporal = within[
        within["head"].isin(["unidirectional_gru", "bidirectional_gru", "causal_tcn"])
    ].copy()
    if control.empty or temporal.empty:
        return pd.DataFrame()
    key = ["backbone", "train_order", "reference_full", "clip_id", "label", "contrast"]
    control = control[key + ["margin_contrast"]].rename(
        columns={"margin_contrast": "endpoint_margin_contrast"}
    )
    result = temporal.merge(control, on=key, how="inner")
    result["margin_contrast"] = result.margin_contrast - result.endpoint_margin_contrast
    result["contrast"] = "temporal_minus_endpoint_control__" + result.contrast.astype(str)
    return result.drop(columns=["endpoint_margin_contrast"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired temporal-order and trajectory-alignment contrasts")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--trajectories")
    parser.add_argument("--stimulus")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--output-dir", default="artifacts/temporal_effects")
    args = parser.parse_args()
    predictions = pd.concat([pd.read_csv(path) for path in args.predictions], ignore_index=True)
    require_columns(predictions, [
        "backbone", "head", "train_order", "reference_full", "clip_id", "label",
        "test_order", "true_class_logit_margin",
    ], "temporal predictions")
    duplicates = predictions.duplicated([
        "backbone", "head", "train_order", "reference_full", "clip_id", "test_order",
    ]).sum()
    if duplicates:
        raise ValueError(f"Temporal predictions contain {duplicates} duplicate run/clip/order rows")
    if predictions.reference_full.dtype == object:
        predictions["reference_full"] = predictions.reference_full.astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if predictions.reference_full.isna().any():
            raise ValueError("reference_full contains values other than true/false")
    predictions["emotion"] = predictions.label.astype(int).map(dict(enumerate(EMOTIONS)))
    raw_within = within_run_contrasts(predictions)
    endpoint_adjusted = temporal_minus_endpoint_control(raw_within)
    within = attach_human_variables(raw_within, args.trajectories, args.stimulus)
    if not endpoint_adjusted.empty:
        endpoint_adjusted = attach_human_variables(
            endpoint_adjusted, args.trajectories, args.stimulus,
        )
    alignment = train_test_alignment_contrast(predictions)
    endpoint_adjusted_alignment = temporal_minus_endpoint_control(alignment)
    if not alignment.empty:
        alignment["emotion"] = alignment.label.astype(int).map(dict(enumerate(EMOTIONS)))
        alignment = attach_human_variables(alignment, args.trajectories, args.stimulus)
    if not endpoint_adjusted_alignment.empty:
        endpoint_adjusted_alignment["emotion"] = endpoint_adjusted_alignment.label.astype(int).map(
            dict(enumerate(EMOTIONS))
        )
        endpoint_adjusted_alignment = attach_human_variables(
            endpoint_adjusted_alignment, args.trajectories, args.stimulus,
        )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    clip = pd.concat(
        [within, alignment, endpoint_adjusted, endpoint_adjusted_alignment],
        ignore_index=True, sort=False,
    )
    clip.to_csv(output / "temporal_contrasts_by_clip.csv.gz", index=False, compression="gzip")
    summarize(clip, "margin_contrast", args.bootstrap).to_csv(
        output / "temporal_contrast_summary.csv", index=False,
    )
    print(f"Temporal contrasts written for {clip.clip_id.nunique()} test clips")


if __name__ == "__main__":
    main()
