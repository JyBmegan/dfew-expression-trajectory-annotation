from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import EMOTIONS, classification_metrics, parse_curve, require_columns, true_class_margin


def load_logits(paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    logit_columns = sorted([column for column in frame.columns if column.startswith("logit_")], key=lambda value: int(value.split("_")[-1]))
    require_columns(frame, ["model", "clip_id", "frame"] + logit_columns, "frame logits")
    label_column = "emotion_id" if "emotion_id" in frame else "label"
    require_columns(frame, [label_column], "frame logits")
    frame = frame.sort_values(["model", "clip_id", "frame"])
    sizes = frame.groupby(["model", "clip_id"]).size()
    if not (sizes == 16).all():
        raise ValueError("Each model/clip must contain exactly 16 frame records")
    keys = frame[["model", "clip_id"]].drop_duplicates().reset_index(drop=True)
    logits = frame[logit_columns].to_numpy(dtype=np.float32).reshape(len(keys), 16, len(logit_columns))
    labels = frame.groupby(["model", "clip_id"], sort=False)[label_column].first().to_numpy(dtype=int)
    if labels.min() == 1 and labels.max() == 7:
        labels = labels - 1
    return keys.model.to_numpy(), keys.clip_id.to_numpy(dtype=int), labels, logits


def coverage_arrays(
    clip_ids: np.ndarray, trajectories_path: str | None,
    frame_ratings_path: str | None, labels: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    trajectory = eligible = apex_positions = None
    if trajectories_path:
        frame = pd.read_csv(trajectories_path).set_index("clip_id")
        trajectory = np.vstack([parse_curve(frame.loc[clip_id]) for clip_id in clip_ids])
        excluded = {"Neutral", "Face not visible", "No consensus"}
        apex_positions = np.asarray([
            float(frame.loc[clip_id].get("apex_position", np.nan)) - 1 for clip_id in clip_ids
        ])
        eligible = np.asarray([
            frame.loc[clip_id].get("dominant_category", "") not in excluded
            and np.isfinite(apex_positions[position])
            for position, clip_id in enumerate(clip_ids)
        ])
    concordance = None
    if frame_ratings_path:
        ratings = pd.read_csv(frame_ratings_path)
        emotion = {index: value for index, value in enumerate(["Happiness","Sadness","Neutral","Anger","Surprise","Disgust","Fear"])}
        ratings["target"] = ratings.clip_id.map(dict(zip(clip_ids, labels))).map(emotion)
        ratings["concordant"] = ratings.visible_category == ratings.target
        pivot = ratings.pivot(index="clip_id", columns="frame", values="concordant")
        concordance = pivot.reindex(clip_ids).reindex(columns=range(1,17), fill_value=False).fillna(False).to_numpy(bool)
    return trajectory, eligible, apex_positions, concordance


def model_subset_statistics(values: np.ndarray, labels: np.ndarray, subset: tuple[int, ...]) -> dict:
    full_logits = values.mean(axis=1)
    selected_logits = values[:, subset, :].mean(axis=1)
    full_margin = true_class_margin(full_logits, labels)
    selected_margin = true_class_margin(selected_logits, labels)
    prediction = selected_logits.argmax(axis=1)
    full_prediction = full_logits.argmax(axis=1)
    absolute_error = np.abs(selected_margin - full_margin)
    return {
        "mean_absolute_margin_difference": float(absolute_error.mean()),
        "median_absolute_margin_difference": float(np.median(absolute_error)),
        "prediction_agreement": float((prediction == full_prediction).mean()),
        **classification_metrics(labels, prediction),
    }


def subset_geometry(subset: tuple[int, ...]) -> dict:
    positions = np.asarray(subset, dtype=float) + 1
    gaps = np.diff(positions)
    return {
        "mean_position": float(positions.mean()),
        "temporal_span": int(positions[-1] - positions[0]),
        "minimum_gap": int(gaps.min()), "maximum_gap": int(gaps.max()),
        "gap_standard_deviation": float(gaps.std(ddof=0)),
        "includes_first": int(1 in positions), "includes_last": int(16 in positions),
        "temporal_quartiles_covered": int(len(set(((positions - 1) // 4).astype(int)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-logits", required=True, nargs="+")
    parser.add_argument(
        "--training-frame-logits", nargs="+",
        help="Optional official-training logits used to choose one fixed subset before test evaluation",
    )
    parser.add_argument("--trajectories")
    parser.add_argument("--frame-ratings")
    parser.add_argument("--output-dir", default="artifacts/frame_subsets")
    parser.add_argument("--store-per-clip", action="store_true")
    args = parser.parse_args()
    models, clip_ids, labels, logits = load_logits(args.frame_logits)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    summaries, oracle_rows, grouped_rows = [], [], []
    per_clip_handle = gzip.open(output / "subset_clip_results.csv.gz", "wt", newline="", encoding="utf-8") if args.store_per_clip else None
    per_clip_writer = None
    try:
        for model in sorted(set(models)):
            mask = models == model
            ids, y, values = clip_ids[mask], labels[mask], logits[mask]
            trajectory, eligible, apex_positions, concordance = coverage_arrays(
                ids, args.trajectories, args.frame_ratings, y
            )
            trajectory_source = (
                pd.read_csv(args.trajectories).set_index("clip_id").reindex(ids)
                if args.trajectories else None
            )
            full_logits = values.mean(axis=1)
            full_margin = true_class_margin(full_logits, y)
            full_prediction = full_logits.argmax(axis=1)
            best_error = np.full(len(ids), np.inf)
            best_subset = np.empty(len(ids), dtype=object)
            best_disagreement = np.zeros(len(ids), dtype=np.int8)
            named_subsets = {(0, 1, 2, 3): "early_4", (12, 13, 14, 15): "late_4", (0, 5, 10, 15): "uniform_4"}
            for subset in itertools.combinations(range(16), 4):
                selected = values[:, subset, :].mean(axis=1)
                margin = true_class_margin(selected, y)
                prediction = selected.argmax(axis=1)
                absolute_error = np.abs(margin - full_margin)
                key = "-".join(str(index + 1) for index in subset)
                improved = absolute_error < best_error
                best_error[improved] = absolute_error[improved]
                best_subset[improved] = key
                best_disagreement[improved] = (prediction[improved] != full_prediction[improved]).astype(np.int8)
                row = {
                    "model": model, "subset": key,
                    **subset_geometry(subset),
                    "mean_absolute_margin_difference": float(absolute_error.mean()),
                    "median_absolute_margin_difference": float(np.median(absolute_error)),
                    "prediction_agreement": float((prediction == full_prediction).mean()),
                    **classification_metrics(y, prediction),
                }
                if trajectory is not None:
                    apex_hits = np.array([value in subset for value in apex_positions])
                    row["apex_coverage"] = float(apex_hits[eligible].mean()) if eligible.any() else np.nan
                    high = trajectory >= 4
                    high_available = eligible & high.any(axis=1)
                    row["high_intensity_available_fraction"] = float(high_available.mean())
                    row["high_intensity_coverage"] = (
                        float(high[high_available][:, subset].any(axis=1).mean())
                        if high_available.any() else np.nan
                    )
                    present_hits = (trajectory[:, subset] >= 1).any(axis=1)
                    row["expression_interval_coverage"] = float(present_hits[eligible].mean()) if eligible.any() else np.nan
                if concordance is not None:
                    row["label_concordant_coverage"] = float(concordance[:, subset].any(axis=1).mean())
                summaries.append(row)
                grouping_values = {"emotion": np.asarray([EMOTIONS[int(value)] for value in y])}
                if trajectory is not None:
                    if "cluster" in trajectory_source:
                        grouping_values["trajectory_cluster"] = trajectory_source.cluster.to_numpy()
                    grouping_values["human_direction"] = np.select(
                        [
                            trajectory_source.start_end_difference.to_numpy(dtype=float) >= 1,
                            trajectory_source.start_end_difference.to_numpy(dtype=float) <= -1,
                        ],
                        ["weak_to_strong", "strong_to_weak"],
                        default="limited_start_end_change",
                    )
                for grouping, group_values in grouping_values.items():
                    for group_name in pd.unique(group_values):
                        if pd.isna(group_name):
                            continue
                        group_mask = group_values == group_name
                        grouped_rows.append({
                            "model": model, "subset": key,
                            "grouping": grouping, "group": group_name,
                            "clips": int(group_mask.sum()),
                            "mean_absolute_margin_difference": float(absolute_error[group_mask].mean()),
                            "prediction_agreement": float((prediction[group_mask] == full_prediction[group_mask]).mean()),
                        })
                if subset in named_subsets:
                    for position, (cid, target_label, error, pred, target) in enumerate(
                        zip(ids, y, absolute_error, prediction, full_prediction)
                    ):
                        record = {
                            "result_type": "predefined", "sampling_method": named_subsets[subset],
                            "model": model, "clip_id": int(cid), "subset": key,
                            "label": int(target_label), "emotion": EMOTIONS[int(target_label)],
                            "absolute_margin_difference": float(error),
                            "prediction_disagreement": int(pred != target),
                        }
                        if trajectory is not None:
                            neutral = not bool(eligible[position])
                            record["apex_covered"] = np.nan if neutral else int(apex_positions[position] in subset)
                            has_high = bool((trajectory[position] >= 4).any())
                            record["high_intensity_covered"] = (
                                int((trajectory[position, subset] >= 4).any())
                                if not neutral and has_high else np.nan
                            )
                            record["expression_interval_covered"] = np.nan if neutral else int((trajectory[position, subset] >= 1).any())
                        if concordance is not None:
                            record["label_concordant_frame_covered"] = int(concordance[position, subset].any())
                        oracle_rows.append(record)
                if per_clip_handle:
                    records = pd.DataFrame({
                        "model": model, "clip_id": ids, "subset": key,
                        "absolute_margin_difference": absolute_error,
                        "prediction_agreement": prediction == full_prediction,
                    })
                    records.to_csv(per_clip_handle, index=False, header=per_clip_writer is None)
                    per_clip_writer = True
            oracle_rows.extend({
                "result_type": "per_clip_oracle", "sampling_method": "per_clip_oracle",
                "model": model, "clip_id": int(cid), "subset": subset,
                "label": int(target_label), "emotion": EMOTIONS[int(target_label)],
                "absolute_margin_difference": float(error), "prediction_disagreement": int(disagreement),
            } for cid, target_label, subset, error, disagreement in zip(ids, y, best_subset, best_error, best_disagreement))
    finally:
        if per_clip_handle:
            per_clip_handle.close()
    summary_frame = pd.DataFrame(summaries)
    summary_frame["selection_scope"] = "same_split_descriptive"
    summary_frame.to_csv(output / "subset_results.csv.gz", index=False, compression="gzip")
    pd.DataFrame(grouped_rows).to_csv(
        output / "subset_group_results.csv.gz", index=False, compression="gzip"
    )
    best_fixed = []
    for model, group in summary_frame.groupby("model"):
        error_best = group.sort_values(["mean_absolute_margin_difference", "subset"]).iloc[0]
        f1_best = group.sort_values(["macro_f1", "subset"], ascending=[False, True]).iloc[0]
        best_fixed.extend([
            {"model": model, "criterion": "minimum_mean_absolute_margin_difference", "subset": error_best.subset, "value": error_best.mean_absolute_margin_difference, "interpretation": "same-split descriptive fixed subset"},
            {"model": model, "criterion": "maximum_macro_f1", "subset": f1_best.subset, "value": f1_best.macro_f1, "interpretation": "same-split descriptive fixed subset"},
        ])
    pd.DataFrame(best_fixed).to_csv(output / "best_fixed_descriptive.csv", index=False)

    if args.training_frame_logits:
        train_models, train_ids, train_labels, train_logits = load_logits(args.training_frame_logits)
        del train_ids
        train_selection_rows, held_out_rows = [], []
        for model in sorted(set(models)):
            test_mask = models == model
            train_mask = train_models == model
            if not train_mask.any():
                raise ValueError(f"Training logits do not contain model {model}")
            candidates = []
            for subset in itertools.combinations(range(16), 4):
                statistics = model_subset_statistics(train_logits[train_mask], train_labels[train_mask], subset)
                candidates.append({
                    "model": model,
                    "subset": "-".join(str(index + 1) for index in subset),
                    **statistics,
                })
            candidate_frame = pd.DataFrame(candidates)
            train_selection_rows.extend(candidates)
            chosen = {
                "minimum_mean_absolute_margin_difference": candidate_frame.sort_values(
                    ["mean_absolute_margin_difference", "subset"]
                ).iloc[0],
                "maximum_macro_f1": candidate_frame.sort_values(
                    ["macro_f1", "subset"], ascending=[False, True]
                ).iloc[0],
            }
            for criterion, training_row in chosen.items():
                subset = tuple(int(value) - 1 for value in training_row.subset.split("-"))
                test_statistics = model_subset_statistics(logits[test_mask], labels[test_mask], subset)
                held_out_rows.append({
                    "model": model, "criterion": criterion,
                    "subset": training_row.subset,
                    "selection_split": "official_train", "evaluation_split": "official_test",
                    "training_criterion_value": float(
                        training_row.mean_absolute_margin_difference
                        if criterion.startswith("minimum") else training_row.macro_f1
                    ),
                    **{f"test_{key}": value for key, value in test_statistics.items()},
                })
        pd.DataFrame(train_selection_rows).to_csv(
            output / "training_subset_selection.csv.gz", index=False, compression="gzip"
        )
        pd.DataFrame(held_out_rows).to_csv(output / "train_selected_test_results.csv", index=False)
    outcomes = pd.DataFrame(oracle_rows)
    outcomes[outcomes.result_type == "predefined"].to_csv(output / "sampling_outcomes.csv.gz", index=False, compression="gzip")
    outcomes[outcomes.result_type == "per_clip_oracle"].to_csv(output / "per_clip_oracle.csv", index=False)
    pd.DataFrame([
        {"comparison": "predefined_method", "scope": "new clips without outcome access", "examples": "early_4|uniform_4|late_4"},
        {"comparison": "fixed_subset_search", "scope": "one subset shared by all clips; descriptive unless selected on separate training data", "examples": "best_fixed_descriptive.csv"},
        {"comparison": "training_selected_fixed_subset", "scope": "one subset selected on official training clips and evaluated once on official test clips", "examples": "train_selected_test_results.csv"},
        {"comparison": "per_clip_oracle", "scope": "uses each clip's full-sequence outcome; upper-bound description only", "examples": "per_clip_oracle.csv"},
    ]).to_csv(output / "selection_scope_key.csv", index=False)
    print(f"Evaluated 1,820 four-frame subsets per model; outputs in {output}")


if __name__ == "__main__":
    main()
