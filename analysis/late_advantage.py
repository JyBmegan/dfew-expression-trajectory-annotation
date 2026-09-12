from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from .common import EMOTIONS, parse_curve, true_class_margin
from .frame_subsets import load_logits


def stratified_bootstrap(values: np.ndarray, labels: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    draws = np.empty(repeats, dtype=float)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    for repeat in range(repeats):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        draws[repeat] = values[sample].mean()
    return tuple(np.quantile(draws, [.025, .975]))


def direction_group(value: float, threshold: float) -> str:
    if value >= threshold:
        return "weak_to_strong"
    if value <= -threshold:
        return "strong_to_weak"
    return "limited_start_end_change"


def adjusted_estimates(frame: pd.DataFrame) -> tuple[float, float]:
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    category = encoder.fit_transform(frame[["emotion"]])
    trajectory = frame[["start_end_difference"]].to_numpy(dtype=float)
    x = np.c_[trajectory, category]
    model = LinearRegression().fit(x, frame["late_minus_early_margin"])
    at_no_start_end_change = np.c_[np.zeros(len(frame)), category]
    return float(model.coef_[0]), float(model.predict(at_no_start_end_change).mean())


def bootstrap_adjusted(frame: pd.DataFrame, repeats: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    slopes, adjusted_means = [], []
    grouped = [group.index.to_numpy() for _, group in frame.groupby("emotion", sort=False)]
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in grouped])
        slope, adjusted_mean = adjusted_estimates(frame.loc[sample].reset_index(drop=True))
        slopes.append(slope); adjusted_means.append(adjusted_mean)
    slope_lower, slope_upper = np.quantile(slopes, [.025, .975])
    mean_lower, mean_upper = np.quantile(adjusted_means, [.025, .975])
    return float(slope_lower), float(slope_upper), float(mean_lower), float(mean_upper)


def adjusted_coefficients(frame: pd.DataFrame, predictors: list[str]) -> dict[str, float]:
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    category = encoder.fit_transform(frame[["emotion"]])
    numeric = frame[predictors].to_numpy(dtype=float)
    model = LinearRegression().fit(np.c_[numeric, category], frame["late_minus_early_margin"])
    return {name: float(model.coef_[index]) for index, name in enumerate(predictors)}


def bootstrap_focal_coefficient(
    frame: pd.DataFrame, predictors: list[str], focal: str,
    repeats: int, rng: np.random.Generator,
) -> tuple[float, float]:
    grouped = [group.index.to_numpy() for _, group in frame.groupby("emotion", sort=False)]
    values = []
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in grouped])
        coefficients = adjusted_coefficients(frame.loc[sample].reset_index(drop=True), predictors)
        values.append(coefficients[focal])
    return tuple(np.quantile(values, [.025, .975]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Relate early-late model evidence to human expression trajectories.")
    parser.add_argument("--frame-logits", required=True, nargs="+")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--stimulus-position")
    parser.add_argument("--output-dir", default="artifacts/late_advantage")
    parser.add_argument("--direction-threshold", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    models, clip_ids, labels, logits = load_logits(args.frame_logits)
    trajectories = pd.read_csv(args.trajectories)
    if "split" in trajectories:
        trajectories = trajectories[trajectories.split == "test"]
    trajectories = trajectories.set_index("clip_id")
    rows = []
    for model in sorted(set(models)):
        model_mask = models == model
        ids, y, values = clip_ids[model_mask], labels[model_mask], logits[model_mask]
        for budget, early_positions, late_positions in [(1, [0], [15]), (4, [0, 1, 2, 3], [12, 13, 14, 15])]:
            early_logits = values[:, early_positions].mean(axis=1)
            late_logits = values[:, late_positions].mean(axis=1)
            difference = true_class_margin(late_logits, y) - true_class_margin(early_logits, y)
            for index, clip_id in enumerate(ids):
                if int(clip_id) not in trajectories.index:
                    continue
                rating = trajectories.loc[int(clip_id)]
                curve = parse_curve(rating)
                start_end = float(curve[-1] - curve[0])
                rows.append({
                    "clip_id": int(clip_id), "model": model, "budget": budget,
                    "label": int(y[index]), "emotion": EMOTIONS[int(y[index])],
                    "late_minus_early_margin": float(difference[index]),
                    "early_prediction": int(early_logits[index].argmax()),
                    "late_prediction": int(late_logits[index].argmax()),
                    "start_end_difference": start_end,
                    "direction_group": direction_group(start_end, args.direction_threshold),
                    "apex_position": rating.get("apex_position", np.nan),
                    "trajectory_cluster": rating.get("cluster", np.nan),
                })
    clip = pd.DataFrame(rows)
    if clip.empty:
        raise ValueError("No overlapping test clips were found in logits and trajectory ratings")
    if args.stimulus_position:
        position = pd.read_csv(args.stimulus_position).drop(columns=["split"], errors="ignore")
        clip = clip.merge(position, on="clip_id", how="left")

    rng = np.random.default_rng(20260912)
    summary, decomposition, associations, position_associations = [], [], [], []
    for (model, budget), group in clip.groupby(["model", "budget"], sort=True):
        lower, upper = stratified_bootstrap(
            group.late_minus_early_margin.to_numpy(), group.label.to_numpy(), args.bootstrap, rng,
        )
        summary.append({
            "model": model, "budget": budget, "grouping": "all", "group": "all",
            "clips": len(group), "mean_late_minus_early_margin": group.late_minus_early_margin.mean(),
            "ci_lower": lower, "ci_upper": upper,
        })
        for grouping in ["direction_group", "trajectory_cluster", "emotion"]:
            available = group.dropna(subset=[grouping])
            for name, subset in available.groupby(grouping, sort=True):
                lo, hi = stratified_bootstrap(
                    subset.late_minus_early_margin.to_numpy(), subset.label.to_numpy(), args.bootstrap, rng,
                )
                proportion = len(subset) / len(group)
                mean = subset.late_minus_early_margin.mean()
                summary.append({
                    "model": model, "budget": budget, "grouping": grouping, "group": name,
                    "clips": len(subset), "mean_late_minus_early_margin": mean,
                    "ci_lower": lo, "ci_upper": hi,
                })
                decomposition.append({
                    "model": model, "budget": budget, "grouping": grouping, "group": name,
                    "clips": len(subset), "clip_proportion": proportion,
                    "within_group_mean": mean, "weighted_contribution_to_overall_mean": proportion * mean,
                })
        analysis = group.dropna(subset=["start_end_difference", "emotion"]).reset_index(drop=True)
        slope, adjusted_mean = adjusted_estimates(analysis)
        lo, hi, adjusted_lo, adjusted_hi = bootstrap_adjusted(analysis, args.bootstrap, rng)
        associations.append({
            "model": model, "budget": budget, "clips": len(analysis),
            "margin_change_per_intensity_level": slope, "ci_lower": lo, "ci_upper": hi,
            "adjusted_late_advantage_at_no_start_end_change": adjusted_mean,
            "adjusted_advantage_ci_lower": adjusted_lo, "adjusted_advantage_ci_upper": adjusted_hi,
            "covariate": "DFEW emotion category",
        })
        for measure in [
            "late_minus_early_brightness",
            "late_minus_early_contrast",
            "late_minus_early_sharpness",
        ]:
            if measure not in group:
                continue
            quality = group.dropna(subset=["start_end_difference", "emotion", measure]).reset_index(drop=True)
            standard_deviation = float(quality[measure].std(ddof=0))
            if len(quality) < 20 or not np.isfinite(standard_deviation) or standard_deviation == 0:
                continue
            quality = quality.assign(position_quality_z=(quality[measure] - quality[measure].mean()) / standard_deviation)
            predictors = ["start_end_difference", "position_quality_z"]
            coefficients = adjusted_coefficients(quality, predictors)
            lower, upper = bootstrap_focal_coefficient(
                quality, predictors, "position_quality_z",
                min(args.bootstrap, 2000), rng,
            )
            position_associations.append({
                "model": model, "budget": budget, "clips": len(quality),
                "position_measure": measure,
                "margin_change_per_position_measure_sd": coefficients["position_quality_z"],
                "ci_lower": float(lower), "ci_upper": float(upper),
                "trajectory_coefficient_in_same_model": coefficients["start_end_difference"],
                "covariates": "human start-end intensity difference and DFEW emotion category",
            })

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    clip.to_csv(output / "late_advantage_clip.csv.gz", index=False, compression="gzip")
    pd.DataFrame(summary).to_csv(output / "late_advantage_summary.csv", index=False)
    pd.DataFrame(decomposition).to_csv(output / "late_advantage_decomposition.csv", index=False)
    pd.DataFrame(associations).to_csv(output / "trajectory_margin_association.csv", index=False)
    pd.DataFrame(position_associations, columns=[
        "model", "budget", "clips", "position_measure",
        "margin_change_per_position_measure_sd", "ci_lower", "ci_upper",
        "trajectory_coefficient_in_same_model", "covariates",
    ]).to_csv(output / "position_quality_margin_association.csv", index=False)
    print(f"Late-advantage trajectory analysis: {clip.clip_id.nunique()} test clips, {clip.model.nunique()} models")


if __name__ == "__main__":
    main()
