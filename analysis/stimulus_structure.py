from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .common import parse_curve


def stratified_spearman(frame: pd.DataFrame, x: str, y: str, strata: str, repeats: int = 2000) -> dict:
    valid = frame.dropna(subset=[x, y, strata]).reset_index(drop=True)
    if len(valid) < 10 or valid[x].nunique() < 2 or valid[y].nunique() < 2:
        return {"n": len(valid), "spearman": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    observed = float(spearmanr(valid[x], valid[y]).statistic)
    groups = [group.index.to_numpy() for _, group in valid.groupby(strata, sort=False)]
    rng = np.random.default_rng(20260912)
    values = []
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        value = spearmanr(valid.loc[sample, x], valid.loc[sample, y]).statistic
        if np.isfinite(value):
            values.append(value)
    lower, upper = np.quantile(values, [.025, .975]) if values else (np.nan, np.nan)
    return {"n": len(valid), "spearman": observed, "ci_lower": lower, "ci_upper": upper}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize temporal position, boundary quality, and original-timebase effects.")
    parser.add_argument("--frame-features", required=True)
    parser.add_argument("--transition-features", required=True)
    parser.add_argument("--stimulus-features", required=True)
    parser.add_argument("--trajectories")
    parser.add_argument("--sampling-outcomes")
    parser.add_argument("--output-dir", default="artifacts/stimulus_structure")
    args = parser.parse_args()

    frames = pd.read_csv(args.frame_features)
    transitions = pd.read_csv(args.transition_features)
    clips = pd.read_csv(args.stimulus_features)
    frame_measures = [column for column in [
        "brightness", "contrast", "sharpness", "face_detected", "face_center_x", "face_center_y", "face_scale",
    ] if column in frames]
    transition_measures = [column for column in [
        "mean_abs_change", "brightness_change", "sharpness_change", "exact_duplicate", "near_duplicate",
        "registration_valid", "rigid_translation", "rigid_rotation_deg", "rigid_scale_change",
        "nonrigid_flow", "photometric_residual",
    ] if column in transitions]
    position_summary = frames.groupby(["split", "frame"])[frame_measures].agg(["mean", "median", "count"])
    position_summary.columns = ["_".join(column) for column in position_summary.columns]
    position_summary = position_summary.reset_index()
    transition_summary = transitions.groupby(["split", "transition"])[transition_measures].agg(["mean", "median", "count"])
    transition_summary.columns = ["_".join(column) for column in transition_summary.columns]
    transition_summary = transition_summary.reset_index()

    frames["region"] = np.where(frames.frame.isin([1, 16]), "boundary", "interior")
    transitions["region"] = np.where(transitions.transition.isin([1, 15]), "boundary", "interior")
    boundary_frame = frames.groupby(["split", "region"])[frame_measures].mean().reset_index()
    boundary_transition = transitions.groupby(["split", "region"])[transition_measures].mean().reset_index()

    association_rows = []
    duration_cluster = pd.DataFrame(columns=[
        "original_length_quartile", "cluster", "clips", "within_length_proportion"
    ])
    if args.trajectories:
        trajectories = pd.read_csv(args.trajectories)
        trajectory_rows = []
        for _, row in trajectories.iterrows():
            curve = parse_curve(row)
            trajectory_rows.append({
                "clip_id": int(row.clip_id), "dominant_category": row.get("dominant_category", ""),
                "trajectory_total_variation": float(np.abs(np.diff(curve)).sum()),
                "trajectory_mean_adjacent_change": float(np.abs(np.diff(curve)).mean()),
                "peak_width": row.get("peak_width", np.nan),
                "high_intensity_fraction": row.get("high_intensity_fraction", np.nan),
                "cluster": row.get("cluster", np.nan),
            })
        analysis = clips.merge(pd.DataFrame(trajectory_rows), on="clip_id", how="inner")
        analysis["log_original_frame_count"] = np.log(analysis.original_frame_count)
        for outcome in ["trajectory_total_variation", "trajectory_mean_adjacent_change", "peak_width", "high_intensity_fraction"]:
            result = stratified_spearman(analysis, "log_original_frame_count", outcome, "dominant_category")
            association_rows.append({"predictor": "log_original_frame_count", "outcome": outcome, **result})
        if analysis.original_frame_count.notna().sum() >= 4:
            analysis["original_length_quartile"] = pd.qcut(
                analysis.original_frame_count.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"],
            )
            duration_cluster = analysis.groupby(["original_length_quartile", "cluster"], observed=True).size().rename("clips").reset_index()
            totals = duration_cluster.groupby("original_length_quartile").clips.transform("sum")
            duration_cluster["within_length_proportion"] = duration_cluster.clips / totals

    if args.sampling_outcomes:
        sampling = pd.read_csv(args.sampling_outcomes).merge(
            clips[["clip_id", "original_frame_count"]], on="clip_id", how="left",
        )
        sampling["log_original_frame_count"] = np.log(sampling.original_frame_count)
        for (model, method), group in sampling.groupby(["model", "sampling_method"]):
            group = group.assign(stratum="all")
            result = stratified_spearman(group, "log_original_frame_count", "absolute_margin_difference", "stratum")
            association_rows.append({
                "predictor": "log_original_frame_count", "outcome": "absolute_margin_difference",
                "model": model, "sampling_method": method, **result,
            })

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    position_summary.to_csv(output / "frame_position_summary.csv", index=False)
    transition_summary.to_csv(output / "transition_position_summary.csv", index=False)
    boundary_frame.to_csv(output / "frame_boundary_comparison.csv", index=False)
    boundary_transition.to_csv(output / "transition_boundary_comparison.csv", index=False)
    pd.DataFrame(association_rows, columns=[
        "predictor", "outcome", "model", "sampling_method", "n",
        "spearman", "ci_lower", "ci_upper",
    ]).to_csv(output / "timebase_associations.csv", index=False)
    duration_cluster.to_csv(output / "trajectory_cluster_by_length.csv", index=False)
    print(f"Stimulus-structure summaries written to {output}")


if __name__ == "__main__":
    main()
