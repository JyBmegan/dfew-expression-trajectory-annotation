from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .common import parse_curve, require_columns, trajectory_descriptors


def derived_features(ratings: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    rows, curves = [], []
    for _, source in ratings.iterrows():
        curve = parse_curve(source)
        curves.append(curve)
        category = source.get("dominant_category", "")
        rows.append({
            "clip_id": int(source.clip_id),
            "split": source.get("split", "test"),
            "dominant_category": category,
            **trajectory_descriptors(curve, category),
        })
    return pd.DataFrame(rows), np.vstack(curves)


def choose_clusters(differences: np.ndarray, clip_ids: np.ndarray, k_min: int = 2, k_max: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluations = []
    candidate_labels = {}
    for k in range(k_min, min(k_max, len(differences) - 1) + 1):
        full = KMeans(n_clusters=k, n_init=30, random_state=42).fit_predict(differences)
        candidate_labels[k] = full
        silhouette = silhouette_score(differences, full) if len(set(full)) > 1 else np.nan
        stability = []
        rng = np.random.default_rng(12345)
        for repeat in range(20):
            sample = np.sort(rng.choice(len(differences), size=max(k * 4, round(.8 * len(differences))), replace=False))
            subsample_model = KMeans(n_clusters=k, n_init=20, random_state=1000 + repeat).fit(differences[sample])
            stability.append(adjusted_rand_score(full, subsample_model.predict(differences)))
        evaluations.append({"k": k, "silhouette": silhouette, "stability_ari": float(np.mean(stability))})
    evaluation = pd.DataFrame(evaluations)
    eligible = evaluation[evaluation.stability_ari >= 0.70]
    chosen_k = int((eligible if not eligible.empty else evaluation).sort_values(["silhouette", "stability_ari"], ascending=False).iloc[0].k)
    labels = candidate_labels[chosen_k]
    assignment = pd.DataFrame({"clip_id": clip_ids.astype(int), "cluster": labels + 1, "chosen_k": chosen_k})
    return assignment, evaluation.assign(chosen_k=chosen_k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--label-clarity")
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    ratings = pd.read_csv(args.ratings)
    require_columns(ratings, ["clip_id"], "trajectory ratings")
    features, curves = derived_features(ratings)
    cluster_eligible = (
        ~features.dominant_category.isin({"Neutral", "Face not visible", "No consensus"})
        & features.apex_position.notna()
    )
    if cluster_eligible.sum() < 7:
        raise ValueError("At least seven trajectories with a visible non-neutral expression are required for clustering")
    assignments, evaluation = choose_clusters(
        np.diff(curves[cluster_eligible.to_numpy()], axis=1),
        features.loc[cluster_eligible, "clip_id"].to_numpy(),
    )
    features = features.merge(assignments, on="clip_id", how="left")
    features["cluster_eligible"] = cluster_eligible.to_numpy(dtype=bool)
    clarity_summary = pd.DataFrame(columns=[
        "clarity_quartile", "clips", "mean_vote_entropy",
        "mean_start_end_difference", "mean_total_variation",
        "mean_adjacent_change", "mean_high_intensity_fraction",
    ])
    if args.label_clarity:
        clarity = pd.read_csv(args.label_clarity)
        features = features.merge(clarity, on="clip_id", how="left")
        if features.vote_entropy.notna().sum() >= 4:
            features["clarity_quartile"] = pd.qcut(
                features.vote_entropy.rank(method="first"), 4,
                labels=["lowest_entropy", "lower_middle", "upper_middle", "highest_entropy"],
            )
            clarity_summary = features.groupby("clarity_quartile", observed=True, as_index=False).agg(
                clips=("clip_id", "size"), mean_vote_entropy=("vote_entropy", "mean"),
                mean_start_end_difference=("start_end_difference", "mean"),
                mean_total_variation=("trajectory_total_variation", "mean"),
                mean_adjacent_change=("mean_absolute_adjacent_change", "mean"),
                mean_high_intensity_fraction=("high_intensity_fraction", "mean"),
            )
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "trajectory_consensus.csv", index=False)
    evaluation.to_csv(output / "trajectory_cluster_selection.csv", index=False)
    clarity_summary.to_csv(output / "trajectory_by_label_clarity.csv", index=False)
    curve_rows = []
    for cluster, ids in assignments.groupby("cluster"):
        mask = features.clip_id.isin(ids.clip_id).to_numpy()
        center = curves[mask].mean(axis=0)
        curve_rows.extend({"cluster": int(cluster), "frame": i + 1, "mean_intensity": float(value)} for i, value in enumerate(center))
    pd.DataFrame(curve_rows).to_csv(output / "trajectory_cluster_curves.csv", index=False)
    print(f"Trajectory analysis complete; selected k={int(evaluation.chosen_k.iloc[0])}")


if __name__ == "__main__":
    main()
