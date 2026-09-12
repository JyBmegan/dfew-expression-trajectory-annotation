from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import EMOTIONS, require_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DFEW votes with isolated-frame and continuous-sequence judgments.")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--label-clarity", required=True)
    parser.add_argument("--frame-primary", required=True)
    parser.add_argument("--trajectory-consensus", required=True)
    parser.add_argument("--output-dir", default="artifacts/human_label_comparison")
    args = parser.parse_args()

    split = pd.read_csv(args.test_csv).rename(columns={"video_name": "clip_id"})
    split = split[split.label.between(1, 7)][["clip_id", "label"]].drop_duplicates("clip_id")
    split["dfew_category"] = split.label.astype(int).map({index + 1: value for index, value in enumerate(EMOTIONS)})
    clarity = pd.read_csv(args.label_clarity)
    frames = pd.read_csv(args.frame_primary)
    trajectories = pd.read_csv(args.trajectory_consensus)
    if "split" in trajectories:
        trajectories = trajectories[trajectories.split == "test"]
    require_columns(frames, ["clip_id", "frame", "visible_category"], "isolated-frame consensus")
    require_columns(trajectories, ["clip_id", "dominant_category"], "trajectory consensus")

    clip = split.merge(clarity, on="clip_id", how="left").merge(
        trajectories[["clip_id", "dominant_category"]], on="clip_id", how="left",
    )
    clip["sequence_matches_dfew"] = clip.dominant_category == clip.dfew_category
    frame = frames.merge(clip[["clip_id", "dfew_category", "dominant_category"]], on="clip_id", how="left")
    frame["isolated_matches_dfew"] = frame.visible_category == frame.dfew_category
    frame["isolated_matches_sequence"] = frame.visible_category == frame.dominant_category
    per_clip_frame = frame.groupby("clip_id", as_index=False).agg(
        isolated_dfew_agreement_fraction=("isolated_matches_dfew", "mean"),
        isolated_sequence_agreement_fraction=("isolated_matches_sequence", "mean"),
        isolated_frames_available=("frame", "nunique"),
    )
    clip = clip.merge(per_clip_frame, on="clip_id", how="left")
    if clip.vote_entropy.notna().sum() >= 4:
        clip["clarity_quartile"] = pd.qcut(
            clip.vote_entropy.rank(method="first"), 4,
            labels=["lowest_entropy", "lower_middle", "upper_middle", "highest_entropy"],
        )

    position = frame.groupby("frame", as_index=False).agg(
        clips=("clip_id", "nunique"),
        isolated_dfew_agreement=("isolated_matches_dfew", "mean"),
        isolated_sequence_agreement=("isolated_matches_sequence", "mean"),
        mean_isolated_intensity=("intensity", "mean"),
    )
    confusion_sequence = pd.crosstab(
        clip.dfew_category, clip.dominant_category, normalize="index", dropna=False,
    ).reindex(index=EMOTIONS, fill_value=0)
    confusion_isolated = pd.crosstab(
        frame.dfew_category, frame.visible_category, normalize="index", dropna=False,
    ).reindex(index=EMOTIONS, fill_value=0)
    clarity_summary = pd.DataFrame(columns=[
        "clarity_quartile", "clips", "sequence_dfew_agreement",
        "isolated_dfew_agreement", "mean_vote_entropy", "mean_vote_margin",
    ])
    if "clarity_quartile" in clip:
        clarity_summary = clip.groupby("clarity_quartile", observed=True, as_index=False).agg(
            clips=("clip_id", "size"), sequence_dfew_agreement=("sequence_matches_dfew", "mean"),
            isolated_dfew_agreement=("isolated_dfew_agreement_fraction", "mean"),
            mean_vote_entropy=("vote_entropy", "mean"), mean_vote_margin=("vote_margin", "mean"),
        )

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    clip.to_csv(output / "clip_label_alignment.csv", index=False)
    position.to_csv(output / "frame_position_agreement.csv", index=False)
    confusion_sequence.to_csv(output / "dfew_vs_sequence_confusion.csv")
    confusion_isolated.to_csv(output / "dfew_vs_isolated_confusion.csv")
    clarity_summary.to_csv(output / "clarity_agreement_summary.csv", index=False)
    print(f"Human-label comparison complete for {clip.clip_id.nunique()} test clips")


if __name__ == "__main__":
    main()
