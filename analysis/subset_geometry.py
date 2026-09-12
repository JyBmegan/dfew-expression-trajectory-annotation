from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


GEOMETRY = [
    "mean_position", "temporal_span", "minimum_gap", "maximum_gap",
    "gap_standard_deviation", "includes_first", "includes_last",
    "temporal_quartiles_covered",
]
OUTCOMES = [
    "mean_absolute_margin_difference", "prediction_agreement",
    "macro_f1", "uar", "war",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe how four-frame temporal geometry relates to fixed-subset outcomes")
    parser.add_argument("--subset-results", required=True)
    parser.add_argument("--output-dir", default="artifacts/subset_geometry")
    args = parser.parse_args()

    frame = pd.read_csv(args.subset_results)
    missing = set(GEOMETRY + OUTCOMES + ["model", "subset"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Subset results are missing {sorted(missing)}")
    correlations = []
    for model, group in frame.groupby("model", sort=True):
        for geometry in GEOMETRY:
            for outcome in OUTCOMES:
                value = spearmanr(group[geometry], group[outcome]).statistic
                correlations.append({
                    "model": model, "geometry": geometry, "outcome": outcome,
                    "spearman": float(value), "subsets": len(group),
                })

    grouped = []
    for keys in [
        ["model", "temporal_span"],
        ["model", "temporal_quartiles_covered"],
        ["model", "includes_first", "includes_last"],
    ]:
        current = frame.groupby(keys, as_index=False)[OUTCOMES].mean()
        current["grouping"] = "|".join(keys[1:])
        grouped.append(current)

    ranks = []
    for model, group in frame.groupby("model", sort=True):
        for criterion, outcome, ascending in [
            ("closest_to_full_margin", "mean_absolute_margin_difference", True),
            ("highest_prediction_agreement", "prediction_agreement", False),
            ("highest_macro_f1", "macro_f1", False),
        ]:
            selected = group.sort_values([outcome, "subset"], ascending=[ascending, True]).head(25).copy()
            selected.insert(1, "criterion", criterion)
            selected.insert(2, "rank", range(1, len(selected) + 1))
            ranks.append(selected[["model", "criterion", "rank", "subset", *GEOMETRY, *OUTCOMES]])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(correlations).to_csv(output / "geometry_correlations.csv", index=False)
    pd.concat(grouped, ignore_index=True, sort=False).to_csv(output / "geometry_group_summaries.csv", index=False)
    pd.concat(ranks, ignore_index=True).to_csv(output / "top_fixed_subsets.csv", index=False)
    print(f"Subset-geometry summaries written to {output}")


if __name__ == "__main__":
    main()
