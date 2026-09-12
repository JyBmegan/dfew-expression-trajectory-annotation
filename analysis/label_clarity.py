from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import read_config, resolve_path


def compute(annotation_path: Path) -> pd.DataFrame:
    source = pd.read_excel(annotation_path)
    required = ["1happy", "2sad", "3neutral", "4angry", "5surprise", "6disgust", "7fear", "order", "label"]
    missing = set(required) - set(source.columns)
    if missing:
        raise ValueError(f"DFEW annotation table is missing {sorted(missing)}")
    votes = source[required[:7]].to_numpy(dtype=float)
    if (votes < 0).any() or not np.allclose(votes.sum(axis=1), 10):
        raise ValueError("Each DFEW annotation row must contain ten non-negative votes")
    totals = votes.sum(axis=1, keepdims=True)
    proportions = np.divide(votes, totals, out=np.zeros_like(votes), where=totals > 0)
    safe = np.where(proportions > 0, proportions, 1)
    ordered = np.sort(votes, axis=1)
    return pd.DataFrame({
        "clip_id": source["order"].astype(int),
        "dfew_label": source["label"].astype(int),
        "vote_max": votes.max(axis=1),
        "vote_margin": ordered[:, -1] - ordered[:, -2],
        "vote_entropy": -(proportions * np.log(safe)).sum(axis=1) / np.log(7),
        "neutral_votes": votes[:, 2],
        **{f"votes_{index + 1}": votes[:, index] for index in range(7)},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/label_clarity.csv")
    args = parser.parse_args()
    config = read_config(args.config)
    result = compute(resolve_path(config, config["paths"]["annotation_xlsx"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Wrote {len(result)} clip-level label-clarity records to {output}")


if __name__ == "__main__":
    main()
