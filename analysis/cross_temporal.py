from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score

from .temporal_models import cache_arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["alexnet", "resnet18"], required=True)
    parser.add_argument("--feature-root", default="local_data/features")
    parser.add_argument("--output", default="artifacts/cross_temporal.csv")
    args = parser.parse_args()
    train_x, _train_ids, train_y, _ = cache_arrays(Path(args.feature_root) / f"{args.backbone}_train")
    test_x, _test_ids, test_y, _ = cache_arrays(Path(args.feature_root) / f"{args.backbone}_test")
    rows = []
    for train_position in range(16):
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1500, tol=1e-4, random_state=42, class_weight="balanced")
        model.fit(np.asarray(train_x[:, train_position], dtype=np.float32), train_y)
        for test_position in range(16):
            prediction = model.predict(np.asarray(test_x[:, test_position], dtype=np.float32))
            rows.append({
                "backbone": args.backbone, "train_position": train_position + 1, "test_position": test_position + 1,
                "accuracy": float((prediction == test_y).mean()),
                "macro_f1": float(f1_score(test_y, prediction, labels=np.arange(7), average="macro", zero_division=0)),
            })
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Cross-temporal matrix written to {output}")


if __name__ == "__main__":
    main()

