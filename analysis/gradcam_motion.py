from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from torchvision import transforms as T

from app.frame_source import FrameSource
from .cache_features import build_model
from .common import EMOTIONS, read_config, resolve_path, true_class_margin


TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def target_layer(model_name: str, model: torch.nn.Module) -> torch.nn.Module:
    if model_name == "alexnet":
        return [module for module in model.features.modules() if isinstance(module, torch.nn.Conv2d)][-1]
    return model.backbone.layer4[-1]


def gradcam(model_name: str, model: torch.nn.Module, image: Image.Image, target: int) -> np.ndarray:
    activation = gradient = None

    def forward_hook(_module, _inputs, output):
        nonlocal activation, gradient
        activation = output
        output.register_hook(lambda value: store_gradient(value))

    def store_gradient(value):
        nonlocal gradient
        gradient = value

    layer = target_layer(model_name, model)
    first = layer.register_forward_hook(forward_hook)
    try:
        tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0)
        model.zero_grad(set_to_none=True)
        score = model(tensor)[0, target]
        score.backward()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heat = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heat = torch.nn.functional.interpolate(heat, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
        heat = heat.detach().cpu().numpy()
        maximum = float(heat.max())
        return heat / maximum if maximum > 0 else np.zeros_like(heat)
    finally:
        first.remove()


def residual_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    points = cv2.goodFeaturesToTrack(previous, maxCorners=250, qualityLevel=.01, minDistance=5, blockSize=5)
    transform = None
    if points is not None and len(points) >= 6:
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        valid = status.reshape(-1).astype(bool)
        if valid.sum() >= 6:
            transform, inliers = cv2.estimateAffinePartial2D(
                points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid],
                method=cv2.RANSAC, ransacReprojThreshold=2.5,
            )
            if transform is not None:
                linear = transform[:, :2]
                scale = float(np.sqrt(linear[0, 0] ** 2 + linear[1, 0] ** 2))
                rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
                translation = float(np.linalg.norm(transform[:, 2]))
                inlier_rate = float(inliers.mean()) if inliers is not None else 0
                if not (.80 <= scale <= 1.25 and abs(rotation) <= 30 and translation <= .25 * np.hypot(*previous.shape) and inlier_rate >= .25):
                    transform = None
    if transform is None:
        transform = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    warped = cv2.warpAffine(previous, transform, (previous.shape[1], previous.shape[0]), borderMode=cv2.BORDER_REFLECT)
    flow = cv2.calcOpticalFlowFarneback(warped, current, None, .5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    high = np.quantile(magnitude, .995)
    return np.clip(magnitude / high, 0, 1) if high > 0 else np.zeros_like(magnitude)


def spatial_agreement(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    a, b = first.reshape(-1), second.reshape(-1)
    correlation = float(spearmanr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    a_high = a >= np.quantile(a, .80)
    b_high = b >= np.quantile(b, .80)
    union = np.logical_or(a_high, b_high).sum()
    iou = float(np.logical_and(a_high, b_high).sum() / union) if union else np.nan
    return correlation, iou


def color_overlay(image: Image.Image, heat: np.ndarray, alpha: float = .48) -> Image.Image:
    base = np.asarray(image.resize((224, 224))).astype(np.float32)
    color = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)
    return Image.fromarray(np.uint8(np.clip((1 - alpha) * base + alpha * color, 0, 255)))


def save_transition_panel(path: Path, first: Image.Image, second: Image.Image, first_cam: np.ndarray, second_cam: np.ndarray, deformation: np.ndarray) -> None:
    tiles = [
        color_overlay(first, first_cam), color_overlay(second, second_cam),
        color_overlay(second, np.abs(second_cam - first_cam)), color_overlay(second, deformation),
    ]
    labels = ["Attribution t", "Attribution t+1", "Attribution change", "Non-rigid deformation"]
    canvas = Image.new("RGB", (4 * 224 + 3 * 8, 258), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        x = index * 232
        canvas.paste(tile, (x, 0))
        draw.text((x + 5, 232), label, fill=(32, 53, 80))
    canvas.save(path)


def choose_transitions(logits: pd.DataFrame, trajectories: pd.DataFrame, per_stratum: int) -> pd.DataFrame:
    logit_columns = [f"logit_{index}" for index in range(7)]
    records = []
    for (model, clip_id), clip in logits.sort_values("frame").groupby(["model", "clip_id"]):
        if len(clip) != 16:
            continue
        label = int(clip.label.iloc[0])
        values = clip[logit_columns].to_numpy(dtype=np.float32)
        margins = true_class_margin(values, np.full(16, label))
        for transition in range(1, 16):
            records.append({
                "model": model, "clip_id": int(clip_id), "label": label,
                "emotion": EMOTIONS[label], "transition": transition,
                "absolute_margin_change": float(abs(margins[transition] - margins[transition - 1])),
                "signed_margin_change": float(margins[transition] - margins[transition - 1]),
            })
    frame = pd.DataFrame(records).merge(trajectories, on="clip_id", how="left", suffixes=("", "_trajectory"))
    strata = ["model", "emotion"] + (["cluster"] if "cluster" in frame and frame.cluster.notna().any() else [])
    best_per_clip = (
        frame.sort_values(["model", "clip_id", "absolute_margin_change"], ascending=[True, True, False])
        .groupby(["model", "clip_id"], as_index=False, sort=False)
        .head(1)
    )
    selected = (
        best_per_clip.sort_values("absolute_margin_change", ascending=False)
        .groupby(strata, dropna=False)
        .head(per_stratum)
    )
    return selected.sort_values(strata + ["absolute_margin_change"], ascending=[True] * len(strata) + [False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare true-class spatial attribution changes with non-rigid image deformation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--frame-logits", required=True, nargs="+")
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--output-dir", default="artifacts/gradcam_motion")
    args = parser.parse_args()

    config = read_config(args.config); paths = config["paths"]
    source_value = paths.get("frames_16_archive", paths["frames"])
    source = FrameSource(resolve_path(config, source_value), paths.get("frames_archive_password"))
    logits = pd.concat([pd.read_csv(path) for path in args.frame_logits], ignore_index=True)
    trajectories = pd.read_csv(args.trajectories)
    if "split" in trajectories:
        trajectories = trajectories[trajectories.split == "test"]
    selected = choose_transitions(logits, trajectories, args.per_stratum)
    output = Path(args.output_dir); maps_dir = output / "maps"; maps_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output / "overlays"; overlays_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_transitions.csv", index=False)

    results = []; aggregates = {}
    models = {}
    for model_name in selected.model.unique():
        checkpoint = torch.load(resolve_path(config, paths[f"{model_name}_checkpoint"]), map_location="cpu", weights_only=False)
        models[model_name] = build_model(model_name, checkpoint)
    for number, row in enumerate(selected.itertuples(), start=1):
        first_bytes = source.read(int(row.clip_id), int(row.transition))
        second_bytes = source.read(int(row.clip_id), int(row.transition) + 1)
        first_image = Image.open(BytesIO(first_bytes)).convert("RGB")
        second_image = Image.open(BytesIO(second_bytes)).convert("RGB")
        first_cam = gradcam(row.model, models[row.model], first_image, int(row.label))
        second_cam = gradcam(row.model, models[row.model], second_image, int(row.label))
        attribution_change = np.abs(second_cam - first_cam)
        first_gray = cv2.cvtColor(np.asarray(first_image), cv2.COLOR_RGB2GRAY)
        second_gray = cv2.cvtColor(np.asarray(second_image), cv2.COLOR_RGB2GRAY)
        deformation = residual_flow(first_gray, second_gray)
        current_correlation, current_iou = spatial_agreement(second_cam, deformation)
        change_correlation, change_iou = spatial_agreement(attribution_change, deformation)
        identifier = f"{row.model}_{int(row.clip_id):05d}_{int(row.transition):02d}_{int(row.transition)+1:02d}"
        np.savez_compressed(
            maps_dir / f"{identifier}.npz", first_gradcam=first_cam, second_gradcam=second_cam,
            attribution_change=attribution_change, nonrigid_deformation=deformation,
        )
        save_transition_panel(
            overlays_dir / f"{identifier}.png", first_image, second_image,
            first_cam, second_cam, deformation,
        )
        cluster = getattr(row, "cluster", np.nan)
        aggregate_key = (row.model, row.emotion, str(cluster))
        if aggregate_key not in aggregates:
            aggregates[aggregate_key] = {
                "count": 0, "second": np.zeros_like(second_cam),
                "change": np.zeros_like(attribution_change), "deformation": np.zeros_like(deformation),
            }
        aggregates[aggregate_key]["count"] += 1
        aggregates[aggregate_key]["second"] += second_cam
        aggregates[aggregate_key]["change"] += attribution_change
        aggregates[aggregate_key]["deformation"] += deformation
        results.append({
            "model": row.model, "clip_id": int(row.clip_id), "label": int(row.label),
            "emotion": row.emotion, "transition": int(row.transition),
            "absolute_margin_change": float(row.absolute_margin_change),
            "current_cam_deformation_spearman": current_correlation,
            "current_cam_deformation_top20_iou": current_iou,
            "cam_change_deformation_spearman": change_correlation,
            "cam_change_deformation_top20_iou": change_iou,
            "map_file": str((maps_dir / f"{identifier}.npz").resolve()),
        })
        if number % 20 == 0:
            pd.DataFrame(results).to_csv(output / "gradcam_motion_overlap.csv", index=False)
            print(f"Processed {number}/{len(selected)} selected transitions")
    source.close()
    pd.DataFrame(results).to_csv(output / "gradcam_motion_overlap.csv", index=False)
    aggregate_dir = output / "aggregate_maps"; aggregate_dir.mkdir(exist_ok=True)
    aggregate_rows = []
    for (model_name, emotion, cluster), values in aggregates.items():
        count = values["count"]
        second = values["second"] / count; change = values["change"] / count; deformation = values["deformation"] / count
        current_correlation, current_iou = spatial_agreement(second, deformation)
        change_correlation, change_iou = spatial_agreement(change, deformation)
        name = f"{model_name}_{emotion}_cluster-{cluster}".replace(" ", "_")
        np.savez_compressed(aggregate_dir / f"{name}.npz", mean_gradcam=second, mean_attribution_change=change, mean_nonrigid_deformation=deformation)
        aggregate_rows.append({
            "model": model_name, "emotion": emotion, "trajectory_cluster": cluster, "transitions": count,
            "mean_cam_deformation_spearman": current_correlation, "mean_cam_deformation_top20_iou": current_iou,
            "mean_cam_change_deformation_spearman": change_correlation, "mean_cam_change_deformation_top20_iou": change_iou,
        })
    pd.DataFrame(aggregate_rows).to_csv(output / "balanced_aggregate_overlap.csv", index=False)
    print(f"Spatial attribution analysis complete: {len(results)} transitions")


if __name__ == "__main__":
    main()
