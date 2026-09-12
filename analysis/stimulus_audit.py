from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.frame_source import FrameSource
from .common import read_config, resolve_path


FRAME_PATTERN = re.compile(r"(?:^|/)(\d{5})/(?:\1_)?\d+\.jpe?g$", re.IGNORECASE)


def difference_hash(gray: np.ndarray) -> str:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def full_length_counts(root: Path) -> pd.DataFrame:
    counts: Counter[int] = Counter()
    sources = {}
    for archive in sorted(root.rglob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as handle:
                for name in handle.namelist():
                    match = FRAME_PATTERN.search(name)
                    if match:
                        clip_id = int(match.group(1))
                        counts[clip_id] += 1
                        sources.setdefault(clip_id, archive.name)
        except zipfile.BadZipFile:
            continue
    return pd.DataFrame({
        "clip_id": sorted(counts),
        "original_frame_count": [counts[key] for key in sorted(counts)],
        "archive_source": [sources[key] for key in sorted(counts)],
    })


def pose_from_landmarks(points: np.ndarray, width: int, height: int) -> tuple[float, float, float]:
    indices = [1, 152, 33, 263, 61, 291]
    image_points = points[indices].astype(np.float64)
    model_points = np.array([
        [0.0, 0.0, 0.0], [0.0, -63.6, -12.5], [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0], [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1],
    ], dtype=np.float64)
    focal = float(width)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return math.nan, math.nan, math.nan
    matrix, _ = cv2.Rodrigues(rotation)
    angles = cv2.RQDecomp3x3(matrix)[0]
    return float(angles[0]), float(angles[1]), float(angles[2])


class FaceAnalyzer:
    def __init__(self, enabled: bool):
        self.mesh = None
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(cascade_path)) if cascade_path.is_file() else None
        if enabled:
            try:
                import mediapipe as mp
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=0.5,
                )
            except Exception as error:
                print(f"MediaPipe unavailable in this environment; continuing with CPU image features ({error})")

    def close(self):
        if self.mesh:
            self.mesh.close()

    def detect(self, image: np.ndarray) -> np.ndarray | None:
        if self.mesh is None:
            return None
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None
        h, w = image.shape[:2]
        return np.asarray([(point.x * w, point.y * h) for point in result.multi_face_landmarks[0].landmark], dtype=np.float32)

    def detect_box(self, gray: np.ndarray) -> tuple[float, float, float] | None:
        if self.cascade is None or self.cascade.empty():
            return None
        boxes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(boxes) == 0:
            return None
        x, y, width, height = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
        return float(x + width / 2), float(y + height / 2), float(np.sqrt(width * height))


def central_face_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, (width // 2, height // 2), (int(width * .42), int(height * .48)), 0, 0, 360, 1, -1)
    return mask.astype(bool)


def estimate_motion(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    points = cv2.goodFeaturesToTrack(previous, maxCorners=250, qualityLevel=.01, minDistance=5, blockSize=5)
    transform = None
    tracked_count = inlier_count = 0
    if points is not None and len(points) >= 6:
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
        valid = status.reshape(-1).astype(bool)
        tracked_count = int(valid.sum())
        if valid.sum() >= 6:
            transform, inliers = cv2.estimateAffinePartial2D(
                points.reshape(-1, 2)[valid], tracked.reshape(-1, 2)[valid],
                method=cv2.RANSAC, ransacReprojThreshold=2.5,
            )
            inlier_count = int(inliers.sum()) if inliers is not None else 0
    registration_valid = transform is not None
    if transform is not None:
        linear = transform[:, :2]
        proposed_scale = float(np.sqrt(linear[0, 0] ** 2 + linear[1, 0] ** 2))
        proposed_rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        proposed_translation = float(np.linalg.norm(transform[:, 2]))
        diagonal = float(np.hypot(*previous.shape))
        registration_valid = (
            .80 <= proposed_scale <= 1.25 and abs(proposed_rotation) <= 30
            and proposed_translation <= .25 * diagonal and inlier_count >= 6
            and inlier_count / max(tracked_count, 1) >= .25
        )
    if not registration_valid:
        transform = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    linear = transform[:, :2]
    scale = float(np.sqrt(linear[0, 0] ** 2 + linear[1, 0] ** 2))
    rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    translation = float(np.linalg.norm(transform[:, 2]))
    warped = cv2.warpAffine(
        previous, transform, (previous.shape[1], previous.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    flow = cv2.calcOpticalFlowFarneback(warped, current, None, .5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=2)
    mask = central_face_mask(previous.shape)
    nonrigid = float(np.median(magnitude[mask]))
    yy, xx = np.indices(previous.shape, dtype=np.float32)
    reconstructed = cv2.remap(
        warped, xx - flow[..., 0], yy - flow[..., 1],
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    photometric = float(np.mean(np.abs(current[mask].astype(np.float32) - reconstructed[mask].astype(np.float32))))
    aligned = float(np.mean(np.abs(current[mask].astype(np.float32) - warped[mask].astype(np.float32))))
    return {
        "registration_valid": int(registration_valid),
        "tracked_points": tracked_count, "registration_inliers": inlier_count,
        "rigid_translation": translation if registration_valid else math.nan,
        "rigid_rotation_deg": rotation if registration_valid else math.nan,
        "rigid_scale_change": abs(scale - 1) if registration_valid else math.nan,
        "nonrigid_flow": nonrigid,
        "aligned_pixel_change": aligned,
        "photometric_residual": photometric,
    }


def audit_clip(source: FrameSource, clip_id: int, split: str, detector: FaceAnalyzer) -> dict:
    frame_rows, transition_rows = [], []
    images, grays, landmarks = [], [], []
    for index in range(1, 17):
        encoded = np.frombuffer(source.read(clip_id, index), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unreadable image {clip_id:05d}/{index}.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        points = detector.detect(image)
        box = detector.detect_box(gray)
        images.append(image); grays.append(gray); landmarks.append(points)
        if points is None:
            if box is None:
                center_x = center_y = face_scale = math.nan
            else:
                center_x, center_y, face_scale = box
            pitch = yaw = roll = math.nan
        else:
            center_x, center_y = points.mean(axis=0)
            face_scale = float(np.sqrt(np.prod(np.ptp(points, axis=0))))
            pitch, yaw, roll = pose_from_landmarks(points, image.shape[1], image.shape[0])
        frame_rows.append({
            "clip_id": clip_id, "split": split, "frame": index,
            "brightness": float(gray.mean()), "contrast": float(gray.std()),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "dhash": difference_hash(gray), "landmarks_available": int(points is not None),
            "face_detected": int(box is not None or points is not None),
            "face_center_x": center_x, "face_center_y": center_y, "face_scale": face_scale,
            "pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll,
        })
    for index in range(15):
        previous, current = grays[index], grays[index + 1]
        mean_abs = float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))
        row = {
            "clip_id": clip_id, "split": split, "transition": index + 1,
            "mean_abs_change": mean_abs,
            "brightness_change": float(abs(current.mean() - previous.mean())),
            "sharpness_change": float(abs(frame_rows[index + 1]["sharpness"] - frame_rows[index]["sharpness"])),
            "exact_duplicate": int(np.array_equal(previous, current)),
            "near_duplicate": int(mean_abs < 0.5),
            **estimate_motion(previous, current),
        }
        if landmarks[index] is not None and landmarks[index + 1] is not None:
            source, target = landmarks[index], landmarks[index + 1]
            transform, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
            if transform is not None:
                linear = transform[:, :2]
                scale = float(np.sqrt(linear[0, 0] ** 2 + linear[1, 0] ** 2))
                rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
                translation = float(np.linalg.norm(transform[:, 2]))
                homogeneous = np.c_[source, np.ones(len(source))]
                fitted = homogeneous @ transform.T
                nonrigid = float(np.linalg.norm(target - fitted, axis=1).mean())
                warped = cv2.warpAffine(previous, transform, (previous.shape[1], previous.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                aligned = float(np.mean(np.abs(current.astype(np.float32) - warped.astype(np.float32))))
                row.update(
                    registration_valid=1, tracked_points=len(source), registration_inliers=len(source),
                    rigid_translation=translation, rigid_rotation_deg=rotation,
                    rigid_scale_change=abs(scale - 1), landmark_nonrigid=nonrigid,
                )
        else:
            row["landmark_nonrigid"] = math.nan
        transition_rows.append(row)
    frame = pd.DataFrame(frame_rows)
    transition = pd.DataFrame(transition_rows)
    clip = {
        "clip_id": clip_id, "split": split,
        "mean_brightness": frame.brightness.mean(), "brightness_range": frame.brightness.max() - frame.brightness.min(),
        "mean_contrast": frame.contrast.mean(), "minimum_sharpness": frame.sharpness.min(),
        "landmark_coverage": frame.landmarks_available.mean(), "face_detection_rate": frame.face_detected.mean(),
        "face_center_drift": float(np.nanmean(np.sqrt(np.diff(frame.face_center_x) ** 2 + np.diff(frame.face_center_y) ** 2))) if frame.face_center_x.notna().sum() >= 2 else math.nan,
        "face_scale_range": float(frame.face_scale.max() - frame.face_scale.min()),
        "pose_yaw_range": float(frame.yaw_deg.max() - frame.yaw_deg.min()),
        "mean_abs_change": transition.mean_abs_change.mean(),
        "maximum_abs_change": transition.mean_abs_change.max(),
        "exact_duplicate_transitions": int(transition.exact_duplicate.sum()),
        "near_duplicate_transitions": int(transition.near_duplicate.sum()),
        "registration_success_rate": transition.registration_valid.mean(),
        "mean_rigid_translation": transition.rigid_translation.mean(),
        "mean_rigid_rotation_deg": transition.rigid_rotation_deg.abs().mean(),
        "mean_rigid_scale_change": transition.rigid_scale_change.mean(),
        "mean_nonrigid_flow": transition.nonrigid_flow.mean(),
        "mean_landmark_nonrigid": transition.landmark_nonrigid.mean(),
        "mean_aligned_pixel_change": transition.aligned_pixel_change.mean(),
        "mean_photometric_residual": transition.photometric_residual.mean(),
        "clip_fingerprint": "".join(frame.loc[frame.frame.isin([1, 8, 16]), "dhash"].tolist()),
    }
    return {"frames": frame_rows, "transitions": transition_rows, "clip": clip}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="artifacts/stimulus_audit")
    parser.add_argument("--with-landmarks", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--skip-full-length-index", action="store_true")
    parser.add_argument("--full-length-index", help="Optional precomputed original-frame-count table")
    args = parser.parse_args()
    config = read_config(args.config)
    paths = config["paths"]
    source_value = paths.get("frames_16_archive", paths["frames"])
    frame_source = FrameSource(resolve_path(config, source_value), paths.get("frames_archive_password"))
    split_rows = []
    for split, key in [("train", "train_csv"), ("test", "test_csv")]:
        source = pd.read_csv(resolve_path(config, paths[key]))
        split_rows.extend((int(row.video_name), split) for row in source.itertuples() if 1 <= int(row.label) <= 7)
    split_rows = split_rows[:args.limit] if args.limit else split_rows
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "audit_cache.sqlite"
    cache = sqlite3.connect(cache_path)
    cache.execute("PRAGMA journal_mode=WAL")
    cache.execute("CREATE TABLE IF NOT EXISTS clip_cache(clip_id INTEGER PRIMARY KEY, split TEXT NOT NULL, payload_json TEXT NOT NULL)")
    if args.rebuild:
        cache.execute("DELETE FROM clip_cache")
        cache.commit()
    detector = FaceAnalyzer(args.with_landmarks)
    try:
        for number, (clip_id, split) in enumerate(split_rows, start=1):
            if cache.execute("SELECT 1 FROM clip_cache WHERE clip_id=?", (clip_id,)).fetchone():
                continue
            result = audit_clip(frame_source, clip_id, split, detector)
            cache.execute(
                "INSERT OR REPLACE INTO clip_cache(clip_id, split, payload_json) VALUES (?,?,?)",
                (clip_id, split, json.dumps(result, allow_nan=True)),
            )
            cache.commit()
            if number % 100 == 0:
                print(f"Audited {number}/{len(split_rows)} clips")
    finally:
        detector.close()
        frame_source.close()
    records = [json.loads(row[0]) for row in cache.execute("SELECT payload_json FROM clip_cache ORDER BY clip_id")]
    cache.close()
    pd.DataFrame([row for record in records for row in record["frames"]]).to_csv(output / "frame_features.csv.gz", index=False, compression="gzip")
    pd.DataFrame([row for record in records for row in record["transitions"]]).to_csv(output / "transition_features.csv.gz", index=False, compression="gzip")
    clips = pd.DataFrame([record["clip"] for record in records])
    index_value = args.full_length_index or paths.get("full_length_index")
    if index_value and not args.skip_full_length_index:
        index_path = resolve_path(config, index_value)
        if not index_path.exists():
            raise FileNotFoundError(f"Run analysis.full_length_index first or use --skip-full-length-index: {index_path}")
        index = pd.read_csv(index_path)
        clips = clips.merge(index, on="clip_id", how="left")
    clips.to_csv(output / "stimulus_features.csv", index=False)
    print(f"Stimulus audit contains {len(clips)} clips in {output}")


if __name__ == "__main__":
    main()
