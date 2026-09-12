#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from app import create_app
from app.db import connect, init_database
from app.frame_source import FrameSource
from app.study import (
    EMOTIONS,
    label_clarity,
    load_split,
    needs_adjudication,
    prepare_study,
    select_training_alignment_subset,
    stable_uuid,
    third_annotator,
)
from app.trajectory import trajectory_descriptors


ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE = ROOT / "local_data" / "study.sqlite"


def load_config(path: str | Path) -> dict:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    config["_base"] = str(path.parent)
    return config


def resolve_config_path(config: dict, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = Path(config["_base"]) / path
    if candidate.exists():
        return candidate.resolve()
    return (ROOT / path).resolve()


def database_backup(database: Path, output_dir: Path = ROOT / "backups") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = output_dir / f"study_{stamp}.sqlite"
    with connect(database) as source, sqlite3.connect(destination) as target:
        source.backup(target)
    return destination


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compute_label_clarity(annotation_path: Path) -> pd.DataFrame:
    source = pd.read_excel(annotation_path)
    required = ["1happy", "2sad", "3neutral", "4angry", "5surprise", "6disgust", "7fear", "order", "label"]
    missing = set(required) - set(source.columns)
    if missing:
        raise ValueError(f"DFEW annotation table is missing {sorted(missing)}")
    votes = source[required[:7]].to_numpy(dtype=float)
    if (votes < 0).any() or not np.allclose(votes.sum(axis=1), 10):
        raise ValueError("Each DFEW annotation row must contain ten non-negative votes")
    proportions = votes / votes.sum(axis=1, keepdims=True)
    safe = np.where(proportions > 0, proportions, 1)
    ordered = np.sort(votes, axis=1)
    return pd.DataFrame({
        "clip_id": source["order"].astype(int), "dfew_label": source["label"].astype(int),
        "vote_max": votes.max(axis=1), "vote_margin": ordered[:, -1] - ordered[:, -2],
        "vote_entropy": -(proportions * np.log(safe)).sum(axis=1) / np.log(7),
        "neutral_votes": votes[:, 2],
        **{f"votes_{index + 1}": votes[:, index] for index in range(7)},
    })


def command_init(args) -> None:
    init_database(args.database)
    print(f"Initialized {Path(args.database).resolve()}")


def command_prepare(args) -> None:
    config = load_config(args.config)
    paths = config["paths"]
    study = config["study"]
    test = load_split(resolve_config_path(config, paths["test_csv"]))
    train = load_split(resolve_config_path(config, paths["train_csv"]))
    clarity = label_clarity(resolve_config_path(config, paths["annotation_xlsx"]))
    stimulus_path = paths.get("stimulus_features")
    stimulus = pd.read_csv(resolve_config_path(config, stimulus_path)) if stimulus_path else None
    if stimulus is not None:
        required = {"clip_id", "original_frame_count", "mean_abs_change"}
        missing_columns = required - set(stimulus.columns)
        if missing_columns:
            raise ValueError(f"Stimulus feature table is missing {sorted(missing_columns)}")
        required_ids = set(test.clip_id.astype(int)) | set(train.clip_id.astype(int))
        available_ids = set(stimulus.clip_id.astype(int))
        missing_ids = required_ids - available_ids
        if missing_ids:
            raise ValueError(
                f"Stimulus audit is incomplete: {len(missing_ids)} study clips are missing. "
                "Resume the local stimulus-feature audit before creating the master database."
            )
        duplicated = stimulus.clip_id.astype(int).duplicated().sum()
        if duplicated:
            raise ValueError(f"Stimulus feature table contains {duplicated} duplicate clip IDs")
    selection_train = train
    excluded_duplicate_ids: set[int] = set()
    duplicate_path = paths.get("duplicate_candidates")
    if duplicate_path:
        duplicate_file = resolve_config_path(config, duplicate_path)
        if duplicate_file.exists():
            duplicates = pd.read_csv(duplicate_file)
            if "review_tier" in duplicates:
                duplicates = duplicates[
                    duplicates.review_tier.isin(["very_high_similarity", "high_similarity"])
                ]
            else:
                # Fingerprint candidates are screening records until the full
                # 16-frame verification step assigns a review tier.
                duplicates = duplicates.iloc[0:0]
            for row in duplicates.itertuples():
                if getattr(row, "split_1", None) == "train":
                    excluded_duplicate_ids.add(int(row.clip_id_1))
                if getattr(row, "split_2", None) == "train":
                    excluded_duplicate_ids.add(int(row.clip_id_2))
            selection_train = train[~train.clip_id.isin(excluded_duplicate_ids)].copy()
    selected = select_training_alignment_subset(
        selection_train, clarity, stimulus,
        per_non_disgust=int(study.get("training_per_non_disgust_class", 380)),
    )
    target_per_class = int(study.get("training_per_non_disgust_class", 380))
    expected_by_label = {
        int(label): len(group) if int(label) == 6 else min(target_per_class, len(group))
        for label, group in selection_train.groupby("label")
    }
    observed_by_label = selected.groupby("label").size().astype(int).to_dict()
    if observed_by_label != expected_by_label:
        raise ValueError(
            f"Training alignment subset has unexpected class counts: {observed_by_label}; "
            f"expected {expected_by_label}"
        )
    source_value = paths.get("frames_16_archive", paths.get("frames"))
    if not source_value:
        raise ValueError("Configuration must define frames_16_archive or frames")
    frame_source = FrameSource(
        resolve_config_path(config, source_value), paths.get("frames_archive_password")
    )
    try:
        requested_ids = sorted(set(test.clip_id.astype(int)) | set(selected.clip_id.astype(int)))
        missing_frames = [clip_id for clip_id in requested_ids if not frame_source.has_clip(clip_id)]
        if missing_frames:
            preview = ", ".join(f"{value:05d}" for value in missing_frames[:10])
            raise ValueError(
                f"The configured 16-frame source is missing {len(missing_frames)} selected clips: {preview}"
            )
        if requested_ids:
            frame_source.read(requested_ids[0], 1)
    finally:
        frame_source.close()
    database = Path(args.database)
    if not database.exists():
        init_database(database)
    with connect(database) as connection:
        summary = prepare_study(
            connection,
            list(study.get("annotators", ["R01", "R02", "R03"])),
            test,
            selected,
            float(study.get("frame_second_rating_fraction", 0.10)),
            float(study.get("frame_hidden_repeat_fraction", 0.02)),
            int(study.get("frame_minimum_clip_gap", 50)),
            stimulus_features=stimulus,
        )
        connection.commit()
    output = Path(args.selection_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Training alignment selection: {output}")
    if excluded_duplicate_ids:
        print(
            f"Excluded {len(excluded_duplicate_ids)} high-similarity cross-split "
            "training candidates from alignment-subset selection"
        )


def command_run(args) -> None:
    config = load_config(args.config) if args.config else None
    if config:
        source_value = config["paths"].get("frames_16_archive", config["paths"]["frames"])
        frames = resolve_config_path(config, source_value)
        frame_password = config["paths"].get("frames_archive_password")
        server = config.get("server", {})
    else:
        frames = Path(args.frames or ROOT / "local_data" / "dfew" / "frames_16")
        frame_password = None
        server = {}
    host = args.host or server.get("host", "127.0.0.1")
    port = args.port or int(server.get("port", 5050))
    secret = server.get("secret_key", "local-study-change-me")
    app = create_app(args.database, frames, secret, frame_password)
    if not args.no_browser:
        import threading
        threading.Timer(0.8, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    app.run(host=host, port=port, debug=False)


def command_status(args) -> None:
    with connect(args.database) as connection:
        rows = connection.execute(
            """
            SELECT annotator_code, task_type, split, COUNT(*) total,
                   SUM(status='complete') complete, SUM(status='started') started
            FROM assignments JOIN tasks USING(task_uuid)
            GROUP BY annotator_code, task_type, split
            ORDER BY annotator_code, task_type, split
            """
        ).fetchall()
    print("annotator\ttask\tsplit\tcomplete\ttotal\tstarted")
    for row in rows:
        print(f"{row['annotator_code']}\t{row['task_type']}\t{row['split']}\t{row['complete']}\t{row['total']}\t{row['started']}")


def command_workload(args) -> None:
    with connect(args.database) as connection:
        frame = pd.DataFrame([dict(row) for row in connection.execute(
            """SELECT a.annotator_code, t.task_type, t.split, a.status, a.duration_ms
               FROM assignments a JOIN tasks t USING(task_uuid)"""
        )])
    completed = frame[(frame.status == "complete") & frame.duration_ms.notna() & (frame.duration_ms > 0)]
    medians = completed.groupby(["annotator_code", "task_type"]).duration_ms.median().to_dict()
    pooled = completed.groupby("task_type").duration_ms.median().to_dict()
    rows = []
    for (annotator, task_type, split), group in frame.groupby(["annotator_code", "task_type", "split"]):
        typical = medians.get((annotator, task_type), pooled.get(task_type, np.nan))
        remaining = int((group.status != "complete").sum())
        rows.append({
            "annotator": annotator, "task_type": task_type, "split": split,
            "total": len(group), "completed": int((group.status == "complete").sum()), "remaining": remaining,
            "median_completed_seconds": typical / 1000 if np.isfinite(typical) else np.nan,
            "estimated_remaining_hours": remaining * typical / 3_600_000 if np.isfinite(typical) else np.nan,
        })
    result = pd.DataFrame(rows)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Workload report: {output.resolve()}")


def command_backup(args) -> None:
    print(database_backup(Path(args.database), Path(args.output)))


def command_select_calibration(args) -> None:
    config = load_config(args.config)
    paths = config["paths"]
    study = config["study"]
    train = load_split(resolve_config_path(config, paths["train_csv"]))
    test = load_split(resolve_config_path(config, paths["test_csv"]))
    candidates = compute_label_clarity(resolve_config_path(config, paths["annotation_xlsx"]))
    stimulus_path = paths.get("stimulus_features")
    stimulus = pd.read_csv(resolve_config_path(config, stimulus_path)) if stimulus_path else None
    alignment = select_training_alignment_subset(
        train,
        candidates,
        stimulus,
        per_non_disgust=int(study.get("training_per_non_disgust_class", 380)),
    )
    alignment_ids = set(alignment.clip_id.astype(int))
    test_ids = set(test.clip_id.astype(int))
    candidates = candidates[~candidates.clip_id.isin(test_ids)].copy()
    candidates["main_study_overlap"] = candidates.clip_id.isin(alignment_ids)
    source_value = paths.get("frames_16_archive", paths.get("frames"))
    source = FrameSource(resolve_config_path(config, source_value), paths.get("frames_archive_password"))
    candidates = candidates[candidates.clip_id.map(lambda value: source.has_clip(int(value)))]
    source.close()
    selected = []
    frame_positions = [2, 6, 11, 15]
    for label in range(1, 8):
        class_candidates = candidates[candidates.dfew_label == label]
        outside = class_candidates[~class_candidates.main_study_overlap]
        group = (outside if len(outside) >= args.per_class else class_candidates).sort_values(
            ["vote_entropy", "vote_margin", "clip_id"], ascending=[True, False, True]
        )
        if len(group) < args.per_class:
            raise ValueError(f"Only {len(group)} eligible calibration clips for class {label}")
        clear_n = (args.per_class + 1) // 2
        clear = group.head(clear_n)
        boundary = group.tail(args.per_class - clear_n)
        chosen = pd.concat([clear, boundary]).drop_duplicates("clip_id")
        if len(chosen) != args.per_class:
            raise ValueError(f"Could not select {args.per_class} unique clips for class {label}")
        chosen = chosen.assign(
            calibration_profile=["clear"] * len(clear) + ["category_boundary"] * len(boundary)
        )
        selected.append(chosen)
    key = pd.concat(selected, ignore_index=True).sort_values(
        ["dfew_label", "calibration_profile", "clip_id"]
    ).reset_index(drop=True)
    task_rows = []
    for within_label, row in key.groupby("dfew_label", sort=True):
        for index, item in enumerate(row.itertuples()):
            task_rows.append({
                "clip_id": int(item.clip_id), "task_type": "frame",
                "frame_index": frame_positions[index % len(frame_positions)],
            })
    task_rows.extend({
        "clip_id": int(item.clip_id), "task_type": "clip", "frame_index": ""
    } for item in key.itertuples())
    manifest = pd.DataFrame(task_rows)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    key_output = Path(args.key_output); key_output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    key.assign(reference_status="discussion_anchor_not_gold").to_csv(key_output, index=False)
    print(
        f"Selected {len(key)} calibration clips ({int(key.main_study_overlap.sum())} also occur in the "
        "human-rated training subset); "
        f"manifest contains {len(manifest)} tasks per annotator"
    )
    print(f"Blind manifest: {output.resolve()}")
    print(f"Coordinator-only key: {key_output.resolve()}")


def command_calibration(args) -> None:
    manifest = pd.read_csv(args.manifest)
    required = {"clip_id", "task_type"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Calibration manifest requires columns {sorted(required)}")
    if not set(manifest.task_type).issubset({"frame", "clip"}):
        raise ValueError("Calibration task_type must be frame or clip")
    if (manifest.task_type == "frame").any() and "frame_index" not in manifest:
        raise ValueError("Frame calibration tasks require frame_index")
    with connect(args.database) as connection:
        codes = [row["code"] for row in connection.execute("SELECT code FROM annotators WHERE active=1 ORDER BY code")]
        if len(codes) != 3:
            raise ValueError("Calibration preparation expects three active annotators")
        if connection.execute("SELECT 1 FROM tasks WHERE split='calibration' LIMIT 1").fetchone():
            raise ValueError("Calibration tasks already exist; use a fresh master database to change them")
        for code in codes:
            connection.execute("UPDATE assignments SET queue_position=queue_position+? WHERE annotator_code=?", (len(manifest), code))
        for position, row in enumerate(manifest.itertuples(), start=1):
            clip_id = int(row.clip_id)
            frame_index = int(row.frame_index) if row.task_type == "frame" else None
            if frame_index is not None and not 1 <= frame_index <= 16:
                raise ValueError(f"Invalid calibration frame {clip_id:05d}/{frame_index}")
            task_uuid = stable_uuid("calibration", row.task_type, clip_id, frame_index)
            connection.execute(
                "INSERT OR IGNORE INTO tasks(task_uuid, task_type, split, clip_id, frame_index) VALUES (?,?,?,?,?)",
                (task_uuid, row.task_type, "calibration", clip_id, frame_index),
            )
            for code in codes:
                assignment_uuid = stable_uuid("assignment", task_uuid, code, "calibration")
                connection.execute(
                    "INSERT OR IGNORE INTO assignments(assignment_uuid, task_uuid, annotator_code, role, queue_position) VALUES (?,?,?,?,?)",
                    (assignment_uuid, task_uuid, code, "primary", position),
                )
        connection.execute("INSERT OR REPLACE INTO study_meta(key, value) VALUES ('calibration_tasks', ?)", (str(len(manifest)),))
        connection.commit()
    print(f"Added {len(manifest)} calibration tasks for each annotator at the start of the queue")


def command_calibration_report(args) -> None:
    rows = []
    with connect(args.database) as connection:
        tasks = connection.execute("SELECT * FROM tasks WHERE split='calibration' ORDER BY task_type, clip_id, frame_index").fetchall()
        for task in tasks:
            if task["task_type"] == "frame":
                ratings = connection.execute(
                    """SELECT a.annotator_code, r.visible_category category, r.intensity
                       FROM assignments a JOIN frame_ratings r USING(assignment_uuid) WHERE a.task_uuid=?""",
                    (task["task_uuid"],),
                ).fetchall()
                curves = None
            else:
                ratings = connection.execute(
                    """SELECT a.annotator_code, r.dominant_category category, r.intensities_json
                       FROM assignments a JOIN clip_ratings r USING(assignment_uuid) WHERE a.task_uuid=?""",
                    (task["task_uuid"],),
                ).fetchall()
                curves = [np.asarray(json.loads(row["intensities_json"]), dtype=float) for row in ratings]
            categories = [row["category"] for row in ratings]
            row = {
                "task_type": task["task_type"], "clip_id": task["clip_id"],
                "frame": task["frame_index"] or "", "n_completed": len(ratings),
                "category_unanimous": int(len(categories) == 3 and len(set(categories)) == 1),
                "categories": "|".join(categories),
            }
            if task["task_type"] == "frame" and len(ratings) >= 2:
                values = np.asarray([item["intensity"] for item in ratings], dtype=float)
                row["mean_pairwise_intensity_gap"] = float(np.mean([abs(a-b) for i, a in enumerate(values) for b in values[i+1:]]))
            elif curves and len(curves) >= 2:
                row["mean_pairwise_intensity_gap"] = float(np.mean([
                    np.mean(np.abs(a-b)) for i, a in enumerate(curves) for b in curves[i+1:]
                ]))
            else:
                row["mean_pairwise_intensity_gap"] = ""
            rows.append(row)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    completed = [row for row in rows if row["n_completed"] == 3]
    if completed:
        unanimous = np.mean([row["category_unanimous"] for row in completed])
        gap = np.mean([row["mean_pairwise_intensity_gap"] for row in completed])
        print(f"Completed calibration tasks: {len(completed)}; category unanimity: {unanimous:.1%}; mean pairwise intensity gap: {gap:.2f}")
    print(f"Calibration report: {output.resolve()}")


def command_make_bundle(args) -> None:
    source = Path(args.database)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    bundle_database = destination / "study.sqlite"
    with connect(source) as master, sqlite3.connect(bundle_database) as copy:
        master.backup(copy)
    master.close()
    copy.close()
    with connect(bundle_database) as connection:
        if not connection.execute("SELECT 1 FROM annotators WHERE code=?", (args.annotator,)).fetchone():
            raise ValueError(f"Unknown annotator {args.annotator}")
        other_assignments = [row[0] for row in connection.execute("SELECT assignment_uuid FROM assignments WHERE annotator_code!=?", (args.annotator,))]
        if other_assignments:
            marks = ",".join("?" for _ in other_assignments)
            connection.execute(f"DELETE FROM drafts WHERE assignment_uuid IN ({marks})", other_assignments)
            connection.execute(f"DELETE FROM frame_ratings WHERE assignment_uuid IN ({marks})", other_assignments)
            connection.execute(f"DELETE FROM clip_ratings WHERE assignment_uuid IN ({marks})", other_assignments)
            connection.execute(f"DELETE FROM assignments WHERE assignment_uuid IN ({marks})", other_assignments)
        connection.execute("DELETE FROM annotators WHERE code!=?", (args.annotator,))
        connection.execute("DELETE FROM import_log")
        connection.execute("DELETE FROM adjudication_cases")
        connection.execute("INSERT OR REPLACE INTO study_meta(key, value) VALUES ('bundle_annotator', ?)", (args.annotator,))
        connection.commit()
        task_count = connection.execute("SELECT COUNT(*) FROM assignments").fetchone()[0]
    connection.close()
    portable = sqlite3.connect(bundle_database)
    try:
        portable.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        portable.execute("PRAGMA journal_mode=DELETE")
    finally:
        portable.close()
    for suffix in ("-wal", "-shm"):
        bundle_database.with_name(bundle_database.name + suffix).unlink(missing_ok=True)
    instructions = (
        f"Annotator: {args.annotator}\n\n"
        "1. Place this study.sqlite file in trajectory_sampling_study/local_data/.\n"
        "2. Place the authorized DFEW 16-frame archive at the path configured locally.\n"
        "3. Start the app and sign in using the annotator code above.\n"
        f"4. Export results with: python manage.py export --annotator {args.annotator} --output exports/{args.annotator}\n"
        "5. Return the exported folder to the study coordinator.\n"
    )
    (destination / "ANNOTATOR_README.txt").write_text(instructions, encoding="utf-8")
    print(f"Private bundle for {args.annotator}: {destination.resolve()} ({task_count} assignments)")


def command_export(args) -> None:
    database = Path(args.database)
    output = Path(args.output)
    backup = database_backup(database)
    output.mkdir(parents=True, exist_ok=True)
    with connect(database) as connection:
        annotator = connection.execute("SELECT code FROM annotators WHERE code=?", (args.annotator,)).fetchone()
        if not annotator:
            raise SystemExit(f"Unknown annotator: {args.annotator}")
        assignment_rows = [dict(row) for row in connection.execute(
            "SELECT * FROM assignments WHERE annotator_code=? AND status='complete'", (args.annotator,)
        )]
        ids = [row["assignment_uuid"] for row in assignment_rows]
        frame_rows, clip_rows = [], []
        if ids:
            marks = ",".join("?" for _ in ids)
            frame_rows = [dict(row) for row in connection.execute(f"SELECT * FROM frame_ratings WHERE assignment_uuid IN ({marks})", ids)]
            clip_rows = [dict(row) for row in connection.execute(f"SELECT * FROM clip_ratings WHERE assignment_uuid IN ({marks})", ids)]
        design_version = connection.execute("SELECT value FROM study_meta WHERE key='design_version'").fetchone()
    write_csv(output / "assignments.csv", assignment_rows, [
        "assignment_uuid", "task_uuid", "annotator_code", "role", "queue_position", "status", "started_at", "completed_at", "duration_ms"
    ])
    write_csv(output / "frame_ratings.csv", frame_rows, ["assignment_uuid", "visible_category", "intensity", "submitted_at"])
    write_csv(output / "clip_ratings.csv", clip_rows, [
        "assignment_uuid", "dominant_category", "intensities_json", "occlusion", "speaking", "abrupt_change", "subject_switch", "submitted_at"
    ])
    payload = {
        "bundle_id": hashlib.sha256((args.annotator + "|" + "|".join(sorted(ids))).encode()).hexdigest()[:20],
        "annotator": args.annotator,
        "design_version": design_version["value"] if design_version else "unknown",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "completed_assignments": len(ids),
    }
    (output / "bundle.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Exported {len(ids)} completed assignments to {output.resolve()}")
    print(f"Backup created: {backup}")


def command_merge(args) -> None:
    with connect(args.database) as connection:
        expected_version = connection.execute("SELECT value FROM study_meta WHERE key='design_version'").fetchone()
        expected_version = expected_version["value"] if expected_version else None
        total = 0
        for folder_name in args.folders:
            folder = Path(folder_name)
            bundle = json.loads((folder / "bundle.json").read_text(encoding="utf-8"))
            if expected_version and bundle.get("design_version") != expected_version:
                raise ValueError(f"Design version mismatch in {folder}")
            if connection.execute("SELECT 1 FROM import_log WHERE bundle_id=?", (bundle["bundle_id"],)).fetchone():
                print(f"Already imported: {folder}")
                continue
            assignments = pd.read_csv(folder / "assignments.csv").fillna("")
            frame = pd.read_csv(folder / "frame_ratings.csv").fillna("")
            clip = pd.read_csv(folder / "clip_ratings.csv").fillna("")
            for row in assignments.to_dict("records"):
                existing = connection.execute(
                    "SELECT annotator_code, task_uuid FROM assignments WHERE assignment_uuid=?", (row["assignment_uuid"],)
                ).fetchone()
                if not existing or existing["annotator_code"] != row["annotator_code"] or existing["task_uuid"] != row["task_uuid"]:
                    raise ValueError(f"Unknown or mismatched assignment {row['assignment_uuid']}")
                connection.execute(
                    "UPDATE assignments SET status='complete', started_at=?, completed_at=?, duration_ms=? WHERE assignment_uuid=?",
                    (row["started_at"] or None, row["completed_at"] or None, int(row["duration_ms"]) if str(row["duration_ms"]) else None, row["assignment_uuid"]),
                )
            for row in frame.to_dict("records"):
                connection.execute(
                    "INSERT OR REPLACE INTO frame_ratings(assignment_uuid, visible_category, intensity, submitted_at) VALUES (?,?,?,?)",
                    (row["assignment_uuid"], row["visible_category"], int(row["intensity"]), row["submitted_at"] or datetime.now().isoformat()),
                )
            for row in clip.to_dict("records"):
                connection.execute(
                    """INSERT OR REPLACE INTO clip_ratings(
                       assignment_uuid, dominant_category, intensities_json, occlusion, speaking, abrupt_change, subject_switch, submitted_at
                       ) VALUES (?,?,?,?,?,?,?,?)""",
                    (row["assignment_uuid"], row["dominant_category"], row["intensities_json"],
                     int(row["occlusion"]), int(row["speaking"]), int(row["abrupt_change"]), int(row["subject_switch"]),
                     row["submitted_at"] or datetime.now().isoformat()),
                )
            imported = len(assignments)
            connection.execute(
                "INSERT INTO import_log(bundle_id, source_path, rows_imported) VALUES (?,?,?)",
                (bundle["bundle_id"], str(folder.resolve()), imported),
            )
            total += imported
        connection.commit()
    print(f"Imported {total} completed assignments")


def command_adjudications(args) -> None:
    created = 0
    with connect(args.database) as connection:
        codes = [row["code"] for row in connection.execute("SELECT code FROM annotators WHERE active=1 ORDER BY code")]
        tasks = connection.execute("SELECT task_uuid FROM tasks WHERE task_type='clip' AND split!='calibration'").fetchall()
        for task in tasks:
            ratings = connection.execute(
                """
                SELECT a.annotator_code, r.* FROM assignments a JOIN clip_ratings r USING(assignment_uuid)
                WHERE a.task_uuid=? AND a.role!='adjudication' ORDER BY a.annotator_code
                """,
                (task["task_uuid"],),
            ).fetchall()
            needed, reasons = needs_adjudication(ratings)
            if not needed:
                continue
            existing_codes = [row["annotator_code"] for row in ratings]
            code = third_annotator(codes, existing_codes)
            assignment_uuid = stable_uuid("assignment", task["task_uuid"], code, "adjudication")
            queue = connection.execute("SELECT COALESCE(MAX(queue_position),0)+1 FROM assignments WHERE annotator_code=?", (code,)).fetchone()[0]
            cursor = connection.execute(
                "INSERT OR IGNORE INTO assignments(assignment_uuid, task_uuid, annotator_code, role, queue_position) VALUES (?,?,?,?,?)",
                (assignment_uuid, task["task_uuid"], code, "adjudication", queue),
            )
            connection.execute(
                "INSERT OR REPLACE INTO adjudication_cases(task_uuid, reasons_json, status) VALUES (?,?,COALESCE((SELECT status FROM adjudication_cases WHERE task_uuid=?),'pending'))",
                (task["task_uuid"], json.dumps(reasons), task["task_uuid"]),
            )
            created += int(cursor.rowcount == 1)
        connection.commit()
    print(f"Adjudication scan complete; {created} cases created or updated")


def consensus_category(categories: list[str]) -> str:
    if len(categories) == 1:
        return str(categories[0])
    counts = pd.Series(categories).value_counts()
    if len(counts) and counts.iloc[0] >= 2:
        return str(counts.index[0])
    return "No consensus"


def command_consensus(args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    consensus_output, frame_consensus_output, adjudication_output = [], [], []
    with connect(args.database) as connection:
        raw_frames = pd.DataFrame([dict(row) for row in connection.execute(
            """SELECT t.task_uuid, t.source_task_uuid, a.annotator_code, a.role, a.duration_ms,
                      t.clip_id, t.frame_index AS frame, t.repeat_kind,
                      r.visible_category, r.intensity, r.submitted_at
               FROM assignments a JOIN tasks t USING(task_uuid) JOIN frame_ratings r USING(assignment_uuid)
               ORDER BY t.clip_id, t.frame_index, a.annotator_code"""
        )])
        raw_clips = pd.DataFrame([dict(row) for row in connection.execute(
            """SELECT a.annotator_code, a.role, a.duration_ms, t.clip_id, t.split,
                      r.dominant_category, r.intensities_json, r.occlusion, r.speaking,
                      r.abrupt_change, r.subject_switch, r.submitted_at
               FROM assignments a JOIN tasks t USING(task_uuid) JOIN clip_ratings r USING(assignment_uuid)
               ORDER BY t.split, t.clip_id, a.annotator_code"""
        )])
        if not raw_clips.empty:
            curves = raw_clips.intensities_json.map(json.loads)
            for index in range(16):
                raw_clips[f"intensity_{index + 1}"] = curves.map(lambda values: values[index])
        clip_tasks = connection.execute("SELECT * FROM tasks WHERE task_type='clip' ORDER BY split, clip_id").fetchall()
        for task in clip_tasks:
            ratings = connection.execute(
                """SELECT a.annotator_code, a.role, r.* FROM assignments a JOIN clip_ratings r USING(assignment_uuid)
                   WHERE a.task_uuid=? ORDER BY (a.role='adjudication'), a.annotator_code""", (task["task_uuid"],)
            ).fetchall()
            base = [row for row in ratings if row["role"] != "adjudication"]
            needed, reasons = needs_adjudication(base)
            adj = [row for row in ratings if row["role"] == "adjudication"]
            if len(base) < 2 or (needed and not adj):
                continue
            used = base + adj[:1]
            curves = np.vstack([np.asarray(json.loads(row["intensities_json"]), dtype=float) for row in used])
            curve = np.median(curves, axis=0) if len(used) == 3 else curves.mean(axis=0)
            category = consensus_category([row["dominant_category"] for row in used])
            row = {
                "clip_id": task["clip_id"], "split": task["split"],
                "n_ratings": len(used), "intensities_json": json.dumps(curve.round(6).tolist()),
                "dominant_category": category, **trajectory_descriptors(curve, category),
            }
            for index, value in enumerate(curve, start=1):
                row[f"intensity_{index}"] = float(value)
            consensus_output.append(row)
            if needed:
                third = adj[0]
                adjudication_output.append({
                    "clip_id": task["clip_id"], "split": task["split"], "reasons": "|".join(reasons),
                    "third_annotator": third["annotator_code"], "third_category": third["dominant_category"],
                    "third_intensities_json": third["intensities_json"],
                    "final_category": category, "final_intensities_json": row["intensities_json"], "status": "complete",
                })

        reliability_output = []
        if not raw_frames.empty:
            primary = raw_frames[raw_frames.repeat_kind == "none"].copy()
            if primary.duplicated(["clip_id", "frame"]).any():
                raise ValueError("A test frame has more than one primary isolated-frame rating")
            frame_consensus_output = primary[[
                "clip_id", "frame", "annotator_code", "visible_category", "intensity"
            ]].to_dict("records")
            primary_by_task = primary.set_index("task_uuid")
            repeats = raw_frames[raw_frames.repeat_kind.isin(["second_rater", "within_rater"])]
            for repeat in repeats.itertuples():
                if repeat.source_task_uuid not in primary_by_task.index:
                    continue
                base = primary_by_task.loc[repeat.source_task_uuid]
                reliability_output.append({
                    "clip_id": int(repeat.clip_id), "frame": int(repeat.frame),
                    "repeat_kind": repeat.repeat_kind,
                    "primary_annotator": base.annotator_code,
                    "repeat_annotator": repeat.annotator_code,
                    "primary_category": base.visible_category,
                    "repeat_category": repeat.visible_category,
                    "category_agreement": int(base.visible_category == repeat.visible_category),
                    "primary_intensity": float(base.intensity),
                    "repeat_intensity": float(repeat.intensity),
                    "absolute_intensity_gap": float(abs(base.intensity - repeat.intensity)),
                })

    clip_columns = ["clip_id", "split", "n_ratings", "dominant_category", "intensities_json"] + [f"intensity_{i}" for i in range(1, 17)] + [
        "start_intensity", "end_intensity", "start_end_difference", "mean_intensity",
        "onset_position", "apex_position", "offset_position", "peak_width", "peak_count", "peak_type",
        "expression_present_fraction", "high_intensity_fraction", "trajectory_total_variation",
        "mean_absolute_adjacent_change", "adjacent_differences_json"
    ]
    raw_frames.to_csv(output / "frame_ratings.csv.gz", index=False, compression="gzip")
    raw_clips.to_csv(output / "clip_ratings.csv.gz", index=False, compression="gzip")
    write_csv(output / "trajectory_consensus.csv", consensus_output, clip_columns)
    write_csv(output / "frame_primary.csv", frame_consensus_output, [
        "clip_id", "frame", "annotator_code", "visible_category", "intensity"
    ])
    write_csv(output / "frame_reliability_pairs.csv", reliability_output, [
        "clip_id", "frame", "repeat_kind", "primary_annotator", "repeat_annotator",
        "primary_category", "repeat_category", "category_agreement",
        "primary_intensity", "repeat_intensity", "absolute_intensity_gap",
    ])
    write_csv(output / "adjudications.csv", adjudication_output, [
        "clip_id", "split", "reasons", "third_annotator", "third_category", "third_intensities_json",
        "final_category", "final_intensities_json", "status",
    ])
    print(
        f"Consensus clips: {len(consensus_output)}; primary frames: {len(frame_consensus_output)}; "
        f"reliability pairs: {len(reliability_output)}"
    )


def command_validate(args) -> None:
    errors = []
    with connect(args.database) as connection:
        meta = {row["key"]: row["value"] for row in connection.execute("SELECT * FROM study_meta")}
        test_clips = int(meta.get("test_clips", 0))
        base_frames = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_type='frame' AND split='test' AND repeat_kind='none'"
        ).fetchone()[0]
        if base_frames != test_clips * 16:
            errors.append(f"Expected {test_clips * 16} base frame tasks, found {base_frames}")
        for code in [row["code"] for row in connection.execute("SELECT code FROM annotators")]:
            clip_positions = [row[0] for row in connection.execute(
                """SELECT t.clip_id FROM assignments a JOIN tasks t USING(task_uuid)
                   WHERE a.annotator_code=? AND t.task_type='frame' AND t.split='test'
                   ORDER BY a.queue_position""", (code,)
            )]
            last = {}
            minimum_gap = int(meta.get("minimum_clip_gap", 50))
            for position, clip_id in enumerate(clip_positions):
                if clip_id in last and position - last[clip_id] <= minimum_gap:
                    errors.append(f"{code}: clip {clip_id} repeats within {minimum_gap} tasks")
                    break
                last[clip_id] = position
        invalid_curves = connection.execute(
            "SELECT assignment_uuid, intensities_json FROM clip_ratings"
        ).fetchall()
        for row in invalid_curves:
            try:
                values = json.loads(row["intensities_json"])
                if len(values) != 16 or any(int(value) not in range(7) for value in values):
                    raise ValueError
            except Exception:
                errors.append(f"Invalid trajectory: {row['assignment_uuid']}")
        if args.require_complete:
            incomplete = connection.execute("SELECT COUNT(*) FROM assignments WHERE status!='complete'").fetchone()[0]
            if incomplete:
                errors.append(f"{incomplete} assignments are not complete")
            clip_tasks = connection.execute("SELECT task_uuid FROM tasks WHERE task_type='clip' AND split!='calibration'").fetchall()
            unresolved = 0
            for task in clip_tasks:
                ratings = connection.execute(
                    """SELECT a.annotator_code, a.role, r.* FROM assignments a JOIN clip_ratings r USING(assignment_uuid)
                       WHERE a.task_uuid=? ORDER BY a.annotator_code""", (task["task_uuid"],)
                ).fetchall()
                base = [row for row in ratings if row["role"] != "adjudication"]
                needed, _ = needs_adjudication(base)
                if needed and not any(row["role"] == "adjudication" for row in ratings):
                    unresolved += 1
            if unresolved:
                errors.append(f"{unresolved} disagreement cases lack a completed third rating")
    if errors:
        print("Validation failed:")
        for error in errors[:30]:
            print(f"- {error}")
        raise SystemExit(1)
    print("Study database validation passed")


def command_demo(_args) -> None:
    frames_root = ROOT / "demo_data" / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for clip_id in range(1, 22):
        label = (clip_id - 1) % 7 + 1
        rows.append((clip_id, label))
        folder = frames_root / f"{clip_id:05d}"
        folder.mkdir(exist_ok=True)
        for frame in range(1, 17):
            image = Image.new("RGB", (224, 224), (236, 242, 247))
            draw = ImageDraw.Draw(image)
            strength = frame / 16 if clip_id % 2 else (17 - frame) / 16
            draw.ellipse((42, 24, 182, 190), fill=(250, 209, 173), outline=(32, 53, 80), width=4)
            draw.ellipse((82, 76, 94, 88), fill=(32, 53, 80)); draw.ellipse((130, 76, 142, 88), fill=(32, 53, 80))
            y = int(135 + (strength - .5) * (16 if label in [1, 5] else -12))
            draw.arc((80, 110, 145, y + 28), 5 if label in [1, 5] else 185, 175 if label in [1, 5] else 355, fill=(239, 109, 97), width=5)
            draw.text((8, 8), f"DEMO {clip_id:02d} · {frame:02d}", fill=(78, 94, 228))
            image.save(folder / f"{frame}.jpg", quality=88)
    test = pd.DataFrame(rows[:7], columns=["video_name", "label"])
    train = pd.DataFrame(rows[7:], columns=["video_name", "label"])
    test.to_csv(ROOT / "demo_data" / "test.csv", index=False)
    train.to_csv(ROOT / "demo_data" / "train.csv", index=False)
    votes = []
    for clip_id, label in rows:
        values = [0] * 7; values[label - 1] = 8; values[label % 7] = 2
        votes.append([*values, clip_id, label])
    annotation_path = ROOT / "demo_data" / "annotation.xlsx"
    # Keep the tracked synthetic fixture stable across repeated launcher runs.
    # Delete it explicitly when a fresh workbook is desired.
    if not annotation_path.exists():
        pd.DataFrame(votes, columns=["1happy","2sad","3neutral","4angry","5surprise","6disgust","7fear","order","label"]).to_excel(annotation_path, index=False)
    config = """[paths]
frames = "demo_data/frames"
annotation_xlsx = "demo_data/annotation.xlsx"
train_csv = "demo_data/train.csv"
test_csv = "demo_data/test.csv"

[study]
annotators = ["R01", "R02", "R03"]
frame_second_rating_fraction = 0.10
frame_hidden_repeat_fraction = 0.02
frame_minimum_clip_gap = 2
training_per_non_disgust_class = 380

[server]
host = "127.0.0.1"
port = 5050
secret_key = "demo-only-secret"
"""
    (ROOT / "demo_data" / "project.toml").write_text(config, encoding="utf-8")
    print(f"Synthetic demo created in {(ROOT / 'demo_data').resolve()}")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Local DFEW trajectory annotation manager")
    root.set_defaults(func=None)
    sub = root.add_subparsers(dest="command")
    init = sub.add_parser("init-db"); init.add_argument("--database", default=DEFAULT_DATABASE); init.set_defaults(func=command_init)
    prepare = sub.add_parser("prepare-study"); prepare.add_argument("--config", required=True); prepare.add_argument("--database", default=DEFAULT_DATABASE); prepare.add_argument("--selection-output", default=ROOT / "artifacts" / "training_alignment_selection.csv"); prepare.set_defaults(func=command_prepare)
    calibration = sub.add_parser("prepare-calibration"); calibration.add_argument("--manifest", required=True); calibration.add_argument("--database", default=DEFAULT_DATABASE); calibration.set_defaults(func=command_calibration)
    select_calibration = sub.add_parser("select-calibration"); select_calibration.add_argument("--config", required=True); select_calibration.add_argument("--per-class", type=int, default=4); select_calibration.add_argument("--output", default=ROOT / "local_data" / "calibration_manifest.csv"); select_calibration.add_argument("--key-output", default=ROOT / "local_data" / "calibration_key.csv"); select_calibration.set_defaults(func=command_select_calibration)
    calibration_report = sub.add_parser("calibration-report"); calibration_report.add_argument("--database", default=DEFAULT_DATABASE); calibration_report.add_argument("--output", default=ROOT / "exports" / "calibration_report.csv"); calibration_report.set_defaults(func=command_calibration_report)
    run = sub.add_parser("run"); run.add_argument("--config"); run.add_argument("--database", default=DEFAULT_DATABASE); run.add_argument("--frames"); run.add_argument("--host"); run.add_argument("--port", type=int); run.add_argument("--no-browser", action="store_true"); run.set_defaults(func=command_run)
    status = sub.add_parser("status"); status.add_argument("--database", default=DEFAULT_DATABASE); status.set_defaults(func=command_status)
    workload = sub.add_parser("workload-report"); workload.add_argument("--database", default=DEFAULT_DATABASE); workload.add_argument("--output", default=ROOT / "exports" / "workload_report.csv"); workload.set_defaults(func=command_workload)
    backup = sub.add_parser("backup"); backup.add_argument("--database", default=DEFAULT_DATABASE); backup.add_argument("--output", default=ROOT / "backups"); backup.set_defaults(func=command_backup)
    export = sub.add_parser("export"); export.add_argument("--database", default=DEFAULT_DATABASE); export.add_argument("--annotator", required=True); export.add_argument("--output", required=True); export.set_defaults(func=command_export)
    bundle = sub.add_parser("make-annotator-bundle"); bundle.add_argument("--database", default=DEFAULT_DATABASE); bundle.add_argument("--annotator", required=True); bundle.add_argument("--output", required=True); bundle.set_defaults(func=command_make_bundle)
    merge = sub.add_parser("merge"); merge.add_argument("folders", nargs="+"); merge.add_argument("--database", default=DEFAULT_DATABASE); merge.set_defaults(func=command_merge)
    adjudicate = sub.add_parser("create-adjudications"); adjudicate.add_argument("--database", default=DEFAULT_DATABASE); adjudicate.set_defaults(func=command_adjudications)
    consensus = sub.add_parser("export-consensus"); consensus.add_argument("--database", default=DEFAULT_DATABASE); consensus.add_argument("--output", required=True); consensus.set_defaults(func=command_consensus)
    validate = sub.add_parser("validate-study"); validate.add_argument("--database", default=DEFAULT_DATABASE); validate.add_argument("--require-complete", action="store_true"); validate.set_defaults(func=command_validate)
    demo = sub.add_parser("make-demo"); demo.set_defaults(func=command_demo)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    if args.func is None:
        parser().print_help()
        raise SystemExit(2)
    args.func(args)
