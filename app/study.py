from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import random
import sqlite3
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from .trajectory import trajectory_descriptors


EMOTIONS = ["Happiness", "Sadness", "Neutral", "Anger", "Surprise", "Disgust", "Fear"]
LABEL_TO_EMOTION = {index + 1: emotion for index, emotion in enumerate(EMOTIONS)}
UUID_NAMESPACE = uuid.UUID("3350bc80-5218-4d89-ae31-87ee4bdcc1c5")


def stable_uuid(kind: str, *parts: object) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, ":".join([kind, *map(str, parts)])))


def load_split(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"video_name", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    frame = frame[["video_name", "label"]].rename(columns={"video_name": "clip_id"})
    frame["clip_id"] = frame["clip_id"].astype(int)
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].between(1, 7)].drop_duplicates("clip_id")
    frame["emotion"] = frame["label"].map(LABEL_TO_EMOTION)
    return frame.sort_values("clip_id").reset_index(drop=True)


def label_clarity(annotation_xlsx: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(annotation_xlsx)
    vote_columns = list(frame.columns[:7])
    votes = frame[vote_columns].to_numpy(dtype=float)
    totals = votes.sum(axis=1, keepdims=True)
    proportions = np.divide(votes, totals, out=np.zeros_like(votes), where=totals > 0)
    safe = np.where(proportions > 0, proportions, 1)
    ordered = np.sort(votes, axis=1)
    return pd.DataFrame({
        "clip_id": frame["order"].astype(int),
        "vote_max": votes.max(axis=1),
        "vote_margin": ordered[:, -1] - ordered[:, -2],
        "vote_entropy": -(proportions * np.log(safe)).sum(axis=1) / np.log(7),
        "neutral_votes": votes[:, 2],
    })


def select_training_alignment_subset(
    train: pd.DataFrame,
    clarity: pd.DataFrame,
    stimulus_features: pd.DataFrame | None,
    per_non_disgust: int = 380,
) -> pd.DataFrame:
    merged = train.merge(clarity, on="clip_id", how="left")
    if stimulus_features is not None:
        available = [column for column in ["clip_id", "original_frame_count", "mean_abs_change"] if column in stimulus_features]
        merged = merged.merge(stimulus_features[available].drop_duplicates("clip_id"), on="clip_id", how="left")

    strata_columns = []
    for source, target in [
        ("vote_entropy", "clarity_bin"),
        ("original_frame_count", "length_bin"),
        ("mean_abs_change", "motion_bin"),
    ]:
        if source in merged and merged[source].notna().sum() >= 4:
            merged[target] = pd.qcut(merged[source].rank(method="first"), 4, labels=False, duplicates="drop")
            strata_columns.append(target)

    selected = []
    rng = np.random.default_rng(20260912)
    for label, group in merged.groupby("label", sort=True):
        if label == 6:
            selected.append(group)
            continue
        target_n = min(per_non_disgust, len(group))
        if not strata_columns:
            chosen = group.iloc[rng.choice(len(group), size=target_n, replace=False)]
        else:
            stratum = group[strata_columns].fillna(-1).astype(str).agg("|".join, axis=1)
            group = group.assign(_stratum=stratum)
            pieces = []
            allocations = group["_stratum"].value_counts(normalize=True).mul(target_n)
            floor = allocations.apply(math.floor).astype(int)
            remainder = target_n - floor.sum()
            for key in (allocations - floor).sort_values(ascending=False).index[:remainder]:
                floor.loc[key] += 1
            for key, count in floor.items():
                pool = group[group["_stratum"] == key]
                count = min(int(count), len(pool))
                if count:
                    pieces.append(pool.iloc[rng.choice(len(pool), size=count, replace=False)])
            chosen = pd.concat(pieces, ignore_index=True) if pieces else group.iloc[0:0]
            if len(chosen) < target_n:
                remaining = group[~group.clip_id.isin(chosen.clip_id)]
                extra_n = min(target_n - len(chosen), len(remaining))
                chosen = pd.concat([chosen, remaining.iloc[rng.choice(len(remaining), size=extra_n, replace=False)]])
        selected.append(chosen)
    result = pd.concat(selected, ignore_index=True).drop(columns=["_stratum"], errors="ignore")
    return result.sort_values(["label", "clip_id"]).reset_index(drop=True)


@dataclass(frozen=True)
class AssignmentItem:
    assignment_uuid: str
    task_uuid: str
    annotator: str
    role: str
    clip_id: int


def _schedule_with_gap(
    items: list[AssignmentItem], gap: int, seed: int,
) -> list[AssignmentItem]:
    by_clip: dict[int, deque[AssignmentItem]] = defaultdict(deque)
    rng = random.Random(seed)
    rng.shuffle(items)
    for item in items:
        by_clip[item.clip_id].append(item)
    heap = [(-len(queue), rng.random(), clip_id) for clip_id, queue in by_clip.items()]
    heapq.heapify(heap)
    cooldown: deque[tuple[int, float, int]] = deque()
    output: list[AssignmentItem] = []

    while heap or cooldown:
        while cooldown and cooldown[0][0] <= len(output):
            _, tie, clip_id = cooldown.popleft()
            heapq.heappush(heap, (-len(by_clip[clip_id]), tie, clip_id))
        if not heap:
            raise ValueError(f"Unable to schedule frame tasks with a {gap}-task clip gap")
        chosen = heapq.heappop(heap)
        _, _, clip_id = chosen
        output.append(by_clip[clip_id].popleft())
        if by_clip[clip_id]:
            cooldown.append((len(output) + gap, rng.random(), clip_id))
    return output


def _stratified_index_sample(
    targets: list[tuple[int, int, int]], fraction: float, rng: np.random.Generator
) -> set[int]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (_clip_id, label, frame_index) in enumerate(targets):
        groups[(label, frame_index)].append(index)
    target_total = round(len(targets) * fraction)
    exact = {key: len(indices) * fraction for key, indices in groups.items()}
    allocation = {key: math.floor(value) for key, value in exact.items()}
    remaining = target_total - sum(allocation.values())
    for key in sorted(groups, key=lambda value: (-(exact[value] - allocation[value]), value))[:remaining]:
        allocation[key] += 1
    selected: set[int] = set()
    for key in sorted(groups):
        count = allocation[key]
        if count:
            selected.update(rng.choice(groups[key], size=count, replace=False).tolist())
    if len(selected) != target_total:
        raise RuntimeError("Stratified repeat allocation did not reach its requested total")
    return selected


def prepare_study(
    connection: sqlite3.Connection,
    annotators: list[str],
    test: pd.DataFrame,
    train_alignment: pd.DataFrame,
    frame_second_fraction: float,
    hidden_repeat_fraction: float,
    minimum_gap: int,
    stimulus_features: pd.DataFrame | None = None,
) -> dict[str, int]:
    if len(annotators) != 2:
        raise ValueError("正式任务只需要两位独立标注者")
    if connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]:
        raise ValueError("Database already contains tasks; use a new database")

    for code in annotators:
        connection.execute(
            "INSERT INTO annotators(code, display_name) VALUES (?, ?)", (code, code)
        )

    task_rows: list[tuple] = []
    assignment_items: dict[str, list[AssignmentItem]] = {code: [] for code in annotators}
    raters = annotators
    rng = np.random.default_rng(20260912)

    frame_targets = [(int(row.clip_id), int(row.label), frame) for row in test.itertuples() for frame in range(1, 17)]
    rng.shuffle(frame_targets)
    second_indices = _stratified_index_sample(frame_targets, frame_second_fraction, rng)
    hidden_indices = _stratified_index_sample(frame_targets, hidden_repeat_fraction, rng)
    primary_counts: dict[tuple[int, int], int] = defaultdict(int)
    second_counts: dict[tuple[int, int, str], int] = defaultdict(int)

    for index, (clip_id, label, frame_index) in enumerate(frame_targets):
        base_task = stable_uuid("frame", "test", clip_id, frame_index, "primary")
        stratum = (label, frame_index)
        primary = raters[(primary_counts[stratum] + label + frame_index) % 2]
        primary_counts[stratum] += 1
        task_rows.append((base_task, "frame", "test", clip_id, frame_index, None, "none"))
        assignment = stable_uuid("assignment", base_task, primary, "primary")
        assignment_items[primary].append(AssignmentItem(assignment, base_task, primary, "primary", clip_id))

        if index in second_indices:
            eligible = [code for code in raters if code != primary]
            second = min(
                eligible,
                key=lambda code: (second_counts[(label, frame_index, code)], annotators.index(code)),
            )
            second_counts[(label, frame_index, second)] += 1
            second_task = stable_uuid("frame", "test", clip_id, frame_index, "second")
            task_rows.append((second_task, "frame", "test", clip_id, frame_index, base_task, "second_rater"))
            assignment = stable_uuid("assignment", second_task, second, "second")
            assignment_items[second].append(AssignmentItem(assignment, second_task, second, "second", clip_id))

        if index in hidden_indices:
            repeat_task = stable_uuid("frame", "test", clip_id, frame_index, "hidden", primary)
            task_rows.append((repeat_task, "frame", "test", clip_id, frame_index, base_task, "within_rater"))
            assignment = stable_uuid("assignment", repeat_task, primary, "hidden_repeat")
            assignment_items[primary].append(AssignmentItem(assignment, repeat_task, primary, "hidden_repeat", clip_id))

    clip_pairs = [(raters[0], raters[1])]
    for split, frame in [("test", test), ("train_alignment", train_alignment)]:
        ordered = frame.sort_values(["label", "clip_id"]).reset_index(drop=True)
        for label, group in ordered.groupby("label", sort=True):
            for within_label_index, row in enumerate(group.itertuples()):
                task_uuid = stable_uuid("clip", split, row.clip_id)
                task_rows.append((task_uuid, "clip", split, int(row.clip_id), None, None, "none"))
                pair = clip_pairs[0]
                for role_index, code in enumerate(pair):
                    assignment_uuid = stable_uuid("assignment", task_uuid, code, f"clip{role_index}")
                    assignment_items[code].append(AssignmentItem(
                        assignment_uuid, task_uuid, code, "primary" if role_index == 0 else "second", int(row.clip_id)
                    ))

    connection.executemany(
        "INSERT INTO tasks(task_uuid, task_type, split, clip_id, frame_index, source_task_uuid, repeat_kind) VALUES (?,?,?,?,?,?,?)",
        task_rows,
    )
    for annotator_index, code in enumerate(annotators):
        frame_task_ids = {row[0] for row in task_rows if row[1] == "frame"}
        frames = [item for item in assignment_items[code] if item.task_uuid in frame_task_ids]
        frame_assignment_ids = {item.assignment_uuid for item in frames}
        clips = [item for item in assignment_items[code] if item.assignment_uuid not in frame_assignment_ids]
        scheduled_frames = _schedule_with_gap(
            frames, minimum_gap, seed=9100 + annotator_index,
        )
        random.Random(9200 + annotator_index).shuffle(clips)
        queue = scheduled_frames + clips
        connection.executemany(
            "INSERT INTO assignments(assignment_uuid, task_uuid, annotator_code, role, queue_position) VALUES (?,?,?,?,?)",
            [(item.assignment_uuid, item.task_uuid, item.annotator, item.role, position)
             for position, item in enumerate(queue, start=1)],
        )

    meta = {
        "design_version": "1",
        "test_clips": str(len(test)),
        "train_alignment_clips": str(len(train_alignment)),
        "frame_second_fraction": str(frame_second_fraction),
        "hidden_repeat_fraction": str(hidden_repeat_fraction),
        "minimum_clip_gap": str(minimum_gap),
        "independent_annotators": ",".join(raters),
    }
    connection.executemany("INSERT INTO study_meta(key, value) VALUES (?, ?)", meta.items())
    connection.execute("PRAGMA optimize")
    return {
        "tasks": len(task_rows),
        "assignments": sum(len(items) for items in assignment_items.values()),
        "test_clips": len(test),
        "train_alignment_clips": len(train_alignment),
    }


def third_annotator(all_codes: list[str], existing: Iterable[str]) -> str:
    remaining = sorted(set(all_codes) - set(existing))
    if len(remaining) != 1:
        raise ValueError("Cannot determine unique third annotator")
    return remaining[0]


def needs_adjudication(ratings: list[sqlite3.Row]) -> tuple[bool, list[str]]:
    if len(ratings) != 2:
        return False, []
    reasons = []
    if ratings[0]["dominant_category"] != ratings[1]["dominant_category"]:
        reasons.append("category")
    curves = [np.asarray(json.loads(row["intensities_json"]), dtype=float) for row in ratings]
    if any(curve.shape != (16,) for curve in curves):
        reasons.append("invalid_curve")
        return True, reasons
    if np.max(np.abs(curves[0] - curves[1])) >= 3:
        reasons.append("intensity_gap")
    descriptors = [trajectory_descriptors(curve, row["dominant_category"]) for curve, row in zip(curves, ratings)]
    apexes = [item["apex_position"] for item in descriptors]
    if all(np.isfinite(value) for value in apexes) and abs(apexes[0] - apexes[1]) >= 4:
        reasons.append("peak_gap")
    peak_types = [item["peak_type"] for item in descriptors]
    if peak_types[0] != peak_types[1]:
        reasons.append("peak_type")
    return bool(reasons), reasons
