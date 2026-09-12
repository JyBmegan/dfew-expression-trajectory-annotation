from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import read_config, resolve_path


PART = re.compile(r"clip_224x224_part_(\d+)\.zip$", re.IGNORECASE)
FRAME = re.compile(r"(?:^|/)(\d{5})/(?:\1_)?(\d+)\.jpe?g$", re.IGNORECASE)


def direct_parts(root: Path) -> dict[int, Path]:
    found = {}
    if not root.exists():
        return found
    for path in root.rglob("clip_224x224_part_*.zip"):
        if path.name.startswith("._"):
            continue
        match = PART.search(path.name)
        if match and zipfile.is_zipfile(path):
            found[int(match.group(1))] = path
    return found


def nested_parts(root: Path) -> dict[int, tuple[Path, str]]:
    found = {}
    if not root.exists():
        return found
    for outer in root.glob("clip_224x224-*.zip"):
        if outer.name.startswith("._"):
            continue
        try:
            with zipfile.ZipFile(outer) as handle:
                for name in handle.namelist():
                    match = PART.search(name)
                    if match:
                        found[int(match.group(1))] = (outer, name)
        except zipfile.BadZipFile:
            continue
    return found


def count_archive(path: Path) -> dict[int, int]:
    counts: Counter[int] = Counter()
    with zipfile.ZipFile(path) as handle:
        for name in handle.namelist():
            match = FRAME.search(name)
            if match:
                counts[int(match.group(1))] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover original face-frame counts from full-length DFEW archives.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/full_length_index.csv")
    parser.add_argument("--cache", default="local_data/full_length_index_cache.sqlite")
    parser.add_argument("--parts", help="Comma-separated archive parts for a partial run")
    parser.add_argument("--temp-dir", default=tempfile.gettempdir())
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    config = read_config(args.config); paths = config["paths"]
    direct_root = resolve_path(config, paths.get("full_length_direct_archives", paths["full_length_archives"]))
    outer_root = resolve_path(config, paths["full_length_archives"])
    direct = direct_parts(direct_root); nested = nested_parts(outer_root)
    available = sorted(set(direct) | set(nested))
    requested = sorted({int(value) for value in args.parts.split(",")}) if args.parts else available
    unavailable = sorted(set(requested) - set(available))
    if unavailable:
        raise FileNotFoundError(f"No full-length archive source for parts {unavailable}")

    cache_path = Path(args.cache); cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = sqlite3.connect(cache_path)
    cache.execute("CREATE TABLE IF NOT EXISTS part_cache(part INTEGER PRIMARY KEY, source TEXT NOT NULL, payload_json TEXT NOT NULL)")
    if args.rebuild:
        marks = ",".join("?" for _ in requested)
        cache.execute(f"DELETE FROM part_cache WHERE part IN ({marks})", requested)
        cache.commit()

    for part in requested:
        if cache.execute("SELECT 1 FROM part_cache WHERE part=?", (part,)).fetchone():
            print(f"Part {part}: already indexed")
            continue
        if part in direct:
            source_name = str(direct[part])
            counts = count_archive(direct[part])
        else:
            outer, member = nested[part]
            source_name = f"{outer.name}:{member}"
            with tempfile.NamedTemporaryFile(prefix=f"dfew_part_{part}_", suffix=".zip", dir=args.temp_dir) as temporary:
                print(f"Part {part}: streaming nested archive to temporary storage")
                with zipfile.ZipFile(outer) as handle, handle.open(member) as source:
                    shutil.copyfileobj(source, temporary, length=8 * 1024 * 1024)
                temporary.flush()
                counts = count_archive(Path(temporary.name))
        cache.execute(
            "INSERT INTO part_cache(part, source, payload_json) VALUES (?,?,?)",
            (part, source_name, json.dumps(counts, separators=(",", ":"))),
        )
        cache.commit()
        print(f"Part {part}: {len(counts)} clips indexed")

    rows = []
    for part, source, payload in cache.execute("SELECT part, source, payload_json FROM part_cache ORDER BY part"):
        for clip_id, count in json.loads(payload).items():
            rows.append({
                "clip_id": int(clip_id), "archive_part": int(part),
                "original_frame_count": int(count),
                # This is a scale descriptor only.  Exact source-frame indices
                # are not asserted unless the official 16-frame files are
                # explicitly matched back to the full sequence.
                "nominal_uniform_interval_frames": (int(count) - 1) / 15,
                "archive_source": source,
            })
    cache.close()
    result = pd.DataFrame(rows).sort_values("clip_id")
    if result.clip_id.duplicated().any():
        duplicates = result.loc[result.clip_id.duplicated(), "clip_id"].head().tolist()
        raise ValueError(f"Clips found in multiple archive parts: {duplicates}")
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Full-length index: {len(result)} clips across {result.archive_part.nunique()} parts -> {output}")


if __name__ == "__main__":
    main()
