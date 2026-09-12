from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .common import read_config, resolve_path


MEMBER = re.compile(r"(?:^|/)(\d{5})/(\d{1,2})\.jpe?g$", re.IGNORECASE)


def requested_clips(config: dict, split: str) -> dict[int, str]:
    rows: dict[int, str] = {}
    choices = [split] if split != "all" else ["train", "test"]
    for name in choices:
        table = pd.read_csv(resolve_path(config, config["paths"][f"{name}_csv"]))
        table = table[table["label"].between(1, 7)]
        for clip_id in table["video_name"].astype(int).unique():
            if clip_id in rows and rows[clip_id] != name:
                raise ValueError(f"Clip {clip_id:05d} occurs in both train and test")
            rows[clip_id] = name
    return rows


def valid_folder(folder: Path) -> bool:
    return all((folder / f"{index}.jpg").is_file() and (folder / f"{index}.jpg").stat().st_size > 0 for index in range(1, 17))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the official DFEW 16-frame inputs used by this study.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Development-only clip limit")
    args = parser.parse_args()

    config = read_config(args.config)
    paths = config["paths"]
    archive = resolve_path(config, paths["frames_16_archive"])
    password = paths.get("frames_archive_password", "").encode() or None
    destination = resolve_path(config, paths["frames"])
    destination.mkdir(parents=True, exist_ok=True)
    wanted = requested_clips(config, args.split)
    if args.limit:
        wanted = dict(sorted(wanted.items())[: args.limit])

    with zipfile.ZipFile(archive) as handle:
        members: dict[int, dict[int, str]] = defaultdict(dict)
        for info in handle.infolist():
            match = MEMBER.search(info.filename)
            if not match:
                continue
            clip_id, frame = int(match.group(1)), int(match.group(2))
            if clip_id in wanted and 1 <= frame <= 16:
                members[clip_id][frame] = info.filename

        missing = {clip_id: sorted(set(range(1, 17)) - set(members.get(clip_id, {}))) for clip_id in wanted}
        missing = {clip_id: frames for clip_id, frames in missing.items() if frames}
        if missing:
            example = ", ".join(f"{key:05d}:{value}" for key, value in list(missing.items())[:5])
            raise RuntimeError(f"Official archive lacks required frames for {len(missing)} clips ({example})")

        records = []
        extracted = skipped = 0
        for number, (clip_id, split) in enumerate(sorted(wanted.items()), start=1):
            folder = destination / f"{clip_id:05d}"
            if valid_folder(folder) and not args.overwrite:
                skipped += 1
            else:
                folder.mkdir(parents=True, exist_ok=True)
                for frame in range(1, 17):
                    target = folder / f"{frame}.jpg"
                    with handle.open(members[clip_id][frame], pwd=password) as source, target.open("wb") as output:
                        while chunk := source.read(1024 * 1024):
                            output.write(chunk)
                if not valid_folder(folder):
                    raise RuntimeError(f"Extraction validation failed for {clip_id:05d}")
                extracted += 1
            records.append({"clip_id": clip_id, "split": split, "frames": 16, "validated": 1})
            if number % 500 == 0:
                print(f"Prepared {number}/{len(wanted)} clips")

    manifest = destination.parent / "frames_16_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=["clip_id", "split", "frames", "validated"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Official 16-frame inputs ready: {len(records)} clips ({extracted} extracted, {skipped} already valid)")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
