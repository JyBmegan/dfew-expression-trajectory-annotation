#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", ".sites-runtime", "local_data", "exports", "backups", "artifacts", "__pycache__"}
SKIP_FILES = {"project.toml"}
BINARY_FORBIDDEN = {".pth", ".pt", ".npy", ".npz", ".sqlite", ".db", ".zip", ".mp4", ".avi"}
SENSITIVE_PATTERNS = [
    re.compile(r"/Volumes/Lenovo/"), re.compile(r"server_backup"),
    re.compile(r"frames_archive_password\s*=\s*\"[^\"]+\""),
]


def main() -> None:
    errors = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.resolve() == Path(__file__).resolve():
            continue
        if any(part in SKIP_DIRS for part in relative.parts) or path.name.startswith("._"):
            continue
        if path.is_dir() or path.name in SKIP_FILES:
            continue
        if path.suffix.lower() in BINARY_FORBIDDEN:
            errors.append(f"forbidden binary: {relative}")
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and "demo_data" not in relative.parts:
            errors.append(f"unexpected image outside synthetic demo: {relative}")
            continue
        if path.suffix.lower() in {".py", ".md", ".toml", ".txt", ".html", ".css", ".js", ".bat", ".command"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(text):
                    errors.append(f"sensitive local value in {relative}: {pattern.pattern}")
    if errors:
        print("Public-release check failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("Public-release check passed: no local data, checkpoints, archives, passwords, or machine-specific paths found")


if __name__ == "__main__":
    main()
