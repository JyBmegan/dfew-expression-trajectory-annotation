from __future__ import annotations

import re
import threading
import zipfile
from pathlib import Path


MEMBER = re.compile(r"(?:^|/)(\d{5})/(\d{1,2})\.jpe?g$", re.IGNORECASE)


class FrameSource:
    """Read 16-frame clips from a folder tree or the official DFEW archive."""

    def __init__(self, source: str | Path, password: str | bytes | None = None):
        self.source = Path(source).resolve()
        self.password = password.encode() if isinstance(password, str) else password
        self._archive: zipfile.ZipFile | None = None
        self._members: dict[tuple[int, int], str] = {}
        self._lock = threading.Lock()
        if self.source.is_file():
            self._archive = zipfile.ZipFile(self.source)
            for name in self._archive.namelist():
                match = MEMBER.search(name)
                if match:
                    frame = int(match.group(2))
                    if 1 <= frame <= 16:
                        self._members[(int(match.group(1)), frame)] = name
        elif not self.source.is_dir():
            raise FileNotFoundError(self.source)

    @property
    def kind(self) -> str:
        return "archive" if self._archive else "directory"

    def has_clip(self, clip_id: int) -> bool:
        if self._archive:
            return all((clip_id, frame) in self._members for frame in range(1, 17))
        folder = self.source / f"{clip_id:05d}"
        return all((folder / f"{frame}.jpg").is_file() for frame in range(1, 17))

    def read(self, clip_id: int, frame: int) -> bytes:
        if not 1 <= frame <= 16:
            raise FileNotFoundError(f"Invalid frame position {frame}")
        if self._archive:
            member = self._members.get((clip_id, frame))
            if member is None:
                raise FileNotFoundError(f"{clip_id:05d}/{frame}.jpg")
            with self._lock:
                return self._archive.read(member, pwd=self.password)
        path = self.source / f"{clip_id:05d}" / f"{frame}.jpg"
        return path.read_bytes()

    def close(self) -> None:
        if self._archive:
            self._archive.close()
