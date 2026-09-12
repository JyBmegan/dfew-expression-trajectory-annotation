from __future__ import annotations

from pathlib import Path

from flask import Flask

from .db import close_db
from .frame_source import FrameSource
from .web import bp


def create_app(
    database: str | Path,
    frames_root: str | Path,
    secret_key: str,
    frame_password: str | bytes | None = None,
) -> Flask:
    app = Flask(__name__)
    frame_source = FrameSource(frames_root, frame_password)
    app.config.update(
        DATABASE=str(Path(database).resolve()),
        FRAME_SOURCE=frame_source,
        SECRET_KEY=secret_key,
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    )
    app.teardown_appcontext(close_db)
    app.register_blueprint(bp)
    return app
