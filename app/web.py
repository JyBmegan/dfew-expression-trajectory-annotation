from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from functools import wraps
from io import BytesIO, StringIO
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)

from .db import get_db


bp = Blueprint("web", __name__)
CATEGORIES = [
    "Happiness", "Sadness", "Neutral", "Anger", "Surprise", "Disgust", "Fear",
    "Mixed", "Unclear", "Face not visible",
]
CATEGORY_LABELS = {
    "Happiness": "高兴", "Sadness": "悲伤", "Neutral": "中性", "Anger": "愤怒",
    "Surprise": "惊讶", "Disgust": "厌恶", "Fear": "恐惧", "Mixed": "混合",
    "Unclear": "无法判断", "Face not visible": "无法看清面部",
}


def _meta(db, key: str, default: str = "") -> str:
    row = db.execute("SELECT value FROM study_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def _progress_rows(db, code: str):
    return db.execute(
        """
        SELECT t.task_type, t.split, COUNT(*) total, SUM(a.status='complete') complete
        FROM assignments a JOIN tasks t USING(task_uuid)
        WHERE a.annotator_code=? GROUP BY t.task_type, t.split ORDER BY t.task_type, t.split
        """,
        (code,),
    ).fetchall()


def _all_complete(rows) -> bool:
    return bool(rows) and all((row["complete"] or 0) == row["total"] for row in rows)


def _csv_bytes(rows, columns: list[str]) -> bytes:
    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8-sig")


@bp.app_context_processor
def inject_labels():
    return {"category_labels": CATEGORY_LABELS}


def require_login(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("annotator"):
            return redirect(url_for("web.login"))
        return view(*args, **kwargs)
    return wrapped


def _assignment_or_404(assignment_uuid: str):
    row = get_db().execute(
        """
        SELECT a.*, t.task_type, t.split, t.clip_id, t.frame_index, t.repeat_kind
        FROM assignments a JOIN tasks t USING(task_uuid)
        WHERE a.assignment_uuid = ? AND a.annotator_code = ?
        """,
        (assignment_uuid, session.get("annotator")),
    ).fetchone()
    if row is None:
        abort(404)
    return row


@bp.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        code = request.form.get("code", "").strip().upper()
        row = get_db().execute("SELECT code FROM annotators WHERE code = ? AND active = 1", (code,)).fetchone()
        if row:
            session.clear()
            session["annotator"] = row["code"]
            return redirect(url_for("web.index"))
        error = "访问码无效，请检查负责人提供的代码。"
    return render_template("login.html", error=error)


@bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("web.login"))


@bp.get("/instructions")
@require_login
def instructions():
    return render_template("instructions.html", categories=CATEGORIES)


@bp.get("/")
@require_login
def index():
    db = get_db()
    code = session["annotator"]
    assignment = db.execute(
        """
        SELECT a.assignment_uuid, t.task_type
        FROM assignments a JOIN tasks t USING(task_uuid)
        WHERE a.annotator_code = ? AND a.status != 'complete'
        ORDER BY CASE a.status WHEN 'started' THEN 0 ELSE 1 END, a.queue_position
        LIMIT 1
        """,
        (code,),
    ).fetchone()
    if assignment is None:
        return redirect(url_for("web.dashboard"))
    return redirect(url_for("web.task", assignment_uuid=assignment["assignment_uuid"]))


@bp.get("/task/<assignment_uuid>")
@require_login
def task(assignment_uuid: str):
    db = get_db()
    assignment = _assignment_or_404(assignment_uuid)
    if assignment["status"] == "complete":
        return redirect(url_for("web.index"))
    if assignment["status"] == "pending":
        db.execute(
            "UPDATE assignments SET status='started', started_at=CURRENT_TIMESTAMP WHERE assignment_uuid=?",
            (assignment_uuid,),
        )
        db.commit()
    progress = db.execute(
        """
        SELECT COUNT(*) total, SUM(status='complete') complete
        FROM assignments WHERE annotator_code=?
        """,
        (session["annotator"],),
    ).fetchone()
    draft = db.execute("SELECT payload_json FROM drafts WHERE assignment_uuid=?", (assignment_uuid,)).fetchone()
    context = {
        "assignment": assignment,
        "categories": CATEGORIES,
        "progress": progress,
        "draft": draft["payload_json"] if draft else "{}",
        "is_practice": _meta(db, "practice_mode") == "1",
    }
    template = "frame_task.html" if assignment["task_type"] == "frame" else "clip_task.html"
    return render_template(template, **context)


@bp.post("/api/rating/<assignment_uuid>")
@require_login
def save_rating(assignment_uuid: str):
    assignment = _assignment_or_404(assignment_uuid)
    payload = request.get_json(force=True)
    db = get_db()
    started = assignment["started_at"]
    duration_ms = payload.get("duration_ms")
    if duration_ms is not None:
        duration_ms = max(0, min(int(duration_ms), 24 * 60 * 60 * 1000))

    if assignment["task_type"] == "frame":
        category = payload.get("visible_category")
        intensity = int(payload.get("intensity", -1))
        if category not in CATEGORIES or not 0 <= intensity <= 6:
            return jsonify(error="请选择可见表情类别，并将强度设为 0–6。"), 400
        if category in {"Neutral", "Face not visible"} and intensity != 0:
            return jsonify(error="中性和无法看清面部的画面，强度必须为 0。"), 400
        if category not in {"Neutral", "Unclear", "Face not visible"} and intensity == 0:
            return jsonify(error="可见表情的强度应为 1–6。"), 400
        db.execute(
            "INSERT OR REPLACE INTO frame_ratings(assignment_uuid, visible_category, intensity) VALUES (?,?,?)",
            (assignment_uuid, category, intensity),
        )
    else:
        category = payload.get("dominant_category")
        try:
            intensities = [int(value) for value in payload.get("intensities", [])]
        except (TypeError, ValueError):
            intensities = []
        if category not in CATEGORIES or len(intensities) != 16 or any(not 0 <= value <= 6 for value in intensities):
            return jsonify(error="请选择主要表情，并完成 16 个位置的强度评分（0–6）。"), 400
        if category in {"Neutral", "Face not visible"} and any(intensities):
            return jsonify(error="中性和无法看清面部的序列，16 个位置的强度都必须为 0。"), 400
        if category not in {"Neutral", "Unclear", "Face not visible"} and not any(intensities):
            return jsonify(error="可见的主要表情至少需要一个位置的强度高于 0。"), 400
        flags = [int(bool(payload.get(name))) for name in ["occlusion", "speaking", "abrupt_change", "subject_switch"]]
        db.execute(
            """
            INSERT OR REPLACE INTO clip_ratings(
                assignment_uuid, dominant_category, intensities_json,
                occlusion, speaking, abrupt_change, subject_switch
            ) VALUES (?,?,?,?,?,?,?)
            """,
            (assignment_uuid, category, json.dumps(intensities), *flags),
        )

    db.execute(
        "UPDATE assignments SET status='complete', completed_at=CURRENT_TIMESTAMP, duration_ms=? WHERE assignment_uuid=?",
        (duration_ms, assignment_uuid),
    )
    db.execute("DELETE FROM drafts WHERE assignment_uuid=?", (assignment_uuid,))
    db.commit()
    return jsonify(ok=True, next=url_for("web.index"))


@bp.post("/api/draft/<assignment_uuid>")
@require_login
def save_draft(assignment_uuid: str):
    _assignment_or_404(assignment_uuid)
    payload = request.get_json(force=True)
    get_db().execute(
        """
        INSERT INTO drafts(assignment_uuid, payload_json, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(assignment_uuid) DO UPDATE SET payload_json=excluded.payload_json, updated_at=CURRENT_TIMESTAMP
        """,
        (assignment_uuid, json.dumps(payload)),
    )
    get_db().commit()
    saved = get_db().execute(
        "SELECT updated_at FROM drafts WHERE assignment_uuid=?", (assignment_uuid,)
    ).fetchone()
    return jsonify(ok=True, saved_at=saved["updated_at"] if saved else None)


@bp.get("/media/<int:clip_id>/<int:frame_index>")
@require_login
def media(clip_id: int, frame_index: int):
    if not 1 <= frame_index <= 16:
        abort(404)
    allowed = get_db().execute(
        """
        SELECT 1 FROM assignments a JOIN tasks t USING(task_uuid)
        WHERE a.annotator_code=? AND t.clip_id=? LIMIT 1
        """,
        (session["annotator"], clip_id),
    ).fetchone()
    if not allowed:
        abort(404)
    try:
        data = current_app.config["FRAME_SOURCE"].read(clip_id, frame_index)
    except (FileNotFoundError, KeyError):
        abort(404)
    response = send_file(
        BytesIO(data), mimetype="image/jpeg", download_name=f"{clip_id:05d}_{frame_index}.jpg",
        conditional=True, max_age=86400,
    )
    response.cache_control.public = False
    response.cache_control.private = True
    return response


@bp.get("/dashboard")
@require_login
def dashboard():
    db = get_db()
    rows = _progress_rows(db, session["annotator"])
    return render_template(
        "dashboard.html",
        rows=rows,
        all_complete=_all_complete(rows),
        is_practice=_meta(db, "practice_mode") == "1",
    )


@bp.get("/export-results")
@require_login
def export_results():
    db = get_db()
    code = session["annotator"]
    rows = _progress_rows(db, code)
    if _meta(db, "practice_mode") == "1":
        abort(404)
    if not _all_complete(rows):
        return redirect(url_for("web.dashboard"))

    assignments = [dict(row) for row in db.execute(
        "SELECT * FROM assignments WHERE annotator_code=? ORDER BY queue_position", (code,)
    )]
    assignment_ids = [row["assignment_uuid"] for row in assignments]
    tasks = [dict(row) for row in db.execute(
        """SELECT t.* FROM tasks t JOIN assignments a USING(task_uuid)
           WHERE a.annotator_code=? ORDER BY a.queue_position""", (code,)
    )]
    if assignment_ids:
        marks = ",".join("?" for _ in assignment_ids)
        frame_ratings = [dict(row) for row in db.execute(
            f"SELECT * FROM frame_ratings WHERE assignment_uuid IN ({marks}) ORDER BY assignment_uuid",
            assignment_ids,
        )]
        clip_ratings = [dict(row) for row in db.execute(
            f"SELECT * FROM clip_ratings WHERE assignment_uuid IN ({marks}) ORDER BY assignment_uuid",
            assignment_ids,
        )]
    else:
        frame_ratings, clip_ratings = [], []

    members = {
        "assignments.csv": _csv_bytes(assignments, [
            "assignment_uuid", "task_uuid", "annotator_code", "role", "queue_position",
            "status", "started_at", "completed_at", "duration_ms",
        ]),
        "tasks.csv": _csv_bytes(tasks, [
            "task_uuid", "task_type", "split", "clip_id", "frame_index",
            "source_task_uuid", "repeat_kind", "created_at",
        ]),
        "frame_ratings.csv": _csv_bytes(frame_ratings, [
            "assignment_uuid", "visible_category", "intensity", "submitted_at",
        ]),
        "clip_ratings.csv": _csv_bytes(clip_ratings, [
            "assignment_uuid", "dominant_category", "intensities_json", "occlusion",
            "speaking", "abrupt_change", "subject_switch", "submitted_at",
        ]),
    }
    db.commit()
    with tempfile.TemporaryDirectory(prefix="dfew_export_") as temporary:
        snapshot = Path(temporary) / "study.sqlite"
        with sqlite3.connect(current_app.config["DATABASE"]) as source, sqlite3.connect(snapshot) as target:
            source.backup(target)
        members["study.sqlite"] = snapshot.read_bytes()

    now = datetime.now(timezone.utc)
    manifest = {
        "bundle_id": hashlib.sha256(
            (code + "|" + "|".join(sorted(assignment_ids))).encode("utf-8")
        ).hexdigest()[:20],
        "annotator": code,
        "design_version": _meta(db, "design_version", "unknown"),
        "exported_at": now.isoformat(),
        "completed_assignments": len(assignments),
        "total_assignments": len(assignments),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in members.items()},
    }
    members["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")

    archive = BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, data in members.items():
            output.writestr(name, data)
    archive.seek(0)
    filename = f"{code}_annotation_results_{now.strftime('%Y%m%d')}.zip"
    return send_file(archive, mimetype="application/zip", as_attachment=True, download_name=filename)


@bp.post("/reset-practice")
@require_login
def reset_practice():
    db = get_db()
    if _meta(db, "practice_mode") != "1" or session["annotator"] != "TEST":
        abort(404)
    assignment_ids = [row[0] for row in db.execute(
        "SELECT assignment_uuid FROM assignments WHERE annotator_code='TEST'"
    )]
    if assignment_ids:
        marks = ",".join("?" for _ in assignment_ids)
        db.execute(f"DELETE FROM drafts WHERE assignment_uuid IN ({marks})", assignment_ids)
        db.execute(f"DELETE FROM frame_ratings WHERE assignment_uuid IN ({marks})", assignment_ids)
        db.execute(f"DELETE FROM clip_ratings WHERE assignment_uuid IN ({marks})", assignment_ids)
    db.execute(
        """UPDATE assignments SET status='pending', started_at=NULL, completed_at=NULL,
           duration_ms=NULL WHERE annotator_code='TEST'"""
    )
    db.commit()
    return redirect(url_for("web.index"))
