from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image

from app import create_app
from app.db import connect, init_database
from app.frame_source import FrameSource
from app.study import prepare_study


class AnnotationPlatformTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "study.sqlite"
        init_database(self.database)
        test = pd.DataFrame({
            "clip_id": range(1, 22),
            "label": [(index % 7) + 1 for index in range(21)],
            "emotion": ["x"] * 21,
        })
        train = pd.DataFrame({
            "clip_id": range(101, 115),
            "label": [(index % 7) + 1 for index in range(14)],
            "emotion": ["x"] * 14,
        })
        with connect(self.database) as connection:
            self.summary = prepare_study(
                connection, ["R01", "R02"], test, train, .10, .02, 9
            )
            connection.commit()
        self.frames = self.root / "frames"
        for clip_id in [*test.clip_id, *train.clip_id]:
            folder = self.frames / f"{clip_id:05d}"
            folder.mkdir(parents=True)
            for frame in range(1, 17):
                Image.new("RGB", (32, 32), (clip_id % 255, frame, 50)).save(
                    folder / f"{frame}.jpg"
                )

    def tearDown(self):
        self.temp.cleanup()

    def test_only_two_independent_accounts_and_nine_intervening_tasks(self):
        with connect(self.database) as connection:
            self.assertEqual(
                [row[0] for row in connection.execute("SELECT code FROM annotators ORDER BY code")],
                ["R01", "R02"],
            )
            for code in ["R01", "R02"]:
                rows = connection.execute(
                    """SELECT t.clip_id, a.queue_position FROM assignments a
                       JOIN tasks t USING(task_uuid) WHERE a.annotator_code=?
                       AND t.task_type='frame' ORDER BY a.queue_position""",
                    (code,),
                ).fetchall()
                last = {}
                for clip_id, position in rows:
                    if clip_id in last:
                        self.assertGreaterEqual(position - last[clip_id] - 1, 9)
                    last[clip_id] = position

    def test_archive_lookup_uses_exact_clip_and_frame(self):
        archive = self.root / "frames.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for frame in range(1, 17):
                output.write(
                    self.frames / "00001" / f"{frame}.jpg",
                    f"clip_224x224_16f/00001/{frame}.jpg",
                )
        source = FrameSource(archive)
        self.assertEqual(source.available_clip_ids(), [1])
        self.assertTrue(source.read(1, 7).startswith(b"\xff\xd8"))
        with self.assertRaises(FileNotFoundError):
            source.read(2, 7)
        source.close()

    def test_draft_is_saved_and_restored_without_submitting(self):
        app = create_app(self.database, self.frames, "test")
        app.testing = True
        client = app.test_client()
        client.post("/login", data={"code": "r01"})
        with connect(self.database) as connection:
            assignment = connection.execute(
                """SELECT assignment_uuid FROM assignments JOIN tasks USING(task_uuid)
                   WHERE annotator_code='R01' AND task_type='frame'
                   ORDER BY queue_position LIMIT 1"""
            ).fetchone()[0]
        response = client.post(
            f"/api/draft/{assignment}",
            json={"visible_category": "Happiness", "intensity": 4},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["saved_at"])
        page = client.get(f"/task/{assignment}")
        self.assertIn(b'Happiness', page.data)
        self.assertIn(b'"intensity": 4', page.data)
        with connect(self.database) as connection:
            status = connection.execute(
                "SELECT status FROM assignments WHERE assignment_uuid=?", (assignment,)
            ).fetchone()[0]
        self.assertEqual(status, "started")

    def test_complete_dashboard_downloads_one_checked_zip(self):
        with connect(self.database) as connection:
            frame_ids = connection.execute(
                """SELECT a.assignment_uuid FROM assignments a JOIN tasks t USING(task_uuid)
                   WHERE a.annotator_code='R01' AND t.task_type='frame'"""
            ).fetchall()
            clip_ids = connection.execute(
                """SELECT a.assignment_uuid FROM assignments a JOIN tasks t USING(task_uuid)
                   WHERE a.annotator_code='R01' AND t.task_type='clip'"""
            ).fetchall()
            connection.executemany(
                """INSERT INTO frame_ratings(assignment_uuid,visible_category,intensity)
                   VALUES (?,'Happiness',3)""",
                frame_ids,
            )
            connection.executemany(
                """INSERT INTO clip_ratings(assignment_uuid,dominant_category,intensities_json)
                   VALUES (?,'Happiness',?)""",
                [(row[0], json.dumps([3] * 16)) for row in clip_ids],
            )
            connection.execute(
                "UPDATE assignments SET status='complete' WHERE annotator_code='R01'"
            )
            connection.commit()
        app = create_app(self.database, self.frames, "test")
        app.testing = True
        client = app.test_client()
        client.post("/login", data={"code": "R01"})
        page = client.get("/dashboard")
        self.assertIn("全部正式任务已完成".encode(), page.data)
        response = client.get("/export-results")
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(BytesIO(response.data)) as archive:
            self.assertEqual(set(archive.namelist()), {
                "assignments.csv", "tasks.csv", "frame_ratings.csv",
                "clip_ratings.csv", "study.sqlite", "manifest.json",
            })
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(
                manifest["completed_assignments"], manifest["total_assignments"]
            )


if __name__ == "__main__":
    unittest.main()
