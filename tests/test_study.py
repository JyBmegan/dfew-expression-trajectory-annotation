from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from app import create_app
from app.db import connect, init_database
from app.frame_source import FrameSource
from app.study import needs_adjudication, prepare_study
from analysis.common import trajectory_descriptors, true_class_margin
from analysis.annotation_agreement import frame_agreement
from analysis.cache_features import extract_batch
from analysis.frame_subsets import model_subset_statistics
from analysis.temporal_models import AveragePoolHead, make_head, order_indices
from analysis.temporal_effects import (
    temporal_minus_endpoint_control, train_test_alignment_contrast, within_run_contrasts,
)
from manage import command_adjudications, command_calibration, command_consensus, command_make_bundle, consensus_category


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.database = self.root / "study.sqlite"
        init_database(self.database)
        test = pd.DataFrame({
            "clip_id": range(1, 10),
            "label": [(index % 7) + 1 for index in range(9)],
            "emotion": ["x"] * 9,
        })
        train = pd.DataFrame({
            "clip_id": range(101, 116),
            "label": [(index % 7) + 1 for index in range(15)],
            "emotion": ["x"] * 15,
        })
        with connect(self.database) as connection:
            self.summary = prepare_study(connection, ["R01", "R02", "R03"], test, train, .10, .02, 2)
            connection.commit()
        self.frames = self.root / "frames"
        for clip_id in list(test.clip_id) + list(train.clip_id):
            folder = self.frames / f"{clip_id:05d}"; folder.mkdir(parents=True)
            for frame in range(1, 17):
                Image.new("RGB", (16, 16), (clip_id % 255, frame, 50)).save(folder / f"{frame}.jpg")

    def tearDown(self): self.temp.cleanup()

    def test_assignment_counts_and_gap(self):
        self.assertEqual(self.summary["test_clips"], 9)
        with connect(self.database) as connection:
            base = connection.execute("SELECT COUNT(*) FROM tasks WHERE task_type='frame' AND repeat_kind='none'").fetchone()[0]
            self.assertEqual(base, 9 * 16)
            for code in ["R01", "R02", "R03"]:
                clips = [row[0] for row in connection.execute(
                    "SELECT clip_id FROM assignments JOIN tasks USING(task_uuid) WHERE annotator_code=? AND task_type='frame' ORDER BY queue_position", (code,)
                )]
                last = {}
                for position, clip_id in enumerate(clips):
                    if clip_id in last:
                        self.assertGreater(position - last[clip_id], 2)
                    last[clip_id] = position

    def test_frame_submission_does_not_expose_label(self):
        app = create_app(self.database, self.frames, "test")
        app.testing = True
        client = app.test_client()
        response = client.post("/login", data={"code": "R01"}, follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        with connect(self.database) as connection:
            assignment = connection.execute(
                "SELECT assignment_uuid FROM assignments JOIN tasks USING(task_uuid) WHERE annotator_code='R01' AND task_type='frame' ORDER BY queue_position LIMIT 1"
            ).fetchone()[0]
        page = client.get(f"/task/{assignment}")
        self.assertEqual(page.status_code, 200)
        self.assertNotIn(b"DFEW label", page.data)
        saved = client.post(f"/api/rating/{assignment}", json={"visible_category": "Happiness", "intensity": 4, "duration_ms": 1200})
        self.assertEqual(saved.status_code, 200)
        with connect(self.database) as connection:
            status = connection.execute("SELECT status FROM assignments WHERE assignment_uuid=?", (assignment,)).fetchone()[0]
            self.assertEqual(status, "complete")

    def test_neutral_frame_requires_zero_intensity(self):
        app = create_app(self.database, self.frames, "test")
        app.testing = True
        client = app.test_client()
        client.post("/login", data={"code": "R01"})
        with connect(self.database) as connection:
            assignment = connection.execute(
                "SELECT assignment_uuid FROM assignments JOIN tasks USING(task_uuid) WHERE annotator_code='R01' AND task_type='frame' ORDER BY queue_position LIMIT 1"
            ).fetchone()[0]
        invalid = client.post(
            f"/api/rating/{assignment}",
            json={"visible_category": "Neutral", "intensity": 3, "duration_ms": 100},
        )
        self.assertEqual(invalid.status_code, 400)
        valid = client.post(
            f"/api/rating/{assignment}",
            json={"visible_category": "Neutral", "intensity": 0, "duration_ms": 100},
        )
        self.assertEqual(valid.status_code, 200)

    def test_visible_expression_requires_nonzero_intensity(self):
        app = create_app(self.database, self.frames, "test")
        app.testing = True
        client = app.test_client()
        client.post("/login", data={"code": "R01"})
        with connect(self.database) as connection:
            assignment = connection.execute(
                "SELECT assignment_uuid FROM assignments JOIN tasks USING(task_uuid) WHERE annotator_code='R01' AND task_type='frame' ORDER BY queue_position LIMIT 1"
            ).fetchone()[0]
        invalid = client.post(
            f"/api/rating/{assignment}",
            json={"visible_category": "Happiness", "intensity": 0, "duration_ms": 100},
        )
        self.assertEqual(invalid.status_code, 400)

    def test_adjudication_rules(self):
        class Row(dict):
            def __getitem__(self, item): return super().__getitem__(item)
        a = Row(dominant_category="Happiness", intensities_json=json.dumps([0,0,1,1,2,2,3,3,4,4,5,5,6,6,6,6]))
        b = Row(dominant_category="Sadness", intensities_json=json.dumps([6,6,5,5,4,4,3,3,2,2,1,1,0,0,0,0]))
        needed, reasons = needs_adjudication([a, b])
        self.assertTrue(needed)
        self.assertIn("category", reasons)
        self.assertIn("intensity_gap", reasons)

    def test_temporal_orders_use_same_frames(self):
        curve = np.asarray([0,2,1,4,3,6,5,1,2,3,4,5,6,2,1,0])
        weak = order_indices(curve, "weak_to_strong")
        strong = order_indices(curve, "strong_to_weak")
        self.assertEqual(set(weak), set(range(16)))
        self.assertEqual(set(strong), set(range(16)))
        self.assertTrue(np.all(np.diff(curve[weak]) >= 0))
        self.assertTrue(np.all(np.diff(curve[strong]) <= 0))
        for value in np.unique(curve):
            self.assertTrue(np.all(np.diff(weak[curve[weak] == value]) > 0))
            self.assertTrue(np.all(np.diff(strong[curve[strong] == value]) > 0))
        np.testing.assert_array_equal(order_indices(curve, "reverse"), np.arange(15, -1, -1))

    def test_temporal_heads_return_seven_logits(self):
        values = torch.randn(3, 16, 24)
        for name in ["average", "last_frame", "unidirectional_gru", "bidirectional_gru", "causal_tcn"]:
            with self.subTest(head=name):
                self.assertEqual(tuple(make_head(name, 24)(values).shape), (3, 7))

    def test_average_pooling_is_order_invariant(self):
        model = AveragePoolHead(12).eval()
        values = torch.randn(4, 16, 12)
        with torch.inference_mode():
            natural = model(values)
            reversed_order = model(values[:, torch.arange(15, -1, -1)])
        torch.testing.assert_close(natural, reversed_order)

    def test_last_frame_control_reads_only_the_endpoint(self):
        model = make_head("last_frame", 12).eval()
        first = torch.randn(3, 16, 12)
        second = first.clone()
        second[:, :-1] = torch.randn_like(second[:, :-1])
        with torch.inference_mode():
            torch.testing.assert_close(model(first), model(second))

    def test_temporal_effect_contrasts_are_paired(self):
        rows = []
        orders = ["natural", "reverse", "weak_to_strong", "strong_to_weak"] + [f"shuffle_{i:02d}" for i in range(1, 21)]
        for training in ["weak_to_strong", "strong_to_weak"]:
            for clip_id in [1, 2]:
                for test_order in orders:
                    value = 2.0 if training == test_order else 1.0
                    rows.append({
                        "backbone": "alexnet", "head": "unidirectional_gru",
                        "train_order": training, "reference_full": False,
                        "clip_id": clip_id, "label": clip_id - 1,
                        "test_order": test_order, "true_class_logit_margin": value,
                    })
        frame = pd.DataFrame(rows)
        within = within_run_contrasts(frame)
        self.assertEqual(within.clip_id.nunique(), 2)
        self.assertEqual(set(within.contrast), {
            "natural_minus_reverse", "weak_to_strong_minus_strong_to_weak", "natural_minus_mean_shuffle",
        })
        alignment = train_test_alignment_contrast(frame)
        np.testing.assert_allclose(alignment.margin_contrast, 1.0)

    def test_temporal_order_effect_can_be_compared_with_endpoint_control(self):
        rows = []
        orders = ["natural", "reverse", "weak_to_strong", "strong_to_weak"] + [f"shuffle_{i:02d}" for i in range(1, 21)]
        for head, multiplier in [("last_frame", 1.0), ("unidirectional_gru", 2.0)]:
            for clip_id in [1, 2]:
                for order in orders:
                    value = multiplier if order == "natural" else 0.0
                    rows.append({
                        "backbone": "alexnet", "head": head, "train_order": "natural",
                        "reference_full": False, "clip_id": clip_id, "label": clip_id - 1,
                        "test_order": order, "true_class_logit_margin": value,
                    })
        within = within_run_contrasts(pd.DataFrame(rows))
        adjusted = temporal_minus_endpoint_control(within)
        natural_reverse = adjusted[adjusted.contrast.str.endswith("natural_minus_reverse")]
        np.testing.assert_allclose(natural_reverse.margin_contrast, 1.0)

    def test_margin(self):
        logits = np.asarray([[3.0, 1.0, 2.0], [0.0, 4.0, 2.0]])
        np.testing.assert_allclose(true_class_margin(logits, np.asarray([0, 1])), [1.0, 2.0])

    def test_feature_cache_uses_model_feature_contract(self):
        class FrozenModel(torch.nn.Module):
            def forward(self, values, return_features=False):
                self.return_features = return_features
                return values[:, :2, 0, 0], values[:, :, 0, 0] - 3
        model = FrozenModel()
        values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        logits, features = extract_batch("alexnet", model, values)
        self.assertTrue(model.return_features)
        torch.testing.assert_close(features, values[:, :, 0, 0] - 3)
        torch.testing.assert_close(logits, values[:, :2, 0, 0])

    def test_subset_statistics_compares_same_four_frames_with_full_sequence(self):
        values = np.zeros((2, 16, 7), dtype=np.float32)
        values[0, :, 0] = np.arange(16)
        values[1, :, 1] = np.arange(16)[::-1]
        result = model_subset_statistics(values, np.asarray([0, 1]), (0, 5, 10, 15))
        self.assertEqual(result["prediction_agreement"], 1.0)
        self.assertEqual(result["war"], 1.0)

    def test_frame_repeats_are_summarized_separately(self):
        pairs = pd.DataFrame([
            {
                "repeat_kind": "second_rater", "primary_category": "Happiness",
                "repeat_category": "Happiness", "primary_intensity": 4, "repeat_intensity": 5,
            },
            {
                "repeat_kind": "within_rater", "primary_category": "Sadness",
                "repeat_category": "Neutral", "primary_intensity": 3, "repeat_intensity": 0,
            },
        ])
        summary, _ = frame_agreement(pairs)
        by_kind = summary.set_index("rating_type")
        self.assertEqual(by_kind.loc["second_rater", "category_agreement"], 1.0)
        self.assertEqual(by_kind.loc["within_rater", "intensity_mean_absolute_gap"], 3.0)

    def test_category_consensus_does_not_invent_mixed(self):
        self.assertEqual(consensus_category(["Happiness"]), "Happiness")
        self.assertEqual(consensus_category(["Happiness", "Happiness"]), "Happiness")
        self.assertEqual(consensus_category(["Happiness", "Sadness"]), "No consensus")
        self.assertEqual(consensus_category(["Happiness", "Sadness", "Fear"]), "No consensus")

    def test_unusable_trajectories_have_no_apex(self):
        for category in ["Neutral", "Face not visible", "No consensus"]:
            result = trajectory_descriptors(np.zeros(16), category)
            self.assertTrue(np.isnan(result["apex_position"]))
        absent = trajectory_descriptors(np.zeros(16), "Happiness")
        self.assertTrue(np.isnan(absent["apex_position"]))
        self.assertEqual(absent["peak_type"], "no_visible_expression")

    def test_archive_frame_source(self):
        archive = self.root / "frames.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            for frame in range(1, 17):
                handle.write(self.frames / "00001" / f"{frame}.jpg", f"clip_224x224_16f/00001/{frame}.jpg")
        source = FrameSource(archive)
        self.assertTrue(source.has_clip(1))
        self.assertTrue(source.read(1, 1).startswith(b"\xff\xd8"))
        source.close()

    def test_calibration_is_inserted_before_main_queue(self):
        manifest = self.root / "calibration.csv"
        pd.DataFrame([
            {"clip_id": 1, "task_type": "clip", "frame_index": np.nan},
            {"clip_id": 2, "task_type": "frame", "frame_index": 4},
        ]).to_csv(manifest, index=False)
        command_calibration(SimpleNamespace(database=self.database, manifest=manifest))
        with connect(self.database) as connection:
            for code in ["R01", "R02", "R03"]:
                rows = connection.execute(
                    """SELECT t.split FROM assignments a JOIN tasks t USING(task_uuid)
                       WHERE a.annotator_code=? ORDER BY a.queue_position LIMIT 2""", (code,),
                ).fetchall()
                self.assertEqual([row["split"] for row in rows], ["calibration", "calibration"])

    def test_adjudication_and_consensus_export(self):
        with connect(self.database) as connection:
            task = connection.execute("SELECT task_uuid FROM tasks WHERE task_type='clip' AND split='test' LIMIT 1").fetchone()[0]
            assignments = connection.execute(
                "SELECT assignment_uuid FROM assignments WHERE task_uuid=? ORDER BY annotator_code", (task,)
            ).fetchall()
            curves = [[0] * 8 + [4] * 8, [5] * 8 + [0] * 8]
            categories = ["Happiness", "Sadness"]
            for assignment, curve, category in zip(assignments, curves, categories):
                connection.execute(
                    "INSERT INTO clip_ratings(assignment_uuid, dominant_category, intensities_json) VALUES (?,?,?)",
                    (assignment[0], category, json.dumps(curve)),
                )
                connection.execute("UPDATE assignments SET status='complete' WHERE assignment_uuid=?", (assignment[0],))
            connection.commit()
        command_adjudications(SimpleNamespace(database=self.database))
        with connect(self.database) as connection:
            third = connection.execute(
                "SELECT assignment_uuid FROM assignments WHERE task_uuid=? AND role='adjudication'", (task,)
            ).fetchone()
            self.assertIsNotNone(third)
            connection.execute(
                "INSERT INTO clip_ratings(assignment_uuid, dominant_category, intensities_json) VALUES (?,?,?)",
                (third[0], "Happiness", json.dumps([0] * 6 + [4] * 10)),
            )
            connection.execute("UPDATE assignments SET status='complete' WHERE assignment_uuid=?", (third[0],))
            connection.commit()
        export = self.root / "consensus"
        command_consensus(SimpleNamespace(database=self.database, output=export))
        adjudications = pd.read_csv(export / "adjudications.csv")
        consensus = pd.read_csv(export / "trajectory_consensus.csv")
        self.assertEqual(len(adjudications), 1)
        self.assertEqual(len(consensus), 1)
        self.assertEqual(consensus.iloc[0].n_ratings, 3)

    def test_annotator_bundle_contains_one_rater(self):
        output = self.root / "bundle"
        command_make_bundle(SimpleNamespace(database=self.database, annotator="R02", output=output))
        with connect(output / "study.sqlite") as connection:
            self.assertEqual([row[0] for row in connection.execute("SELECT code FROM annotators")], ["R02"])
            self.assertEqual([row[0] for row in connection.execute("SELECT DISTINCT annotator_code FROM assignments")], ["R02"])


if __name__ == "__main__":
    unittest.main()
