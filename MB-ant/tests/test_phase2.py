import tempfile
import unittest
from pathlib import Path

import numpy as np

from phase2.capture import HeadingLogWriter
from phase2.log_schema import load_log, write_log
from phase2.synthetic import make_dataset
from phase2.sync import gyro_deltas_at_frames
from phase2.visual_memory import FamiliarityMemory, SparseKCEncoder, preprocess_proxy_panorama


class Phase2Test(unittest.TestCase):
    def test_gyro_integration_between_camera_frames(self):
        deltas = gyro_deltas_at_frames(
            [0, 0.05, 0.1, 0.15, 0.2], [20, 20, 20, 20, 20], [0, 0.1, 0.2]
        )
        np.testing.assert_allclose(deltas, [0, 2, 2])

    def test_writer_and_validator_preserve_synchronized_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            with HeadingLogWriter(path, run_id="run-a", route_id="route-a",
                                  split="validation", origin="simulator",
                                  condition="nominal", visual_source="legacy_en_scan",
                                  context_source="odometry_place_progress") as writer:
                writer.append(timestamp_s=0.0, gyro_delta_deg=0.0,
                              heading_truth_deg=20, sector_angles_deg=[10, 20, 30],
                              familiarity_scores=[4, 1, 5], context_ok=True)
                writer.append(timestamp_s=0.1, gyro_delta_deg=2.0,
                              heading_truth_deg=22, sector_angles_deg=[],
                              familiarity_scores=[], context_ok=False)
            records = load_log(path)
            self.assertEqual([r["frame"] for r in records], [0, 1])
            self.assertEqual(records[1]["gyro_delta_deg"], 2.0)

    def test_route_split_leakage_is_rejected(self):
        records = make_dataset(steps=20, replicates_per_split=1)
        test_record = next(r for r in records if r["split"] == "test")
        test_record["route_id"] = records[0]["route_id"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                write_log(Path(directory) / "bad.jsonl", records)

    def test_sparse_visual_memory_self_match(self):
        image = np.zeros((19, 74), dtype=np.uint8)
        pn = preprocess_proxy_panorama(image)
        self.assertEqual(pn.shape, (360,))
        self.assertAlmostEqual(np.linalg.norm(pn), 1.0)
        encoder = SparseKCEncoder(num_pn=4, num_kc=64, fan_in=2,
                                  active_fraction=0.125, seed=4)
        memory = FamiliarityMemory(encoder).fit(np.array([[1, 0, 0, 0], [0, 0, 0, 1]]))
        score, index = memory.score(np.array([1, 0, 0, 0]))
        self.assertEqual(score, 0.0)
        self.assertEqual(index, 0)


if __name__ == "__main__":
    unittest.main()
