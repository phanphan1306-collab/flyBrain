import unittest

from heading_fusion import HeadingFusion, wrap


class HeadingFusionTest(unittest.TestCase):
    def test_wrap_and_gyro_prediction(self):
        model = HeadingFusion(179)
        result = model.update(5)
        self.assertEqual(result.heading_deg, -176)
        self.assertEqual(result.reason, "gyro_only")
        self.assertEqual(wrap(-181), 179)

    def test_equal_visual_scores_do_not_correct(self):
        model = HeadingFusion(70)
        result = model.update(2, [40, 70, 100], [0, 0, 0])
        self.assertEqual(result.heading_deg, 72)
        self.assertEqual(result.confidence, 0)
        self.assertEqual(result.reason, "ambiguous_visual")

    def test_large_disagreement_requires_context_and_repeat(self):
        model = HeadingFusion(70, repeat_count=3)
        sectors = [130, 140, 150, 160, 170]
        scores = [0, 7, 8, 9, 10]
        first = model.update(0, sectors, scores, context_ok=True)
        second = model.update(0, sectors, scores, context_ok=True)
        third = model.update(0, sectors, scores, context_ok=False)
        self.assertEqual(first.correction_deg, 0)
        self.assertEqual(second.correction_deg, 0)
        self.assertEqual(third.reason, "large_innovation_requires_context")
        fourth = model.update(0, sectors, scores, context_ok=True)
        self.assertEqual(fourth.reason, "repeated_large_innovation")
        self.assertGreater(fourth.correction_deg, 0)
        self.assertLessEqual(fourth.correction_deg, model.max_correction_deg)

    def test_small_consistent_visual_is_bounded(self):
        model = HeadingFusion(70, max_correction_deg=2)
        result = model.update(0, [80, 90, 100], [0, 7, 9])
        self.assertEqual(result.reason, "consistent_visual")
        self.assertLessEqual(abs(result.correction_deg), 2)


if __name__ == "__main__":
    unittest.main()
