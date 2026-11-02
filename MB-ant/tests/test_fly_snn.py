"""Tests for scientific invariants in the synthetic SNN baseline."""

import unittest
from dataclasses import replace

import numpy as np

from fly_snn.benchmark import reference_turn, run_seed, summarize
from fly_snn.comparison import ARMS, architecture
from fly_snn.model import (Config, ContextMemorySNN, HeadingRingSNN,
                           SteeringSNN, circular_error, decode_spikes)


class FlySNNBaselineTests(unittest.TestCase):
    def test_ring_wraps_and_integrates_turn_without_landmark(self):
        cfg = Config()
        ring = HeadingRingSNN(cfg)
        first, tie = decode_spikes(ring.update(0, cfg.sectors - 1))
        self.assertEqual((first, tie), (cfg.sectors - 1, False))
        second, tie = decode_spikes(ring.update(1, None))
        self.assertEqual((second, tie), (0, False))
        third, tie = decode_spikes(ring.update(-1, None))
        self.assertEqual((third, tie), (cfg.sectors - 1, False))
        self.assertEqual(circular_error(0, cfg.sectors - 1, cfg.sectors), 1)

    def test_mb_learning_changes_goal_spikes_and_spares_untrained_weights(self):
        cfg = Config()
        memory = ContextMemorySNN(cfg, seed=4)
        cue = np.ones(cfg.cue_bits, dtype=np.int8)
        before, active = memory.recall(cue)
        self.assertEqual(int(before.sum()), 0)
        self.assertGreater(active, 0)
        for _ in range(cfg.train_repeats):
            memory.train(cue, 3)
        after, _ = memory.recall(cue)
        self.assertEqual(decode_spikes(after)[0], 3)
        self.assertGreater(int(after[3]), 0)
        self.assertTrue(np.all(memory.weights >= 0))

    def test_steering_sign_and_opposite_ambiguity(self):
        cfg = Config()
        steering = SteeringSNN(cfg)
        heading = np.zeros(cfg.sectors, dtype=np.int32)
        heading[0] = 8
        goal = np.zeros_like(heading)
        goal[2] = 8
        self.assertEqual(steering.decide(goal, heading)[0], 1)
        goal.fill(0)
        goal[-2] = 8
        self.assertEqual(steering.decide(goal, heading)[0], -1)
        self.assertIsNone(reference_turn(cfg.sectors // 2, 0, cfg.sectors))

    def test_paired_arms_reproducible_and_no_heading_label_in_ring(self):
        cfg = Config(contexts=3, episode_steps=6)
        rows = run_seed(cfg, 73)
        self.assertEqual(rows, run_seed(cfg, 73))
        self.assertEqual(len(rows), 3 * 6 * 4)
        for offset in range(0, len(rows), 4):
            block = rows[offset:offset + 4]
            self.assertEqual({r["arm"] for r in block},
                             {"cx_fixed", "direct_cx", "mb_cx", "mb_shuffled"})
            self.assertEqual(len({r["heading_prediction"] for r in block}), 1)
            self.assertEqual(len({r["true_heading"] for r in block}), 1)
            self.assertEqual(len({r["landmark_cue"] for r in block}), 1)
        summaries = summarize(rows)
        self.assertEqual(len(summaries), 4)
        self.assertTrue(all(s["n"] == 15 for s in summaries))

    def test_noise_conditions_share_underlying_trajectory(self):
        base = Config(contexts=3, episode_steps=7)
        clean = run_seed(replace(base, cue_flip_probability=0,
                                 landmark_dropout=0, landmark_noise_probability=0,
                                 turn_noise_probability=0), 91)
        noisy = run_seed(replace(base, cue_flip_probability=0.3,
                                 landmark_dropout=0.8, landmark_noise_probability=0.2,
                                 turn_noise_probability=0.2), 91)
        for a, b in zip(clean, noisy):
            self.assertEqual((a["context"], a["step"], a["true_heading"], a["goal"]),
                             (b["context"], b["step"], b["true_heading"], b["goal"]))

    def test_four_family_comparison_is_paired_and_budget_is_explicit(self):
        cfg = Config(contexts=2, episode_steps=4)
        rows = run_seed(cfg, 17, comparison=True)
        self.assertEqual(rows, run_seed(cfg, 17, comparison=True))
        self.assertEqual(len(rows), 2 * 4 * len(ARMS))
        for offset in range(0, len(rows), len(ARMS)):
            block = rows[offset:offset + len(ARMS)]
            self.assertEqual({r["arm"] for r in block}, set(ARMS))
            self.assertEqual(len({r["heading_prediction"] for r in block}), 1)
            self.assertEqual(len({r["true_heading"] for r in block}), 1)
        self.assertEqual(architecture(cfg, "direct_budget_cx")["trainable_synapses"],
                         architecture(cfg, "mb_sparse_t4_cx")["trainable_synapses"])
        self.assertLess(architecture(cfg, "direct_cx")["trainable_synapses"],
                        architecture(cfg, "mb_sparse_t4_cx")["trainable_synapses"])


if __name__ == "__main__":
    unittest.main()
