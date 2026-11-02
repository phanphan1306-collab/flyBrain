"""Minimal circular heading state with a gated visual correction.

Input `visual_scores` is one familiarity score per scanned heading sector;
smaller values mean more familiar. Legacy MB_ant obtains this by scanning one
EN over angles. It does not contain separate sector-specific MBON neurons.
Angles and gyro deltas use degrees. This is a computational CX-inspired state,
not a biophysical E-PG/P-EN ring simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Update:
    predicted_deg: float
    heading_deg: float
    visual_deg: float | None
    confidence: float
    innovation_deg: float | None
    correction_deg: float
    reason: str


class HeadingFusion:
    def __init__(self, heading_deg: float, *, temperature: float = 2.0,
                 min_confidence: float = 0.08, direct_gate_deg: float = 35.0,
                 repeat_tolerance_deg: float = 12.0, repeat_count: int = 3,
                 max_correction_deg: float = 8.0, visual_gain: float = 0.5,
                 confidence_reference: float = 0.1):
        if temperature <= 0 or not 0 <= min_confidence <= 1 or confidence_reference <= 0:
            raise ValueError("invalid confidence parameters")
        if direct_gate_deg <= 0 or repeat_count < 2 or max_correction_deg <= 0:
            raise ValueError("invalid gate parameters")
        if not 0 <= visual_gain <= 1:
            raise ValueError("visual_gain must be in [0, 1]")
        self.heading_deg = wrap(heading_deg)
        self.temperature = temperature
        self.min_confidence = min_confidence
        self.direct_gate_deg = direct_gate_deg
        self.repeat_tolerance_deg = repeat_tolerance_deg
        self.repeat_count = repeat_count
        self.max_correction_deg = max_correction_deg
        self.visual_gain = visual_gain
        self.confidence_reference = confidence_reference
        self.pending_deg: float | None = None
        self.pending_count = 0

    def update(self, gyro_delta_deg: float, sector_deg=None,
               visual_scores=None, *, context_ok: bool = False) -> Update:
        if not np.isfinite(gyro_delta_deg):
            raise ValueError("nonfinite gyro delta")
        predicted = wrap(self.heading_deg + gyro_delta_deg)
        self.heading_deg = predicted
        if sector_deg is None or visual_scores is None:
            self.pending_deg, self.pending_count = None, 0
            return Update(predicted, predicted, None, 0.0, None, 0.0, "gyro_only")

        angles = np.asarray(sector_deg, dtype=float).reshape(-1)
        scores = np.asarray(visual_scores, dtype=float).reshape(-1)
        if angles.size != scores.size or angles.size < 3:
            raise ValueError("need >=3 matching sector angles and scores")
        if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(scores)):
            raise ValueError("nonfinite visual observation")
        order = np.argsort(scores)
        best = int(order[0])
        visual = wrap(angles[best])
        innovation = wrap(visual - predicted)

        # Both a unique best sector and a concentrated score distribution are
        # required. Equal minima yield zero confidence regardless of level.
        gap = scores[order[1]] - scores[best]
        weights = np.exp(-(scores - scores[best]) / self.temperature)
        probs = weights / weights.sum()
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-300)))
        concentration = 1.0 - entropy / np.log(len(scores))
        confidence = float(np.clip((gap / (gap + self.temperature)) * concentration, 0, 1))
        if confidence < self.min_confidence:
            self.pending_deg, self.pending_count = None, 0
            return Update(predicted, predicted, visual, confidence, innovation, 0.0,
                          "ambiguous_visual")

        if abs(innovation) <= self.direct_gate_deg:
            self.pending_deg, self.pending_count = None, 0
            reason = "consistent_visual"
        else:
            # Compare to the last visual heading advanced by this gyro step.
            if self.pending_deg is not None and abs(wrap(visual - self.pending_deg - gyro_delta_deg)) <= self.repeat_tolerance_deg:
                self.pending_count += 1
            else:
                self.pending_count = 1
            self.pending_deg = visual
            if self.pending_count < self.repeat_count:
                return Update(predicted, predicted, visual, confidence, innovation, 0.0,
                              "large_innovation_waiting_for_repeat")
            if not context_ok:
                return Update(predicted, predicted, visual, confidence, innovation, 0.0,
                              "large_innovation_requires_context")
            reason = "repeated_large_innovation"

        relative_confidence = min(1.0, confidence / self.confidence_reference)
        correction = float(np.clip(self.visual_gain * relative_confidence * innovation,
                                   -self.max_correction_deg, self.max_correction_deg))
        self.heading_deg = wrap(predicted + correction)
        return Update(predicted, self.heading_deg, visual, confidence, innovation,
                      correction, reason)
