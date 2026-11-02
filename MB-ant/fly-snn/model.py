"""Minimal fully spiking modules for a synthetic heading-selection task.

The design borrows *questions* from Ardin (sparse KC memory), Stentiford
(spiking CX compass), and Pires (goal/heading comparison). It does not claim
to reproduce their neuron parameters, anatomy, or published scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    sectors: int = 16
    cue_bits: int = 48
    kenyon_cells: int = 512
    kc_connections: int = 6
    kc_threshold: int = 4
    memory_ticks: int = 8
    ring_ticks: int = 8
    episode_steps: int = 24
    train_repeats: int = 5
    contexts: int = 8
    learning_rate: float = 0.14
    cue_flip_probability: float = 0.08
    landmark_dropout: float = 0.35
    landmark_noise_probability: float = 0.10
    turn_noise_probability: float = 0.10

    def validate(self) -> None:
        if self.sectors < 8 or self.sectors % 2:
            raise ValueError("sectors must be even and >= 8")
        if not 1 <= self.kc_threshold <= self.kc_connections <= self.cue_bits:
            raise ValueError("invalid KC connectivity or threshold")
        if min(self.contexts, self.episode_steps, self.memory_ticks,
               self.ring_ticks, self.train_repeats) < 1:
            raise ValueError("counts must be positive")
        if self.episode_steps < 2:
            raise ValueError("episode_steps must include initialization and test steps")
        for value in (self.cue_flip_probability, self.landmark_dropout,
                      self.landmark_noise_probability, self.turn_noise_probability):
            if not 0 <= value <= 1:
                raise ValueError("probabilities must be in [0, 1]")

    def to_dict(self) -> dict:
        return asdict(self)


def circular_error(predicted: int, target: int, sectors: int) -> int:
    """Unsigned sector error, with the wrap at 0/360 degrees handled."""
    return int(abs((int(predicted) - int(target) + sectors // 2) % sectors
                   - sectors // 2))


def decode_spikes(counts: np.ndarray) -> tuple[int, bool]:
    """Population winner and explicit tie status. First-index tie rule."""
    maximum = int(np.max(counts))
    return int(np.argmax(counts)), bool(maximum == 0 or np.count_nonzero(counts == maximum) > 1)


class ContextMemorySNN:
    """Binary input spikes -> sparse KC spikes -> plastic goal output spikes.

    Training is reward-gated potentiation for the teacher goal and mild
    depression for competing goals. A shared inhibitory term implements output
    competition. This is Ardin-inspired sparse coding, not Ardin's EN circuit.
    """

    def __init__(self, cfg: Config, seed: int):
        cfg.validate()
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.connections = np.zeros((cfg.kenyon_cells, cfg.cue_bits), dtype=np.int8)
        for row in self.connections:
            row[rng.choice(cfg.cue_bits, cfg.kc_connections, replace=False)] = 1
        self.weights = np.zeros((cfg.sectors, cfg.kenyon_cells), dtype=np.float32)

    def kc_spikes(self, cue: np.ndarray) -> np.ndarray:
        if cue.shape != (self.cfg.cue_bits,):
            raise ValueError("cue shape mismatch")
        return (self.connections @ cue.astype(np.int8) >= self.cfg.kc_threshold)

    def train(self, cue: np.ndarray, goal: int) -> int:
        active = self.kc_spikes(cue)
        if not 0 <= goal < self.cfg.sectors:
            raise ValueError("goal outside ring")
        if np.any(active):
            self.weights[goal, active] += self.cfg.learning_rate
            self.weights[:, active] -= 0.02 * self.cfg.learning_rate
            self.weights[goal, active] += 0.02 * self.cfg.learning_rate
            np.maximum(self.weights, 0, out=self.weights)
        return int(active.sum())

    def recall(self, cue: np.ndarray) -> tuple[np.ndarray, int]:
        """Run output LIF units; return their spike counts and KC activity."""
        active = self.kc_spikes(cue)
        counts = np.zeros(self.cfg.sectors, dtype=np.int32)
        voltage = np.zeros(self.cfg.sectors, dtype=np.float32)
        current = self.weights[:, active].sum(axis=1)
        if np.any(active):
            current = current / max(1, int(active.sum())) * 12.0
        for _ in range(self.cfg.memory_ticks):
            voltage = 0.55 * voltage + current - 0.12 * counts.sum()
            spikes = voltage >= 0.8
            counts += spikes
            voltage[spikes] = 0
        return counts, int(active.sum())


class DirectMemorySNN(ContextMemorySNN):
    """Matched learning/readout with input spikes connected directly to goals."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.connections = np.eye(cfg.cue_bits, dtype=np.int8)
        self.weights = np.zeros((cfg.sectors, cfg.cue_bits), dtype=np.float32)

    def kc_spikes(self, cue: np.ndarray) -> np.ndarray:
        if cue.shape != (self.cfg.cue_bits,):
            raise ValueError("cue shape mismatch")
        return cue.astype(bool)


class BudgetMatchedDirectMemorySNN(ContextMemorySNN):
    """Direct cue-to-goal synapses with the MB output-weight budget.

    Repeated cue channels carry no new information. This is a parameter-count
    control, not an independently expressive architecture or a compute match.
    """

    def __init__(self, cfg: Config, seed: int):
        cfg.validate()
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        self.input_index = np.arange(cfg.kenyon_cells) % cfg.cue_bits
        rng.shuffle(self.input_index)
        self.weights = np.zeros((cfg.sectors, cfg.kenyon_cells), dtype=np.float32)

    def kc_spikes(self, cue: np.ndarray) -> np.ndarray:
        if cue.shape != (self.cfg.cue_bits,):
            raise ValueError("cue shape mismatch")
        return cue[self.input_index].astype(bool)


class HeadingRingSNN:
    """A small E-PG-like spiking ring with velocity-gated recurrent synapses.

    Each update provides a relative turn (-1, 0, +1) and optional landmark
    population spikes. The ring's own previous spikes, routed through fixed
    clockwise/counterclockwise matrices, supply the motion prediction. No
    ground-truth heading is passed to update().
    """

    def __init__(self, cfg: Config):
        cfg.validate()
        self.cfg = cfg
        self.voltage = np.zeros(cfg.sectors, dtype=np.float32)
        self.previous = np.zeros(cfg.sectors, dtype=np.float32)
        self.last_counts = np.zeros(cfg.sectors, dtype=np.int32)
        self.initialized = False

    def reset(self) -> None:
        self.voltage.fill(0)
        self.previous.fill(0)
        self.last_counts.fill(0)
        self.initialized = False

    def update(self, turn: int, landmark: int | None) -> np.ndarray:
        if turn not in (-1, 0, 1):
            raise ValueError("turn must be -1, 0 or +1 sector")
        if landmark is not None and not 0 <= landmark < self.cfg.sectors:
            raise ValueError("landmark outside ring")
        if not self.initialized and landmark is None:
            return self.last_counts.copy()
        if not self.initialized:
            self.previous[landmark] = 1
            self.initialized = True

        # A velocity channel selects one of three fixed recurrent projections.
        # Applying it once per frame prevents one turn from shifting the bump
        # repeatedly during the internal neural ticks.
        predicted = np.roll(self.previous, turn)
        cue = np.zeros(self.cfg.sectors, dtype=np.float32)
        if landmark is not None:
            cue[landmark] = 1.0
        counts = np.zeros(self.cfg.sectors, dtype=np.int32)
        voltage = np.zeros(self.cfg.sectors, dtype=np.float32)
        spikes = predicted.copy()
        for _ in range(self.cfg.ring_ticks):
            local = 0.3 * spikes + 0.05 * (np.roll(spikes, 1) + np.roll(spikes, -1))
            inhibition = 0.2 * spikes.sum()
            voltage = 0.45 * voltage + 1.15 * predicted + local + 1.6 * cue - inhibition
            fired = voltage >= 1.0
            if np.count_nonzero(fired) > 1:
                # Fast global inhibition: only the strongest cell gets to
                # complete a spike in this tick. Index order breaks exact ties.
                winner = int(np.argmax(voltage))
                fired.fill(False)
                fired[winner] = True
            counts += fired
            voltage[fired] = 0
            spikes = fired.astype(np.float32)
        # Feedback remains a neural population, including when a cue vanishes.
        self.previous = counts.astype(np.float32) / max(1, int(counts.max()))
        self.last_counts = counts
        return counts.copy()


class SteeringSNN:
    """Spike-coincidence goal/heading comparison with left/right output cells.

    The fixed sin-shaped synapses are a minimal PFL-like hypothesis; they are
    not fitted or claimed to reproduce the fly PFL3 connectome.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        angle = 2 * np.pi * np.arange(cfg.sectors) / cfg.sectors
        difference = angle[:, None] - angle[None, :]
        self.left = np.maximum(np.sin(difference), 0)
        self.right = np.maximum(-np.sin(difference), 0)

    def decide(self, goal_counts: np.ndarray, heading_counts: np.ndarray) -> tuple[int, tuple[int, int]]:
        if goal_counts.shape != (self.cfg.sectors,) or heading_counts.shape != (self.cfg.sectors,):
            raise ValueError("population shape mismatch")
        if goal_counts.max() == 0 or heading_counts.max() == 0:
            return 0, (0, 0)
        g = goal_counts / goal_counts.sum()
        h = heading_counts / heading_counts.sum()
        drives = np.array([g @ self.left @ h, g @ self.right @ h])
        voltage = np.zeros(2)
        counts = np.zeros(2, dtype=np.int32)
        for _ in range(8):
            voltage = 0.5 * voltage + 3 * drives - 0.2 * counts[::-1]
            spikes = voltage >= 0.8
            counts += spikes
            voltage[spikes] = 0
        decision = int(np.sign(counts[0] - counts[1]))
        return decision, (int(counts[0]), int(counts[1]))
