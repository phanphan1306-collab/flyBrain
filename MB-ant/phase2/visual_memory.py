"""A lightweight sparse-KC visual baseline for new panorama scans.

This is a non-spiking computational proxy, not the MATLAB PN/KC/EN network.
Train only on a route's learning images; evaluate on separate observations.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_proxy_panorama(image: np.ndarray) -> np.ndarray:
    """Resize to 10x36 grayscale, invert intensity, and L2-normalize.

    This deliberately does not claim to reproduce MATLAB `adapthisteq`.
    Use the same transform at learning and query time.
    """
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")
    if array.dtype != np.uint8:
        raise ValueError("image must be uint8")
    resized = Image.fromarray(array).convert("L").resize((36, 10), Image.Resampling.BILINEAR)
    vector = 1.0 - np.asarray(resized, dtype=float).reshape(-1) / 255.0
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("all-white image has zero inverted norm")
    return vector / norm


class SparseKCEncoder:
    def __init__(self, *, num_pn: int = 360, num_kc: int = 20000,
                 fan_in: int = 10, active_fraction: float = 0.0127,
                 seed: int = 0):
        if num_pn < 2 or num_kc < 2 or not 1 <= fan_in <= num_pn:
            raise ValueError("invalid PN/KC architecture")
        if not 0 < active_fraction < 1:
            raise ValueError("active_fraction must be in (0, 1)")
        self.num_pn = num_pn
        self.num_kc = num_kc
        self.active_count = max(1, round(num_kc * active_fraction))
        rng = np.random.default_rng(seed)
        self.connections = np.vstack(
            [rng.choice(num_pn, fan_in, replace=False) for _ in range(num_kc)]
        )

    def encode(self, pn_vector: np.ndarray) -> np.ndarray:
        values = np.asarray(pn_vector, dtype=float).reshape(-1)
        if values.size != self.num_pn or not np.all(np.isfinite(values)):
            raise ValueError(f"expected {self.num_pn} finite PN values")
        activation = values[self.connections].mean(axis=1)
        # Stable tie-break by KC index so repeated images have identical codes.
        winners = np.lexsort((np.arange(self.num_kc), -activation))[:self.active_count]
        code = np.zeros(self.num_kc, dtype=bool)
        code[winners] = True
        return code


class FamiliarityMemory:
    def __init__(self, encoder: SparseKCEncoder):
        self.encoder = encoder
        self.codes: np.ndarray | None = None

    def fit(self, route_pn_vectors: np.ndarray) -> "FamiliarityMemory":
        vectors = np.asarray(route_pn_vectors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != self.encoder.num_pn or len(vectors) < 1:
            raise ValueError("route vectors must be N x num_pn")
        self.codes = np.stack([self.encoder.encode(vector) for vector in vectors])
        return self

    def score(self, query_pn_vector: np.ndarray) -> tuple[float, int]:
        if self.codes is None:
            raise ValueError("memory has not been fitted")
        query = self.encoder.encode(query_pn_vector)
        overlaps = self.codes.astype(np.int32) @ query.astype(np.int32)
        best = int(np.argmax(overlaps))
        # All codes have the same active count; 0 means identical, 1 no overlap.
        familiarity_distance = 1.0 - float(overlaps[best]) / self.encoder.active_count
        return familiarity_distance, best

    def score_sectors(self, candidate_pn_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = np.asarray(candidate_pn_vectors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != self.encoder.num_pn or len(vectors) < 3:
            raise ValueError("candidate vectors must be S x num_pn, S>=3")
        result = [self.score(vector) for vector in vectors]
        return (np.array([item[0] for item in result], dtype=float),
                np.array([item[1] for item in result], dtype=int))
