"""Integrate high-rate z-axis gyro samples between camera frame times.

Caller must convert the IMU's sign convention and units to degrees/second.
Camera and gyro timestamps must share one clock.
"""

from __future__ import annotations

import numpy as np


def gyro_deltas_at_frames(gyro_time_s, yaw_rate_deg_s, frame_time_s) -> np.ndarray:
    gyro_t = np.asarray(gyro_time_s, dtype=float).reshape(-1)
    rate = np.asarray(yaw_rate_deg_s, dtype=float).reshape(-1)
    frame_t = np.asarray(frame_time_s, dtype=float).reshape(-1)
    if len(gyro_t) < 2 or len(frame_t) < 1 or len(gyro_t) != len(rate):
        raise ValueError("need >=2 paired gyro samples and >=1 camera frame")
    if not all(np.all(np.isfinite(x)) for x in (gyro_t, rate, frame_t)):
        raise ValueError("timestamps and gyro rates must be finite")
    if np.any(np.diff(gyro_t) <= 0) or np.any(np.diff(frame_t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if frame_t[0] < gyro_t[0] or frame_t[-1] > gyro_t[-1]:
        raise ValueError("gyro samples must cover every camera frame")

    deltas = np.zeros(len(frame_t), dtype=float)
    for i in range(1, len(frame_t)):
        left, right = frame_t[i - 1], frame_t[i]
        interior = gyro_t[(gyro_t > left) & (gyro_t < right)]
        times = np.concatenate(([left], interior, [right]))
        rates = np.interp(times, gyro_t, rate)
        deltas[i] = float(np.trapezoid(rates, times))
    return deltas
