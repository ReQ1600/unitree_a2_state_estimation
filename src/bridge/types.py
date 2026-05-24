# src/bridge/types.py
"""Immutable dataclass for simulator snapshot at a single timestep."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SimMeasurement:
    """Snapshot of simulator state at one timestep.

    All arrays are fresh copies to prevent reference aliasing.
    """
    timestamp: float

    # shape (3,) m/s^2, IMU/base frame
    imu_acc: np.ndarray

    # shape (3,) rad/s, IMU/base frame
    imu_gyro: np.ndarray

    # shape (3,), world frame
    base_pos: np.ndarray

    # shape (4,), (w, x, y, z)
    base_quat: np.ndarray

    # shape (12,)
    joint_pos: np.ndarray

    # shape (12,)
    joint_vel: np.ndarray

    # shape (4,), binary, [FL, FR, RL, RR]
    foot_contacts: np.ndarray
