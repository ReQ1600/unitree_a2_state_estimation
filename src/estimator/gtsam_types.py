"""GTSAM type aliases and deterministic key generators for the factor graph.

Key numbering scheme (avoids collisions between different variable types):
  PoseKey(i)   = i         (gtsam.Pose3 / gtsam.NavState)
  VelKey(i)    = 10000 + i (gtsam.Vector, 3D velocity)
  BiasKey(i)   = 20000 + i (gtsam.imuBias.ConstantBias)

Utility functions:
  make_navstate(pos, quat_xyzw, vel)  -> gtsam.NavState
  navstate_to_pose3(ns)               -> gtsam.Pose3
  navstate_to_vector3(ns)             -> numpy (3,) velocity

Relies on gtsam (>=4.2) for PreintegratedCombinedMeasurements, CombinedImuFactor,
NavState, Pose3, Rot3, Point3, etc.
"""
# basically just translate between numpy/sensor and gtsam

import numpy as np
from typing import Tuple
import gtsam
from gtsam import Pose3, Rot3, Point3, NavState
from gtsam.imuBias import ConstantBias

# gtsam stores variables in key-value structures
# this guarantees global key uniquness 

def PoseKey(i: int) -> int:
    """Deterministic key for the robot pose (NavState) at timestep i."""
    return i


def VelKey(i: int) -> int:
    """Deterministic key for the robot velocity (Vector3) at timestep i."""
    return 10000 + i


def BiasKey(i: int) -> int:
    """Deterministic key for the IMU bias (ConstantBias) at timestep i."""
    return 20000 + i


# mujoco uses [xyzw], gtsam wants [wxyz] smh
# NavState = (pose3, vector3) 
def make_navstate(position: np.ndarray,
                  quaternion_xyzw: np.ndarray,
                  velocity: np.ndarray) -> NavState:
    """Build a GTSAM NavState from numpy arrays.

    Args:
        position:       (3,)  world-frame position [x, y, z].
        quaternion_xyzw: (4,) orientation quaternion [x, y, z, w].
        velocity:       (3,)  world-frame linear velocity.

    Returns:
        gtsam.NavState.
    """
    rot = Rot3.Quaternion(quaternion_xyzw[3],  # w
                          quaternion_xyzw[0],  # x
                          quaternion_xyzw[1],  # y
                          quaternion_xyzw[2])  # z
    pose = Pose3(rot, Point3(*position))
    return NavState(pose, velocity)


# gtsam types arent plottable by default
def navstate_to_pose3(ns: NavState) -> Pose3:
    """Extract Pose3 from a NavState."""
    return ns.pose()


def navstate_to_vector3(ns: NavState) -> np.ndarray:
    """Extract 3D velocity (numpy) from a NavState."""
    v = ns.velocity()
    return np.array([v[0], v[1], v[2]], dtype=float)


def navstate_to_position(ns: NavState) -> np.ndarray:
    """Extract 3D position (numpy) from a NavState."""
    p = ns.position()
    return np.array([p[0], p[1], p[2]], dtype=float)


# uses rpy to be readavle
def initial_navstate(position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> NavState:
    """Convenience: create initial NavState from RPY angles (rad) and position/velocity."""
    rot = Rot3.RzRyRx(*rpy)
    pose = Pose3(rot, Point3(*position))
    return NavState(pose, velocity)

# match the paper's assumption that bias starts near 0
def zero_bias() -> ConstantBias:
    """Return a zero ConstantBias (accel + gyro bias = 0)."""
    return ConstantBias()