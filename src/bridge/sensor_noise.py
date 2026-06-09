"""IMU noise generator: white noise + bias random walk."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ImuNoiseParams:
    acc_white_density: float
    gyro_white_density: float
    acc_bias_density: float
    gyro_bias_density: float


class ImuNoiseGenerator:
    def __init__(self, params: ImuNoiseParams):
        self.p = params
        self.acc_bias = np.zeros(3, dtype=float)
        self.gyro_bias = np.zeros(3, dtype=float)

    def corrupt(self, acc_true: np.ndarray, gyro_true: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (acc_noisy, gyro_noisy).

        White noise sigma for discrete time step dt is density / sqrt(dt).
        Bias random walk increment sigma is bias_density * sqrt(dt).
        """
        dt = float(dt)
        # white noise
        sigma_acc = self.p.acc_white_density / np.sqrt(dt)
        sigma_gyro = self.p.gyro_white_density / np.sqrt(dt)
        white_acc = np.random.normal(0.0, sigma_acc, 3)
        white_gyro = np.random.normal(0.0, sigma_gyro, 3)

        # bias random walk (discrete integration)
        bias_acc_sigma = self.p.acc_bias_density * np.sqrt(dt)
        bias_gyro_sigma = self.p.gyro_bias_density * np.sqrt(dt)
        self.acc_bias += np.random.normal(0.0, bias_acc_sigma, 3)
        self.gyro_bias += np.random.normal(0.0, bias_gyro_sigma, 3)

        acc_noisy = acc_true + white_acc + self.acc_bias
        gyro_noisy = gyro_true + white_gyro + self.gyro_bias
        return acc_noisy, gyro_noisy
    
@dataclass
class JointNoiseParams:
    joint_white_density: float
    joint_bias_density: float


class JointNoiseGenerator:
    def __init__(self, params: JointNoiseParams):
        self.p = params
        self.joint_bias = np.zeros((4, 3), dtype=float)

    def corrupt(self, joint_true: np.ndarray, dt: float) -> np.ndarray:
        """Return (joint_noisy).

        White noise sigma for discrete time step dt is density / sqrt(dt).
        Bias random walk increment sigma is bias_density * sqrt(dt).
        """
        dt = float(dt)
        # white noise
        sigma_joint = self.p.joint_white_density / np.sqrt(dt)
        white_joint = np.random.normal(0.0, sigma_joint, (4, 3))

        # bias random walk (discrete integration)
        bias_joint_sigma = self.p.joint_bias_density * np.sqrt(dt)
        self.joint_bias += np.random.normal(0.0, bias_joint_sigma, (4, 3))

        joint_noisy = joint_true + white_joint + self.joint_bias
        return joint_noisy
    
