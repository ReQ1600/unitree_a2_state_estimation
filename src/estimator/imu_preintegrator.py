"""IMU preintegration wrapper around GTSAM's PreintegratedCombinedMeasurements.

Usage pattern:
    params = PreintegrationCombinedParams.MakeSharedU(9.81)
    params.setAccelerometerCovariance(...)
    params.setGyroscopeCovariance(...)
    params.setBiasAccCovariance(...)
    params.setBiasOmegaCovariance(...)

    pim = ImuPreintegrator(params, bias_init)

    # In loop:
    pim.integrate(acc, gyro, dt)
    # At keyframe:
    delta = pim.delta()          # returns dict of accumulated delta values
    predicted_navstate = pim.predict(current_navstate, current_bias)
    pim.reset(new_bias)
"""

# mathematical (@mikołaj) engine of the estimator
# wraps gtsam preintegration into pythong api

from typing import Dict
import numpy as np
from gtsam import PreintegrationCombinedParams, PreintegratedCombinedMeasurements, NavState
from gtsam.imuBias import ConstantBias


class ImuPreintegrator:
    """Thin wrapper around GTSAM.PreintegratedCombinedMeasurements.

    Accumulates IMU measurements between keyframes and provides
    preintegrated deltas for factor construction.
    """

    def __init__(self, params: PreintegrationCombinedParams, bias: ConstantBias) -> None:
        """Create a fresh preintegrator.

        Args:
            params:  PreintegrationCombinedParams (gravity, noise covariances).
            bias:    Initial IMU bias estimate (ConstantBias).
        """
        self._params = params
        # gtsam preintegration buffer
        # accumulates acc, gyro, dt and updates 15x15 cov matrix in real time
        self._pim = PreintegratedCombinedMeasurements(params, bias)
        self._bias = bias


    def integrate(self,
                  acc_measured: np.ndarray,
                  gyro_measured: np.ndarray,
                  dt: float) -> None:
        """Integrate one IMU measurement.

        Args:
            acc_measured:  (3,)  accelerometer reading in sensor frame [m/s²].
            gyro_measured: (3,)  gyroscope reading in sensor frame [rad/s].
            dt:            Time since last measurement [s].
        """

        self._pim.integrateMeasurement(
            # gtsam expects python lists
            acc_measured.ravel().tolist(),
            gyro_measured.ravel().tolist(),
            float(dt)
        )


    # represents uncertainty in 15x15 covariance (p, R, v, b_acc, b_gyro)
    # return numpy to make plotable
    def delta_rotation(self) -> np.ndarray:
        """Accumulated rotation increment (3x3 rotation matrix, numpy)."""
        return np.array(self._pim.deltaRij().matrix())

    def delta_position(self) -> np.ndarray:
        """Accumulated position increment [m] (world-frame, 3,)."""
        dp = self._pim.deltaPij()
        return np.array([dp[0], dp[1], dp[2]])

    def delta_velocity(self) -> np.ndarray:
        """Accumulated velocity increment [m/s] (world-frame, 3,)."""
        dv = self._pim.deltaVij()
        return np.array([dv[0], dv[1], dv[2]])

    def delta(self) -> Dict[str, np.ndarray]:
        """Return all accumulated deltas as a dict."""
        return {
            'dR': self.delta_rotation(),
            'dp': self.delta_position(),
            'dv': self.delta_velocity(),
            'cov': self.preintegration_covariance()
        }

    def preintegration_covariance(self) -> np.ndarray:
        """Return the 15x15 preintegrated covariance matrix."""
        cov = self._pim.preintMeasCov()
        return np.array(cov)


    def predict(self,
                navstate_from: NavState,
                bias_from: ConstantBias) -> NavState:
        """Predict the next NavState given the starting state and bias.

        Args:
            navstate_from: Starting navigation state (pose + velocity).
            bias_from:     IMU bias at the starting keyframe.

        Returns:
            Predicted NavState at the end of the preintegration window.
        """
        # isam needs a good guess at step_idx before optimising
        return self._pim.predict(navstate_from, bias_from)

    def reset(self, new_bias: ConstantBias) -> None:
        """Reset the preintegrator for the next keyframe interval."""
        self._bias = new_bias
        self._pim.resetIntegrationAndSetBias(new_bias)

    def bias(self) -> ConstantBias:
        """Return the current bias estimate stored in the preintegrator."""
        return self._bias


    @staticmethod
    def make_params(accel_noise_density: float = 0.01,
                    gyro_noise_density: float = 0.001,
                    accel_bias_rw: float = 0.0001,
                    gyro_bias_rw: float = 0.00001,
                    gravity: float = 9.81) -> PreintegrationCombinedParams:
        """Create PreintegrationCombinedParams with default noise values.

        Noise densities follow continuous-time convention (units / sqrt(Hz)):
          accel_noise_density   [m/s² / sqrt(Hz)]
          gyro_noise_density    [rad/s / sqrt(Hz)]
          accel_bias_rw         [m/s³ / sqrt(Hz)] (bias random walk)
          gyro_bias_rw          [rad/s² / sqrt(Hz)] (bias random walk)
        """
        # noise config yeah
        params = PreintegrationCombinedParams.MakeSharedU(gravity)
        I3 = np.eye(3)
        params.setAccelerometerCovariance(I3 * (accel_noise_density ** 2))
        params.setGyroscopeCovariance(I3 * (gyro_noise_density ** 2))
        params.setBiasAccCovariance(I3 * (accel_bias_rw ** 2))
        params.setBiasOmegaCovariance(I3 * (gyro_bias_rw ** 2))
        params.setIntegrationCovariance(I3 * 1e-8)
        return params
