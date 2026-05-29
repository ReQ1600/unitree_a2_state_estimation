"""Main iSAM2 estimator loop for legged robot state estimation.

Wires the MuJoCo bridge, IMU preintegrator, and factor registry into a
real-time incremental factor-graph solver.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import gtsam
from gtsam import ISAM2, ISAM2Params, NonlinearFactorGraph, Values, NavState

from .gtsam_types import (
    PoseKey, VelKey, BiasKey,
    navstate_to_position, navstate_to_vector3, zero_bias
)
from .imu_preintegrator import ImuPreintegrator
from .contact_preintegrator import ContactPreintegrator
from .factor_registry import FactorRegistry


class Estimator:
    """Incremental factor-graph state estimator using GTSAM iSAM2.

    Usage:
        estimator = Estimator(config, registry, preint_params)
        estimator.initialise(sensor_data)
        while running:
            sensor_data = bridge.step_and_collect()
            estimator.step(sensor_data)
            pose, vel = estimator.latest_estimate()
    """

    def __init__(self,
                 config: Dict[str, Any],
                 registry: FactorRegistry,
                 preint_params: Any) -> None:
        """
        Args:
            config:         Dict with keys: 'keyframe_every_n_steps'.
            registry:       FactorRegistry with registered factors.
            preint_params:  PreintegrationCombinedParams for IMU preintegration.
        """
        self._registry = registry
        self._preint_params = preint_params

        # solver
        isam_params = ISAM2Params()
        # relinerise graph on every update
        isam_params.relinearizeSkip = config.get('isam_relinearize_skip', 1)
        self._isam = ISAM2(isam_params)

        # states
        self._step_counter: int = 0     # total steps
        self._keyframe_idx: int = 0     # index in the graph factor
        self._keyframe_every: int = int(config.get('keyframe_every_n_steps', 100))      # how many imy steps between optimisations
        self._imu_steps_since_keyframe: int = 0     # counter to trigger keyframes

        # imu preintegrator (created fresh at each keyframe)
        # preintegrator is not created untili initilalise()
        self._pim: Optional[ImuPreintegrator] = None
        self._current_bias = zero_bias()

        self._contact_pim: Optional[ContactPreintegrator] = None
        self._contact_preint_config = config["contact_preintegration"]

        # logs
        self._estimates: List[Dict[str, Any]] = []


    def initialise(self, sensor_data: Dict[str, Any]) -> None:
        """Set up the first frame with priors."""
        # isam requires a valid bayes tree before it can perform incremental updates
        graph = NonlinearFactorGraph()
        values = Values()
        context = self._make_context()

        self._registry.add_all_priors(graph, values, sensor_data, context)
        self._isam.update(graph, values)

        # create fresh preintegrator
        self._pim = ImuPreintegrator(self._preint_params, self._current_bias)
        self._imu_steps_since_keyframe = 0
        self._keyframe_idx = 0

        self._contact_pim = ContactPreintegrator.from_config(
            self._contact_preint_config
        )

        # log initial estimate
        self._log_estimate(0, sensor_data)

    def step(self, sensor_data: Dict[str, Any]) -> None:
        """Process one simulation timestep.

        Args:
            sensor_data: Dict with at least 'imu_acc', 'imu_gyro', 'dt'.
        """
        # accumulate imu measurement
        # just feed them into the preintegrator and dgaf
        acc = np.asarray(sensor_data['imu_acc'], dtype=float).ravel()
        gyro = np.asarray(sensor_data['imu_gyro'], dtype=float).ravel()
        dt = float(sensor_data.get('dt', 0.002))

        if self._pim is not None:
            self._pim.integrate(acc, gyro, dt)

        if self._contact_pim is not None and self._pim is not None:
            foot_contacts = np.asarray(sensor_data["foot_contacts"], dtype=float).reshape(4)

            delta_rotation_ik = self._pim.delta_rotation()

            fk_contact_rotation = np.asarray(
                sensor_data.get("fk_contact_rotation", np.eye(3)),
                dtype=float,
            )

            self._contact_pim.integrate(
                contact_flags=foot_contacts,
                dt=dt,
                delta_rotation_ik=delta_rotation_ik,
                fk_contact_rotation=fk_contact_rotation,
            )

        self._imu_steps_since_keyframe += 1
        self._step_counter += 1

        # keyframe — build graph and update isam
        if self._imu_steps_since_keyframe >= self._keyframe_every:
            self._do_keyframe(sensor_data)

    def latest_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, velocity) of the latest keyframe."""

        
        result = self._isam.calculateEstimate()
        k = self._keyframe_idx
        if k > 0:
            k -= 1  # last inserted keyframe
        pk = PoseKey(k)
        vk = VelKey(k)
        pos = np.zeros(3)
        vel = np.zeros(3)
        if result.exists(pk):
            p = result.atPose3(pk)
            pos = np.array([p.x(), p.y(), p.z()])
        if result.exists(vk):
            v = result.atVector(vk)
            vel = np.array([v[0], v[1], v[2]])
        return pos, vel

    def get_log(self) -> List[Dict[str, Any]]:
        return list(self._estimates)


    def _do_keyframe(self, sensor_data: Dict[str, Any]) -> None:
        """Build a batch graph for the current keyframe and update iSAM2."""

        # incremental graph smoothing
        self._keyframe_idx += 1
        graph = NonlinearFactorGraph()
        values = Values()
        context = self._make_context()

        # provide current isam estimate so factors can predict next state
        context['current_estimate'] = self._isam.calculateEstimate()

        # all registered factors add their factors + initial estimates
        self._registry.add_all_to_graph(graph, values,
                                        self._keyframe_idx, sensor_data, context)

        # update solver
        # adds new factors to the graph, realinerses, updates only affected parts of tree
        self._isam.update(graph, values)

        # reset preintegrator for next window
        self._pim = ImuPreintegrator(self._preint_params, self._current_bias)
        self._imu_steps_since_keyframe = 0

        self._contact_pim = ContactPreintegrator.from_config(
            self._contact_preint_config
        )

        # log log log
        self._log_estimate(self._keyframe_idx, sensor_data)

    def _make_context(self) -> Dict[str, Any]:
        return {
            'pose_key': PoseKey,
            'vel_key': VelKey,
            'bias_key': BiasKey,
            'pim': self._pim,
            'contact_pim': self._contact_pim,
            'initial_contact_points': np.zeros((4, 3)),
        }

    def _log_estimate(self, kf_idx: int, sensor_data: Dict[str, Any]) -> None:
        est_pos, est_vel = self.latest_estimate()
        gt_pos = sensor_data.get('base_pos', np.zeros(3))
        log = {
            'keyframe': kf_idx,
            'step': self._step_counter,
            'est_pos': est_pos,
            'est_vel': est_vel,
            'gt_pos': np.asarray(gt_pos, dtype=float).ravel(),
        }
        self._estimates.append(log)
