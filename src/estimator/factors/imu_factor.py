"""IMU factor using GTSAM's CombinedImuFactor with preintegrated measurements."""

from typing import Any, Dict, List
import numpy as np
import gtsam
from gtsam import (
    CombinedImuFactor, PriorFactorPose3, PriorFactorVector, NavState, Pose3, Rot3, Point3, noiseModel
)
from gtsam.imuBias import ConstantBias

from ..factor_registry import BaseFactor
from ..gtsam_types import PoseKey, VelKey, BiasKey, make_navstate, zero_bias


class ImuFactorWrapper(BaseFactor):
    """Connects consecutive robot states via IMU preintegration.

    At step 0: adds priors on pose, velocity, and IMU bias.
    At step i>0: adds CombinedImuFactor between state i-1 and state i using
    the preintegrated measurements stored in context['pim'].
    """

    def __init__(self,
                 prior_pose_sigma: float = 0.001,
                 prior_vel_sigma: float = 0.01,
                 prior_bias_sigma: float = 0.1) -> None:
        
        # if these are too large, graph drifts immediately, increase conservatively
        # add some noise to the priors
        self._prior_pose_noise = noiseModel.Diagonal.Sigmas(np.array([
            prior_pose_sigma, prior_pose_sigma, prior_pose_sigma,  # translation
            prior_pose_sigma, prior_pose_sigma, prior_pose_sigma,  # rotation
        ]))
        self._prior_vel_noise = noiseModel.Diagonal.Sigmas(np.array([
            prior_vel_sigma, prior_vel_sigma, prior_vel_sigma
        ]))
        self._prior_bias_noise = noiseModel.Diagonal.Sigmas(np.array([
            prior_bias_sigma, prior_bias_sigma, prior_bias_sigma,  # accel bias
            prior_bias_sigma, prior_bias_sigma, prior_bias_sigma,  # gyro bias
        ]))

    # generating prior for when the sim starts
    def add_prior(self,
                  graph: gtsam.NonlinearFactorGraph,
                  values: gtsam.Values,
                  sensor_data: Dict[str, Any],
                  context: Dict[str, Any]) -> None:
        """Add priors for the first frame."""
        pk = context['pose_key'](0)
        vk = context['vel_key'](0)
        bk = context['bias_key'](0)

        # pose prior from ground-truth (or first estimate)
        # constant bias 0 at first as we will estimate it
        pos = sensor_data.get('base_pos', np.zeros(3))
        quat = sensor_data.get('base_quat', np.array([0.0, 0.0, 0.0, 1.0]))
        ns = make_navstate(pos, quat, np.zeros(3))
        graph.add(PriorFactorPose3(pk, ns.pose(), self._prior_pose_noise))
        graph.add(PriorFactorVector(vk, ns.velocity(), self._prior_vel_noise))
        graph.add(gtsam.PriorFactorConstantBias(bk, ConstantBias(), self._prior_bias_noise))

        # initial estimate
        values.insert(pk, ns.pose())
        values.insert(vk, ns.velocity())
        values.insert(bk, ConstantBias())

    # transition between subsequent moments
    def add_to_graph(self,
                     graph: gtsam.NonlinearFactorGraph,
                     values: gtsam.Values,
                     step_idx: int,
                     sensor_data: Dict[str, Any],
                     context: Dict[str, Any]) -> None:
        """Add CombinedImuFactor connecting step_idx-1 → step_idx."""
        if step_idx == 0:
            return  # at first step, add_prior handles ts

        # read this as pose key at i (previous step)
        # j - current step
        # in gtsam, every variable in optimisation must have a unique id
        pi = context['pose_key'](step_idx - 1)
        vi = context['vel_key'](step_idx - 1)
        pj = context['pose_key'](step_idx)
        vj = context['vel_key'](step_idx)
        bi = context['bias_key'](step_idx - 1)
        bj = context['bias_key'](step_idx)

        pre_integrator = context['pim']
        if pre_integrator is None:
            return

        # fetches current estimate, computes the imu residual, extracts the covariance, builds a jacobian, linearises around the current sam_prev_result and adds a weighted edge to the bayes tree
        graph.add(CombinedImuFactor(pi, vi, pj, vj, bi, bj, pre_integrator._pim))

        # initial estimate: predict from isam2 current solution
        # sam needs a starting point before it can optimise
        sam_prev_result = context.get('current_sam_prev_result')
        if sam_prev_result is not None and sam_prev_result.exists(pi):
            current_pose = sam_prev_result.atPose3(pi)
            current_vel = sam_prev_result.atVector(vi)
            if sam_prev_result.exists(bi):
                current_bias = sam_prev_result.atConstantBias(bi)
            else:
                from gtsam.imuBias import ConstantBias as CB
                current_bias = CB()
        else:
            # if that fails :(( use ground truth from sensor
            pos = sensor_data.get('base_pos', np.zeros(3))
            quat = sensor_data.get('base_quat', np.array([0., 0., 0., 1.]))
            ns = make_navstate(pos, quat, np.zeros(3))
            current_pose = ns.pose()
            current_vel = ns.velocity()
            from gtsam.imuBias import ConstantBias as CB
            current_bias = CB()

        # pim.predict() propagates state for initialisation
        ns_pred = pre_integrator._pim.predict(NavState(current_pose, current_vel), current_bias)

        values.insert(pj, ns_pred.pose())
        values.insert(vj, ns_pred.velocity())
        values.insert(bj, current_bias)  # bias accumulates over time, botching everything

    def sensor_fields(self) -> List[str]:
        return ['imu_acc', 'imu_gyro', 'base_pos', 'base_quat']
