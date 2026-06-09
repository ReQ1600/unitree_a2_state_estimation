"""Preintegrated point-contact factor from Hartley et al. (2017).

Implements the point-contact residual from Section VII.B:

    r_Cij = R_i^T (d_j - d_i)

where:
    R_i  is the robot base orientation at keyframe i,
    d_i  is the world-frame contact point position at keyframe i,
    d_j  is the world-frame contact point position at keyframe j.

The covariance is supplied by ContactPreintegrator according to Eq. (41)-(42).
"""

from typing import Any, Dict, List, Optional

import numpy as np
import gtsam
from gtsam import noiseModel

from ..factor_registry import BaseFactor
from ..gtsam_types import PoseKey, ContactKey
from ..contact_preintegrator import ContactPreintegrator


class ContactFactor(BaseFactor):
    """Preintegrated point-contact factor.

    Graph structure per foot:

        PoseKey(i)
            |
            v
        ContactKey(foot, i) ---- point-contact factor ---- ContactKey(foot, j)

    FK factor is responsible for connecting:

        PoseKey(k), joint_positions(k) ---- FK factor ---- ContactKey(foot, k)

    This factor consumes ContactPreintegrator covariance and inserts the
    paper's point-contact residual:

        r = R_i^T (d_j - d_i)
    """

    def __init__(self, prior_contact_sigma: float = 0.01) -> None:
        """Initialize the contact factor.

        Args:
            prior_contact_sigma:
                Standard deviation [m] for initial contact point priors.
                This anchors ContactKey(foot, 0). It is not used for the
                preintegrated point-contact residual itself.
        """
        super().__init__()

        self.foot_names = ["FL", "FR", "RL", "RR"]

        self._prior_contact_noise = noiseModel.Isotropic.Sigma(
            3,
            float(prior_contact_sigma),
        )

    def add_prior(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        sensor_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Contact Pose3 priors are handled by ForwardKinematicFactor."""
        return

    def add_to_graph(
        self,
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        step_idx: int,
        sensor_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Add preintegrated point-contact factors for the current keyframe."""
        if step_idx == 0:
            return

        contact_pim: ContactPreintegrator = self._require_contact_pim(context)
        current_estimate: Optional[gtsam.Values] = context.get("current_estimate")

        pose_key_i = context.get("pose_key", PoseKey)(step_idx - 1)

        for foot_idx in range(4):
            if not contact_pim.is_valid(foot_idx):
                continue

            contact_key_i = ContactKey(foot_idx, step_idx - 1)
            contact_key_j = ContactKey(foot_idx, step_idx)

            if not self._insert_initial_estimate(
                values=values,
                contact_key_i=contact_key_i,
                contact_key_j=contact_key_j,
                foot_idx=foot_idx,
                step_idx=step_idx,
                current_estimate=current_estimate,
            ):
                continue

            covariance_ij = np.asarray(
                contact_pim.covariance_for_foot(foot_idx),
                dtype=float,
            ).reshape(3, 3)

            contact_noise = noiseModel.Gaussian.Covariance(covariance_ij)
            # print(
            #     "CONTACT ADD",
            #     foot_idx,
            #     step_idx,
            #     current_estimate.exists(contact_key_i) if current_estimate is not None else None,
            #     values.exists(contact_key_j),
            # )

            graph.add(
                self._make_point_contact_factor(
                    pose_key_i=pose_key_i,
                    contact_key_i=contact_key_i,
                    contact_key_j=contact_key_j,
                    noise=contact_noise,
                )
            )

    def sensor_fields(self) -> List[str]:
        """Return sensor_data keys required by this factor."""
        return ["foot_contacts"]

    def _insert_initial_estimate(
        self,
        values: gtsam.Values,
        contact_key_i: int,
        contact_key_j: int,
        foot_idx: int,
        step_idx: int,
        current_estimate: Optional[gtsam.Values],
    ) -> bool:
        """Insert initial estimate for ContactKey(foot, step_idx)."""
        # print(
        #     "CONTACT INIT CALL",
        #     foot_idx,
        #     step_idx,
        #     contact_key_i,
        #     contact_key_j,
        # )
        if values.exists(contact_key_j):
            return True

        if current_estimate is not None and current_estimate.exists(contact_key_i):
            previous_pose = current_estimate.atPose3(contact_key_i)
            # print(
            #     "CONTACT INIT",
            #     foot_idx,
            #     step_idx,
            #     contact_key_j,
            #     values.exists(contact_key_j)
            # )
            values.insert(contact_key_j, previous_pose)
            # print(
            #     "CONTACT INIT OK",
            #     foot_idx,
            #     step_idx,
            #     contact_key_i,
            # )
            return True

        # print(
        #     "CONTACT INIT FAILED",
        #     foot_idx,
        #     step_idx,
        #     contact_key_i,
        # )
        return False

    @staticmethod
    def _make_point_contact_factor(
        pose_key_i: int,
        contact_key_i: int,
        contact_key_j: int,
        noise: gtsam.noiseModel.Base,
    ) -> gtsam.CustomFactor:
        """Create the custom point-contact factor.

        Residual from paper Eq. (39):
            r = R_i^T (d_j - d_i)

        Variables:
            pose_i: robot base Pose3 at keyframe i
            d_i:    contact point position at keyframe i, world frame
            d_j:    contact point position at keyframe j, world frame

        Jacobians:
            This implementation uses numerical Jacobians for robustness and
            clarity. The residual itself follows the paper exactly.
        """

        def residual_from_values(values: gtsam.Values) -> np.ndarray:
            pose_i = values.atPose3(pose_key_i)
            contact_pose_i = values.atPose3(contact_key_i)
            contact_pose_j = values.atPose3(contact_key_j)

            r_i = pose_i.rotation().matrix()

            d_i = contact_pose_i.translation()
            d_j = contact_pose_j.translation()

            d_i_np = np.array([d_i[0], d_i[1], d_i[2]], dtype=float)
            d_j_np = np.array([d_j[0], d_j[1], d_j[2]], dtype=float)

            return r_i.T @ (d_j_np - d_i_np)

        def error_func(
            factor: gtsam.CustomFactor,
            values: gtsam.Values,
            jacobians: Optional[List[np.ndarray]],
        ) -> np.ndarray:
            residual = residual_from_values(values)

            if jacobians is not None:
                # GTSAM Pose3 tangent convention: [rot, trans], dimension 6.
                jacobians[0] = ContactFactor._numerical_jacobian_pose(
                    values=values,
                    pose_key=pose_key_i,
                    residual_func=residual_from_values,
                )

                r_i_T = values.atPose3(pose_key_i).rotation().matrix().T

                # Contact variables are Pose3(C_i, d_i), but Eq. (39)
                # depends only on d_i and d_j. Therefore rotation columns are zero,
                # translation columns are -R_i^T and +R_i^T.
                jacobians[1] = np.zeros((3, 6), dtype=float)
                jacobians[2] = np.zeros((3, 6), dtype=float)

                jacobians[1][:, 3:6] = -r_i_T
                jacobians[2][:, 3:6] = r_i_T

            return residual

        return gtsam.CustomFactor(
            noise,
            [pose_key_i, contact_key_i, contact_key_j],
            error_func,
        )

    @staticmethod
    def _numerical_jacobian_pose(
        values: gtsam.Values,
        pose_key: int,
        residual_func,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """Numerical Jacobian of residual wrt Pose3 tangent perturbation.

        Uses right perturbation:
            Pose_plus = Pose.retract(delta)

        This avoids hand-derived Pose3 Jacobian mistakes during integration.
        """
        pose = values.atPose3(pose_key)
        base_residual = residual_func(values)

        jac = np.zeros((3, 6), dtype=float)

        for col in range(6):
            delta = np.zeros(6, dtype=float)
            delta[col] = eps

            values_plus = gtsam.Values(values)
            values_plus.update(pose_key, pose.retract(delta))

            residual_plus = residual_func(values_plus)
            jac[:, col] = (residual_plus - base_residual) / eps

        return jac

    @staticmethod
    def _require_contact_pim(context: Dict[str, Any]) -> ContactPreintegrator:
        """Fetch contact preintegrator from context."""
        if "contact_pim" not in context:
            raise ValueError(
                "ContactFactor requires context['contact_pim'] at keyframes."
            )

        contact_pim = context["contact_pim"]

        if not isinstance(contact_pim, ContactPreintegrator):
            raise TypeError(
                "context['contact_pim'] must be an instance of ContactPreintegrator."
            )

        return contact_pim
