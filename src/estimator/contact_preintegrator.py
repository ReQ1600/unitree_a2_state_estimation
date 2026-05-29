"""Preintegrated point-contact covariance model.

Implements the point-contact contact preintegration from Hartley et al. (2017),
Section VII.B.

For point contact, the contact point position is assumed fixed in the world.
The preintegrated measurement is therefore zero displacement, while the
preintegrator propagates the covariance of contact slip/noise:

    r_Cij = R_i^T (d_j - d_i)

    Sigma_Cik+1 = Sigma_Cik + B Sigma_vd B^T

where:

    B = DeltaR_ik fR(alpha_k) dt

and fR(alpha_k) is the contact-frame orientation relative to the base frame,
provided by FK. Until FK is available, callers may pass identity rotations as
a placeholder.
"""

from typing import Any, Dict, Optional

import numpy as np


class ContactPreintegrator:
    """Point-contact covariance preintegrator.

    Foot order:
        0 = FL
        1 = FR
        2 = RL
        3 = RR
    """

    def __init__(
        self,
        contact_velocity_noise_sigma: float,
        minimum_contact_probability: float,
        require_continuous_contact: bool,
        covariance_epsilon: float,
    ) -> None:
        """Create a contact preintegrator.

        Args:
            contact_velocity_noise_sigma:
                Standard deviation [m/s] of the discrete contact linear
                velocity noise eta_vd from the paper.

            minimum_contact_probability:
                Threshold above which a foot is treated as being in contact.

            require_continuous_contact:
                If True, a contact factor is valid only if contact persisted
                throughout the whole keyframe interval.

            covariance_epsilon:
                Small diagonal term added to avoid singular covariance.
        """
        self._contact_velocity_noise_sigma = float(contact_velocity_noise_sigma)
        self._minimum_contact_probability = float(minimum_contact_probability)
        self._require_continuous_contact = bool(require_continuous_contact)
        self._covariance_epsilon = float(covariance_epsilon)

        self._covariance = np.zeros((4, 3, 3), dtype=float)

        self._active = np.zeros(4, dtype=bool)
        self._valid = np.zeros(4, dtype=bool)
        self._contact_broken = np.zeros(4, dtype=bool)
        self._elapsed_time = np.zeros(4, dtype=float)

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
    ) -> "ContactPreintegrator":
        """Create ContactPreintegrator from config dictionary."""
        return cls(
            contact_velocity_noise_sigma=config["contact_velocity_noise_sigma"],
            minimum_contact_probability=config["minimum_contact_probability"],
            require_continuous_contact=config["require_continuous_contact"],
            covariance_epsilon=config["covariance_epsilon"],
        )

    def integrate(
        self,
        contact_flags: np.ndarray,
        dt: float,
        delta_rotation_ik: np.ndarray,
        fk_contact_rotation: np.ndarray,
        contact_velocity_covariance: Optional[np.ndarray] = None,
    ) -> None:
        """Integrate one contact measurement sample.

        Args:
            contact_flags:
                Shape (4,). Binary/probabilistic contact flags [FL, FR, RL, RR].

            dt:
                Integration timestep [s].

            delta_rotation_ik:
                Shape (3, 3). IMU preintegrated relative rotation DeltaR_ik.

            fk_contact_rotation:
                Either shape (3, 3) or (4, 3, 3). Contact frame orientation
                fR(alpha_k) relative to base frame. Until FK is available,
                pass identity.

            contact_velocity_covariance:
                Optional Sigma_vd from the paper.

                Supported shapes:
                    (3, 3): same covariance for all feet
                    (4, 3, 3): per-foot covariance

                If omitted, isotropic covariance is built from
                contact_velocity_noise_sigma.
        """
        contact_flags = np.asarray(contact_flags, dtype=float).reshape(4)
        dt = float(dt)

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        delta_rotation_ik = np.asarray(delta_rotation_ik, dtype=float).reshape(3, 3)
        fk_rotations = self._resolve_fk_rotations(fk_contact_rotation)
        velocity_covariance = self._resolve_velocity_covariance(
            contact_velocity_covariance
        )

        for foot_idx in range(4):
            in_contact = (
                contact_flags[foot_idx] > self._minimum_contact_probability
            )

            if not in_contact:
                self._active[foot_idx] = False
                self._contact_broken[foot_idx] = True

                if self._require_continuous_contact:
                    self._valid[foot_idx] = False

                continue

            if self._require_continuous_contact and self._contact_broken[foot_idx]:
                self._valid[foot_idx] = False
                continue

            self._active[foot_idx] = True
            self._valid[foot_idx] = True

            # Paper Eq. (42):
            #   B = DeltaR_ik fR(alpha_k) dt
            #
            # For k = i, DeltaR_ii = I, so this also gives
            #   B = fR(alpha_i) dt = R_i^T C_i dt.
            b_matrix = delta_rotation_ik @ fk_rotations[foot_idx] * dt

            # Paper Eq. (41):
            #   Sigma_Cik+1 = Sigma_Cik + B Sigma_vd B^T
            sigma_vd = velocity_covariance[foot_idx]
            self._covariance[foot_idx] += b_matrix @ sigma_vd @ b_matrix.T
            self._elapsed_time[foot_idx] += dt

    def displacement(self, foot_idx: int) -> np.ndarray:
        """Return preintegrated point-contact displacement.

        For the point-contact factor in the paper, the nominal preintegrated
        displacement is zero. This method exists for API symmetry, but the
        paper's point-contact residual does not use a nonzero delta_d_ij.
        """
        self._validate_foot_idx(foot_idx)
        return np.zeros(3, dtype=float)

    def covariance_for_foot(self, foot_idx: int) -> np.ndarray:
        """Return 3x3 covariance for a foot's preintegrated contact factor."""
        self._validate_foot_idx(foot_idx)
        return (
            self._covariance[foot_idx].copy()
            + np.eye(3) * self._covariance_epsilon
        )

    def is_valid(self, foot_idx: int) -> bool:
        """Return whether the foot has valid continuous contact in interval."""
        self._validate_foot_idx(foot_idx)
        return bool(self._valid[foot_idx])

    def elapsed_time(self, foot_idx: int) -> float:
        """Return accumulated valid contact time for this foot."""
        self._validate_foot_idx(foot_idx)
        return float(self._elapsed_time[foot_idx])

    def delta(self) -> Dict[str, np.ndarray]:
        """Return all preintegrated contact data."""
        return {
            "displacement": np.zeros((4, 3), dtype=float),
            "covariance": np.array(
                [self.covariance_for_foot(i) for i in range(4)]
            ),
            "valid": self._valid.copy(),
            "contact_broken": self._contact_broken.copy(),
            "elapsed_time": self._elapsed_time.copy(),
        }

    def reset(self) -> None:
        """Reset the preintegrator for the next keyframe interval."""
        self._covariance = np.zeros((4, 3, 3), dtype=float)
        self._active = np.zeros(4, dtype=bool)
        self._valid = np.zeros(4, dtype=bool)
        self._contact_broken = np.zeros(4, dtype=bool)
        self._elapsed_time = np.zeros(4, dtype=float)

    def _resolve_velocity_covariance(
        self,
        contact_velocity_covariance: Optional[np.ndarray],
    ) -> np.ndarray:
        """Return per-foot Sigma_vd with shape (4, 3, 3)."""
        if contact_velocity_covariance is None:
            sigma_vd = np.eye(3) * (self._contact_velocity_noise_sigma ** 2)
            return np.repeat(sigma_vd[None, :, :], 4, axis=0)

        cov = np.asarray(contact_velocity_covariance, dtype=float)

        if cov.shape == (3, 3):
            return np.repeat(cov[None, :, :], 4, axis=0)

        if cov.shape == (4, 3, 3):
            return cov

        raise ValueError(
            "contact_velocity_covariance must have shape (3, 3) or (4, 3, 3)."
        )

    @staticmethod
    def _resolve_fk_rotations(fk_contact_rotation: np.ndarray) -> np.ndarray:
        """Return per-foot FK contact rotations with shape (4, 3, 3)."""
        rot = np.asarray(fk_contact_rotation, dtype=float)

        if rot.shape == (3, 3):
            return np.repeat(rot[None, :, :], 4, axis=0)

        if rot.shape == (4, 3, 3):
            return rot

        raise ValueError(
            "fk_contact_rotation must have shape (3, 3) or (4, 3, 3)."
        )

    @staticmethod
    def _validate_foot_idx(foot_idx: int) -> None:
        """Validate foot index."""
        if foot_idx < 0 or foot_idx >= 4:
            raise ValueError("foot_idx must be in range 0..3.")
            