import numpy as np
import pytest

from src.estimator.contact_preintegrator import ContactPreintegrator


def make_preintegrator() -> ContactPreintegrator:
    return ContactPreintegrator(
        contact_velocity_noise_sigma=0.1,
        minimum_contact_probability=0.5,
        require_continuous_contact=True,
        covariance_epsilon=0.0,
    )


def test_contact_preintegrator_covariance_single_step():
    """
    Paper Eq. (41)-(42):

        Sigma_{k+1} = Sigma_k + B Sigma_vd B^T
        B = DeltaR_ik fR(alpha_k) dt

    Single integration step with:
        DeltaR = I
        fR = I

    Expected:
        Sigma = dt^2 * Sigma_vd
    """
    pim = make_preintegrator()

    dt = 0.01

    contact_flags = np.array([1.0, 0.0, 0.0, 0.0])
    delta_rotation = np.eye(3)
    fk_rotation = np.eye(3)

    pim.integrate(
        contact_flags=contact_flags,
        dt=dt,
        delta_rotation_ik=delta_rotation,
        fk_contact_rotation=fk_rotation,
    )

    sigma_vd = np.eye(3) * (0.1 ** 2)
    B = delta_rotation @ fk_rotation * dt

    expected_cov = B @ sigma_vd @ B.T

    np.testing.assert_allclose(
        pim.covariance_for_foot(0),
        expected_cov,
        atol=1e-12,
    )


def test_contact_preintegrator_covariance_multiple_steps():
    """
    Covariance should accumulate over multiple integration steps.
    """
    pim = make_preintegrator()

    dt = 0.01
    n_steps = 100

    contact_flags = np.array([1.0, 0.0, 0.0, 0.0])

    for _ in range(n_steps):
        pim.integrate(
            contact_flags=contact_flags,
            dt=dt,
            delta_rotation_ik=np.eye(3),
            fk_contact_rotation=np.eye(3),
        )

    sigma_vd = np.eye(3) * (0.1 ** 2)
    B = np.eye(3) * dt

    expected_cov = n_steps * (B @ sigma_vd @ B.T)

    np.testing.assert_allclose(
        pim.covariance_for_foot(0),
        expected_cov,
        atol=1e-12,
    )


def test_contact_preintegrator_continuous_contact_requirement():
    """
    If contact breaks during the interval and
    require_continuous_contact=True,
    the factor must become invalid.
    """
    pim = make_preintegrator()

    dt = 0.01

    pim.integrate(
        contact_flags=np.array([1.0, 0.0, 0.0, 0.0]),
        dt=dt,
        delta_rotation_ik=np.eye(3),
        fk_contact_rotation=np.eye(3),
    )

    pim.integrate(
        contact_flags=np.array([0.0, 0.0, 0.0, 0.0]),
        dt=dt,
        delta_rotation_ik=np.eye(3),
        fk_contact_rotation=np.eye(3),
    )

    assert not pim.is_valid(0)


def test_contact_preintegrator_elapsed_time():
    """
    Valid contact time should accumulate correctly.
    """
    pim = make_preintegrator()

    dt = 0.01
    n_steps = 25

    for _ in range(n_steps):
        pim.integrate(
            contact_flags=np.array([1.0, 0.0, 0.0, 0.0]),
            dt=dt,
            delta_rotation_ik=np.eye(3),
            fk_contact_rotation=np.eye(3),
        )

    assert np.isclose(
        pim.elapsed_time(0),
        n_steps * dt,
    )


def test_contact_preintegrator_zero_displacement():
    """
    Point-contact factor in the paper assumes:

        Delta d_ij = 0

    Therefore displacement() must always return zero.
    """
    pim = make_preintegrator()

    for foot_idx in range(4):
        np.testing.assert_allclose(
            pim.displacement(foot_idx),
            np.zeros(3),
            atol=1e-12,
        )


def test_contact_preintegrator_reset():
    """
    Reset should clear all accumulated state.
    """
    pim = make_preintegrator()

    pim.integrate(
        contact_flags=np.array([1.0, 0.0, 0.0, 0.0]),
        dt=0.01,
        delta_rotation_ik=np.eye(3),
        fk_contact_rotation=np.eye(3),
    )

    pim.reset()

    np.testing.assert_allclose(
        pim.covariance_for_foot(0),
        np.zeros((3, 3)),
        atol=1e-12,
    )

    assert pim.elapsed_time(0) == 0.0
    assert not pim.is_valid(0)
    