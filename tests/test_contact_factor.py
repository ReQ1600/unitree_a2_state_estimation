import numpy as np
import gtsam

from src.estimator.factors.contact_factor import ContactFactor
from src.estimator.gtsam_types import PoseKey, ContactKey


def make_values(
    rotation: gtsam.Rot3,
    d_i: np.ndarray,
    d_j: np.ndarray,
) -> gtsam.Values:
    values = gtsam.Values()

    pose_i = gtsam.Pose3(
        rotation,
        gtsam.Point3(0.0, 0.0, 0.0),
    )

    values.insert(PoseKey(0), pose_i)
    values.insert(ContactKey(0, 0), gtsam.Point3(*d_i))
    values.insert(ContactKey(0, 1), gtsam.Point3(*d_j))

    return values


def evaluate_factor_error(factor: gtsam.CustomFactor, values: gtsam.Values) -> np.ndarray:
    """
    Evaluate unwhitened factor residual.

    GTSAM CustomFactor exposes the user-defined error through unwhitenedError().
    """
    return np.asarray(factor.unwhitenedError(values), dtype=float).reshape(3)


def test_contact_factor_zero_residual():
    """
    Paper Eq. (39):

        r_Cij = R_i^T (d_j - d_i)

    If d_j == d_i, residual must be zero.
    """
    factor = ContactFactor._make_point_contact_factor(
        pose_key_i=PoseKey(0),
        contact_key_i=ContactKey(0, 0),
        contact_key_j=ContactKey(0, 1),
        noise=gtsam.noiseModel.Isotropic.Sigma(3, 1.0),
    )

    values = make_values(
        rotation=gtsam.Rot3.Identity(),
        d_i=np.array([1.0, 2.0, 3.0]),
        d_j=np.array([1.0, 2.0, 3.0]),
    )

    residual = evaluate_factor_error(factor, values)

    np.testing.assert_allclose(
        residual,
        np.zeros(3),
        atol=1e-12,
    )


def test_contact_factor_identity_rotation_residual():
    """
    With R_i = I, residual should equal d_j - d_i.
    """
    factor = ContactFactor._make_point_contact_factor(
        pose_key_i=PoseKey(0),
        contact_key_i=ContactKey(0, 0),
        contact_key_j=ContactKey(0, 1),
        noise=gtsam.noiseModel.Isotropic.Sigma(3, 1.0),
    )

    d_i = np.array([1.0, 2.0, 3.0])
    d_j = np.array([2.0, 4.0, 6.0])

    values = make_values(
        rotation=gtsam.Rot3.Identity(),
        d_i=d_i,
        d_j=d_j,
    )

    residual = evaluate_factor_error(factor, values)

    np.testing.assert_allclose(
        residual,
        d_j - d_i,
        atol=1e-12,
    )


def test_contact_factor_rotated_pose_residual():
    """
    This is the most important paper-specific test.

    Paper Eq. (39):

        r_Cij = R_i^T (d_j - d_i)

    If R_i is a 90 degree rotation around Z and d_j - d_i = [1, 0, 0],
    then:

        R_i^T [1, 0, 0] = [0, -1, 0]

    This catches the old incorrect implementation:

        r = d_j - d_i
    """
    factor = ContactFactor._make_point_contact_factor(
        pose_key_i=PoseKey(0),
        contact_key_i=ContactKey(0, 0),
        contact_key_j=ContactKey(0, 1),
        noise=gtsam.noiseModel.Isotropic.Sigma(3, 1.0),
    )

    rotation = gtsam.Rot3.Rz(np.pi / 2.0)

    values = make_values(
        rotation=rotation,
        d_i=np.array([0.0, 0.0, 0.0]),
        d_j=np.array([1.0, 0.0, 0.0]),
    )

    residual = evaluate_factor_error(factor, values)

    expected = np.array([0.0, -1.0, 0.0])

    np.testing.assert_allclose(
        residual,
        expected,
        atol=1e-12,
    )


def test_contact_factor_180deg_rotation_residual():
    """
    Additional check of paper Eq. (39):

        r_Cij = R_i^T (d_j - d_i)

    If R_i is a 180 degree rotation around Z and d_j - d_i = [1, 0, 0],
    then:

        R_i^T [1, 0, 0] = [-1, 0, 0]
    """
    factor = ContactFactor._make_point_contact_factor(
        pose_key_i=PoseKey(0),
        contact_key_i=ContactKey(0, 0),
        contact_key_j=ContactKey(0, 1),
        noise=gtsam.noiseModel.Isotropic.Sigma(3, 1.0),
    )

    rotation = gtsam.Rot3.Rz(np.pi)

    values = make_values(
        rotation=rotation,
        d_i=np.array([0.0, 0.0, 0.0]),
        d_j=np.array([1.0, 0.0, 0.0]),
    )

    residual = evaluate_factor_error(factor, values)

    expected = np.array([-1.0, 0.0, 0.0])

    np.testing.assert_allclose(
        residual,
        expected,
        atol=1e-12,
    )

def test_contact_factor_optimizes_contact_point():
    """
    Smoke test: GTSAM optimizer should be able to use ContactFactor.

    Setup:
        R_i = I
        d_i fixed at [0, 0, 0]
        d_j initialized incorrectly at [1, 0, 0]

    Contact factor residual:
        r = R_i^T (d_j - d_i)

    After optimization, d_j should move close to d_i.
    """
    graph = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    pose_key = PoseKey(0)
    contact_i_key = ContactKey(0, 0)
    contact_j_key = ContactKey(0, 1)

    pose_i = gtsam.Pose3(
        gtsam.Rot3.Identity(),
        gtsam.Point3(0.0, 0.0, 0.0),
    )

    d_i = gtsam.Point3(0.0, 0.0, 0.0)
    d_j_initial = gtsam.Point3(1.0, 0.0, 0.0)

    values.insert(pose_key, pose_i)
    values.insert(contact_i_key, d_i)
    values.insert(contact_j_key, d_j_initial)

    graph.add(
        gtsam.PriorFactorPose3(
            pose_key,
            pose_i,
            gtsam.noiseModel.Isotropic.Sigma(6, 1e-6),
        )
    )

    graph.add(
        gtsam.PriorFactorPoint3(
            contact_i_key,
            d_i,
            gtsam.noiseModel.Isotropic.Sigma(3, 1e-6),
        )
    )

    graph.add(
        ContactFactor._make_point_contact_factor(
            pose_key_i=pose_key,
            contact_key_i=contact_i_key,
            contact_key_j=contact_j_key,
            noise=gtsam.noiseModel.Isotropic.Sigma(3, 0.01),
        )
    )

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)
    result = optimizer.optimize()

    d_j_result = result.atPoint3(contact_j_key)
    d_j_np = np.array([d_j_result[0], d_j_result[1], d_j_result[2]])

    np.testing.assert_allclose(
        d_j_np,
        np.zeros(3),
        atol=1e-6,
    )
