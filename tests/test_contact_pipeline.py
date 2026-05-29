import numpy as np

from src.estimator.estimator import Estimator
from src.estimator.factor_registry import FactorRegistry
from src.estimator.imu_preintegrator import ImuPreintegrator
from src.estimator.factors.imu_factor import ImuFactorWrapper
from src.estimator.factors.contact_factor import ContactFactor


def make_test_estimator() -> Estimator:
    config = {
        "keyframe_every_n_steps": 3,
        "isam_relinearize_skip": 1,
        "prior_pose_sigma": 1e-3,
        "prior_vel_sigma": 1e-2,
        "prior_bias_sigma": 1e-1,
        "contact_preintegration": {
            "contact_velocity_noise_sigma": 0.1,
            "minimum_contact_probability": 0.5,
            "require_continuous_contact": True,
            "covariance_epsilon": 1e-9,
        },
    }

    preint_params = ImuPreintegrator.make_params(
        accel_noise_density=0.01,
        gyro_noise_density=0.001,
        accel_bias_rw=0.0001,
        gyro_bias_rw=0.00001,
        gravity=9.81,
    )

    registry = FactorRegistry()

    registry.register(
        ImuFactorWrapper(
            prior_pose_sigma=config["prior_pose_sigma"],
            prior_vel_sigma=config["prior_vel_sigma"],
            prior_bias_sigma=config["prior_bias_sigma"],
        )
    )

    registry.register(
        ContactFactor(
            prior_contact_sigma=0.01,
        )
    )

    return Estimator(config, registry, preint_params)


def make_sensor_data(dt: float = 0.002) -> dict:
    return {
        "imu_acc": np.array([0.0, 0.0, 9.81]),
        "imu_gyro": np.zeros(3),
        "base_pos": np.zeros(3),
        "base_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "foot_contacts": np.ones(4),
        "fk_contact_rotation": np.repeat(np.eye(3)[None, :, :], 4, axis=0),
        "dt": dt,
    }


def test_contact_pipeline_runs_one_keyframe():
    """
    Integration smoke test.

    This checks that the whole pipeline runs together:

        Estimator
        -> ImuPreintegrator
        -> ContactPreintegrator
        -> ImuFactorWrapper
        -> ContactFactor
        -> iSAM2 update

    It does not require MuJoCo or FK.
    FK is represented by identity contact rotations.
    """
    estimator = make_test_estimator()

    sensor_data = make_sensor_data()

    estimator.initialise(sensor_data)

    for _ in range(3):
        estimator.step(sensor_data)

    log = estimator.get_log()

    # Initial log + one keyframe log.
    assert len(log) >= 2

    pos, vel = estimator.latest_estimate()

    assert pos.shape == (3,)
    assert vel.shape == (3,)
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(vel))


def test_contact_pipeline_runs_multiple_keyframes():
    """
    Longer smoke test: several keyframes should be added without exceptions.
    """
    estimator = make_test_estimator()

    sensor_data = make_sensor_data()

    estimator.initialise(sensor_data)

    for _ in range(10):
        estimator.step(sensor_data)

    log = estimator.get_log()

    assert len(log) >= 4

    for entry in log:
        assert "est_pos" in entry
        assert "est_vel" in entry
        assert np.all(np.isfinite(entry["est_pos"]))
        assert np.all(np.isfinite(entry["est_vel"]))
        