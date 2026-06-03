#!/usr/bin/env python3
"""Entry point: run the A2 factor-graph state estimator.

Usage:
    PYTHONPATH=. python3 scripts/run_estimator.py [--config config/default.yaml]
"""

import argparse
import sys
import os
import numpy as np
import yaml
from tqdm import tqdm

import time
import mujoco.viewer

# ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bridge.sim_bridge import SimBridge
from src.bridge.gait_generator import GaitGenerator
from src.bridge.sensor_noise import ImuNoiseGenerator, ImuNoiseParams
from src.estimator.imu_preintegrator import ImuPreintegrator
from src.estimator.factor_registry import FactorRegistry
from src.estimator.factors.imu_factor import ImuFactorWrapper
from src.estimator.factors.forward_kinematic_factor import ForwardKinematicFactor
from src.estimator.estimator import Estimator


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='A2 Factor-Graph State Estimator')
    parser.add_argument('--config', default='config/default.yaml',
                        help='Path to YAML configuration file')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # compute nominal stance angles for initialisation
    nom = cfg['gait']['nominal']
    init_joints = np.zeros(12, dtype=float)
    for leg in range(4):
        base = leg * 3
        init_joints[base + 0] = float(nom['hip'])
        init_joints[base + 1] = float(nom['thigh'])
        init_joints[base + 2] = float(nom['calf'])

    # set up the physics bridge with robot starting in stance pose
    bridge = SimBridge(cfg['simulation']['model_path'],
                       dt=cfg['simulation']['timestep'],
                       init_joint_angles=init_joints)

    # set up the gait generator (trot pattern)
    gait_gen = GaitGenerator(cfg['gait'])

    # optionally noise
    noise_gen = None
    if cfg['noise']['enabled']:
        np_cfg = cfg['noise']
        noise_gen = ImuNoiseGenerator(ImuNoiseParams(
            acc_white_density=np_cfg['accel_white_density'],
            gyro_white_density=np_cfg['gyro_white_density'],
            acc_bias_density=np_cfg['accel_bias_density'],
            gyro_bias_density=np_cfg['gyro_bias_density'],
        ))

    # create gtsam parameter object using ct noise densities
    imu_cfg = cfg['imu']
    preint_params = ImuPreintegrator.make_params(
        accel_noise_density=imu_cfg['accel_noise_density'],
        gyro_noise_density=imu_cfg['gyro_noise_density'],
        accel_bias_rw=imu_cfg['accel_bias_rw'],
        gyro_bias_rw=imu_cfg['gyro_bias_rw'],
        gravity=imu_cfg['gravity'],
    )

    # register factors
    registry = FactorRegistry()
    est_cfg = cfg['estimator']
    imu_factor = ImuFactorWrapper(
        prior_pose_sigma=est_cfg.get('prior_pose_sigma', 0.001),
        prior_vel_sigma=est_cfg.get('prior_vel_sigma', 0.01),
        prior_bias_sigma=est_cfg.get('prior_bias_sigma', 0.1),
    )
    registry.register(imu_factor)

    # registering forward kinematic factor for 4 legs
    for i in range(4):
        registry.register(ForwardKinematicFactor(i, 0.00873, cfg['simulation']['model_path']))

    # inisialise main solver object
    estimator = Estimator(est_cfg, registry, preint_params)

    # main loop
    duration = cfg['simulation']['duration']
    dt = bridge._dt
    n_steps = int(duration / dt)

    # the first frame — apply initial stance before stepping
    init_targets = gait_gen.step(dt)
    bridge.step(ctrl=init_targets)
    acc, gyro = bridge._extract_imu()
    pos, quat = bridge._extract_base_pose()
    contacts = bridge._extract_contacts()
    joint_states = bridge._extract_joint_states()

    sensor_data = {
        'imu_acc': acc, #    corrupted by noise
        'imu_gyro': gyro,
        'base_pos': pos, #   usually ground truth
        'base_quat': quat,
        'foot_contacts': contacts,
        'joint_states': joint_states,
        'dt': dt,
    }

    if noise_gen:
        acc, gyro = noise_gen.corrupt(acc, gyro, dt)
        sensor_data['imu_acc'] = acc
        sensor_data['imu_gyro'] = gyro

    # trigger the prior factors
    estimator.initialise(sensor_data)

    mj_model = bridge.model
    mj_data = bridge.data

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        # pre-allocate foot-site IDs for contact visualization
        foot_site_names = ["FL", "FR", "RL", "RR"]
        foot_site_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in foot_site_names
        ]

        # remaining steps
        for step in tqdm(range(1, n_steps), desc='Simulating'):
            step_start = time.time()

            # apply sinusoidal gait targets before physics step
            targets = gait_gen.step(dt)
            bridge.step(ctrl=targets)

            # extract all sensor data
            acc, gyro = bridge._extract_imu()
            pos, quat = bridge._extract_base_pose()
            contacts = bridge._extract_contacts()
            joint_states = bridge._extract_joint_states()

            viewer.sync()

            # ── foot-contact visualisation ──────────────────────────────
            # colored spheres at foot sites: green = in contact, red = in air
            viewer.user_scn.ngeom = 0  # clear previous markers
            for leg_idx, site_id in enumerate(foot_site_ids):
                if site_id < 0:
                    continue
                foot_pos = mj_data.site_xpos[site_id].copy()
                in_contact = contacts[leg_idx] > 0.5
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[leg_idx],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.025, 0.0, 0.0]),
                    pos=foot_pos,
                    mat=np.eye(3).ravel(),
                    rgba=np.array(
                        [0.0, 1.0, 0.0, 0.8] if in_contact
                        else [1.0, 0.2, 0.2, 0.8]
                    ),
                )
                viewer.user_scn.geoms[leg_idx].category = mujoco.mjtCatBit.mjCAT_DECOR
            viewer.user_scn.ngeom = 4
            # ────────────────────────────────────────────────────────────

            # corrupt our readings
            if noise_gen:
                acc, gyro = noise_gen.corrupt(acc, gyro, dt)

            sensor_data = {
                'imu_acc': acc,
                'imu_gyro': gyro,
                'base_pos': pos,
                'base_quat': quat,
                'foot_contacts': contacts,
                'joint_states': joint_states,
                'dt': dt,
            }
            # pass noisy measurements to imu preintegrator
            estimator.step(sensor_data)
            # estimator internally decides when to run isam2

            #slow down the simulation to real time
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
                
            if not viewer.is_running():
                print("Window closed - sumulation terminated")
                break

    # results and visualisation
    log = estimator.get_log()
    gt = np.array([e['gt_pos'] for e in log])
    est = np.array([e['est_pos'] for e in log])
    times = np.arange(len(log)) * (cfg['simulation']['timestep'] *
                                     cfg['estimator']['keyframe_every_n_steps'])

    print(f"\nProcessed {n_steps} steps ({duration:.1f} s), "
          f"{len(log)} keyframes.")
    print(f"Final GT  pos: {gt[-1]}")
    print(f"Final est pos: {est[-1]}")

    if cfg['output'].get('plot', True):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ['X', 'Y', 'Z']
        for ax_idx, ax in enumerate(axes):
            ax.plot(times, gt[:, ax_idx], 'k-', label='Ground truth')
            ax.plot(times, est[:, ax_idx], 'r--', label='Estimate')
            ax.set_ylabel(f'{labels[ax_idx]} [m]')
            ax.legend()
            ax.grid(True)
        axes[-1].set_xlabel('Time [s]')
        fig.suptitle('A2 State Estimation — IMU+FK (iSAM2)')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
