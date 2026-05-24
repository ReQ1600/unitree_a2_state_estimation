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

# ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bridge.sim_bridge import SimBridge
from src.bridge.sensor_noise import ImuNoiseGenerator, ImuNoiseParams
from src.estimator.imu_preintegrator import ImuPreintegrator
from src.estimator.factor_registry import FactorRegistry
from src.estimator.factors.imu_factor import ImuFactorWrapper
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

    # set up the physics bridge
    bridge = SimBridge(cfg['simulation']['model_path'],
                       dt=cfg['simulation']['timestep'])

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

    # inisialise main solver object
    estimator = Estimator(est_cfg, registry, preint_params)

    # main loop
    duration = cfg['simulation']['duration']
    dt = bridge._dt
    n_steps = int(duration / dt)

    # the first frame
    bridge.step()
    acc, gyro = bridge._extract_imu()
    pos, quat = bridge._extract_base_pose()
    contacts = bridge._extract_contacts()

    sensor_data = {
        'imu_acc': acc, #    corrupted by noise
        'imu_gyro': gyro,
        'base_pos': pos, #   usually ground truth
        'base_quat': quat,
        'foot_contacts': contacts,
        'dt': dt,
    }

    if noise_gen:
        acc, gyro = noise_gen.corrupt(acc, gyro, dt)
        sensor_data['imu_acc'] = acc
        sensor_data['imu_gyro'] = gyro

    # trigger the prior factors
    estimator.initialise(sensor_data)

    # remaining steps
    for step in tqdm(range(1, n_steps), desc='Simulating'):
        # iterate through the simulation duration
        bridge.step()   # advances time
        acc, gyro = bridge._extract_imu() # extracts raw data
        pos, quat = bridge._extract_base_pose()
        contacts = bridge._extract_contacts()

        # corrupt our readings
        if noise_gen:
            acc, gyro = noise_gen.corrupt(acc, gyro, dt)

        sensor_data = {
            'imu_acc': acc,
            'imu_gyro': gyro,
            'base_pos': pos,
            'base_quat': quat,
            'foot_contacts': contacts,
            'dt': dt,
        }
        # pass noisy measurements to imu preintegrator
        estimator.step(sensor_data)
        # estimator internally decides when to run isam2

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
        fig.suptitle('A2 State Estimation — IMU-only (iSAM2)')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
