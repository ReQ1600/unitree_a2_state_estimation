"""Quick smoke test for the simulation bridge."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bridge.sim_bridge import SimBridge
from src.bridge.sensor_noise import ImuNoiseGenerator, ImuNoiseParams
import numpy as np

bridge = SimBridge("models/a2.xml", dt = 0.002)
noise_maker = ImuNoiseGenerator(ImuNoiseParams(0.03, 0.001, 0.001, 0.0001))
history = []

for _ in range (5000):
    bridge.step()
    acc, gyro = bridge._extract_imu()
    pos, quar = bridge._extract_base_pose()
    contacts = bridge._extract_contacts()
    
    acc_n, gyro_n = noise_maker.corrupt(acc, gyro, bridge._dt)
    history.append((bridge.data.time, acc_n, gyro_n, pos, contacts))