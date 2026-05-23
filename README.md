# unitree_a2_state_estimation

Legged Robot State-Estimation Through Combined Forward Kinematic and Preintegrated Contact Factors utilizing GTSAM

Implementation of the factor-graph state estimator from Hartley et al. (2017, arXiv:1712.05873) for the Unitree A2 bipedal robot.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
├── config/default.yaml      # All tunable parameters
├── scripts/run_estimator.py # Entry point
├── src/
│   ├── bridge/              # MuJoCo ↔ estimator communication
│   │   ├── sim_bridge.py    # Simulation bridge
│   │   └── sensor_noise.py  # IMU noise injection
│   └── estimator/           # GTSAM iSAM2 estimator core
│       ├── gtsam_types.py       # Key generators, type aliases
│       ├── imu_preintegrator.py # IMU preintegration wrapper
│       ├── factor_registry.py   # Factor plugin system
│       ├── estimator.py         # Main iSAM2 loop
│       └── factors/             # Custom factors (IMU, contact, FK, ...)
├── assets/                  # Symlinks to robot models
├── third_party/             # Cloned dependencies (unitree_rl_mjlab)
└── tests/
```

## Dependencies

- **GTSAM 4.2+** — Factor graph optimization (iSAM2, IMU preintegration)
- **MuJoCo 3.8+** — Physics simulation (Unitree A2 model from unitree_rl_mjlab)
- **NumPy, PyYAML, Matplotlib, tqdm**

