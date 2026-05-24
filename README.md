# unitree_a2_state_estimation

Legged Robot State-Estimation Through Combined Forward Kinematic and Preintegrated Contact Factors utilizing GTSAM

Implementation of the factor-graph state estimator from Hartley et al. (2017, arXiv:1712.05873) for the Unitree A2 bipedal robot.

## Setup

Create and activate a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the minimum required packages:

```bash
pip install gtsam mujoco numpy pyyaml matplotlib tqdm
```

## Project structure (high level)

```
├── config/default.yaml           # Runtime & estimator parameters
├── scripts/run_estimator.py      # Example runner (simulation -> estimator)
├── src/
│   ├── bridge/                   # MuJoCo ↔ estimator bridge
│   │   ├── sim_bridge.py         # Simulation bridge (SimBridge)
│   │   └── sensor_noise.py       # IMU noise injection helpers
│   └── estimator/                # GTSAM / iSAM2 estimator core
│       ├── gtsam_types.py        # Key generators, NavState helpers
│       ├── imu_preintegrator.py  # Preintegration wrapper
│       ├── factor_registry.py    # Factor plugin system (Phase D)
│       ├── estimator.py          # Main iSAM2 loop (Phase E)
│       └── factors/               # Factor implementations (IMU, FK, contact...)
├── docs/                         # (optional) generated walkthroughs and docs
├── assets/                       # Symlinks to robot models / meshes
├── third_party/                  # Cloned repos (unitree_rl_mjlab)
└── tests/                        # Unit tests for bridge / preintegrator
```

## What is implemented

- A MuJoCo simulation bridge (`src/bridge/sim_bridge.py`) that exposes IMU, base pose, joint states and contacts.
- IMU noise model and generator (`src/bridge/sensor_noise.py`).
- GTSAM helpers and an IMU preintegrator wrapper (`src/estimator/gtsam_types.py`, `src/estimator/imu_preintegrator.py`).
- A factor plugin system (`src/estimator/factor_registry.py`) so contributors can add factors without modifying the estimator core.
- An IMU factor wrapper and an iSAM2-based estimator loop (`src/estimator/factors/imu_factor.py`, `src/estimator/estimator.py`).
- Example runner `scripts/run_estimator.py` which wires the bridge, registry and estimator and runs a short simulation.

These components implement a working skeleton of the factor-graph estimator from Hartley et al. (2017). Forward-kinematics and contact factors are left as extensible plugins.

## Quick start: run the example

Ensure the Unitree A2 MuJoCo model is available in `third_party/unitree_rl_mjlab` (or that `assets/` points to a valid `a2.xml`). Then run:

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/run_estimator.py --config config/default.yaml
```

The runner will step the simulator, integrate IMU measurements, and update the factor graph at configured keyframes. By default it will plot a comparison of ground-truth vs estimated base pose when finished.

## Configuration

Main parameters live in `config/default.yaml`:

- `simulation.timestep`, `simulation.duration`, `simulation.model_path`
- `imu.*` — preintegration and IMU noise parameters
- `estimator.keyframe_every_n_steps`, `estimator.isam_relinearize_skip`, prior sigmas
- `noise.*` — bridge noise toggles for debugging

Tune `keyframe_every_n_steps` to trade off computation vs accuracy.

## Running tests

Run the unit tests (bridge and preintegrator):

```bash
source .venv/bin/activate
PYTHONPATH=. python3 tests/test_brige.py
```

## Developer notes (how to add a new factor)

1. Implement a class that conforms to the `BaseFactor` interface in `src/estimator/factor_registry.py`:
	- `add_initial_estimate(values, step_idx, sensor_data, context)` — seed any `gtsam.Values` your factor will reference.
	- `add_to_graph(graph, values, step_idx, sensor_data, context)` — add factor(s) to the provided `gtsam.NonlinearFactorGraph`.
2. Place the file under `src/estimator/factors/` and register the factor in `scripts/run_estimator.py` via `registry.register(MyFactor(...))`.
3. Add unit tests that exercise `add_initial_estimate` + `add_to_graph` in isolation.

See the estimator implementation in `src/estimator/` for details on the registry, IMU factor wrapper, and estimator loop.

## Troubleshooting

- "Key X does not exist in Values": ensure your factor's `add_initial_estimate` inserts any referenced keys before `isam.update()` is called.
- GTSAM constructor mismatches: the Python wrapper may expose different constructors than C++. Use `dir(gtsam)` and `help()` to inspect available signatures.
- MuJoCo mesh load errors: ensure `simulation.model_path` points to the original model location in `third_party/unitree_rl_mjlab` or that `assets/` contains the required meshes.

## Contacts

If you have questions about the estimator implementation, inspect `src/estimator/` or open an issue in the project repository.

---


