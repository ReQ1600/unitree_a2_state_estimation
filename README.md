# unitree_a2_state_estimation

Legged Robot State Estimation Through Combined Forward Kinematic and Preintegrated Contact Factors utilizing GTSAM.

Implementation of the factor-graph state estimator from Hartley et al. (2017, arXiv:1712.05873) for the Unitree A2 robot.

---

## Setup

Create and activate a Python virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not have a `requirements.txt`, install the minimum required packages:

```bash
pip install gtsam mujoco numpy pyyaml matplotlib tqdm pytest
```

## Third-party dependency

This project expects the Unitree MuJoCo assets and XML files from `unitree_rl_mjlab` to live under `third_party/unitree_rl_mjlab`.

Clone it like this:

```bash
mkdir -p third_party
git clone https://github.com/unitreerobotics/unitree_rl_mjlab.git third_party/unitree_rl_mjlab
```

If you keep the repo elsewhere, update the model path in `config/default.yaml` or the bridge code so it can still find the A2 XML and meshes.

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
│       ├── factor_registry.py    # Factor plugin system
│       ├── estimator.py          # Main iSAM2 loop
│       └── factors/               # Factor implementations (IMU, FK, contact...)
├── docs/                         # (optional) generated walkthroughs and docs
├── assets/                       # Symlinks to robot models / meshes
├── third_party/                  # Local clone of unitree_rl_mjlab
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
PYTHONPATH=. python3 tests/test_phase_c.py && PYTHONPATH=. python3 tests/test_brige.py
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

---

## Project structure (high level)

```text
├── config/default.yaml           # Runtime & estimator parameters
├── scripts/run_estimator.py      # Example runner (simulation -> estimator)
├── src/
│   ├── bridge/
│   │   ├── sim_bridge.py         # MuJoCo ↔ estimator bridge
│   │   ├── sensor_noise.py       # IMU noise injection helpers
│   │   └── types.py              # Bridge data structures
│   │
│   └── estimator/
│       ├── gtsam_types.py
│       ├── imu_preintegrator.py
│       ├── contact_preintegrator.py
│       ├── factor_registry.py
│       ├── estimator.py
│       │
│       └── factors/
│           ├── imu_factor.py
│           ├── contact_factor.py
│           └── ...
│
├── tests/
│   ├── test_contact_preintegrator.py
│   ├── test_contact_factor.py
│   └── test_contact_pipeline.py
│
├── docs/
├── assets/
└── third_party/
```

---

## What is implemented

### Simulation bridge

* MuJoCo simulation bridge (`sim_bridge.py`)
* IMU measurements
* Base pose measurements
* Joint state extraction
* Contact state extraction

### IMU subsystem

* IMU noise injection helpers
* GTSAM IMU preintegration wrapper
* IMU factor integration with iSAM2

### Contact subsystem

Implemented according to Hartley et al. (2017), Section VII.B:

* Point-contact state representation
* Preintegrated point-contact factor
* Contact covariance preintegration
* Contact state keys (`ContactKey`)
* Contact factor integration with the estimator

The implemented point-contact residual is:

```text
r_Cij = R_i^T (d_j - d_i)
```

and contact covariance propagation follows:

```text
Σ_Cik+1 = Σ_Cik + B Σ_vd B^T
```

with

```text
B = ΔR_ik fR(α_k) Δt
```

as described in the paper.

### Estimator framework

* Factor plugin system
* iSAM2 incremental optimization
* IMU factor registration
* Contact factor registration
* Keyframe-based estimation pipeline

---

## Current limitations

The point-contact factor and contact preintegrator are implemented and validated.

The following placeholders remain until the forward-kinematics module is completed:

### 1. Contact-frame rotations

The quantity

```text
fR(α_k)
```

is currently represented by identity matrices.

This placeholder is located in:

```python
SimBridge._extract_fk_contact_rotation()
```

The FK module is expected to provide the paper quantity `fR(α_k)`, i.e. the contact-frame orientation relative to the base frame, with shape `(4, 3, 3)` and foot order `[FL, FR, RL, RR]`.

Once FK is available, no changes to `ContactFactor` or `ContactPreintegrator` should be required.

### 2. Initial contact point estimates

Initial contact point states are currently initialized using placeholder world-frame coordinates.

These should ultimately be initialized from FK.

### 3. Contact noise tuning

The parameter

```yaml
contact_velocity_noise_sigma
```

is currently hand-tuned.

Once FK uncertainty modelling is available, it should be revisited.

---

## Quick start

Ensure the Unitree model is available and run:

```bash
source .venv/bin/activate

PYTHONPATH=. python3 scripts/run_estimator.py \
    --config config/default.yaml
```

The estimator will:

1. Step the simulator.
2. Integrate IMU measurements.
3. Integrate contact measurements.
4. Update the factor graph at keyframes.
5. Plot estimated vs ground-truth trajectories.

---

## Configuration

Main parameters live in:

```text
config/default.yaml
```

Important groups:

### Simulation

```yaml
simulation:
```

* timestep
* duration
* model_path

### IMU

```yaml
imu:
```

* accel_noise_density
* gyro_noise_density
* accel_bias_rw
* gyro_bias_rw
* gravity

### Estimator

```yaml
estimator:
```

* keyframe_every_n_steps
* isam_relinearize_skip
* prior_pose_sigma
* prior_vel_sigma
* prior_bias_sigma

### Contact preintegration

```yaml
contact_preintegration:
```

* contact_velocity_noise_sigma
* minimum_contact_probability
* require_continuous_contact
* covariance_epsilon

---

## Running tests

Run the full test suite:

```bash
PYTHONPATH=. \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
pytest tests/ -v
```

Important tests:

### Contact preintegrator

```text
tests/test_contact_preintegrator.py
```

Verifies:

* covariance propagation
* covariance accumulation
* contact validity logic
* zero-displacement point-contact assumption
* reset behaviour

### Contact factor

```text
tests/test_contact_factor.py
```

Verifies:

* zero residual case
* identity-rotation case
* rotated-frame residuals
* optimization through GTSAM

### Pipeline integration

```text
tests/test_contact_pipeline.py
```

Verifies:

* estimator integration
* IMU preintegration
* contact preintegration
* factor insertion
* iSAM2 updates

---

## Developer notes

### Adding a new factor

1. Implement a class conforming to `BaseFactor`.
2. Place it under:

```text
src/estimator/factors/
```

3. Register it in:

```python
registry.register(MyFactor(...))
```

4. Add unit tests.

### Notes on ContactFactor

The current implementation follows the point-contact formulation from Hartley et al. (2017).

Future FK integration should only replace FK placeholders and should not require modifications to the mathematical formulation of:

* `ContactFactor`
* `ContactPreintegrator`

---

## Troubleshooting

### Missing GTSAM symbols

Inspect available Python bindings:

```python
import gtsam
dir(gtsam)
```

### MuJoCo mesh loading issues

Ensure:

```text
third_party/unitree_rl_mjlab
```

contains the original model assets.

### Pytest loading ROS plugins

Use:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

when running tests.

---

## References

Hartley, R., Ghaffari, M., Eustice, R., & Grizzle, J. (2017).

**Legged Robot State-Estimation Through Combined Forward Kinematic and Preintegrated Contact Factors**
arXiv:1712.05873
