"""Sinusoidal trot gait generator for the Unitree A2 quadruped.

Produces desired joint-angle targets for 12 position actuators at each
timestep, matching the XML actuator order:

    [FR_hip, FR_thigh, FR_calf,
     FL_hip, FL_thigh, FL_calf,
     RR_hip, RR_thigh, RR_calf,
     RL_hip, RL_thigh, RL_calf]

Leg-to-index mapping (diagonal pairs for trot):
    FR=0, FL=1, RR=2, RL=3
    Trot pairs: (FL, RR) swing together, (FR, RL) swing together.
"""

import numpy as np
from typing import Dict, Any, Tuple


# joint range limits from a2.xml (lower, upper) per joint type 
_JOINT_LIMITS: Dict[str, Tuple[float, float]] = {
    "hip":   (-1.01, 1.01),
    "thigh_front": (-2.34, 3.15),
    "thigh_rear":  (-1.56, 3.94),
    "calf":  (-2.77, -0.54),
}


class GaitGenerator:
    """Sinusoidal trot gait producing position targets for all 12 actuators.

    Usage:
        gen = GaitGenerator(cfg["gait"])
        while simulating:
            targets = gen.step(dt)
            bridge.apply_control(targets)
            bridge.step()
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        # timing
        self._freq = float(config["frequency"])
        self._swing_ratio = float(config["swing_ratio"])
        self._phase_offsets = np.asarray(config["phase_offset"], dtype=float)  # [FR, FL, RR, RL]

        # nominal stance angles
        nom = config["nominal"]
        self._nom_hip = float(nom["hip"])
        self._nom_thigh = float(nom["thigh"])
        self._nom_calf = float(nom["calf"])

        # swing amplitudes
        amp = config["amplitude"]
        self._amp_hip = float(amp["hip"])
        self._amp_thigh = float(amp["thigh"])
        self._amp_calf = float(amp["calf"])

        # per-leg joint limits: thigh range depends on front/rear
        self._limits = np.zeros((4, 3, 2), dtype=float)  # [leg, joint, {lo,hi}]
        for leg in range(4):
            is_rear = leg >= 2
            thigh_lim = _JOINT_LIMITS["thigh_rear"] if is_rear else _JOINT_LIMITS["thigh_front"]
            self._limits[leg, 0] = _JOINT_LIMITS["hip"]
            self._limits[leg, 1] = thigh_lim
            self._limits[leg, 2] = _JOINT_LIMITS["calf"]

        # settling: hold nominal stance for settle_duration before gait starts
        self._settle_duration = float(config.get("settle_duration", 0.5))
        self._weight_shift_fraction = float(config.get("weight_shift_fraction", 0.15))
        self._hip_shift = float(config.get("hip_shift", 0.08))
        self._elapsed: float = 0.0
        self._settled: bool = self._settle_duration <= 0.0

        # pre-compute nominal stance target (all legs in stance)
        self._stance_targets = np.zeros(12, dtype=float)
        for leg in range(4):
            base = leg * 3
            self._stance_targets[base + 0] = self._nom_hip
            self._stance_targets[base + 1] = self._nom_thigh
            self._stance_targets[base + 2] = self._nom_calf

        # state
        self._phase: float = 0.0          # global phase [0, 2π)

        # current targets (last computed, for inspection)
        self._last_targets: np.ndarray = self._stance_targets.copy()

    # public API 

    def step(self, dt: float) -> np.ndarray:
        """Advance phase by dt and return joint targets for this timestep.

        During the initial settling window, holds the nominal stance pose
        so the robot stabilises on the ground before the gait begins.
        The SimBridge should initialise qpos to the stance angles so there
        is no jump when control begins.

        Returns (12,) float64 array in XML actuator order.
        """
        self._elapsed += dt

        if not self._settled:
            if self._elapsed >= self._settle_duration:
                self._settled = True
                self._phase = 0.0
            self._last_targets = self._stance_targets.copy()
            return self._last_targets

        self._phase += 2.0 * np.pi * self._freq * dt
        if self._phase >= 2.0 * np.pi:
            self._phase -= 2.0 * np.pi

        self._last_targets = self._compute_targets(self._phase)
        return self._last_targets

    def reset(self) -> None:
        """Reset phase and settling timer to zero."""
        self._phase = 0.0
        self._elapsed = 0.0
        self._settled = self._settle_duration <= 0.0
        self._last_targets = self._stance_targets.copy()

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def last_targets(self) -> np.ndarray:
        return self._last_targets.copy()

    # internal 

    def _compute_targets(self, phase: float) -> np.ndarray:
        """Compute (12,) joint targets for the given global phase.

        Creep gait: one leg swings at a time, three legs stance for stability.
        Leg order: FL → RR → FR → RL (diagonal alternation).
        Uses a smooth lift-swing-plant sinusoidal trajectory with lateral weight shifting.
        """
        targets = np.zeros(12, dtype=float)

        # swing start phase for each leg: FR=np.pi, FL=0.0, RR=np.pi/2, RL=3*np.pi/2
        swing_starts = [np.pi, 0.0, np.pi/2, 3*np.pi/2]

        # Determine current quarter and smooth transition value
        q = int(phase / (np.pi / 2.0)) % 4
        q_prev = (q - 1) % 4
        theta = phase % (np.pi / 2.0)
        trans_duration = self._weight_shift_fraction * (np.pi / 2.0)

        # Target shifts for the 4 quarters corresponding to swing leg sides:
        # Q0 (FL swings - left side) -> Right shift (+hip_shift)
        # Q1 (RR swings - right side) -> Left shift (-hip_shift)
        # Q2 (FR swings - right side) -> Left shift (-hip_shift)
        # Q3 (RL swings - left side) -> Right shift (+hip_shift)
        target_shifts = [self._hip_shift, -self._hip_shift, -self._hip_shift, self._hip_shift]

        if theta < trans_duration:
            r = theta / trans_duration
            s_val = 0.5 * (1.0 - np.cos(np.pi * r))
            curr_shift = target_shifts[q_prev] + (target_shifts[q] - target_shifts[q_prev]) * s_val
        else:
            curr_shift = target_shifts[q]

        for leg in range(4):
            rel_phase = (phase - swing_starts[leg]) % (2.0 * np.pi)
            tau = rel_phase / (2.0 * np.pi)

            if tau < self._swing_ratio:
                # ── SWING leg ──
                swing_progress = tau / self._swing_ratio
                if swing_progress < self._weight_shift_fraction:
                    # Keep foot on the ground during initial weight-shift sub-phase
                    thigh_target = self._nom_thigh
                    calf_target = self._nom_calf
                else:
                    # Lift and sweep foot forward
                    s = (swing_progress - self._weight_shift_fraction) / (1.0 - self._weight_shift_fraction)
                    thigh_target = self._nom_thigh + self._amp_thigh * np.cos(np.pi * s)
                    calf_target = self._nom_calf - self._amp_calf * np.sin(np.pi * s)
                hip_target = 0.0
            else:
                # ── STANCE leg ──
                st_prog = (tau - self._swing_ratio) / (1.0 - self._swing_ratio)
                thigh_target = self._nom_thigh - self._amp_thigh * np.cos(np.pi * st_prog)
                calf_target = self._nom_calf
                hip_target = curr_shift

            # clip to joint limits
            hip_target = np.clip(hip_target, *self._limits[leg, 0])
            thigh_target = np.clip(thigh_target, *self._limits[leg, 1])
            calf_target = np.clip(calf_target, *self._limits[leg, 2])

            base = leg * 3
            targets[base + 0] = hip_target
            targets[base + 1] = thigh_target
            targets[base + 2] = calf_target

        return targets
