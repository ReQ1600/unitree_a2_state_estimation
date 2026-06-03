"""MuJoCo simulation bridge exposing a minimal LowState-like API.

Provides:
- `step()` to advance simulation by one timestep
- `_extract_imu()` -> (accel, gyro)
- `_extract_base_pose()` -> (position(3,), quaternion(4,))
- `_extract_contacts()` -> array(4,) with contact flags
"""
import os
from typing import Tuple
import mujoco
import numpy as np


class SimBridge:
    def __init__(self, xml_path: str, dt: float = 0.002,
                 init_joint_angles: "np.ndarray | None" = None) -> None:
        # resolve path: allow short paths like "models/a2.xml" by
        # trying several likely locations whatever
        if not os.path.exists(xml_path):
            # prefer the original third_party path so relative mesh references resolve
            alt2 = os.path.join('third_party', 'unitree_rl_mjlab', 'src', 'assets', 'robots', 'unitree_a2', 'xmls', os.path.basename(xml_path))
            if os.path.exists(alt2):
                xml_path = alt2
            else:
                # try assets/ symlink as a fallback
                alt = os.path.join('assets', os.path.basename(xml_path))
                if os.path.exists(alt):
                    xml_path = alt

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # store dt for downstream users
        self._dt = float(dt)

        # if model.opt.timestep exists, keep it; otherwise leave model default
        try:
            self.model.opt.timestep = float(dt)
        except Exception:
            pass

        # reset data cuz else first imu trash
        mujoco.mj_resetData(self.model, self.data)

        # set initial joint angles if provided (after reset so base pose is preserved)
        # Position actuators in MuJoCo may not map 1:1 via actuator_trnid;
        # we write directly to qpos[7:] which holds the 12 joint positions
        # (the first 7 entries are the floating-base pose).
        if init_joint_angles is not None:
            n_act = min(len(init_joint_angles), self.model.nu)
            # joint qpos starts after the 7-DOF floating base
            jnt_start = 7
            for j in range(n_act):
                if jnt_start + j < self.model.nq:
                    self.data.qpos[jnt_start + j] = float(init_joint_angles[j])

    def step(self, ctrl: "np.ndarray | None" = None) -> None:
        """Advance simulation by one timestep.

        Args:
            ctrl: Optional (nu,) array of actuator commands. Applied before
                  the physics step. For position actuators this is desired
                  joint angles [rad]; for torque motors it's torque [Nm].
        """
        if ctrl is not None:
            self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)

    def _extract_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (accel, gyro) as numpy arrays of shape (3,).

        Uses sensor names `imu_lin_acc` and `imu_ang_vel` defined in the A2 model.
        """
        # did not hardcode in case we change model
        acc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_lin_acc")
        gyro_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")

        # sensor addr gives index in data.sensordata
        # pack everything together tightly
        acc_adr = int(self.model.sensor_adr[acc_id])
        gyro_adr = int(self.model.sensor_adr[gyro_id])
        acc = self.data.sensordata[acc_adr:acc_adr + 3].copy()
        gyro = self.data.sensordata[gyro_adr:gyro_adr + 3].copy()
        return acc, gyro

    def _extract_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return base position and quaternion (w,x,y,z).

        The A2 model does not always name the base the same; search for a
        body name containing 'base' or fallback to body id 0.
        """
        # find tf 
        body_id = 0
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and 'base' in name.lower():
                body_id = i
                break

        pos = self.data.xpos[body_id].copy()
        # xquat exists and stores world quaternion per body when available
        try:
            quat = self.data.xquat[body_id].copy()
        except Exception:
            # fallback: build quaternion from rotation matrix
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        return pos, quat

    def _extract_contacts(self) -> np.ndarray:
        """Return contact flags for the four feet: [FL, FR, RL, RR].

        Contacts are detected from `self.data.contact` and the model geom names.
        """
        # this is complex :(
        # mujoco doesn;t have a foot sensor, just generic collision engine
        contacts = np.zeros(4, dtype=float)
        # map common foot geom name prefixes to indices
        foot_map = {
            'fl_foot': 0, 'fl_foot_collision': 0, 'fl_foot_collision_geom': 0,
            'fr_foot': 1, 'fr_foot_collision': 1,
            'rl_foot': 2, 'rl_foot_collision': 2,
            'rr_foot': 3, 'rr_foot_collision': 3
        }

        # iterate contacts
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            # obtain geom names
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)

            for g in (g1, g2):
                if not g:
                    continue
                key = g.lower()
                # try to match any known foot geom substring
                for foot_name, idx in foot_map.items():
                    if foot_name in key:
                        # approximate contact force magnitude from contact frame
                        # use contact.frame. The contact struct has 'frame' fields; but
                        # we can check distance and normal force via 'efc' or use zero-threshold.
                        # simpler: mark as contact if contact.dist < 0.01
                        try:
                            if hasattr(c, 'dist') and c.dist < 0.01:
                                contacts[idx] = 1.0
                                break
                            # else check contact force magnitude if available
                            if hasattr(c, 'force') and np.linalg.norm(c.force) > 0.5:
                                contacts[idx] = 1.0
                                break
                        except Exception:
                            contacts[idx] = 1.0
                            break
        return contacts

    def _extract_joint_states(self) -> np.ndarray:
        leg_joint_map = {
            0: ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
            1: ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            2: ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
            3: ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"]
        }

        joint_states = np.zeros((4, 3), dtype=np.float64)

        for leg_id, joint_names in leg_joint_map.items():
            for j, joint_name in enumerate(joint_names):

                jid = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name
                )

                if jid == -1:
                    raise RuntimeError(f"Joint not found: {joint_name}")

                qadr = self.model.jnt_qposadr[jid]

                joint_states[leg_id, j] = self.data.qpos[qadr]

        return joint_states