import gtsam
import numpy as np
import scipy
from ..factor_registry import BaseFactor
import mujoco
import os
from typing import Any, Dict, List

class ForwardKinematicFactor(BaseFactor):
    def __init__(self, leg_id, encoder_sigma, xml_path):
        """
        leg_id: leg identifier (0-3 for a2)
        encoder_sigma: encoder noise standard deviation
        xml_path: path to a xml mujoco config file
        """

        self.leg_id = leg_id
        self.encoder_sigma = encoder_sigma

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

        self.kinematic_constants = self._build_kinematic_constants(xml_path)[leg_id]
    
    @property
    def sensor_fields(self):
        return ['joint_states']
    
    def add_initial_estimate(self, vals, step_idx, sensor_data, ctx):
        """
        contact frame(Ci, di) position and orientation init
        """

        if sensor_data['foot_contacts'][self.leg_id] == 0:
            return
        
        # base and leg contact keys  
        base_key = gtsam.symbol('x', step_idx)
        contact_key = gtsam.symbol('c', self.leg_id * 1000 + step_idx)

        # init x_k from ground truth if doesnt exist 
        if not vals.exists(base_key):
            base_pos = sensor_data['base_pos']
            base_quat = sensor_data['base_quat']
            base_rot = gtsam.Rot3.Quaternion(base_quat[0], base_quat[1], base_quat[2], base_quat[3])
            vals.insert(base_key, gtsam.Pose3(base_rot, base_pos))

        # init based on current base estimation and measurements
        if not vals.exists(contact_key):
            if vals.exists(base_key):
                base_pose = vals.atPose3(base_key)
                alpha = sensor_data['joint_states'][self.leg_id]

                #calculated based on a2 model
                R_bc = self._f_R(alpha)
                p_bc = self._f_p(alpha)

                # C = R * f_R
                # d = p + R * f_p
                contact_rot = base_pose.rotation().compose(R_bc)
                contact_pos = base_pose.translation() + base_pose.rotation().rotate(p_bc)

                vals.insert(contact_key, gtsam.Pose3(contact_rot, contact_pos))
    
    def add_to_graph(self, graph, values, step_idx, sensor_data, context):
        """
        adds gtsam.CustomFactor to the graph calculating residues f_Ri and f_pi 
        """
        print("FK add_to_graph", self.leg_id, step_idx, sensor_data["foot_contacts"][self.leg_id])
        # if leg is not on the ground calculating fc would only make the estimation worse
        if sensor_data['foot_contacts'][self.leg_id] == 0:
            return

        # base and leg contact keys  
        base_key = gtsam.symbol('x', step_idx)
        contact_key = gtsam.symbol('c', self.leg_id * 1000 + step_idx)
        
        if not values.exists(base_key) or not values.exists(contact_key):
            return
            
        print(f"DEBUG: keys available in  sensor_data: {sensor_data.keys()}")
        leg_encoder_data = sensor_data['joint_states'][self.leg_id]

        fk_R = self._f_R(leg_encoder_data)
        fk_p = self._f_p(leg_encoder_data)
        covariance = self._calculate_covariance(leg_encoder_data)
        noise_model = gtsam.noiseModel.Gaussian.Covariance(covariance)

        def err_func(this, v, H):
            """
            error function that calculates r_fRi and r_fpi
            """

            base_pose = v.atPose3(base_key)
            contact_pose = v.atPose3(contact_key)

            R_i = base_pose.rotation()
            p_i = base_pose.translation()
            C_i = contact_pose.rotation()
            d_i = contact_pose.translation()

            # rotation residual calculation Log(f_R^T * R^T * C) (21)
            r_R = gtsam.Rot3.Logmap(
                fk_R.inverse().compose(R_i.inverse().compose(C_i))
            )

            #position residual calculation R^T * (d - p) - f_p (21)
            r_p = R_i.unrotate(d_i - p_i) - fk_p

            #H jacobian calculation
            #H0 - jacobian with respect to the base
            H0 = np.zeros((6, 6))

            #rotation with respect to the base rotation
            H0[0:3, 0:3] = -inverse_right_jacobian_so3(r_R) @ (C_i.inverse().compose(R_i)).matrix()
            
            # position with respect tot the base rotation
            H0[3:6, 0:3] = skew(R_i.unrotate(d_i - p_i))

            #H1 - jacobian with respect to the contact frame
            H1 = np.zeros((6, 6))

            #rotation with respect to the contact rotation
            H1[0:3, 0:3] = inverse_right_jacobian_so3(r_R)

            #position with respect to the contact translaction
            H1[3:6, 3:6] = R_i.inverse().compose(C_i).matrix()

            if H is not None:
                H[0] = H0
                H[1] = H1
                
            return np.hstack((r_R, r_p))

        factor = gtsam.CustomFactor(noise_model, [base_key, contact_key], err_func)
        graph.add(factor)
    
    def add_prior(self,
                  graph: gtsam.NonlinearFactorGraph,
                  values: gtsam.Values,
                  sensor_data: Dict[str, Any],
                  context: Dict[str, Any]) -> None:
        
        contact_key = gtsam.symbol('c', self.leg_id * 1000)
        base_key = gtsam.symbol('x', 0)
        
        #init from ground truth if doesnt exist
        if not values.exists(base_key):
            base_pos = sensor_data['base_pos']
            base_quat = sensor_data['base_quat'] # MuJoCo podaje [w, x, y, z]
            base_rot = gtsam.Rot3.Quaternion(base_quat[0], base_quat[1], base_quat[2], base_quat[3])
            base_pose = gtsam.Pose3(base_rot, base_pos)
            
            values.insert(base_key, base_pose)
            
            # base prior
            base_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.01)
            graph.add(gtsam.PriorFactorPose3(base_key, base_pose, base_noise))
            
        base_pose = values.atPose3(base_key)

        alpha = sensor_data['joint_states'][self.leg_id]
        fk_R = self._f_R(alpha)
        fk_p = self._f_p(alpha)

        contact_R = base_pose.rotation().compose(fk_R)
        contact_p = base_pose.translation() + base_pose.rotation().rotate(fk_p)
        contact_pose = gtsam.Pose3(contact_R, contact_p)

        if not values.exists(contact_key):
            values.insert(contact_key, contact_pose)

        prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        graph.add(gtsam.PriorFactorPose3(contact_key, contact_pose, prior_noise))

    def contact_rotation(self, alpha: np.ndarray) -> np.ndarray:
        """Return fR(alpha): contact-frame orientation relative to base frame."""
        return self._f_R(alpha).matrix()

    def _f_R(self, alpha):
        """
        calculates orintation based on (12) & (13)
        alpha: leg encoder data
        """
        
        A = self.kinematic_constants['A']
        axes = self.kinematic_constants['axes']

        # accumulated rotation init
        R_total = gtsam.Rot3()

        for n in range(len(alpha)):
            A_n = gtsam.Rot3(A[n])

            # Exp(alpha_n^dagger)
            joint_axis_vec = np.zeros(3)
            joint_axis_vec[axes[n]] = alpha[n]
            joint_rot = gtsam.Rot3.Expmap(joint_axis_vec)

            #R = R * A_n * Exp(alpha_n^dagger), multiplies all rotation matrices
            R_total = R_total.compose(A_n).compose(joint_rot)

        #transform to contact point
        R_total = R_total.compose(gtsam.Rot3(A[-1]))

        return R_total
        
    def _f_p(self, alpha):
        """
        calculates position based on (12) & (13)
        alpha: leg encoder data
        """

        A = self.kinematic_constants['A']
        t = self.kinematic_constants['t']
        axes = self.kinematic_constants['axes']

        p_total = np.zeros(3)
        current_R = gtsam.Rot3() # current rotation matrix relative to the base

        for n in range(len(alpha)):
            t_in_base = current_R.rotate(np.array(t[n]))
            p_total += t_in_base

            A_n = gtsam.Rot3(A[n])

            #A_1n
            joint_axis_vec = np.zeros(3)
            joint_axis_vec[axes[n]] = alpha[n]
            joint_rot = gtsam.Rot3.Expmap(joint_axis_vec)

            current_R = current_R.compose(A_n).compose(joint_rot)

        #transform to contact point
        p_total += current_R.rotate(np.array(t[-1]))

        return p_total


    def _calculate_covariance(self, alpha):
        Q_blocks = []
        S_blocks = []

        sigma_blocks = []
        for i in range(len(alpha)):
            block = np.zeros((3,3))
            axis = self.kinematic_constants['axes'][i]
            block[axis, axis] = self.encoder_sigma**2
            sigma_blocks.append(block)
        
        sigma_alpha_dagger = scipy.linalg.block_diag(*sigma_blocks)

        #calculating Qi and Si blocks for every state
        for i in range(len(alpha)):
            #calculating Qi = A_{i+1, N+1}^T
            Qi = self._get_rotation_between(i + 1, len(alpha) + 1, alpha).transpose()
            Q_blocks.append(Qi)

            #calculating Si (23)
            Si = np.zeros((3, 3))
            for n in range(i, len(alpha)):
                A1_nplus1 = self._get_rotation_between(1, n + 2, alpha)
                Aiplus1_nplus1_T = self._get_rotation_between(i + 1, n + 2, alpha).transpose()
                
                tn_plus_1 = self.kinematic_constants['t'][n]
                t_hat = skew(tn_plus_1)
                
                Si -= A1_nplus1 @ t_hat @ Aiplus1_nplus1_T
            S_blocks.append(Si)

        Q = np.hstack(Q_blocks)
        S = np.hstack(S_blocks)

        #calculating jacobian M = [Q; S]
        M = np.vstack([Q, S])

        return M @ sigma_alpha_dagger @ M.T

    def _get_rotation_between(self, start, end, alpha):
        """
        alpha: current joint states
        """

        if start == end:
            return gtsam.Rot3().matrix()
        
        A = self.kinematic_constants['A']
        axes = self.kinematic_constants['axes']

        r_R = gtsam.Rot3()

        #publication starts at 1 but python indexes from 0
        for i in range(start - 1, end - 1):
            A_i = gtsam.Rot3(A[i])

            # if not the last transform which is constant
            if i < len(alpha):
                joint_vec = np.zeros(3)
                joint_vec[axes[i]] = alpha[i]
                joint_rot = gtsam.Rot3.Expmap(joint_vec)
                r_R = r_R.compose(A_i).compose(joint_rot)
            else:
                #if it is the last transformation to foot the transformation is constant so no need to rotate it
                r_R = r_R.compose(A_i)
        
        return r_R.matrix()
    

    def _build_kinematic_constants(self, xml_path):
        """
        Build kinematic constants directly from MuJoCo A2 model.

        Returns:
        {
            leg_id: {
                "A": [...],
                "t": [...],
                "axes": [...]
            }
        }
        """

        model = mujoco.MjModel.from_xml_path(xml_path)
        leg_map = {
            0: "FL",
            1: "FR",
            2: "RL",
            3: "RR",
        }

        kinematics = {}

        for leg_id, prefix in leg_map.items():
            joint_names = [
                f"{prefix}_hip_joint",
                f"{prefix}_thigh_joint",
                f"{prefix}_calf_joint",
            ]

            body_names = [
                f"{prefix}_hip",
                f"{prefix}_thigh",
                f"{prefix}_calf",
            ]

            A = []
            t = []
            axes = []

            for body_name in body_names:
                body_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    body_name
                )

                if body_id == -1:
                    raise RuntimeError(f"Body not found: {body_name}")

                # translation parent -> child
                body_pos = model.body_pos[body_id].copy()

                # quaternion parent -> child
                body_quat = model.body_quat[body_id].copy()

                # quat -> rotation matrix
                R = np.zeros((3, 3))
                mujoco.mju_quat2Mat(
                    R.reshape(-1),
                    body_quat
                )

                A.append(R)
                t.append(body_pos)

            # joint axes
            for joint_name in joint_names:

                jid = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    joint_name
                )

                if jid == -1:
                    raise RuntimeError(f"Joint not found: {joint_name}")

                axis = model.jnt_axis[jid]

                axis_idx = int(np.argmax(np.abs(axis)))

                axes.append(axis_idx)

            # transform calf -> foot
            foot_site = prefix

            site_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_SITE,
                foot_site
            )

            if site_id == -1:
                raise RuntimeError(f"Foot site not found: {foot_site}")

            foot_pos = model.site_pos[site_id].copy()

            # foot rotation
            foot_R = np.eye(3)

            A.append(foot_R)
            t.append(foot_pos)

            kinematics[leg_id] = {
                "A": A,
                "t": t,
                "axes": axes
            }

        return kinematics
    
def skew(v):
    v = np.asarray(v).reshape(3)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def inverse_right_jacobian_so3(phi):
    """Inverse right Jacobian for SO(3).

    For small phi, use first-order approximation.
    """
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    phi_hat = skew(phi)

    if theta < 1e-8:
        return np.eye(3) - 0.5 * phi_hat

    half_theta = 0.5 * theta
    cot_half_theta = 1.0 / np.tan(half_theta)

    return (
        np.eye(3)
        - 0.5 * phi_hat
        + (1.0 - theta * cot_half_theta / 2.0)
        / (theta ** 2)
        * (phi_hat @ phi_hat)
    )
