import gtsam
import numpy as np
import scipy
from ..factor_registry import BaseFactor

class ForwardKinematicFactor(BaseFactor):
    def __init__(self, leg_id, encoder_sigma):
        """
        leg_id: leg identifier (0-3 for a2)
        encoder_sigma: encoder noise standard deviation
        """

        self.leg_id = leg_id
        self.encoder_sigma = encoder_sigma
    
    def add_initial_estimate(self, vals, step_id, sensor_data, ctx):
        """
        contact frame(Ci, di) position and orientation init
        """

        # base and leg contact keys  
        base_key = gtsam.symbol('x', step_id)
        contact_key = gtsam.symbol('c', self.leg_id * 1000 + step_id)

        #init based on current base estimation and measurements
        if not vals.exsists(contact_key):
            if vals.exists(base_key):
                base_pose = vals.atPose3(base_key)
                alpha = sensor_data['joint_positions'][self.leg_id]

                #calculated based on a2 model
                R_bc = self._f_R(alpha)
                p_bc = self._f_p(alpha)

                # C = R * f_R
                # d = p + R * f_p
                contact_rot = base_pose.rotation().compose(gtsam.Rot3(R_bc))
                contact_pos = base_pose.translation() + base_pose.rotation()

                vals.insert(contact_key, gtsam.Pose(contact_rot), contact_pos)
    
    def add_to_graph(self, graph, values, step_id, sensor_data, ctx):
        """
        adds gtsam.CustomFactor to the graph calculating residues f_Ri and f_pi 
        """

        # base and leg contact keys  
        base_key = gtsam.symbol('x', step_id)
        contact_key = gtsam.symbol('c', self.leg_id * 1000 + step_id)

        leg_encoder_data = sensor_data['joint_positions'][self.leg_id]

        fk_R = self._f_R(leg_encoder_data)
        fk_p = self._f_p(leg_encoder_data)
        covariance = self._calculate_covariance(leg_encoder_data)
        noise_model = gtsam.noiseModel.Covariance(covariance)

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
            r_R = gtsam.Rot3(fk_R).inverse().compose(R_i.inverse().compose(C_i)).logmap()

            #position residual calculation R^T * (d - p) - f_p (21)
            r_p = R_i.unrotate(d_i - p_i) - fk_p

            #H jacobian calculation
            #H0 - jacobian with respect to the base
            H0 = np.zeros((6, 6))

            #rotation with respect to the base rotation
            H0[0:3, 0:3] = -gtsam.Rot3.InverseRightJacobian(r_R) @ C_i.transpose().compose(R_i).matrix()
            
            # position with respect tot the base rotation
            H0[3:6, 0:3] = gtsam.skewSymmetric(R_i.unrotate(d_i - p_i))

            #H1 - jacobian with respect to the contact frame
            H1 = np.zeros((6, 6))

            #rotation with respect to the contact rotation
            H1[0:3, 0:3] = gtsam.Rot3.InverseRightJacobian(r_R)

            #position with respect to the contact translaction
            H1[3:6, 3:6] = R_i.transpose().compose(C_i).matrix()

            if H is not None:
                H = H0
                H[4] = H1

            factor = gtsam.CustomFactor(noise_model, [base_key, contact_key], err_func)
            graph.add(factor)
    
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
        
        sigma_alpha_dagger = scipy.linalg(*sigma_blocks)

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
                t_hat = gtsam.skewSymmetric(tn_plus_1)
                
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
                res_R = res_R.compose(A_i)
        
        return res_R.compose(A_i)