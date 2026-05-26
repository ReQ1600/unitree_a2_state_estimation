import gtsam
import numpy as np
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
            r_fRi = gtsam.Rot3(fk_R).inverse().compose(R_i.inverse().compose(C_i)).logmap()

            #position residual calculation R^T * (d - p) - f_p (21)
            res_p = R_i.unrotate(d_i - p_i) - fk_p

            #TODO: calculate H jacobians for base and contact

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
        f_R_total = gtsam.Rot3()

        for n in range(len(alpha)):
            A_n = gtsam.Rot3(A[n])

            #alpha_n^dagger
            joint_axis_vec = np.zeros(3)
            joint_axis_vec[axes[n]] = alpha[n]
            joint_rot = gtsam.Rot3.Expmap(joint_axis_vec)

            #R = R * A_n * Exp(alpha_n^dagger)
            f_R_total = f_R_total.compose(A_n).compose(joint_rot)

        #transform to contact point
        f_R_total = f_R_total.compose(gtsam.Rot3(A[-1]))

        return f_R_total
        
    def _f_p(self, alpha):
        pass

    def _calculate_covariance(self, alpha):
        pass