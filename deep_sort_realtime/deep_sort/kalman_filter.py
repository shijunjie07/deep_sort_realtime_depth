import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 10-dimensional state space

        x, y, a, h, d, vx, vy, va, vh, vd

    contains the bounding box center position (x, y), depth d, aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        
        dt = 1.0     # time step
        ndim = 5     # number of measurements

        # motion model matrix setup
        self._motion_mat = np.eye(2 * ndim, 2 * ndim) # state transition matrix A
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim) # measurement matrix H

        # standard deviations for position and velocity
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        
    def initiate(self, measurement):
        # initialise the state and covariance when a new object is detected
        mean_pos = measurement    # [x,y,a,h,d]
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        # standard deviations for uncertainty initialisation
        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,                                            # a
            2 * self._std_weight_position * measurement[3],  # h
            2 * self._std_weight_position * measurement[3],  # d
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            1e-5,                                            # va
            10 * self._std_weight_velocity * measurement[3], # vh
            10 * self._std_weight_velocity * measurement[3], # vd
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance
    
    def predict(self, mean, covariance):
        # predict next state using the motion model
        std_pos = [
            self._std_weight_position * mean[3], # x
            self._std_weight_position * mean[3], # y
            1e-2,                                # a
            self._std_weight_position * mean[3], # h
            self._std_weight_position * mean[3], # d
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],  # vx
            self._std_weight_velocity * mean[3],  # vy
            1e-5,                                 # va
            self._std_weight_velocity * mean[3],  # vh
            self._std_weight_velocity * mean[3],  # vd
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        # 
        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        return mean, covariance

    def project(self, mean, covariance):
        # project the state space to measurement space
        std = [
            self._std_weight_position * mean[3], # x
            self._std_weight_position * mean[3], # y
            1e-1,                                # a
            self._std_weight_position * mean[3], # h
            self._std_weight_position * mean[3], # d
        ]
        innovation_cov = np.diag(np.square(std))
        
        # measurement mean and covariance
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        ) + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        # update step: measurement update
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Lower triangular matrix L, Bool of whether returned a L
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        # compute the kalman gain
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(kalman_gain.T, innovation)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements, only_position=False):
        # compute the gating distance between state space and measurements
        # Mahalanobis distance
        
        # project predicted state into measurement space
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # use only x y
        if only_position:
            projected_mean, projected_cov = projected_mean[:2], projected_cov[:2]
            # measurements is 2D
            measurements = measurements[:, :2]
            
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_trangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z*z, axis=0)
        return squared_maha
