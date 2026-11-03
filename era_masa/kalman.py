"""Linear Gaussian xyah model; noise scales follow the paper's DeepSORT setting.

This implements the prediction/correction equations independently. It does not
copy the GPL source in the upstream MASA-OA Kalman module.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .geometry import from_xyah


class BoxKalman:
    def __init__(self, measurement):
        self.mean = np.r_[measurement, np.zeros(4)].astype(float)
        h = max(measurement[3], 1e-3)
        self.covariance = np.diag(np.square([
            h / 10, h / 10, 1e-2, h / 10,
            h / 16, h / 16, 1e-5, h / 16]))
        self.transition = np.eye(8)
        self.transition[:4, 4:] = np.eye(4)

    def predict(self):
        h = max(self.mean[3], 1e-3)
        noise = np.diag(np.square([
            h / 20, h / 20, 1e-2, h / 20,
            h / 160, h / 160, 1e-5, h / 160]))
        self.mean = self.transition @ self.mean
        self.covariance = self.transition @ self.covariance @ self.transition.T + noise

    def project(self):
        h = max(self.mean[3], 1e-3)
        noise = np.diag(np.square([h / 20, h / 20, 0.1, h / 20]))
        return self.mean[:4], self.covariance[:4, :4] + noise

    def distance(self, measurements):
        mean, cov = self.project()
        diff = measurements - mean
        return np.maximum(np.einsum("ij,ji->i", diff, cho_solve(cho_factor(cov), diff.T)), 0)

    def update(self, measurement):
        mean, innovation_cov = self.project()
        gain = cho_solve(cho_factor(innovation_cov), self.covariance[:4, :]).T
        self.mean += gain @ (measurement - mean)
        # Joseph form keeps covariance symmetric and positive semidefinite.
        transform = np.eye(8)
        transform[:, :4] -= gain
        noise = innovation_cov - self.covariance[:4, :4]
        self.covariance = transform @ self.covariance @ transform.T + gain @ noise @ gain.T
        self.covariance = (self.covariance + self.covariance.T) / 2

    @property
    def box(self):
        return from_xyah(self.mean)
