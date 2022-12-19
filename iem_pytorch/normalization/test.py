import numpy as np


class TestNorm():
    def __init__(self, norm_path):
        self.mean = np.asarray([50, 100]).reshape(2, 1)
        self.scale = np.asarray([100, 100]).reshape(2, 1)

    def norm(self, x, mean, scale):
        return (x - mean) / scale

    def denorm(self, x, mean, scale):
        return x * scale + mean
