from scipy.signal import convolve2d
from math import pi
import numpy as np

class Filter():
    def __init__(self, type, size) -> None:
        self.size = size
        self.type = type

    def __str__(self):
        return f'{self.type} filter [{self.size}x{self.size}]'
    
    def apply(self, image):
        return convolve2d(image, self._kernels[self.size],
                          boundary='symm', mode='same')

class Gaussian(Filter):
    _kernels = {
        3: 0.0625 * np.array([
                    [   1,  2,  1],
                    [   2,  4,  2],
                    [   1,  2,  1]
                ], dtype=np.float32),
        5: 0.003663004 * np.array([
                    [   1,  4,  7,  4,  1],
                    [   4, 16, 26, 16,  4],
                    [   7, 26, 41, 26,  7],
                    [   4, 16, 26, 16,  4],
                    [   1,  4,  7,  4,  1],
                ], dtype=np.float32)
    }
    
    def __init__(self, size = 3):
        super().__init__(type = 'Gaussian', size = size)

class Laplacian(Filter):
    _kernels = {
        3: np.array([
                    [   0,  1,  0],
                    [   1, -4,  1],
                    [   0,  1,  0]
                ], dtype=np.float32),
    }
    
    def __init__(self, size = 3):
        super().__init__(type = 'Laplacian', size = size)

class Sobel(Filter):
    _kernels = {
        'x': np.array([
                    [  -1,  0,  1],
                    [  -2,  0,  2],
                    [  -1,  0,  1]
                ], dtype=np.float32),
        'y': np.array([
                    [  -1, -2, -1],
                    [   0,  0,  0],
                    [   1,  2,  1]
                ], dtype=np.float32),
    }

    def __init__(self, size = 3):
        super().__init__(type = 'Sobel', size = size)

    def apply(self, image):
        dx = convolve2d(image, self._kernels['x'], boundary='symm', mode='same')
        dy = convolve2d(image, self._kernels['y'], boundary='symm', mode='same')

        mag = np.sqrt(dx**2 + dy**2)
        ang = np.arctan2(dy, dx) * 180 / pi

        return mag, ang