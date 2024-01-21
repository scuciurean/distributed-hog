from math import floor
import numpy as np
import sys

def decimate(image, factor = 2):
    if factor <= 0:
        raise ValueError("Decimation factor should be a positive integer.")

    height = image.shape[0] // factor
    width = image.shape[1] // factor

    reshaped_image = image[:height * factor, :width * factor].reshape((height, factor, width, factor))
    decimated_image = reshaped_image.mean(axis=(1, 3))

    return decimated_image

def expand(image, factor = 2):
    if factor <= 0:
        raise ValueError("Expansion factor should be a positive integer.")

    new_height = image.shape[0] * factor
    new_width = image.shape[1] * factor

    expanded_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            expanded_image[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor] = image[i, j]

    return expanded_image

def residual(original_image, upsampled_image):
    if original_image.shape != upsampled_image.shape:
        raise ValueError("Original and upsampled images must have the same dimensions.")

    residuals = original_image - upsampled_image

    return residuals

def threshold(image, threshold_value, max_value):
    return np.where(image > threshold_value, max_value, 0)

class Pyramid:
    def __init__(self, levels, filter, kernel_size = 3):
        self.levels = levels
        if filter != None:
            self.filter_type = getattr(sys.modules[__name__], filter)
            self.filter = self.filter_type(kernel_size)
        else:
            self.filter = None

    @property
    def filter_name(self):
        return self.filter_type.__name__ if self.filter else 'Unfiltered'

    def __str__(self):
        return f"{self.filter_name} pyramid [{self.levels} levels]"

    def _decompose(self, image):
        if len(image.shape) != 2:
            raise Exception("wrong input image")
        sample = image
        if self.filter != None:
            filtered = self.filter.apply(image)
            sample = filtered

            if self.filter_name == 'Sobel':
                sample = filtered[0]

        decimated = decimate(sample)
        expanded = expand(decimated)
        residuals = residual(sample, expanded)

        return decimated, residuals

    def extract(self, image):
        samples = []
        residuals = []

        target = image
        for _ in range(0, self.levels):
            dec, res = self._decompose(target)
            if dec.shape[0] % 2 != 0 or dec.shape[1] % 2 != 0:
                dec = dec[0:(int)(floor(dec.shape[0] / 2) * 2), 0:(int)(floor(dec.shape[1] / 2) * 2)]
            samples.append(dec)
            residuals.append(res)
            target = dec

        return samples, residuals
