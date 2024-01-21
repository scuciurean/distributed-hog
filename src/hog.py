import numpy as np

from data_handling import split_into_blocks

def gradient_patch(patch, bin_size=9, normalize=False):
    # max_angle = 180
    max_angle = 360
    bin_delta = (int)(max_angle / bin_size)
    ranges = np.arange(0, max_angle, bin_delta)
    bins = [(x, x + bin_delta) for x in ranges]
    mag, ang = patch
    gradient = np.zeros(bin_size)

    for angle, magnitude in zip(ang.flatten(), mag.flatten()):
        #corner cases
        if angle == max_angle:
            gradient[0] += magnitude
        elif angle % bin_delta == 0:
            gradient[(int)(angle / bin_delta)] += magnitude

        for bin_range in bins:
            if angle > bin_range[0] and angle < bin_range[1]:
                delta_left = angle - bin_range[0]
                delta_right = bin_range[1] - angle
                bin_left = delta_left / bin_delta * magnitude
                bin_right = delta_right / bin_delta * magnitude
                gradient[(int)(bin_range[0] / bin_delta)] += bin_left
                if angle < max_angle - bin_delta:
                    gradient[(int)(bin_range[1] / bin_delta)] += bin_right
                else: #another corner case
                    gradient[0] += bin_right
    
    if normalize and gradient.max() != 0:
        gradient *= patch.shape[1]/gradient.max()*.75

    return gradient

def histogram_of_oriented_gradients(magnitudes, angles, patch_size=(8,8), normalize=False):

    def _adapt_angle(angle):
        # return angle + 180 if angle < 0 else angle - 180 if angle > 180 else angle
        return angle + 180

    _angles = np.vectorize(_adapt_angle)(angles)
    # _magnitudes = magnitudes / magnitudes.sum(axis=1)[:, np.newaxis]
    _magnitudes = magnitudes
    mag_blocks = split_into_blocks(_magnitudes, patch_size)
    ang_blocks = split_into_blocks(_angles, patch_size)
    patches = np.empty((mag_blocks.shape[0], 2, mag_blocks.shape[1], mag_blocks.shape[2]))
    for index in range(0, len(mag_blocks)):
        patches[index][0] = mag_blocks[index]
        patches[index][1] = ang_blocks[index]

    return np.array([gradient_patch(patch, normalize=normalize) for patch in patches])
