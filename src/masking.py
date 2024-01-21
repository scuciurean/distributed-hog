from scipy.ndimage import rotate
import numpy as np

def apply_mask_and_extract(matrix1, mask, inverted_mask=False):
    h1, w1 = matrix1.shape
    h2, w2 = mask.shape

    start_row = (h1 - h2) // 2
    end_row = start_row + h2
    start_col = (w1 - w2) // 2
    end_col = start_col + w2

    masked_matrix = np.zeros_like(matrix1)
    masked_matrix[start_row:end_row, start_col:end_col] = matrix1[start_row:end_row, start_col:end_col] * ((1 - mask) if inverted_mask else (255 - mask))

    result = masked_matrix[start_row:end_row, start_col:end_col]

    return result

def rotate(mat, angle, offset):
    height, width = mat.shape
    center_x, center_y = width // 2, height // 2
    y, x = np.indices((height, width)) - np.array([center_y, center_x])[:, np.newaxis, np.newaxis]

    y -= offset

    y_rot = np.round(y * np.cos(np.radians(angle)) - x * np.sin(np.radians(angle))) + center_y
    x_rot = np.round(y * np.sin(np.radians(angle)) + x * np.cos(np.radians(angle))) + center_x

    y_rot, x_rot = y_rot.astype(int), x_rot.astype(int)
    valid_mask = (y_rot >= 0) & (y_rot < height) & (x_rot >= 0) & (x_rot < width)

    rotated_mat = np.zeros_like(mat)
    rotated_mat[valid_mask] = mat[y_rot[valid_mask], x_rot[valid_mask]]

    return rotated_mat

def distribute(mask, gamma, delta, slope, angle, offset, inverted_mask=False, num_ranges=10):
    height, width = mask.shape[0]*2, mask.shape[1]*2
    result_mat = np.zeros((height, width))

    def map_gamma(x, gamma):
        return int((x / gamma) ** (.25*(slope + 1)) * gamma)

    for x in range(0, width, delta):
        for y in range(0, height, delta):
            if x < width and y < height:
                mapped_x = map_gamma(x, gamma)
                mapped_x = min(mapped_x, width - 1)
                result_mat[y, mapped_x] = 255
    result_mat = group_even(result_mat, num_ranges)
    rotated_mat = rotate(result_mat, angle, offset)
    masked = apply_mask_and_extract(rotated_mat, mask, inverted_mask=inverted_mask)

    result_masked = np.where(masked > 0, masked, 0).astype(np.uint8)

    return result_masked

def group_even(mat, num_ranges):
    result = np.where(mat > 0, 1, 0)
    waves = []
    for x in range(0, mat.shape[0] - 1):
        for y in range(0, mat.shape[1] -1):
            if result[x, y] != 0:
                waves.append(y)

    waves = list(set(waves))
    waves.sort()

    indices = np.linspace(0, len(waves), num_ranges + 1, dtype=int)
    ranges = [waves[indices[i]:indices[i + 1]] for i in range(num_ranges)]
    classes = np.arange(1, num_ranges + 1)

    for index, _range in enumerate(ranges):
        for item in _range:
            result[:, item] *= classes[index]

    return result