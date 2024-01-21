import numpy as np

def convert_to_grayscale(image):
    # Convert the image to grayscale using Rec.ITU-R BT.601-7 formula
    # Y = 0.2989 * R + 0.5870 * G + 0.1440 * B
    return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1440 * image[:, :, 2]

def polar_to_cartesian(angle, magnitude):
    angle_rad = np.radians(angle)
    x_component = magnitude * np.cos(angle_rad)
    y_component = magnitude * np.sin(angle_rad)
    return x_component, y_component

def sum_directions(directions):
    vectors = [polar_to_cartesian(angle % 360, magnitude) for angle, magnitude in directions]
    result_vector = np.sum(vectors, axis=0)
    magnitude = np.linalg.norm(result_vector)
    angle = np.degrees(np.arctan2(result_vector[1], result_vector[0])) % 360
    # angle = np.degrees(np.arctan2(result_vector[1], result_vector[0])) % 180
    return magnitude, angle