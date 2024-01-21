
import requests
import numpy as np
from io import BytesIO
from imageio.v3 import imread
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from transform import sum_directions, polar_to_cartesian

def plot_image(image, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.show()

def get_image(url):
    url_content = requests.get(url).content

    return imread(BytesIO(url_content))

def draw_rectangle(raw_image, rectangle_coords):
    _, ax = plt.subplots()
    ax.imshow(raw_image, cmap='gray')
    rect = patches.Rectangle((rectangle_coords[1], rectangle_coords[0]),
                             rectangle_coords[3] - rectangle_coords[1],
                             rectangle_coords[2] - rectangle_coords[0],
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.axis('off')
    plt.show()

def draw_arrow(ax, x, y, angle_degrees, length, color='red', arrowstyle='->', mutation_scale=15):
    # Compute the endpoint of the arrow
    end_x = x + length * np.cos(np.radians(angle_degrees))
    end_y = y + length * np.sin(np.radians(angle_degrees))

    arrowprops = dict(arrowstyle=arrowstyle, color=color, mutation_scale=mutation_scale)
    ax.annotate('', xy=(end_x, end_y), xytext=(x, y), arrowprops=arrowprops)

def draw_oriented_gradient(ax, i, j, patch_len, bin_index, bin_val, bin_delta):
    draw_arrow(ax,
                j * patch_len + (patch_len / 2) + 1,
                i * patch_len + (patch_len / 2) + 1,
                bin_index * bin_delta, bin_val)

    rect = patches.Rectangle((j * patch_len, i * patch_len),
                                (j + 1) * patch_len,
                                (i + 1) * patch_len,
                                linewidth=1, edgecolor='blue', facecolor='none')
    ax.add_patch(rect)

def draw_vector(ax, x, y, dx, dy, color='blue'):
    ax.arrow(x, y, dx, dy, head_width=1, head_length=1, fc=color, ec=color)

def draw_vector_gradient(image, patch_shape, hog, max_angle = 180, bin_size = 20):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    for i in range(0, (int)(image.shape[0]/patch_shape[0])):
        for j in range(0, (int)(image.shape[1]/patch_shape[1])):
            directions = hog[j * (int)(image.shape[0]/patch_shape[0]) + i]
            # polar = np.arange(0,360 + 40,40)
            polar = np.arange(0, max_angle + bin_size, bin_size)
            resultant_magnitude, resultant_angle = sum_directions(zip(polar,directions))
            resultant_magnitude/=7
            # for a, m in zip(polar,directions):
            #     dx, dy = polar_to_cartesian(a, m)
            #     draw_vector(ax,
            #                 i * patch_shape[0] + (patch_shape[0] / 2) - 1,
            #                 j * patch_shape[0] + (patch_shape[0] / 2) - 1,
            #                 dx, dy, color='lightblue')

            resultant_dx, resultant_dy = polar_to_cartesian(resultant_angle, resultant_magnitude)
            draw_vector(ax,
                        i * patch_shape[0] + (patch_shape[0] / 2),
                        j * patch_shape[0] + (patch_shape[0] / 2),
                        resultant_dx, resultant_dy, color='red')
    plt.show()