import numpy as np

def split_into_blocks(image, block_shape):
    image_height, image_width = image.shape[:2]
    block_height, block_width = block_shape

    num_blocks_vertical = image_height // block_height
    num_blocks_horizontal = image_width // block_width

    blocks = []

    # Iterate through the blocks and append them to the list
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = image[i * block_height: (i + 1) * block_height, 
                          j * block_width: (j + 1) * block_width]
            blocks.append(block)

    return np.array(blocks)

def feature_stack(image, representation, block_shape):

    samples, _ = representation.extract(image)
    feature_set = []
    for level, sample in enumerate(samples):
        blocks = split_into_blocks(sample, block_shape)
        print(f"Level {level} got {len(blocks)} subsamples")
        for index, feature in enumerate(blocks):
            x_max = sample.shape[0] / block_shape[0]
            y_max = sample.shape[1] / block_shape[1]
            x0 = round(index / x_max)
            y0 = round(index % y_max)
            position = (level, x0, y0)
            feature_set.append((feature, position))

    return feature_set
