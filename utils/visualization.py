import os
import cv2
import numpy as np


def make_tile_image(images, num_cols, num_rows):
    num_images, height, width, ch = images.shape
    assert num_images == num_cols * num_rows, "Images must have the same number of rows * columns"
    tile = np.zeros((height * num_rows, width * num_cols, ch), dtype=np.uint8)

    for r in range(num_rows):
        for c in range(num_cols):
            tile[r * height: (r + 1) * height, (c * width): (c + 1) * width, :] = images[r * num_cols + c]

    return tile

def write_image(image, path, image_index, pre_fix=None, image_name=None):
    if image_name is not None:
        cv2.imwrite(os.path.join(path, image_name), image)
        return

    if pre_fix is None:
        cv2.imwrite(os.path.join(path, '%03d.png' % image_index), image)
    else:
        cv2.imwrite(os.path.join(path, '%s_%03d.png' % (pre_fix, image_index)), image)

