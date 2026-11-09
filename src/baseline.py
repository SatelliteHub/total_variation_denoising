#################################################################
# Baseline isotropic total-variation image denoising
#################################################################

import numpy as np

def check_image(image):
    img = np.asarray(image, dtype=np.float64)

    if img.ndim != 2:
        raise ValueError("Image must be a 2D greyscale array !")

    if img.size == 0:
        raise ValueError("Image must not be empty !")

    if not np.isfinite(img).all():
        raise ValueError("Image must contain finite values !")

    return img



# calculate brightness change between neighbour pixels
def grad(image):
    img = check_image(image)
    
    # Calculate the difference between each pixel and the pixel below it and to the right of it
    row_diff, col_diff = np.zeros_like(img)
    row_diff[:-1, :] = img[1:, :] - img[:-1., :]
    col_diff[:, :-1] = img[:, 1:] - img[:, :-1]

    return row_diff, col_diff


# Combine 2 pixel-wise components of a vector field into a single one
def div(px, py):
    px = check_image(px)
    py = check_image(py)

    result = np.zeros_like(px)

    # Add and substract the neighbour values to calculate the divergence
    result[:-1, :] += px[:-1, :]
    result[1:, :] -= px[:-1, :]
    result[:, :-1] += py[:, :-1]
    result[:, 1:] -= py[:, :-1]

    return result








