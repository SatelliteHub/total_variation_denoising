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



# calculates one number measuring total brightness change in the image
# summing the lengths of the gradient vectors at each pixel
def isotropic_tv(image):
    row_diff, col_diff = grad(image)
    total_variation = np.sum(np.hypot(row_diff, col_diff)) #np.hypot is sqrt(row_diff**2 + col_diff**2)

    return float(total_variation)


# Denoise a grayscale image using Chambolle's dual fixed-point method
def denoise_tv(noisy_image, weight, n_iter, tau=1.0 / 8.0):
    # Weight is the regularisation length (lambda)
    # n_iter is the number of update steps
    # tau is the step size; this discretisation uses 0 < tau <= 1/8

    noisy = check_image(noisy_image)

    # p is a temporary arrays: auxiliary vector field with two values at every pixel
    px = np.zeros_like(noisy)
    py = np.zeros_like(noisy)

    # Repeat the update step n_iter times by adjusting the helper vector field p at each pixel
    for _ in range(n_iter):
        field = div(px, py) - noisy / weight
        gx, gy = grad(field)

        # Normalise each two-component update
        # This keeps the length of p at each pixel within one
        magnitude = np.hypot(gx, gy)
        denominator = 1.0 + tau * magnitude
        px = (px + tau * gx) / denominator
        py = (py + tau * gy) / denominator

    # Recover the denoised image from the final vector field
    return noisy - weight * div(px, py)


# Give one score for how well the reconstructed image fits the noisy observation and how smooth it is
# ROF is the Rudin-Osher-Fatemi model (see Chambolle paper) for image denoising
# which is a combination of a data term and a total variation (TV) regularization term
def rof_objective(noisy_image, reconstructed_image, weight):
    noisy = check_image(noisy_image)
    reconstructed = check_image(reconstructed_image)

    # The data term rewards closeness to the noisy observation
    data_cost = 0.5 * np.sum((reconstructed - noisy) ** 2)

    # The TV term rewards images with less total brightness change
    smoothness_cost = weight * isotropic_tv(reconstructed)

    return float(data_cost + smoothness_cost)


