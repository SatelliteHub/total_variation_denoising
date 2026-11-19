import numpy as np


# Check that both images are finite greyscale arrays with the same shape
def _check_image_pair(reference_image, estimate_image):
    reference = np.asarray(reference_image, dtype=np.float64)
    estimate = np.asarray(estimate_image, dtype=np.float64)

    if reference.ndim != 2 or estimate.ndim != 2:
        raise ValueError("Both images must be 2D greyscale arrays.")
    if reference.size == 0 or estimate.size == 0:
        raise ValueError("Images must not be empty.")
    if not np.isfinite(reference).all() or not np.isfinite(estimate).all():
        raise ValueError("Images must contain only finite values.")
    if reference.shape != estimate.shape:
        raise ValueError("The two images must have the same shape.")

    return reference, estimate


# Find the average squared difference between two images
def mean_squared_error(reference_image, estimate_image):
    reference, estimate = _check_image_pair(reference_image, estimate_image)

    return float(np.mean((estimate - reference) ** 2))


# Return peak signal to noise ratio in decibels
# An exact match has infinite PSNR
def psnr(reference_image, estimate_image, data_range=1.0):
    try:
        data_range = float(data_range)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("data_range must be a finite positive number.") from error
    if not np.isfinite(data_range) or data_range <= 0.0:
        raise ValueError("data_range must be a finite positive number.")

    error = mean_squared_error(reference_image, estimate_image)
    if error == 0.0:
        return float("inf")

    return float(10.0 * np.log10((data_range**2) / error))
