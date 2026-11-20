"""Measure image reconstruction quality"""

import numpy as np

def _check_image_pair(reference_image, estimate_image):
    """Check two greyscale images

    Input images must be finite 2D arrays with the same shape
    Return both images as float arrays
    """
    # Convert both inputs to arrays of decimal numbers
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


def mean_squared_error(reference_image, estimate_image):
    """Calculate mean squared error (MSE) between two images

    Input images must have the same 2D shape
    Return one score where a lower value is better
    """
    reference, estimate = _check_image_pair(reference_image, estimate_image)

    # Square pixel errors so positive and negative errors do not cancel
    squared_errors = (estimate - reference) ** 2
    # Average all pixel errors to get one image score
    return float(np.mean(squared_errors))


def psnr(reference_image, estimate_image, data_range=1.0):
    """Calculate peak signal to noise ratio (PSNR)

    Input images must have the same 2D shape and the data range must be positive
    Return one score in decibels where a higher value is better
    An exact match returns infinity
    """
    try:
        data_range = float(data_range)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("data_range must be a finite positive number.") from error
    if not np.isfinite(data_range) or data_range <= 0.0:
        raise ValueError("data_range must be a finite positive number.")

    error = mean_squared_error(reference_image, estimate_image)
    if error == 0.0:
        # A perfect match has infinite PSNR
        return float("inf")

    # Convert the squared error into a decibel score
    return float(10.0 * np.log10((data_range**2) / error))
