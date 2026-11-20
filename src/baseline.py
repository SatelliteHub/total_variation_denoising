# Baseline isotropic TV denoising with the Chambolle dual method
# Images are 2D greyscale arrays
"""Tools for baseline total variation denoising"""

import numpy as np

from src.data import estimate_sigma, make_image
from src.metrics import mean_squared_error, psnr


def check_image(image):
    """Check a greyscale image

    Input is a 2D array with shape height by width
    Return a finite float array with the same shape
    """
    img = np.asarray(image, dtype=np.float64)

    if img.ndim != 2:
        raise ValueError("Image must be a 2D greyscale array.")
    if img.size == 0:
        raise ValueError("Image must not be empty.")
    if not np.isfinite(img).all():
        raise ValueError("Image must contain only finite values.")

    return img


def _check_weight(weight):
    """Check the regularisation weight

    Input is one finite positive number
    Return the weight as a float
    """
    # Convert numeric text as well as number inputs
    try:
        value = float(weight)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("weight must be a finite positive number.") from error

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("weight must be a finite positive number.")

    return value


def grad(image):
    """Find changes between neighbouring pixels

    Input is a 2D image
    Return row and column differences with the same shape
    The last row and column of differences are zero
    """
    img = check_image(image)

    row_diff = np.zeros_like(img)
    col_diff = np.zeros_like(img)

    # Compare each pixel with the pixel below it
    row_diff[:-1, :] = img[1:, :] - img[:-1, :]
    # Compare each pixel with the pixel to its right
    col_diff[:, :-1] = img[:, 1:] - img[:, :-1]

    return row_diff, col_diff


def div(px, py):
    """Combine row and column fields into one image

    Input fields must be 2D arrays with the same shape
    Return one divergence array with that shape
    """
    px = check_image(px)
    py = check_image(py)
    if px.shape != py.shape:
        raise ValueError("px and py must have the same shape.")

    result = np.zeros_like(px)

    # Add vertical flow to one pixel and subtract it from the next row
    result[:-1, :] += px[:-1, :]
    result[1:, :] -= px[:-1, :]
    # Apply the same rule to horizontal flow and the next column
    result[:, :-1] += py[:, :-1]
    result[:, 1:] -= py[:, :-1]

    return result


def isotropic_tv(image):
    """Measure total variation in a greyscale image

    Input is a 2D image
    Return one total variation value as a float
    """
    row_diff, col_diff = grad(image)
    # Combine both directional changes at each pixel
    gradient_lengths = np.hypot(row_diff, col_diff)
    total_variation = np.sum(gradient_lengths)

    return float(total_variation)


def denoise_tv(noisy_image, weight, n_iter, tau=1.0 / 8.0, debug=False):
    """Denoise an image with the fixed point method by Chambolle

    Input is a noisy 2D image and the regularisation settings
    Return a denoised float array with the same shape
    """
    noisy = check_image(noisy_image)
    weight = _check_weight(weight)

    if isinstance(n_iter, (bool, np.bool_)) or not isinstance(
        n_iter, (int, np.integer)
    ):
        raise ValueError("n_iter must be a non-negative integer.")
    if n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")

    try:
        tau = float(tau)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("tau must be positive and no larger than 1/8.") from error
    if not np.isfinite(tau) or tau <= 0.0 or tau > 1.0 / 8.0:
        raise ValueError("tau must be positive and no larger than 1/8.")

    # Store the two parts of the dual field for the row and column directions
    px = np.zeros_like(noisy)
    py = np.zeros_like(noisy)

    if debug:
        print(
            f"TV debug start: shape={noisy.shape}, weight={weight:.4f}, "
            f"tau={tau:.4f}, iterations={n_iter}"
        )
        print(
            f"TV debug input: finite={np.isfinite(noisy).all()}, "
            f"range=[{noisy.min():.4f}, {noisy.max():.4f}], "
            f"mean={noisy.mean():.6f}"
        )

    # Print progress about ten times when debugging is switched on
    progress_step = max(1, n_iter // 10)
    for iteration in range(1, n_iter + 1):
        # Combine the dual field with the noisy image before taking its gradient
        field = div(px, py) - noisy / weight
        gx, gy = grad(field)

        # Find the gradient length at every pixel
        magnitude = np.hypot(gx, gy)
        # Scale each update so the dual vector stays within its allowed length
        denominator = 1.0 + tau * magnitude
        next_px = (px + tau * gx) / denominator
        next_py = (py + tau * gy) / denominator

        if debug and (
            iteration == 1
            or iteration % progress_step == 0
            or iteration == n_iter
        ):
            max_update = np.max(np.hypot(next_px - px, next_py - py))
            max_vector_length = np.max(np.hypot(next_px, next_py))
            print(
                f"TV debug iteration {iteration}/{n_iter}: "
                f"largest update={max_update:.3e}, "
                f"largest vector length={max_vector_length:.6f}"
            )

        px, py = next_px, next_py

    # Use the final dual field to recover the denoised image
    reconstructed = noisy - weight * div(px, py)

    if debug:
        mean_change = reconstructed.mean() - noisy.mean()
        print(
            f"TV debug output: finite={np.isfinite(reconstructed).all()}, "
            f"range=[{reconstructed.min():.4f}, {reconstructed.max():.4f}]"
        )
        print(
            f"TV debug mean check: input={noisy.mean():.6f}, "
            f"output={reconstructed.mean():.6f}, change={mean_change:.3e}"
        )
        print(
            f"TV debug variation check: input={isotropic_tv(noisy):.4f}, "
            f"output={isotropic_tv(reconstructed):.4f}"
        )
        print(
            f"TV debug objective check: input={rof_objective(noisy, noisy, weight):.4f}, "
            f"output={rof_objective(noisy, reconstructed, weight):.4f}"
        )

    return reconstructed


def rof_objective(noisy_image, reconstructed_image, weight):
    """Score the data fit and image smoothness

    Input images must have the same 2D shape
    Return the ROF objective as a float
    """
    noisy = check_image(noisy_image)
    reconstructed = check_image(reconstructed_image)
    if noisy.shape != reconstructed.shape:
        raise ValueError("The two images must have the same shape.")
    weight = _check_weight(weight)

    # Add the fit cost to the weighted smoothness cost
    data_cost = 0.5 * np.sum((reconstructed - noisy) ** 2)
    smoothness_cost = weight * isotropic_tv(reconstructed)

    return float(data_cost + smoothness_cost)


def run_demo(seed=0, size=96, sigma=0.1, weight=0.12, n_iter=200, debug=True):
    """Show one baseline denoising example

    Input settings choose the image and denoising strength
    Print image scores and show the comparison figure
    """
    clean, noisy = make_image(seed=seed, size=size, sigma=sigma)
    sigma_estimate = estimate_sigma(noisy)
    reconstructed = denoise_tv(
        noisy, weight=weight, n_iter=n_iter, debug=debug
    )

    print("Chambolle isotropic TV denoising baseline")
    print(f"Seed: {seed}; image size: {size} x {size}; iterations: {n_iter}")
    print(f"Noise sigma: {sigma:.3f}; estimated sigma: {sigma_estimate:.3f}")
    print(f"Fixed regularisation weight: {weight:.3f}")
    print(
        "Noisy image:     "
        f"MSE={mean_squared_error(clean, noisy):.6f}, "
        f"PSNR={psnr(clean, noisy):.2f} dB, TV={isotropic_tv(noisy):.2f}"
    )
    print(
        "Reconstruction:  "
        f"MSE={mean_squared_error(clean, reconstructed):.6f}, "
        f"PSNR={psnr(clean, reconstructed):.2f} dB, "
        f"TV={isotropic_tv(reconstructed):.2f}"
    )
    print(
        "ROF objective:   "
        f"noisy={rof_objective(noisy, noisy, weight):.2f}, "
        f"reconstructed={rof_objective(noisy, reconstructed, weight):.2f}"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to display the three example images.")
    else:
        images = (clean, noisy, reconstructed)
        titles = ("Clean image", "Noisy image", "TV reconstruction")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for axis, image, title in zip(axes, images, titles):
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"Isotropic TV denoising (weight={weight:g}, iterations={n_iter})"
        )
        plt.show()

    return clean, noisy, reconstructed


if __name__ == "__main__":
    run_demo()
