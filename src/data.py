import numpy as np


# Make clean image with four rectangles and two disks
# Add Gaussian noise without clipping the pixel values
# Return the clean and noisy images with shape size by size
def make_image(seed, size=96, sigma=0.1):
    if isinstance(size, (bool, np.bool_)) or not isinstance(
        size, (int, np.integer)
    ):
        raise ValueError("size must be an integer of at least 32.")
    size = int(size)
    if size < 32:
        raise ValueError("size must be at least 32.")

    try:
        sigma = float(sigma)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("sigma must be a finite non-negative number.") from error
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be a finite non-negative number.")

    rng = np.random.default_rng(seed)
    clean = np.full((size, size), 0.15, dtype=np.float64)
    yy, xx = np.mgrid[:size, :size]

    for _ in range(4):
        row, col = rng.integers(0, size - 20, size=2)
        height, width = rng.integers(10, 22, size=2)
        clean[row : row + height, col : col + width] = rng.uniform(0.3, 0.9)

    for _ in range(2):
        cy, cx = rng.integers(15, size - 15, size=2)
        radius = rng.integers(6, 14)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius**2
        clean[mask] = rng.uniform(0.3, 0.9)

    noisy = clean + sigma * rng.normal(size=clean.shape)

    return clean, noisy


# Add a small checkerboard texture to the centre of a synthetic image
# Keep the same Gaussian noise across the whole image
# Return the clean image, noisy image and texture region
def make_texture_stress_image(seed, size=96, sigma=0.1):
    clean, noisy = make_image(seed=seed, size=size, sigma=sigma)
    noise = noisy - clean

    patch_size = min(size, max(16, size // 3))
    start = (size - patch_size) // 2
    stop = start + patch_size
    texture_region = (slice(start, stop), slice(start, stop))

    row, col = np.mgrid[:patch_size, :patch_size]
    checkerboard = ((row // 2 + col // 2) % 2) * 2 - 1
    texture_patch = 0.5 + 0.16 * checkerboard

    clean[texture_region] = texture_patch
    noisy[texture_region] = texture_patch + noise[texture_region]

    return clean, noisy, texture_region


# Estimate noise from alternating horizontal pixel pairs
# Edges and texture can affect the estimate
def estimate_sigma(noisy_image):
    noisy = np.asarray(noisy_image, dtype=np.float64)

    if noisy.ndim != 2:
        raise ValueError("Image must be a 2D greyscale array.")
    if noisy.size == 0:
        raise ValueError("Image must not be empty.")
    if not np.isfinite(noisy).all():
        raise ValueError("Image must contain only finite values.")
    if noisy.shape[1] < 2:
        raise ValueError("Noise estimation needs an image at least two columns wide.")

    usable_width = 2 * (noisy.shape[1] // 2)
    differences = (
        noisy[:, 1:usable_width:2] - noisy[:, :usable_width:2]
    ) / np.sqrt(2.0)
    centre = np.median(differences)
    mad = np.median(np.abs(differences - centre))

    return float(mad / 0.67449)
