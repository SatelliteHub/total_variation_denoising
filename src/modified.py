
import numpy as np

if __package__:
    from .baseline import (
        denoise_tv,
        estimate_sigma,
        make_image,
        mean_squared_error,
        psnr,
    )
else:
    from baseline import (
        denoise_tv,
        estimate_sigma,
        make_image,
        mean_squared_error,
        psnr,
    )


# Modified method that adapts the TV weight to the estimated noise
# Tune alpha on validation images and relate the rule to cited parameter selection work


def _check_alpha(alpha):
    try:
        value = float(alpha)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("alpha must be a finite positive number.") from error

    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("alpha must be a finite positive number.")

    return value


def denoise_tv_adaptive(
    noisy_image,
    alpha,
    n_iter=200,
    tau=1.0 / 8.0,
    debug=False,
):
    # Estimate noise and use it to choose the regularisation weight
    # Keep Chambolle's baseline updates unchanged
    # Return the image, noise estimate and weight used
    alpha = _check_alpha(alpha)
    sigma_estimate = estimate_sigma(noisy_image)
    weight = max(alpha * sigma_estimate, 1e-8)

    if debug:
        print(
            f"Adaptive weight: noise estimate={sigma_estimate:.4f}, "
            f"alpha={alpha:.3f}, weight={weight:.4f}"
        )

    reconstructed = denoise_tv(
        noisy_image,
        weight=weight,
        n_iter=n_iter,
        tau=tau,
        debug=debug,
    )

    return reconstructed, sigma_estimate, weight


def run_demo(
    seed = 0,
    size = 96,
    sigma = 0.1,
    ###########################################################################
    baseline_weight = 0.10,     # Change value to see effect on baseline method
    alpha = 1.0,                # Change value to see effect on adaptive method
    ###########################################################################
    n_iter = 200,
    debug = True,
):
    # Compare both methods on the same noisy image
    # Demo values are starting points and should be chosen with validation images
    clean, noisy = make_image(seed=seed, size=size, sigma=sigma)
    baseline_result = denoise_tv(
        noisy,
        weight=baseline_weight,
        n_iter=n_iter,
        debug=debug,
    )
    adaptive_result, sigma_estimate, adaptive_weight = denoise_tv_adaptive(
        noisy,
        alpha=alpha,
        n_iter=n_iter,
        debug=debug,
    )

    print("Fixed weight baseline and noise adaptive method")
    print(f"Seed: {seed}; noise sigma: {sigma:.3f}")
    print(f"Estimated noise sigma: {sigma_estimate:.3f}")
    print(f"Baseline weight: {baseline_weight:.4f}")
    print(f"Adaptive alpha: {alpha:.3f}; adaptive weight: {adaptive_weight:.4f}")
    print(
        f"Noisy image: MSE={mean_squared_error(clean, noisy):.6f}, "
        f"PSNR={psnr(clean, noisy):.2f} dB"
    )
    print(
        f"Baseline: MSE={mean_squared_error(clean, baseline_result):.6f}, "
        f"PSNR={psnr(clean, baseline_result):.2f} dB"
    )
    print(
        f"Adaptive: MSE={mean_squared_error(clean, adaptive_result):.6f}, "
        f"PSNR={psnr(clean, adaptive_result):.2f} dB"
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to display the comparison images.")
    else:
        images = (clean, noisy, baseline_result, adaptive_result)
        titles = ("Clean image", "Noisy image", "Fixed weight", "Adaptive weight")
        fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
        for axis, image, title in zip(axes, images, titles):
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(title)
            axis.axis("off")
        fig.suptitle(
            f"TV denoising comparison at noise sigma={sigma:g}"
        )
        plt.show()

    return clean, noisy, baseline_result, adaptive_result


if __name__ == "__main__":
    run_demo()
