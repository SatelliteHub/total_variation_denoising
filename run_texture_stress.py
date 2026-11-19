import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from src.baseline import denoise_tv
from src.data import make_texture_stress_image
from src.metrics import mean_squared_error, psnr
from src.modified import denoise_tv_adaptive


RESULT_FIELDS = [
    "seed",
    "noise_sigma",
    "image_size",
    "iterations",
    "baseline_weight",
    "alpha",
    "estimated_sigma",
    "adaptive_weight",
    "noisy_mse",
    "noisy_psnr",
    "noisy_texture_mse",
    "noisy_texture_psnr",
    "noisy_texture_retention",
    "noisy_texture_correlation",
    "baseline_mse",
    "adaptive_mse",
    "baseline_psnr",
    "adaptive_psnr",
    "baseline_texture_mse",
    "adaptive_texture_mse",
    "baseline_texture_psnr",
    "adaptive_texture_psnr",
    "clean_texture_std",
    "baseline_texture_retention",
    "adaptive_texture_retention",
    "baseline_texture_correlation",
    "adaptive_texture_correlation",
    "baseline_runtime_seconds",
    "adaptive_runtime_seconds",
]


def _project_path(path):
    project_root = Path(__file__).resolve().parent
    if path.is_absolute():
        return path
    return project_root / path


def _load_config(path):
    config_path = _project_path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read parameter config {config_path}: {error}") from error

    if not isinstance(config, dict) or config.get("selection_split") != "validation":
        raise ValueError("The parameter config must contain validation-selected values")

    for name in ("baseline_weight", "alpha", "iterations", "image_size"):
        if name not in config:
            raise ValueError(f"The parameter config is missing {name}")

    return config


def _timed_run(function, repeats):
    function()
    durations = []
    result = None

    for _ in range(repeats):
        start = perf_counter()
        result = function()
        durations.append(perf_counter() - start)

    return result, median(durations)


def _texture_retention(clean, result, texture_region):
    clean_patch = clean[texture_region]
    result_patch = result[texture_region]
    clean_std = float(np.std(clean_patch))
    result_std = float(np.std(result_patch))
    if clean_std == 0.0:
        return float("nan"), float("nan")

    retention = result_std / clean_std
    if result_std == 0.0:
        correlation = 0.0
    else:
        correlation = float(
            np.corrcoef(clean_patch.ravel(), result_patch.ravel())[0, 1]
        )

    return retention, correlation


def _save_example(path, clean, noisy, baseline, adaptive, texture_region, seed, sigma):
    images = (clean, noisy, baseline, adaptive)
    titles = ("Clean", "Noisy", "Fixed weight", "Noise adaptive")
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)

    for column, (image, title) in enumerate(zip(images, titles)):
        axes[0, column].imshow(
            image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")

        axes[1, column].imshow(
            image[texture_region],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[1, column].axis("off")

    axes[1, 0].set_ylabel("Texture patch", rotation=90)
    figure.suptitle(f"Checkerboard stress case, sigma={sigma:g}, seed={seed}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare TV denoisers on synthetic images with fine texture"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("outputs/selected_parameters.json"),
        help="Validation-selected parameters",
    )
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--num-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=3000)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/texture_stress_results.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("outputs/texture_stress_example.png"),
    )

    args = parser.parse_args()
    if not np.isfinite(args.sigma) or args.sigma < 0.0:
        parser.error("--sigma must be finite and non-negative")
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.seed_start < 0:
        parser.error("--seed-start must be non-negative")
    if args.size is not None and args.size < 32:
        parser.error("--size must be at least 32")
    if args.iterations is not None and args.iterations < 0:
        parser.error("--iterations must not be negative")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1")

    return args


def main():
    args = _parse_args()
    try:
        config = _load_config(args.config)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    baseline_weight = float(config["baseline_weight"])
    alpha = float(config["alpha"])
    iterations = int(
        config["iterations"] if args.iterations is None else args.iterations
    )
    image_size = int(config["image_size"] if args.size is None else args.size)
    if not np.isfinite(baseline_weight) or baseline_weight <= 0.0:
        raise SystemExit("The configured baseline weight must be finite and positive")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise SystemExit("The configured alpha must be finite and positive")

    output_path = _project_path(args.output)
    figure_path = _project_path(args.figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Checkerboard stress case: sigma={args.sigma:g}, image size={image_size}, "
        f"weight={baseline_weight:g}, alpha={alpha:g}, iterations={iterations}"
    )
    print(
        f"Seeds: {args.seed_start} to {args.seed_start + args.num_seeds - 1}"
    )

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.num_seeds):
        print(f"Running checkerboard stress case with seed {seed}")
        clean, noisy, texture_region = make_texture_stress_image(
            seed=seed,
            size=image_size,
            sigma=args.sigma,
        )

        def run_baseline():
            return denoise_tv(
                noisy.copy(),
                weight=baseline_weight,
                n_iter=iterations,
            )

        def run_adaptive():
            return denoise_tv_adaptive(
                noisy.copy(),
                alpha=alpha,
                n_iter=iterations,
            )

        baseline_result, baseline_runtime = _timed_run(
            run_baseline, args.timing_repeats
        )
        (adaptive_result, sigma_estimate, adaptive_weight), adaptive_runtime = (
            _timed_run(run_adaptive, args.timing_repeats)
        )

        baseline_texture_retention, baseline_texture_correlation = _texture_retention(
            clean, baseline_result, texture_region
        )
        adaptive_texture_retention, adaptive_texture_correlation = _texture_retention(
            clean, adaptive_result, texture_region
        )
        clean_patch = clean[texture_region]
        noisy_patch = noisy[texture_region]
        baseline_patch = baseline_result[texture_region]
        adaptive_patch = adaptive_result[texture_region]
        noisy_texture_retention, noisy_texture_correlation = _texture_retention(
            clean, noisy, texture_region
        )

        rows.append(
            {
                "seed": seed,
                "noise_sigma": args.sigma,
                "image_size": image_size,
                "iterations": iterations,
                "baseline_weight": baseline_weight,
                "alpha": alpha,
                "estimated_sigma": sigma_estimate,
                "adaptive_weight": adaptive_weight,
                "noisy_mse": mean_squared_error(clean, noisy),
                "noisy_psnr": psnr(clean, noisy),
                "noisy_texture_mse": mean_squared_error(clean_patch, noisy_patch),
                "noisy_texture_psnr": psnr(clean_patch, noisy_patch),
                "noisy_texture_retention": noisy_texture_retention,
                "noisy_texture_correlation": noisy_texture_correlation,
                "baseline_mse": mean_squared_error(clean, baseline_result),
                "adaptive_mse": mean_squared_error(clean, adaptive_result),
                "baseline_psnr": psnr(clean, baseline_result),
                "adaptive_psnr": psnr(clean, adaptive_result),
                "baseline_texture_mse": mean_squared_error(
                    clean_patch, baseline_patch
                ),
                "adaptive_texture_mse": mean_squared_error(
                    clean_patch, adaptive_patch
                ),
                "baseline_texture_psnr": psnr(clean_patch, baseline_patch),
                "adaptive_texture_psnr": psnr(clean_patch, adaptive_patch),
                "clean_texture_std": float(np.std(clean_patch)),
                "baseline_texture_retention": baseline_texture_retention,
                "adaptive_texture_retention": adaptive_texture_retention,
                "baseline_texture_correlation": baseline_texture_correlation,
                "adaptive_texture_correlation": adaptive_texture_correlation,
                "baseline_runtime_seconds": baseline_runtime,
                "adaptive_runtime_seconds": adaptive_runtime,
            }
        )

        if seed == args.seed_start:
            _save_example(
                figure_path,
                clean,
                noisy,
                baseline_result,
                adaptive_result,
                texture_region,
                seed,
                args.sigma,
            )

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved stress results to {output_path}")
    print(f"Saved example figure to {figure_path}")
    print(
        "Mean whole-image MSE: "
        f"baseline={np.mean([row['baseline_mse'] for row in rows]):.6f}, "
        f"adaptive={np.mean([row['adaptive_mse'] for row in rows]):.6f}"
    )
    print(
        "Mean checkerboard-region MSE: "
        f"baseline={np.mean([row['baseline_texture_mse'] for row in rows]):.6f}, "
        f"adaptive={np.mean([row['adaptive_texture_mse'] for row in rows]):.6f}"
    )
    print(
        "Mean checkerboard contrast retention (1 means unchanged): "
        f"baseline={np.mean([row['baseline_texture_retention'] for row in rows]):.3f}, "
        f"adaptive={np.mean([row['adaptive_texture_retention'] for row in rows]):.3f}"
    )
    print(
        "Mean checkerboard correlation (1 means matching texture): "
        f"baseline={np.mean([row['baseline_texture_correlation'] for row in rows]):.3f}, "
        f"adaptive={np.mean([row['adaptive_texture_correlation'] for row in rows]):.3f}"
    )


if __name__ == "__main__":
    main()
