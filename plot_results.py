"""Create figures and summaries from saved test results"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.baseline import denoise_tv
from src.data import make_image
from src.metrics import mean_squared_error, psnr
from src.modified import denoise_tv_adaptive


SUMMARY_FIELDS = [
    "noise_sigma",
    "paired_cases",
    "baseline_mean_mse",
    "baseline_mse_ci_lower",
    "baseline_mse_ci_upper",
    "adaptive_mean_mse",
    "adaptive_mse_ci_lower",
    "adaptive_mse_ci_upper",
    "paired_mse_gain_baseline_minus_adaptive",
    "paired_mse_gain_ci_lower",
    "paired_mse_gain_ci_upper",
    "paired_psnr_gain_adaptive_minus_baseline_db",
    "paired_psnr_gain_ci_lower_db",
    "paired_psnr_gain_ci_upper_db",
]


def _project_path(path):
    """Resolve a path from the project folder

    Input is an absolute or relative path
    Return the full path
    """
    project_root = Path(__file__).resolve().parent
    if path.is_absolute():
        return path
    return project_root / path


def _read_test_results(path):
    """Read and check paired test results

    Input is the path to a CSV file
    Return one dictionary for each test case
    """
    results_path = _project_path(path)
    try:
        with results_path.open("r", newline="", encoding="utf-8") as results_file:
            rows = list(csv.DictReader(results_file))
    except OSError as error:
        raise ValueError(f"Could not read results file {results_path}: {error}") from error

    if not rows:
        raise ValueError("The results CSV has no data rows")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("The results CSV must contain independent test cases")

    required = {
        "noise_sigma",
        "seed",
        "image_size",
        "iterations",
        "baseline_weight",
        "alpha",
        "baseline_mse",
        "adaptive_mse",
        "baseline_psnr",
        "adaptive_psnr",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"The results CSV is missing columns: {sorted(missing)}")

    return rows


def _one_value(rows, name):
    """Check that a field has one value in all rows

    Input is a list of result rows and a field name
    Return the field value as a float
    """
    values = {float(row[name]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"The results CSV has more than one {name} value")
    return values.pop()


def _bootstrap_mean_interval(values, rng, samples):
    """Estimate 95 percent bounds for the mean

    Input is a list of values and a random generator
    Return lower and upper bounds as floats
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 1:
        return float(values[0]), float(values[0])

    # Resample the values to estimate the mean bounds
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _summarise(rows, samples, random_seed):
    """Summarise paired metrics for each noise level

    Input is the test rows and bootstrap settings
    Return one summary dictionary for each noise level
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["noise_sigma"])].append(row)

    rng = np.random.default_rng(random_seed)
    summary = []
    for sigma in sorted(grouped):
        group = grouped[sigma]
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate test seeds found for sigma {sigma:g}")

        baseline_mse = np.asarray(
            [float(row["baseline_mse"]) for row in group], dtype=np.float64
        )
        adaptive_mse = np.asarray(
            [float(row["adaptive_mse"]) for row in group], dtype=np.float64
        )
        mse_gain = baseline_mse - adaptive_mse
        psnr_gain = np.asarray(
            [float(row["adaptive_psnr"]) - float(row["baseline_psnr"]) for row in group],
            dtype=np.float64,
        )

        baseline_ci = _bootstrap_mean_interval(baseline_mse, rng, samples)
        adaptive_ci = _bootstrap_mean_interval(adaptive_mse, rng, samples)
        mse_gain_ci = _bootstrap_mean_interval(mse_gain, rng, samples)
        psnr_gain_ci = _bootstrap_mean_interval(psnr_gain, rng, samples)

        summary.append(
            {
                "noise_sigma": sigma,
                "paired_cases": len(group),
                "baseline_mean_mse": float(np.mean(baseline_mse)),
                "baseline_mse_ci_lower": baseline_ci[0],
                "baseline_mse_ci_upper": baseline_ci[1],
                "adaptive_mean_mse": float(np.mean(adaptive_mse)),
                "adaptive_mse_ci_lower": adaptive_ci[0],
                "adaptive_mse_ci_upper": adaptive_ci[1],
                "paired_mse_gain_baseline_minus_adaptive": float(np.mean(mse_gain)),
                "paired_mse_gain_ci_lower": mse_gain_ci[0],
                "paired_mse_gain_ci_upper": mse_gain_ci[1],
                "paired_psnr_gain_adaptive_minus_baseline_db": float(np.mean(psnr_gain)),
                "paired_psnr_gain_ci_lower_db": psnr_gain_ci[0],
                "paired_psnr_gain_ci_upper_db": psnr_gain_ci[1],
            }
        )

    return summary


def _save_summary_figure(summary, path, baseline_weight, alpha, iterations):
    """Save the overall test comparison figure

    Input is the summary and the figure settings
    Save a PNG figure with confidence intervals
    """
    sigmas = np.asarray([row["noise_sigma"] for row in summary])
    x_positions = np.arange(sigmas.size)
    baseline_mean = np.asarray([row["baseline_mean_mse"] for row in summary])
    baseline_low = np.asarray([row["baseline_mse_ci_lower"] for row in summary])
    baseline_high = np.asarray([row["baseline_mse_ci_upper"] for row in summary])
    adaptive_mean = np.asarray([row["adaptive_mean_mse"] for row in summary])
    adaptive_low = np.asarray([row["adaptive_mse_ci_lower"] for row in summary])
    adaptive_high = np.asarray([row["adaptive_mse_ci_upper"] for row in summary])
    mse_gain = np.asarray(
        [row["paired_mse_gain_baseline_minus_adaptive"] for row in summary]
    )
    gain_low = np.asarray([row["paired_mse_gain_ci_lower"] for row in summary])
    gain_high = np.asarray([row["paired_mse_gain_ci_upper"] for row in summary])

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].errorbar(
        x_positions,
        baseline_mean,
        yerr=np.vstack((baseline_mean - baseline_low, baseline_high - baseline_mean)),
        marker="o",
        capsize=4,
        linewidth=2,
        label=f"Fixed weight ({baseline_weight:g})",
    )
    axes[0].errorbar(
        x_positions,
        adaptive_mean,
        yerr=np.vstack((adaptive_mean - adaptive_low, adaptive_high - adaptive_mean)),
        marker="s",
        capsize=4,
        linewidth=2,
        label=f"Noise adaptive (alpha={alpha:g})",
    )
    axes[0].set_title("Mean test MSE")
    axes[0].set_ylabel("MSE (lower is better)")
    axes[0].set_xticks(x_positions, [f"{sigma:g}" for sigma in sigmas])
    axes[0].set_xlabel("Noise standard deviation, sigma")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].errorbar(
        x_positions,
        mse_gain,
        yerr=np.vstack((mse_gain - gain_low, gain_high - mse_gain)),
        marker="o",
        capsize=4,
        linewidth=2,
        color="#2a7f62",
    )
    axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.6)
    axes[1].set_title("Paired MSE improvement")
    axes[1].set_ylabel("Fixed MSE minus adaptive MSE")
    axes[1].set_xticks(x_positions, [f"{sigma:g}" for sigma in sigmas])
    axes[1].set_xlabel("Noise standard deviation, sigma")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].text(
        0.03,
        0.97,
        "Positive values favour the adaptive method",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )

    case_counts = {row["paired_cases"] for row in summary}
    case_text = ", ".join(str(count) for count in sorted(case_counts))
    figure.suptitle(
        f"TV denoising on independent test cases\n"
        f"95% bootstrap intervals, {case_text} paired cases per noise level, "
        f"{iterations} iterations"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _save_example_figure(
    path, seed, sigma, image_size, iterations, baseline_weight, alpha
):
    """Save one normal test image comparison

    Input is the image seed noise level and denoising settings
    Save a PNG with clean noisy and reconstructed images
    """
    clean, noisy = make_image(seed=seed, size=image_size, sigma=sigma)
    baseline = denoise_tv(
        noisy.copy(), weight=baseline_weight, n_iter=iterations
    )
    adaptive, _, _ = denoise_tv_adaptive(
        noisy.copy(), alpha=alpha, n_iter=iterations
    )

    images = (clean, noisy, baseline, adaptive)
    titles = (
        "Clean image",
        f"Noisy\nMSE={mean_squared_error(clean, noisy):.5f}",
        f"Fixed weight\nMSE={mean_squared_error(clean, baseline):.5f}",
        f"Noise adaptive\nMSE={mean_squared_error(clean, adaptive):.5f}",
    )
    figure, axes = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)
    for axis, image, title in zip(axes, images, titles):
        axis.imshow(
            image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest"
        )
        axis.set_title(title)
        axis.axis("off")

    figure.suptitle(
        f"Representative test example: seed={seed}, sigma={sigma:g}, "
        f"weight={baseline_weight:g}, alpha={alpha:g}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _parse_args():
    """Read and check the figure settings

    Input comes from the command line
    Return all settings in one namespace
    """
    parser = argparse.ArgumentParser(
        description="Create reproducible figures from saved TV test results"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/test_alpha-1_weight-0.2.csv"),
        help="Paired test results CSV",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/test_overall_summary.csv"),
        help="Where to save the summary metrics and confidence intervals",
    )
    parser.add_argument(
        "--comparison-figure",
        type=Path,
        default=Path("outputs/test_overall_comparison.png"),
        help="Where to save the overall test comparison figure",
    )
    parser.add_argument(
        "--example-figure",
        type=Path,
        default=Path("outputs/test_representative_example.png"),
        help="Where to save the normal image example",
    )
    parser.add_argument("--example-seed", type=int, default=2000)
    parser.add_argument("--example-sigma", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=20261120)

    args = parser.parse_args()
    if args.example_seed < 0:
        parser.error("--example-seed must be non-negative")
    if not np.isfinite(args.example_sigma) or args.example_sigma < 0.0:
        parser.error("--example-sigma must be finite and non-negative")
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if args.random_seed < 0:
        parser.error("--random-seed must be non-negative")

    return args


def main():
    """Create the test figures and summary table

    Input comes from the command line
    Save two figures and one CSV file
    """
    args = _parse_args()
    try:
        rows = _read_test_results(args.results)
        summary = _summarise(rows, args.bootstrap_samples, args.random_seed)
        baseline_weight = _one_value(rows, "baseline_weight")
        alpha = _one_value(rows, "alpha")
        image_size = int(_one_value(rows, "image_size"))
        iterations = int(_one_value(rows, "iterations"))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    example_exists = any(
        int(row["seed"]) == args.example_seed
        and float(row["noise_sigma"]) == args.example_sigma
        for row in rows
    )
    if not example_exists:
        raise SystemExit(
            "The example seed and noise level must be present in the test results"
        )

    summary_path = _project_path(args.summary)
    comparison_path = _project_path(args.comparison_figure)
    example_path = _project_path(args.example_figure)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)

    _save_summary_figure(
        summary,
        comparison_path,
        baseline_weight,
        alpha,
        iterations,
    )
    # Recreate the example from its seed for a repeatable figure
    _save_example_figure(
        example_path,
        args.example_seed,
        args.example_sigma,
        image_size,
        iterations,
        baseline_weight,
        alpha,
    )

    print(f"Read paired test cases from {_project_path(args.results)}")
    print(f"Saved summary table to {summary_path}")
    print(f"Saved overall comparison figure to {comparison_path}")
    print(f"Saved representative image figure to {example_path}")
    for row in summary:
        print(
            f"Sigma {row['noise_sigma']:g}: paired MSE gain="
            f"{row['paired_mse_gain_baseline_minus_adaptive']:.6f}, "
            f"95% bootstrap interval="
            f"[{row['paired_mse_gain_ci_lower']:.6f}, "
            f"{row['paired_mse_gain_ci_upper']:.6f}]"
        )


if __name__ == "__main__":
    main()
