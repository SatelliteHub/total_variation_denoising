"""Run paired TV denoising experiments"""

import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from src.baseline import denoise_tv
from src.data import make_image
from src.metrics import mean_squared_error, psnr
from src.modified import denoise_tv_adaptive


# Keep the random seeds separate for each type of experiment
SEED_STARTS = {
    "development": 0,
    "validation": 1000,
    "test": 2000,
}

CSV_FIELDS = [
    "split",
    "noise_sigma",
    "seed",
    "image_size",
    "iterations",
    "timing_repeats",
    "baseline_weight",
    "alpha",
    "estimated_sigma",
    "adaptive_weight",
    "noisy_mse",
    "noisy_psnr",
    "baseline_mse",
    "baseline_psnr",
    "adaptive_mse",
    "adaptive_psnr",
    "baseline_runtime_seconds",
    "adaptive_runtime_seconds",
    "mse_gain_baseline_minus_adaptive",
    "psnr_gain_adaptive_minus_baseline",
]


def _timed_run(function, repeats):
    """Measure repeated runs of one function

    Input is a function with no arguments and a repeat count
    Return the last result and the median time in seconds
    """
    # Warm up the function before measuring its speed
    function()
    durations = []
    result = None

    # Repeat the timed run to reduce the effect of one slow measurement
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        durations.append(perf_counter() - start)

    return result, median(durations)


def _bootstrap_interval(values, rng, samples=5000):
    """Estimate a confidence interval for the mean

    Input is a list of paired differences and a random generator
    Return lower and upper 95 percent bounds
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan"), float("nan")

    # Select new groups of results to estimate variation in the mean
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = np.mean(values[indices], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])

    return float(lower), float(upper)


def _parse_args():
    """Read and check the experiment settings

    Input comes from the command line
    Return all settings in one namespace
    """
    parser = argparse.ArgumentParser(
        description="Compare fixed-weight and noise-adaptive TV denoising"
    )
    parser.add_argument(
        "--split",
        choices=tuple(SEED_STARTS),
        default="development",
        help="Development, validation or final test cases",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="First random seed for this run",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=20,
        help="Number of paired cases at each noise level",
    )
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
        help="Noise standard deviations for the generated images",
    )
    parser.add_argument(
        "--baseline-weight",
        type=float,
        default=0.10,
        help="Fixed weight used by the baseline method",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scale factor used by the adaptive method",
    )
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--timing-repeats",
        type=int,
        default=3,
        help="Timed runs for each method and case after one warm-up",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path for experiment or tuning results",
    )
    parser.add_argument(
        "--tune-validation",
        action="store_true",
        help="Choose parameters using validation images, then save them",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file with parameters selected on validation images",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("outputs/selected_parameters.json"),
        help="Where to save the selected validation parameters",
    )
    parser.add_argument(
        "--baseline-candidates",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20],
        help="Fixed weights to compare during validation tuning",
    )
    parser.add_argument(
        "--alpha-candidates",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Adaptive scale factors to compare during validation tuning",
    )

    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.timing_repeats < 1:
        parser.error("--timing-repeats must be at least 1")
    if args.size < 32:
        parser.error("--size must be at least 32")
    if args.iterations < 0:
        parser.error("--iterations must not be negative")
    if not np.isfinite(args.baseline_weight) or args.baseline_weight < 0.0:
        parser.error("--baseline-weight must be finite and non-negative")
    if not np.isfinite(args.alpha) or args.alpha <= 0.0:
        parser.error("--alpha must be finite and positive")
    if any(not np.isfinite(sigma) or sigma < 0.0 for sigma in args.sigmas):
        parser.error("Every noise level must be finite and non-negative")
    if len(set(args.sigmas)) != len(args.sigmas):
        parser.error("Noise levels must not contain duplicates")
    if any(not np.isfinite(value) or value < 0.0 for value in args.baseline_candidates):
        parser.error("Every baseline candidate must be finite and non-negative")
    if any(not np.isfinite(value) or value <= 0.0 for value in args.alpha_candidates):
        parser.error("Every alpha candidate must be finite and positive")
    if len(set(args.baseline_candidates)) != len(args.baseline_candidates):
        parser.error("Baseline candidates must not contain duplicates")
    if len(set(args.alpha_candidates)) != len(args.alpha_candidates):
        parser.error("Alpha candidates must not contain duplicates")
    if args.seed_start is not None and args.seed_start < 0:
        parser.error("--seed-start must be non-negative")
    if args.tune_validation and args.config is not None:
        parser.error("Do not use --config while tuning validation parameters")
    if args.split == "test" and args.config is None and not args.tune_validation:
        parser.error("A test run needs --config from a validation tuning run")

    return args


def _project_path(path):
    """Resolve a path from the project folder

    Input is an absolute or relative path
    Return the full path
    """
    # Save relative paths inside the project folder
    project_root = Path(__file__).resolve().parent
    if path.is_absolute():
        return path
    return project_root / path


def _candidate_scores(method, candidates, cases, args, case_writer):
    """Score candidate settings on validation images

    Input contains candidate values and paired validation cases
    Return the mean MSE score for each candidate
    """
    score_rows = []

    # Try each possible setting on the same validation images
    for candidate in candidates:
        name = "baseline weight" if method == "baseline" else "adaptive alpha"
        print(f"Tuning {name} {candidate:g}")
        mse_by_sigma = {sigma: [] for sigma in args.sigmas}

        for sigma, seed, clean, noisy in cases:
            # Score one candidate on one paired image case
            if method == "baseline":
                denoised = denoise_tv(
                    noisy.copy(),
                    weight=candidate,
                    n_iter=args.iterations,
                )
            else:
                denoised, _, _ = denoise_tv_adaptive(
                    noisy.copy(),
                    alpha=candidate,
                    n_iter=args.iterations,
                )

            case_mse = mean_squared_error(clean, denoised)
            mse_by_sigma[sigma].append(case_mse)
            case_writer.writerow(
                {
                    "method": method,
                    "parameter_value": candidate,
                    "noise_sigma": sigma,
                    "seed": seed,
                    "mse": case_mse,
                }
            )

        mean_mse_by_sigma = {
            str(sigma): float(np.mean(values))
            for sigma, values in mse_by_sigma.items()
        }
        # Give each noise level equal weight in the selection score
        macro_mean_mse = float(np.mean(list(mean_mse_by_sigma.values())))
        score_rows.append(
            {
                "value": float(candidate),
                "macro_mean_mse": macro_mean_mse,
                "mean_mse_by_sigma": mean_mse_by_sigma,
            }
        )

    return score_rows


def tune_validation_parameters(args):
    """Select settings using validation images

    Input is the experiment settings
    Save case scores and selected values
    Return the selected values and their scores
    """
    # Tuning must use validation data and not the final test cases
    args.split = "validation"
    seed_start = args.seed_start
    if seed_start is None:
        seed_start = SEED_STARTS["validation"]

    # Create each validation image once so candidates see the same cases
    cases = []
    for sigma in args.sigmas:
        for seed in range(seed_start, seed_start + args.num_seeds):
            print(f"Preparing validation case with seed {seed} and sigma {sigma:.2f}")
            clean, noisy = make_image(seed=seed, size=args.size, sigma=sigma)
            cases.append((sigma, seed, clean, noisy))

    tuning_csv_path = (
        _project_path(args.output)
        if args.output is not None
        else _project_path(Path("outputs/validation_tuning.csv"))
    )
    tuning_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the candidate scores so the parameter choice can be reviewed
    with tuning_csv_path.open("w", newline="", encoding="utf-8") as output_file:
        fields = ["method", "parameter_value", "noise_sigma", "seed", "mse"]
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()

        baseline_scores = _candidate_scores(
            "baseline", args.baseline_candidates, cases, args, writer
        )
        adaptive_scores = _candidate_scores(
            "adaptive", args.alpha_candidates, cases, args, writer
        )

    # Choose the settings with the lowest validation error
    best_baseline = min(
        baseline_scores, key=lambda row: (row["macro_mean_mse"], row["value"])
    )
    best_adaptive = min(
        adaptive_scores, key=lambda row: (row["macro_mean_mse"], row["value"])
    )

    config = {
        "selection_split": "validation",
        "selection_metric": "mean MSE averaged equally across noise levels",
        "seed_start": seed_start,
        "num_seeds": args.num_seeds,
        "sigmas": [float(sigma) for sigma in args.sigmas],
        "image_size": args.size,
        "iterations": args.iterations,
        "baseline_weight": best_baseline["value"],
        "alpha": best_adaptive["value"],
        "baseline_candidate_scores": baseline_scores,
        "alpha_candidate_scores": adaptive_scores,
    }

    # Save selected settings for the final test run
    config_path = _project_path(args.config_output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")

    print(f"\nValidation tuning results saved to {tuning_csv_path}")
    print(f"Selected parameters saved to {config_path}")
    print(
        f"Selected baseline weight: {config['baseline_weight']:.4g} "
        f"(validation mean MSE {best_baseline['macro_mean_mse']:.6f})"
    )
    print(
        f"Selected adaptive alpha: {config['alpha']:.4g} "
        f"(validation mean MSE {best_adaptive['macro_mean_mse']:.6f})"
    )
    return config


def load_parameter_config(path):
    """Load settings selected on validation images

    Input is the path to a JSON file
    Return the selected settings as a dictionary
    """
    # Read the saved validation settings from a JSON file
    config_path = _project_path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read parameter config {config_path}: {error}") from error

    if not isinstance(config, dict):
        raise ValueError("The parameter config must contain a JSON object")
    # Do not accept settings selected using the final test cases
    if config.get("selection_split") != "validation":
        raise ValueError("The parameter config must record validation as its selection split")
    for name in ("baseline_weight", "alpha"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"The parameter config needs a finite {name}")
        if not np.isfinite(value):
            raise ValueError(f"The parameter config needs a finite {name}")
    if config["baseline_weight"] < 0.0 or config["alpha"] <= 0.0:
        raise ValueError("The selected weight and alpha must be non-negative and positive")

    return config


def run_experiments(args):
    """Run paired denoising cases and save the scores

    Input is the experiment settings
    Save one CSV row for each seed and noise level
    """
    seed_start = args.seed_start
    if seed_start is None:
        seed_start = SEED_STARTS[args.split]

    if args.output is None:
        output_path = _project_path(
            Path(
                f"outputs/{args.split}_alpha-{args.alpha:g}_weight-"
                f"{args.baseline_weight:g}.csv"
            )
        )
    else:
        output_path = _project_path(args.output)

    # Reuse each noisy image for both denoising methods
    rows = []
    for sigma in args.sigmas:
        for seed in range(seed_start, seed_start + args.num_seeds):
            print(f"Running {args.split} case with seed {seed} and sigma {sigma:.2f}")
            clean, noisy = make_image(seed=seed, size=args.size, sigma=sigma)

            def run_baseline():
                """Run baseline denoising on the current image

                Return an image with the same shape as the input
                """
                return denoise_tv(
                    noisy.copy(),
                    weight=args.baseline_weight,
                    n_iter=args.iterations,
                )

            def run_adaptive():
                """Run adaptive denoising on the current image

                Return the image noise estimate and selected weight
                """
                return denoise_tv_adaptive(
                    noisy.copy(),
                    alpha=args.alpha,
                    n_iter=args.iterations,
                )

            # Measure both methods on this same noisy image
            baseline_result, baseline_runtime = _timed_run(
                run_baseline, args.timing_repeats
            )
            (adaptive_result, sigma_estimate, adaptive_weight), adaptive_runtime = (
                _timed_run(run_adaptive, args.timing_repeats)
            )

            # Compare each result with the known clean image
            baseline_mse = mean_squared_error(clean, baseline_result)
            baseline_psnr = psnr(clean, baseline_result)
            adaptive_mse = mean_squared_error(clean, adaptive_result)
            adaptive_psnr = psnr(clean, adaptive_result)

            # Store scores and timing together as one paired result row
            rows.append(
                {
                    "split": args.split,
                    "noise_sigma": sigma,
                    "seed": seed,
                    "image_size": args.size,
                    "iterations": args.iterations,
                    "timing_repeats": args.timing_repeats,
                    "baseline_weight": args.baseline_weight,
                    "alpha": args.alpha,
                    "estimated_sigma": sigma_estimate,
                    "adaptive_weight": adaptive_weight,
                    "noisy_mse": mean_squared_error(clean, noisy),
                    "noisy_psnr": psnr(clean, noisy),
                    "baseline_mse": baseline_mse,
                    "baseline_psnr": baseline_psnr,
                    "adaptive_mse": adaptive_mse,
                    "adaptive_psnr": adaptive_psnr,
                    "baseline_runtime_seconds": baseline_runtime,
                    "adaptive_runtime_seconds": adaptive_runtime,
                    "mse_gain_baseline_minus_adaptive": baseline_mse - adaptive_mse,
                    "psnr_gain_adaptive_minus_baseline": adaptive_psnr - baseline_psnr,
                }
            )

    # Create the output folder before writing the CSV file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Use a fixed seed so the bootstrap intervals can be repeated
    bootstrap_rng = np.random.default_rng(20261119)
    print(f"\nSaved paired results to {output_path}")
    print("Positive MSE and PSNR gains favour the adaptive method")

    for sigma in args.sigmas:
        group = [row for row in rows if row["noise_sigma"] == sigma]
        mse_gains = [row["mse_gain_baseline_minus_adaptive"] for row in group]
        psnr_gains = [row["psnr_gain_adaptive_minus_baseline"] for row in group]
        # Estimate uncertainty in the average paired improvements
        mse_interval = _bootstrap_interval(mse_gains, bootstrap_rng)
        psnr_interval = _bootstrap_interval(psnr_gains, bootstrap_rng)
        mean_baseline_time = float(
            np.mean([row["baseline_runtime_seconds"] for row in group])
        )
        mean_adaptive_time = float(
            np.mean([row["adaptive_runtime_seconds"] for row in group])
        )

        print(f"\nSigma {sigma:.2f}, {len(group)} paired cases")
        print(
            f"Mean MSE gain: {np.mean(mse_gains):.6f}, "
            f"95% bootstrap interval: [{mse_interval[0]:.6f}, {mse_interval[1]:.6f}]"
        )
        print(
            f"Mean PSNR gain: {np.mean(psnr_gains):.3f} dB, "
            f"95% bootstrap interval: [{psnr_interval[0]:.3f}, {psnr_interval[1]:.3f}] dB"
        )
        print(
            f"Mean runtime: baseline={mean_baseline_time:.4f}s, "
            f"adaptive={mean_adaptive_time:.4f}s"
        )


# Read command line settings only when this file is run as a script
if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.tune_validation:
        selected = tune_validation_parameters(arguments)
        arguments.baseline_weight = selected["baseline_weight"]
        arguments.alpha = selected["alpha"]
        arguments.output = None
        run_experiments(arguments)
    else:
        if arguments.config is not None:
            selected = load_parameter_config(arguments.config)
            arguments.baseline_weight = selected["baseline_weight"]
            arguments.alpha = selected["alpha"]
            print(
                f"Loaded validation parameters from {_project_path(arguments.config)}"
            )
        run_experiments(arguments)
