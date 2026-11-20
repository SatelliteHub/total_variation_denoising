"""Display one comparison of fixed and adaptive TV denoising"""

from src.modified import run_demo


if __name__ == "__main__":
    run_demo(
        seed=2000,
        size=96,
        sigma=0.10,
        baseline_weight=0.20,
        alpha=1.0,
        n_iter=200,
        debug=False,
    )
