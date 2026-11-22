"""Display one comparison of fixed and adaptive TV denoising"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from modified import run_demo


# Run this part only when this file is started directly
if __name__ == "__main__":
    # Use settings that match the saved test example
    run_demo(
        seed=2000,
        size=96,
        sigma=0.10,
        baseline_weight=0.20,
        alpha=1.0,
        n_iter=200,
        debug=False,
    )
