import unittest

import numpy as np

from src.baseline import denoise_tv, div, grad
from src.modified import denoise_tv_adaptive


class TestFiniteDifferenceOperators(unittest.TestCase):
    def test_gradient_on_small_image(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]])

        row_diff, col_diff = grad(image)

        np.testing.assert_array_equal(
            row_diff,
            np.array([[2.0, 2.0], [0.0, 0.0]]),
        )
        np.testing.assert_array_equal(
            col_diff,
            np.array([[1.0, 0.0], [1.0, 0.0]]),
        )

    def test_divergence_on_small_fields(self):
        px = np.array([[0.1, 0.2], [0.0, 0.0]])
        py = np.array([[0.3, 0.0], [0.4, 0.0]])

        result = div(px, py)

        np.testing.assert_allclose(
            result,
            np.array([[0.4, -0.1], [0.3, -0.6]]),
        )

    def test_divergence_is_negative_adjoint_of_gradient(self):
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        px = np.array([[0.1, 0.2], [0.0, 0.0]])
        py = np.array([[0.3, 0.0], [0.4, 0.0]])
        row_diff, col_diff = grad(image)

        gradient_product = np.sum(row_diff * px + col_diff * py)
        divergence_product = -np.sum(image * div(px, py))

        self.assertAlmostEqual(gradient_product, divergence_product)


class TestDenoisers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.noisy = 0.5 + 0.15 * rng.normal(size=(8, 9))

    def test_constant_image_stays_constant(self):
        image = np.full((8, 9), 0.37)

        result = denoise_tv(image, weight=0.2, n_iter=25)

        np.testing.assert_allclose(result, image, atol=1e-12)

    def test_baseline_returns_finite_image_without_changing_input(self):
        original = self.noisy.copy()

        result = denoise_tv(self.noisy, weight=0.2, n_iter=20)

        self.assertEqual(result.shape, original.shape)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(self.noisy, original)

    def test_adaptive_returns_finite_image_without_changing_input(self):
        original = self.noisy.copy()

        result, sigma_estimate, weight = denoise_tv_adaptive(
            self.noisy,
            alpha=1.0,
            n_iter=20,
        )

        self.assertEqual(result.shape, original.shape)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.isfinite(sigma_estimate))
        self.assertTrue(np.isfinite(weight))
        self.assertGreater(weight, 0.0)
        np.testing.assert_array_equal(self.noisy, original)


if __name__ == "__main__":
    unittest.main()
