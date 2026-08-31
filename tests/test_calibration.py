import unittest

import numpy as np

from sculptor.npe.calibration import empirical_coverage


class CalibrationTests(unittest.TestCase):
    def test_empirical_coverage_returns_expected_keys(self) -> None:
        truths = np.array([[0.0, 0.0], [1.0, 1.0]])
        samples = np.repeat(truths[:, None, :], 32, axis=1)
        coverage = empirical_coverage(samples, truths)
        self.assertIn(0.9, coverage)
        self.assertGreaterEqual(coverage[0.9], 1.0)

    def test_empirical_coverage_is_marginal_not_joint(self) -> None:
        rng = np.random.default_rng(0)
        truths = rng.normal(size=(512, 10))
        samples = rng.normal(size=(512, 256, 10))
        coverage = empirical_coverage(samples, truths)
        self.assertGreater(coverage[0.9], 0.82)
        self.assertLess(coverage[0.9], 0.98)


if __name__ == "__main__":
    unittest.main()
