import unittest

import numpy as np

from sculptor.config import get_default_track_config
from sculptor.npe.training import resolve_parameter_layout, split_parameter_array


class TrainingLayoutTests(unittest.TestCase):
    def test_conditioning_parameters_are_excluded_from_targets(self) -> None:
        config = get_default_track_config("g395h")
        layout = resolve_parameter_layout(config.parameter_names, config.conditioning_parameter_names)
        self.assertEqual(layout.conditioning_names, ("snr",))
        self.assertEqual(len(layout.target_names), 10)
        self.assertNotIn("snr", layout.target_names)

        values = np.arange(22, dtype=np.float32).reshape(2, 11)
        targets, conditioning = split_parameter_array(values, layout)
        self.assertEqual(targets.shape, (2, 10))
        self.assertEqual(conditioning.shape, (2, 1))
        np.testing.assert_array_equal(conditioning[:, 0], values[:, -1])
