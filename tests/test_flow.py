import unittest

from sculptor.npe.embedding import EmbeddingConfig, SpectrumEmbeddingNet, torch
from sculptor.npe.flow import ConditionalFlow, FlowConfig


class FlowTests(unittest.TestCase):
    def test_optional_training_dependency_is_guarded(self) -> None:
        if torch is None:
            with self.assertRaises(ImportError):
                SpectrumEmbeddingNet(EmbeddingConfig(input_length=1024))
            with self.assertRaises(ImportError):
                ConditionalFlow(FlowConfig(parameter_dim=10))


if __name__ == "__main__":
    unittest.main()
