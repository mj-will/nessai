from unittest.mock import MagicMock, create_autospec
from nessai.flows.distributions import ResampledGaussian


def test_finalise():
    """Test the finalise method"""

    dist = create_autospec(ResampledGaussian)
    dist.estimate_normalisation_constant = MagicMock()

    ResampledGaussian.finalise(dist, n_samples=100, n_batches=10)

    dist.estimate_normalisation_constant.assert_called_once_with(
        n_samples=100,
        n_batches=10,
    )
