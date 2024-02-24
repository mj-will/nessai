"""Integration test for the reparameterisation in GWFlowProposal"""

from nessai.gw.proposal import GWFlowProposal
import numpy as np
import pytest


@pytest.mark.requires("bilby")
@pytest.mark.requires("lal")
@pytest.mark.requires("astropy")
@pytest.mark.integration_test
@pytest.mark.parametrize("reverse", [False, True])
def test_invertibility(
    tmp_path, get_bilby_gw_model, injection_parameters, reverse
):
    """Make sure the default GW reparameterisation is invertible.

    Uses the `verify_rescaling` method from `FlowProposal`.

    Also checks if the number of unique samples is the same as the input. This
    will depend on the order of the reparameterisations when using inversion.
    """
    model = get_bilby_gw_model(
        list(injection_parameters.keys()), injection_parameters
    )
    output = tmp_path / "test"
    output.mkdir()

    proposal = GWFlowProposal(
        model=model,
        poolsize=1000,
        output=output,
        reverse_reparameterisations=reverse,
    )

    proposal.set_rescaling()
    proposal.verify_rescaling()

    reparam = proposal._reparameterisation
    assert reparam.order == list(reparam.keys())

    if not reverse:
        pytest.xfail(
            "Checking unique samples with reverse=False will always fail."
        )
    # Check the number of unique samples
    n = 10
    x = model.new_point(n)

    x_prime, _ = proposal.rescale(x)
    x_out, _ = proposal.inverse_rescale(x_prime)

    flags = {n: False for n in proposal.names}
    for name in proposal.names:
        flags[name] = np.unique(x_out[name].round(decimals=10)).size == n
    if not all(flags.values()):
        msg = f"""Number of unique samples has changed.
        Breakdown per parameter: {flags}
        """
        raise AssertionError(msg)
