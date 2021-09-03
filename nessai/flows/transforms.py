# -*- coding: utf-8 -*-
"""
Transforms for constructing flows.
"""
import math
from typing import Callable, Tuple, Union

from nflows.transforms import Transform
from nflows.transforms.coupling import CouplingTransform
from nflows.utils import torchutils
import torch


class NLSqCouplingTransform(CouplingTransform):
    """Non-Linear Squared Coupling Transform

    Reference: Ziegler & Rush 2019, arXiv:1901.10548

    Based of the original implementation:\
        https://github.com/harvardnlp/TextFlow

    Parameters
    ----------
    mask : Union[torch.Tensor, list, tuple]
        Mask defining which inputs will be transformed. > 0 implies the input
        will be transformed and < 0 implies it will remain unchanged.
    transform_net_create_fn : typing.Callable
        Function to create the NN that outputs the transform parameters.
    unconditional_transform : torch.transform.Transform
        Transform applied to the non-transformed parameters.
    alpha : float
        Constant included for stability as described in Appendix C of the
        reference.
    """

    def __init__(
        self,
        mask: Union[torch.Tensor, list, tuple],
        transform_net_create_fn: Callable,
        unconditional_transform: Transform = None,
        alpha: float = 0.95,
    ) -> None:

        super().__init__(
            mask,
            transform_net_create_fn,
            unconditional_transform=unconditional_transform,
        )
        self.log_c_const = math.log(8.0 * math.sqrt(3.0) / 9.0 * alpha)

    def _transform_dim_multiplier(self) -> int:
        return 5

    def _get_params(
        self, transform_params: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # Input tensor will have shape [batch_size, 5 * n_transform_features]
        # Split along the second dim into 5.
        a, logb, c_p, logd, g = torch.split(
            transform_params, self.num_transform_features, dim=1
        )
        b = torch.exp(logb)
        d = torch.exp(logd)
        c = torch.tanh(c_p) * torch.exp(self.log_c_const + logb - logd)
        return a, b, c, d, g

    def _get_derivative(
        self,
        arg: torch.Tensor,
        denom: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        return torchutils.sum_except_batch(
            torch.log(b - 2 * c * d * arg / denom.pow(2)), num_batch_dims=1
        )

    def _coupling_transform_forward(
        self,
        inputs: torch.Tensor,
        transform_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = self._get_params(transform_params)
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        g = g.double()
        inputs = inputs.double()

        # Four components of the cubic equation (Appendix C)
        # Follows G. C. Holmes 2002 for solving polynomials
        # a=aa etc
        aa = -b * d.pow(2)
        bb = (inputs - a) * d.pow(2) - 2 * b * d * g
        cc = (inputs - a) * 2 * d * g - b * (1 + g.pow(2))
        dd = (inputs - a) * (1 + g.pow(2)) - c
        # Find the real root
        # Don't quite follow all this :/
        p = (3 * aa * cc - bb.pow(2)) / (3 * aa.pow(2))
        q = (2 * bb.pow(3) - 9 * aa * bb * cc + 27 * aa.pow(2) * dd) / (
            27 * aa.pow(3)
        )

        t = -2 * torch.abs(q) / q * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = (
            -3 * torch.abs(q) / (2 * p) * torch.sqrt(3 / torch.abs(p))
        )
        inter_term2 = 1 / 3 * torch.arccosh(torch.abs(inter_term1 - 1) + 1)
        t = t * torch.cosh(inter_term2)

        tpos = -2 * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = 3 * q / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * torch.arcsinh(inter_term1)
        tpos = tpos * torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        outputs = t - bb / (3 * aa)

        # Compute log-Jacobian determinant
        arg = d * outputs + g
        denom = 1 + arg.pow(2)
        # Original version is missing the minus sign.
        logabsdet = -self._get_derivative(arg, denom, b, c, d)
        # Map back to ploats
        return outputs.float(), logabsdet.float()

    def _coupling_transform_inverse(
        self,
        inputs: torch.Tensor,
        transform_params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = self._get_params(transform_params)
        arg = d * inputs + g
        denom = 1 + arg.pow(2)
        outputs = a + b * inputs + c / denom
        # Compute log-Jaxobian determinant
        logabsdet = self._get_derivative(arg, denom, b, c, d)
        return outputs, logabsdet
