"""
Example showing how to handle discrete parameters using reparametersations.
"""

import numpy as np

from nessai.flowsampler import FlowSampler
from nessai.livepoint import empty_structured_array
from nessai.model import Model
from nessai.utils import configure_logger

# Configure the output directory and logger
output = "./outdir/discrete_example/"
logger = configure_logger(output=output, log_level="INFO")

# Set the random number generator
rng = np.random.default_rng(1234)


# Define the model class, this model has two signal models and a discrete
# variable that determines which is used for the computing the log-likelihood
class MultiModelLikelihood(Model):
    def __init__(self, x_data, y_data):
        # x and y data
        self.x_data = x_data
        self.y_data = y_data
        # Assume the standard deviation is known
        self.sigma = 1
        self.bounds = {"m": [-5.0, 5.0], "c": [-5.0, 5.0], "model": [0, 1]}
        self.names = list(self.bounds.keys())
        # Specify the list of discrete parameters
        self.discrete_parameters = ["model"]

    def new_point(self, N=1):
        # Redefine the new point method so that it correctly samples the
        # discrete parameter
        x = empty_structured_array(N, self.names)
        x["m"] = rng.uniform(*self.bounds["m"], size=N)
        x["c"] = rng.uniform(*self.bounds["c"], size=N)
        x["model"] = rng.choice([0, 1], size=N)
        return x

    def new_point_log_prob(self, x):
        # Update the function that defines the log-probability for the
        # `new_point` method.
        log_prob = -np.log(
            np.ptp(self.bounds["m"]) * np.ptp(self.bounds["c"]) * 0.5
        ) * np.ones(len(x))
        return log_prob

    def log_prior(self, x):
        # Check the values are within the prior bounds
        log_p = np.log(self.in_bounds(x), dtype="float")
        # Compute the log-prior for the uniform parameters
        for n in ["m", "c"]:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        # Only accept the allowed values for 'model', log(0) = -inf
        log_p += np.log(~(x["model"] % 1).astype(bool))
        return log_p

    def log_likelihood(self, x):
        # Use a different likelihood depending on the value of the "model"
        # parameter. We use the following two models:
        # 1. (mx + c)
        # 2. (mx^1.1 + c)
        y_fit = np.where(
            x["model"] == 1,
            (x["m"] * self.x_data + x["c"]),
            (x["m"] * self.x_data**1.1 + x["c"]),
        )
        # Standard Gaussian likelihood
        log_l = np.sum(
            -0.5 * (((self.y_data - y_fit) / self.sigma) ** 2)
            - np.log(2 * np.pi * self.sigma**2),
            axis=0,
        )
        return log_l


# Generate some fake data.
# We use the straight line model for this example.
x_data = np.linspace(0, 10, 100)
y_data = 2.5 * x_data + 1.4 + rng.standard_normal(len(x_data))

# Define the sampler and specify the 'dequantise' reparameterisation for the
# discrete parameter.
# This adds random noise from [0, 1) to the discrete values, without this
# the sampling would be inefficient since the flow will not propose integer
# values.
fs = FlowSampler(
    MultiModelLikelihood(x_data, y_data),
    output=output,
    resume=False,
    seed=1234,
    reparameterisations={"model": "dequantise"},
)

# And run as usual
fs.run()
