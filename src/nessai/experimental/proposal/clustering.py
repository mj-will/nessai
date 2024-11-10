import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ...livepoint import live_points_to_array
from ...model import Model
from ...plot import (
    nessai_style,
    plot_1d_comparison,
)
from ...proposal.flowproposal import FlowProposal
from ..flowmodel.clustering import ClusteringFlowModel


class ClusteringFlowProposal(FlowProposal):
    """FlowProposal that includes clustering for multi-modal live points.

    Parameters
    ----------
    model
        The model class used for computing the log-prior and log-likelihood.
    max_n_clusters
        The maximum number of clusters allowed. This is passed to
        :py:class:`~nessai.flowmodel.clustering.ClusteringFlowModel`. If not
        specified, the default in the flow model class will be used.
    kwargs
        Keyword arguments passed to
        :py:class:`~nessai.proposal.flowproposal.FlowProposal`, see that class
        for a complete list.
    """

    _FlowModelClass = ClusteringFlowModel

    def __init__(
        self, model: Model, max_n_clusters: int = None, **kwargs
    ) -> None:
        flow_config = kwargs.pop("flow_config", {}) or {}
        if max_n_clusters is not None:
            flow_config["max_n_clusters"] = max_n_clusters
        super().__init__(model, flow_config=flow_config, **kwargs)

    @nessai_style()
    def _plot_training_data(self, output: str) -> None:
        """Plot the training data and compare to the results.

        Plots each cluster individually.
        """
        z_training_data, _ = self.forward_pass(
            self.training_data, rescale=True
        )
        prime_array = live_points_to_array(
            self.training_data_prime,
            self.prime_parameters,
        )
        cluster_labels = self.flow.get_cluster_labels(prime_array)

        fig = plt.figure()
        plt.scatter(
            prime_array[:, 0],
            prime_array[:, 1],
            c=cluster_labels,
        )
        fig.savefig(os.path.join(output, "training_clusters"))
        plt.close(fig)

        z_gen = self.rng.standard_normal((self.training_data.size, self.dims))

        fig = plt.figure()
        plt.hist(np.sqrt(np.sum(z_training_data**2, axis=1)), "auto")
        plt.xlabel("Radius")
        fig.savefig(os.path.join(output, "radial_dist.png"))
        plt.close(fig)

        plot_1d_comparison(
            z_training_data,
            z_gen,
            labels=["z_live_points", "z_generated"],
            convert_to_live_points=True,
            filename=os.path.join(output, "z_comparison.png"),
        )

        x_prime_gen, log_prob = self.backward_pass(z_gen, rescale=False)
        gen_cluster_labels = self.flow.get_cluster_labels(
            live_points_to_array(x_prime_gen, self.prime_parameters)
        )
        x_prime_gen["logL"] = log_prob
        x_gen, log_J = self.inverse_rescale(x_prime_gen)
        (
            x_gen,
            log_J,
            x_prime_gen,
            gen_cluster_labels,
        ) = self.check_prior_bounds(
            x_gen, log_J, x_prime_gen, gen_cluster_labels
        )
        x_gen["logL"] += log_J

        labels = set(np.unique(gen_cluster_labels)) | set(
            np.unique(cluster_labels)
        )
        to_plot = [
            x_gen[np.squeeze(gen_cluster_labels == label)] for label in labels
        ]
        plot_labels = ["live_points"] + [f"cluster_{i}" for i in labels]
        colours = ["k"] + list(
            sns.color_palette("colorblind", n_colors=len(plot_labels) - 1),
        )

        plot_1d_comparison(
            self.training_data,
            *to_plot,
            parameters=self.model.names,
            labels=plot_labels,
            colours=colours,
            filename=os.path.join(output, "x_comparison.png"),
        )

        if self.parameters_to_rescale:
            plot_1d_comparison(
                self.training_data_prime,
                x_prime_gen,
                parameters=self.prime_parameters,
                labels=["live points", "generated"],
                filename=os.path.join(output, "x_prime_comparison.png"),
            )
