# -*- coding: utf-8 -*-
"""
Object and functions to handle training the normalising flow.
"""
import copy
import logging
import numpy as np
import os
import shutil
import torch
from torch.nn.utils import clip_grad_norm_

from .utils import update_config

from ..flows import configure_model, reset_weights, reset_permutations
from ..flows.base import BaseFlow
from ..plot import plot_loss
from ..utils import save_to_json, compute_minimum_distances

logger = logging.getLogger(__name__)


class FlowModel:
    """
    Object that contains the normalising flows and handles training and data
    pre-processing.

    Does NOT use structured arrays for live points, \
            :obj:`~nessai.proposal.base.Proposal`
    object should act as the interface between structured used by the sampler
    and unstructured arrays of live points used for training.

    Parameters
    ----------
    config : dict, optional
        Configuration used for the normalising flow. If None, default values
        are used.
    output : str, optional
        Path for output, this includes weights files and the loss plot. If
        not specified, the current working directory is used.
    """

    model_config = None
    noise_scale = None
    noise_type = None
    model: BaseFlow = None

    def __init__(self, config=None, output=None):
        if output is None:
            output = os.getcwd()
        self.model = None
        self.initialised = False
        self.output = output
        os.makedirs(self.output, exist_ok=True)
        self.setup_from_input_dict(config)
        self.weights_file = None

        self.device = None
        self.inference_device = None
        self.use_dataloader = False
        self._batch_size = None

    def save_input(self, config, output_file=None):
        """
        Save the dictionary used as an inputs as a JSON file in the output
        directory.

        Parameters
        ----------
        config : dict
            Dictionary to save.
        output_file : str, optional
            File to save the config to.
        """
        config = copy.deepcopy(config)
        if output_file is None:
            output_file = os.path.join(self.output, "flow_config.json")
        for k, v in list(config.items()):
            if isinstance(v, np.ndarray):
                config[k] = np.array2string(config[k], separator=",")
        for k, v in list(config["model_config"].items()):
            if isinstance(v, np.ndarray):
                config["model_config"][k] = np.array2string(
                    config["model_config"][k], separator=","
                )

        if "flow" in config["model_config"]:
            config["model_config"]["flow"] = str(
                config["model_config"]["flow"]
            )

        save_to_json(config, output_file)

    def setup_from_input_dict(self, config):
        """
        Setup the trainer from a dictionary, all keys in the dictionary are
        added as methods to the object. Input is automatically saved.

        Parameters
        ----------
        config : dict
            Dictionary with parameters that are used to update the defaults.
        """
        config = update_config(config)
        logger.debug(f"Flow configuration: {config}")
        for key, value in config.items():
            setattr(self, key, value)
        self.save_input(config)

    def update_mask(self):
        """Method to update the ask upon calling ``initialise``

        By default the mask is left unchanged.
        """
        pass

    def get_optimiser(self, optimiser="adam", **kwargs):
        """
        Get the optimiser and ensure it is always correctly initialised.

        Returns
        -------
        :obj:`torch.optim.Adam`
            Instance of the Adam optimiser from torch.optim
        """
        optimisers = {
            "adam": (torch.optim.Adam, {"weight_decay": 1e-6}),
            "adamw": (torch.optim.AdamW, {}),
            "sgd": (torch.optim.SGD, {}),
        }
        if self.model is None:
            raise RuntimeError("Cannot initialise optimiser before model")
        optim, default_kwargs = optimisers.get(optimiser.lower())
        default_kwargs.update(kwargs)
        return optim(self.model.parameters(), lr=self.lr, **default_kwargs)

    def initialise(self):
        """
        Initialise the model and optimiser.

        This includes:

            - Updating the model configuration
            - Initialising the normalising flow
            - Initialising the optimiser
            - Configuring the inference device
        """
        self.update_mask()
        self.model, self.device = configure_model(self.model_config)
        logger.debug(f"Training device: {self.device}")
        self.inference_device = torch.device(
            self.model_config.get("inference_device_tag", self.device)
            or self.device
        )
        logger.debug(f"Inference device: {self.inference_device}")

        self._optimiser = self.get_optimiser(
            self.optimiser, **self.optimiser_kwargs
        )
        self.initialised = True

    def move_to(self, device, update_default=False):
        """Move the flow to a different device.

        Parameters
        ----------
        device : str
            Device to move flow to.
        update_default : bool, optional
            If True, the default device for the flow (and data) will be
            updated.
        """
        device = torch.device(device)
        self.model.to(device)
        if update_default:
            self.device = device

    @staticmethod
    def check_batch_size(x, batch_size, min_fraction=0.1):
        """Check that the batch size is valid.

        Tries to ensure that the last batch is at least a minimum fraction of
        the size of the batch size.

        Parameters
        ----------
        x : numpy.ndarray
            Training data.
        batch_size : int
            The user-specified batch size
        min_fraction : float
            Fraction of user-specified batch size to use a lower limit.
        """
        logger.debug("Checking batch size")
        if batch_size == 1:
            raise ValueError("Cannot use a batch size of 1!")
        min_batch_size = int(min_fraction * batch_size)
        final_batch_size = len(x) % batch_size
        if final_batch_size and (final_batch_size < min_batch_size):
            logger.debug(
                "Adjusting batch size to ensure final batch has at least "
                f"{min_batch_size} samples in it."
            )
            while True:
                batch_size -= 1
                final_batch_size = len(x) % batch_size
                if batch_size < 2:
                    raise RuntimeError("Could not find a valid batch size")
                elif (final_batch_size == 0) or (
                    final_batch_size >= min_batch_size
                ):
                    break
                elif (batch_size <= min_batch_size) and final_batch_size > 1:
                    logger.warning(
                        f"Batch size is less than {min_batch_size} but valid. "
                        f"Setting batch size to: {batch_size}"
                    )
                    break
        logger.debug(f"Using valid batch size of: {batch_size}")
        return batch_size

    def prep_data(
        self, samples, val_size, batch_size, weights=None, use_dataloader=False
    ):
        """
        Prep data and return dataloaders for training

        Parameters
        ----------
        samples : array_like
            Array of samples to split in to training and validation.
        val_size : float
            Float between 0 and 1 that defines the fraction of data used for
            validation.
        batch_size : int
            Batch size used when constructing dataloaders.
        use_dataloader : bool, optional
            If True data is returned in a dataloader else a tensor is returned.
        weights : array_like, optional
            Array of weights for each sample, weights will used when computing
            the loss.

        Returns
        -------
        train_data, val_data :
            Training and validation data as either a tensor or dataloader
        """
        if not self.initialised:
            self.initialise()

        if not np.isfinite(samples).all():
            raise ValueError("Cannot train with non-finite samples!")

        idx = np.random.permutation(samples.shape[0])
        samples = samples[idx]
        if weights is not None:
            if not np.isfinite(weights).all():
                raise ValueError("Weights contain non-finite values!")
            weights = weights[idx]
            use_dataloader = True

        logger.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        if val_size is None:
            val_size = 0
        n = int((1 - val_size) * samples.shape[0])
        x_train, x_val = samples[:n], samples[n:]
        if weights is not None:
            weights_train = weights[:n]
            weights_val = weights[n:]
        logger.debug(f"{x_train.shape} training samples")
        logger.debug(f"{x_val.shape} validation samples")

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            if batch_size == "all" or batch_size is None:
                batch_size = x_train.shape[0]
            else:
                raise RuntimeError(f"Unknown batch size: {batch_size}")

        batch_size = self.check_batch_size(x_train, batch_size)
        self._batch_size = batch_size
        # Validation size cannot be bigger than the batch size
        val_batch_size = min(len(x_val), batch_size) if len(x_val) else None

        dtype = torch.get_default_dtype()
        logger.debug(f"Using dtype {dtype} for tensors")
        if use_dataloader:
            logger.debug("Using dataloaders")
            train_tensors = [torch.from_numpy(x_train).type(dtype)]
            val_tensors = [torch.from_numpy(x_val).type(dtype)]

            if weights is not None:
                train_tensors.append(
                    torch.from_numpy(weights_train).type(dtype)
                )
                val_tensors.append(torch.from_numpy(weights_val).type(dtype))

            train_dataset = torch.utils.data.TensorDataset(*train_tensors)
            train_data = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            val_dataset = torch.utils.data.TensorDataset(*val_tensors)
            val_data = torch.utils.data.DataLoader(
                val_dataset, batch_size=val_batch_size, shuffle=False
            )
        else:
            logger.debug("Using tensors")
            train_data = torch.from_numpy(x_train).type(dtype).to(self.device)
            val_data = torch.from_numpy(x_val).type(dtype).to(self.device)

        return train_data, val_data, batch_size

    def end_iteration(self):
        """Calls any functions that should be applied at the end of the \
            iteration.

        This functions is called after the flow has been updated on all batches
        of data but before the validation step.

        Calls :py:meth:`nessai.flows.base.BaseFlow.end_iteration`
        """
        self.model.end_iteration()

    def _train(
        self,
        train_data,
        noise_scale=0.0,
        is_dataloader=False,
        weighted=False,
    ):
        """
        Loop over the data and update the weights

        Parameters
        ----------
        train_data : :obj:`torch.util.data.Dataloader` or :obj:`torch.Tensor`
            Training data. If a tensor is provided, it is split into batches
            using the batch size.
        noise_scale : float, optional
            Scale of Gaussian noise added to data.
        is_dataloader : bool, optional
            Must be True when using a dataloader
        weighted : bool
            If True the weighted KL will be used to compute the loss. Requires
            data to include weights.

        Returns
        -------
        float
            Mean of training loss for each batch.
        """
        model = self.model
        model.train()
        train_loss = 0

        if hasattr(model, "loss_function"):
            loss_fn = model.loss_function
        elif weighted:

            def loss_fn(data, weights):
                return -torch.sum(model.log_prob(data) * weights) / torch.sum(
                    weights
                )

        else:

            def loss_fn(data):
                return -model.log_prob(data).mean()

        if not is_dataloader:
            p = torch.randperm(train_data.shape[0])
            train_data = train_data[p, :].split(self._batch_size)

        n = 0
        for data in train_data:
            if is_dataloader:
                x = data[0].to(self.device)
            else:
                x = data
            if noise_scale:
                x = x + noise_scale * torch.randn_like(x)
            for param in model.parameters():
                param.grad = None
            if weighted:
                weights = data[1].to(self.device)
                loss = loss_fn(x, weights)
            else:
                loss = loss_fn(x)
            train_loss += loss.item()
            loss.backward()
            if self.clip_grad_norm:
                clip_grad_norm_(model.parameters(), self.clip_grad_norm)
            self._optimiser.step()
            n += 1

        self.end_iteration()

        if self.annealing:
            self.scheduler.step()

        return train_loss / n

    def _validate(self, val_data, is_dataloader=False, weighted=False):
        """
        Loop over the data and get validation loss

        Parameters
        ----------
        val_data : :obj:`torch.util.data.Dataloader` or :obj:`torch.Tensor
            Dataloader with data to validate on
        is_dataloader : bool, optional
            Boolean to indicate if the data is a dataloader
        weighted : bool
            If True the weighted KL will be used to compute the loss. Requires
            data to include weights.

        Returns
        -------
        float
            Mean of training loss for each batch.
        """
        if not len(val_data):
            return np.nan
        model = self.model
        model.eval()
        val_loss = 0

        if hasattr(model, "loss_function"):
            loss_fn = model.loss_function
        elif weighted:

            def loss_fn(data, weights):
                return -torch.sum(model.log_prob(data) * weights) / torch.sum(
                    weights
                )

        else:

            def loss_fn(data):
                return -model.log_prob(data).mean()

        if is_dataloader:
            n = 0
            for data in val_data:
                if is_dataloader:
                    x = data[0].to(self.device)
                if weighted:
                    weights = data[1].to(self.device)
                    with torch.inference_mode():
                        val_loss += loss_fn(x, weights).item()
                else:
                    with torch.inference_mode():
                        val_loss += loss_fn(x).item()
                n += 1

            return val_loss / n
        else:
            with torch.inference_mode():
                val_loss += loss_fn(val_data).item()
            return val_loss

    def finalise(self):
        """Method to finalise the flow before using it for inference."""
        logger.debug("Finalising model before inference.")
        self.model.finalise()

    def train(
        self,
        samples,
        weights=None,
        max_epochs=None,
        patience=None,
        output=None,
        val_size=None,
        plot=True,
    ):
        """
        Train the flow on a set of samples.

        Allows for training parameters to specified instead of those
        given in initial config.

        Parameters
        ----------
        samples : ndarray
            Unstructured numpy array containing data to train on
        weights : array_like
            Array of weights to use with the weight KL when computing the loss.
        max_epochs : int, optional
            Maximum number of epochs that is used instead of value
            in the configuration.
        patience : int, optional
            Patience in number of epochs that is used instead of value
            in the configuration.
        val_size : float, optional
           Fraction of the samples to use for validation
        output : str, optional
            Path to output directory that is used instead of the path
            specified when the object is initialised
        plot : bool, optional
            Boolean to enable or disable plotting the loss function

        Returns
        -------
        history : dict
            Dictionary that contains the training and validation losses.
        """
        if not self.initialised:
            logger.debug("Initialising")
            self.initialise()

        if not np.isfinite(samples).all():
            raise ValueError("Training data is not finite")

        logger.debug("Data summary:")
        logger.debug(f"Mean: {np.mean(samples, axis=0)}")
        logger.debug(f"Standard deviation: {np.std(samples, axis=0, ddof=1)}")

        if output is None:
            output = self.output
        else:
            os.makedirs(output, exist_ok=True)

        if val_size is None:
            val_size = self.val_size

        if val_size == 0.0:
            validate = False
        else:
            validate = True

        if self.noise_type == "adaptive":
            noise_scale = self.noise_scale * np.mean(
                compute_minimum_distances(samples)
            )
            logger.debug(f"Using adaptive scale: {noise_scale:.3f}")
        elif self.noise_type == "constant":
            noise_scale = self.noise_scale
        else:
            logger.debug("No noise will be added in training")
            noise_scale = None

        self.move_to(self.device)

        weighted = True if weights is not None else False
        use_dataloader = self.use_dataloader or weighted

        train_data, val_data, _ = self.prep_data(
            samples,
            val_size=val_size,
            batch_size=self.batch_size,
            weights=weights,
            use_dataloader=use_dataloader,
        )

        if max_epochs is None:
            max_epochs = self.max_epochs
        if self.annealing:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimiser, max_epochs
            )
        if patience is None:
            patience = self.patience
        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model.state_dict())
        logger.debug("Starting training")
        logger.debug("Training parameters:")
        logger.debug(f"Max. epochs: {max_epochs}")
        logger.debug(f"Patience: {patience}")
        logger.debug(f"Training with {samples.shape[0]} samples")

        history = dict(loss=[], val_loss=[])

        current_weights_file = os.path.join(output, "model.pt")

        for epoch in range(1, max_epochs + 1):

            loss = self._train(
                train_data,
                noise_scale=noise_scale,
                is_dataloader=use_dataloader,
                weighted=weighted,
            )
            val_loss = self._validate(
                val_data,
                is_dataloader=use_dataloader,
                weighted=weighted,
            )
            history["loss"].append(loss)
            history["val_loss"].append(val_loss)

            if validate and (val_loss < best_val_loss):
                best_epoch = epoch
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model.state_dict())

            if not epoch % 50:
                logger.debug(
                    f"Epoch {epoch}: loss: {loss:.3} val loss: {val_loss:.3}"
                )

            if validate and (epoch - best_epoch > patience):
                logger.debug(f"Epoch {epoch}: Reached patience")
                break

        logger.debug("Finished training")

        # Make sure cache for LU is reset.
        self.model.train()
        self.model.eval()
        if validate:
            logger.debug(f"Loading best model from epoch {best_epoch}")
            self.model.load_state_dict(best_model)
        self.finalise()

        self.save_weights(current_weights_file)
        self.move_to(self.inference_device)
        self.model.eval()

        if plot:
            plot_loss(
                epoch, history, filename=os.path.join(output, "loss.png")
            )
        return history

    def save_weights(self, weights_file):
        """
        Save the weights file. If the file already exists move it to
        ``<weights_file>.old`` and then save the file.

        Parameters
        ----------
        weights_file : str
            Path to to file to save weights. Recommended file type is ``.pt``.
        """
        if os.path.exists(weights_file):
            shutil.move(weights_file, weights_file + ".old")

        torch.save(self.model.state_dict(), weights_file)
        self.weights_file = weights_file

    def load_weights(self, weights_file):
        """
        Load weights for the model and initialises the model if it is not
        initialised. The weights_file attribute is also updated.

        Model is loaded in evaluation mode (``model.eval()``)

        Parameters
        ----------
        weights_files : str
            Path to weights file
        """
        # TODO: these two methods are basically the same
        if not self.initialised:
            self.initialise()
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()
        self.weights_file = weights_file

    def reload_weights(self, weights_file):
        """
        Tries to the load the weights file and if not, tries to load
        the weights file stored internally.

        Parameters
        ----------
        weights_files : str
            Path to weights file
        """
        if weights_file is None:
            weights_file = self.weights_file
        logger.debug(f"Reloading weights from {weights_file}")
        self.load_weights(weights_file)

    def reset_model(self, weights=True, permutations=False):
        """
        Reset the weights of the model and optimiser.

        Parameters
        ----------
        weights : bool, optional
            If true the model weights are reset.
        permutations : bool, optional
            If true any permutations (linear transforms) are reset.
        """
        if not any([weights, permutations]):
            logger.debug("Nothing to reset")
            return
        if weights and permutations:
            logger.debug("Complete reset of model")
            self.model, self.device = configure_model(self.model_config)
        elif weights:
            self.model.apply(reset_weights)
            logger.debug("Reset weights")
        elif permutations:
            self.model.apply(reset_permutations)
            logger.debug("Reset linear transforms")
        self._optimiser = self.get_optimiser(
            self.optimiser, **self.optimiser_kwargs
        )
        logger.debug("Resetting optimiser")

    def forward_and_log_prob(self, x):
        """
        Forward pass through the model and return the samples in the latent
        space with their log probabilities

        Parameters
        ----------
        x : ndarray
            Array of samples

        Returns
        -------
        z : ndarray
            Samples in the latent space
        log_prob : ndarray
            Log probabilities for each samples
        """
        x = (
            torch.from_numpy(x)
            .type(torch.get_default_dtype())
            .to(self.model.device)
        )
        self.model.eval()
        with torch.inference_mode():
            z, log_prob = self.model.forward_and_log_prob(x)

        z = z.detach().cpu().numpy().astype(np.float64)
        log_prob = log_prob.detach().cpu().numpy().astype(np.float64)
        return z, log_prob

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Compute the log-probability of a sample.

        Parameters
        ----------
        x : ndarray
            Array of samples in the X-prime space.

        Returns
        -------
        ndarray
            Array of log-probabilities.
        """
        x = (
            torch.from_numpy(x)
            .type(torch.get_default_dtype())
            .to(self.model.device)
        )
        self.model.eval()
        with torch.inference_mode():
            log_prob = self.model.log_prob(x)
        log_prob = log_prob.cpu().numpy().astype(np.float64)
        return log_prob

    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from the flow.

        Parameters
        ----------
        n : int
            Number of samples to draw

        Returns
        -------
        numpy.ndarray
            Array of samples
        """
        with torch.inference_mode():
            x = self.model.sample(int(n))
        return x.cpu().numpy().astype(np.float64)

    def sample_latent_distribution(self, n: int = 1) -> np.ndarray:
        """Sample from the latent distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw

        Returns
        -------
        numpy.ndarray
            Array of samples
        """
        with torch.inference_mode():
            z = self.model.sample_latent_distribution(n)
        return z.cpu().numpy().astype(np.float64)

    def sample_and_log_prob(self, N=1, z=None, alt_dist=None):
        """
        Generate samples from samples drawn from the base distribution or
        and alternative distribution from provided latent samples

        Parameters
        ----------
        N : int, optional
            Number of samples to draw if z is not specified
        z : ndarray, optional
            Array of latent samples to map the the data space, if ``alt_dist``
            is not specified they are assumed to be drawn from the base
            distribution of the flow.
        alt_dist : :obj:`glasflow.nflows.distribution.Distribution`
            Distribution object from which the latent samples z were
            drawn from. Must have a ``log_prob`` method that accepts an
            instance of ``torch.Tensor``

        Returns
        -------
        samples : ndarray
            Array containing samples in the latent space.
        log_prob : ndarray
            Array containing the log probability that corresponds to each
            sample.
        """
        if self.model is None:
            raise RuntimeError("Model is not initialised yet!")
        if self.model.training:
            self.model.eval()
        if z is None:
            with torch.inference_mode():
                x, log_prob = self.model.sample_and_log_prob(int(N))
        else:
            if alt_dist is not None:
                log_prob_fn = alt_dist.log_prob
            else:
                log_prob_fn = self.model.base_distribution_log_prob

            with torch.inference_mode():
                if isinstance(z, np.ndarray):
                    z = (
                        torch.from_numpy(z)
                        .type(torch.get_default_dtype())
                        .to(self.model.device)
                    )
                log_prob = log_prob_fn(z)
                x, log_J = self.model.inverse(z, context=None)
                log_prob -= log_J

        x = x.detach().cpu().numpy().astype(np.float64)
        log_prob = log_prob.detach().cpu().numpy().astype(np.float64)
        return x, log_prob

    def __getstate__(self):
        state = self.__dict__.copy()
        state["initialised"] = False
        del state["optimiser"]
        del state["model"]
        del state["model_config"]
        return state
