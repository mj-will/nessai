import six
import os
import copy
import logging
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .flows import BatchNormFlow, setup_model
from .plot import plot_loss, plot_inputs, plot_samples

logger = logging.getLogger(__name__)


def update_config(d):
    """
    Update the default dictionary for a trainer
    """
    default_model = dict(n_inputs=None, n_neurons=32, n_blocks=4, n_layers=2,
            ftype='RealNVP', mask=None)

    if 'model_config' in d.keys():
        default_model.update(d['model_config'])

    default = dict(lr=0.0001,                  # learning rate
                   batch_size=100,             # batch size
                   val_size=0.1,               # validation per cent (0.1 = 10%)
                   max_epochs=500,             # maximum number of training epochs
                   patience=20,                # stop after n epochs with no improvement
                   device_tag="cuda")          # device for training

    if not isinstance(d, dict):
        raise TypeError('Must pass a dictionary to update the default trainer settings')
    else:
        default.update(d)
        default['model_config'] = default_model

    return default


def weight_reset(m):
    """
    Reset parameters of a given model
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()


class FlowModel:

    def __init__(self, config=None, output='./'):
        self.initialised = False
        self.output = output
        self._setup_from_input_dict(config)

    def save_input(self, config):
        """
        Save the dictionary used as an inputs as a JSON file
        """
        output_file = self.output + "flow_config.json"
        for k, v in list(config.items()):
            if type(v) == np.ndarray:
                config[k] = np.array_str(config[k])
        for k, v in list(config['model_config'].items()):
            if type(v) == np.ndarray:
                config['model_config'][k] = np.array_str(config['model_config'][k])
        with open(output_file, "w") as f:
            json.dump(config, f, indent=4)

    def _setup_from_input_dict(self, config):
        """
        Setup the trainer from a dictionary
        """
        config = update_config(config)
        logger.debug(f'Flow configuration: {config}')
        for key, value in config.items():
            setattr(self, key, value)
        self.save_input(config)
        # TODO: mask

    def get_mask(self, mask):
        """
        Get a the mask
        """
        return None

    def initialise(self):
        """
        Initialise the model and optimiser
        """
        self.device = torch.device(self.device_tag)
        #self.model_config['mask'] = self.get_mask(self.mask)
        self.model = setup_model(**self.model_config, device=self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.initialised = True

    def _prep_data(self, samples, val_size):
        """
        Prep data and return dataloaders for training
        """
        idx = np.random.permutation(samples.shape[0])
        samples = samples[idx]

        logger.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        n = int(val_size * samples.shape[0])
        x_train, x_val = samples[:n], samples[n:]
        train_tensor = torch.from_numpy(x_train.astype(np.float32))
        train_dataset = torch.utils.data.TensorDataset(train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_tensor = torch.from_numpy(x_val.astype(np.float32))
        val_dataset = torch.utils.data.TensorDataset(val_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=x_val.shape[0], shuffle=False)

        return train_loader, val_loader

    def _train(self, loader, noise_scale=0.0):
        """
        Loop over the data and update the weights

        Returns
        -------
        Mean of training loss for each batch
        """
        model = self.model
        model.train()
        train_loss = 0

        for idx, data in enumerate(loader):
            data = (data[0] + noise_scale * torch.randn_like(data[0])).to(self.device)
            self.optimiser.zero_grad()
            loss = -model.log_probs(data).mean()
            train_loss += loss.item()
            loss.backward()
            self.optimiser.step()

            for module in model.modules():
                if isinstance(module, BatchNormFlow):
                    module.momentum = 0

            with torch.no_grad():
                model(loader.dataset.tensors[0].to(data.device))

            for module in model.modules():
                if isinstance(module, BatchNormFlow):
                    module.momentum = 1

        return train_loss / len(loader)

    def _validate(self, loader):
        """
        Loop over the data and get validation loss

        Returns
        -------
        Mean of training loss for each batch
        """
        model = self.model
        model.eval()
        val_loss = 0

        for idx, data in enumerate(loader):
            data = data[0].to(self.device)
            with torch.no_grad():
                val_loss += -model.log_probs(data).mean().item()

        return val_loss / len(loader)

    def train(self, samples, max_epochs=None, patience=None, output=None,
            val_size=None):
        """
        Train the flow on samples
        """
        if not self.initialised:
            logger.info("Initialising")
            self.initialise()

        if output is None:
            output = self.output
        else:
            os.makedirs(output, exist_ok=True)

        if val_size is None:
            val_size = self.val_size

        train_loader, val_loader = self._prep_data(samples, val_size=val_size)

        # train
        if max_epochs is None:
            max_epochs = self.max_epochs
        if patience is None:
            patience = self.patience
        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model)
        logger.info("Starting training")
        logger.info("Training parameters:")
        logger.info(f"Max. epochs: {max_epochs}")
        logger.info(f"Patience: {patience}")
        history = dict(loss=[], val_loss=[])

        self.weights_file = output + 'model.pt'
        logger.debug(f'Training with {samples.shape[0]} samples')
        for epoch in range(1, max_epochs + 1):

            loss = self._train(train_loader)
            val_loss = self._validate(val_loader)
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
                best_model = copy.deepcopy(self.model)

            if not epoch % 50:
                logger.info(f"Epoch {epoch}: loss: {loss:.3}, val loss: {val_loss:.3}")

            if epoch - best_epoch > patience and epoch > 100:
                logger.info(f"Epoch {epoch}: Reached patience")
                break

        self.model.load_state_dict(best_model.state_dict())
        self.model.eval()

    def load_weights(self, weights_file):
        """
        Load weights for the model

        Model is loaded in evaluation mode (model.eval())
        """
        if not self.initialised:
            self.initialise()
        self.model.load_state_dict(torch.load(weights_file))
        self.model.eval()

    def reset_model(self):
        """
        Reset the weights and optimiser
        """
        logger.debug('Reseting model weights and optimiser')
        self.model.apply(weight_reset)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
