import six
import os
import copy
import logging
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from scipy import stats as stats

import matplotlib.pyplot as plt

from .trainer import Trainer
from .flows import BatchNormFlow, setup_model
from .plot import plot_loss, plot_inputs, plot_samples

def update_config(d):
    """
    Update the default dictionary for a trainer
    """
    default_model = dict(n_inputs=None, n_neurons=32, n_blocks=4, n_layers=2,
            ftype='RealNVP')

    default = dict(lr=0.0001,                  # learning rate
                   batch_size=100,             # batch size
                   val_size=0.1,               # validation per cent (0.1 = 10%)
                   max_epochs=500,             # maximum number of training epochs
                   patience=20,                # stop after n epochs with no improvement
                   device_tag="cuda",          # device for training
                   model_config=default_model)

    if not isinstance(d, dict):
        raise TypeError('Must pass a dictionary to update the default trainer settings')
    else:
        default.update(d)
    return default


def weight_reset(m):
    """
    Reset parameters of a given model
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
        m.reset_parameters()


class FlowModel:

    def __init__(self, config=None, output='./'):
        self.intialised = False
        self.output = output
        trainer_dict = update_config(trainer_dict)
        self._setup_from_input_dict(config)

    def save_input(self, attr_dict):
        """
        Save the dictionary used as an inputs as a JSON file
        """
        d = attr_dict.copy()
        d['model_dict'] = attr_dict['model_dict'].copy()
        # Pop bilby priors object if present
        d.pop('bilby_priors', None)
        output_file = self.outdir + "trainer_dict.json"
        for k, v in list(d.items()):
            if type(v) == np.ndarray:
                d[k] = np.array_str(d[k])
        for k, v in list(d['model_dict'].items()):
            if type(v) == np.ndarray:
                d['model_dict'][k] = np.array_str(d['model_dict'][k])
        with open(output_file, "w") as f:
            json.dump(d, f, indent=4)

    def _setup_from_input_dict(self, config):
        """
        Setup the trainer from a dictionary
        """
        if self.outdir is not None:
            attr_dict.pop('outdir')

        if attr_dict['normalise']:
            if 'prior_bounds' not in attr_dict and cpnest_model is None:
                raise RuntimeError('Must provided CPNest model or prior_bounds to use normalisation')
            else:
             if cpnest_model is not None:
                attr_dict["prior_bounds"] = np.array(cpnest_model.bounds)
        else:
            self.prior_bounds = None
        for key, value in six.iteritems(attr_dict):
            setattr(self, key, value)
        self.n_inputs = self.model_dict["n_inputs"]
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
        self.save_input(attr_dict)

        if 'mask' in self.model_dict.keys():
            self.mask = self.model_dict['mask'].copy()
        else:
            self.mask = None

    def get_mask(self, mask):
        """
        Get a the mask
        """
        return None

    def initialise(self):
        """
        Intialise the model and optimiser
        """
        self.device = torch.device(self.device_tag)
        self.model_dict['mask'] = self.get_mask(self.mask)
        self.model = setup_model(**self.model_dict, device=self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
        self.intialised = True

    def _prep_data(self, samples):
        """
        Prep data and return dataloaders for training
        """
        idx = np.random.permutation(samples.shape[0])
        samples = samples[idx]

        logging.debug("N input samples: {}".format(len(samples)))

        # setup data loading
        x_train, x_val = train_test_split(samples, test_size=self.val_size, shuffle=False)
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

    def train(self, samples, max_epochs=None, patience=None):
        """
        Train the flow on samples
        """
        if not self.intialised:
            logging.info("Initialising")
            self.initialise()

        train_loader, val_loader = self._prep_data(samples)

        # train
        if max_epochs is None:
            max_epochs = self.max_epochs
        if patience is None:
            patience = self.patience
        best_epoch = 0
        best_val_loss = np.inf
        best_model = copy.deepcopy(self.model)
        logging.info("Starting training")
        logging.info("Training parameters:")
        logging.info(f"Max. epochs: {max_epochs}")
        logging.info(f"Patience: {patience}")
        history = dict(loss=[], val_loss=[])

        self.weights_file = block_outdir + 'model.pt'
        logging.debug(f'Training with {samples.shape[0]} samples')
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
                logging.info(f"Epoch {epoch}: loss: {loss:.3}, val loss: {val_loss:.3}")

            if epoch - best_epoch > patience and epoch > 100:
                logging.info(f"Epoch {epoch}: Reached patience")
                break

            if self.manager is not None:
                if self.manager.stop_training.value:
                    if epoch >= patience:
                        break

        self.training_count += 1
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
        self.model.apply(weight_reset)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-6)
