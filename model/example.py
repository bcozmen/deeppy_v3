"""Example concrete model built on :class:`BaseModel`."""

from itertools import chain

import torch.nn as nn

from network import Network
from optimizer import Optimizer

from .base import BaseModel


class ExampleModel(BaseModel):
    """A simple supervised model with one optimizer over all networks."""

    def __init__(self, runtime, network_params, optimizer_params, logger=None):
        super().__init__(
            runtime,
            network_params=network_params,
            optimizer_params=optimizer_params,
            logger=logger,
        )

    def build_networks(self, network_params):
        return [
            Network(runtime=self.runtime, **configuration)
            for configuration in network_params
        ]

    def build_optimizers(self, optimizer_params):
        parameters = chain.from_iterable(network.parameters() for network in self.networks)
        return [Optimizer(parameters, runtime=self.runtime, **optimizer_params)]

    def forward(self, inputs):
        if len(self.networks) != 1:
            raise NotImplementedError("ExampleModel expects one network")
        return self.networks[0](inputs)

    def training_step(self, batch):
        inputs, targets = batch
        return nn.functional.mse_loss(self(inputs), targets)


Model = ExampleModel
