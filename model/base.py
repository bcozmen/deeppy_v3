"""Runtime-aware model and training-step orchestration."""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for user-defined runtime-aware training models.

    Subclasses define how their networks and optimizers are built, as well as
    their forward and training-step logic. The base class owns device transfer,
    autocasting, loss aggregation, optimization, evaluation, and state
    serialization.
    """

    def __init__(
        self,
        runtime,
        network_params=None,
        optimizer_params=None,
        model_params=None,
        logger=None,
    ):
        super().__init__()
        self.runtime = runtime
        self.logger = logger
        if self.logger is not None and hasattr(self.logger, "bind"):
            self.logger.bind(runtime)
        self.model_params = deepcopy(model_params)
        self.network_params = deepcopy(network_params)
        self.optimizer_params = deepcopy(optimizer_params)
        self.networks = nn.ModuleList(self.build_networks(self.network_params))
        self.optimizers = list(self.build_optimizers(self.optimizer_params))
        if not self.optimizers:
            raise ValueError("build_optimizers() must return at least one optimizer")
        self.global_step = 0
        self.latest_loss = None

    @abstractmethod
    def build_networks(self, network_params):
        """Return the model's networks from the saved network parameters."""
        raise NotImplementedError

    @abstractmethod
    def build_optimizers(self, optimizer_params):
        """Return the model's optimizers from the saved optimizer parameters."""
        raise NotImplementedError

    def attach_logger(self, logger):
        """Attach an externally managed logger and return ``self``."""
        self.logger = logger
        if hasattr(logger, "bind"):
            logger.bind(self.runtime)
        return self

    @property
    def optimizer(self):
        """Return the primary optimizer used by the default training loop."""
        if len(self.optimizers) != 1:
            raise AttributeError("override optimize() when using multiple optimizers")
        return self.optimizers[0]

    @torch.no_grad()
    def evaluate(self, batch, phase="eval"):
        self.eval()
        batch = self.runtime.to_device(batch)
        scope = self.logger.scope(phase, self.global_step) if self.logger else nullcontext()
        with scope, self.runtime.autocast():
            return self.training_step(batch)

    def test(self, batch):
        """Evaluate a batch using the ``test`` logging phase."""
        return self.evaluate(batch, phase="test")

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch):
        raise NotImplementedError

    def optimize(self, batch):
        self.train()
        batch = self.runtime.to_device(batch)
        scope = self.logger.scope("train", self.global_step) if self.logger else nullcontext()
        with scope, self.runtime.autocast():
            loss = self.training_step(batch)
        metric_loss = self.runtime.reduce_mean(loss.detach())

        optimizer = self.optimizer
        updated = optimizer.backward(loss, metric_loss=metric_loss)
        if optimizer.accumulation.last_loss is not None:
            self.latest_loss = optimizer.accumulation.last_loss
        if updated:
            self.global_step += 1
        return updated, loss

    def save_states(self):
        return {
            "model_params": deepcopy(self.model_params),
            "network_params": deepcopy(self.network_params),
            "optimizer_params": deepcopy(self.optimizer_params),
            "networks": [network.state_dict() for network in self.networks],
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "global_step": self.global_step,
        }

    def load_states(self, state):
        for network, network_state in zip(self.networks, state["networks"]):
            network.load_state_dict(network_state)
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)
        self.global_step = state.get("global_step", 0)

    @classmethod
    def from_checkpoint(cls, runtime, checkpoint, **kwargs):
        """Construct and restore a model directly from a saved model state."""
        if isinstance(checkpoint, (str, Path)):
            checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "model" in checkpoint and "networks" not in checkpoint:
            checkpoint = checkpoint["model"]
        if checkpoint.get("model_params") is not None:
            model = cls(runtime, **checkpoint["model_params"], **kwargs)
        else:
            model = cls(
                runtime,
                network_params=checkpoint["network_params"],
                optimizer_params=checkpoint["optimizer_params"],
                **kwargs,
            )
        model.load_states(checkpoint)
        return model


Model = BaseModel
