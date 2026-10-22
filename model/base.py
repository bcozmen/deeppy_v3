"""Runtime-aware model and training-step orchestration."""

from abc import ABC

import torch
import torch.nn as nn

from callbacks.base import CallbackManager
from network import Network
from optimizer import Optimizer
from runtime import Checkpointer


class BaseModel(nn.Module, ABC):
    """Build networks and optimizers from one shared runtime.

    Subclasses can override ``training_step`` for task-specific loss logic and
    ``forward`` for multi-network outputs. The default implementation expects
    batches shaped as ``(inputs, targets)`` and one configured network. Custom
    multi-network training should override ``forward`` and ``optimize``.
    """

    def __init__(
        self,
        runtime,
        networks,
        optimizers=None,
        criterion=None,
        callbacks=None,
        checkpoint_dir=None,
        dataset=None,
    ):
        super().__init__()
        self.runtime = runtime
        self.networks = nn.ModuleList(self._build_networks(networks))
        self.criterion = criterion or nn.MSELoss()
        self.optimizers = self._build_optimizers(optimizers)
        self.callbacks = CallbackManager(callbacks)
        self.global_step = 0
        self.latest_loss = None
        self.checkpointer = (
            Checkpointer(checkpoint_dir, self, runtime, dataset=dataset)
            if checkpoint_dir is not None
            else None
        )

    def _build_networks(self, configurations):
        result = []
        for configuration in configurations:
            if isinstance(configuration, nn.Module):
                result.append(configuration)
            else:
                result.append(Network(runtime=self.runtime, **configuration))
        return result

    def _build_optimizers(self, configurations):
        if configurations is None:
            configurations = [{} for _ in self.networks]
        if len(configurations) != len(self.networks):
            raise ValueError("one optimizer configuration is required per network")
        return [
            Optimizer(network.parameters(), runtime=self.runtime, **configuration)
            for network, configuration in zip(self.networks, configurations)
        ]

    def forward(self, inputs):
        if len(self.networks) != 1:
            raise NotImplementedError("override forward() when using multiple networks")
        return self.networks[0](inputs)

    def training_step(self, batch):
        inputs, targets = batch
        return self.criterion(self(inputs), targets)

    def optimize(self, batch):
        self.train()
        with self.runtime.autocast():
            loss = self.training_step(batch)
        metric_loss = self.runtime.reduce_mean(loss.detach())
        self.callbacks.emit("on_backward_end", self, loss)

        if len(self.optimizers) != 1:
            raise NotImplementedError("override optimize() for multiple optimizers")
        optimizer = self.optimizers[0]
        updated = optimizer.backward(loss, metric_loss=metric_loss)
        if optimizer.accumulation.last_loss is not None:
            self.latest_loss = optimizer.accumulation.last_loss
        if updated:
            self.global_step += 1
        self.callbacks.emit("on_optimizer_step_end", self, updated)
        return updated, loss

    @torch.no_grad()
    def evaluate(self, batch):
        self.eval()
        with self.runtime.autocast():
            return self.training_step(batch)

    def save_states(self):
        return {
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


Model = BaseModel
