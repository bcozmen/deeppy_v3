"""High-level network construction facade."""

import torch.nn as nn

from .initialization import WeightInitializer
from .layers import LayerGenerator
from .tasks import build_task


class Network(nn.Module):
    """Build, initialize, compile, and optionally DDP-wrap a network.

    ``arch_params`` and ``decoder_params`` contain layer specifications for
    :class:`LayerGenerator`. The constructed PyTorch module is available as
    ``network.model`` and the facade itself remains callable.
    """

    TASKS = {"reg", "classify", "autoencoder"}

    def __init__(self, arch_params, *, 
                    decoder_params=None, task="reg", classify_threshold=0.5, 
                    weight_init="uniform", runtime=None):
                    
        super().__init__()
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {sorted(self.TASKS)}")

        arch_params = self._as_specs(arch_params)
        decoder_params = self._as_specs(decoder_params)
        if task == "autoencoder" and decoder_params is None:
            decoder_params = self._symmetric_specs(arch_params)
        generator = LayerGenerator()

        layers = [layer for spec in arch_params for layer in generator.generate(spec)]
        encoder_len = len(layers)
        if task == "autoencoder":
            layers.extend(layer for spec in decoder_params for layer in generator.generate(spec))

        task_model = build_task(nn.Sequential(*layers), task, classify_threshold, encoder_len)
        self.model = WeightInitializer(weight_init)(task_model)

        if runtime is not None:
            self.model = runtime.prepare_model(self.model)

        self.task = task
        self.arch_params = arch_params
        self.decoder_params = decoder_params
        self.classify_threshold = classify_threshold

    @staticmethod
    def _as_specs(specifications):
        if specifications is None:
            return None
        return [specifications] if isinstance(specifications, dict) else list(specifications)

    @staticmethod
    def _symmetric_specs(specifications):
        """Mirror architecture specifications for an autoencoder decoder."""
        mirrored = []
        for specification in reversed(specifications):
            specification = dict(specification)
            specification["layers"] = list(reversed(specification.get("layers", [None, None])))
            mirrored.append(specification)
        return mirrored

    def forward(self, inputs):
        return self.model(inputs)

    def encode(self, inputs):
        return self._task_module().encode(inputs)

    def decode(self, encoded):
        return self._task_module().decode(encoded)

    def _task_module(self):
        model = self.model
        return model.module if hasattr(model, "module") else model

    def save_states(self):
        return {"model": self._task_module().state_dict()}

    def load_states(self, state):
        self._task_module().load_state_dict(state["model"])
