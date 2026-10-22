"""Model compilation configuration."""

import torch


class Compiler:
    """Compile models when enabled by the runtime configuration."""

    def __init__(self, enabled=False, **kwargs):
        self.enabled = enabled
        self.kwargs = kwargs

    def __call__(self, model):
        if not self.enabled:
            return model
        return torch.compile(model, **self.kwargs)
