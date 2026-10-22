"""Weight initialization strategies."""

import torch.nn as nn


class WeightInitializer:
    """Initialize learnable weights in a model with a named strategy."""

    def __init__(self, strategy="uniform", bias_range=(-0.01, 0.01)):
        if strategy not in (None, "uniform", "normal"):
            raise ValueError("strategy must be None, 'uniform', or 'normal'")
        self.strategy = strategy
        self.bias_range = bias_range

    def __call__(self, model):
        if self.strategy is None:
            return model

        for layer in model.modules():
            if not isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                continue
            initializer = (
                nn.init.xavier_uniform_
                if self.strategy == "uniform"
                else nn.init.xavier_normal_
            )
            initializer(layer.weight)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, *self.bias_range)
        return model
