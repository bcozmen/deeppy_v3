import torch.nn as nn


class TaskModel(nn.Module):
    """Base task model with a shared sequential backbone."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs)


class ClassificationTask(TaskModel):
    """Return thresholded outputs only during evaluation."""

    def __init__(self, model, threshold):
        super().__init__(model)
        self.threshold = threshold

    def forward(self, inputs):
        outputs = super().forward(inputs)
        if not self.training:
            return (outputs > self.threshold).float()
        return outputs


class AutoencoderTask(nn.Module):
    """Separate an encoder and decoder from one sequential model."""

    def __init__(self, model, encoder_len):
        super().__init__()
        self.encoder = model[:encoder_len]
        self.decoder = model[encoder_len:]

    def forward(self, inputs):
        return self.decode(self.encode(inputs))

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, encoded):
        return self.decoder(encoded)


def build_task(model, task, classify_threshold=0.5, encoder_len=0):
    """Create the task-specific module for ``task``."""
    if task == "autoencoder":
        return AutoencoderTask(model, encoder_len)
    if task == "classify":
        return ClassificationTask(model, classify_threshold)
    return TaskModel(model)
