"""Integration with the shared distributed-training helper."""


class DDPWrapper:
    """Use one initialized ``Distributed`` object for device and DDP setup."""

    def __init__(self, distributed=None):
        self.distributed = distributed

    def __call__(self, model):
        if self.distributed is not None:
            return self.distributed.wrap_model(model)
        return model
