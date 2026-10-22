"""Runtime-owned automatic mixed-precision configuration."""

from contextlib import nullcontext

import torch


class AMPConfig:
    """Store AMP settings and provide the forward autocast context."""

    def __init__(self, config=None, *, device_type="cuda"):
        if config is None or config is False:
            self.options = None
            return
        options = {} if config is True else dict(config)
        self.options = {
            "device_type": device_type,
            "dtype": torch.float16,
            "enabled": True,
            **options,
        }

    @property
    def enabled(self):
        return self.options is not None and self.options["enabled"]

    def __bool__(self):
        return self.enabled

    def autocast(self):
        if not self.enabled:
            return nullcontext()
        return torch.autocast(
            device_type=self.options["device_type"],
            dtype=self.options["dtype"],
            enabled=self.options["enabled"],
        )
