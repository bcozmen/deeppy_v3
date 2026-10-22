"""Automatic mixed-precision support for optimizer wrapper chains."""

from contextlib import nullcontext

import torch

from .base import OptimizerWrapper


class AMP(OptimizerWrapper):
    """Add mixed-precision autocasting and gradient scaling.

    ``GradScaler`` is used only for enabled float16 CUDA training. The
    wrapper also unscales gradients before downstream wrappers, so clipping
    operates on gradients in their true scale.
    """

    def __init__(self, inner, device_type, dtype, enabled):
        """Configure autocasting for the wrapped optimizer.

        Args:
            inner: Optimizer being wrapped.
            device_type: Device type passed to PyTorch AMP (for example,
                ``"cuda"`` or ``"cpu"``).
            dtype: Autocast data type.
            enabled: Whether float16 gradient scaling should be enabled.
        """
        super().__init__(inner)

        self.device_type = device_type
        self.dtype = dtype

        self.enabled = bool(enabled) and dtype == torch.float16

        self.scaler = torch.amp.GradScaler(device=device_type, enabled=self.enabled)

    def autocast(self):
        """Return the autocast context used around the forward pass."""
        if not self.enabled:
            return nullcontext()

        return torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=True)

    def backward(self, loss):
        """Scale ``loss`` and backpropagate through the wrapped optimizer."""
        self.scaler.scale(loss).backward()

    def prepare_step(self):
        """Unscale gradients before downstream preparation steps."""
        # Unscale the gradients before downstream wrappers
        # perform operations such as gradient clipping.
        if self.enabled:
            self.scaler.unscale_(self.raw_optimizer)

        self.inner.prepare_step()

    def update(self):
        """Apply a scaled update and report whether AMP accepted it."""
        if not self.enabled:
            return self.inner.update()

        old_scale = self.scaler.get_scale()

        self.scaler.step(self.raw_optimizer)
        self.scaler.update()

        new_scale = self.scaler.get_scale()

        # A decreased scale indicates that GradScaler skipped
        # the optimizer update because of non-finite gradients.
        return new_scale >= old_scale

    def after_step(self, updated):
        """Delegate post-update processing."""
        self.inner.after_step(updated)

    def state_dict(self):
        """Save both wrapped optimizer and gradient-scaler state."""
        return {
            "inner": self.inner.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state):
        """Restore both wrapped optimizer and gradient-scaler state."""
        self.inner.load_state_dict(state["inner"])
        self.scaler.load_state_dict(state["scaler"])
